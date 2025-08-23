import asyncio
import uuid
from datetime import datetime
from typing import Dict, Optional, Any
from dataclasses import dataclass

from .config import Config
from .models import JobStatus, JobState
from .agents.base import BaseAgent
from .agents.codex_agent import CodexAgent
from .agents.claude_agent import ClaudeAgent
from .agents.ollama_agent import OllamaAgent
from .storage.base import BaseStorage
from .storage.memory import MemoryStorage


@dataclass
class AgentJob:
    job_id: str
    agent_type: str
    task: str
    timeout: int
    status: JobStatus
    agent: Optional[BaseAgent] = None
    task_handle: Optional[asyncio.Task] = None


class AgentManager:
    """Manages agent lifecycle and job execution."""
    
    def __init__(self, config: Config):
        self.config = config
        self.jobs: Dict[str, AgentJob] = {}
        self.storage = self._create_storage()
        
        # Agent type mapping
        self.agent_classes = {
            "codex": CodexAgent,
            "claude": ClaudeAgent, 
            "ollama": OllamaAgent
        }
    
    def _create_storage(self) -> BaseStorage:
        """Create storage backend based on configuration."""
        storage_type = self.config.storage.type
        storage_config = self.config.storage.config
        
        if storage_type == "memory":
            return MemoryStorage()
        elif storage_type == "sqlite":
            from .storage.sqlite import SQLiteStorage
            return SQLiteStorage(storage_config.get("database_path", "agentsmcp.db"))
        elif storage_type == "postgresql":
            from .storage.postgresql import PostgreSQLStorage
            return PostgreSQLStorage(storage_config)
        elif storage_type == "redis":
            from .storage.redis import RedisStorage
            return RedisStorage(storage_config)
        else:
            raise ValueError(f"Unknown storage type: {storage_type}")
    
    def _create_agent(self, agent_type: str) -> BaseAgent:
        """Create an agent instance based on type."""
        if agent_type not in self.agent_classes:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        agent_config = self.config.get_agent_config(agent_type)
        if not agent_config:
            raise ValueError(f"No configuration found for agent type: {agent_type}")
        
        agent_class = self.agent_classes[agent_type]
        return agent_class(agent_config, self.config)
    
    async def spawn_agent(self, agent_type: str, task: str, timeout: int = 300) -> str:
        """Spawn a new agent to handle a task."""
        job_id = str(uuid.uuid4())
        
        try:
            # Create agent
            agent = self._create_agent(agent_type)
            
            # Create job status
            status = JobStatus(job_id=job_id, state=JobState.PENDING)
            
            # Create job
            job = AgentJob(
                job_id=job_id,
                agent_type=agent_type,
                task=task,
                timeout=timeout,
                status=status,
                agent=agent
            )
            
            # Store job
            self.jobs[job_id] = job
            await self.storage.store_job_status(status)
            
            # Start task
            task_handle = asyncio.create_task(self._run_agent_task(job))
            job.task_handle = task_handle
            
            return job_id
            
        except Exception as e:
            # Clean up on failure
            if job_id in self.jobs:
                del self.jobs[job_id]
            raise e
    
    async def _run_agent_task(self, job: AgentJob):
        """Run an agent task with timeout and error handling."""
        try:
            # Update status to running
            job.status.state = JobState.RUNNING
            job.status.updated_at = datetime.utcnow()
            await self.storage.store_job_status(job.status)
            
            # Run the task with timeout
            result = await asyncio.wait_for(
                job.agent.execute_task(job.task),
                timeout=job.timeout
            )
            
            # Update status on success
            job.status.state = JobState.COMPLETED
            job.status.output = result
            job.status.updated_at = datetime.utcnow()
            
        except asyncio.TimeoutError:
            job.status.state = JobState.TIMEOUT
            job.status.error = f"Task timed out after {job.timeout} seconds"
            job.status.updated_at = datetime.utcnow()
            
        except asyncio.CancelledError:
            job.status.state = JobState.CANCELLED
            job.status.error = "Task was cancelled"
            job.status.updated_at = datetime.utcnow()
            
        except Exception as e:
            job.status.state = JobState.FAILED
            job.status.error = str(e)
            job.status.updated_at = datetime.utcnow()
        
        finally:
            # Store final status
            await self.storage.store_job_status(job.status)
            
            # Clean up agent resources
            if job.agent:
                await job.agent.cleanup()
    
    async def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Get the status of a job."""
        # Check in-memory first
        if job_id in self.jobs:
            return self.jobs[job_id].status
        
        # Fall back to storage
        return await self.storage.get_job_status(job_id)
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        
        # Cancel the task if it's running
        if job.task_handle and not job.task_handle.done():
            job.task_handle.cancel()
        
        # Update status
        job.status.state = JobState.CANCELLED
        job.status.error = "Job cancelled by user"
        job.status.updated_at = datetime.utcnow()
        
        await self.storage.store_job_status(job.status)
        return True
    
    async def wait_for_completion(self, job_id: str, poll_interval: float = 1.0) -> JobStatus:
        """Wait for a job to complete and return the final status."""
        while True:
            status = await self.get_job_status(job_id)
            
            if not status:
                raise ValueError(f"Job not found: {job_id}")
            
            if status.state in [JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED, JobState.TIMEOUT]:
                return status
            
            await asyncio.sleep(poll_interval)
    
    async def cleanup_completed_jobs(self, max_age_hours: int = 24):
        """Clean up old completed jobs."""
        cutoff = datetime.utcnow().replace(hour=datetime.utcnow().hour - max_age_hours)
        
        to_remove = []
        for job_id, job in self.jobs.items():
            if (job.status.state in [JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED, JobState.TIMEOUT] 
                and job.status.updated_at < cutoff):
                to_remove.append(job_id)
        
        for job_id in to_remove:
            del self.jobs[job_id]
    
    async def shutdown(self):
        """Shutdown the agent manager and clean up resources."""
        # Cancel all running jobs
        for job in self.jobs.values():
            if job.task_handle and not job.task_handle.done():
                job.task_handle.cancel()
        
        # Wait for all tasks to complete
        pending_tasks = [job.task_handle for job in self.jobs.values() 
                        if job.task_handle and not job.task_handle.done()]
        
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)
        
        # Clean up storage
        await self.storage.close()