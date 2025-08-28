import asyncio
import logging
import os
import time
import collections
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, List

from .agents.base import BaseAgent
from .agents.claude_agent import ClaudeAgent
from .agents.codex_agent import CodexAgent
from .agents.ollama_agent import OllamaAgent
from .config import Config
from .models import JobState, JobStatus, TaskEnvelopeV1, ResultEnvelopeV1, EnvelopeStatus
from .events import EventBus
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

    def __init__(self, config: Config, events: EventBus | None = None, orchestrator_bus: Optional[object] = None):
        self.config = config
        self.log = logging.getLogger(__name__)
        self.jobs: Dict[str, AgentJob] = {}
        self.storage = self._create_storage()
        self.events = events
        # Optional typed event bus from agentsmcp.orchestration
        self.orchestrator_bus = orchestrator_bus
        # Global concurrency semaphore to provide lightweight backpressure.
        try:
            max_conc = int(getattr(self.config, "concurrent_agents", 4))
        except Exception:
            max_conc = 4
        self._concurrency = asyncio.Semaphore(max(1, max_conc))

        # --- Queue + worker pool (lazy start) ---
        self._queue: asyncio.Queue[AgentJob | None] = asyncio.Queue()
        self._workers: List[asyncio.Task] = []
        self._workers_started: bool = False
        self._shutdown: bool = False

        # Per-provider semaphores (default cap=2; override via AGENTSMCP_PROVIDER_CAP_{PROVIDER})
        self._provider_caps: Dict[str, asyncio.Semaphore] = {}
        self._default_provider_cap = int(os.getenv("AGENTSMCP_PROVIDER_CAP_DEFAULT", "2"))

        # Basic in-memory metrics
        self.metrics = {
            "queued": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "durations": collections.deque(maxlen=200),  # type: ignore[var-annotated]
        }

        # Agent type mapping
        self.agent_classes = {
            "codex": CodexAgent,
            "claude": ClaudeAgent,
            "ollama": OllamaAgent,
        }

    # -----------------------------
    # Role-based execution (P1)
    # -----------------------------
    async def execute_role_task(
        self,
        task: TaskEnvelopeV1,
        *,
        timeout: int = 300,
        max_retries: int = 1,
    ) -> ResultEnvelopeV1:
        """Route a TaskEnvelopeV1 to an appropriate role and execute it.

        Maintains backward compatibility by leveraging the existing agent
        spawning pipeline. Publishes typed orchestration events when the
        orchestration bus is provided.
        """
        # Lazy import to avoid hard dependency if not used
        try:
            from .roles.registry import RoleRegistry
        except Exception as e:  # pragma: no cover
            self.log.error("Failed to import RoleRegistry: %s", e)
            return ResultEnvelopeV1(status=EnvelopeStatus.ERROR, notes=str(e))

        registry = RoleRegistry()
        role, decision = registry.route(task)
        self.log.info("Routing task to role=%s agent=%s", role.name().value, decision.agent_type)

        # Publish orchestration typed event if available
        if self.orchestrator_bus is not None:
            try:
                from .orchestration import JobStarted
                await self.orchestrator_bus.publish(
                    JobStarted(
                        job_id=str(uuid.uuid4()),
                        agent_type=decision.agent_type,
                        task=task.objective,
                        timestamp=datetime.utcnow(),
                    )
                )
            except Exception as e:  # pragma: no cover - non-fatal
                self.log.debug("Orchestrator publish failed: %s", e)

        # Execute via role wrapper (which uses spawn_agent internally)
        start = datetime.utcnow()
        try:
            result = await role.execute(
                task, self, timeout=timeout, max_retries=max_retries
            )
            duration = (datetime.utcnow() - start).total_seconds()

            if self.orchestrator_bus is not None:
                try:
                    from .orchestration import JobCompleted
                    await self.orchestrator_bus.publish(
                        JobCompleted(
                            job_id="",
                            result=result.model_dump(),  # type: ignore[union-attr]
                            duration=duration,
                            timestamp=datetime.utcnow(),
                        )
                    )
                except Exception as e:  # pragma: no cover
                    self.log.debug("Orchestrator publish failed: %s", e)
            return result
        except Exception as e:
            # Publish failure
            if self.orchestrator_bus is not None:
                try:
                    from .orchestration import JobFailed
                    await self.orchestrator_bus.publish(
                        JobFailed(job_id="", error=e, timestamp=datetime.utcnow())
                    )
                except Exception:  # pragma: no cover
                    pass
            return ResultEnvelopeV1(status=EnvelopeStatus.ERROR, notes=str(e))

    def _create_storage(self) -> BaseStorage:
        """Create storage backend based on configuration."""
        storage_type = self.config.storage.type
        storage_config = self.config.storage.config

        if (
            str(storage_type) == "StorageType.MEMORY"
            or storage_type == getattr(type(storage_type), "MEMORY", None)
            or getattr(storage_type, "value", storage_type) == "memory"
        ):
            return MemoryStorage()
        elif getattr(storage_type, "value", storage_type) == "sqlite":
            from .storage.sqlite import SQLiteStorage

            return SQLiteStorage(storage_config.get("database_path", "agentsmcp.db"))
        elif getattr(storage_type, "value", storage_type) == "postgresql":
            from .storage.postgresql import PostgreSQLStorage

            return PostgreSQLStorage(storage_config)
        elif getattr(storage_type, "value", storage_type) == "redis":
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
                agent=agent,
            )

            # Store job
            self.jobs[job_id] = job
            await self.storage.store_job_status(status)

            # Enqueue job for workers
            if not self._workers_started:
                self._start_workers()
            await self._queue.put(job)
            self.metrics["queued"] += 1
            self.log.info("Enqueued job %s with agent %s", job_id, agent_type)
            if self.events:
                await self.events.publish({
                    "type": "job.spawned",
                    "job_id": job_id,
                    "agent_type": agent_type,
                })

            return job_id

        except Exception as e:
            # Clean up on failure
            self.log.exception("Failed to spawn job %s: %s", job_id, e)
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
            self.log.info("Job %s started", job.job_id)
            if self.events:
                await self.events.publish({
                    "type": "job.started",
                    "job_id": job.job_id,
                })

            # Enforce global + per-provider caps with lightweight backpressure
            provider_name = getattr(getattr(job.agent, "agent_config", None), "provider", None)
            provider_name = getattr(provider_name, "value", None) or "default"
            prov_sem = self._provider_caps.get(provider_name)
            if prov_sem is None:
                cap_env = int(os.getenv(f"AGENTSMCP_PROVIDER_CAP_{provider_name.upper()}", str(self._default_provider_cap)))
                prov_sem = asyncio.Semaphore(max(1, cap_env))
                self._provider_caps[provider_name] = prov_sem

            async with self._concurrency:
                async with prov_sem:
                    # Run the task with timeout
                    result = await asyncio.wait_for(
                        job.agent.execute_task(job.task), timeout=job.timeout
                    )

            # Update status on success
            job.status.state = JobState.COMPLETED
            job.status.output = result
            job.status.updated_at = datetime.utcnow()
            self.log.info("Job %s completed", job.job_id)
            if self.events:
                await self.events.publish({
                    "type": "job.completed",
                    "job_id": job.job_id,
                })

        except asyncio.TimeoutError:
            job.status.state = JobState.TIMEOUT
            job.status.error = f"Task timed out after {job.timeout} seconds"
            job.status.updated_at = datetime.utcnow()
            self.log.warning("Job %s timed out", job.job_id)
            if self.events:
                await self.events.publish({
                    "type": "job.timeout",
                    "job_id": job.job_id,
                })

        except asyncio.CancelledError:
            job.status.state = JobState.CANCELLED
            job.status.error = "Task was cancelled"
            job.status.updated_at = datetime.utcnow()
            self.log.info("Job %s cancelled", job.job_id)
            if self.events:
                await self.events.publish({
                    "type": "job.cancelled",
                    "job_id": job.job_id,
                })

        except Exception as e:
            job.status.state = JobState.FAILED
            job.status.error = str(e)
            job.status.updated_at = datetime.utcnow()
            self.log.exception("Job %s failed: %s", job.job_id, e)
            if self.events:
                await self.events.publish({
                    "type": "job.failed",
                    "job_id": job.job_id,
                })

        finally:
            # Store final status
            await self.storage.store_job_status(job.status)

            # Clean up agent resources
            if job.agent:
                await job.agent.cleanup()
            self.log.debug("Job %s resources cleaned", job.job_id)

    def _start_workers(self) -> None:
        if self._workers_started:
            return
        self._workers_started = True
        worker_count = max(1, int(getattr(self.config, "concurrent_agents", 4)))
        for _ in range(worker_count):
            self._workers.append(asyncio.create_task(self._worker_loop()))

    async def _worker_loop(self) -> None:
        while not self._shutdown:
            try:
                job_or_none = await self._queue.get()
            except asyncio.CancelledError:
                break
            if job_or_none is None:
                break
            job = job_or_none
            # Metrics transitions
            self.metrics["queued"] = max(0, self.metrics["queued"] - 1)
            self.metrics["running"] += 1
            start = time.monotonic()
            try:
                await self._run_agent_task(job)
                self.metrics["completed"] += 1
            except Exception:
                self.metrics["failed"] += 1
            finally:
                self.metrics["running"] = max(0, self.metrics["running"] - 1)
                self.metrics["durations"].append(time.monotonic() - start)
                self._queue.task_done()

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
        self.log.info("Job %s cancelled by user", job_id)
        if self.events:
            await self.events.publish({
                "type": "job.cancelled",
                "job_id": job_id,
            })
        return True

    async def wait_for_completion(
        self, job_id: str, poll_interval: float = 1.0
    ) -> JobStatus:
        """Wait for a job to complete and return the final status."""
        while True:
            status = await self.get_job_status(job_id)

            if not status:
                raise ValueError(f"Job not found: {job_id}")

            if status.state in [
                JobState.COMPLETED,
                JobState.FAILED,
                JobState.CANCELLED,
                JobState.TIMEOUT,
            ]:
                return status

            await asyncio.sleep(poll_interval)

    async def cleanup_completed_jobs(self, max_age_hours: int = 24):
        """Clean up old completed jobs."""
        cutoff = datetime.utcnow().replace(hour=datetime.utcnow().hour - max_age_hours)

        to_remove = []
        for job_id, job in self.jobs.items():
            if (
                job.status.state
                in [
                    JobState.COMPLETED,
                    JobState.FAILED,
                    JobState.CANCELLED,
                    JobState.TIMEOUT,
                ]
                and job.status.updated_at < cutoff
            ):
                to_remove.append(job_id)

        for job_id in to_remove:
            del self.jobs[job_id]

    async def shutdown(self):
        """Shutdown the agent manager and clean up resources."""
        self._shutdown = True
        # Signal workers to exit and cancel
        for _ in self._workers:
            await self._queue.put(None)
        for t in self._workers:
            t.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        # Cancel all running jobs
        for job in self.jobs.values():
            if job.task_handle and not job.task_handle.done():
                job.task_handle.cancel()

        # Wait for all tasks to complete
        pending_tasks = [
            job.task_handle
            for job in self.jobs.values()
            if job.task_handle and not job.task_handle.done()
        ]

        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)

        # Clean up storage
        await self.storage.close()
        self.log.info("Agent manager shutdown complete")

    def queue_size(self) -> int:
        """Return the number of jobs waiting in the queue."""
        return self._queue.qsize()
