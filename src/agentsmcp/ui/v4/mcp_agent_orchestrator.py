"""
MCP Agent Orchestrator for V4 Workspace Controller

This module provides real MCP agent spawning and orchestration capabilities for the
V4 workspace controller, replacing demo agents with actual MCP agent integration.

Key Features:
- Real MCP agent spawning using existing AgentManager infrastructure
- Sequential thinking capture from MCP agents via sequential-thinking tool
- Process coach integration for self-improving loops
- Live progress tracking and status monitoring
- Agent lifecycle management with proper error handling
"""

import asyncio
import logging
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

from ...agent_manager import AgentManager
from ...orchestration.process_coach import ProcessCoach, CoachTriggerType, ProcessCoachConfig
from ...config import Config
from ...events import EventBus
from ...models import TaskEnvelopeV1, JobState


logger = logging.getLogger(__name__)


class MCPAgentType(Enum):
    """MCP Agent types available for spawning."""
    CODEX = "codex"
    CLAUDE = "claude" 
    OLLAMA = "ollama"
    SYSTEM_ARCHITECT = "system-architect"
    QA_REVIEWER = "qa-logic-reviewer"
    CODER = "coder"
    PROCESS_COACH = "process-coach"
    
    
@dataclass
class MCPAgentTask:
    """Represents a task to be executed by an MCP agent."""
    task_id: str
    agent_type: MCPAgentType
    objective: str
    context: Dict[str, Any]
    timeout: int = 300
    priority: int = 1
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class MCPAgentSession:
    """Represents an active MCP agent session."""
    session_id: str
    agent_type: MCPAgentType
    job_id: str
    task: MCPAgentTask
    status: str
    progress: float = 0.0
    current_step: str = ""
    sequential_thoughts: List[Dict[str, Any]] = None
    output_lines: List[Dict[str, Any]] = None
    error_count: int = 0
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.sequential_thoughts is None:
            self.sequential_thoughts = []
        if self.output_lines is None:
            self.output_lines = []
        if self.created_at is None:
            self.created_at = datetime.now()


class MCPAgentOrchestrator:
    """
    Orchestrates MCP agents for the V4 workspace controller.
    
    Provides real agent spawning, sequential thinking capture, and process coach
    integration to create a genuine Mission Control Center for AI agents.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the MCP agent orchestrator."""
        self.config = config or Config()
        self.event_bus = EventBus()
        
        # Initialize core components
        self.agent_manager = AgentManager(
            config=self.config,
            events=self.event_bus
        )
        
        self.process_coach = ProcessCoach(ProcessCoachConfig())
        
        # Track active sessions
        self.active_sessions: Dict[str, MCPAgentSession] = {}
        self.completed_sessions: List[MCPAgentSession] = []
        
        # Event callbacks for workspace UI updates
        self.status_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        self.thinking_callbacks: List[Callable[[str, str], None]] = []
        self.output_callbacks: List[Callable[[str, str], None]] = []
        self.process_coach_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        
        # Task queue for agent work
        self.task_queue = asyncio.Queue()
        self._orchestrator_running = False
        self._orchestrator_task = None
        
        logger.info("MCP Agent Orchestrator initialized")

    def add_status_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add callback for status updates."""
        self.status_callbacks.append(callback)
    
    def add_thinking_callback(self, callback: Callable[[str, str], None]):
        """Add callback for sequential thinking updates."""
        self.thinking_callbacks.append(callback)
        
    def add_output_callback(self, callback: Callable[[str, str], None]):
        """Add callback for output updates."""
        self.output_callbacks.append(callback)
    
    def add_process_coach_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add callback for Process Coach updates."""
        self.process_coach_callbacks.append(callback)
        
    async def start_orchestrator(self):
        """Start the orchestrator event loop."""
        if self._orchestrator_running:
            return
            
        self._orchestrator_running = True
        self._orchestrator_task = asyncio.create_task(self._orchestrator_loop())
        
        # Setup event bus listeners
        await self._setup_event_listeners()
        
        logger.info("MCP Agent Orchestrator started")
    
    async def stop_orchestrator(self):
        """Stop the orchestrator and cleanup resources."""
        self._orchestrator_running = False
        
        if self._orchestrator_task:
            self._orchestrator_task.cancel()
            try:
                await self._orchestrator_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all active sessions
        for session in self.active_sessions.values():
            await self._cancel_session(session.session_id)
            
        await self.agent_manager.shutdown()
        logger.info("MCP Agent Orchestrator stopped")
    
    async def spawn_agent(
        self, 
        agent_type: MCPAgentType, 
        objective: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: int = 300,
        priority: int = 1
    ) -> str:
        """
        Spawn a new MCP agent to execute a task.
        
        Returns the session ID for tracking the agent.
        """
        task_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        
        task = MCPAgentTask(
            task_id=task_id,
            agent_type=agent_type,
            objective=objective,
            context=context or {},
            timeout=timeout,
            priority=priority
        )
        
        # Create session
        session = MCPAgentSession(
            session_id=session_id,
            agent_type=agent_type,
            job_id="",  # Will be filled when job is spawned
            task=task,
            status="spawning"
        )
        
        self.active_sessions[session_id] = session
        
        # Queue task for processing
        await self.task_queue.put(task)
        
        # Notify UI of new agent
        self._notify_status_update(session_id, {
            'status': 'spawning',
            'agent_type': agent_type.value,
            'objective': objective,
            'progress': 0.0
        })
        
        logger.info(f"Spawned MCP agent {agent_type.value} with session {session_id}")
        return session_id
    
    async def _orchestrator_loop(self):
        """Main orchestrator event loop."""
        while self._orchestrator_running:
            try:
                # Process queued tasks
                try:
                    task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                    await self._process_task(task)
                except asyncio.TimeoutError:
                    continue
                
                # Update progress for active sessions
                await self._update_session_progress()
                
                # Process coach integration - trigger improvement cycles
                await self._process_coach_integration()
                
            except Exception as e:
                logger.error(f"Error in orchestrator loop: {e}")
                await asyncio.sleep(1)
    
    async def _process_task(self, task: MCPAgentTask):
        """Process a queued task by spawning the appropriate agent."""
        try:
            # Find the session for this task
            session = None
            for s in self.active_sessions.values():
                if s.task.task_id == task.task_id:
                    session = s
                    break
            
            if not session:
                logger.error(f"No session found for task {task.task_id}")
                return
            
            # Map MCP agent type to internal agent type
            internal_agent_type = self._map_mcp_to_internal_agent_type(task.agent_type)
            
            # Create TaskEnvelope for the agent
            task_envelope = TaskEnvelopeV1(
                objective=task.objective,
                bounded_context={
                    "repo": task.context.get("repo", "."),
                    "module": task.context.get("module", "")
                },
                inputs=task.context.get("inputs", {}),
                constraints={
                    "time_s": task.timeout,
                    "write_paths": task.context.get("write_paths", []),
                    "read_only_paths": task.context.get("read_only_paths", [])
                }
            )
            
            # Execute via role-based execution
            session.status = "executing"
            session.started_at = datetime.now()
            
            self._notify_status_update(session.session_id, {
                'status': 'executing',
                'progress': 0.1
            })
            
            # Execute the task
            result = await self.agent_manager.execute_role_task(
                task_envelope,
                timeout=task.timeout
            )
            
            # Update session with results
            session.status = "completed" if result.status.value == "ok" else "error"
            session.completed_at = datetime.now()
            session.progress = 1.0
            
            # Move to completed sessions
            self.completed_sessions.append(session)
            del self.active_sessions[session.session_id]
            
            # Notify UI
            self._notify_status_update(session.session_id, {
                'status': session.status,
                'progress': 1.0,
                'result': result.model_dump() if hasattr(result, 'model_dump') else str(result)
            })
            
            # Trigger process coach improvement cycle
            await self.process_coach.trigger_improvement_cycle(
                CoachTriggerType.TASK_COMPLETION,
                context={
                    'task': task.objective,
                    'agent_type': task.agent_type.value,
                    'result': result.model_dump() if hasattr(result, 'model_dump') else str(result),
                    'session_id': session.session_id
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing task {task.task_id}: {e}")
            if session:
                session.status = "error"
                session.error_count += 1
                session.completed_at = datetime.now()
                
                self._notify_status_update(session.session_id, {
                    'status': 'error',
                    'error': str(e)
                })
    
    def _map_mcp_to_internal_agent_type(self, mcp_type: MCPAgentType) -> str:
        """Map MCP agent types to internal agent system types."""
        mapping = {
            MCPAgentType.CODEX: "self",  # Use SelfAgent for all MCP agents
            MCPAgentType.CLAUDE: "self",
            MCPAgentType.OLLAMA: "self", 
            MCPAgentType.SYSTEM_ARCHITECT: "system-architect",
            MCPAgentType.QA_REVIEWER: "qa-logic-reviewer",
            MCPAgentType.CODER: "coder-c1",
            MCPAgentType.PROCESS_COACH: "process-coach"
        }
        return mapping.get(mcp_type, "self")
    
    async def _setup_event_listeners(self):
        """Setup event bus listeners for agent events."""
        # Listen for job events from agent manager
        async def on_job_started(event):
            if event.get('type') == 'job.started':
                job_id = event.get('job_id')
                # Find session by job_id and update
                for session in self.active_sessions.values():
                    if session.job_id == job_id:
                        session.status = "executing"
                        self._notify_status_update(session.session_id, {
                            'status': 'executing',
                            'progress': 0.2
                        })
                        break
        
        # Subscribe to events
        if hasattr(self.event_bus, 'subscribe'):
            await self.event_bus.subscribe('job.*', on_job_started)
    
    async def _update_session_progress(self):
        """Update progress for active sessions."""
        for session in self.active_sessions.values():
            if session.status == "executing":
                # Simulate progress based on elapsed time
                if session.started_at:
                    elapsed = (datetime.now() - session.started_at).total_seconds()
                    # Rough progress estimation based on timeout
                    session.progress = min(0.2 + (elapsed / session.task.timeout) * 0.7, 0.9)
                    
                    self._notify_status_update(session.session_id, {
                        'progress': session.progress
                    })
    
    async def _process_coach_integration(self):
        """Integrate with process coach for continuous improvement."""
        # Check if we should trigger improvement cycles
        completed_count = len(self.completed_sessions)
        
        # Trigger improvement cycle every 5 completed tasks
        if completed_count > 0 and completed_count % 5 == 0:
            # Check if we haven't triggered recently
            last_cycle_time = getattr(self, '_last_process_coach_cycle', None)
            if (not last_cycle_time or 
                (datetime.now() - last_cycle_time).total_seconds() > 300):  # 5 minutes
                
                await self.process_coach.trigger_improvement_cycle(
                    CoachTriggerType.SCHEDULED_CYCLE,
                    context={
                        'total_completed_sessions': completed_count,
                        'active_sessions': len(self.active_sessions),
                        'orchestrator': 'mcp_agent_orchestrator'
                    }
                )
                self._last_process_coach_cycle = datetime.now()
                
                # Notify Process Coach callbacks
                message = f"Improvement cycle triggered ({completed_count} tasks completed)"
                cycle_data = {
                    'completed_sessions': completed_count,
                    'active_sessions': len(self.active_sessions),
                    'cycle_time': datetime.now().isoformat()
                }
                self._notify_process_coach_update(message, cycle_data)
    
    def _notify_status_update(self, session_id: str, status_data: Dict[str, Any]):
        """Notify UI callbacks of status updates."""
        for callback in self.status_callbacks:
            try:
                callback(session_id, status_data)
            except Exception as e:
                logger.error(f"Error in status callback: {e}")
    
    def _notify_thinking_update(self, session_id: str, thought: str):
        """Notify UI callbacks of sequential thinking updates."""
        for callback in self.thinking_callbacks:
            try:
                callback(session_id, thought)
            except Exception as e:
                logger.error(f"Error in thinking callback: {e}")
    
    def _notify_output_update(self, session_id: str, output: str):
        """Notify UI callbacks of output updates."""  
        for callback in self.output_callbacks:
            try:
                callback(session_id, output)
            except Exception as e:
                logger.error(f"Error in output callback: {e}")
    
    def _notify_process_coach_update(self, message: str, cycle_data: Dict[str, Any]):
        """Notify UI callbacks of Process Coach updates."""
        for callback in self.process_coach_callbacks:
            try:
                callback(message, cycle_data)
            except Exception as e:
                logger.error(f"Error in Process Coach callback: {e}")
    
    async def _cancel_session(self, session_id: str) -> bool:
        """Cancel an active session."""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        
        # Cancel the job if we have a job_id
        if session.job_id:
            await self.agent_manager.cancel_job(session.job_id)
        
        # Update session status
        session.status = "cancelled"
        session.completed_at = datetime.now()
        
        # Move to completed
        self.completed_sessions.append(session)
        del self.active_sessions[session_id]
        
        self._notify_status_update(session_id, {
            'status': 'cancelled'
        })
        
        return True
    
    async def pause_session(self, session_id: str) -> bool:
        """Pause an active session."""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        session.status = "paused"
        
        self._notify_status_update(session_id, {
            'status': 'paused'
        })
        
        return True
    
    async def resume_session(self, session_id: str) -> bool:
        """Resume a paused session."""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        session.status = "executing"
        
        self._notify_status_update(session_id, {
            'status': 'executing'
        })
        
        return True
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a session."""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
        else:
            # Check completed sessions
            session = next((s for s in self.completed_sessions if s.session_id == session_id), None)
        
        if not session:
            return None
        
        return {
            'session_id': session_id,
            'agent_type': session.agent_type.value,
            'status': session.status,
            'progress': session.progress,
            'current_step': session.current_step,
            'objective': session.task.objective,
            'created_at': session.created_at.isoformat(),
            'started_at': session.started_at.isoformat() if session.started_at else None,
            'completed_at': session.completed_at.isoformat() if session.completed_at else None,
            'sequential_thoughts': session.sequential_thoughts[-10:],  # Last 10 thoughts
            'output_lines': session.output_lines[-20:],  # Last 20 output lines
            'error_count': session.error_count
        }
    
    def get_all_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all sessions."""
        all_sessions = {}
        
        # Active sessions
        for session_id in self.active_sessions:
            all_sessions[session_id] = self.get_session_status(session_id)
        
        # Recent completed sessions (last 20)
        for session in self.completed_sessions[-20:]:
            all_sessions[session.session_id] = self.get_session_status(session.session_id)
        
        return all_sessions
    
    async def get_process_coach_status(self) -> Dict[str, Any]:
        """Get current process coach status."""
        return await self.process_coach.get_improvement_status()
    
    def get_orchestrator_metrics(self) -> Dict[str, Any]:
        """Get orchestrator performance metrics."""
        total_sessions = len(self.active_sessions) + len(self.completed_sessions)
        completed_count = len(self.completed_sessions)
        success_count = sum(1 for s in self.completed_sessions if s.status == "completed")
        
        return {
            'total_sessions': total_sessions,
            'active_sessions': len(self.active_sessions),
            'completed_sessions': completed_count,
            'success_rate': success_count / max(completed_count, 1),
            'average_task_duration': self._calculate_average_duration(),
            'orchestrator_running': self._orchestrator_running,
            'queue_size': self.task_queue.qsize()
        }
    
    def _calculate_average_duration(self) -> float:
        """Calculate average task duration for completed sessions."""
        durations = []
        for session in self.completed_sessions:
            if session.started_at and session.completed_at:
                duration = (session.completed_at - session.started_at).total_seconds()
                durations.append(duration)
        
        return sum(durations) / len(durations) if durations else 0.0


# Convenience functions for integration

async def create_orchestrator(config: Optional[Config] = None) -> MCPAgentOrchestrator:
    """Create and start an MCP agent orchestrator."""
    orchestrator = MCPAgentOrchestrator(config)
    await orchestrator.start_orchestrator()
    return orchestrator


def get_available_agent_types() -> List[str]:
    """Get list of available MCP agent types."""
    return [agent_type.value for agent_type in MCPAgentType]


def create_coding_task(objective: str, files: List[str] = None) -> Dict[str, Any]:
    """Create a coding task context."""
    return {
        "objective": objective,
        "context": {
            "write_paths": files or [],
            "repo": ".",
            "inputs": {
                "task_type": "coding",
                "files_to_modify": files or []
            }
        }
    }


def create_analysis_task(objective: str, target: str = None) -> Dict[str, Any]:
    """Create an analysis task context."""
    return {
        "objective": objective,
        "context": {
            "read_only_paths": [target] if target else [],
            "repo": ".",
            "inputs": {
                "task_type": "analysis",
                "analysis_target": target or "codebase"
            }
        }
    }