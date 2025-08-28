"""
Main coordination loop for AGENTS.md v2 two-tier architecture.

Implements single main loop with max-one branch rule, delegating tasks
to stateless agent functions via structured envelopes.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from ..models import (
    TaskEnvelopeV1, 
    ResultEnvelopeV1, 
    EnvelopeStatus,
    build_task_envelope,
    build_result_envelope
)
from ..roles import get_role, RoleName
from .delegation import DelegationEngine
from .state_machine import TaskState, TaskStateMachine
from .quality_gates import QualityGateManager


class MainCoordinator:
    """
    Primary orchestration loop implementing AGENTS.md v2 coordination patterns.
    
    Maintains single-branch, single-loop operation with quality gates,
    delegating work to stateless agent functions via structured envelopes.
    """
    
    def __init__(
        self,
        agent_manager,
        event_bus: Optional[object] = None,
        max_concurrent_tasks: int = 3
    ):
        """Initialize coordinator with dependencies."""
        self.agent_manager = agent_manager
        # Event bus is a typed bus from src.agentsmcp.orchestration (module). Keep loose typing here to avoid import cycles.
        if event_bus is None:
            # Lazy import to avoid circular import during package init
            from src.agentsmcp.orchestration import EventBus  # type: ignore
            self.event_bus = EventBus()
        else:
            self.event_bus = event_bus
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # Core coordination components
        self.delegation_engine = DelegationEngine(agent_manager)
        self.state_machine = TaskStateMachine()
        self.quality_gates = QualityGateManager()
        
        # Active task tracking
        self.active_tasks: Dict[str, TaskState] = {}
        self.task_results: Dict[str, ResultEnvelopeV1] = {}
        
        # Coordination state
        self._running = False
        self._shutdown_event = asyncio.Event()
    
    async def start(self) -> None:
        """Start the main coordination loop."""
        if self._running:
            return
            
        self._running = True
        self._shutdown_event.clear()
        
        # Start coordination loop
        asyncio.create_task(self._coordination_loop())
    
    async def stop(self) -> None:
        """Stop the coordination loop gracefully."""
        self._running = False
        self._shutdown_event.set()
        
        # Wait for active tasks to complete or timeout
        if self.active_tasks:
            await asyncio.wait_for(
                self._wait_for_active_tasks(),
                timeout=30.0
            )
    
    async def submit_task(
        self,
        objective: str,
        bounded_context: Optional[str] = None,
        inputs: Optional[Dict] = None,
        constraints: Optional[List[str]] = None,
        preferred_role: Optional[RoleName] = None
    ) -> str:
        """
        Submit a new task for orchestrated execution.
        
        Returns task_id for tracking progress.
        """
        task_id = str(uuid.uuid4())
        
        # Create task envelope
        task_envelope = TaskEnvelopeV1(
            objective=objective,
            bounded_context=bounded_context,
            inputs=inputs or {},
            constraints=constraints or [],
            routing={"preferred_role": preferred_role.value if preferred_role else None}
        )
        
        # Initialize task state
        task_state = TaskState(
            task_id=task_id,
            envelope=task_envelope,
            status=EnvelopeStatus.PENDING,
            created_at=datetime.now(timezone.utc)
        )
        
        # Queue for processing
        self.active_tasks[task_id] = task_state
        
        # Emit typed JobStarted event lazily-imported to avoid cycle
        from src.agentsmcp.orchestration import JobStarted  # type: ignore
        await self.event_bus.publish(JobStarted(
            job_id=task_id,
            agent_type="coordinator",
            task=objective,
            timestamp=task_state.created_at
        ))
        
        return task_id
    
    async def get_task_result(self, task_id: str) -> Optional[ResultEnvelopeV1]:
        """Get result for completed task."""
        return self.task_results.get(task_id)
    
    async def get_task_status(self, task_id: str) -> Optional[TaskState]:
        """Get current status of task."""
        return self.active_tasks.get(task_id)
    
    async def _coordination_loop(self) -> None:
        """Main coordination loop - processes tasks and maintains quality gates."""
        while self._running:
            try:
                # Process pending tasks
                await self._process_pending_tasks()
                
                # Check quality gates
                await self._evaluate_quality_gates()
                
                # Clean up completed tasks
                await self._cleanup_completed_tasks()
                
                # Brief pause to prevent busy wait
                await asyncio.sleep(0.1)
                
            except Exception as e:
                # Log error but continue coordination
                print(f"Coordination loop error: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_pending_tasks(self) -> None:
        """Process tasks in PENDING state."""
        pending_tasks = [
            task for task in self.active_tasks.values()
            if task.status == EnvelopeStatus.PENDING
        ]
        
        # Respect concurrency limits
        available_slots = self.max_concurrent_tasks - len([
            task for task in self.active_tasks.values()
            if task.status == EnvelopeStatus.PENDING  # Count currently active/running pending
        ])
        
        for task in pending_tasks[:max(1, available_slots)]:
            await self._execute_task(task)
    
    async def _execute_task(self, task_state: TaskState) -> None:
        """Execute a single task through the delegation engine."""
        task_id = task_state.task_id
        
        try:
            # Update state to running (PENDING -> PENDING marks started_at)
            await self.state_machine.transition(task_state, EnvelopeStatus.PENDING)
            
            # Get role assignment
            role = await self._determine_role(task_state.envelope)
            
            # Execute via delegation engine
            result = await self.delegation_engine.execute_task(
                task_state.envelope,
                role
            )
            
            # Store result and update state
            self.task_results[task_id] = result
            await self.state_machine.transition(task_state, result.status)
            
            # Emit completion event (lazy import to avoid circular import at module init)
            if result.status == EnvelopeStatus.SUCCESS:
                from src.agentsmcp.orchestration import JobCompleted  # type: ignore
                await self.event_bus.publish(JobCompleted(
                    job_id=task_id,
                    result=result.model_dump(),
                    duration=(datetime.now(timezone.utc) - task_state.created_at).total_seconds(),
                    timestamp=datetime.now(timezone.utc)
                ))
            else:
                from src.agentsmcp.orchestration import JobFailed  # type: ignore
                await self.event_bus.publish(JobFailed(
                    job_id=task_id,
                    error=Exception(result.notes or "Task failed"),
                    timestamp=datetime.now(timezone.utc)
                ))
                
        except Exception as e:
            # Handle execution error
            error_result = ResultEnvelopeV1(
                status=EnvelopeStatus.ERROR,
                notes=f"Execution failed: {str(e)}",
                confidence=0.0
            )
            
            self.task_results[task_id] = error_result
            await self.state_machine.transition(task_state, EnvelopeStatus.ERROR)
            
            from src.agentsmcp.orchestration import JobFailed  # type: ignore
            await self.event_bus.publish(JobFailed(
                job_id=task_id,
                error=e,
                timestamp=datetime.now(timezone.utc)
            ))
    
    async def _determine_role(self, task: TaskEnvelopeV1) -> RoleName:
        """Determine appropriate role for task based on content analysis."""
        routing_hint = task.routing.get("preferred_role") if task.routing else None
        if routing_hint:
            try:
                return RoleName(routing_hint)
            except ValueError:
                pass
        
        # Analyze task content for role assignment
        objective_lower = task.objective.lower()
        
        if any(word in objective_lower for word in ["design", "architect", "plan", "structure"]):
            return RoleName.ARCHITECT
        elif any(word in objective_lower for word in ["implement", "code", "write", "fix"]):
            return RoleName.CODER
        elif any(word in objective_lower for word in ["test", "verify", "validate", "qa"]):
            return RoleName.QA
        elif any(word in objective_lower for word in ["merge", "review", "approve"]):
            return RoleName.MERGE_BOT
        elif any(word in objective_lower for word in ["document", "docs", "readme"]):
            return RoleName.DOCS
        else:
            # Default to architect for planning
            return RoleName.ARCHITECT
    
    async def _evaluate_quality_gates(self) -> None:
        """Evaluate quality gates for running tasks."""
        for task_state in self.active_tasks.values():
            if task_state.task_id in self.task_results:
                result = self.task_results[task_state.task_id]
                await self.quality_gates.evaluate(task_state, result)
    
    async def _cleanup_completed_tasks(self) -> None:
        """Clean up tasks that have completed or failed."""
        completed_task_ids = [
            task_id for task_id, task in self.active_tasks.items()
            if task.status in [EnvelopeStatus.SUCCESS, EnvelopeStatus.ERROR]
        ]
        
        # Keep results but remove from active tracking after delay
        for task_id in completed_task_ids:
            task = self.active_tasks[task_id]
            if task.completed_at and (
                datetime.now(timezone.utc) - task.completed_at
            ).total_seconds() > 300:  # 5 minute retention
                del self.active_tasks[task_id]
    
    async def _wait_for_active_tasks(self) -> None:
        """Wait for all active tasks to complete."""
        while self.active_tasks:
            await asyncio.sleep(0.1)
