"""Execution engine for coordinated agent team execution.

This module implements various execution strategies for agent teams including
parallel, sequential, and hierarchical coordination. It provides progress tracking,
result aggregation, timeout handling, and error recovery.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable, Union, Set, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum

from .models import TeamComposition, CoordinationStrategy, AgentSpec, TaskResult, TeamPerformanceMetrics
from .agent_loader import AgentLoader, LoadedAgent, AgentState
from .resource_manager import ResourceManager
from ..roles.base import RoleName, ModelAssignment, BaseRole
from ..models import TaskEnvelopeV1, ResultEnvelopeV1, EnvelopeStatus

if TYPE_CHECKING:
    from ..agent_manager import AgentManager


class ExecutionStatus(str, Enum):
    """Execution status for team coordination."""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskPriority(int, Enum):
    """Task priority levels."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class ExecutionTask:
    """Individual task within team execution."""
    task_id: str
    agent_spec: AgentSpec
    envelope: TaskEnvelopeV1
    priority: TaskPriority = TaskPriority.MEDIUM
    dependencies: Set[str] = field(default_factory=set)
    timeout_seconds: int = 300
    retries_remaining: int = 2
    assigned_agent: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[ResultEnvelopeV1] = None
    error: Optional[str] = None


@dataclass
class ExecutionProgress:
    """Progress tracking for team execution."""
    total_tasks: int
    completed_tasks: int = 0
    failed_tasks: int = 0
    running_tasks: int = 0
    
    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage."""
        return (self.completed_tasks / self.total_tasks * 100) if self.total_tasks > 0 else 0.0
    
    @property
    def is_complete(self) -> bool:
        """Check if execution is complete."""
        return self.completed_tasks + self.failed_tasks >= self.total_tasks


@dataclass
class TeamExecution:
    """Represents a complete team execution with results."""
    execution_id: str
    team_composition: TeamComposition
    objective: str
    status: ExecutionStatus
    progress: ExecutionProgress
    tasks: Dict[str, ExecutionTask] = field(default_factory=dict)
    results: Dict[str, ResultEnvelopeV1] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_duration_seconds: float = 0.0
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class ExecutionStrategy(ABC):
    """Abstract base class for execution strategies."""
    
    def __init__(
        self,
        agent_loader: AgentLoader,
        resource_manager: ResourceManager,
        agent_manager: "AgentManager",
    ):
        self.agent_loader = agent_loader
        self.resource_manager = resource_manager
        self.agent_manager = agent_manager
        self.log = logging.getLogger(__name__)
    
    @abstractmethod
    async def execute(
        self,
        execution: TeamExecution,
        progress_callback: Optional[Callable[[ExecutionProgress], None]] = None,
    ) -> TeamExecution:
        """Execute the team coordination strategy."""
        pass
    
    async def _notify_progress(
        self,
        progress: ExecutionProgress,
        callback: Optional[Callable[[ExecutionProgress], None]],
    ) -> None:
        """Notify progress callback if provided."""
        if callback:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(progress)
                else:
                    callback(progress)
            except Exception as e:
                self.log.error("Progress callback failed: %s", e)


class ParallelExecution(ExecutionStrategy):
    """Parallel execution strategy - run all agents concurrently."""
    
    def __init__(
        self,
        agent_loader: AgentLoader,
        resource_manager: ResourceManager,
        agent_manager: "AgentManager",
        max_concurrent_tasks: int = 10,
    ):
        super().__init__(agent_loader, resource_manager, agent_manager)
        self.max_concurrent_tasks = max_concurrent_tasks
    
    async def execute(
        self,
        execution: TeamExecution,
        progress_callback: Optional[Callable[[ExecutionProgress], None]] = None,
    ) -> TeamExecution:
        """Execute all tasks in parallel with concurrency control."""
        self.log.info("Starting parallel execution for %s", execution.execution_id)
        
        execution.status = ExecutionStatus.RUNNING
        execution.started_at = datetime.now(timezone.utc)
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        
        # Create tasks for all agents
        async_tasks = []
        for task_id, task in execution.tasks.items():
            async_task = asyncio.create_task(
                self._execute_single_task(task, semaphore, execution, progress_callback)
            )
            async_tasks.append(async_task)
        
        # Wait for all tasks to complete
        try:
            await asyncio.gather(*async_tasks, return_exceptions=True)
        except Exception as e:
            execution.errors.append(f"Parallel execution error: {e}")
            execution.status = ExecutionStatus.FAILED
        
        # Finalize execution
        execution.completed_at = datetime.now(timezone.utc)
        if execution.started_at:
            execution.total_duration_seconds = (
                execution.completed_at - execution.started_at
            ).total_seconds()
        
        # Determine final status
        if execution.status == ExecutionStatus.RUNNING:
            if execution.progress.failed_tasks > 0:
                execution.status = ExecutionStatus.FAILED
            else:
                execution.status = ExecutionStatus.COMPLETED
        
        self.log.info("Parallel execution completed for %s in %.2fs", 
                     execution.execution_id, execution.total_duration_seconds)
        return execution
    
    async def _execute_single_task(
        self,
        task: ExecutionTask,
        semaphore: asyncio.Semaphore,
        execution: TeamExecution,
        progress_callback: Optional[Callable[[ExecutionProgress], None]],
    ) -> None:
        """Execute a single task with concurrency control."""
        async with semaphore:
            try:
                await self._run_task_with_agent(task, execution)
                execution.progress.completed_tasks += 1
            except Exception as e:
                self.log.error("Task %s failed: %s", task.task_id, e)
                task.error = str(e)
                execution.progress.failed_tasks += 1
                execution.errors.append(f"Task {task.task_id}: {e}")
            finally:
                execution.progress.running_tasks = max(0, execution.progress.running_tasks - 1)
                await self._notify_progress(execution.progress, progress_callback)


class SequentialExecution(ExecutionStrategy):
    """Sequential execution strategy - run agents one after another."""
    
    async def execute(
        self,
        execution: TeamExecution,
        progress_callback: Optional[Callable[[ExecutionProgress], None]] = None,
    ) -> TeamExecution:
        """Execute tasks sequentially in dependency order."""
        self.log.info("Starting sequential execution for %s", execution.execution_id)
        
        execution.status = ExecutionStatus.RUNNING
        execution.started_at = datetime.now(timezone.utc)
        
        # Sort tasks by priority and dependencies
        ordered_tasks = self._sort_tasks_for_sequential_execution(execution.tasks)
        
        # Execute tasks one by one
        for task in ordered_tasks:
            try:
                execution.progress.running_tasks = 1
                await self._notify_progress(execution.progress, progress_callback)
                
                await self._run_task_with_agent(task, execution)
                execution.progress.completed_tasks += 1
                
            except Exception as e:
                self.log.error("Task %s failed: %s", task.task_id, e)
                task.error = str(e)
                execution.progress.failed_tasks += 1
                execution.errors.append(f"Task {task.task_id}: {e}")
                
                # For sequential execution, one failure might break the chain
                if task.priority <= TaskPriority.HIGH:
                    self.log.warning("Critical task failed, stopping sequential execution")
                    break
            
            finally:
                execution.progress.running_tasks = 0
                await self._notify_progress(execution.progress, progress_callback)
        
        # Finalize execution
        execution.completed_at = datetime.now(timezone.utc)
        if execution.started_at:
            execution.total_duration_seconds = (
                execution.completed_at - execution.started_at
            ).total_seconds()
        
        if execution.progress.failed_tasks > 0:
            execution.status = ExecutionStatus.FAILED
        else:
            execution.status = ExecutionStatus.COMPLETED
        
        self.log.info("Sequential execution completed for %s in %.2fs",
                     execution.execution_id, execution.total_duration_seconds)
        return execution
    
    def _sort_tasks_for_sequential_execution(
        self, 
        tasks: Dict[str, ExecutionTask]
    ) -> List[ExecutionTask]:
        """Sort tasks for sequential execution considering dependencies and priority."""
        # Simple topological sort with priority
        sorted_tasks = []
        remaining_tasks = dict(tasks)
        
        while remaining_tasks:
            # Find tasks with no unmet dependencies
            ready_tasks = []
            for task_id, task in remaining_tasks.items():
                if not task.dependencies or task.dependencies.issubset({
                    t.task_id for t in sorted_tasks
                }):
                    ready_tasks.append(task)
            
            if not ready_tasks:
                # Circular dependency or unresolvable - take highest priority
                ready_tasks = [min(remaining_tasks.values(), key=lambda t: t.priority)]
            
            # Sort ready tasks by priority
            ready_tasks.sort(key=lambda t: t.priority)
            
            # Add highest priority task
            next_task = ready_tasks[0]
            sorted_tasks.append(next_task)
            remaining_tasks.pop(next_task.task_id)
        
        return sorted_tasks


class HierarchicalExecution(ExecutionStrategy):
    """Hierarchical execution strategy - coordinate through primary agent."""
    
    def __init__(
        self,
        agent_loader: AgentLoader,
        resource_manager: ResourceManager,
        agent_manager: "AgentManager",
        coordination_agent_role: RoleName = RoleName.ARCHITECT,
    ):
        super().__init__(agent_loader, resource_manager, agent_manager)
        self.coordination_agent_role = coordination_agent_role
    
    async def execute(
        self,
        execution: TeamExecution,
        progress_callback: Optional[Callable[[ExecutionProgress], None]] = None,
    ) -> TeamExecution:
        """Execute with hierarchical coordination through a primary agent."""
        self.log.info("Starting hierarchical execution for %s", execution.execution_id)
        
        execution.status = ExecutionStatus.RUNNING
        execution.started_at = datetime.now(timezone.utc)
        
        try:
            # Load coordination agent first
            coordinator = await self._load_coordination_agent(execution)
            
            # Create coordination plan
            plan = await self._create_coordination_plan(coordinator, execution)
            
            # Execute plan phases
            for phase in plan.get("phases", []):
                await self._execute_phase(phase, execution, progress_callback)
            
            execution.status = ExecutionStatus.COMPLETED
            
        except Exception as e:
            self.log.error("Hierarchical execution failed: %s", e)
            execution.status = ExecutionStatus.FAILED
            execution.errors.append(f"Hierarchical execution error: {e}")
        
        # Finalize execution
        execution.completed_at = datetime.now(timezone.utc)
        if execution.started_at:
            execution.total_duration_seconds = (
                execution.completed_at - execution.started_at
            ).total_seconds()
        
        self.log.info("Hierarchical execution completed for %s in %.2fs",
                     execution.execution_id, execution.total_duration_seconds)
        return execution
    
    async def _load_coordination_agent(self, execution: TeamExecution) -> BaseRole:
        """Load the coordination agent."""
        from .agent_loader import AgentContext
        
        # Find coordination agent spec or create one
        coordinator_spec = None
        for spec in execution.team_composition.primary_team:
            if spec.role == self.coordination_agent_role.value:
                coordinator_spec = spec
                break
        
        if not coordinator_spec:
            coordinator_spec = AgentSpec(
                role=self.coordination_agent_role.value,
                model_assignment="codex",  # Default to codex for coordination
                priority=1,
            )
        
        context = AgentContext(
            team_id=execution.execution_id,
            objective=execution.objective,
            role_spec=coordinator_spec,
            resource_constraints=None,  # Use defaults
        )
        
        return await self.agent_loader.load_agent(
            self.coordination_agent_role,
            ModelAssignment(agent_type=coordinator_spec.model_assignment),
            context,
        )
    
    async def _create_coordination_plan(
        self,
        coordinator: BaseRole,
        execution: TeamExecution,
    ) -> Dict[str, Any]:
        """Create coordination plan through the coordination agent."""
        planning_envelope = TaskEnvelopeV1(
            objective=f"Create coordination plan for: {execution.objective}",
            inputs={
                "team_composition": execution.team_composition.model_dump(),
                "tasks": {tid: task.envelope.model_dump() for tid, task in execution.tasks.items()},
            },
            constraints=["Create phases with task dependencies", "Optimize for efficiency"],
        )
        
        result = await coordinator.execute(
            planning_envelope,
            self.agent_manager,
        )
        
        # Extract plan from result
        plan = result.artifacts.get("coordination_plan", {
            "phases": [{"name": "default", "tasks": list(execution.tasks.keys())}]
        })
        
        return plan
    
    async def _execute_phase(
        self,
        phase: Dict[str, Any],
        execution: TeamExecution,
        progress_callback: Optional[Callable[[ExecutionProgress], None]],
    ) -> None:
        """Execute a single phase of the hierarchical plan."""
        phase_name = phase.get("name", "unnamed")
        task_ids = phase.get("tasks", [])
        
        self.log.info("Executing phase '%s' with %d tasks", phase_name, len(task_ids))
        
        # Execute tasks in phase (can be parallel or sequential based on phase config)
        phase_strategy = phase.get("strategy", "parallel")
        
        if phase_strategy == "parallel":
            # Run phase tasks in parallel
            async_tasks = []
            for task_id in task_ids:
                if task_id in execution.tasks:
                    task = execution.tasks[task_id]
                    async_task = asyncio.create_task(
                        self._run_task_with_agent(task, execution)
                    )
                    async_tasks.append(async_task)
            
            # Wait for phase completion
            results = await asyncio.gather(*async_tasks, return_exceptions=True)
            
            # Update progress
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    execution.progress.failed_tasks += 1
                    execution.errors.append(f"Phase {phase_name} task failed: {result}")
                else:
                    execution.progress.completed_tasks += 1
        
        else:
            # Run phase tasks sequentially
            for task_id in task_ids:
                if task_id in execution.tasks:
                    task = execution.tasks[task_id]
                    try:
                        await self._run_task_with_agent(task, execution)
                        execution.progress.completed_tasks += 1
                    except Exception as e:
                        execution.progress.failed_tasks += 1
                        execution.errors.append(f"Phase {phase_name} task {task_id} failed: {e}")
        
        await self._notify_progress(execution.progress, progress_callback)


class ExecutionEngine:
    """Main execution engine that manages different coordination strategies."""
    
    def __init__(
        self,
        agent_loader: AgentLoader,
        resource_manager: ResourceManager,
        agent_manager: "AgentManager",
    ):
        self.agent_loader = agent_loader
        self.resource_manager = resource_manager
        self.agent_manager = agent_manager
        self.log = logging.getLogger(__name__)
        
        # Initialize strategies
        self.strategies: Dict[CoordinationStrategy, ExecutionStrategy] = {
            CoordinationStrategy.PARALLEL: ParallelExecution(
                agent_loader, resource_manager, agent_manager
            ),
            CoordinationStrategy.SEQUENTIAL: SequentialExecution(
                agent_loader, resource_manager, agent_manager
            ),
            CoordinationStrategy.HIERARCHICAL: HierarchicalExecution(
                agent_loader, resource_manager, agent_manager
            ),
        }
        
        # Execution tracking
        self.active_executions: Dict[str, TeamExecution] = {}
        self._execution_lock = asyncio.Lock()
    
    async def execute_team(
        self,
        team_composition: TeamComposition,
        objective: str,
        tasks: List[TaskEnvelopeV1],
        progress_callback: Optional[Callable[[ExecutionProgress], None]] = None,
        timeout_seconds: Optional[int] = None,
    ) -> TeamExecution:
        """Execute a team composition with the specified strategy."""
        
        execution_id = f"exec_{int(time.time())}_{id(team_composition)}"
        
        # Create execution object
        execution = TeamExecution(
            execution_id=execution_id,
            team_composition=team_composition,
            objective=objective,
            status=ExecutionStatus.PENDING,
            progress=ExecutionProgress(total_tasks=len(tasks)),
        )
        
        # Create execution tasks
        for i, task_envelope in enumerate(tasks):
            # Assign agent spec (simple round-robin for now)
            agent_spec = team_composition.primary_team[i % len(team_composition.primary_team)]
            
            task_id = f"task_{i}"
            execution_task = ExecutionTask(
                task_id=task_id,
                agent_spec=agent_spec,
                envelope=task_envelope,
                timeout_seconds=timeout_seconds or 300,
            )
            execution.tasks[task_id] = execution_task
        
        # Register execution
        async with self._execution_lock:
            self.active_executions[execution_id] = execution
        
        try:
            # Get strategy and execute
            strategy = self.strategies[team_composition.coordination_strategy]
            
            if timeout_seconds:
                execution = await asyncio.wait_for(
                    strategy.execute(execution, progress_callback),
                    timeout=timeout_seconds
                )
            else:
                execution = await strategy.execute(execution, progress_callback)
            
        except asyncio.TimeoutError:
            execution.status = ExecutionStatus.TIMEOUT
            execution.errors.append("Execution timed out")
        
        except Exception as e:
            execution.status = ExecutionStatus.FAILED
            execution.errors.append(f"Execution failed: {e}")
            self.log.error("Team execution failed: %s", e)
        
        finally:
            # Unregister execution
            async with self._execution_lock:
                self.active_executions.pop(execution_id, None)
        
        return execution
    
    async def _run_task_with_agent(
        self,
        task: ExecutionTask,
        execution: TeamExecution,
    ) -> None:
        """Run a single task with an appropriate agent."""
        task.started_at = datetime.now(timezone.utc)
        
        # Load agent
        from .agent_loader import AgentContext
        
        role = RoleName(task.agent_spec.role)
        model_assignment = ModelAssignment(agent_type=task.agent_spec.model_assignment)
        
        context = AgentContext(
            team_id=execution.execution_id,
            objective=execution.objective,
            role_spec=task.agent_spec,
            resource_constraints=None,  # Use defaults
        )
        
        agent = await self.agent_loader.load_agent(role, model_assignment, context)
        
        # Execute task
        try:
            result = await agent.execute(task.envelope, self.agent_manager)
            task.result = result
            task.completed_at = datetime.now(timezone.utc)
            
            # Store result
            execution.results[task.task_id] = result
            
        except Exception as e:
            task.error = str(e)
            raise e
    
    def get_execution_status(self, execution_id: str) -> Optional[TeamExecution]:
        """Get status of a specific execution."""
        return self.active_executions.get(execution_id)
    
    def get_active_executions(self) -> Dict[str, TeamExecution]:
        """Get all currently active executions."""
        return self.active_executions.copy()
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running execution."""
        execution = self.active_executions.get(execution_id)
        if execution and execution.status == ExecutionStatus.RUNNING:
            execution.status = ExecutionStatus.CANCELLED
            self.log.info("Cancelled execution %s", execution_id)
            return True
        return False