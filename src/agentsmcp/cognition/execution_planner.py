"""
Sub-task scheduling and dependency management with resource allocation.

This module implements intelligent execution planning that optimizes task scheduling,
manages resource allocation, and creates checkpoints for progress validation.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from .models import (
    SubTask, DependencyGraph, ExecutionSchedule, ResourceConstraints,
    ResourceAllocation, Checkpoint, TaskType
)
from .config import ExecutionPlannerConfig, DEFAULT_PLANNER_CONFIG

logger = logging.getLogger(__name__)


class ResourceConflict(Exception):
    """Raised when resource allocation conflicts occur."""
    pass


class UnresolvableDependency(Exception):
    """Raised when dependencies cannot be resolved."""
    pass


class SchedulingFailure(Exception):
    """Raised when scheduling process fails."""
    pass


@dataclass
class SchedulingContext:
    """Context for execution scheduling."""
    start_time: datetime = field(default_factory=datetime.now)
    resource_constraints: Optional[ResourceConstraints] = None
    priority_boost: Dict[str, float] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)


class ExecutionPlanner:
    """
    Sub-task scheduling and dependency management system.
    
    This planner creates optimized execution schedules that respect dependencies,
    manage resource allocation, and include validation checkpoints.
    """
    
    def __init__(self, config: Optional[ExecutionPlannerConfig] = None):
        """Initialize the execution planner."""
        self.config = config or DEFAULT_PLANNER_CONFIG
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Scheduling metrics
        self._total_schedules_created = 0
        self._total_tasks_scheduled = 0
        self._resource_conflicts_resolved = 0
        
        # Resource availability tracking
        self._available_resources: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info("ExecutionPlanner initialized")
    
    async def create_schedule(
        self,
        subtasks: List[SubTask],
        dependency_graph: DependencyGraph,
        constraints: Optional[ResourceConstraints] = None,
        context: Optional[SchedulingContext] = None
    ) -> ExecutionSchedule:
        """
        Create optimized execution schedule for sub-tasks.
        
        Args:
            subtasks: List of sub-tasks to schedule
            dependency_graph: Task dependency relationships
            constraints: Resource constraints and limits
            context: Optional scheduling context
            
        Returns:
            ExecutionSchedule with optimized task ordering and resource allocation
            
        Raises:
            ResourceConflict: If resource allocation conflicts occur
            UnresolvableDependency: If dependencies cannot be resolved
            SchedulingFailure: If scheduling process fails
        """
        if not subtasks:
            return ExecutionSchedule()
        
        context = context or SchedulingContext()
        constraints = constraints or ResourceConstraints()
        
        try:
            self._total_schedules_created += 1
            self._total_tasks_scheduled += len(subtasks)
            
            # Validate dependencies are resolvable
            await self._validate_dependencies(dependency_graph)
            
            # Create execution order based on dependencies and strategy
            execution_order = await self._create_execution_order(
                subtasks, dependency_graph, constraints, context
            )
            
            # Identify parallel execution groups
            parallel_groups = await self._identify_parallel_groups(
                execution_order, dependency_graph, constraints
            )
            
            # Allocate resources to tasks
            resource_allocations = await self._allocate_resources(
                subtasks, execution_order, constraints, context
            )
            
            # Create checkpoints for progress validation
            checkpoints = await self._create_checkpoints(
                execution_order, subtasks, constraints
            )
            
            # Calculate critical path and duration estimates
            critical_path = await self._calculate_critical_path(
                subtasks, dependency_graph, resource_allocations
            )
            
            total_duration = await self._estimate_total_duration(
                execution_order, subtasks, parallel_groups, resource_allocations
            )
            
            # Create execution schedule
            schedule = ExecutionSchedule(
                tasks=subtasks,
                execution_order=execution_order,
                parallel_groups=parallel_groups,
                resource_allocations=resource_allocations,
                checkpoints=checkpoints,
                estimated_total_duration=total_duration,
                critical_path=critical_path
            )
            
            self.logger.info(
                f"Created execution schedule with {len(subtasks)} tasks, "
                f"{len(parallel_groups)} parallel groups, "
                f"estimated duration: {total_duration}"
            )
            
            return schedule
            
        except Exception as e:
            self.logger.error(f"Error creating execution schedule: {e}", exc_info=True)
            raise SchedulingFailure(f"Failed to create execution schedule: {e}")
    
    async def _validate_dependencies(self, dependency_graph: DependencyGraph):
        """Validate that all dependencies can be resolved."""
        # Check for circular dependencies
        cycles = dependency_graph.detect_cycles()
        if cycles:
            raise UnresolvableDependency(f"Circular dependencies detected: {cycles}")
        
        # Validate all dependency references exist
        for task_id, task in dependency_graph.tasks.items():
            for dep_id in task.dependencies:
                if dep_id not in dependency_graph.tasks:
                    raise UnresolvableDependency(
                        f"Task {task_id} depends on non-existent task {dep_id}"
                    )
    
    async def _create_execution_order(
        self,
        subtasks: List[SubTask],
        dependency_graph: DependencyGraph,
        constraints: ResourceConstraints,
        context: SchedulingContext
    ) -> List[str]:
        """Create execution order based on dependencies and optimization strategy."""
        
        if self.config.optimization_strategy == "critical_path":
            return await self._create_critical_path_order(
                subtasks, dependency_graph, constraints, context
            )
        elif self.config.optimization_strategy == "load_balanced":
            return await self._create_load_balanced_order(
                subtasks, dependency_graph, constraints, context
            )
        elif self.config.optimization_strategy == "fastest":
            return await self._create_fastest_order(
                subtasks, dependency_graph, constraints, context
            )
        else:
            # Default to topological sort
            return await self._create_topological_order(
                subtasks, dependency_graph
            )
    
    async def _create_critical_path_order(
        self,
        subtasks: List[SubTask],
        dependency_graph: DependencyGraph,
        constraints: ResourceConstraints,
        context: SchedulingContext
    ) -> List[str]:
        """Create execution order optimized for critical path."""
        # Calculate task priorities based on critical path
        task_priorities = {}
        
        for task in subtasks:
            # Calculate forward path length (how many tasks depend on this one)
            forward_path = await self._calculate_forward_path_length(
                task.id, dependency_graph
            )
            
            # Calculate backward path length (how many dependencies this task has)
            backward_path = len(task.dependencies)
            
            # Combine with task priority and duration
            duration_factor = (
                task.estimated_duration.total_seconds() / 3600 
                if task.estimated_duration else 1
            )
            
            priority_score = (
                task.priority * 0.3 +
                forward_path * 0.4 +
                backward_path * 0.2 +
                duration_factor * 0.1
            )
            
            # Apply context priority boost
            if task.id in context.priority_boost:
                priority_score *= context.priority_boost[task.id]
            
            task_priorities[task.id] = priority_score
        
        # Create topological order respecting dependencies
        ordered_tasks = []
        remaining_tasks = set(task.id for task in subtasks)
        task_dict = {task.id: task for task in subtasks}
        
        while remaining_tasks:
            # Find tasks with all dependencies satisfied
            ready_tasks = []
            for task_id in remaining_tasks:
                task = task_dict[task_id]
                if all(dep_id not in remaining_tasks for dep_id in task.dependencies):
                    ready_tasks.append(task_id)
            
            if not ready_tasks:
                raise UnresolvableDependency("No ready tasks found - possible circular dependency")
            
            # Sort ready tasks by priority
            ready_tasks.sort(key=lambda tid: task_priorities.get(tid, 0), reverse=True)
            
            # Add highest priority ready task
            next_task = ready_tasks[0]
            ordered_tasks.append(next_task)
            remaining_tasks.remove(next_task)
        
        return ordered_tasks
    
    async def _create_load_balanced_order(
        self,
        subtasks: List[SubTask],
        dependency_graph: DependencyGraph,
        constraints: ResourceConstraints,
        context: SchedulingContext
    ) -> List[str]:
        """Create execution order optimized for load balancing."""
        # Group tasks by type for better load distribution
        tasks_by_type = {}
        for task in subtasks:
            if task.task_type not in tasks_by_type:
                tasks_by_type[task.task_type] = []
            tasks_by_type[task.task_type].append(task)
        
        # Create interleaved order to balance different types of work
        ordered_tasks = []
        remaining_tasks = {task.id: task for task in subtasks}
        
        while remaining_tasks:
            # Find ready tasks grouped by type
            ready_by_type = {}
            for task_id, task in remaining_tasks.items():
                if all(dep_id not in remaining_tasks for dep_id in task.dependencies):
                    if task.task_type not in ready_by_type:
                        ready_by_type[task.task_type] = []
                    ready_by_type[task.task_type].append(task_id)
            
            if not ready_by_type:
                raise UnresolvableDependency("No ready tasks found")
            
            # Select tasks to balance workload
            for task_type, ready_tasks in ready_by_type.items():
                if ready_tasks:
                    # Sort by priority within type
                    ready_tasks.sort(key=lambda tid: remaining_tasks[tid].priority, reverse=True)
                    next_task = ready_tasks[0]
                    ordered_tasks.append(next_task)
                    del remaining_tasks[next_task]
                    break
        
        return ordered_tasks
    
    async def _create_fastest_order(
        self,
        subtasks: List[SubTask],
        dependency_graph: DependencyGraph,
        constraints: ResourceConstraints,
        context: SchedulingContext
    ) -> List[str]:
        """Create execution order optimized for fastest completion."""
        # Prioritize shorter tasks that can be completed quickly
        task_scores = {}
        
        for task in subtasks:
            # Score based on inverse duration and priority
            duration_seconds = (
                task.estimated_duration.total_seconds()
                if task.estimated_duration else 3600  # Default 1 hour
            )
            
            # Shorter tasks get higher scores
            duration_score = 1.0 / (duration_seconds / 3600 + 0.1)  # Avoid division by zero
            priority_score = task.priority / 10.0  # Normalize priority
            
            task_scores[task.id] = duration_score * 0.7 + priority_score * 0.3
        
        # Create topological order with fast task preference
        ordered_tasks = []
        remaining_tasks = set(task.id for task in subtasks)
        task_dict = {task.id: task for task in subtasks}
        
        while remaining_tasks:
            ready_tasks = []
            for task_id in remaining_tasks:
                task = task_dict[task_id]
                if all(dep_id not in remaining_tasks for dep_id in task.dependencies):
                    ready_tasks.append(task_id)
            
            if not ready_tasks:
                raise UnresolvableDependency("No ready tasks found")
            
            # Select fastest ready task
            ready_tasks.sort(key=lambda tid: task_scores.get(tid, 0), reverse=True)
            next_task = ready_tasks[0]
            ordered_tasks.append(next_task)
            remaining_tasks.remove(next_task)
        
        return ordered_tasks
    
    async def _create_topological_order(
        self,
        subtasks: List[SubTask],
        dependency_graph: DependencyGraph
    ) -> List[str]:
        """Create basic topological ordering of tasks."""
        ordered_tasks = []
        in_degree = {}
        
        # Calculate in-degrees
        for task in subtasks:
            in_degree[task.id] = len(task.dependencies)
        
        # Find tasks with no dependencies
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        
        while queue:
            # Process task with no remaining dependencies
            current_task = queue.pop(0)
            ordered_tasks.append(current_task)
            
            # Update in-degrees of dependent tasks
            if current_task in dependency_graph.edges:
                for dependent_task in dependency_graph.edges[current_task]:
                    in_degree[dependent_task] -= 1
                    if in_degree[dependent_task] == 0:
                        queue.append(dependent_task)
        
        if len(ordered_tasks) != len(subtasks):
            raise UnresolvableDependency("Circular dependency detected in topological sort")
        
        return ordered_tasks
    
    async def _identify_parallel_groups(
        self,
        execution_order: List[str],
        dependency_graph: DependencyGraph,
        constraints: ResourceConstraints
    ) -> List[List[str]]:
        """Identify groups of tasks that can execute in parallel."""
        if not constraints.max_concurrent_tasks or constraints.max_concurrent_tasks <= 1:
            return [[task_id] for task_id in execution_order]
        
        parallel_groups = []
        processed = set()
        
        for task_id in execution_order:
            if task_id in processed:
                continue
            
            # Start new parallel group
            current_group = [task_id]
            processed.add(task_id)
            
            # Find other tasks that can run in parallel with this one
            for other_task_id in execution_order:
                if (other_task_id not in processed and 
                    len(current_group) < constraints.max_concurrent_tasks):
                    
                    # Check if other task can run in parallel
                    if await self._can_run_in_parallel(
                        current_group, other_task_id, dependency_graph
                    ):
                        current_group.append(other_task_id)
                        processed.add(other_task_id)
            
            parallel_groups.append(current_group)
        
        return parallel_groups
    
    async def _can_run_in_parallel(
        self,
        current_group: List[str],
        candidate_task: str,
        dependency_graph: DependencyGraph
    ) -> bool:
        """Check if a task can run in parallel with current group."""
        candidate_deps = set(dependency_graph.tasks[candidate_task].dependencies)
        
        for group_task in current_group:
            group_task_deps = set(dependency_graph.tasks[group_task].dependencies)
            
            # Can't run in parallel if either depends on the other
            if (candidate_task in group_task_deps or 
                group_task in candidate_deps):
                return False
            
            # Can't run in parallel if they have overlapping dependencies
            # that haven't been satisfied
            if candidate_deps & group_task_deps:
                return False
        
        return True
    
    async def _allocate_resources(
        self,
        subtasks: List[SubTask],
        execution_order: List[str],
        constraints: ResourceConstraints,
        context: SchedulingContext
    ) -> Dict[str, ResourceAllocation]:
        """Allocate resources to tasks."""
        allocations = {}
        task_dict = {task.id: task for task in subtasks}
        
        # Track resource usage over time
        resource_timeline = {}
        current_time = context.start_time
        
        for task_id in execution_order:
            task = task_dict[task_id]
            
            # Allocate agents
            allocated_agents = await self._allocate_agents_to_task(
                task, constraints, resource_timeline, current_time
            )
            
            # Allocate memory
            allocated_memory = await self._allocate_memory_to_task(
                task, constraints, resource_timeline, current_time
            )
            
            # Calculate timing
            start_time = current_time
            duration = task.estimated_duration or timedelta(hours=1)
            end_time = start_time + duration
            
            allocation = ResourceAllocation(
                task_id=task_id,
                assigned_agents=allocated_agents,
                allocated_memory_mb=allocated_memory,
                estimated_start_time=start_time,
                estimated_end_time=end_time
            )
            
            allocations[task_id] = allocation
            
            # Update timeline for next task
            current_time = end_time
        
        return allocations
    
    async def _allocate_agents_to_task(
        self,
        task: SubTask,
        constraints: ResourceConstraints,
        timeline: Dict[str, Any],
        start_time: datetime
    ) -> List[str]:
        """Allocate agents to a specific task."""
        if not constraints.available_agents:
            return ["default_agent"]
        
        # Determine how many agents this task needs
        required_agents = min(
            len(task.required_resources) or 1,
            self.config.default_agent_concurrency,
            len(constraints.available_agents)
        )
        
        # Simple allocation - assign first available agents
        # In a real implementation, this would consider agent capabilities,
        # current assignments, etc.
        allocated = constraints.available_agents[:required_agents]
        
        return allocated
    
    async def _allocate_memory_to_task(
        self,
        task: SubTask,
        constraints: ResourceConstraints,
        timeline: Dict[str, Any],
        start_time: datetime
    ) -> int:
        """Allocate memory to a specific task."""
        # Estimate memory needs based on task type and complexity
        base_memory = {
            TaskType.ANALYSIS: 512,
            TaskType.IMPLEMENTATION: 1024,
            TaskType.TESTING: 256,
            TaskType.DOCUMENTATION: 128,
            TaskType.COORDINATION: 256,
            TaskType.VALIDATION: 256
        }
        
        estimated_memory = base_memory.get(task.task_type, 512)
        
        # Apply safety margin
        safety_margin = 1.0 + self.config.memory_safety_margin
        allocated_memory = int(estimated_memory * safety_margin)
        
        return allocated_memory
    
    async def _create_checkpoints(
        self,
        execution_order: List[str],
        subtasks: List[SubTask],
        constraints: ResourceConstraints
    ) -> List[Checkpoint]:
        """Create validation checkpoints in the execution plan."""
        if not self.config.insert_validation_checkpoints:
            return []
        
        checkpoints = []
        task_dict = {task.id: task for task in subtasks}
        
        # Insert checkpoints at regular intervals
        checkpoint_interval = self.config.checkpoint_frequency
        
        for i in range(0, len(execution_order), checkpoint_interval):
            if i == 0:
                continue  # Don't create checkpoint at the very beginning
            
            completed_tasks = execution_order[:i]
            checkpoint_name = f"Checkpoint {len(checkpoints) + 1}"
            
            # Determine validation criteria based on completed tasks
            validation_criteria = []
            for task_id in completed_tasks[-checkpoint_interval:]:
                task = task_dict[task_id]
                if task.task_type == TaskType.IMPLEMENTATION:
                    validation_criteria.append(f"Verify {task.name} implementation")
                elif task.task_type == TaskType.TESTING:
                    validation_criteria.append(f"Confirm {task.name} test results")
                else:
                    validation_criteria.append(f"Validate {task.name} completion")
            
            checkpoint = Checkpoint(
                name=checkpoint_name,
                description=f"Validate completion of tasks {i-checkpoint_interval+1} through {i}",
                completed_tasks=completed_tasks.copy(),
                validation_criteria=validation_criteria,
                rollback_instructions=await self._create_rollback_instructions(
                    completed_tasks, task_dict
                )
            )
            
            checkpoints.append(checkpoint)
        
        # Add final checkpoint if needed
        if len(execution_order) % checkpoint_interval != 0:
            final_checkpoint = Checkpoint(
                name="Final Validation",
                description="Validate all tasks completed successfully",
                completed_tasks=execution_order.copy(),
                validation_criteria=["All tasks completed", "System functioning correctly"],
                rollback_instructions="Review execution log and address any issues"
            )
            checkpoints.append(final_checkpoint)
        
        return checkpoints
    
    async def _create_rollback_instructions(
        self,
        completed_tasks: List[str],
        task_dict: Dict[str, SubTask]
    ) -> str:
        """Create rollback instructions for a checkpoint."""
        if not self.config.enable_rollback_planning:
            return "Manual rollback required - review completed tasks"
        
        rollback_steps = []
        
        # Reverse order for rollback
        for task_id in reversed(completed_tasks[-3:]):  # Last 3 tasks
            task = task_dict[task_id]
            if task.task_type == TaskType.IMPLEMENTATION:
                rollback_steps.append(f"Revert changes from {task.name}")
            elif task.task_type == TaskType.TESTING:
                rollback_steps.append(f"Remove test artifacts from {task.name}")
            else:
                rollback_steps.append(f"Undo {task.name}")
        
        return "; ".join(rollback_steps) if rollback_steps else "No specific rollback needed"
    
    async def _calculate_critical_path(
        self,
        subtasks: List[SubTask],
        dependency_graph: DependencyGraph,
        resource_allocations: Dict[str, ResourceAllocation]
    ) -> List[str]:
        """Calculate the critical path through the task network."""
        # Find the longest path through the dependency graph
        task_durations = {}
        for task in subtasks:
            allocation = resource_allocations.get(task.id)
            if allocation and allocation.estimated_start_time and allocation.estimated_end_time:
                duration = allocation.estimated_end_time - allocation.estimated_start_time
            else:
                duration = task.estimated_duration or timedelta(hours=1)
            
            task_durations[task.id] = duration.total_seconds()
        
        # Calculate longest path using topological ordering
        longest_paths = {}
        critical_predecessors = {}
        
        # Initialize
        for task in subtasks:
            longest_paths[task.id] = task_durations[task.id]
            critical_predecessors[task.id] = None
        
        # Process in topological order
        for task in subtasks:
            for dep_id in task.dependencies:
                candidate_path = longest_paths[dep_id] + task_durations[task.id]
                if candidate_path > longest_paths[task.id]:
                    longest_paths[task.id] = candidate_path
                    critical_predecessors[task.id] = dep_id
        
        # Find task with longest total path (critical path end)
        critical_end = max(longest_paths.keys(), key=lambda tid: longest_paths[tid])
        
        # Reconstruct critical path
        critical_path = []
        current = critical_end
        
        while current is not None:
            critical_path.append(current)
            current = critical_predecessors[current]
        
        critical_path.reverse()
        return critical_path
    
    async def _estimate_total_duration(
        self,
        execution_order: List[str],
        subtasks: List[SubTask],
        parallel_groups: List[List[str]],
        resource_allocations: Dict[str, ResourceAllocation]
    ) -> timedelta:
        """Estimate total execution duration considering parallelization."""
        total_seconds = 0
        task_dict = {task.id: task for task in subtasks}
        
        for group in parallel_groups:
            # For parallel groups, duration is the maximum of all tasks in the group
            group_durations = []
            
            for task_id in group:
                task = task_dict[task_id]
                
                # Use allocation timing if available
                allocation = resource_allocations.get(task_id)
                if allocation and allocation.estimated_start_time and allocation.estimated_end_time:
                    duration = allocation.estimated_end_time - allocation.estimated_start_time
                else:
                    duration = task.estimated_duration or timedelta(hours=1)
                
                group_durations.append(duration.total_seconds())
            
            # Add the maximum duration for this parallel group
            total_seconds += max(group_durations) if group_durations else 0
        
        return timedelta(seconds=total_seconds)
    
    async def _calculate_forward_path_length(
        self,
        task_id: str,
        dependency_graph: DependencyGraph
    ) -> int:
        """Calculate the length of the forward dependency path."""
        visited = set()
        
        def dfs(current_id: str) -> int:
            if current_id in visited:
                return 0
            
            visited.add(current_id)
            max_path = 0
            
            for dependent_id in dependency_graph.edges.get(current_id, []):
                path_length = 1 + dfs(dependent_id)
                max_path = max(max_path, path_length)
            
            return max_path
        
        return dfs(task_id)
    
    def get_scheduling_stats(self) -> Dict[str, Any]:
        """Get execution planning performance statistics."""
        avg_tasks = (
            self._total_tasks_scheduled / self._total_schedules_created
            if self._total_schedules_created > 0 else 0
        )
        
        return {
            "total_schedules_created": self._total_schedules_created,
            "total_tasks_scheduled": self._total_tasks_scheduled,
            "average_tasks_per_schedule": round(avg_tasks, 2),
            "resource_conflicts_resolved": self._resource_conflicts_resolved,
            "config": {
                "optimization_strategy": self.config.optimization_strategy,
                "enable_resource_optimization": self.config.enable_resource_optimization,
                "enable_checkpoint_insertion": self.config.enable_checkpoint_insertion,
                "default_agent_concurrency": self.config.default_agent_concurrency
            }
        }