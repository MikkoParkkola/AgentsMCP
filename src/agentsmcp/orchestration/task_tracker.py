"""Task tracking and coordination system for AgentsMCP.

This module provides comprehensive task tracking that integrates sequential thinking,
progress display, and agent coordination for transparent AI operations.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

from .sequential_planner import SequentialPlanner, SequentialPlan, PlanningStep
from ..ui.v3.progress_display import ProgressDisplay, AgentStatus


logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Overall task status."""
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskContext:
    """Context information for a task."""
    task_id: str
    user_input: str
    created_at: float = field(default_factory=time.time)
    complexity: str = "medium"  # low, medium, high
    priority: int = 5  # 1-10 scale
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentAssignment:
    """Agent assignment within a task."""
    agent_id: str
    agent_type: str
    assigned_steps: List[str]
    estimated_duration_ms: int
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: AgentStatus = AgentStatus.IDLE
    results: Dict[str, Any] = field(default_factory=dict)


class TaskTracker:
    """
    Comprehensive task tracking system that coordinates sequential thinking,
    progress display, and agent management for transparent AI operations.
    """
    
    def __init__(self, 
                 progress_update_callback: Optional[Callable[[str], None]] = None,
                 status_update_callback: Optional[Callable[[str], None]] = None):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Core components
        self.sequential_planner = SequentialPlanner()
        self.progress_display = ProgressDisplay(update_callback=progress_update_callback)
        
        # Callbacks
        self.progress_update_callback = progress_update_callback
        self.status_update_callback = status_update_callback
        
        # Task tracking
        self.active_tasks: Dict[str, TaskContext] = {}
        self.task_plans: Dict[str, SequentialPlan] = {}
        self.task_assignments: Dict[str, List[AgentAssignment]] = {}
        self.task_status: Dict[str, TaskStatus] = {}
        
        # Performance tracking
        self.total_tasks_processed = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.average_task_duration_ms = 0
    
    async def start_task(self, 
                        user_input: str,
                        context: Dict[str, Any] = None,
                        estimated_duration_ms: int = 60000) -> str:
        """
        Start a new task with sequential thinking and progress tracking.
        
        Args:
            user_input: The user's request
            context: Additional context information
            estimated_duration_ms: Estimated total duration
            
        Returns:
            Task ID for tracking
        """
        task_id = f"task_{int(time.time())}_{hash(user_input) % 10000}"
        context = context or {}
        
        self.logger.info(f"Starting task {task_id}: {user_input[:100]}...")
        
        # Create task context
        task_context = TaskContext(
            task_id=task_id,
            user_input=user_input,
            complexity=context.get("complexity", "medium"),
            priority=context.get("priority", 5),
            tags=context.get("tags", []),
            metadata=context
        )
        
        # Initialize tracking
        self.active_tasks[task_id] = task_context
        self.task_status[task_id] = TaskStatus.PLANNING
        self.total_tasks_processed += 1
        
        # Start progress display
        self.progress_display.start_task(task_id, user_input, estimated_duration_ms)
        self.progress_display.update_orchestrator_status("Starting sequential planning...")
        
        try:
            # Phase 1: Sequential Planning
            self._notify_status("🎯 Phase 1: Sequential thinking and planning...")
            
            planning_start = time.time()
            plan = await self.sequential_planner.create_plan(
                user_input=user_input,
                context=context,
                progress_callback=self._on_planning_progress
            )
            planning_duration = int((time.time() - planning_start) * 1000)
            
            self.task_plans[task_id] = plan
            self.logger.info(f"Created sequential plan for {task_id} with {len(plan.steps)} steps in {planning_duration}ms")
            
            # Phase 2: Agent Assignment
            self._notify_status("🤖 Phase 2: Assigning agents and preparing execution...")
            self.progress_display.add_task_phase("agent_assignment")
            
            agent_assignments = await self._assign_agents_to_plan(plan, context)
            self.task_assignments[task_id] = agent_assignments
            
            # Add agents to progress display
            for assignment in agent_assignments:
                self.progress_display.add_agent(
                    assignment.agent_id,
                    f"{assignment.agent_type.title()} Agent",
                    assignment.estimated_duration_ms
                )
            
            # Phase 3: Ready for Execution
            self.task_status[task_id] = TaskStatus.EXECUTING
            self._notify_status("✅ Planning complete - ready for execution")
            
            return task_id
            
        except Exception as e:
            self.logger.error(f"Failed to start task {task_id}: {e}")
            self.task_status[task_id] = TaskStatus.FAILED
            self.failed_tasks += 1
            self.progress_display.update_orchestrator_status(f"Failed to start task: {str(e)}")
            raise
    
    async def execute_task(self, task_id: str) -> Dict[str, Any]:
        """
        Execute a task using its sequential plan and agent assignments.
        
        Args:
            task_id: ID of the task to execute
            
        Returns:
            Execution results
        """
        if task_id not in self.active_tasks:
            raise ValueError(f"Task {task_id} not found")
        
        plan = self.task_plans.get(task_id)
        assignments = self.task_assignments.get(task_id, [])
        
        if not plan:
            raise ValueError(f"No plan found for task {task_id}")
        
        self.logger.info(f"Executing task {task_id} with {len(plan.steps)} steps and {len(assignments)} agents")
        
        execution_start = time.time()
        results = {"task_id": task_id, "steps": [], "agents": [], "success": False}
        
        try:
            self._notify_status("🚀 Executing sequential plan...")
            
            # Execute plan step by step
            async def step_executor(step: PlanningStep, plan: SequentialPlan) -> Any:
                return await self._execute_plan_step(step, plan, task_id, assignments)
            
            success = await self.sequential_planner.execute_plan(task_id, step_executor)
            
            if success:
                self.task_status[task_id] = TaskStatus.COMPLETED
                self.successful_tasks += 1
                self.progress_display.complete_task()
                self._notify_status("✅ Task completed successfully")
                
                results["success"] = True
                results["execution_duration_ms"] = int((time.time() - execution_start) * 1000)
                
                # Update performance tracking
                total_duration = results["execution_duration_ms"]
                if self.successful_tasks == 1:
                    self.average_task_duration_ms = total_duration
                else:
                    self.average_task_duration_ms = int(
                        (self.average_task_duration_ms * (self.successful_tasks - 1) + total_duration) / self.successful_tasks
                    )
                
                self.logger.info(f"Task {task_id} completed successfully in {total_duration}ms")
            else:
                self.task_status[task_id] = TaskStatus.FAILED
                self.failed_tasks += 1
                self.progress_display.update_orchestrator_status("❌ Task execution failed")
                self.logger.error(f"Task {task_id} execution failed")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Task {task_id} execution error: {e}")
            self.task_status[task_id] = TaskStatus.FAILED
            self.failed_tasks += 1
            self.progress_display.update_orchestrator_status(f"❌ Execution error: {str(e)}")
            
            results["error"] = str(e)
            return results
    
    async def _assign_agents_to_plan(self, plan: SequentialPlan, context: Dict[str, Any]) -> List[AgentAssignment]:
        """Assign agents to plan steps based on requirements."""
        assignments = []
        agent_counter = 1
        
        # Group steps by agent type requirements
        step_groups = {}
        for step in plan.steps:
            # Determine required agent type for this step
            agent_type = self._determine_agent_type_for_step(step, context)
            
            if agent_type not in step_groups:
                step_groups[agent_type] = []
            step_groups[agent_type].append(step.step_id)
        
        # Create agent assignments
        for agent_type, step_ids in step_groups.items():
            total_duration = sum(
                step.estimated_duration_ms for step in plan.steps
                if step.step_id in step_ids
            )
            
            assignment = AgentAssignment(
                agent_id=f"{agent_type}_{agent_counter}",
                agent_type=agent_type,
                assigned_steps=step_ids,
                estimated_duration_ms=total_duration
            )
            assignments.append(assignment)
            agent_counter += 1
        
        return assignments
    
    def _determine_agent_type_for_step(self, step: PlanningStep, context: Dict[str, Any]) -> str:
        """Determine the best agent type for a planning step."""
        step_desc = step.description.lower()
        
        # Map step descriptions to agent types
        if any(keyword in step_desc for keyword in ['analyze', 'analysis', 'understand', 'examine']):
            return 'analyst'
        elif any(keyword in step_desc for keyword in ['code', 'implement', 'develop', 'program', 'build']):
            return 'developer'
        elif any(keyword in step_desc for keyword in ['test', 'verify', 'validate', 'check']):
            return 'qa_engineer'
        elif any(keyword in step_desc for keyword in ['design', 'architecture', 'structure']):
            return 'architect'
        elif any(keyword in step_desc for keyword in ['deploy', 'release', 'publish']):
            return 'devops'
        elif any(keyword in step_desc for keyword in ['document', 'write', 'explain']):
            return 'technical_writer'
        else:
            # Default based on task complexity
            complexity = context.get("complexity", "medium")
            if complexity == "high":
                return 'senior_engineer'
            elif complexity == "low":
                return 'assistant'
            else:
                return 'general_agent'
    
    async def _execute_plan_step(self, 
                                step: PlanningStep, 
                                plan: SequentialPlan, 
                                task_id: str,
                                assignments: List[AgentAssignment]) -> Any:
        """Execute a single plan step with agent coordination."""
        # Find the agent assignment for this step
        assigned_agent = None
        for assignment in assignments:
            if step.step_id in assignment.assigned_steps:
                assigned_agent = assignment
                break
        
        if not assigned_agent:
            self.logger.warning(f"No agent assigned for step {step.step_id}")
            return {"result": "No agent assigned", "success": False}
        
        # Start the agent if not already started
        if not assigned_agent.started_at:
            assigned_agent.started_at = time.time()
            assigned_agent.status = AgentStatus.IN_PROGRESS
            self.progress_display.start_agent(assigned_agent.agent_id, step.description)
        
        # Update agent progress
        step_index = next(i for i, s in enumerate(plan.steps) if s.step_id == step.step_id)
        agent_steps_total = len(assigned_agent.assigned_steps)
        agent_steps_completed = sum(1 for sid in assigned_agent.assigned_steps 
                                  if any(s.step_id == sid and s.is_completed for s in plan.steps))
        
        progress_percentage = ((agent_steps_completed + 0.5) / agent_steps_total) * 100
        self.progress_display.update_agent_progress(
            assigned_agent.agent_id, 
            progress_percentage, 
            step.description
        )
        
        try:
            # Simulate step execution (in real implementation, this would delegate to actual agents)
            self.logger.debug(f"Executing step {step.step_id}: {step.description}")
            
            # Simulate processing time with progress updates
            execution_time = step.estimated_duration_ms / 1000.0
            update_interval = min(1.0, execution_time / 5)  # Update 5 times during execution
            
            for i in range(5):
                await asyncio.sleep(update_interval)
                intermediate_progress = progress_percentage + (i + 1) * (10 / agent_steps_total)
                self.progress_display.update_agent_progress(
                    assigned_agent.agent_id,
                    min(100.0, intermediate_progress),
                    f"{step.description} - {20 * (i + 1)}%"
                )
            
            # Mark step as completed
            if step.step_id == assigned_agent.assigned_steps[-1]:  # Last step for this agent
                self.progress_display.complete_agent(assigned_agent.agent_id)
                assigned_agent.status = AgentStatus.COMPLETED
                assigned_agent.completed_at = time.time()
            
            result = {
                "step_id": step.step_id,
                "agent_id": assigned_agent.agent_id,
                "result": f"Completed: {step.description}",
                "success": True,
                "duration_ms": step.duration_ms
            }
            
            assigned_agent.results[step.step_id] = result
            return result
            
        except Exception as e:
            self.logger.error(f"Step {step.step_id} failed: {e}")
            self.progress_display.set_agent_error(assigned_agent.agent_id, f"Error: {str(e)}")
            assigned_agent.status = AgentStatus.ERROR
            
            return {
                "step_id": step.step_id,
                "agent_id": assigned_agent.agent_id,
                "error": str(e),
                "success": False
            }
    
    def _on_planning_progress(self, message: str, percentage: float, data: Dict[str, Any]) -> None:
        """Handle planning progress updates."""
        self.progress_display.update_orchestrator_status(f"Planning: {message}")
        self.logger.debug(f"Planning progress: {percentage:.1f}% - {message}")
    
    def _notify_status(self, status: str) -> None:
        """Notify status update."""
        if self.status_update_callback:
            self.status_update_callback(status)
        self.progress_display.update_orchestrator_status(status)
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive status for a task."""
        if task_id not in self.active_tasks:
            return None
        
        task = self.active_tasks[task_id]
        plan = self.task_plans.get(task_id)
        assignments = self.task_assignments.get(task_id, [])
        status = self.task_status.get(task_id, TaskStatus.PENDING)
        
        return {
            "task_id": task_id,
            "user_input": task.user_input,
            "status": status.value,
            "created_at": task.created_at,
            "complexity": task.complexity,
            "plan_steps": len(plan.steps) if plan else 0,
            "assigned_agents": len(assignments),
            "progress_percentage": plan.progress_percentage if plan else 0.0,
            "elapsed_time_ms": int((time.time() - task.created_at) * 1000),
            "agents": [
                {
                    "agent_id": a.agent_id,
                    "agent_type": a.agent_type,
                    "status": a.status.value,
                    "assigned_steps": len(a.assigned_steps),
                    "results_count": len(a.results)
                }
                for a in assignments
            ]
        }
    
    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """Get status for all active tasks."""
        return [self.get_task_status(task_id) for task_id in self.active_tasks.keys()]
    
    def get_progress_display(self) -> str:
        """Get formatted progress display."""
        return self.progress_display.format_progress_display()
    
    def get_status_line(self) -> str:
        """Get compact status line."""
        return self.progress_display.format_status_line()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        planner_stats = self.sequential_planner.get_performance_stats()
        display_stats = self.progress_display.get_performance_stats()
        
        success_rate = (self.successful_tasks / max(1, self.total_tasks_processed)) * 100
        
        return {
            "task_tracking": {
                "total_tasks_processed": self.total_tasks_processed,
                "successful_tasks": self.successful_tasks,
                "failed_tasks": self.failed_tasks,
                "success_rate_percentage": success_rate,
                "average_task_duration_ms": self.average_task_duration_ms,
                "active_tasks": len(self.active_tasks)
            },
            "sequential_planning": planner_stats,
            "progress_display": display_stats
        }
    
    def cleanup_completed_tasks(self, max_age_hours: int = 24) -> int:
        """Clean up old completed tasks."""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        tasks_to_remove = []
        for task_id, task in self.active_tasks.items():
            task_status = self.task_status.get(task_id)
            if (task_status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED] and
                (current_time - task.created_at) > max_age_seconds):
                tasks_to_remove.append(task_id)
        
        # Remove old tasks
        for task_id in tasks_to_remove:
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
            if task_id in self.task_plans:
                del self.task_plans[task_id]
            if task_id in self.task_assignments:
                del self.task_assignments[task_id]
            if task_id in self.task_status:
                del self.task_status[task_id]
        
        # Clean up progress display
        self.progress_display.cleanup_completed_agents()
        
        if tasks_to_remove:
            self.logger.info(f"Cleaned up {len(tasks_to_remove)} completed tasks")
        
        return len(tasks_to_remove)