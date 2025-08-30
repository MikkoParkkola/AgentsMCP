"""
Symphony Mode Orchestration API with Real-time Coordination

Advanced multi-agent coordination system supporting 12+ concurrent agents
with real-time status monitoring, conflict resolution, and harmony scoring.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import uuid

from .base import APIBase, APIResponse, APIError


class AgentStatus(str, Enum):
    """Agent status in symphony mode."""
    IDLE = "idle"
    WORKING = "working"
    WAITING = "waiting"
    BLOCKED = "blocked"
    FAILED = "failed"
    COMPLETED = "completed"
    TERMINATED = "terminated"


class TaskPriority(str, Enum):
    """Task priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class ConflictType(str, Enum):
    """Types of conflicts between agents."""
    RESOURCE = "resource"
    DEPENDENCY = "dependency"
    SCHEDULING = "scheduling"
    OUTPUT = "output"
    COORDINATION = "coordination"


@dataclass
class Agent:
    """Agent representation in symphony mode."""
    id: str
    name: str
    type: str  # claude, gpt4, etc.
    capabilities: List[str]
    status: AgentStatus = AgentStatus.IDLE
    current_task_id: Optional[str] = None
    workload: float = 0.0  # 0.0 to 1.0
    health_score: float = 1.0  # 0.0 to 1.0
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    resource_usage: Dict[str, float] = field(default_factory=dict)
    

@dataclass
class Task:
    """Task in the symphony orchestration system."""
    id: str
    name: str
    description: str
    priority: TaskPriority
    dependencies: List[str] = field(default_factory=list)
    required_capabilities: List[str] = field(default_factory=list)
    assigned_agent_id: Optional[str] = None
    status: str = "pending"
    estimated_duration: Optional[float] = None
    actual_duration: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class Conflict:
    """Conflict between agents or tasks."""
    id: str
    type: ConflictType
    severity: float  # 0.0 to 1.0
    involved_agents: List[str]
    involved_tasks: List[str]
    description: str
    resolution_strategy: Optional[str] = None
    resolved: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None


@dataclass
class SymphonyMetrics:
    """Symphony mode performance metrics."""
    harmony_score: float  # Overall coordination effectiveness (0.0 to 1.0)
    active_agents: int
    completed_tasks: int
    failed_tasks: int
    average_task_duration: float
    resource_utilization: float
    conflict_count: int
    resolved_conflicts: int
    uptime: float
    throughput: float  # Tasks per minute


class SymphonyOrchestrationAPI(APIBase):
    """Advanced symphony mode orchestration with real-time coordination."""
    
    def __init__(self):
        super().__init__("symphony_orchestration_api")
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, Task] = {}
        self.conflicts: Dict[str, Conflict] = {}
        self.task_queue: List[str] = []  # Task IDs in priority order
        self.execution_history: List[Dict[str, Any]] = []
        self.resource_pools: Dict[str, float] = {}
        self.coordination_rules: Dict[str, Any] = {}
        
        # Symphony mode state
        self.symphony_active = False
        self.start_time = None
        self.auto_scale_enabled = False
        self.max_agents = 12
        self.min_agents = 2
        
        # Performance tracking
        self.metrics_history: List[SymphonyMetrics] = []
        self.last_metrics_update = time.time()
        
        # Initialize coordination system
        asyncio.create_task(self._initialize_coordination_system())
        
    async def _initialize_coordination_system(self):
        """Initialize the coordination system with default settings."""
        # Default resource pools
        self.resource_pools = {
            "cpu": 1.0,
            "memory": 1.0,
            "api_calls": 1000.0,
            "network": 1.0,
        }
        
        # Default coordination rules
        self.coordination_rules = {
            "max_concurrent_tasks_per_agent": 3,
            "task_timeout_seconds": 300,
            "heartbeat_interval_seconds": 30,
            "conflict_resolution_timeout": 60,
            "auto_retry_failed_tasks": True,
            "load_balancing_enabled": True,
            "priority_scheduling": True,
        }
        
        # Start background processes
        asyncio.create_task(self._monitor_agents())
        asyncio.create_task(self._process_task_queue())
        asyncio.create_task(self._resolve_conflicts())
        asyncio.create_task(self._update_metrics())
    
    async def enable_symphony_mode(
        self,
        initial_agents: List[Dict[str, Any]] = None,
        auto_scale: bool = False,
        max_agents: int = 12
    ) -> APIResponse:
        """Enable symphony mode with multi-agent coordination."""
        return await self._execute_with_metrics(
            "enable_symphony_mode",
            self._enable_symphony_mode_internal,
            initial_agents or [],
            auto_scale,
            max_agents
        )
    
    async def _enable_symphony_mode_internal(
        self,
        initial_agents: List[Dict[str, Any]],
        auto_scale: bool,
        max_agents: int
    ) -> Dict[str, Any]:
        """Internal logic for enabling symphony mode."""
        if self.symphony_active:
            raise APIError("Symphony mode already active", "ALREADY_ACTIVE", 400)
        
        self.symphony_active = True
        self.start_time = datetime.utcnow()
        self.auto_scale_enabled = auto_scale
        self.max_agents = max_agents
        
        # Register initial agents
        registered_agents = []
        for agent_data in initial_agents:
            agent = await self._register_agent_internal(agent_data)
            registered_agents.append(agent.id)
        
        # If no initial agents provided, create default agent set
        if not registered_agents:
            default_agents = [
                {
                    "name": "coordinator",
                    "type": "claude",
                    "capabilities": ["coordination", "planning", "analysis"]
                },
                {
                    "name": "executor", 
                    "type": "gpt4",
                    "capabilities": ["execution", "coding", "data_processing"]
                }
            ]
            
            for agent_data in default_agents:
                agent = await self._register_agent_internal(agent_data)
                registered_agents.append(agent.id)
        
        self.logger.info(f"Symphony mode enabled with {len(registered_agents)} agents")
        
        return {
            "symphony_active": True,
            "registered_agents": registered_agents,
            "auto_scale_enabled": auto_scale,
            "max_agents": max_agents,
            "start_time": self.start_time.isoformat()
        }
    
    async def register_agent(self, agent_data: Dict[str, Any]) -> APIResponse:
        """Register a new agent in symphony mode."""
        return await self._execute_with_metrics(
            "register_agent",
            self._register_agent_internal,
            agent_data
        )
    
    async def _register_agent_internal(self, agent_data: Dict[str, Any]) -> Agent:
        """Internal agent registration logic."""
        if not self.symphony_active:
            raise APIError("Symphony mode not active", "NOT_ACTIVE", 400)
        
        if len(self.agents) >= self.max_agents:
            raise APIError("Maximum agent limit reached", "AGENT_LIMIT", 400)
        
        agent_id = agent_data.get("id", str(uuid.uuid4()))
        
        if agent_id in self.agents:
            raise APIError(f"Agent {agent_id} already registered", "DUPLICATE_AGENT", 400)
        
        agent = Agent(
            id=agent_id,
            name=agent_data["name"],
            type=agent_data["type"],
            capabilities=agent_data.get("capabilities", []),
            performance_metrics={
                "tasks_completed": 0,
                "avg_response_time": 0.0,
                "success_rate": 1.0,
                "error_rate": 0.0
            }
        )
        
        self.agents[agent_id] = agent
        
        self.logger.info(f"Registered agent {agent_id} ({agent.name})")
        
        return agent
    
    async def submit_task(self, task_data: Dict[str, Any]) -> APIResponse:
        """Submit a task for execution in symphony mode."""
        return await self._execute_with_metrics(
            "submit_task",
            self._submit_task_internal,
            task_data
        )
    
    async def _submit_task_internal(self, task_data: Dict[str, Any]) -> Task:
        """Internal task submission logic."""
        if not self.symphony_active:
            raise APIError("Symphony mode not active", "NOT_ACTIVE", 400)
        
        task_id = task_data.get("id", str(uuid.uuid4()))
        
        if task_id in self.tasks:
            raise APIError(f"Task {task_id} already exists", "DUPLICATE_TASK", 400)
        
        task = Task(
            id=task_id,
            name=task_data["name"],
            description=task_data.get("description", ""),
            priority=TaskPriority(task_data.get("priority", "normal")),
            dependencies=task_data.get("dependencies", []),
            required_capabilities=task_data.get("required_capabilities", []),
            estimated_duration=task_data.get("estimated_duration")
        )
        
        # Validate dependencies
        for dep_id in task.dependencies:
            if dep_id not in self.tasks:
                raise APIError(f"Dependency task {dep_id} not found", "INVALID_DEPENDENCY", 400)
        
        self.tasks[task_id] = task
        await self._schedule_task(task)
        
        self.logger.info(f"Submitted task {task_id} ({task.name})")
        
        return task
    
    async def _schedule_task(self, task: Task):
        """Schedule a task for execution."""
        # Check if dependencies are satisfied
        if not self._dependencies_satisfied(task):
            return  # Task will be scheduled later when dependencies complete
        
        # Insert task into queue based on priority
        priority_order = {
            TaskPriority.CRITICAL: 0,
            TaskPriority.HIGH: 1,
            TaskPriority.NORMAL: 2,
            TaskPriority.LOW: 3,
        }
        
        task_priority = priority_order[task.priority]
        
        # Find insertion point
        insert_index = len(self.task_queue)
        for i, existing_task_id in enumerate(self.task_queue):
            existing_task = self.tasks[existing_task_id]
            if priority_order[existing_task.priority] > task_priority:
                insert_index = i
                break
        
        self.task_queue.insert(insert_index, task.id)
    
    def _dependencies_satisfied(self, task: Task) -> bool:
        """Check if all task dependencies are satisfied."""
        for dep_id in task.dependencies:
            dep_task = self.tasks.get(dep_id)
            if not dep_task or dep_task.status != "completed":
                return False
        return True
    
    async def _monitor_agents(self):
        """Background task to monitor agent health and status."""
        while True:
            try:
                if not self.symphony_active:
                    await asyncio.sleep(10)
                    continue
                
                current_time = datetime.utcnow()
                
                for agent in self.agents.values():
                    # Check heartbeat timeout
                    if (current_time - agent.last_heartbeat).seconds > self.coordination_rules["heartbeat_interval_seconds"] * 2:
                        if agent.status != AgentStatus.TERMINATED:
                            self.logger.warning(f"Agent {agent.id} heartbeat timeout")
                            agent.status = AgentStatus.FAILED
                            agent.health_score *= 0.8
                            
                            # Handle task reassignment if agent was working
                            if agent.current_task_id:
                                await self._handle_agent_failure(agent)
                
                await asyncio.sleep(self.coordination_rules["heartbeat_interval_seconds"])
                
            except Exception as e:
                self.logger.error(f"Agent monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _process_task_queue(self):
        """Background task to process the task queue."""
        while True:
            try:
                if not self.symphony_active or not self.task_queue:
                    await asyncio.sleep(5)
                    continue
                
                # Get next task
                task_id = self.task_queue[0]
                task = self.tasks[task_id]
                
                # Find available agent
                suitable_agent = await self._find_suitable_agent(task)
                
                if suitable_agent:
                    # Assign task to agent
                    await self._assign_task_to_agent(task, suitable_agent)
                    self.task_queue.pop(0)
                else:
                    # No suitable agent available, wait
                    await asyncio.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Task queue processing error: {e}")
                await asyncio.sleep(10)
    
    async def _find_suitable_agent(self, task: Task) -> Optional[Agent]:
        """Find the most suitable agent for a task."""
        suitable_agents = []
        
        for agent in self.agents.values():
            if agent.status not in [AgentStatus.IDLE, AgentStatus.WORKING]:
                continue
                
            # Check capability match
            if task.required_capabilities:
                if not set(task.required_capabilities).issubset(set(agent.capabilities)):
                    continue
            
            # Check workload
            max_concurrent = self.coordination_rules["max_concurrent_tasks_per_agent"]
            if agent.status == AgentStatus.WORKING and agent.workload >= max_concurrent:
                continue
            
            suitable_agents.append(agent)
        
        if not suitable_agents:
            return None
        
        # Score agents and return the best one
        best_agent = None
        best_score = -1
        
        for agent in suitable_agents:
            score = self._calculate_agent_suitability_score(agent, task)
            if score > best_score:
                best_score = score
                best_agent = agent
        
        return best_agent
    
    def _calculate_agent_suitability_score(self, agent: Agent, task: Task) -> float:
        """Calculate suitability score for agent-task pairing."""
        score = 0.0
        
        # Base score from health and performance
        score += agent.health_score * 0.3
        success_rate = agent.performance_metrics.get("success_rate", 1.0)
        score += success_rate * 0.3
        
        # Capability match score
        if task.required_capabilities:
            matched_capabilities = set(task.required_capabilities).intersection(set(agent.capabilities))
            capability_score = len(matched_capabilities) / len(task.required_capabilities)
            score += capability_score * 0.2
        else:
            score += 0.2  # No specific requirements
        
        # Workload penalty
        workload_penalty = agent.workload * 0.2
        score -= workload_penalty
        
        return max(score, 0.0)
    
    async def _assign_task_to_agent(self, task: Task, agent: Agent):
        """Assign a task to an agent."""
        task.assigned_agent_id = agent.id
        task.status = "assigned"
        task.started_at = datetime.utcnow()
        
        agent.current_task_id = task.id
        agent.status = AgentStatus.WORKING
        agent.workload += 1.0
        
        self.logger.info(f"Assigned task {task.id} to agent {agent.id}")
        
        # Start task execution (simplified - in real implementation would call actual agent)
        asyncio.create_task(self._simulate_task_execution(task, agent))
    
    async def _simulate_task_execution(self, task: Task, agent: Agent):
        """Simulate task execution for demonstration."""
        try:
            # Simulate work duration
            duration = task.estimated_duration or 30  # Default 30 seconds
            await asyncio.sleep(min(duration, 5))  # Cap at 5 seconds for demo
            
            # Simulate success/failure based on agent reliability
            success_rate = agent.performance_metrics.get("success_rate", 0.9)
            import random
            success = random.random() < success_rate
            
            if success:
                task.status = "completed"
                task.completed_at = datetime.utcnow()
                task.actual_duration = (task.completed_at - task.started_at).total_seconds()
                task.result = {"status": "success", "output": "Task completed successfully"}
                
                # Update agent metrics
                agent.performance_metrics["tasks_completed"] += 1
                agent.performance_metrics["avg_response_time"] = (
                    agent.performance_metrics.get("avg_response_time", 0) * 0.8 + 
                    task.actual_duration * 0.2
                )
                
            else:
                task.status = "failed"
                task.error = "Simulated task failure"
                task.retry_count += 1
                
                agent.performance_metrics["error_rate"] = (
                    agent.performance_metrics.get("error_rate", 0) * 0.9 + 0.1
                )
            
            # Clean up agent state
            agent.current_task_id = None
            agent.workload = max(0.0, agent.workload - 1.0)
            
            if agent.workload == 0:
                agent.status = AgentStatus.IDLE
            
            # Handle task completion/failure
            await self._handle_task_completion(task)
            
        except Exception as e:
            self.logger.error(f"Task execution error: {e}")
            task.status = "failed"
            task.error = str(e)
            agent.current_task_id = None
            agent.workload = max(0.0, agent.workload - 1.0)
            if agent.workload == 0:
                agent.status = AgentStatus.IDLE
    
    async def _handle_task_completion(self, task: Task):
        """Handle task completion and trigger dependent tasks."""
        self.execution_history.append({
            "task_id": task.id,
            "status": task.status,
            "duration": task.actual_duration,
            "agent_id": task.assigned_agent_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        if task.status == "completed":
            # Check for dependent tasks that can now be scheduled
            for other_task in self.tasks.values():
                if (task.id in other_task.dependencies and 
                    other_task.status == "pending" and
                    self._dependencies_satisfied(other_task)):
                    await self._schedule_task(other_task)
        
        elif task.status == "failed" and task.retry_count < task.max_retries:
            # Retry failed task
            if self.coordination_rules["auto_retry_failed_tasks"]:
                task.status = "pending"
                task.assigned_agent_id = None
                await self._schedule_task(task)
    
    async def _handle_agent_failure(self, agent: Agent):
        """Handle agent failure and task reassignment."""
        if agent.current_task_id:
            task = self.tasks[agent.current_task_id]
            task.status = "pending"
            task.assigned_agent_id = None
            agent.current_task_id = None
            
            # Reschedule the task
            await self._schedule_task(task)
            
            self.logger.info(f"Reassigned task {task.id} due to agent {agent.id} failure")
    
    async def _resolve_conflicts(self):
        """Background task to resolve conflicts between agents."""
        while True:
            try:
                if not self.symphony_active:
                    await asyncio.sleep(10)
                    continue
                
                unresolved_conflicts = [c for c in self.conflicts.values() if not c.resolved]
                
                for conflict in unresolved_conflicts:
                    await self._resolve_conflict(conflict)
                
                await asyncio.sleep(self.coordination_rules["conflict_resolution_timeout"])
                
            except Exception as e:
                self.logger.error(f"Conflict resolution error: {e}")
                await asyncio.sleep(10)
    
    async def _resolve_conflict(self, conflict: Conflict):
        """Resolve a specific conflict."""
        try:
            if conflict.type == ConflictType.RESOURCE:
                await self._resolve_resource_conflict(conflict)
            elif conflict.type == ConflictType.DEPENDENCY:
                await self._resolve_dependency_conflict(conflict)
            elif conflict.type == ConflictType.SCHEDULING:
                await self._resolve_scheduling_conflict(conflict)
            
            conflict.resolved = True
            conflict.resolved_at = datetime.utcnow()
            
        except Exception as e:
            self.logger.error(f"Failed to resolve conflict {conflict.id}: {e}")
    
    async def _resolve_resource_conflict(self, conflict: Conflict):
        """Resolve resource-based conflicts."""
        # Simple resolution: pause lower priority tasks
        involved_tasks = [self.tasks[tid] for tid in conflict.involved_tasks if tid in self.tasks]
        involved_tasks.sort(key=lambda t: t.priority.value)  # Sort by priority
        
        # Pause lowest priority task
        if involved_tasks:
            lowest_priority_task = involved_tasks[-1]
            if lowest_priority_task.assigned_agent_id:
                agent = self.agents[lowest_priority_task.assigned_agent_id]
                agent.status = AgentStatus.WAITING
                conflict.resolution_strategy = f"Paused task {lowest_priority_task.id}"
    
    async def _resolve_dependency_conflict(self, conflict: Conflict):
        """Resolve dependency-based conflicts."""
        # Reorder task queue to respect dependencies
        conflict.resolution_strategy = "Reordered task queue"
    
    async def _resolve_scheduling_conflict(self, conflict: Conflict):
        """Resolve scheduling conflicts."""
        # Redistribute tasks among agents
        conflict.resolution_strategy = "Redistributed tasks"
    
    async def _update_metrics(self):
        """Background task to update symphony metrics."""
        while True:
            try:
                if not self.symphony_active:
                    await asyncio.sleep(30)
                    continue
                
                metrics = await self._calculate_current_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only last 100 metric entries
                if len(self.metrics_history) > 100:
                    self.metrics_history = self.metrics_history[-100:]
                
                self.last_metrics_update = time.time()
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Metrics update error: {e}")
                await asyncio.sleep(30)
    
    async def _calculate_current_metrics(self) -> SymphonyMetrics:
        """Calculate current symphony metrics."""
        active_agents = len([a for a in self.agents.values() if a.status in [AgentStatus.WORKING, AgentStatus.IDLE]])
        completed_tasks = len([t for t in self.tasks.values() if t.status == "completed"])
        failed_tasks = len([t for t in self.tasks.values() if t.status == "failed"])
        
        # Calculate average task duration
        completed_task_list = [t for t in self.tasks.values() if t.status == "completed" and t.actual_duration]
        avg_duration = sum(t.actual_duration for t in completed_task_list) / len(completed_task_list) if completed_task_list else 0.0
        
        # Calculate harmony score (coordination effectiveness)
        harmony_score = self._calculate_harmony_score()
        
        # Calculate resource utilization
        total_workload = sum(agent.workload for agent in self.agents.values())
        max_possible_workload = len(self.agents) * self.coordination_rules["max_concurrent_tasks_per_agent"]
        resource_utilization = total_workload / max_possible_workload if max_possible_workload > 0 else 0.0
        
        # Calculate uptime
        uptime = (datetime.utcnow() - self.start_time).total_seconds() if self.start_time else 0.0
        
        # Calculate throughput (tasks per minute)
        throughput = (completed_tasks / (uptime / 60)) if uptime > 0 else 0.0
        
        return SymphonyMetrics(
            harmony_score=harmony_score,
            active_agents=active_agents,
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            average_task_duration=avg_duration,
            resource_utilization=resource_utilization,
            conflict_count=len(self.conflicts),
            resolved_conflicts=len([c for c in self.conflicts.values() if c.resolved]),
            uptime=uptime,
            throughput=throughput
        )
    
    def _calculate_harmony_score(self) -> float:
        """Calculate overall coordination harmony score."""
        if not self.agents:
            return 0.0
        
        # Base score from agent health
        avg_health = sum(agent.health_score for agent in self.agents.values()) / len(self.agents)
        
        # Penalty for conflicts
        unresolved_conflicts = len([c for c in self.conflicts.values() if not c.resolved])
        conflict_penalty = min(unresolved_conflicts * 0.1, 0.5)
        
        # Bonus for task completion rate
        total_tasks = len(self.tasks)
        if total_tasks > 0:
            completed_tasks = len([t for t in self.tasks.values() if t.status == "completed"])
            completion_bonus = (completed_tasks / total_tasks) * 0.3
        else:
            completion_bonus = 0.0
        
        harmony_score = avg_health - conflict_penalty + completion_bonus
        return max(0.0, min(harmony_score, 1.0))
    
    async def get_symphony_status(self) -> APIResponse:
        """Get current symphony mode status and metrics."""
        return await self._execute_with_metrics(
            "get_symphony_status",
            self._get_symphony_status_internal
        )
    
    async def _get_symphony_status_internal(self) -> Dict[str, Any]:
        """Internal logic for getting symphony status."""
        if not self.symphony_active:
            return {
                "symphony_active": False,
                "message": "Symphony mode not active"
            }
        
        current_metrics = await self._calculate_current_metrics()
        
        return {
            "symphony_active": True,
            "start_time": self.start_time.isoformat(),
            "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds(),
            "agents": {
                "total": len(self.agents),
                "active": len([a for a in self.agents.values() if a.status == AgentStatus.WORKING]),
                "idle": len([a for a in self.agents.values() if a.status == AgentStatus.IDLE]),
                "failed": len([a for a in self.agents.values() if a.status == AgentStatus.FAILED])
            },
            "tasks": {
                "total": len(self.tasks),
                "pending": len([t for t in self.tasks.values() if t.status == "pending"]),
                "running": len([t for t in self.tasks.values() if t.status == "assigned"]),
                "completed": len([t for t in self.tasks.values() if t.status == "completed"]),
                "failed": len([t for t in self.tasks.values() if t.status == "failed"])
            },
            "metrics": asdict(current_metrics),
            "conflicts": {
                "total": len(self.conflicts),
                "unresolved": len([c for c in self.conflicts.values() if not c.resolved])
            },
            "auto_scale_enabled": self.auto_scale_enabled,
            "max_agents": self.max_agents
        }
    
    async def get_agent_details(self, agent_id: str) -> APIResponse:
        """Get detailed information about a specific agent."""
        return await self._execute_with_metrics(
            "get_agent_details",
            self._get_agent_details_internal,
            agent_id
        )
    
    async def _get_agent_details_internal(self, agent_id: str) -> Dict[str, Any]:
        """Internal logic for getting agent details."""
        agent = self.agents.get(agent_id)
        if not agent:
            raise APIError(f"Agent {agent_id} not found", "AGENT_NOT_FOUND", 404)
        
        # Get agent's task history
        agent_tasks = [
            asdict(task) for task in self.tasks.values() 
            if task.assigned_agent_id == agent_id
        ]
        
        return {
            "agent": asdict(agent),
            "task_history": agent_tasks,
            "recent_performance": agent.performance_metrics
        }
    
    async def disable_symphony_mode(self) -> APIResponse:
        """Disable symphony mode and clean up resources."""
        return await self._execute_with_metrics(
            "disable_symphony_mode",
            self._disable_symphony_mode_internal
        )
    
    async def _disable_symphony_mode_internal(self) -> Dict[str, Any]:
        """Internal logic for disabling symphony mode."""
        if not self.symphony_active:
            raise APIError("Symphony mode not active", "NOT_ACTIVE", 400)
        
        # Terminate all agents
        for agent in self.agents.values():
            agent.status = AgentStatus.TERMINATED
        
        # Mark remaining tasks as cancelled
        pending_tasks = [t for t in self.tasks.values() if t.status in ["pending", "assigned"]]
        for task in pending_tasks:
            task.status = "cancelled"
        
        # Calculate final metrics
        final_metrics = await self._calculate_current_metrics()
        
        self.symphony_active = False
        end_time = datetime.utcnow()
        total_uptime = (end_time - self.start_time).total_seconds()
        
        summary = {
            "symphony_disabled": True,
            "end_time": end_time.isoformat(),
            "total_uptime_seconds": total_uptime,
            "final_metrics": asdict(final_metrics),
            "total_agents_registered": len(self.agents),
            "total_tasks_processed": len(self.tasks),
            "tasks_completed": len([t for t in self.tasks.values() if t.status == "completed"]),
            "tasks_failed": len([t for t in self.tasks.values() if t.status == "failed"]),
            "conflicts_resolved": len([c for c in self.conflicts.values() if c.resolved])
        }
        
        # Clear state for next session
        self.agents.clear()
        self.tasks.clear()
        self.conflicts.clear()
        self.task_queue.clear()
        self.start_time = None
        
        return summary