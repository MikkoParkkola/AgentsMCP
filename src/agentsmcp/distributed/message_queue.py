"""
Message Queue System for Distributed AgentsMCP

Handles task distribution, result collection, and worker coordination
with reliable delivery and cost tracking integration.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class TaskStatus(Enum):
    QUEUED = "queued"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class Task:
    """Task definition for worker execution."""
    
    task_id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    description: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    required_capabilities: List[str] = field(default_factory=list)
    max_cost: Optional[float] = None
    timeout_seconds: int = 300
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    assigned_to: Optional[str] = None
    status: TaskStatus = TaskStatus.QUEUED
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['priority'] = self.priority.value
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create task from dictionary."""
        # Handle enum conversions
        if 'priority' in data and isinstance(data['priority'], int):
            data['priority'] = TaskPriority(data['priority'])
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = TaskStatus(data['status'])
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        return cls(**data)


@dataclass
class TaskResult:
    """Result from worker task execution."""
    
    task_id: str
    worker_id: str
    status: TaskStatus
    result: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    # Performance metrics
    execution_time_seconds: float = 0.0
    tokens_used: int = 0
    cost: float = 0.0
    quality_score: Optional[float] = None
    
    # Metadata
    completed_at: datetime = field(default_factory=datetime.utcnow)
    worker_capabilities: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        data = asdict(self)
        data['completed_at'] = self.completed_at.isoformat()
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskResult':
        """Create result from dictionary."""
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = TaskStatus(data['status'])
        if 'completed_at' in data and isinstance(data['completed_at'], str):
            data['completed_at'] = datetime.fromisoformat(data['completed_at'])
        
        return cls(**data)


class MessageQueue:
    """
    Async message queue for orchestrator-worker communication.
    
    Features:
    - Priority-based task scheduling
    - Worker capability matching
    - Cost-aware task distribution
    - Reliable delivery with retries
    - Real-time status tracking
    """
    
    def __init__(self, max_queue_size: int = 1000):
        self.max_queue_size = max_queue_size
        
        # Task storage by priority
        self._queues: Dict[TaskPriority, asyncio.Queue] = {
            priority: asyncio.Queue(maxsize=max_queue_size) 
            for priority in TaskPriority
        }
        
        # Task tracking
        self._tasks: Dict[str, Task] = {}
        self._results: Dict[str, TaskResult] = {}
        self._task_assignments: Dict[str, str] = {}  # task_id -> worker_id
        
        # Worker tracking
        self._workers: Dict[str, Dict[str, Any]] = {}
        self._worker_capabilities: Dict[str, List[str]] = {}
        self._worker_load: Dict[str, int] = {}
        
        # Event handlers
        self._task_handlers: List[Callable[[Task], None]] = []
        self._result_handlers: List[Callable[[TaskResult], None]] = []
        
        # Statistics
        self.stats = {
            "tasks_queued": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_cost": 0.0,
            "average_execution_time": 0.0
        }
        
        logger.info("📬 MessageQueue initialized")
    
    async def submit_task(self, task: Task) -> str:
        """Submit a task to the queue."""
        if len(self._tasks) >= self.max_queue_size:
            raise RuntimeError(f"Queue at maximum capacity ({self.max_queue_size})")
        
        # Store task
        self._tasks[task.task_id] = task
        
        # Add to appropriate priority queue
        priority_queue = self._queues[task.priority]
        await priority_queue.put(task)
        
        self.stats["tasks_queued"] += 1
        
        # Notify handlers
        for handler in self._task_handlers:
            try:
                handler(task)
            except Exception as e:
                logger.error(f"Task handler error: {e}")
        
        logger.info(f"📝 Task {task.task_id} queued with {task.priority.name} priority")
        return task.task_id
    
    async def get_next_task(self, worker_id: str, capabilities: List[str]) -> Optional[Task]:
        """Get next task for a worker based on capabilities and priority."""
        
        # Update worker info
        self._workers[worker_id] = {
            "last_seen": datetime.utcnow(),
            "capabilities": capabilities,
            "status": "active"
        }
        self._worker_capabilities[worker_id] = capabilities
        
        # Try to get task from highest priority queue first
        for priority in sorted(TaskPriority, key=lambda p: p.value, reverse=True):
            queue = self._queues[priority]
            
            # Check if queue has tasks
            if queue.empty():
                continue
            
            # Look for compatible task
            temp_tasks = []
            task = None
            
            # Try to find a compatible task
            while not queue.empty():
                candidate_task = await queue.get()
                
                # Check if worker can handle this task
                if self._worker_can_handle_task(capabilities, candidate_task):
                    task = candidate_task
                    break
                else:
                    temp_tasks.append(candidate_task)
            
            # Put back incompatible tasks
            for temp_task in temp_tasks:
                await queue.put(temp_task)
            
            if task:
                # Assign task to worker
                task.assigned_to = worker_id
                task.status = TaskStatus.ASSIGNED
                self._task_assignments[task.task_id] = worker_id
                self._worker_load[worker_id] = self._worker_load.get(worker_id, 0) + 1
                
                logger.info(f"📤 Task {task.task_id} assigned to worker {worker_id}")
                return task
        
        # No compatible tasks found
        return None
    
    async def submit_result(self, result: TaskResult) -> None:
        """Submit task result from worker."""
        
        # Store result
        self._results[result.task_id] = result
        
        # Update task status
        if result.task_id in self._tasks:
            task = self._tasks[result.task_id]
            task.status = result.status
        
        # Update worker load
        if result.worker_id in self._worker_load:
            self._worker_load[result.worker_id] = max(0, self._worker_load[result.worker_id] - 1)
        
        # Update statistics
        if result.status == TaskStatus.COMPLETED:
            self.stats["tasks_completed"] += 1
        elif result.status == TaskStatus.FAILED:
            self.stats["tasks_failed"] += 1
            
            # Handle retries
            if result.task_id in self._tasks:
                task = self._tasks[result.task_id]
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    task.status = TaskStatus.QUEUED
                    task.assigned_to = None
                    
                    # Re-queue for retry
                    await self._queues[task.priority].put(task)
                    logger.info(f"🔄 Task {task.task_id} queued for retry ({task.retry_count}/{task.max_retries})")
        
        self.stats["total_cost"] += result.cost
        
        # Update average execution time
        if result.execution_time_seconds > 0:
            total_completed = self.stats["tasks_completed"]
            if total_completed > 0:
                current_avg = self.stats["average_execution_time"]
                self.stats["average_execution_time"] = (
                    (current_avg * (total_completed - 1) + result.execution_time_seconds) / total_completed
                )
        
        # Notify handlers
        for handler in self._result_handlers:
            try:
                handler(result)
            except Exception as e:
                logger.error(f"Result handler error: {e}")
        
        logger.info(f"✅ Result received for task {result.task_id} from worker {result.worker_id}")
    
    def _worker_can_handle_task(self, worker_capabilities: List[str], task: Task) -> bool:
        """Check if worker has required capabilities for task."""
        if not task.required_capabilities:
            return True  # No specific requirements
        
        return all(cap in worker_capabilities for cap in task.required_capabilities)
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a task."""
        if task_id not in self._tasks:
            return None
        
        task = self._tasks[task_id]
        result = self._results.get(task_id)
        
        status_info = {
            "task_id": task_id,
            "status": task.status.value,
            "created_at": task.created_at.isoformat(),
            "assigned_to": task.assigned_to,
            "retry_count": task.retry_count,
            "priority": task.priority.name
        }
        
        if result:
            status_info.update({
                "completed_at": result.completed_at.isoformat(),
                "execution_time": result.execution_time_seconds,
                "cost": result.cost,
                "quality_score": result.quality_score,
                "error": result.error
            })
        
        return status_info
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get overall queue status."""
        queue_sizes = {
            priority.name: queue.qsize() 
            for priority, queue in self._queues.items()
        }
        
        worker_stats = {}
        for worker_id, capabilities in self._worker_capabilities.items():
            worker_info = self._workers.get(worker_id, {})
            worker_stats[worker_id] = {
                "capabilities": capabilities,
                "current_load": self._worker_load.get(worker_id, 0),
                "last_seen": worker_info.get("last_seen", "unknown"),
                "status": worker_info.get("status", "unknown")
            }
        
        return {
            "queue_sizes": queue_sizes,
            "total_queued": sum(queue_sizes.values()),
            "active_workers": len(self._workers),
            "worker_details": worker_stats,
            "statistics": self.stats.copy()
        }
    
    def add_task_handler(self, handler: Callable[[Task], None]) -> None:
        """Add handler for new tasks."""
        self._task_handlers.append(handler)
    
    def add_result_handler(self, handler: Callable[[TaskResult], None]) -> None:
        """Add handler for task results."""
        self._result_handlers.append(handler)
    
    async def cleanup_old_tasks(self, max_age_hours: int = 24) -> int:
        """Clean up old completed/failed tasks."""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        tasks_to_remove = []
        for task_id, task in self._tasks.items():
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                if task.created_at < cutoff_time:
                    tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del self._tasks[task_id]
            if task_id in self._results:
                del self._results[task_id]
            if task_id in self._task_assignments:
                del self._task_assignments[task_id]
        
        logger.info(f"🧹 Cleaned up {len(tasks_to_remove)} old tasks")
        return len(tasks_to_remove)