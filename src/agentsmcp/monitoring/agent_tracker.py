"""
Agent activity tracking and status monitoring system.

Tracks the status and activities of all agents in the system, providing
real-time insights into agent workload, task progress, and health status.
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import weakref

from .metrics_collector import get_metrics_collector, record_gauge, record_counter, record_histogram

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent operational status."""
    IDLE = "idle"
    STARTING = "starting"
    THINKING = "thinking"
    WORKING = "working"
    WAITING_RESOURCE = "waiting_resource"
    WAITING_INPUT = "waiting_input"
    COMPLETING = "completing"
    ERROR = "error"
    STOPPING = "stopping"
    STOPPED = "stopped"


class TaskPhase(Enum):
    """Phase of task execution."""
    QUEUED = "queued"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    EXECUTING = "executing"
    VALIDATING = "validating"
    COMPLETING = "completing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskInfo:
    """Information about a task being executed by an agent."""
    task_id: str
    description: str
    phase: TaskPhase
    start_time: float
    estimated_duration: Optional[float] = None
    progress_percentage: float = 0.0
    current_step: str = ""
    total_steps: Optional[int] = None
    current_step_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time
    
    @property
    def estimated_remaining(self) -> Optional[float]:
        """Get estimated remaining time in seconds."""
        if not self.estimated_duration:
            return None
        return max(0, self.estimated_duration - self.elapsed_time)


@dataclass 
class AgentActivity:
    """Current activity of an agent."""
    agent_id: str
    agent_type: str
    status: AgentStatus
    current_task: Optional[TaskInfo] = None
    task_queue_size: int = 0
    last_update: float = field(default_factory=time.time)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None
    resource_usage: Dict[str, float] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)


@dataclass
class AgentMetrics:
    """Performance metrics for an agent."""
    tasks_completed: int = 0
    tasks_failed: int = 0
    average_task_duration: float = 0.0
    success_rate: float = 100.0
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    response_time_p95: float = 0.0
    uptime_seconds: float = 0.0
    last_activity: float = field(default_factory=time.time)


class AgentTracker:
    """
    Tracks agent status, activities, and performance metrics.
    
    Provides real-time monitoring of all agents in the system,
    including their current tasks, resource usage, and health status.
    """
    
    def __init__(self, update_interval: float = 0.5):
        """
        Initialize agent tracker.
        
        Args:
            update_interval: How often to update metrics in seconds
        """
        self.update_interval = update_interval
        
        # Agent tracking
        self._agents: Dict[str, AgentActivity] = {}
        self._agent_metrics: Dict[str, AgentMetrics] = defaultdict(AgentMetrics)
        self._task_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Threading
        self._lock = threading.RLock()
        self._running = False
        self._update_task: Optional[asyncio.Task] = None
        
        # Listeners
        self._status_listeners: Set[Callable[[str, AgentStatus, AgentStatus], None]] = set()
        self._task_listeners: Set[Callable[[str, Optional[TaskInfo], Optional[TaskInfo]], None]] = set()
        self._metrics_listeners: Set[Callable[[str, AgentMetrics], None]] = set()
        self._weak_listeners: weakref.WeakSet = weakref.WeakSet()
        
        # Global metrics
        self.metrics_collector = get_metrics_collector()
        
        logger.info(f"AgentTracker initialized with update_interval={update_interval}")
    
    def start(self):
        """Start the agent tracking system."""
        if self._running:
            return
        
        self._running = True
        
        # Start update task in current event loop
        try:
            loop = asyncio.get_event_loop()
            self._update_task = loop.create_task(self._update_loop())
            logger.info("AgentTracker started")
        except RuntimeError:
            logger.warning("No event loop available, agent tracking will be manual")
    
    async def stop(self):
        """Stop the agent tracking system."""
        self._running = False
        
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
            self._update_task = None
        
        logger.info("AgentTracker stopped")
    
    def register_agent(self, agent_id: str, agent_type: str, capabilities: List[str] = None):
        """Register a new agent for tracking."""
        with self._lock:
            if agent_id in self._agents:
                logger.warning(f"Agent {agent_id} already registered, updating")
            
            activity = AgentActivity(
                agent_id=agent_id,
                agent_type=agent_type,
                status=AgentStatus.IDLE,
                capabilities=capabilities or []
            )
            
            self._agents[agent_id] = activity
            
            # Record registration
            record_counter('agent.registered', 1.0, {'agent_type': agent_type})
            record_gauge('agent.count', len(self._agents))
            
            logger.info(f"Registered agent: {agent_id} ({agent_type})")
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent from tracking."""
        with self._lock:
            if agent_id not in self._agents:
                logger.warning(f"Agent {agent_id} not registered")
                return
            
            agent_type = self._agents[agent_id].agent_type
            del self._agents[agent_id]
            
            if agent_id in self._agent_metrics:
                del self._agent_metrics[agent_id]
            
            if agent_id in self._task_history:
                del self._task_history[agent_id]
            
            # Record unregistration
            record_counter('agent.unregistered', 1.0, {'agent_type': agent_type})
            record_gauge('agent.count', len(self._agents))
            
            logger.info(f"Unregistered agent: {agent_id}")
    
    def update_agent_status(self, agent_id: str, new_status: AgentStatus, 
                          error_message: Optional[str] = None):
        """Update agent status."""
        with self._lock:
            if agent_id not in self._agents:
                logger.warning(f"Agent {agent_id} not registered")
                return
            
            activity = self._agents[agent_id]
            old_status = activity.status
            activity.status = new_status
            activity.last_update = time.time()
            activity.error_message = error_message
            
            # Record status change
            record_counter('agent.status_change', 1.0, {
                'agent_id': agent_id,
                'agent_type': activity.agent_type,
                'from_status': old_status.value,
                'to_status': new_status.value
            })
            
            # Update status metrics
            self._update_status_metrics()
            
            # Notify listeners
            self._notify_status_listeners(agent_id, old_status, new_status)
            
            logger.debug(f"Agent {agent_id} status: {old_status.value} -> {new_status.value}")
    
    def start_task(self, agent_id: str, task_id: str, description: str, 
                   estimated_duration: Optional[float] = None, 
                   total_steps: Optional[int] = None):
        """Start a new task for an agent."""
        with self._lock:
            if agent_id not in self._agents:
                logger.warning(f"Agent {agent_id} not registered")
                return
            
            activity = self._agents[agent_id]
            old_task = activity.current_task
            
            new_task = TaskInfo(
                task_id=task_id,
                description=description,
                phase=TaskPhase.QUEUED,
                start_time=time.time(),
                estimated_duration=estimated_duration,
                total_steps=total_steps
            )
            
            activity.current_task = new_task
            activity.last_update = time.time()
            
            # Update queue size (assuming task was queued)
            activity.task_queue_size = max(0, activity.task_queue_size - 1)
            
            # Record task start
            record_counter('agent.task_started', 1.0, {
                'agent_id': agent_id,
                'agent_type': activity.agent_type
            })
            
            # Notify listeners
            self._notify_task_listeners(agent_id, old_task, new_task)
            
            logger.info(f"Agent {agent_id} started task: {task_id}")
    
    def update_task_progress(self, agent_id: str, phase: TaskPhase,
                           progress_percentage: float = 0.0,
                           current_step: str = "",
                           current_step_index: int = 0):
        """Update task progress."""
        with self._lock:
            if agent_id not in self._agents:
                logger.warning(f"Agent {agent_id} not registered")
                return
            
            activity = self._agents[agent_id]
            if not activity.current_task:
                logger.warning(f"Agent {agent_id} has no current task")
                return
            
            task = activity.current_task
            old_phase = task.phase
            
            task.phase = phase
            task.progress_percentage = progress_percentage
            task.current_step = current_step
            task.current_step_index = current_step_index
            activity.last_update = time.time()
            
            # Record progress
            record_gauge('agent.task_progress', progress_percentage, {
                'agent_id': agent_id,
                'task_id': task.task_id,
                'phase': phase.value
            })
            
            if old_phase != phase:
                record_counter('agent.task_phase_change', 1.0, {
                    'agent_id': agent_id,
                    'from_phase': old_phase.value,
                    'to_phase': phase.value
                })
            
            logger.debug(f"Agent {agent_id} task progress: {phase.value} {progress_percentage:.1f}%")
    
    def complete_task(self, agent_id: str, success: bool = True, 
                     result_metadata: Optional[Dict[str, Any]] = None):
        """Complete the current task for an agent."""
        with self._lock:
            if agent_id not in self._agents:
                logger.warning(f"Agent {agent_id} not registered")
                return
            
            activity = self._agents[agent_id]
            if not activity.current_task:
                logger.warning(f"Agent {agent_id} has no current task to complete")
                return
            
            task = activity.current_task
            task.phase = TaskPhase.COMPLETED if success else TaskPhase.FAILED
            task.progress_percentage = 100.0 if success else task.progress_percentage
            
            # Update metrics
            metrics = self._agent_metrics[agent_id]
            if success:
                metrics.tasks_completed += 1
            else:
                metrics.tasks_failed += 1
            
            # Update average duration
            duration = task.elapsed_time
            total_tasks = metrics.tasks_completed + metrics.tasks_failed
            if total_tasks > 1:
                metrics.average_task_duration = (
                    (metrics.average_task_duration * (total_tasks - 1) + duration) / total_tasks
                )
            else:
                metrics.average_task_duration = duration
            
            # Update success rate
            metrics.success_rate = (metrics.tasks_completed / total_tasks) * 100 if total_tasks > 0 else 100
            
            # Store in history
            task.metadata.update(result_metadata or {})
            self._task_history[agent_id].append(task)
            
            # Clear current task
            old_task = activity.current_task
            activity.current_task = None
            activity.last_update = time.time()
            
            # Record completion
            record_counter('agent.task_completed', 1.0, {
                'agent_id': agent_id,
                'agent_type': activity.agent_type,
                'success': str(success).lower()
            })
            record_histogram('agent.task_duration', duration, {
                'agent_id': agent_id,
                'agent_type': activity.agent_type
            })
            
            # Notify listeners
            self._notify_task_listeners(agent_id, old_task, None)
            self._notify_metrics_listeners(agent_id, metrics)
            
            logger.info(f"Agent {agent_id} completed task: {task.task_id} (success: {success})")
    
    def add_to_queue(self, agent_id: str, queue_size_delta: int = 1):
        """Update agent's task queue size."""
        with self._lock:
            if agent_id not in self._agents:
                logger.warning(f"Agent {agent_id} not registered")
                return
            
            activity = self._agents[agent_id]
            activity.task_queue_size = max(0, activity.task_queue_size + queue_size_delta)
            activity.last_update = time.time()
            
            record_gauge('agent.queue_size', activity.task_queue_size, {
                'agent_id': agent_id,
                'agent_type': activity.agent_type
            })
    
    def update_resource_usage(self, agent_id: str, cpu_percent: float = 0.0,
                            memory_mb: float = 0.0, custom_metrics: Optional[Dict[str, float]] = None):
        """Update agent resource usage metrics."""
        with self._lock:
            if agent_id not in self._agents:
                logger.warning(f"Agent {agent_id} not registered")
                return
            
            activity = self._agents[agent_id]
            metrics = self._agent_metrics[agent_id]
            
            # Update resource usage
            activity.resource_usage.update({
                'cpu_percent': cpu_percent,
                'memory_mb': memory_mb
            })
            
            if custom_metrics:
                activity.resource_usage.update(custom_metrics)
            
            # Update metrics
            metrics.cpu_usage_percent = cpu_percent
            metrics.memory_usage_mb = memory_mb
            metrics.last_activity = time.time()
            
            # Record resource metrics
            record_gauge('agent.cpu_usage', cpu_percent, {
                'agent_id': agent_id,
                'agent_type': activity.agent_type
            })
            record_gauge('agent.memory_usage', memory_mb, {
                'agent_id': agent_id,
                'agent_type': activity.agent_type
            })
            
            activity.last_update = time.time()
    
    def get_agent_activity(self, agent_id: str) -> Optional[AgentActivity]:
        """Get current activity for a specific agent."""
        with self._lock:
            return self._agents.get(agent_id)
    
    def get_all_activities(self) -> Dict[str, AgentActivity]:
        """Get current activities for all agents."""
        with self._lock:
            return self._agents.copy()
    
    def get_agent_metrics(self, agent_id: str) -> Optional[AgentMetrics]:
        """Get performance metrics for a specific agent."""
        with self._lock:
            return self._agent_metrics.get(agent_id)
    
    def get_all_metrics(self) -> Dict[str, AgentMetrics]:
        """Get performance metrics for all agents."""
        with self._lock:
            return dict(self._agent_metrics)
    
    def get_task_history(self, agent_id: str, max_count: int = 50) -> List[TaskInfo]:
        """Get task history for an agent."""
        with self._lock:
            history = list(self._task_history.get(agent_id, []))
            return history[-max_count:] if max_count > 0 else history
    
    def get_agents_by_status(self, status: AgentStatus) -> List[str]:
        """Get list of agent IDs with specific status."""
        with self._lock:
            return [
                agent_id for agent_id, activity in self._agents.items()
                if activity.status == status
            ]
    
    def get_agents_by_type(self, agent_type: str) -> List[str]:
        """Get list of agent IDs of specific type."""
        with self._lock:
            return [
                agent_id for agent_id, activity in self._agents.items()
                if activity.agent_type == agent_type
            ]
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get system-wide summary statistics."""
        with self._lock:
            if not self._agents:
                return {
                    'total_agents': 0,
                    'agents_by_status': {},
                    'agents_by_type': {},
                    'active_tasks': 0,
                    'queued_tasks': 0,
                    'total_tasks_completed': 0,
                    'total_tasks_failed': 0,
                    'average_success_rate': 0.0
                }
            
            # Count by status
            status_counts = defaultdict(int)
            for activity in self._agents.values():
                status_counts[activity.status.value] += 1
            
            # Count by type
            type_counts = defaultdict(int)
            for activity in self._agents.values():
                type_counts[activity.agent_type] += 1
            
            # Active and queued tasks
            active_tasks = sum(1 for a in self._agents.values() if a.current_task)
            queued_tasks = sum(a.task_queue_size for a in self._agents.values())
            
            # Overall metrics
            total_completed = sum(m.tasks_completed for m in self._agent_metrics.values())
            total_failed = sum(m.tasks_failed for m in self._agent_metrics.values())
            
            # Average success rate
            success_rates = [m.success_rate for m in self._agent_metrics.values() if m.tasks_completed + m.tasks_failed > 0]
            avg_success_rate = sum(success_rates) / len(success_rates) if success_rates else 0.0
            
            return {
                'total_agents': len(self._agents),
                'agents_by_status': dict(status_counts),
                'agents_by_type': dict(type_counts),
                'active_tasks': active_tasks,
                'queued_tasks': queued_tasks,
                'total_tasks_completed': total_completed,
                'total_tasks_failed': total_failed,
                'average_success_rate': round(avg_success_rate, 1)
            }
    
    def add_status_listener(self, callback: Callable[[str, AgentStatus, AgentStatus], None]):
        """Add listener for agent status changes."""
        self._status_listeners.add(callback)
    
    def add_task_listener(self, callback: Callable[[str, Optional[TaskInfo], Optional[TaskInfo]], None]):
        """Add listener for agent task changes."""
        self._task_listeners.add(callback)
    
    def add_metrics_listener(self, callback: Callable[[str, AgentMetrics], None]):
        """Add listener for agent metrics updates."""
        self._metrics_listeners.add(callback)
    
    def remove_status_listener(self, callback):
        """Remove status change listener."""
        self._status_listeners.discard(callback)
    
    def remove_task_listener(self, callback):
        """Remove task change listener."""
        self._task_listeners.discard(callback)
    
    def remove_metrics_listener(self, callback):
        """Remove metrics listener."""
        self._metrics_listeners.discard(callback)
    
    async def _update_loop(self):
        """Background task to update metrics and cleanup stale data."""
        while self._running:
            try:
                await asyncio.sleep(self.update_interval)
                self._update_metrics()
                self._cleanup_stale_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in agent tracker update loop: {e}")
    
    def _update_metrics(self):
        """Update global metrics."""
        with self._lock:
            # Update status metrics
            self._update_status_metrics()
            
            # Update uptime for active agents
            current_time = time.time()
            for agent_id, metrics in self._agent_metrics.items():
                activity = self._agents.get(agent_id)
                if activity and activity.status != AgentStatus.STOPPED:
                    metrics.uptime_seconds = current_time - (metrics.last_activity or current_time)
    
    def _update_status_metrics(self):
        """Update status distribution metrics."""
        status_counts = defaultdict(int)
        for activity in self._agents.values():
            status_counts[activity.status.value] += 1
        
        for status, count in status_counts.items():
            record_gauge('agent.status_count', count, {'status': status})
    
    def _cleanup_stale_data(self):
        """Clean up stale agent data."""
        current_time = time.time()
        stale_threshold = 300  # 5 minutes
        
        with self._lock:
            stale_agents = []
            for agent_id, activity in self._agents.items():
                if current_time - activity.last_update > stale_threshold:
                    stale_agents.append(agent_id)
            
            for agent_id in stale_agents:
                logger.warning(f"Agent {agent_id} appears stale, marking as stopped")
                self.update_agent_status(agent_id, AgentStatus.STOPPED)
    
    def _notify_status_listeners(self, agent_id: str, old_status: AgentStatus, new_status: AgentStatus):
        """Notify status change listeners."""
        dead_listeners = set()
        for listener in list(self._status_listeners):
            try:
                listener(agent_id, old_status, new_status)
            except Exception as e:
                logger.warning(f"Status listener error: {e}")
                dead_listeners.add(listener)
        
        self._status_listeners -= dead_listeners
    
    def _notify_task_listeners(self, agent_id: str, old_task: Optional[TaskInfo], new_task: Optional[TaskInfo]):
        """Notify task change listeners."""
        dead_listeners = set()
        for listener in list(self._task_listeners):
            try:
                listener(agent_id, old_task, new_task)
            except Exception as e:
                logger.warning(f"Task listener error: {e}")
                dead_listeners.add(listener)
        
        self._task_listeners -= dead_listeners
    
    def _notify_metrics_listeners(self, agent_id: str, metrics: AgentMetrics):
        """Notify metrics update listeners."""
        dead_listeners = set()
        for listener in list(self._metrics_listeners):
            try:
                listener(agent_id, metrics)
            except Exception as e:
                logger.warning(f"Metrics listener error: {e}")
                dead_listeners.add(listener)
        
        self._metrics_listeners -= dead_listeners


# Global agent tracker instance
_global_tracker: Optional[AgentTracker] = None


def get_agent_tracker() -> AgentTracker:
    """Get the global agent tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = AgentTracker()
        _global_tracker.start()
    return _global_tracker