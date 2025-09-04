"""
Agent Health Monitoring System

Implements comprehensive health monitoring for agent swarms with automatic restart
capabilities, health checks, and performance tracking based on AI agent swarm
best practices.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
import json
from pathlib import Path


logger = logging.getLogger(__name__)


class AgentHealthStatus(Enum):
    """Agent health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    FAILED = "failed"
    RESTARTING = "restarting"
    UNKNOWN = "unknown"


@dataclass
class AgentHealthMetrics:
    """Health metrics for an individual agent"""
    agent_id: str
    agent_type: str
    status: AgentHealthStatus = AgentHealthStatus.UNKNOWN
    last_heartbeat: Optional[datetime] = None
    response_time_ms: float = 0.0
    success_rate: float = 0.0
    error_count: int = 0
    restart_count: int = 0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    task_completion_rate: float = 0.0
    last_successful_task: Optional[datetime] = None
    consecutive_failures: int = 0
    health_score: float = 100.0
    
    # Historical data
    response_times: List[float] = field(default_factory=list)
    error_history: List[Dict[str, Any]] = field(default_factory=list)
    performance_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class HealthThresholds:
    """Configurable health check thresholds"""
    max_response_time_ms: float = 5000.0
    min_success_rate: float = 0.8
    max_consecutive_failures: int = 3
    heartbeat_timeout_seconds: int = 30
    max_memory_usage_mb: float = 512.0
    max_cpu_usage_percent: float = 80.0
    min_task_completion_rate: float = 0.7
    unhealthy_threshold_score: float = 60.0
    failed_threshold_score: float = 30.0


class AgentHealthMonitor:
    """
    Comprehensive agent health monitoring system with automatic restart capabilities
    
    Features:
    - Real-time health monitoring
    - Automatic agent restart on failures
    - Performance tracking and analytics
    - Configurable health thresholds
    - Health scoring algorithm
    - Historical metrics storage
    """
    
    def __init__(self, 
                 thresholds: Optional[HealthThresholds] = None,
                 metrics_storage_path: Optional[str] = None,
                 auto_restart: bool = True,
                 monitoring_interval: float = 10.0):
        """
        Initialize the health monitoring system
        
        Args:
            thresholds: Health check thresholds configuration
            metrics_storage_path: Path to store historical metrics
            auto_restart: Enable automatic agent restart on failures
            monitoring_interval: Health check interval in seconds
        """
        self.thresholds = thresholds or HealthThresholds()
        self.auto_restart = auto_restart
        self.monitoring_interval = monitoring_interval
        
        # Agent metrics storage
        self.agent_metrics: Dict[str, AgentHealthMetrics] = {}
        self.agent_restart_callbacks: Dict[str, Callable] = {}
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
        
        # Metrics storage
        self.metrics_storage_path = Path(metrics_storage_path) if metrics_storage_path else None
        if self.metrics_storage_path:
            self.metrics_storage_path.mkdir(parents=True, exist_ok=True)
        
        # Event listeners
        self.health_change_listeners: List[Callable] = []
        self.restart_listeners: List[Callable] = []
        
        logger.info(f"Health monitor initialized with auto_restart={auto_restart}")
    
    def register_agent(self, 
                      agent_id: str, 
                      agent_type: str,
                      restart_callback: Optional[Callable] = None):
        """Register an agent for health monitoring"""
        self.agent_metrics[agent_id] = AgentHealthMetrics(
            agent_id=agent_id,
            agent_type=agent_type,
            last_heartbeat=datetime.now()
        )
        
        if restart_callback:
            self.agent_restart_callbacks[agent_id] = restart_callback
        
        logger.info(f"Registered agent {agent_id} ({agent_type}) for health monitoring")
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent from health monitoring"""
        if agent_id in self.agent_metrics:
            # Save final metrics before removal
            self._save_agent_metrics(agent_id)
            del self.agent_metrics[agent_id]
        
        if agent_id in self.agent_restart_callbacks:
            del self.agent_restart_callbacks[agent_id]
        
        logger.info(f"Unregistered agent {agent_id} from health monitoring")
    
    def update_agent_heartbeat(self, agent_id: str):
        """Update agent heartbeat timestamp"""
        if agent_id in self.agent_metrics:
            self.agent_metrics[agent_id].last_heartbeat = datetime.now()
    
    def record_task_completion(self, 
                             agent_id: str, 
                             success: bool, 
                             response_time_ms: float,
                             error_details: Optional[Dict] = None):
        """Record task completion metrics for an agent"""
        if agent_id not in self.agent_metrics:
            logger.warning(f"Agent {agent_id} not registered for monitoring")
            return
        
        metrics = self.agent_metrics[agent_id]
        
        # Update response time
        metrics.response_times.append(response_time_ms)
        if len(metrics.response_times) > 100:  # Keep last 100 samples
            metrics.response_times.pop(0)
        
        metrics.response_time_ms = sum(metrics.response_times) / len(metrics.response_times)
        
        # Update success/failure tracking
        if success:
            metrics.consecutive_failures = 0
            metrics.last_successful_task = datetime.now()
        else:
            metrics.consecutive_failures += 1
            metrics.error_count += 1
            
            if error_details:
                metrics.error_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'error': error_details
                })
                if len(metrics.error_history) > 50:  # Keep last 50 errors
                    metrics.error_history.pop(0)
        
        # Recalculate health score
        self._calculate_health_score(agent_id)
        
        # Update heartbeat
        self.update_agent_heartbeat(agent_id)
        
        logger.debug(f"Recorded task completion for {agent_id}: success={success}, "
                    f"response_time={response_time_ms}ms")
    
    def update_resource_usage(self, 
                            agent_id: str, 
                            memory_mb: float, 
                            cpu_percent: float):
        """Update agent resource usage metrics"""
        if agent_id not in self.agent_metrics:
            return
        
        metrics = self.agent_metrics[agent_id]
        metrics.memory_usage_mb = memory_mb
        metrics.cpu_usage_percent = cpu_percent
        
        # Store performance history
        metrics.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            'memory_mb': memory_mb,
            'cpu_percent': cpu_percent,
            'response_time_ms': metrics.response_time_ms
        })
        
        if len(metrics.performance_history) > 100:
            metrics.performance_history.pop(0)
        
        # Recalculate health score
        self._calculate_health_score(agent_id)
    
    def _calculate_health_score(self, agent_id: str) -> float:
        """Calculate comprehensive health score for an agent"""
        if agent_id not in self.agent_metrics:
            return 0.0
        
        metrics = self.agent_metrics[agent_id]
        score = 100.0
        
        # Response time penalty (0-20 points)
        if metrics.response_time_ms > self.thresholds.max_response_time_ms:
            response_penalty = min(20.0, 
                                 (metrics.response_time_ms - self.thresholds.max_response_time_ms) / 
                                 self.thresholds.max_response_time_ms * 20)
            score -= response_penalty
        
        # Consecutive failures penalty (0-30 points)
        if metrics.consecutive_failures > 0:
            failure_penalty = min(30.0, metrics.consecutive_failures * 10)
            score -= failure_penalty
        
        # Memory usage penalty (0-15 points)
        if metrics.memory_usage_mb > self.thresholds.max_memory_usage_mb:
            memory_penalty = min(15.0,
                               (metrics.memory_usage_mb - self.thresholds.max_memory_usage_mb) /
                               self.thresholds.max_memory_usage_mb * 15)
            score -= memory_penalty
        
        # CPU usage penalty (0-15 points)
        if metrics.cpu_usage_percent > self.thresholds.max_cpu_usage_percent:
            cpu_penalty = min(15.0,
                            (metrics.cpu_usage_percent - self.thresholds.max_cpu_usage_percent) /
                            self.thresholds.max_cpu_usage_percent * 15)
            score -= cpu_penalty
        
        # Heartbeat timeout penalty (0-20 points)
        if metrics.last_heartbeat:
            time_since_heartbeat = (datetime.now() - metrics.last_heartbeat).total_seconds()
            if time_since_heartbeat > self.thresholds.heartbeat_timeout_seconds:
                heartbeat_penalty = min(20.0, time_since_heartbeat / 
                                      self.thresholds.heartbeat_timeout_seconds * 10)
                score -= heartbeat_penalty
        
        metrics.health_score = max(0.0, score)
        
        # Update status based on score
        old_status = metrics.status
        if metrics.health_score >= 80.0:
            metrics.status = AgentHealthStatus.HEALTHY
        elif metrics.health_score >= self.thresholds.unhealthy_threshold_score:
            metrics.status = AgentHealthStatus.DEGRADED
        elif metrics.health_score >= self.thresholds.failed_threshold_score:
            metrics.status = AgentHealthStatus.UNHEALTHY
        else:
            metrics.status = AgentHealthStatus.FAILED
        
        # Notify listeners of status changes
        if old_status != metrics.status:
            self._notify_health_change(agent_id, old_status, metrics.status)
        
        return metrics.health_score
    
    def _notify_health_change(self, 
                            agent_id: str, 
                            old_status: AgentHealthStatus, 
                            new_status: AgentHealthStatus):
        """Notify listeners of agent health status changes"""
        for listener in self.health_change_listeners:
            try:
                listener(agent_id, old_status, new_status)
            except Exception as e:
                logger.error(f"Error in health change listener: {e}")
        
        logger.info(f"Agent {agent_id} status changed: {old_status.value} -> {new_status.value}")
    
    async def start_monitoring(self):
        """Start the health monitoring loop"""
        if self.is_monitoring:
            logger.warning("Health monitoring already started")
            return
        
        self.is_monitoring = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started agent health monitoring")
    
    async def stop_monitoring(self):
        """Stop the health monitoring loop"""
        self.is_monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped agent health monitoring")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.monitoring_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _perform_health_checks(self):
        """Perform health checks on all registered agents"""
        for agent_id in list(self.agent_metrics.keys()):
            try:
                metrics = self.agent_metrics[agent_id]
                
                # Check heartbeat timeout
                if metrics.last_heartbeat:
                    time_since_heartbeat = (datetime.now() - metrics.last_heartbeat).total_seconds()
                    if time_since_heartbeat > self.thresholds.heartbeat_timeout_seconds:
                        logger.warning(f"Agent {agent_id} heartbeat timeout: {time_since_heartbeat}s")
                
                # Recalculate health score
                self._calculate_health_score(agent_id)
                
                # Check if restart is needed
                if self.auto_restart and self._should_restart_agent(agent_id):
                    await self._restart_agent(agent_id)
                
            except Exception as e:
                logger.error(f"Error checking health for agent {agent_id}: {e}")
    
    def _should_restart_agent(self, agent_id: str) -> bool:
        """Determine if an agent should be restarted"""
        if agent_id not in self.agent_metrics:
            return False
        
        metrics = self.agent_metrics[agent_id]
        
        # Don't restart if already restarting
        if metrics.status == AgentHealthStatus.RESTARTING:
            return False
        
        # Restart conditions
        return (
            metrics.status == AgentHealthStatus.FAILED or
            metrics.consecutive_failures >= self.thresholds.max_consecutive_failures or
            (metrics.last_heartbeat and 
             (datetime.now() - metrics.last_heartbeat).total_seconds() > 
             self.thresholds.heartbeat_timeout_seconds * 2)
        )
    
    async def _restart_agent(self, agent_id: str):
        """Restart a failed agent"""
        if agent_id not in self.agent_metrics:
            return
        
        metrics = self.agent_metrics[agent_id]
        old_status = metrics.status
        metrics.status = AgentHealthStatus.RESTARTING
        metrics.restart_count += 1
        
        logger.warning(f"Restarting agent {agent_id} (restart #{metrics.restart_count})")
        
        # Notify restart listeners
        for listener in self.restart_listeners:
            try:
                await listener(agent_id, metrics.restart_count)
            except Exception as e:
                logger.error(f"Error in restart listener: {e}")
        
        # Call restart callback if available
        if agent_id in self.agent_restart_callbacks:
            try:
                callback = self.agent_restart_callbacks[agent_id]
                if asyncio.iscoroutinefunction(callback):
                    await callback(agent_id)
                else:
                    callback(agent_id)
                
                # Reset failure counters on successful restart
                metrics.consecutive_failures = 0
                metrics.error_count = 0
                metrics.last_heartbeat = datetime.now()
                metrics.status = AgentHealthStatus.HEALTHY
                
                logger.info(f"Successfully restarted agent {agent_id}")
                
            except Exception as e:
                logger.error(f"Failed to restart agent {agent_id}: {e}")
                metrics.status = AgentHealthStatus.FAILED
        else:
            logger.warning(f"No restart callback available for agent {agent_id}")
            metrics.status = old_status
    
    def get_agent_health(self, agent_id: str) -> Optional[AgentHealthMetrics]:
        """Get health metrics for a specific agent"""
        return self.agent_metrics.get(agent_id)
    
    def get_all_agent_health(self) -> Dict[str, AgentHealthMetrics]:
        """Get health metrics for all agents"""
        return self.agent_metrics.copy()
    
    def get_unhealthy_agents(self) -> List[str]:
        """Get list of unhealthy agent IDs"""
        return [
            agent_id for agent_id, metrics in self.agent_metrics.items()
            if metrics.status in [AgentHealthStatus.UNHEALTHY, AgentHealthStatus.FAILED]
        ]
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary"""
        total_agents = len(self.agent_metrics)
        if total_agents == 0:
            return {
                'total_agents': 0,
                'healthy': 0,
                'degraded': 0,
                'unhealthy': 0,
                'failed': 0,
                'overall_health_score': 100.0,
                'timestamp': datetime.now().isoformat()
            }
        
        status_counts = {
            'healthy': 0,
            'degraded': 0, 
            'unhealthy': 0,
            'failed': 0,
            'restarting': 0
        }
        
        total_score = 0.0
        for metrics in self.agent_metrics.values():
            status_counts[metrics.status.value] += 1
            total_score += metrics.health_score
        
        overall_health_score = total_score / total_agents
        
        return {
            'total_agents': total_agents,
            'healthy': status_counts['healthy'],
            'degraded': status_counts['degraded'],
            'unhealthy': status_counts['unhealthy'],
            'failed': status_counts['failed'],
            'restarting': status_counts['restarting'],
            'overall_health_score': overall_health_score,
            'timestamp': datetime.now().isoformat()
        }
    
    def _save_agent_metrics(self, agent_id: str):
        """Save agent metrics to persistent storage"""
        if not self.metrics_storage_path or agent_id not in self.agent_metrics:
            return
        
        try:
            metrics = self.agent_metrics[agent_id]
            metrics_file = self.metrics_storage_path / f"{agent_id}_metrics.json"
            
            # Convert metrics to JSON-serializable format
            metrics_data = {
                'agent_id': metrics.agent_id,
                'agent_type': metrics.agent_type,
                'status': metrics.status.value,
                'health_score': metrics.health_score,
                'response_time_ms': metrics.response_time_ms,
                'success_rate': metrics.success_rate,
                'error_count': metrics.error_count,
                'restart_count': metrics.restart_count,
                'memory_usage_mb': metrics.memory_usage_mb,
                'cpu_usage_percent': metrics.cpu_usage_percent,
                'consecutive_failures': metrics.consecutive_failures,
                'last_heartbeat': metrics.last_heartbeat.isoformat() if metrics.last_heartbeat else None,
                'last_successful_task': metrics.last_successful_task.isoformat() if metrics.last_successful_task else None,
                'response_times': metrics.response_times[-20:],  # Save last 20 samples
                'error_history': metrics.error_history[-10:],    # Save last 10 errors
                'performance_history': metrics.performance_history[-20:],  # Save last 20 samples
                'timestamp': datetime.now().isoformat()
            }
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save metrics for agent {agent_id}: {e}")
    
    def add_health_change_listener(self, listener: Callable):
        """Add a listener for health status changes"""
        self.health_change_listeners.append(listener)
    
    def add_restart_listener(self, listener: Callable):
        """Add a listener for agent restarts"""
        self.restart_listeners.append(listener)
    
    async def force_restart_agent(self, agent_id: str) -> bool:
        """Force restart an agent regardless of health status"""
        if agent_id not in self.agent_metrics:
            logger.error(f"Cannot restart unknown agent: {agent_id}")
            return False
        
        await self._restart_agent(agent_id)
        return True
    
    def update_thresholds(self, new_thresholds: HealthThresholds):
        """Update health monitoring thresholds"""
        self.thresholds = new_thresholds
        logger.info("Updated health monitoring thresholds")
        
        # Recalculate all health scores with new thresholds
        for agent_id in self.agent_metrics.keys():
            self._calculate_health_score(agent_id)