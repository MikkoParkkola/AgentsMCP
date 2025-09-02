"""
Performance monitoring system for AgentsMCP.

Monitors system performance including response times, throughput,
resource utilization, and quality gate performance.
"""

import asyncio
import logging
import time
import psutil
import platform
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict

from .metrics_collector import get_metrics_collector, record_gauge, record_counter, record_histogram, record_timer

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """System performance metrics."""
    # Response times (in seconds)
    response_time_p50: float = 0.0
    response_time_p95: float = 0.0
    response_time_p99: float = 0.0
    average_response_time: float = 0.0
    
    # Throughput
    requests_per_second: float = 0.0
    tasks_completed_per_minute: float = 0.0
    
    # System resources
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    disk_usage_percent: float = 0.0
    
    # Quality metrics
    success_rate: float = 100.0
    error_rate: float = 0.0
    quality_gate_pass_rate: float = 100.0
    
    # Agent metrics
    active_agents: int = 0
    idle_agents: int = 0
    total_agents: int = 0
    
    # Task metrics
    active_tasks: int = 0
    queued_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    
    timestamp: float = field(default_factory=time.time)


@dataclass
class ThroughputStats:
    """Throughput statistics over time windows."""
    last_minute: float = 0.0
    last_5_minutes: float = 0.0
    last_15_minutes: float = 0.0
    last_hour: float = 0.0


class PerformanceMonitor:
    """
    System performance monitoring and metrics collection.
    
    Monitors response times, throughput, resource usage, and overall
    system health to provide real-time performance insights.
    """
    
    def __init__(self, update_interval: float = 1.0):
        """
        Initialize performance monitor.
        
        Args:
            update_interval: How often to update performance metrics in seconds
        """
        self.update_interval = update_interval
        
        # Metrics storage
        self._response_times: deque = deque(maxlen=1000)
        self._request_timestamps: deque = deque(maxlen=10000)
        self._task_completions: deque = deque(maxlen=10000)
        self._error_timestamps: deque = deque(maxlen=1000)
        
        # Current metrics
        self._current_metrics = PerformanceMetrics()
        self._metrics_history: deque = deque(maxlen=3600)  # 1 hour at 1s intervals
        
        # Threading
        self._lock = threading.RLock()
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self._performance_listeners: List[Callable[[PerformanceMetrics], None]] = []
        self._alert_listeners: List[Callable[[str, str, Dict[str, Any]], None]] = []
        
        # Alert thresholds
        self.alert_thresholds = {
            'response_time_p95': 5.0,  # 5 seconds
            'cpu_usage_percent': 80.0,
            'memory_usage_percent': 90.0,
            'error_rate': 5.0,  # 5%
            'success_rate': 95.0,  # Below 95%
        }
        
        # System info
        self._system_info = self._get_system_info()
        
        # Global metrics
        self.metrics_collector = get_metrics_collector()
        
        logger.info(f"PerformanceMonitor initialized with update_interval={update_interval}")
    
    def start(self):
        """Start the performance monitoring system."""
        if self._running:
            return
        
        self._running = True
        
        # Start monitoring task in current event loop
        try:
            loop = asyncio.get_event_loop()
            self._monitor_task = loop.create_task(self._monitor_loop())
            logger.info("PerformanceMonitor started")
        except RuntimeError:
            logger.warning("No event loop available, performance monitoring will be manual")
    
    async def stop(self):
        """Stop the performance monitoring system."""
        self._running = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        
        logger.info("PerformanceMonitor stopped")
    
    def record_request(self, response_time: float, success: bool = True):
        """Record a request with its response time and success status."""
        current_time = time.time()
        
        with self._lock:
            self._response_times.append(response_time)
            self._request_timestamps.append(current_time)
            
            if not success:
                self._error_timestamps.append(current_time)
        
        # Record to global metrics
        record_histogram('request.response_time', response_time)
        record_counter('request.total', 1.0, {'success': str(success).lower()})
        if not success:
            record_counter('request.errors', 1.0)
    
    def record_task_completion(self, duration: float, success: bool = True):
        """Record a task completion with duration and success status."""
        current_time = time.time()
        
        with self._lock:
            self._task_completions.append((current_time, duration, success))
        
        # Record to global metrics
        record_histogram('task.duration', duration)
        record_counter('task.completed', 1.0, {'success': str(success).lower()})
    
    def record_quality_gate(self, gate_type: str, passed: bool, duration: float = 0.0):
        """Record quality gate execution."""
        record_counter('quality_gate.executed', 1.0, {
            'gate_type': gate_type,
            'passed': str(passed).lower()
        })
        
        if duration > 0:
            record_histogram('quality_gate.duration', duration, {'gate_type': gate_type})
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        with self._lock:
            return self._current_metrics
    
    def get_metrics_history(self, max_count: int = 3600) -> List[PerformanceMetrics]:
        """Get historical performance metrics."""
        with self._lock:
            history = list(self._metrics_history)
            return history[-max_count:] if max_count > 0 else history
    
    def get_throughput_stats(self) -> ThroughputStats:
        """Get throughput statistics for various time windows."""
        current_time = time.time()
        
        with self._lock:
            # Count requests in different time windows
            minute_count = sum(1 for ts in self._request_timestamps if current_time - ts <= 60)
            five_minute_count = sum(1 for ts in self._request_timestamps if current_time - ts <= 300)
            fifteen_minute_count = sum(1 for ts in self._request_timestamps if current_time - ts <= 900)
            hour_count = sum(1 for ts in self._request_timestamps if current_time - ts <= 3600)
            
            return ThroughputStats(
                last_minute=minute_count,
                last_5_minutes=five_minute_count / 5,
                last_15_minutes=fifteen_minute_count / 15,
                last_hour=hour_count / 60
            )
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health assessment."""
        metrics = self.get_current_metrics()
        
        # Calculate health score (0-100)
        health_factors = {
            'response_time': min(100, max(0, 100 - (metrics.response_time_p95 / 10 * 100))),
            'cpu_usage': max(0, 100 - metrics.cpu_usage_percent),
            'memory_usage': max(0, 100 - metrics.memory_usage_percent),
            'success_rate': metrics.success_rate,
            'quality_gates': metrics.quality_gate_pass_rate
        }
        
        overall_health = sum(health_factors.values()) / len(health_factors)
        
        # Determine status
        if overall_health >= 90:
            status = "excellent"
        elif overall_health >= 75:
            status = "good"
        elif overall_health >= 50:
            status = "fair"
        else:
            status = "poor"
        
        return {
            'overall_health_score': round(overall_health, 1),
            'status': status,
            'health_factors': {k: round(v, 1) for k, v in health_factors.items()},
            'active_agents': metrics.active_agents,
            'total_agents': metrics.total_agents,
            'system_load': {
                'cpu_percent': metrics.cpu_usage_percent,
                'memory_percent': metrics.memory_usage_percent,
                'active_tasks': metrics.active_tasks
            }
        }
    
    def get_performance_trends(self, minutes: int = 30) -> Dict[str, Any]:
        """Get performance trends over the specified time period."""
        with self._lock:
            # Get metrics for the time period
            cutoff_time = time.time() - (minutes * 60)
            recent_metrics = [
                m for m in self._metrics_history
                if m.timestamp >= cutoff_time
            ]
            
            if not recent_metrics:
                return {}
            
            # Calculate trends
            response_times = [m.response_time_p95 for m in recent_metrics]
            cpu_usage = [m.cpu_usage_percent for m in recent_metrics]
            memory_usage = [m.memory_usage_percent for m in recent_metrics]
            success_rates = [m.success_rate for m in recent_metrics]
            
            def calculate_trend(values):
                if len(values) < 2:
                    return 0.0
                
                # Simple linear trend calculation
                x = list(range(len(values)))
                n = len(values)
                sum_x = sum(x)
                sum_y = sum(values)
                sum_xy = sum(x[i] * values[i] for i in range(n))
                sum_x2 = sum(x[i] ** 2 for i in range(n))
                
                if n * sum_x2 - sum_x ** 2 == 0:
                    return 0.0
                
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
                return slope
            
            return {
                'time_period_minutes': minutes,
                'data_points': len(recent_metrics),
                'trends': {
                    'response_time_p95': calculate_trend(response_times),
                    'cpu_usage': calculate_trend(cpu_usage),
                    'memory_usage': calculate_trend(memory_usage),
                    'success_rate': calculate_trend(success_rates)
                },
                'averages': {
                    'response_time_p95': sum(response_times) / len(response_times) if response_times else 0,
                    'cpu_usage': sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0,
                    'memory_usage': sum(memory_usage) / len(memory_usage) if memory_usage else 0,
                    'success_rate': sum(success_rates) / len(success_rates) if success_rates else 0
                }
            }
    
    def add_performance_listener(self, callback: Callable[[PerformanceMetrics], None]):
        """Add listener for performance metrics updates."""
        self._performance_listeners.append(callback)
    
    def add_alert_listener(self, callback: Callable[[str, str, Dict[str, Any]], None]):
        """Add listener for performance alerts."""
        self._alert_listeners.append(callback)
    
    def remove_performance_listener(self, callback):
        """Remove performance metrics listener."""
        if callback in self._performance_listeners:
            self._performance_listeners.remove(callback)
    
    def remove_alert_listener(self, callback):
        """Remove alert listener."""
        if callback in self._alert_listeners:
            self._alert_listeners.remove(callback)
    
    def set_alert_threshold(self, metric_name: str, threshold: float):
        """Set alert threshold for a specific metric."""
        self.alert_thresholds[metric_name] = threshold
        logger.info(f"Set alert threshold for {metric_name}: {threshold}")
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                await asyncio.sleep(self.update_interval)
                self._update_metrics()
                self._check_alerts()
                self._notify_listeners()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in performance monitor loop: {e}")
    
    def _update_metrics(self):
        """Update current performance metrics."""
        current_time = time.time()
        
        with self._lock:
            # Calculate response time statistics
            if self._response_times:
                sorted_times = sorted(self._response_times)
                count = len(sorted_times)
                
                self._current_metrics.response_time_p50 = sorted_times[int(0.5 * count)]
                self._current_metrics.response_time_p95 = sorted_times[int(0.95 * count)]
                self._current_metrics.response_time_p99 = sorted_times[int(0.99 * count)]
                self._current_metrics.average_response_time = sum(sorted_times) / count
            
            # Calculate throughput
            minute_requests = sum(1 for ts in self._request_timestamps if current_time - ts <= 60)
            self._current_metrics.requests_per_second = minute_requests / 60
            
            minute_tasks = sum(1 for ts, _, _ in self._task_completions if current_time - ts <= 60)
            self._current_metrics.tasks_completed_per_minute = minute_tasks
            
            # Calculate success/error rates
            total_requests = len(self._request_timestamps)
            if total_requests > 0:
                recent_errors = sum(1 for ts in self._error_timestamps if current_time - ts <= 300)  # 5 minutes
                recent_requests = sum(1 for ts in self._request_timestamps if current_time - ts <= 300)
                
                if recent_requests > 0:
                    self._current_metrics.error_rate = (recent_errors / recent_requests) * 100
                    self._current_metrics.success_rate = 100 - self._current_metrics.error_rate
            
            # Get system resource usage
            try:
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                self._current_metrics.cpu_usage_percent = cpu_percent
                self._current_metrics.memory_usage_percent = memory.percent
                self._current_metrics.memory_usage_mb = memory.used / (1024 * 1024)
                self._current_metrics.disk_usage_percent = disk.percent
                
            except Exception as e:
                logger.debug(f"Error getting system metrics: {e}")
            
            # Get agent metrics from agent tracker
            try:
                from .agent_tracker import get_agent_tracker
                tracker = get_agent_tracker()
                summary = tracker.get_system_summary()
                
                self._current_metrics.active_agents = summary.get('agents_by_status', {}).get('working', 0)
                self._current_metrics.idle_agents = summary.get('agents_by_status', {}).get('idle', 0)
                self._current_metrics.total_agents = summary.get('total_agents', 0)
                self._current_metrics.active_tasks = summary.get('active_tasks', 0)
                self._current_metrics.queued_tasks = summary.get('queued_tasks', 0)
                
            except Exception as e:
                logger.debug(f"Error getting agent metrics: {e}")
            
            # Update timestamp
            self._current_metrics.timestamp = current_time
            
            # Store in history
            self._metrics_history.append(self._current_metrics)
            
            # Record to global metrics
            self._record_global_metrics()
    
    def _record_global_metrics(self):
        """Record current metrics to global metrics collector."""
        m = self._current_metrics
        
        # Response time metrics
        record_gauge('performance.response_time_p50', m.response_time_p50)
        record_gauge('performance.response_time_p95', m.response_time_p95)
        record_gauge('performance.response_time_p99', m.response_time_p99)
        
        # Throughput metrics
        record_gauge('performance.requests_per_second', m.requests_per_second)
        record_gauge('performance.tasks_per_minute', m.tasks_completed_per_minute)
        
        # System metrics
        record_gauge('system.cpu_usage_percent', m.cpu_usage_percent)
        record_gauge('system.memory_usage_percent', m.memory_usage_percent)
        record_gauge('system.memory_usage_mb', m.memory_usage_mb)
        
        # Quality metrics
        record_gauge('performance.success_rate', m.success_rate)
        record_gauge('performance.error_rate', m.error_rate)
        
        # Agent metrics
        record_gauge('agents.active', m.active_agents)
        record_gauge('agents.idle', m.idle_agents)
        record_gauge('agents.total', m.total_agents)
        record_gauge('tasks.active', m.active_tasks)
        record_gauge('tasks.queued', m.queued_tasks)
    
    def _check_alerts(self):
        """Check for performance alerts."""
        m = self._current_metrics
        
        alerts = []
        
        # Check response time
        if m.response_time_p95 > self.alert_thresholds.get('response_time_p95', 5.0):
            alerts.append(('warning', 'High Response Time', {
                'current': m.response_time_p95,
                'threshold': self.alert_thresholds['response_time_p95']
            }))
        
        # Check CPU usage
        if m.cpu_usage_percent > self.alert_thresholds.get('cpu_usage_percent', 80.0):
            alerts.append(('warning', 'High CPU Usage', {
                'current': m.cpu_usage_percent,
                'threshold': self.alert_thresholds['cpu_usage_percent']
            }))
        
        # Check memory usage
        if m.memory_usage_percent > self.alert_thresholds.get('memory_usage_percent', 90.0):
            alerts.append(('critical', 'High Memory Usage', {
                'current': m.memory_usage_percent,
                'threshold': self.alert_thresholds['memory_usage_percent']
            }))
        
        # Check error rate
        if m.error_rate > self.alert_thresholds.get('error_rate', 5.0):
            alerts.append(('warning', 'High Error Rate', {
                'current': m.error_rate,
                'threshold': self.alert_thresholds['error_rate']
            }))
        
        # Check success rate
        if m.success_rate < self.alert_thresholds.get('success_rate', 95.0):
            alerts.append(('warning', 'Low Success Rate', {
                'current': m.success_rate,
                'threshold': self.alert_thresholds['success_rate']
            }))
        
        # Notify alert listeners
        for severity, message, data in alerts:
            for listener in self._alert_listeners[:]:  # Copy to avoid modification during iteration
                try:
                    listener(severity, message, data)
                except Exception as e:
                    logger.error(f"Alert listener error: {e}")
    
    def _notify_listeners(self):
        """Notify performance listeners of metric updates."""
        for listener in self._performance_listeners[:]:  # Copy to avoid modification during iteration
            try:
                listener(self._current_metrics)
            except Exception as e:
                logger.error(f"Performance listener error: {e}")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        try:
            return {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'disk_total_gb': psutil.disk_usage('/').total / (1024**3),
                'platform': platform.platform(),
                'python_version': platform.python_version()
            }
        except Exception as e:
            logger.warning(f"Could not get system info: {e}")
            return {}


# Global performance monitor instance
_global_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
        _global_monitor.start()
    return _global_monitor