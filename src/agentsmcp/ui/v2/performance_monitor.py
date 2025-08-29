"""
Performance Monitor for TUI v2 System

Provides real-time performance monitoring and alerting for critical metrics:
- Startup time tracking  
- Typing response latency
- Memory usage monitoring
- Component initialization times
"""

import time
import logging
import asyncio
import sys
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from datetime import datetime, timedelta


logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics."""
    STARTUP_TIME = "startup_time"
    TYPING_LATENCY = "typing_latency"
    MEMORY_USAGE = "memory_usage"
    RENDER_TIME = "render_time"
    INITIALIZATION_TIME = "init_time"


@dataclass
class PerformanceMetric:
    """A performance metric sample."""
    name: str
    value: float
    timestamp: datetime
    metric_type: MetricType
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceThreshold:
    """Performance threshold for alerting."""
    metric_type: MetricType
    warning_threshold: float
    critical_threshold: float
    enabled: bool = True


class PerformanceAlert:
    """Performance alert when thresholds are exceeded."""
    
    def __init__(self, metric: PerformanceMetric, threshold: PerformanceThreshold, 
                 severity: str):
        self.metric = metric
        self.threshold = threshold
        self.severity = severity
        self.timestamp = datetime.now()


class PerformanceMonitor:
    """
    Performance monitoring for TUI v2 system.
    
    Tracks critical performance metrics and provides alerts when
    performance targets are not met.
    """
    
    def __init__(self, max_samples: int = 1000):
        """Initialize performance monitor.
        
        Args:
            max_samples: Maximum number of samples to retain per metric
        """
        self.max_samples = max_samples
        self._metrics: Dict[str, deque] = {}
        self._thresholds: Dict[MetricType, PerformanceThreshold] = {}
        self._alerts: List[PerformanceAlert] = []
        self._alert_handlers: List[Callable[[PerformanceAlert], None]] = []
        self._enabled = True
        
        # Setup default thresholds based on requirements
        self._setup_default_thresholds()
    
    def _setup_default_thresholds(self):
        """Setup default performance thresholds."""
        self._thresholds = {
            MetricType.STARTUP_TIME: PerformanceThreshold(
                metric_type=MetricType.STARTUP_TIME,
                warning_threshold=400.0,  # 400ms warning
                critical_threshold=500.0  # 500ms critical  
            ),
            MetricType.TYPING_LATENCY: PerformanceThreshold(
                metric_type=MetricType.TYPING_LATENCY,
                warning_threshold=12.0,   # 12ms warning
                critical_threshold=16.0   # 16ms critical (60fps)
            ),
            MetricType.MEMORY_USAGE: PerformanceThreshold(
                metric_type=MetricType.MEMORY_USAGE,
                warning_threshold=40.0,   # 40MB warning
                critical_threshold=50.0   # 50MB critical
            ),
            MetricType.RENDER_TIME: PerformanceThreshold(
                metric_type=MetricType.RENDER_TIME,
                warning_threshold=8.0,    # 8ms warning
                critical_threshold=16.0   # 16ms critical
            ),
            MetricType.INITIALIZATION_TIME: PerformanceThreshold(
                metric_type=MetricType.INITIALIZATION_TIME,
                warning_threshold=100.0,  # 100ms warning per component
                critical_threshold=200.0  # 200ms critical per component
            ),
        }
    
    def record_metric(self, name: str, value: float, metric_type: MetricType,
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Record a performance metric.
        
        Args:
            name: Metric name
            value: Metric value 
            metric_type: Type of metric
            metadata: Additional metadata
            
        Returns:
            True if metric was recorded successfully
        """
        if not self._enabled:
            return False
        
        try:
            metric = PerformanceMetric(
                name=name,
                value=value,
                timestamp=datetime.now(),
                metric_type=metric_type,
                metadata=metadata or {}
            )
            
            # Store metric sample
            if name not in self._metrics:
                self._metrics[name] = deque(maxlen=self.max_samples)
            
            self._metrics[name].append(metric)
            
            # Check thresholds and generate alerts
            self._check_thresholds(metric)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to record metric {name}: {e}")
            return False
    
    def _check_thresholds(self, metric: PerformanceMetric):
        """Check if metric exceeds thresholds and generate alerts."""
        threshold = self._thresholds.get(metric.metric_type)
        if not threshold or not threshold.enabled:
            return
        
        severity = None
        if metric.value >= threshold.critical_threshold:
            severity = "CRITICAL"
        elif metric.value >= threshold.warning_threshold:
            severity = "WARNING"
        
        if severity:
            alert = PerformanceAlert(metric, threshold, severity)
            self._alerts.append(alert)
            
            # Trigger alert handlers
            for handler in self._alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Alert handler failed: {e}")
            
            # Log alert
            logger.warning(
                f"PERFORMANCE {severity}: {metric.name} = {metric.value:.1f}ms "
                f"(threshold: {threshold.warning_threshold:.1f}ms warning, "
                f"{threshold.critical_threshold:.1f}ms critical)"
            )
    
    def measure_time(self, name: str, metric_type: MetricType,
                    metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for measuring execution time.
        
        Usage:
            with monitor.measure_time("startup", MetricType.STARTUP_TIME):
                # code to measure
                pass
        """
        return _TimeContextManager(self, name, metric_type, metadata)
    
    async def measure_async_time(self, name: str, metric_type: MetricType,
                                coro, metadata: Optional[Dict[str, Any]] = None):
        """
        Measure execution time of an async coroutine.
        
        Args:
            name: Metric name
            metric_type: Type of metric
            coro: Coroutine to measure
            metadata: Additional metadata
            
        Returns:
            Result of the coroutine
        """
        start_time = time.time()
        try:
            result = await coro
            return result
        finally:
            elapsed = (time.time() - start_time) * 1000  # Convert to milliseconds
            self.record_metric(name, elapsed, metric_type, metadata)
    
    def get_metrics(self, name: Optional[str] = None, 
                   metric_type: Optional[MetricType] = None,
                   since: Optional[datetime] = None) -> List[PerformanceMetric]:
        """
        Get performance metrics with optional filtering.
        
        Args:
            name: Filter by metric name
            metric_type: Filter by metric type
            since: Only return metrics after this timestamp
            
        Returns:
            List of matching metrics
        """
        results = []
        
        for metric_name, samples in self._metrics.items():
            if name and metric_name != name:
                continue
            
            for sample in samples:
                if metric_type and sample.metric_type != metric_type:
                    continue
                
                if since and sample.timestamp < since:
                    continue
                
                results.append(sample)
        
        return sorted(results, key=lambda x: x.timestamp)
    
    def get_recent_alerts(self, since: Optional[datetime] = None,
                         severity: Optional[str] = None) -> List[PerformanceAlert]:
        """Get recent performance alerts with optional filtering."""
        alerts = self._alerts
        
        if since:
            alerts = [a for a in alerts if a.timestamp >= since]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_statistics(self, name: str, window_minutes: int = 60) -> Dict[str, float]:
        """
        Get performance statistics for a metric.
        
        Args:
            name: Metric name
            window_minutes: Time window for statistics
            
        Returns:
            Dictionary with min, max, avg, p95, p99 values
        """
        if name not in self._metrics:
            return {}
        
        # Filter to time window
        since = datetime.now() - timedelta(minutes=window_minutes)
        samples = [s for s in self._metrics[name] if s.timestamp >= since]
        
        if not samples:
            return {}
        
        values = [s.value for s in samples]
        values.sort()
        
        n = len(values)
        return {
            'count': n,
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / n,
            'p50': values[n // 2],
            'p95': values[int(n * 0.95)] if n >= 20 else values[-1],
            'p99': values[int(n * 0.99)] if n >= 100 else values[-1],
        }
    
    def add_alert_handler(self, handler: Callable[[PerformanceAlert], None]):
        """Add a custom alert handler."""
        self._alert_handlers.append(handler)
    
    def set_threshold(self, metric_type: MetricType, warning: float, 
                     critical: float, enabled: bool = True):
        """Set custom threshold for a metric type."""
        self._thresholds[metric_type] = PerformanceThreshold(
            metric_type=metric_type,
            warning_threshold=warning,
            critical_threshold=critical,
            enabled=enabled
        )
    
    def enable(self):
        """Enable performance monitoring."""
        self._enabled = True
    
    def disable(self):
        """Disable performance monitoring."""
        self._enabled = False
    
    def clear_metrics(self, name: Optional[str] = None):
        """Clear stored metrics."""
        if name:
            if name in self._metrics:
                self._metrics[name].clear()
        else:
            self._metrics.clear()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system performance information."""
        try:
            import psutil
            process = psutil.Process()
            
            return {
                'memory_mb': process.memory_info().rss / 1024 / 1024,
                'cpu_percent': process.cpu_percent(),
                'threads': process.num_threads(),
                'open_files': len(process.open_files()),
            }
        except ImportError:
            # Fallback without psutil
            return {
                'memory_mb': self._estimate_memory_usage(),
                'threads': threading.active_count() if hasattr(sys, 'threading') else 1,
            }
    
    def _estimate_memory_usage(self) -> float:
        """Rough estimate of memory usage without psutil."""
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except:
            return 0.0


class _TimeContextManager:
    """Context manager for measuring execution time."""
    
    def __init__(self, monitor: PerformanceMonitor, name: str, 
                 metric_type: MetricType, metadata: Optional[Dict[str, Any]]):
        self.monitor = monitor
        self.name = name
        self.metric_type = metric_type
        self.metadata = metadata
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            elapsed = (time.time() - self.start_time) * 1000  # Convert to milliseconds
            self.monitor.record_metric(self.name, elapsed, self.metric_type, self.metadata)


# Global instance for convenience
_global_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get or create the global performance monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
        
        # Setup default alert handler to log critical performance issues
        def log_critical_alerts(alert: PerformanceAlert):
            if alert.severity == "CRITICAL":
                logger.critical(
                    f"CRITICAL PERFORMANCE ISSUE: {alert.metric.name} = {alert.metric.value:.1f} "
                    f"exceeds {alert.threshold.critical_threshold:.1f} threshold"
                )
        
        _global_monitor.add_alert_handler(log_critical_alerts)
    
    return _global_monitor


# Convenience functions
def record_metric(name: str, value: float, metric_type: MetricType,
                 metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Record a performance metric using the global monitor."""
    return get_performance_monitor().record_metric(name, value, metric_type, metadata)


def measure_time(name: str, metric_type: MetricType,
                metadata: Optional[Dict[str, Any]] = None):
    """Context manager for measuring execution time using global monitor."""
    return get_performance_monitor().measure_time(name, metric_type, metadata)