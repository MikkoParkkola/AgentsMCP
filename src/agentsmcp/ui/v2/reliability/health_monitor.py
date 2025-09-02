"""
Health Monitor - Monitor TUI health and detect hang conditions.

This module provides continuous health monitoring for the TUI to detect hang conditions
and performance issues. It monitors response times, memory usage, and UI responsiveness
to trigger recovery actions when problems are detected.

Key Features:
- Monitor TUI health every 1 second to detect hang conditions
- Track performance metrics (response times, memory usage, FPS)
- Detect when UI becomes unresponsive (>5s without updates)
- Trigger recovery actions when hangs are detected
- Integrate with recovery systems for automatic fixes

ICD Compliance:
- Inputs: health_check_interval, hang_threshold_seconds, performance_metrics
- Outputs: health_status, hang_detected_event, performance_report
- Performance: Health monitoring must detect hangs within 5s threshold
- Error Handling: Failed monitoring should not disrupt TUI operation
"""

import asyncio
import logging
import psutil
import threading
import time
import weakref
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Callable, Union
import traceback
import sys

from .timeout_guardian import TimeoutGuardian, get_global_guardian

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Overall health status of the TUI."""
    HEALTHY = "healthy"         # All systems operating normally
    DEGRADED = "degraded"       # Some performance issues but functional
    UNHEALTHY = "unhealthy"     # Significant issues affecting functionality
    HANGING = "hanging"         # System appears to be hung
    UNKNOWN = "unknown"         # Unable to determine status


class MetricType(Enum):
    """Types of metrics being monitored."""
    RESPONSE_TIME = "response_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    FPS = "fps"
    UPDATE_FREQUENCY = "update_frequency"
    EVENT_PROCESSING = "event_processing"
    RENDER_TIME = "render_time"


class AlertLevel(Enum):
    """Alert levels for health issues."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class HealthMetric:
    """Individual health metric data point."""
    metric_type: MetricType
    value: float
    unit: str
    timestamp: datetime
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    
    @property
    def alert_level(self) -> AlertLevel:
        """Determine alert level based on thresholds."""
        if self.threshold_critical and self.value >= self.threshold_critical:
            return AlertLevel.CRITICAL
        elif self.threshold_warning and self.value >= self.threshold_warning:
            return AlertLevel.WARNING
        else:
            return AlertLevel.INFO


@dataclass
class PerformanceReport:
    """Comprehensive performance report."""
    timestamp: datetime
    overall_status: HealthStatus
    metrics: Dict[MetricType, HealthMetric]
    alerts: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    uptime_seconds: float = 0.0
    last_response_time: Optional[float] = None
    hang_detected: bool = False
    recovery_suggested: bool = False


@dataclass
class HangDetectionConfig:
    """Configuration for hang detection."""
    response_timeout_seconds: float = 5.0    # Max time without UI response
    update_timeout_seconds: float = 5.0      # Max time without UI updates
    event_timeout_seconds: float = 10.0      # Max time without event processing
    memory_threshold_mb: float = 1000.0      # Memory usage threshold
    cpu_threshold_percent: float = 90.0      # CPU usage threshold
    fps_minimum: float = 5.0                 # Minimum acceptable FPS


class HealthMonitor:
    """
    Monitor TUI health and detect hang conditions.
    
    Provides continuous monitoring of TUI performance and responsiveness
    to detect and react to hang conditions before they become critical.
    """
    
    def __init__(
        self,
        check_interval_seconds: float = 1.0,
        hang_config: Optional[HangDetectionConfig] = None,
        timeout_guardian: Optional[TimeoutGuardian] = None
    ):
        """Initialize the health monitor."""
        self._check_interval = check_interval_seconds
        self._hang_config = hang_config or HangDetectionConfig()
        self._guardian = timeout_guardian or get_global_guardian()
        
        # Monitoring state
        self._monitoring_active = False
        self._monitoring_task: Optional[asyncio.Task] = None
        self._start_time = datetime.now()
        self._last_response_time: Optional[datetime] = None
        self._last_update_time: Optional[datetime] = None
        self._last_event_time: Optional[datetime] = None
        
        # Metrics tracking
        self._metrics_history: Dict[MetricType, List[HealthMetric]] = {}
        self._current_metrics: Dict[MetricType, HealthMetric] = {}
        self._performance_samples = 100  # Keep last N samples
        
        # Health callbacks
        self._health_callbacks: Set[Callable[[PerformanceReport], None]] = set()
        self._hang_callbacks: Set[Callable[[str], None]] = set()
        
        # Process monitoring
        self._process = psutil.Process()
        self._baseline_memory = self._process.memory_info().rss / 1024 / 1024  # MB
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        logger.info(f"Health monitor initialized with {self._check_interval}s interval")
        
    async def start_monitoring(self) -> None:
        """Start health monitoring."""
        if self._monitoring_active:
            logger.warning("Health monitoring is already active")
            return
            
        logger.info("Starting TUI health monitoring")
        self._monitoring_active = True
        self._start_time = datetime.now()
        
        # Start monitoring task
        self._monitoring_task = asyncio.create_task(
            self._monitoring_loop(),
            name="health_monitor"
        )
        
    async def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        if not self._monitoring_active:
            return
            
        logger.info("Stopping TUI health monitoring")
        self._monitoring_active = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
            
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        logger.info("Health monitoring loop started")
        
        try:
            while self._monitoring_active:
                try:
                    # Use timeout guardian to protect health check
                    async with self._guardian.protect_operation("health_check", 2.0):
                        await self._perform_health_check()
                        
                except asyncio.TimeoutError:
                    logger.warning("Health check timed out - potential hang detected")
                    await self._handle_potential_hang("health_check_timeout")
                    
                except Exception as e:
                    logger.error(f"Health check failed: {e}")
                    
                # Wait for next check interval
                await asyncio.sleep(self._check_interval)
                
        except asyncio.CancelledError:
            logger.info("Health monitoring loop cancelled")
            
        except Exception as e:
            logger.error(f"Health monitoring loop failed: {e}")
            
    async def _perform_health_check(self) -> PerformanceReport:
        """Perform comprehensive health check."""
        check_start = time.time()
        current_time = datetime.now()
        
        # Collect system metrics
        memory_usage = self._process.memory_info().rss / 1024 / 1024  # MB
        cpu_percent = self._process.cpu_percent()
        
        # Create metrics
        metrics = {}
        
        # Memory metric
        memory_metric = HealthMetric(
            metric_type=MetricType.MEMORY_USAGE,
            value=memory_usage,
            unit="MB",
            timestamp=current_time,
            threshold_warning=self._hang_config.memory_threshold_mb * 0.8,
            threshold_critical=self._hang_config.memory_threshold_mb
        )
        metrics[MetricType.MEMORY_USAGE] = memory_metric
        
        # CPU metric
        cpu_metric = HealthMetric(
            metric_type=MetricType.CPU_USAGE,
            value=cpu_percent,
            unit="%",
            timestamp=current_time,
            threshold_warning=self._hang_config.cpu_threshold_percent * 0.8,
            threshold_critical=self._hang_config.cpu_threshold_percent
        )
        metrics[MetricType.CPU_USAGE] = cpu_metric
        
        # Response time metric
        response_time = time.time() - check_start
        response_metric = HealthMetric(
            metric_type=MetricType.RESPONSE_TIME,
            value=response_time * 1000,  # Convert to milliseconds
            unit="ms",
            timestamp=current_time,
            threshold_warning=100.0,  # 100ms warning
            threshold_critical=500.0  # 500ms critical
        )
        metrics[MetricType.RESPONSE_TIME] = response_metric
        
        # Update metrics history
        async with self._lock:
            for metric_type, metric in metrics.items():
                if metric_type not in self._metrics_history:
                    self._metrics_history[metric_type] = []
                    
                # Add new metric and trim history
                history = self._metrics_history[metric_type]
                history.append(metric)
                if len(history) > self._performance_samples:
                    history.pop(0)
                    
                self._current_metrics[metric_type] = metric
                
        # Detect hang conditions
        hang_detected = await self._detect_hang_conditions(current_time)
        
        # Determine overall health status
        overall_status = self._calculate_overall_status(metrics, hang_detected)
        
        # Generate performance report
        uptime = (current_time - self._start_time).total_seconds()
        report = PerformanceReport(
            timestamp=current_time,
            overall_status=overall_status,
            metrics=metrics,
            uptime_seconds=uptime,
            last_response_time=response_time,
            hang_detected=hang_detected,
            recovery_suggested=hang_detected or overall_status == HealthStatus.UNHEALTHY
        )
        
        # Add alerts and recommendations
        await self._generate_alerts_and_recommendations(report)
        
        # Notify callbacks
        await self._notify_health_callbacks(report)
        
        # Handle hang detection
        if hang_detected:
            await self._handle_hang_detection(report)
            
        return report
        
    async def _detect_hang_conditions(self, current_time: datetime) -> bool:
        """Detect if the TUI appears to be hanging."""
        hang_detected = False
        
        # Check response time hang
        if self._last_response_time:
            response_delta = (current_time - self._last_response_time).total_seconds()
            if response_delta > self._hang_config.response_timeout_seconds:
                logger.warning(f"Response timeout detected: {response_delta:.1f}s since last response")
                hang_detected = True
                
        # Check update time hang
        if self._last_update_time:
            update_delta = (current_time - self._last_update_time).total_seconds()
            if update_delta > self._hang_config.update_timeout_seconds:
                logger.warning(f"Update timeout detected: {update_delta:.1f}s since last update")
                hang_detected = True
                
        # Check event processing hang
        if self._last_event_time:
            event_delta = (current_time - self._last_event_time).total_seconds()
            if event_delta > self._hang_config.event_timeout_seconds:
                logger.warning(f"Event timeout detected: {event_delta:.1f}s since last event")
                hang_detected = True
                
        return hang_detected
        
    def _calculate_overall_status(
        self,
        metrics: Dict[MetricType, HealthMetric],
        hang_detected: bool
    ) -> HealthStatus:
        """Calculate overall health status from metrics."""
        if hang_detected:
            return HealthStatus.HANGING
            
        critical_alerts = 0
        warning_alerts = 0
        
        for metric in metrics.values():
            if metric.alert_level == AlertLevel.CRITICAL:
                critical_alerts += 1
            elif metric.alert_level == AlertLevel.WARNING:
                warning_alerts += 1
                
        if critical_alerts > 0:
            return HealthStatus.UNHEALTHY
        elif warning_alerts > 1:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
            
    async def _generate_alerts_and_recommendations(self, report: PerformanceReport) -> None:
        """Generate alerts and recommendations based on report."""
        alerts = []
        recommendations = []
        
        # Memory alerts
        memory_metric = report.metrics.get(MetricType.MEMORY_USAGE)
        if memory_metric and memory_metric.alert_level == AlertLevel.CRITICAL:
            alerts.append(f"Memory usage critical: {memory_metric.value:.1f}MB")
            recommendations.append("Consider restarting TUI or reducing memory usage")
            
        # CPU alerts
        cpu_metric = report.metrics.get(MetricType.CPU_USAGE)
        if cpu_metric and cpu_metric.alert_level == AlertLevel.CRITICAL:
            alerts.append(f"CPU usage critical: {cpu_metric.value:.1f}%")
            recommendations.append("Reduce TUI update frequency or complexity")
            
        # Response time alerts
        response_metric = report.metrics.get(MetricType.RESPONSE_TIME)
        if response_metric and response_metric.alert_level == AlertLevel.CRITICAL:
            alerts.append(f"Response time critical: {response_metric.value:.1f}ms")
            recommendations.append("Check for blocking operations in TUI")
            
        # Hang detection alerts
        if report.hang_detected:
            alerts.append("TUI hang condition detected")
            recommendations.append("Consider triggering recovery procedures")
            
        report.alerts = alerts
        report.recommendations = recommendations
        
    async def _handle_hang_detection(self, report: PerformanceReport) -> None:
        """Handle detected hang condition."""
        logger.critical("TUI hang detected - triggering recovery procedures")
        
        # Notify hang callbacks
        for callback in self._hang_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback("hang_detected")
                else:
                    callback("hang_detected")
            except Exception as e:
                logger.error(f"Hang callback failed: {e}")
                
    async def _handle_potential_hang(self, reason: str) -> None:
        """Handle potential hang condition."""
        logger.warning(f"Potential hang detected: {reason}")
        
        for callback in self._hang_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(reason)
                else:
                    callback(reason)
            except Exception as e:
                logger.error(f"Potential hang callback failed: {e}")
                
    async def _notify_health_callbacks(self, report: PerformanceReport) -> None:
        """Notify health status callbacks."""
        for callback in self._health_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(report)
                else:
                    callback(report)
            except Exception as e:
                logger.error(f"Health callback failed: {e}")
                
    def add_health_callback(self, callback: Callable[[PerformanceReport], None]) -> None:
        """Add callback for health status updates."""
        self._health_callbacks.add(callback)
        
    def remove_health_callback(self, callback: Callable[[PerformanceReport], None]) -> None:
        """Remove health status callback."""
        self._health_callbacks.discard(callback)
        
    def add_hang_callback(self, callback: Callable[[str], None]) -> None:
        """Add callback for hang detection."""
        self._hang_callbacks.add(callback)
        
    def remove_hang_callback(self, callback: Callable[[str], None]) -> None:
        """Remove hang detection callback."""
        self._hang_callbacks.discard(callback)
        
    async def record_ui_response(self) -> None:
        """Record that UI responded (call from UI update code)."""
        async with self._lock:
            self._last_response_time = datetime.now()
            
    async def record_ui_update(self) -> None:
        """Record that UI was updated (call from display update code)."""
        async with self._lock:
            self._last_update_time = datetime.now()
            
    async def record_event_processed(self) -> None:
        """Record that an event was processed (call from event handling code)."""
        async with self._lock:
            self._last_event_time = datetime.now()
            
    async def get_current_health_status(self) -> HealthStatus:
        """Get current health status."""
        if not self._current_metrics:
            return HealthStatus.UNKNOWN
            
        # Quick health check without full monitoring
        memory_usage = self._process.memory_info().rss / 1024 / 1024  # MB
        if memory_usage > self._hang_config.memory_threshold_mb:
            return HealthStatus.UNHEALTHY
            
        cpu_percent = self._process.cpu_percent()
        if cpu_percent > self._hang_config.cpu_threshold_percent:
            return HealthStatus.UNHEALTHY
            
        return HealthStatus.HEALTHY
        
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        async with self._lock:
            current_metrics = self._current_metrics.copy()
            
        summary = {
            'status': await self.get_current_health_status(),
            'uptime_seconds': (datetime.now() - self._start_time).total_seconds(),
            'monitoring_active': self._monitoring_active,
            'metrics': {}
        }
        
        for metric_type, metric in current_metrics.items():
            summary['metrics'][metric_type.value] = {
                'value': metric.value,
                'unit': metric.unit,
                'alert_level': metric.alert_level.value,
                'timestamp': metric.timestamp.isoformat()
            }
            
        return summary


# Global health monitor instance
_global_monitor: Optional[HealthMonitor] = None


def get_global_health_monitor() -> HealthMonitor:
    """Get the global health monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = HealthMonitor()
    return _global_monitor


async def start_tui_health_monitoring(
    check_interval_seconds: float = 1.0,
    hang_config: Optional[HangDetectionConfig] = None
) -> HealthMonitor:
    """
    Convenience function to start TUI health monitoring.
    
    Args:
        check_interval_seconds: Health check interval (default 1s)
        hang_config: Hang detection configuration
        
    Returns:
        HealthMonitor instance
    """
    monitor = get_global_health_monitor()
    if hang_config:
        monitor._hang_config = hang_config
    if check_interval_seconds != monitor._check_interval:
        monitor._check_interval = check_interval_seconds
        
    await monitor.start_monitoring()
    return monitor