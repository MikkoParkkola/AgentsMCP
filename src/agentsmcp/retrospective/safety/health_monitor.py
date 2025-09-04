"""Health monitoring for retrospective system improvements."""

from __future__ import annotations

import asyncio
import logging
import os
import psutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Awaitable
import statistics

from .safety_config import SafetyConfig, RollbackTrigger

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Overall health status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class MetricTrend(str, Enum):
    """Trend direction for metrics."""
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    UNKNOWN = "unknown"


@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    status: HealthStatus = HealthStatus.HEALTHY
    trend: MetricTrend = MetricTrend.UNKNOWN
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthMetrics:
    """Complete set of system health metrics."""
    timestamp: datetime
    metrics: Dict[str, HealthMetric]
    overall_status: HealthStatus
    summary: str = ""
    
    @classmethod
    def create_empty(cls) -> HealthMetrics:
        """Create empty health metrics."""
        return cls(
            timestamp=datetime.now(timezone.utc),
            metrics={},
            overall_status=HealthStatus.UNKNOWN
        )


@dataclass
class HealthBaseline:
    """Baseline health metrics for comparison."""
    baseline_id: str
    created_at: datetime
    duration_seconds: int
    metrics: Dict[str, Dict[str, float]]  # metric_name -> {mean, median, p95, p99, std_dev}
    sample_count: int
    
    def compare_metric(self, metric: HealthMetric, threshold_percent: float) -> bool:
        """Compare a metric against baseline with threshold."""
        if metric.name not in self.metrics:
            return True  # No baseline to compare against
        
        baseline_stats = self.metrics[metric.name]
        baseline_value = baseline_stats.get('mean', 0.0)
        
        if baseline_value == 0.0:
            return True  # Cannot compare against zero baseline
        
        percentage_change = ((metric.value - baseline_value) / baseline_value) * 100
        return abs(percentage_change) <= threshold_percent


class HealthMonitor:
    """System health monitoring for safety framework."""
    
    def __init__(self, config: SafetyConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Current health state
        self._current_metrics: Optional[HealthMetrics] = None
        self._baseline: Optional[HealthBaseline] = None
        self._is_monitoring = False
        
        # Monitoring task
        self._monitoring_task: Optional[asyncio.Task] = None
        self._stop_monitoring = asyncio.Event()
        
        # Health history for trend analysis
        self._health_history: List[HealthMetrics] = []
        self._max_history_size = 1000
        
        # Custom metric collectors
        self._custom_collectors: List[Callable[[], Awaitable[Dict[str, HealthMetric]]]] = []
        
        # Health change callbacks
        self._health_change_callbacks: List[Callable[[HealthMetrics, HealthMetrics], Awaitable[None]]] = []
    
    async def start_monitoring(self):
        """Start continuous health monitoring."""
        if self._is_monitoring:
            self.logger.warning("Health monitoring already started")
            return
        
        self._is_monitoring = True
        self._stop_monitoring.clear()
        
        # Start monitoring task
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        self.logger.info("Health monitoring started")
    
    async def stop_monitoring(self):
        """Stop health monitoring."""
        if not self._is_monitoring:
            return
        
        self._is_monitoring = False
        self._stop_monitoring.set()
        
        if self._monitoring_task:
            try:
                await asyncio.wait_for(self._monitoring_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass
        
        self.logger.info("Health monitoring stopped")
    
    async def collect_baseline(self, duration_seconds: Optional[int] = None) -> HealthBaseline:
        """Collect baseline health metrics over a period."""
        duration = duration_seconds or self.config.baseline_collection_duration_seconds
        baseline_id = f"baseline_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Collecting baseline metrics for {duration} seconds")
        
        # Collect samples
        samples: Dict[str, List[float]] = {}
        sample_count = 0
        
        start_time = time.time()
        end_time = start_time + duration
        
        while time.time() < end_time:
            try:
                current_metrics = await self.collect_current_metrics()
                
                for metric_name, metric in current_metrics.metrics.items():
                    if metric_name not in samples:
                        samples[metric_name] = []
                    samples[metric_name].append(metric.value)
                
                sample_count += 1
                
                # Wait for next sample
                await asyncio.sleep(min(5.0, self.config.health_check_interval_seconds))
                
            except Exception as e:
                self.logger.error(f"Error collecting baseline sample: {e}")
        
        # Calculate statistics
        baseline_metrics = {}
        for metric_name, values in samples.items():
            if values:
                baseline_metrics[metric_name] = {
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'p95': self._percentile(values, 95),
                    'p99': self._percentile(values, 99),
                    'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0,
                    'min': min(values),
                    'max': max(values)
                }
        
        baseline = HealthBaseline(
            baseline_id=baseline_id,
            created_at=datetime.now(timezone.utc),
            duration_seconds=duration,
            metrics=baseline_metrics,
            sample_count=sample_count
        )
        
        self._baseline = baseline
        self.logger.info(f"Baseline collected: {sample_count} samples, {len(baseline_metrics)} metrics")
        
        return baseline
    
    async def collect_current_metrics(self) -> HealthMetrics:
        """Collect current system health metrics."""
        try:
            metrics = {}
            timestamp = datetime.now(timezone.utc)
            
            # System metrics
            system_metrics = await self._collect_system_metrics()
            metrics.update(system_metrics)
            
            # Application metrics
            app_metrics = await self._collect_application_metrics()
            metrics.update(app_metrics)
            
            # Custom metrics
            for collector in self._custom_collectors:
                try:
                    custom_metrics = await collector()
                    metrics.update(custom_metrics)
                except Exception as e:
                    self.logger.error(f"Custom metric collector failed: {e}")
            
            # Determine overall status
            overall_status = self._determine_overall_status(metrics)
            
            current_metrics = HealthMetrics(
                timestamp=timestamp,
                metrics=metrics,
                overall_status=overall_status,
                summary=self._generate_health_summary(metrics, overall_status)
            )
            
            # Update current metrics
            previous_metrics = self._current_metrics
            self._current_metrics = current_metrics
            
            # Add to history
            self._add_to_history(current_metrics)
            
            # Notify callbacks of changes
            if previous_metrics:
                await self._notify_health_change(previous_metrics, current_metrics)
            
            return current_metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect health metrics: {e}")
            return HealthMetrics.create_empty()
    
    async def check_health_degradation(
        self, 
        current_metrics: Optional[HealthMetrics] = None
    ) -> List[RollbackTrigger]:
        """Check for health degradation that might trigger rollback."""
        if not current_metrics:
            current_metrics = await self.collect_current_metrics()
        
        if not self._baseline:
            self.logger.warning("No baseline available for health comparison")
            return []
        
        triggers = []
        
        # Check response time degradation
        if 'response_time_ms' in current_metrics.metrics:
            response_time = current_metrics.metrics['response_time_ms']
            if not self._baseline.compare_metric(
                response_time, 
                self.config.thresholds.max_response_time_increase_percent
            ):
                triggers.append(RollbackTrigger.RESPONSE_TIME_DEGRADATION)
        
        # Check error rate spike
        if 'error_rate' in current_metrics.metrics:
            error_rate = current_metrics.metrics['error_rate']
            if not self._baseline.compare_metric(
                error_rate,
                self.config.thresholds.max_error_rate_increase_percent
            ):
                triggers.append(RollbackTrigger.ERROR_RATE_SPIKE)
        
        # Check memory leak
        if 'memory_usage_percent' in current_metrics.metrics:
            memory_usage = current_metrics.metrics['memory_usage_percent']
            if not self._baseline.compare_metric(
                memory_usage,
                self.config.thresholds.max_memory_increase_percent
            ):
                triggers.append(RollbackTrigger.MEMORY_LEAK)
        
        # Check critical role failures
        critical_role_failures = any(
            metric.name.startswith('role_') and 
            metric.status in [HealthStatus.CRITICAL, HealthStatus.WARNING]
            for metric in current_metrics.metrics.values()
        )
        
        if critical_role_failures:
            triggers.append(RollbackTrigger.CRITICAL_ROLE_FAILURE)
        
        # Check health check failures
        if current_metrics.overall_status == HealthStatus.CRITICAL:
            triggers.append(RollbackTrigger.HEALTH_CHECK_FAILURE)
        
        if triggers:
            self.logger.warning(f"Health degradation detected: {triggers}")
        
        return triggers
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        try:
            while not self._stop_monitoring.is_set():
                try:
                    await self.collect_current_metrics()
                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {e}")
                
                try:
                    await asyncio.wait_for(
                        self._stop_monitoring.wait(),
                        timeout=self.config.health_check_interval_seconds
                    )
                    break  # Stop monitoring was set
                except asyncio.TimeoutError:
                    continue  # Continue monitoring
                    
        except asyncio.CancelledError:
            self.logger.info("Monitoring loop cancelled")
        except Exception as e:
            self.logger.error(f"Monitoring loop failed: {e}")
    
    async def _collect_system_metrics(self) -> Dict[str, HealthMetric]:
        """Collect system-level health metrics."""
        metrics = {}
        timestamp = datetime.now(timezone.utc)
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1.0)
            metrics['cpu_usage_percent'] = HealthMetric(
                name='cpu_usage_percent',
                value=cpu_percent,
                unit='percent',
                timestamp=timestamp,
                status=HealthStatus.CRITICAL if cpu_percent > 90 else 
                       HealthStatus.WARNING if cpu_percent > 75 else HealthStatus.HEALTHY
            )
            
            # Memory usage
            memory = psutil.virtual_memory()
            metrics['memory_usage_percent'] = HealthMetric(
                name='memory_usage_percent',
                value=memory.percent,
                unit='percent',
                timestamp=timestamp,
                status=HealthStatus.CRITICAL if memory.percent > 95 else
                       HealthStatus.WARNING if memory.percent > 85 else HealthStatus.HEALTHY
            )
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            metrics['disk_usage_percent'] = HealthMetric(
                name='disk_usage_percent',
                value=disk_percent,
                unit='percent',
                timestamp=timestamp,
                status=HealthStatus.CRITICAL if disk_percent > 95 else
                       HealthStatus.WARNING if disk_percent > 85 else HealthStatus.HEALTHY
            )
            
            # Load average (Unix systems)
            try:
                load_avg = os.getloadavg()[0]  # 1-minute load average
                cpu_count = psutil.cpu_count()
                load_percent = (load_avg / cpu_count) * 100 if cpu_count else 0
                
                metrics['load_average_percent'] = HealthMetric(
                    name='load_average_percent',
                    value=load_percent,
                    unit='percent',
                    timestamp=timestamp,
                    status=HealthStatus.CRITICAL if load_percent > 200 else
                           HealthStatus.WARNING if load_percent > 150 else HealthStatus.HEALTHY
                )
            except (OSError, AttributeError):
                # getloadavg() not available on all platforms
                pass
        
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
        
        return metrics
    
    async def _collect_application_metrics(self) -> Dict[str, HealthMetric]:
        """Collect application-specific health metrics."""
        metrics = {}
        timestamp = datetime.now(timezone.utc)
        
        try:
            # Process metrics
            process = psutil.Process()
            
            # Memory usage by current process
            memory_info = process.memory_info()
            metrics['process_memory_mb'] = HealthMetric(
                name='process_memory_mb',
                value=memory_info.rss / (1024 * 1024),  # Convert to MB
                unit='MB',
                timestamp=timestamp
            )
            
            # File descriptors (Unix systems)
            try:
                num_fds = process.num_fds()
                metrics['open_file_descriptors'] = HealthMetric(
                    name='open_file_descriptors',
                    value=num_fds,
                    unit='count',
                    timestamp=timestamp,
                    status=HealthStatus.WARNING if num_fds > 1000 else HealthStatus.HEALTHY
                )
            except (AttributeError, psutil.AccessDenied):
                pass
            
            # Thread count
            num_threads = process.num_threads()
            metrics['thread_count'] = HealthMetric(
                name='thread_count',
                value=num_threads,
                unit='count',
                timestamp=timestamp,
                status=HealthStatus.WARNING if num_threads > 100 else HealthStatus.HEALTHY
            )
            
            # Mock application metrics (in real implementation, these would come from the actual application)
            # Response time simulation
            metrics['response_time_ms'] = HealthMetric(
                name='response_time_ms',
                value=50.0,  # Mock value
                unit='ms',
                timestamp=timestamp
            )
            
            # Error rate simulation
            metrics['error_rate'] = HealthMetric(
                name='error_rate',
                value=0.1,  # Mock value (0.1%)
                unit='percent',
                timestamp=timestamp
            )
        
        except Exception as e:
            self.logger.error(f"Failed to collect application metrics: {e}")
        
        return metrics
    
    def _determine_overall_status(self, metrics: Dict[str, HealthMetric]) -> HealthStatus:
        """Determine overall health status from individual metrics."""
        if not metrics:
            return HealthStatus.UNKNOWN
        
        statuses = [metric.status for metric in metrics.values()]
        
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        elif HealthStatus.HEALTHY in statuses:
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    def _generate_health_summary(self, metrics: Dict[str, HealthMetric], overall_status: HealthStatus) -> str:
        """Generate human-readable health summary."""
        total_metrics = len(metrics)
        healthy_count = sum(1 for m in metrics.values() if m.status == HealthStatus.HEALTHY)
        warning_count = sum(1 for m in metrics.values() if m.status == HealthStatus.WARNING)
        critical_count = sum(1 for m in metrics.values() if m.status == HealthStatus.CRITICAL)
        
        return f"Overall: {overall_status.value} | Metrics: {total_metrics} total, {healthy_count} healthy, {warning_count} warnings, {critical_count} critical"
    
    def _add_to_history(self, metrics: HealthMetrics):
        """Add metrics to history and maintain size limit."""
        self._health_history.append(metrics)
        
        if len(self._health_history) > self._max_history_size:
            self._health_history = self._health_history[-self._max_history_size:]
    
    async def _notify_health_change(self, previous: HealthMetrics, current: HealthMetrics):
        """Notify registered callbacks of health changes."""
        for callback in self._health_change_callbacks:
            try:
                await callback(previous, current)
            except Exception as e:
                self.logger.error(f"Health change callback failed: {e}")
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = (percentile / 100.0) * (len(sorted_values) - 1)
        
        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower = sorted_values[int(index)]
            upper = sorted_values[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def add_custom_collector(self, collector: Callable[[], Awaitable[Dict[str, HealthMetric]]]):
        """Add a custom metric collector."""
        self._custom_collectors.append(collector)
        self.logger.info("Added custom health metric collector")
    
    def add_health_change_callback(self, callback: Callable[[HealthMetrics, HealthMetrics], Awaitable[None]]):
        """Add callback for health changes."""
        self._health_change_callbacks.append(callback)
        self.logger.info("Added health change callback")
    
    @property
    def current_metrics(self) -> Optional[HealthMetrics]:
        """Get current health metrics."""
        return self._current_metrics
    
    @property
    def baseline(self) -> Optional[HealthBaseline]:
        """Get current baseline."""
        return self._baseline
    
    @property
    def is_monitoring(self) -> bool:
        """Check if monitoring is active."""
        return self._is_monitoring