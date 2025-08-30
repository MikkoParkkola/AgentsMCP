"""
Performance Monitoring API for AgentsMCP

Provides comprehensive performance monitoring, profiling, optimization,
and capacity planning for all backend services with real-time metrics
and automated performance tuning.
"""

import asyncio
import logging
import psutil
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from .base import APIBase, APIResponse, APIError


class MetricAggregation(Enum):
    """Metric aggregation methods"""
    AVERAGE = "average"
    MAXIMUM = "maximum"
    MINIMUM = "minimum"
    SUM = "sum"
    PERCENTILE_95 = "p95"
    PERCENTILE_99 = "p99"


class PerformanceIssueType(Enum):
    """Types of performance issues"""
    HIGH_LATENCY = "high_latency"
    HIGH_CPU = "high_cpu"
    HIGH_MEMORY = "high_memory"
    LOW_THROUGHPUT = "low_throughput"
    HIGH_ERROR_RATE = "high_error_rate"
    RESOURCE_LEAK = "resource_leak"


@dataclass
class PerformanceMetric:
    """Represents a performance metric"""
    name: str
    value: float
    timestamp: datetime
    unit: str
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class PerformanceAlert:
    """Performance alert"""
    id: str
    issue_type: PerformanceIssueType
    service: str
    metric_name: str
    current_value: float
    threshold_value: float
    severity: str
    message: str
    timestamp: datetime
    resolved: bool = False


@dataclass
class SystemResources:
    """Current system resource utilization"""
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    network_io_bytes_sent: int
    network_io_bytes_recv: int
    active_connections: int
    process_count: int
    timestamp: datetime


class PerformanceMonitoring(APIBase):
    """
    Comprehensive performance monitoring system for AgentsMCP backend services.
    
    Features:
    - Real-time performance metrics collection
    - System resource monitoring (CPU, memory, disk, network)
    - Application performance profiling
    - Latency and throughput tracking
    - Automated performance alerting
    - Capacity planning and forecasting
    - Performance regression detection
    - Optimization recommendations
    """
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Metrics storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.metric_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.request_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.throughput_counters: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # System monitoring
        self.system_history: deque = deque(maxlen=1440)  # 24 hours at 1-minute intervals
        self.process_stats: Dict[str, Dict[str, Any]] = {}
        
        # Performance alerts
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.alert_thresholds: Dict[str, Dict[str, float]] = {}
        
        # Performance baselines and forecasting
        self.baselines: Dict[str, Dict[str, float]] = {}
        self.trend_data: Dict[str, List[float]] = defaultdict(list)
        
        # Optimization tracking
        self.optimization_suggestions: List[Dict[str, Any]] = []
        self.applied_optimizations: List[Dict[str, Any]] = []
        
        # Background tasks
        self._performance_tasks: Set[asyncio.Task] = set()
        self._initialize_default_thresholds()
        
    async def initialize(self) -> APIResponse:
        """Initialize the performance monitoring system"""
        try:
            # Start background monitoring tasks
            tasks = [
                asyncio.create_task(self._system_monitoring_loop()),
                asyncio.create_task(self._performance_analysis_loop()),
                asyncio.create_task(self._alert_evaluation_loop()),
                asyncio.create_task(self._optimization_loop()),
                asyncio.create_task(self._cleanup_loop())
            ]
            
            self._performance_tasks.update(tasks)
            
            await self._record_metric("performance.monitoring_initialized", 1, "count")
            self.logger.info("Performance monitoring system initialized")
            
            return APIResponse(
                success=True,
                data={"status": "initialized", "active_tasks": len(self._performance_tasks)},
                message="Performance monitoring system initialized successfully"
            )
            
        except Exception as e:
            return APIResponse(
                success=False,
                error=APIError("PERFORMANCE_INIT_ERROR", f"Failed to initialize performance monitoring: {str(e)}")
            )
    
    async def record_request_timing(self, service: str, endpoint: str, duration_ms: float, 
                                  status_code: int = 200) -> APIResponse:
        """Record request timing metrics"""
        try:
            now = datetime.utcnow()
            
            # Record timing
            metric_name = f"request.duration_ms.{service}.{endpoint}"
            await self._record_metric(metric_name, duration_ms, "ms", {"status": str(status_code)})
            
            # Store for latency analysis
            timing_key = f"{service}:{endpoint}"
            self.request_times[timing_key].append(duration_ms)
            
            # Track errors
            if status_code >= 400:
                error_key = f"{service}:{endpoint}:errors"
                self.error_counts[error_key] += 1
                await self._record_metric(f"request.errors.{service}.{endpoint}", 1, "count")
            
            # Track throughput
            throughput_key = f"{service}:{endpoint}:throughput"
            current_minute = int(now.timestamp() // 60)
            
            # Initialize or update throughput counter for current minute
            if not self.throughput_counters[throughput_key] or \
               self.throughput_counters[throughput_key][-1][0] != current_minute:
                self.throughput_counters[throughput_key].append([current_minute, 0])
            
            self.throughput_counters[throughput_key][-1][1] += 1
            
            return APIResponse(success=True, data={"recorded": True})
            
        except Exception as e:
            return APIResponse(
                success=False,
                error=APIError("TIMING_RECORD_ERROR", f"Failed to record request timing: {str(e)}")
            )
    
    async def get_performance_summary(self, time_range_minutes: int = 60) -> APIResponse:
        """Get comprehensive performance summary"""
        try:
            now = datetime.utcnow()
            cutoff_time = now - timedelta(minutes=time_range_minutes)
            
            # System resources
            latest_system = self.system_history[-1] if self.system_history else None
            
            # Calculate request latency statistics
            latency_stats = {}
            for timing_key, timings in self.request_times.items():
                if timings:
                    recent_timings = list(timings)[-100:]  # Last 100 requests
                    latency_stats[timing_key] = {
                        "average_ms": sum(recent_timings) / len(recent_timings),
                        "p95_ms": self._calculate_percentile(recent_timings, 0.95),
                        "p99_ms": self._calculate_percentile(recent_timings, 0.99),
                        "min_ms": min(recent_timings),
                        "max_ms": max(recent_timings),
                        "request_count": len(recent_timings)
                    }
            
            # Calculate throughput
            throughput_stats = {}
            for throughput_key, counters in self.throughput_counters.items():
                if counters:
                    recent_counters = [c[1] for c in counters if c[0] >= (now.timestamp() // 60) - time_range_minutes]
                    if recent_counters:
                        throughput_stats[throughput_key] = {
                            "requests_per_minute": sum(recent_counters) / max(1, len(recent_counters)),
                            "peak_rpm": max(recent_counters),
                            "total_requests": sum(recent_counters)
                        }
            
            # Error rates
            error_stats = {}
            for error_key, error_count in self.error_counts.items():
                if error_count > 0:
                    service_endpoint = error_key.replace(":errors", "")
                    throughput_key = f"{service_endpoint}:throughput"
                    
                    total_requests = sum(c[1] for c in self.throughput_counters.get(throughput_key, []))
                    error_rate = (error_count / max(1, total_requests)) * 100
                    
                    error_stats[service_endpoint] = {
                        "error_count": error_count,
                        "error_rate_percent": error_rate
                    }
            
            # Active alerts
            active_alerts_summary = [
                {
                    "id": alert.id,
                    "issue_type": alert.issue_type.value,
                    "service": alert.service,
                    "severity": alert.severity,
                    "message": alert.message
                }
                for alert in self.active_alerts.values()
                if not alert.resolved
            ]
            
            result = {
                "time_range_minutes": time_range_minutes,
                "system_resources": {
                    "cpu_percent": latest_system.cpu_percent if latest_system else 0,
                    "memory_percent": latest_system.memory_percent if latest_system else 0,
                    "disk_usage_percent": latest_system.disk_usage_percent if latest_system else 0,
                    "active_connections": latest_system.active_connections if latest_system else 0,
                    "timestamp": latest_system.timestamp.isoformat() if latest_system else None
                },
                "latency_stats": latency_stats,
                "throughput_stats": throughput_stats,
                "error_stats": error_stats,
                "active_alerts": active_alerts_summary,
                "optimization_suggestions": len(self.optimization_suggestions),
                "applied_optimizations": len(self.applied_optimizations)
            }
            
            await self._record_metric("performance.summaries_generated", 1, "count")
            
            return APIResponse(success=True, data=result)
            
        except Exception as e:
            return APIResponse(
                success=False,
                error=APIError("PERFORMANCE_SUMMARY_ERROR", f"Failed to generate performance summary: {str(e)}")
            )
    
    async def get_system_resources(self) -> APIResponse:
        """Get current system resource utilization"""
        try:
            # Get current system stats
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            # Get network connections
            try:
                connections = len(psutil.net_connections())
            except:
                connections = 0
            
            # Get process count
            process_count = len(psutil.pids())
            
            resources = SystemResources(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_usage_percent=(disk.used / disk.total) * 100,
                network_io_bytes_sent=network.bytes_sent,
                network_io_bytes_recv=network.bytes_recv,
                active_connections=connections,
                process_count=process_count,
                timestamp=datetime.utcnow()
            )
            
            # Store in history
            self.system_history.append(resources)
            
            # Record as metrics
            await self._record_metric("system.cpu_percent", resources.cpu_percent, "percent")
            await self._record_metric("system.memory_percent", resources.memory_percent, "percent")
            await self._record_metric("system.disk_usage_percent", resources.disk_usage_percent, "percent")
            await self._record_metric("system.active_connections", resources.active_connections, "count")
            await self._record_metric("system.process_count", resources.process_count, "count")
            
            result = {
                "cpu_percent": resources.cpu_percent,
                "memory_percent": resources.memory_percent,
                "memory_available_gb": memory.available / (1024**3),
                "memory_used_gb": memory.used / (1024**3),
                "memory_total_gb": memory.total / (1024**3),
                "disk_usage_percent": resources.disk_usage_percent,
                "disk_free_gb": disk.free / (1024**3),
                "disk_used_gb": disk.used / (1024**3),
                "disk_total_gb": disk.total / (1024**3),
                "network_bytes_sent": resources.network_io_bytes_sent,
                "network_bytes_recv": resources.network_io_bytes_recv,
                "active_connections": resources.active_connections,
                "process_count": resources.process_count,
                "timestamp": resources.timestamp.isoformat()
            }
            
            return APIResponse(success=True, data=result)
            
        except Exception as e:
            return APIResponse(
                success=False,
                error=APIError("SYSTEM_RESOURCES_ERROR", f"Failed to get system resources: {str(e)}")
            )
    
    async def get_service_metrics(self, service: str, time_range_minutes: int = 60) -> APIResponse:
        """Get detailed metrics for a specific service"""
        try:
            now = datetime.utcnow()
            cutoff_time = now - timedelta(minutes=time_range_minutes)
            
            # Find all metrics for this service
            service_metrics = {}
            for metric_name, metric_data in self.metrics.items():
                if service in metric_name:
                    recent_data = [
                        m for m in metric_data
                        if m.timestamp >= cutoff_time
                    ]
                    
                    if recent_data:
                        values = [m.value for m in recent_data]
                        service_metrics[metric_name] = {
                            "current_value": values[-1] if values else 0,
                            "average": sum(values) / len(values),
                            "minimum": min(values),
                            "maximum": max(values),
                            "data_points": len(values),
                            "unit": recent_data[0].unit if recent_data else None
                        }
            
            # Get request timing data
            timing_metrics = {}
            for timing_key, timings in self.request_times.items():
                if service in timing_key:
                    recent_timings = list(timings)
                    if recent_timings:
                        timing_metrics[timing_key] = {
                            "average_ms": sum(recent_timings) / len(recent_timings),
                            "p95_ms": self._calculate_percentile(recent_timings, 0.95),
                            "p99_ms": self._calculate_percentile(recent_timings, 0.99),
                            "requests": len(recent_timings)
                        }
            
            # Get throughput data
            throughput_metrics = {}
            for throughput_key, counters in self.throughput_counters.items():
                if service in throughput_key:
                    recent_counters = [c[1] for c in counters if c[0] >= (now.timestamp() // 60) - time_range_minutes]
                    if recent_counters:
                        throughput_metrics[throughput_key] = {
                            "requests_per_minute": sum(recent_counters) / max(1, len(recent_counters)),
                            "peak_rpm": max(recent_counters),
                            "total_requests": sum(recent_counters)
                        }
            
            # Check for active alerts
            service_alerts = [
                {
                    "id": alert.id,
                    "issue_type": alert.issue_type.value,
                    "metric_name": alert.metric_name,
                    "current_value": alert.current_value,
                    "threshold": alert.threshold_value,
                    "severity": alert.severity,
                    "message": alert.message
                }
                for alert in self.active_alerts.values()
                if alert.service == service and not alert.resolved
            ]
            
            result = {
                "service": service,
                "time_range_minutes": time_range_minutes,
                "metrics": service_metrics,
                "timing_metrics": timing_metrics,
                "throughput_metrics": throughput_metrics,
                "active_alerts": service_alerts,
                "health_score": self._calculate_service_health_score(service)
            }
            
            return APIResponse(success=True, data=result)
            
        except Exception as e:
            return APIResponse(
                success=False,
                error=APIError("SERVICE_METRICS_ERROR", f"Failed to get service metrics: {str(e)}")
            )
    
    async def set_performance_threshold(self, metric_name: str, threshold_value: float, 
                                      comparison: str = "greater_than") -> APIResponse:
        """Set performance threshold for alerting"""
        try:
            if comparison not in ["greater_than", "less_than", "equal_to"]:
                return APIResponse(
                    success=False,
                    error=APIError("INVALID_COMPARISON", "Comparison must be 'greater_than', 'less_than', or 'equal_to'")
                )
            
            self.alert_thresholds[metric_name] = {
                "threshold": threshold_value,
                "comparison": comparison,
                "created_at": datetime.utcnow()
            }
            
            await self._record_metric("performance.thresholds_set", 1, "count")
            
            return APIResponse(
                success=True,
                data={
                    "metric_name": metric_name,
                    "threshold": threshold_value,
                    "comparison": comparison
                },
                message="Performance threshold set successfully"
            )
            
        except Exception as e:
            return APIResponse(
                success=False,
                error=APIError("THRESHOLD_SET_ERROR", f"Failed to set performance threshold: {str(e)}")
            )
    
    async def get_optimization_suggestions(self) -> APIResponse:
        """Get performance optimization suggestions"""
        try:
            # Generate new suggestions based on current performance data
            await self._generate_optimization_suggestions()
            
            # Sort suggestions by priority (impact score)
            sorted_suggestions = sorted(
                self.optimization_suggestions,
                key=lambda s: s.get("impact_score", 0),
                reverse=True
            )
            
            result = {
                "suggestions": sorted_suggestions[:10],  # Top 10 suggestions
                "total_suggestions": len(self.optimization_suggestions),
                "applied_optimizations": len(self.applied_optimizations),
                "generated_at": datetime.utcnow().isoformat()
            }
            
            return APIResponse(success=True, data=result)
            
        except Exception as e:
            return APIResponse(
                success=False,
                error=APIError("OPTIMIZATION_ERROR", f"Failed to get optimization suggestions: {str(e)}")
            )
    
    async def get_performance_trends(self, metric_name: str, days: int = 7) -> APIResponse:
        """Get performance trends and forecasting"""
        try:
            if metric_name not in self.metrics:
                return APIResponse(
                    success=False,
                    error=APIError("METRIC_NOT_FOUND", f"Metric '{metric_name}' not found")
                )
            
            now = datetime.utcnow()
            cutoff_time = now - timedelta(days=days)
            
            # Get historical data
            historical_data = [
                m for m in self.metrics[metric_name]
                if m.timestamp >= cutoff_time
            ]
            
            if len(historical_data) < 10:
                return APIResponse(
                    success=False,
                    error=APIError("INSUFFICIENT_DATA", "Not enough data for trend analysis")
                )
            
            # Calculate trend statistics
            values = [m.value for m in historical_data]
            timestamps = [m.timestamp for m in historical_data]
            
            # Simple linear regression for trend
            n = len(values)
            x_values = list(range(n))
            x_sum = sum(x_values)
            y_sum = sum(values)
            xy_sum = sum(x * y for x, y in zip(x_values, values))
            x_sq_sum = sum(x * x for x in x_values)
            
            slope = (n * xy_sum - x_sum * y_sum) / (n * x_sq_sum - x_sum * x_sum)
            intercept = (y_sum - slope * x_sum) / n
            
            # Determine trend direction
            if slope > 0.01:
                trend_direction = "increasing"
            elif slope < -0.01:
                trend_direction = "decreasing"
            else:
                trend_direction = "stable"
            
            # Calculate forecast for next 24 hours
            future_hours = 24
            forecast_values = []
            for i in range(future_hours):
                forecast_x = n + i
                forecast_y = slope * forecast_x + intercept
                forecast_values.append(max(0, forecast_y))  # Ensure non-negative
            
            # Calculate volatility
            mean_value = sum(values) / len(values)
            variance = sum((v - mean_value) ** 2 for v in values) / len(values)
            volatility = variance ** 0.5
            
            result = {
                "metric_name": metric_name,
                "analysis_period_days": days,
                "data_points": len(historical_data),
                "trend": {
                    "direction": trend_direction,
                    "slope": slope,
                    "confidence": "high" if n > 100 else "medium" if n > 50 else "low"
                },
                "statistics": {
                    "current_value": values[-1],
                    "average": mean_value,
                    "minimum": min(values),
                    "maximum": max(values),
                    "volatility": volatility
                },
                "forecast": {
                    "next_24h_values": forecast_values,
                    "predicted_average": sum(forecast_values) / len(forecast_values),
                    "confidence_interval": volatility * 2
                }
            }
            
            return APIResponse(success=True, data=result)
            
        except Exception as e:
            return APIResponse(
                success=False,
                error=APIError("TREND_ANALYSIS_ERROR", f"Failed to analyze trends: {str(e)}")
            )
    
    async def _record_metric(self, name: str, value: float, unit: str, labels: Optional[Dict[str, str]] = None):
        """Internal method to record a performance metric"""
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            unit=unit,
            labels=labels or {}
        )
        
        self.metrics[name].append(metric)
        
        # Update metadata
        if name not in self.metric_metadata:
            self.metric_metadata[name] = {
                "unit": unit,
                "first_recorded": metric.timestamp,
                "total_recordings": 0
            }
        
        self.metric_metadata[name]["total_recordings"] += 1
        self.metric_metadata[name]["last_recorded"] = metric.timestamp
    
    def _calculate_percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile value"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * percentile
        f = int(k)
        c = k - f
        
        if f + 1 < len(sorted_values):
            return sorted_values[f] + c * (sorted_values[f + 1] - sorted_values[f])
        else:
            return sorted_values[f]
    
    def _calculate_service_health_score(self, service: str) -> float:
        """Calculate health score for a service (0.0 to 1.0)"""
        try:
            score = 1.0
            
            # Check latency (lower is better)
            for timing_key, timings in self.request_times.items():
                if service in timing_key and timings:
                    recent_timings = list(timings)[-50:]
                    avg_latency = sum(recent_timings) / len(recent_timings)
                    
                    # Penalize high latency
                    if avg_latency > 1000:  # > 1 second
                        score *= 0.7
                    elif avg_latency > 500:  # > 500ms
                        score *= 0.85
            
            # Check error rates
            for error_key, error_count in self.error_counts.items():
                if service in error_key:
                    service_endpoint = error_key.replace(":errors", "")
                    throughput_key = f"{service_endpoint}:throughput"
                    
                    total_requests = sum(c[1] for c in self.throughput_counters.get(throughput_key, []))
                    if total_requests > 0:
                        error_rate = error_count / total_requests
                        
                        # Penalize high error rates
                        if error_rate > 0.05:  # > 5%
                            score *= 0.5
                        elif error_rate > 0.01:  # > 1%
                            score *= 0.8
            
            # Check for active alerts
            service_alerts = [a for a in self.active_alerts.values() if a.service == service and not a.resolved]
            if service_alerts:
                score *= max(0.3, 1.0 - (len(service_alerts) * 0.2))
            
            return max(0.0, min(1.0, score))
            
        except Exception:
            return 0.5  # Default to medium health on error
    
    async def _generate_optimization_suggestions(self):
        """Generate performance optimization suggestions"""
        suggestions = []
        
        try:
            # Analyze system resources
            if self.system_history:
                recent_resources = list(self.system_history)[-60:]  # Last hour
                avg_cpu = sum(r.cpu_percent for r in recent_resources) / len(recent_resources)
                avg_memory = sum(r.memory_percent for r in recent_resources) / len(recent_resources)
                
                # High CPU usage suggestion
                if avg_cpu > 80:
                    suggestions.append({
                        "id": "high_cpu_optimization",
                        "type": "resource_optimization",
                        "priority": "high",
                        "impact_score": 9,
                        "title": "High CPU Usage Detected",
                        "description": f"Average CPU usage is {avg_cpu:.1f}%. Consider scaling horizontally or optimizing CPU-intensive operations.",
                        "actions": [
                            "Profile CPU-intensive code paths",
                            "Implement caching for expensive operations",
                            "Consider horizontal scaling",
                            "Review algorithm efficiency"
                        ]
                    })
                
                # High memory usage suggestion
                if avg_memory > 85:
                    suggestions.append({
                        "id": "high_memory_optimization",
                        "type": "resource_optimization", 
                        "priority": "high",
                        "impact_score": 8,
                        "title": "High Memory Usage Detected",
                        "description": f"Average memory usage is {avg_memory:.1f}%. Consider memory optimization or increasing available memory.",
                        "actions": [
                            "Profile memory usage patterns",
                            "Implement memory pooling",
                            "Review data structures for efficiency",
                            "Consider memory leak detection"
                        ]
                    })
            
            # Analyze request latencies
            for timing_key, timings in self.request_times.items():
                if timings:
                    recent_timings = list(timings)[-100:]
                    avg_latency = sum(recent_timings) / len(recent_timings)
                    p95_latency = self._calculate_percentile(recent_timings, 0.95)
                    
                    if avg_latency > 500:  # > 500ms average
                        suggestions.append({
                            "id": f"high_latency_{timing_key}",
                            "type": "latency_optimization",
                            "priority": "medium",
                            "impact_score": 7,
                            "title": f"High Latency in {timing_key}",
                            "description": f"Average latency is {avg_latency:.1f}ms (P95: {p95_latency:.1f}ms)",
                            "actions": [
                                "Implement request caching",
                                "Optimize database queries",
                                "Add request queuing",
                                "Consider CDN for static content"
                            ]
                        })
            
            # Analyze throughput patterns
            for throughput_key, counters in self.throughput_counters.items():
                if counters:
                    recent_counters = [c[1] for c in counters[-60:]]  # Last hour
                    if recent_counters:
                        avg_throughput = sum(recent_counters) / len(recent_counters)
                        peak_throughput = max(recent_counters)
                        
                        # Low throughput suggestion
                        if avg_throughput < peak_throughput * 0.3:  # Current throughput < 30% of peak
                            suggestions.append({
                                "id": f"low_throughput_{throughput_key}",
                                "type": "throughput_optimization",
                                "priority": "medium",
                                "impact_score": 6,
                                "title": f"Low Throughput in {throughput_key}",
                                "description": f"Current throughput ({avg_throughput:.1f} RPM) is much lower than peak ({peak_throughput} RPM)",
                                "actions": [
                                    "Investigate performance bottlenecks",
                                    "Optimize connection pooling",
                                    "Review load balancing configuration",
                                    "Check for resource contention"
                                ]
                            })
            
            # Update suggestions list
            self.optimization_suggestions = suggestions
            
        except Exception as e:
            self.logger.error(f"Error generating optimization suggestions: {e}")
    
    def _initialize_default_thresholds(self):
        """Initialize default performance thresholds"""
        default_thresholds = {
            "system.cpu_percent": {"threshold": 80.0, "comparison": "greater_than"},
            "system.memory_percent": {"threshold": 85.0, "comparison": "greater_than"},
            "system.disk_usage_percent": {"threshold": 90.0, "comparison": "greater_than"},
            "request.duration_ms": {"threshold": 1000.0, "comparison": "greater_than"},
            "request.errors": {"threshold": 10.0, "comparison": "greater_than"}
        }
        
        for metric_name, config in default_thresholds.items():
            config["created_at"] = datetime.utcnow()
            self.alert_thresholds[metric_name] = config
    
    async def _system_monitoring_loop(self):
        """Background task for system resource monitoring"""
        while True:
            try:
                await asyncio.sleep(60)  # Every minute
                
                # Get system resources (this also stores them)
                await self.get_system_resources()
                
            except Exception as e:
                self.logger.error(f"Error in system monitoring loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _performance_analysis_loop(self):
        """Background task for performance analysis"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Update baselines for all metrics
                for metric_name, metric_data in self.metrics.items():
                    if len(metric_data) >= 10:
                        recent_values = [m.value for m in list(metric_data)[-100:]]
                        mean = sum(recent_values) / len(recent_values)
                        variance = sum((x - mean) ** 2 for x in recent_values) / len(recent_values)
                        std = variance ** 0.5
                        
                        self.baselines[metric_name] = {
                            "mean": mean,
                            "std": std,
                            "updated_at": datetime.utcnow()
                        }
                
                await self._record_metric("performance.analysis_cycles", 1, "count")
                
            except Exception as e:
                self.logger.error(f"Error in performance analysis loop: {e}")
                await asyncio.sleep(900)  # Wait 15 minutes on error
    
    async def _alert_evaluation_loop(self):
        """Background task for performance alert evaluation"""
        while True:
            try:
                await asyncio.sleep(30)  # Every 30 seconds
                
                # Evaluate thresholds
                for metric_name, threshold_config in self.alert_thresholds.items():
                    if metric_name not in self.metrics or not self.metrics[metric_name]:
                        continue
                    
                    latest_metric = self.metrics[metric_name][-1]
                    current_value = latest_metric.value
                    threshold = threshold_config["threshold"]
                    comparison = threshold_config["comparison"]
                    
                    # Check if threshold is breached
                    breached = False
                    if comparison == "greater_than" and current_value > threshold:
                        breached = True
                    elif comparison == "less_than" and current_value < threshold:
                        breached = True
                    elif comparison == "equal_to" and current_value == threshold:
                        breached = True
                    
                    if breached:
                        alert_id = f"perf_alert_{metric_name}_{int(time.time())}"
                        
                        # Don't create duplicate alerts
                        existing_alerts = [
                            a for a in self.active_alerts.values()
                            if a.metric_name == metric_name and not a.resolved
                        ]
                        
                        if not existing_alerts:
                            # Determine issue type and severity
                            issue_type, severity = self._classify_performance_issue(metric_name, current_value, threshold)
                            
                            alert = PerformanceAlert(
                                id=alert_id,
                                issue_type=issue_type,
                                service="system",  # Default to system
                                metric_name=metric_name,
                                current_value=current_value,
                                threshold_value=threshold,
                                severity=severity,
                                message=f"{metric_name} is {current_value} (threshold: {threshold})",
                                timestamp=datetime.utcnow()
                            )
                            
                            self.active_alerts[alert_id] = alert
                            await self._record_metric("performance.alerts_triggered", 1, "count")
                            
                            self.logger.warning(f"Performance alert: {alert.message}")
                
            except Exception as e:
                self.logger.error(f"Error in alert evaluation loop: {e}")
                await asyncio.sleep(120)  # Wait 2 minutes on error
    
    async def _optimization_loop(self):
        """Background task for optimization analysis"""
        while True:
            try:
                await asyncio.sleep(1800)  # Every 30 minutes
                
                # Generate optimization suggestions
                await self._generate_optimization_suggestions()
                
                await self._record_metric("performance.optimization_analyses", 1, "count")
                
            except Exception as e:
                self.logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour on error
    
    async def _cleanup_loop(self):
        """Background task for data cleanup"""
        while True:
            try:
                await asyncio.sleep(3600)  # Every hour
                
                cleanup_count = 0
                
                # Clean up old resolved alerts (keep for 24 hours)
                now = datetime.utcnow()
                cutoff_time = now - timedelta(hours=24)
                
                alerts_to_remove = [
                    alert_id for alert_id, alert in self.active_alerts.items()
                    if alert.resolved and alert.timestamp < cutoff_time
                ]
                
                for alert_id in alerts_to_remove:
                    del self.active_alerts[alert_id]
                    cleanup_count += 1
                
                # Clean up old baselines
                baselines_to_remove = [
                    metric_name for metric_name, baseline in self.baselines.items()
                    if baseline.get("updated_at", now) < cutoff_time
                ]
                
                for metric_name in baselines_to_remove:
                    del self.baselines[metric_name]
                    cleanup_count += 1
                
                await self._record_metric("performance.cleanup_items", cleanup_count, "count")
                
                if cleanup_count > 0:
                    self.logger.info(f"Cleaned up {cleanup_count} old performance items")
                
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(7200)  # Wait 2 hours on error
    
    def _classify_performance_issue(self, metric_name: str, current_value: float, threshold: float) -> Tuple[PerformanceIssueType, str]:
        """Classify performance issue type and severity"""
        # Determine issue type based on metric name
        if "cpu" in metric_name.lower():
            issue_type = PerformanceIssueType.HIGH_CPU
        elif "memory" in metric_name.lower():
            issue_type = PerformanceIssueType.HIGH_MEMORY
        elif "duration" in metric_name.lower() or "latency" in metric_name.lower():
            issue_type = PerformanceIssueType.HIGH_LATENCY
        elif "error" in metric_name.lower():
            issue_type = PerformanceIssueType.HIGH_ERROR_RATE
        else:
            issue_type = PerformanceIssueType.LOW_THROUGHPUT
        
        # Determine severity based on how much the threshold is exceeded
        excess_ratio = current_value / threshold if threshold > 0 else 1.0
        
        if excess_ratio >= 2.0:
            severity = "critical"
        elif excess_ratio >= 1.5:
            severity = "high"
        elif excess_ratio >= 1.2:
            severity = "medium"
        else:
            severity = "low"
        
        return issue_type, severity
    
    async def shutdown(self):
        """Shutdown the performance monitoring system"""
        self.logger.info("Shutting down performance monitoring system")
        
        # Cancel all background tasks
        for task in self._performance_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._performance_tasks:
            await asyncio.gather(*self._performance_tasks, return_exceptions=True)
        
        await self._record_metric("performance.monitoring_shutdown", 1, "count")
        self.logger.info("Performance monitoring system shutdown complete")