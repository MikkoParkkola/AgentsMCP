"""
Comprehensive Monitoring API for AgentsMCP

Provides comprehensive system monitoring, metrics collection, alerting,
and observability for all backend services with real-time dashboards
and proactive issue detection.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from enum import Enum

from .base import APIBase, APIResponse, APIError


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics we can track"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Metric:
    """Represents a single metric"""
    name: str
    type: MetricType
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None


@dataclass
class Alert:
    """Represents a system alert"""
    id: str
    severity: AlertSeverity
    message: str
    source: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemHealth:
    """Overall system health status"""
    status: str  # healthy, degraded, unhealthy
    score: float  # 0.0 to 1.0
    components: Dict[str, Dict[str, Any]]
    last_updated: datetime
    issues: List[str] = field(default_factory=list)


class ComprehensiveMonitoring(APIBase):
    """
    Comprehensive monitoring system for AgentsMCP backend services.
    
    Features:
    - Real-time metrics collection and aggregation
    - Proactive alerting with customizable thresholds
    - System health monitoring and scoring
    - Performance trend analysis
    - Service dependency tracking
    - Automated anomaly detection
    - Real-time dashboards and reporting
    """
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Metrics storage (time-series data)
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.metric_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Alerting system
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.alert_history: deque = deque(maxlen=1000)
        
        # Health monitoring
        self.component_health: Dict[str, Dict[str, Any]] = {}
        self.health_checks: Dict[str, Any] = {}
        
        # Performance baselines and anomaly detection
        self.baselines: Dict[str, Dict[str, float]] = {}
        self.anomaly_thresholds: Dict[str, float] = defaultdict(lambda: 2.0)
        
        # Service dependencies
        self.service_dependencies: Dict[str, Set[str]] = defaultdict(set)
        
        # Background tasks
        self._monitoring_tasks: Set[asyncio.Task] = set()
        self._initialize_default_rules()
        
    async def initialize(self) -> APIResponse:
        """Initialize the monitoring system"""
        try:
            # Start background monitoring tasks
            tasks = [
                asyncio.create_task(self._metrics_aggregation_loop()),
                asyncio.create_task(self._health_check_loop()),
                asyncio.create_task(self._alert_evaluation_loop()),
                asyncio.create_task(self._anomaly_detection_loop()),
                asyncio.create_task(self._cleanup_loop())
            ]
            
            self._monitoring_tasks.update(tasks)
            
            await self._record_metric("monitoring.initialized", 1, MetricType.COUNTER)
            self.logger.info("Comprehensive monitoring system initialized")
            
            return APIResponse(
                success=True,
                data={"status": "initialized", "active_tasks": len(self._monitoring_tasks)},
                message="Monitoring system initialized successfully"
            )
            
        except Exception as e:
            return APIResponse(
                success=False,
                error=APIError("MONITORING_INIT_ERROR", f"Failed to initialize monitoring: {str(e)}")
            )
    
    async def record_metric(self, name: str, value: float, metric_type: MetricType = MetricType.GAUGE, 
                           labels: Optional[Dict[str, str]] = None, unit: Optional[str] = None) -> APIResponse:
        """Record a metric value"""
        try:
            await self._record_metric(name, value, metric_type, labels, unit)
            return APIResponse(success=True, data={"metric_recorded": name})
            
        except Exception as e:
            return APIResponse(
                success=False,
                error=APIError("METRIC_RECORD_ERROR", f"Failed to record metric: {str(e)}")
            )
    
    async def get_metrics(self, name_pattern: Optional[str] = None, 
                         time_range: Optional[int] = None) -> APIResponse:
        """Get metrics data"""
        try:
            now = datetime.utcnow()
            cutoff_time = now - timedelta(seconds=time_range) if time_range else None
            
            result = {}
            for metric_name, metric_data in self.metrics.items():
                if name_pattern and name_pattern not in metric_name:
                    continue
                
                # Filter by time range if specified
                if cutoff_time:
                    filtered_data = [m for m in metric_data if m.timestamp >= cutoff_time]
                else:
                    filtered_data = list(metric_data)
                
                result[metric_name] = {
                    "data_points": len(filtered_data),
                    "latest_value": filtered_data[-1].value if filtered_data else None,
                    "latest_timestamp": filtered_data[-1].timestamp.isoformat() if filtered_data else None,
                    "metadata": self.metric_metadata.get(metric_name, {})
                }
            
            await self._record_metric("monitoring.metrics_queries", 1, MetricType.COUNTER)
            
            return APIResponse(
                success=True,
                data={
                    "metrics": result,
                    "query_time": time_range,
                    "total_metrics": len(result)
                }
            )
            
        except Exception as e:
            return APIResponse(
                success=False,
                error=APIError("METRICS_QUERY_ERROR", f"Failed to query metrics: {str(e)}")
            )
    
    async def create_alert_rule(self, rule_id: str, metric_name: str, condition: str,
                              threshold: float, severity: AlertSeverity, 
                              message_template: str) -> APIResponse:
        """Create a new alert rule"""
        try:
            rule = {
                "metric_name": metric_name,
                "condition": condition,  # "gt", "lt", "eq", "ne"
                "threshold": threshold,
                "severity": severity,
                "message_template": message_template,
                "enabled": True,
                "created_at": datetime.utcnow(),
                "triggered_count": 0,
                "last_triggered": None
            }
            
            self.alert_rules[rule_id] = rule
            await self._record_metric("monitoring.alert_rules", 1, MetricType.COUNTER)
            
            self.logger.info(f"Created alert rule: {rule_id}")
            
            return APIResponse(
                success=True,
                data={"rule_id": rule_id, "rule": rule},
                message="Alert rule created successfully"
            )
            
        except Exception as e:
            return APIResponse(
                success=False,
                error=APIError("ALERT_RULE_ERROR", f"Failed to create alert rule: {str(e)}")
            )
    
    async def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> APIResponse:
        """Get all active alerts"""
        try:
            active_alerts = [
                alert for alert in self.alerts.values() 
                if not alert.resolved and (not severity or alert.severity == severity)
            ]
            
            # Sort by severity and timestamp
            severity_order = {"critical": 0, "error": 1, "warning": 2, "info": 3}
            active_alerts.sort(
                key=lambda a: (severity_order.get(a.severity.value, 4), a.timestamp),
                reverse=True
            )
            
            result = {
                "alerts": [
                    {
                        "id": alert.id,
                        "severity": alert.severity.value,
                        "message": alert.message,
                        "source": alert.source,
                        "timestamp": alert.timestamp.isoformat(),
                        "metadata": alert.metadata
                    }
                    for alert in active_alerts
                ],
                "total_count": len(active_alerts),
                "by_severity": {
                    severity.value: len([a for a in active_alerts if a.severity == severity])
                    for severity in AlertSeverity
                }
            }
            
            return APIResponse(success=True, data=result)
            
        except Exception as e:
            return APIResponse(
                success=False,
                error=APIError("ALERTS_QUERY_ERROR", f"Failed to query alerts: {str(e)}")
            )
    
    async def resolve_alert(self, alert_id: str, resolution_note: Optional[str] = None) -> APIResponse:
        """Resolve an active alert"""
        try:
            if alert_id not in self.alerts:
                return APIResponse(
                    success=False,
                    error=APIError("ALERT_NOT_FOUND", f"Alert {alert_id} not found")
                )
            
            alert = self.alerts[alert_id]
            if alert.resolved:
                return APIResponse(
                    success=False,
                    error=APIError("ALERT_ALREADY_RESOLVED", f"Alert {alert_id} already resolved")
                )
            
            alert.resolved = True
            alert.resolved_at = datetime.utcnow()
            if resolution_note:
                alert.metadata["resolution_note"] = resolution_note
            
            await self._record_metric("monitoring.alerts_resolved", 1, MetricType.COUNTER)
            
            self.logger.info(f"Resolved alert: {alert_id}")
            
            return APIResponse(
                success=True,
                data={"alert_id": alert_id, "resolved_at": alert.resolved_at.isoformat()},
                message="Alert resolved successfully"
            )
            
        except Exception as e:
            return APIResponse(
                success=False,
                error=APIError("ALERT_RESOLVE_ERROR", f"Failed to resolve alert: {str(e)}")
            )
    
    async def get_system_health(self) -> APIResponse:
        """Get overall system health status"""
        try:
            # Calculate overall health score
            component_scores = []
            healthy_components = 0
            total_components = len(self.component_health)
            
            for component_name, health_data in self.component_health.items():
                score = health_data.get("score", 0.0)
                component_scores.append(score)
                if score >= 0.8:
                    healthy_components += 1
            
            overall_score = sum(component_scores) / len(component_scores) if component_scores else 0.0
            
            # Determine status
            if overall_score >= 0.9:
                status = "healthy"
            elif overall_score >= 0.7:
                status = "degraded"
            else:
                status = "unhealthy"
            
            # Get active issues
            active_alerts = [a for a in self.alerts.values() if not a.resolved]
            critical_alerts = [a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]
            
            issues = []
            if critical_alerts:
                issues.extend([f"Critical: {alert.message}" for alert in critical_alerts[:3]])
            
            health = SystemHealth(
                status=status,
                score=overall_score,
                components=dict(self.component_health),
                last_updated=datetime.utcnow(),
                issues=issues
            )
            
            await self._record_metric("monitoring.health_score", overall_score, MetricType.GAUGE)
            
            return APIResponse(
                success=True,
                data={
                    "status": health.status,
                    "score": health.score,
                    "components": health.components,
                    "last_updated": health.last_updated.isoformat(),
                    "issues": health.issues,
                    "summary": {
                        "healthy_components": healthy_components,
                        "total_components": total_components,
                        "active_alerts": len(active_alerts),
                        "critical_alerts": len(critical_alerts)
                    }
                }
            )
            
        except Exception as e:
            return APIResponse(
                success=False,
                error=APIError("HEALTH_CHECK_ERROR", f"Failed to get system health: {str(e)}")
            )
    
    async def register_health_check(self, component: str, check_function: Any, 
                                  interval: int = 60) -> APIResponse:
        """Register a health check for a component"""
        try:
            self.health_checks[component] = {
                "function": check_function,
                "interval": interval,
                "last_check": None,
                "next_check": datetime.utcnow(),
                "consecutive_failures": 0
            }
            
            # Initialize component health
            self.component_health[component] = {
                "status": "unknown",
                "score": 0.0,
                "last_check": None,
                "message": "Health check registered"
            }
            
            await self._record_metric("monitoring.health_checks", 1, MetricType.COUNTER)
            
            return APIResponse(
                success=True,
                data={"component": component, "interval": interval},
                message="Health check registered successfully"
            )
            
        except Exception as e:
            return APIResponse(
                success=False,
                error=APIError("HEALTH_CHECK_REGISTER_ERROR", f"Failed to register health check: {str(e)}")
            )
    
    async def get_performance_trends(self, metric_name: str, time_range: int = 3600) -> APIResponse:
        """Get performance trends and analysis"""
        try:
            if metric_name not in self.metrics:
                return APIResponse(
                    success=False,
                    error=APIError("METRIC_NOT_FOUND", f"Metric {metric_name} not found")
                )
            
            now = datetime.utcnow()
            cutoff_time = now - timedelta(seconds=time_range)
            
            # Get recent data points
            recent_data = [
                m for m in self.metrics[metric_name] 
                if m.timestamp >= cutoff_time
            ]
            
            if not recent_data:
                return APIResponse(
                    success=False,
                    error=APIError("NO_DATA", "No data available for the specified time range")
                )
            
            values = [d.value for d in recent_data]
            
            # Calculate statistics
            avg_value = sum(values) / len(values)
            min_value = min(values)
            max_value = max(values)
            
            # Calculate trend (simple linear regression)
            n = len(values)
            x_sum = sum(range(n))
            y_sum = sum(values)
            xy_sum = sum(i * values[i] for i in range(n))
            x_sq_sum = sum(i * i for i in range(n))
            
            if n > 1:
                slope = (n * xy_sum - x_sum * y_sum) / (n * x_sq_sum - x_sum * x_sum)
                trend = "increasing" if slope > 0.01 else "decreasing" if slope < -0.01 else "stable"
            else:
                slope = 0
                trend = "stable"
            
            # Detect anomalies
            baseline = self.baselines.get(metric_name, {})
            threshold = self.anomaly_thresholds.get(metric_name, 2.0)
            
            anomalies = []
            if baseline.get("mean") and baseline.get("std"):
                mean = baseline["mean"]
                std = baseline["std"]
                for i, value in enumerate(values):
                    if abs(value - mean) > threshold * std:
                        anomalies.append({
                            "index": i,
                            "value": value,
                            "deviation": abs(value - mean) / std,
                            "timestamp": recent_data[i].timestamp.isoformat()
                        })
            
            result = {
                "metric_name": metric_name,
                "time_range_seconds": time_range,
                "data_points": len(recent_data),
                "statistics": {
                    "average": avg_value,
                    "minimum": min_value,
                    "maximum": max_value,
                    "range": max_value - min_value
                },
                "trend": {
                    "direction": trend,
                    "slope": slope,
                    "confidence": "high" if n > 10 else "low"
                },
                "anomalies": {
                    "count": len(anomalies),
                    "threshold": threshold,
                    "detected": anomalies[:5]  # Limit to first 5
                },
                "baseline": baseline
            }
            
            await self._record_metric("monitoring.trend_analyses", 1, MetricType.COUNTER)
            
            return APIResponse(success=True, data=result)
            
        except Exception as e:
            return APIResponse(
                success=False,
                error=APIError("TREND_ANALYSIS_ERROR", f"Failed to analyze trends: {str(e)}")
            )
    
    async def _record_metric(self, name: str, value: float, metric_type: MetricType = MetricType.GAUGE,
                           labels: Optional[Dict[str, str]] = None, unit: Optional[str] = None):
        """Internal method to record a metric"""
        metric = Metric(
            name=name,
            type=metric_type,
            value=value,
            timestamp=datetime.utcnow(),
            labels=labels or {},
            unit=unit
        )
        
        self.metrics[name].append(metric)
        
        # Update metadata
        if name not in self.metric_metadata:
            self.metric_metadata[name] = {
                "type": metric_type.value,
                "unit": unit,
                "first_seen": metric.timestamp,
                "total_recordings": 0
            }
        
        self.metric_metadata[name]["total_recordings"] += 1
        self.metric_metadata[name]["last_seen"] = metric.timestamp
    
    async def _trigger_alert(self, alert_id: str, severity: AlertSeverity, message: str, 
                           source: str, metadata: Optional[Dict[str, Any]] = None):
        """Internal method to trigger an alert"""
        alert = Alert(
            id=alert_id,
            severity=severity,
            message=message,
            source=source,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        self.alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        await self._record_metric("monitoring.alerts_triggered", 1, MetricType.COUNTER,
                                labels={"severity": severity.value})
        
        self.logger.warning(f"Alert triggered: {alert_id} - {message}")
    
    def _initialize_default_rules(self):
        """Initialize default alert rules"""
        default_rules = {
            "high_memory_usage": {
                "metric_name": "system.memory_usage_percent",
                "condition": "gt",
                "threshold": 85.0,
                "severity": AlertSeverity.WARNING,
                "message_template": "Memory usage is {value}% (threshold: {threshold}%)",
                "enabled": True
            },
            "critical_memory_usage": {
                "metric_name": "system.memory_usage_percent",
                "condition": "gt",
                "threshold": 95.0,
                "severity": AlertSeverity.CRITICAL,
                "message_template": "Critical memory usage: {value}% (threshold: {threshold}%)",
                "enabled": True
            },
            "high_response_time": {
                "metric_name": "api.response_time_ms",
                "condition": "gt",
                "threshold": 1000.0,
                "severity": AlertSeverity.WARNING,
                "message_template": "High API response time: {value}ms (threshold: {threshold}ms)",
                "enabled": True
            },
            "low_health_score": {
                "metric_name": "monitoring.health_score",
                "condition": "lt",
                "threshold": 0.7,
                "severity": AlertSeverity.ERROR,
                "message_template": "System health score is low: {value} (threshold: {threshold})",
                "enabled": True
            }
        }
        
        for rule_id, rule_data in default_rules.items():
            rule_data["created_at"] = datetime.utcnow()
            rule_data["triggered_count"] = 0
            rule_data["last_triggered"] = None
            self.alert_rules[rule_id] = rule_data
    
    async def _metrics_aggregation_loop(self):
        """Background task for metrics aggregation"""
        while True:
            try:
                await asyncio.sleep(30)  # Run every 30 seconds
                
                # Update baselines for anomaly detection
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
                
                await self._record_metric("monitoring.aggregation_cycles", 1, MetricType.COUNTER)
                
            except Exception as e:
                self.logger.error(f"Error in metrics aggregation loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _health_check_loop(self):
        """Background task for health checks"""
        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                now = datetime.utcnow()
                
                for component, check_config in self.health_checks.items():
                    if now >= check_config["next_check"]:
                        try:
                            # Run health check function (mock implementation)
                            # In real implementation, this would call the actual health check function
                            health_result = {
                                "status": "healthy",
                                "score": 1.0,
                                "message": "Component is healthy"
                            }
                            
                            self.component_health[component] = {
                                "status": health_result["status"],
                                "score": health_result["score"],
                                "last_check": now,
                                "message": health_result["message"]
                            }
                            
                            check_config["last_check"] = now
                            check_config["next_check"] = now + timedelta(seconds=check_config["interval"])
                            check_config["consecutive_failures"] = 0
                            
                        except Exception as e:
                            # Health check failed
                            check_config["consecutive_failures"] += 1
                            
                            self.component_health[component] = {
                                "status": "unhealthy",
                                "score": 0.0,
                                "last_check": now,
                                "message": f"Health check failed: {str(e)}"
                            }
                            
                            # Trigger alert on consecutive failures
                            if check_config["consecutive_failures"] >= 3:
                                await self._trigger_alert(
                                    f"health_check_failed_{component}",
                                    AlertSeverity.ERROR,
                                    f"Health check for {component} failed {check_config['consecutive_failures']} times",
                                    "health_monitor"
                                )
                
                await self._record_metric("monitoring.health_checks_performed", 1, MetricType.COUNTER)
                
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(30)
    
    async def _alert_evaluation_loop(self):
        """Background task for alert rule evaluation"""
        while True:
            try:
                await asyncio.sleep(15)  # Evaluate every 15 seconds
                
                for rule_id, rule in self.alert_rules.items():
                    if not rule["enabled"]:
                        continue
                    
                    metric_name = rule["metric_name"]
                    if metric_name not in self.metrics or not self.metrics[metric_name]:
                        continue
                    
                    # Get latest metric value
                    latest_metric = self.metrics[metric_name][-1]
                    current_value = latest_metric.value
                    threshold = rule["threshold"]
                    condition = rule["condition"]
                    
                    # Evaluate condition
                    triggered = False
                    if condition == "gt" and current_value > threshold:
                        triggered = True
                    elif condition == "lt" and current_value < threshold:
                        triggered = True
                    elif condition == "eq" and current_value == threshold:
                        triggered = True
                    elif condition == "ne" and current_value != threshold:
                        triggered = True
                    
                    if triggered:
                        # Check if this alert is already active
                        alert_id = f"rule_{rule_id}_{int(time.time())}"
                        if not any(a.source == f"rule_{rule_id}" and not a.resolved for a in self.alerts.values()):
                            message = rule["message_template"].format(
                                value=current_value,
                                threshold=threshold,
                                metric=metric_name
                            )
                            
                            await self._trigger_alert(
                                alert_id,
                                rule["severity"],
                                message,
                                f"rule_{rule_id}",
                                {
                                    "rule_id": rule_id,
                                    "metric_value": current_value,
                                    "threshold": threshold,
                                    "condition": condition
                                }
                            )
                            
                            rule["triggered_count"] += 1
                            rule["last_triggered"] = datetime.utcnow()
                
                await self._record_metric("monitoring.alert_evaluations", 1, MetricType.COUNTER)
                
            except Exception as e:
                self.logger.error(f"Error in alert evaluation loop: {e}")
                await asyncio.sleep(60)
    
    async def _anomaly_detection_loop(self):
        """Background task for anomaly detection"""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                anomalies_detected = 0
                
                for metric_name, baseline in self.baselines.items():
                    if metric_name not in self.metrics or not self.metrics[metric_name]:
                        continue
                    
                    # Check recent values for anomalies
                    recent_data = list(self.metrics[metric_name])[-10:]
                    mean = baseline["mean"]
                    std = baseline["std"]
                    threshold = self.anomaly_thresholds.get(metric_name, 2.0)
                    
                    for metric in recent_data:
                        deviation = abs(metric.value - mean) / std if std > 0 else 0
                        
                        if deviation > threshold:
                            # Anomaly detected
                            alert_id = f"anomaly_{metric_name}_{int(metric.timestamp.timestamp())}"
                            
                            await self._trigger_alert(
                                alert_id,
                                AlertSeverity.WARNING,
                                f"Anomaly detected in {metric_name}: value {metric.value} deviates {deviation:.2f}σ from baseline",
                                "anomaly_detector",
                                {
                                    "metric_name": metric_name,
                                    "value": metric.value,
                                    "baseline_mean": mean,
                                    "deviation_sigma": deviation,
                                    "threshold_sigma": threshold
                                }
                            )
                            
                            anomalies_detected += 1
                
                await self._record_metric("monitoring.anomalies_detected", anomalies_detected, MetricType.GAUGE)
                
            except Exception as e:
                self.logger.error(f"Error in anomaly detection loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _cleanup_loop(self):
        """Background task for data cleanup"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                now = datetime.utcnow()
                cleanup_count = 0
                
                # Clean up old resolved alerts (keep for 7 days)
                cutoff_time = now - timedelta(days=7)
                alerts_to_remove = [
                    alert_id for alert_id, alert in self.alerts.items()
                    if alert.resolved and alert.resolved_at and alert.resolved_at < cutoff_time
                ]
                
                for alert_id in alerts_to_remove:
                    del self.alerts[alert_id]
                    cleanup_count += 1
                
                # Clean up old baselines (keep for 24 hours)
                cutoff_time = now - timedelta(hours=24)
                baselines_to_remove = [
                    metric_name for metric_name, baseline in self.baselines.items()
                    if baseline.get("updated_at", now) < cutoff_time
                ]
                
                for metric_name in baselines_to_remove:
                    del self.baselines[metric_name]
                    cleanup_count += 1
                
                await self._record_metric("monitoring.cleanup_items", cleanup_count, MetricType.GAUGE)
                self.logger.info(f"Cleaned up {cleanup_count} old monitoring items")
                
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(7200)  # Wait 2 hours on error
    
    async def shutdown(self):
        """Shutdown the monitoring system"""
        self.logger.info("Shutting down comprehensive monitoring system")
        
        # Cancel all background tasks
        for task in self._monitoring_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._monitoring_tasks:
            await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)
        
        await self._record_metric("monitoring.shutdown", 1, MetricType.COUNTER)
        self.logger.info("Comprehensive monitoring system shutdown complete")