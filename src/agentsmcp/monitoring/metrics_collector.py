"""
Real-time metrics collection system for AgentsMCP.

Collects and aggregates metrics from various system components including
orchestrator, agents, quality gates, and self-improvement systems.
"""

import asyncio
import logging
import time
import weakref
from typing import Dict, List, Optional, Any, Callable, Set, Union, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import threading

logger = logging.getLogger(__name__)

T = TypeVar('T')


class MetricType(Enum):
    """Types of metrics collected by the system."""
    COUNTER = "counter"  # Monotonically increasing values
    GAUGE = "gauge"      # Point-in-time values that can go up or down
    HISTOGRAM = "histogram"  # Distribution of values over time
    TIMER = "timer"      # Time-based measurements
    RATE = "rate"        # Rate of change over time


@dataclass
class Metric:
    """Individual metric data point."""
    name: str
    value: Union[int, float]
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class AggregatedMetric:
    """Aggregated metric with statistical information."""
    name: str
    count: int
    sum: float
    min: float
    max: float
    mean: float
    p50: float
    p95: float
    p99: float
    rate_per_second: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)


class SlidingWindow(Generic[T]):
    """Sliding window data structure for time-series data."""
    
    def __init__(self, window_size: int = 1000, time_window_seconds: float = 300):
        """
        Initialize sliding window.
        
        Args:
            window_size: Maximum number of entries to keep
            time_window_seconds: Maximum age of entries in seconds
        """
        self.window_size = window_size
        self.time_window = time_window_seconds
        self._data: deque = deque(maxlen=window_size)
        self._lock = threading.RLock()
    
    def add(self, value: T, timestamp: Optional[float] = None):
        """Add a value to the window."""
        if timestamp is None:
            timestamp = time.time()
        
        with self._lock:
            self._data.append((value, timestamp))
            self._cleanup_old_entries()
    
    def get_values(self, max_age_seconds: Optional[float] = None) -> List[T]:
        """Get all values within the specified time window."""
        with self._lock:
            if max_age_seconds is None:
                return [value for value, _ in self._data]
            
            cutoff_time = time.time() - max_age_seconds
            return [value for value, timestamp in self._data if timestamp >= cutoff_time]
    
    def _cleanup_old_entries(self):
        """Remove entries older than the time window."""
        if not self.time_window:
            return
        
        cutoff_time = time.time() - self.time_window
        while self._data and self._data[0][1] < cutoff_time:
            self._data.popleft()


class MetricsCollector:
    """
    Real-time metrics collection and aggregation system.
    
    Provides high-performance metrics collection with minimal overhead,
    supporting counters, gauges, histograms, timers, and rates.
    """
    
    def __init__(self, buffer_size: int = 10000, flush_interval: float = 1.0):
        """
        Initialize metrics collector.
        
        Args:
            buffer_size: Maximum metrics to buffer before flushing
            flush_interval: How often to flush metrics in seconds
        """
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        
        # Storage
        self._metrics: Dict[str, SlidingWindow[Metric]] = defaultdict(
            lambda: SlidingWindow(1000, 300)  # 5 minute window
        )
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        
        # Aggregations
        self._aggregated_cache: Dict[str, AggregatedMetric] = {}
        self._last_aggregation: float = 0
        self._aggregation_interval: float = 5.0  # 5 seconds
        
        # Threading
        self._lock = threading.RLock()
        self._running = False
        self._flush_task: Optional[asyncio.Task] = None
        
        # Listeners
        self._listeners: Set[Callable[[List[Metric]], None]] = set()
        self._weak_listeners: weakref.WeakSet = weakref.WeakSet()
        
        logger.info(f"MetricsCollector initialized with buffer_size={buffer_size}, flush_interval={flush_interval}")
    
    def start(self):
        """Start the metrics collection background tasks."""
        if self._running:
            return
        
        self._running = True
        # Start flush task in current event loop
        try:
            loop = asyncio.get_event_loop()
            self._flush_task = loop.create_task(self._flush_loop())
        except RuntimeError:
            logger.warning("No event loop available, metrics will be collected synchronously")
    
    async def stop(self):
        """Stop the metrics collection system."""
        self._running = False
        
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None
        
        # Final flush
        self._flush_metrics()
        logger.info("MetricsCollector stopped")
    
    def record_counter(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """Record a counter metric (monotonically increasing)."""
        with self._lock:
            key = self._metric_key(name, tags)
            self._counters[key] += value
            
            metric = Metric(
                name=name,
                value=value,
                timestamp=time.time(),
                tags=tags or {},
                metric_type=MetricType.COUNTER
            )
            self._metrics[key].add(metric)
    
    def record_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a gauge metric (point-in-time value)."""
        with self._lock:
            key = self._metric_key(name, tags)
            self._gauges[key] = value
            
            metric = Metric(
                name=name,
                value=value,
                timestamp=time.time(),
                tags=tags or {},
                metric_type=MetricType.GAUGE
            )
            self._metrics[key].add(metric)
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a histogram metric (distribution of values)."""
        with self._lock:
            key = self._metric_key(name, tags)
            self._histograms[key].append(value)
            
            # Keep only recent values
            if len(self._histograms[key]) > 1000:
                self._histograms[key] = self._histograms[key][-500:]
            
            metric = Metric(
                name=name,
                value=value,
                timestamp=time.time(),
                tags=tags or {},
                metric_type=MetricType.HISTOGRAM
            )
            self._metrics[key].add(metric)
    
    def record_timer(self, name: str, duration_seconds: float, tags: Optional[Dict[str, str]] = None):
        """Record a timer metric (time duration)."""
        with self._lock:
            key = self._metric_key(name, tags)
            
            metric = Metric(
                name=name,
                value=duration_seconds,
                timestamp=time.time(),
                tags=tags or {},
                metric_type=MetricType.TIMER
            )
            self._metrics[key].add(metric)
    
    def start_timer(self, name: str, tags: Optional[Dict[str, str]] = None) -> Callable[[], None]:
        """Start a timer and return a function to stop it."""
        start_time = time.time()
        
        def stop_timer():
            duration = time.time() - start_time
            self.record_timer(name, duration, tags)
        
        return stop_timer
    
    def get_counter(self, name: str, tags: Optional[Dict[str, str]] = None) -> float:
        """Get current counter value."""
        with self._lock:
            key = self._metric_key(name, tags)
            return self._counters.get(key, 0.0)
    
    def get_gauge(self, name: str, tags: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get current gauge value."""
        with self._lock:
            key = self._metric_key(name, tags)
            return self._gauges.get(key)
    
    def get_histogram_stats(self, name: str, tags: Optional[Dict[str, str]] = None) -> Optional[Dict[str, float]]:
        """Get histogram statistics."""
        with self._lock:
            key = self._metric_key(name, tags)
            values = self._histograms.get(key, [])
            
            if not values:
                return None
            
            sorted_values = sorted(values)
            return {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'mean': statistics.mean(values),
                'p50': statistics.median(values),
                'p95': sorted_values[int(0.95 * len(sorted_values))],
                'p99': sorted_values[int(0.99 * len(sorted_values))],
                'sum': sum(values)
            }
    
    def get_aggregated_metrics(self, max_age_seconds: float = 300) -> List[AggregatedMetric]:
        """Get aggregated metrics for all collected data."""
        current_time = time.time()
        
        # Check if we need to refresh aggregations
        if current_time - self._last_aggregation > self._aggregation_interval:
            self._update_aggregations()
            self._last_aggregation = current_time
        
        # Filter by age
        cutoff_time = current_time - max_age_seconds
        return [
            metric for metric in self._aggregated_cache.values()
            if metric.timestamp >= cutoff_time
        ]
    
    def get_metric_names(self) -> Set[str]:
        """Get all available metric names."""
        with self._lock:
            names = set()
            for key in self._metrics.keys():
                name, _ = self._parse_metric_key(key)
                names.add(name)
            return names
    
    def get_recent_metrics(self, name: str, max_count: int = 100, 
                          max_age_seconds: float = 60) -> List[Metric]:
        """Get recent metrics for a specific name."""
        with self._lock:
            all_metrics = []
            
            # Collect from all tag combinations for this name
            for key, window in self._metrics.items():
                metric_name, _ = self._parse_metric_key(key)
                if metric_name == name:
                    all_metrics.extend(window.get_values(max_age_seconds))
            
            # Sort by timestamp and limit count
            all_metrics.sort(key=lambda m: m.timestamp, reverse=True)
            return all_metrics[:max_count]
    
    def add_listener(self, callback: Callable[[List[Metric]], None], weak: bool = True):
        """
        Add a listener for metric updates.
        
        Args:
            callback: Function to call with new metrics
            weak: If True, uses weak reference to avoid memory leaks
        """
        if weak:
            self._weak_listeners.add(callback)
        else:
            self._listeners.add(callback)
    
    def remove_listener(self, callback: Callable[[List[Metric]], None]):
        """Remove a metric listener."""
        self._listeners.discard(callback)
    
    async def _flush_loop(self):
        """Background task to flush metrics periodically."""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)
                self._flush_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics flush loop: {e}")
    
    def _flush_metrics(self):
        """Flush collected metrics to listeners."""
        try:
            with self._lock:
                # Collect recent metrics
                recent_metrics = []
                for window in self._metrics.values():
                    recent_metrics.extend(window.get_values(self.flush_interval * 2))
                
                if not recent_metrics:
                    return
                
                # Notify listeners
                dead_listeners = set()
                for listener in list(self._listeners):
                    try:
                        listener(recent_metrics)
                    except Exception as e:
                        logger.warning(f"Metric listener error: {e}")
                        dead_listeners.add(listener)
                
                # Clean up dead listeners
                self._listeners -= dead_listeners
                
                # Notify weak listeners
                for listener in list(self._weak_listeners):
                    try:
                        listener(recent_metrics)
                    except ReferenceError:
                        # Weak reference died, will be cleaned up automatically
                        pass
                    except Exception as e:
                        logger.warning(f"Weak metric listener error: {e}")
                
        except Exception as e:
            logger.error(f"Error flushing metrics: {e}")
    
    def _update_aggregations(self):
        """Update aggregated metric cache."""
        with self._lock:
            for key, window in self._metrics.items():
                name, tags = self._parse_metric_key(key)
                
                # Get recent values
                values = [m.value for m in window.get_values(300)]  # 5 minutes
                
                if not values:
                    continue
                
                # Calculate aggregations
                sorted_values = sorted(values)
                count = len(values)
                
                # Calculate rate (values per second)
                time_span = min(300, time.time() - window.get_values(1)[0].timestamp if count > 0 else 1)
                rate_per_second = count / time_span if time_span > 0 else 0
                
                aggregated = AggregatedMetric(
                    name=name,
                    count=count,
                    sum=sum(values),
                    min=min(values),
                    max=max(values),
                    mean=statistics.mean(values),
                    p50=statistics.median(values),
                    p95=sorted_values[int(0.95 * count)] if count > 0 else 0,
                    p99=sorted_values[int(0.99 * count)] if count > 0 else 0,
                    rate_per_second=rate_per_second,
                    timestamp=time.time(),
                    tags=tags
                )
                
                self._aggregated_cache[key] = aggregated
    
    def _metric_key(self, name: str, tags: Optional[Dict[str, str]]) -> str:
        """Generate unique key for metric with tags."""
        if not tags:
            return name
        
        tag_str = ','.join(f'{k}={v}' for k, v in sorted(tags.items()))
        return f"{name}#{tag_str}"
    
    def _parse_metric_key(self, key: str) -> tuple[str, Dict[str, str]]:
        """Parse metric key back into name and tags."""
        if '#' not in key:
            return key, {}
        
        name, tag_str = key.split('#', 1)
        tags = {}
        
        if tag_str:
            for tag in tag_str.split(','):
                if '=' in tag:
                    k, v = tag.split('=', 1)
                    tags[k] = v
        
        return name, tags
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collector statistics."""
        with self._lock:
            return {
                'metrics_count': len(self._metrics),
                'counters_count': len(self._counters),
                'gauges_count': len(self._gauges),
                'histograms_count': len(self._histograms),
                'listeners_count': len(self._listeners) + len(self._weak_listeners),
                'running': self._running,
                'last_aggregation': self._last_aggregation,
                'aggregated_metrics_count': len(self._aggregated_cache)
            }


# Global metrics collector instance
_global_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
        _global_collector.start()
    return _global_collector


def record_counter(name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
    """Record a counter metric using the global collector."""
    get_metrics_collector().record_counter(name, value, tags)


def record_gauge(name: str, value: float, tags: Optional[Dict[str, str]] = None):
    """Record a gauge metric using the global collector."""
    get_metrics_collector().record_gauge(name, value, tags)


def record_histogram(name: str, value: float, tags: Optional[Dict[str, str]] = None):
    """Record a histogram metric using the global collector."""
    get_metrics_collector().record_histogram(name, value, tags)


def record_timer(name: str, duration_seconds: float, tags: Optional[Dict[str, str]] = None):
    """Record a timer metric using the global collector."""
    get_metrics_collector().record_timer(name, duration_seconds, tags)


def start_timer(name: str, tags: Optional[Dict[str, str]] = None) -> Callable[[], None]:
    """Start a timer using the global collector."""
    return get_metrics_collector().start_timer(name, tags)