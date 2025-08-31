"""Performance monitoring and benchmarking utilities for AgentsMCP.

This module provides tools to measure startup time, memory usage, and track
the effectiveness of lazy loading optimizations.
"""

import functools
import gc
import logging
import os
import psutil
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta

from .lazy_loading import get_lazy_registry_status

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Single performance measurement."""
    name: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StartupBenchmark:
    """Startup performance benchmark results."""
    total_time: float
    memory_usage: float
    lazy_components_loaded: Dict[str, bool]
    import_times: Dict[str, float]
    initialization_times: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.utcnow)


class PerformanceMonitor:
    """Performance monitoring and benchmarking system."""
    
    def __init__(self):
        self.metrics: List[PerformanceMetric] = []
        self._import_start_times: Dict[str, float] = {}
        self._initialization_times: Dict[str, float] = {}
        self._startup_start_time: Optional[float] = None
        self._process = psutil.Process()
    
    def start_startup_benchmark(self) -> None:
        """Start measuring startup performance."""
        self._startup_start_time = time.perf_counter()
        logger.debug("Started startup performance benchmark")
    
    def record_import_start(self, module_name: str) -> None:
        """Record the start of a module import."""
        self._import_start_times[module_name] = time.perf_counter()
    
    def record_import_end(self, module_name: str) -> None:
        """Record the end of a module import."""
        if module_name in self._import_start_times:
            import_time = time.perf_counter() - self._import_start_times[module_name]
            self.record_metric(f"import_{module_name}", import_time, "seconds")
            del self._import_start_times[module_name]
    
    def record_initialization_time(self, component_name: str, duration: float) -> None:
        """Record component initialization time."""
        self._initialization_times[component_name] = duration
        self.record_metric(f"init_{component_name}", duration, "seconds")
    
    def record_metric(self, name: str, value: float, unit: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record a performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            metadata=metadata or {}
        )
        self.metrics.append(metric)
        logger.debug(f"Recorded metric: {name}={value}{unit}")
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self._process.memory_info().rss / 1024 / 1024
    
    def get_startup_benchmark(self) -> Optional[StartupBenchmark]:
        """Get startup benchmark results."""
        if self._startup_start_time is None:
            return None
        
        total_time = time.perf_counter() - self._startup_start_time
        memory_usage = self.get_memory_usage()
        lazy_components = get_lazy_registry_status()
        
        # Extract import times
        import_times = {}
        for metric in self.metrics:
            if metric.name.startswith("import_"):
                module_name = metric.name[7:]  # Remove "import_" prefix
                import_times[module_name] = metric.value
        
        return StartupBenchmark(
            total_time=total_time,
            memory_usage=memory_usage,
            lazy_components_loaded=lazy_components,
            import_times=import_times,
            initialization_times=self._initialization_times.copy(),
        )
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all collected metrics."""
        if not self.metrics:
            return {}
        
        summary = {
            "total_metrics": len(self.metrics),
            "memory_usage_mb": self.get_memory_usage(),
            "metrics_by_type": {},
            "recent_metrics": []
        }
        
        # Group metrics by type
        for metric in self.metrics:
            metric_type = metric.name.split("_")[0]
            if metric_type not in summary["metrics_by_type"]:
                summary["metrics_by_type"][metric_type] = []
            summary["metrics_by_type"][metric_type].append({
                "name": metric.name,
                "value": metric.value,
                "unit": metric.unit,
                "timestamp": metric.timestamp.isoformat()
            })
        
        # Get recent metrics (last 10)
        recent_metrics = sorted(self.metrics, key=lambda m: m.timestamp, reverse=True)[:10]
        summary["recent_metrics"] = [
            {
                "name": m.name,
                "value": m.value,
                "unit": m.unit,
                "timestamp": m.timestamp.isoformat()
            }
            for m in recent_metrics
        ]
        
        return summary
    
    def clear_metrics(self) -> None:
        """Clear all collected metrics."""
        self.metrics.clear()
        self._import_start_times.clear()
        self._initialization_times.clear()
        self._startup_start_time = None
        logger.debug("Cleared all performance metrics")


# Global performance monitor instance
_performance_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return _performance_monitor


@contextmanager
def measure_time(operation_name: str, record_metric: bool = True):
    """Context manager to measure operation time.
    
    Args:
        operation_name: Name of the operation being measured
        record_metric: Whether to record the metric in the global monitor
        
    Yields:
        Dictionary with timing information
    """
    start_time = time.perf_counter()
    start_memory = _performance_monitor.get_memory_usage()
    
    timing_info = {"start_time": start_time}
    
    try:
        yield timing_info
    finally:
        end_time = time.perf_counter()
        end_memory = _performance_monitor.get_memory_usage()
        
        duration = end_time - start_time
        memory_delta = end_memory - start_memory
        
        timing_info.update({
            "end_time": end_time,
            "duration": duration,
            "memory_start_mb": start_memory,
            "memory_end_mb": end_memory,
            "memory_delta_mb": memory_delta
        })
        
        if record_metric:
            _performance_monitor.record_metric(
                f"time_{operation_name}", 
                duration, 
                "seconds",
                {"memory_delta_mb": memory_delta}
            )
        
        logger.debug(f"{operation_name} took {duration:.3f}s (memory delta: {memory_delta:+.1f}MB)")


def benchmark_startup() -> StartupBenchmark:
    """Run a complete startup benchmark.
    
    This function simulates a full system startup and measures performance.
    """
    monitor = get_performance_monitor()
    monitor.clear_metrics()
    monitor.start_startup_benchmark()
    
    with measure_time("full_startup"):
        # Simulate system initialization
        with measure_time("config_loading"):
            from . import runtime_config
            # Force config loading
            config = runtime_config.Config.load()
        
        with measure_time("role_registry"):
            from .roles import registry
            # Force role registry loading
            role_registry = registry.RoleRegistry()
            _ = role_registry.ROLE_CLASSES
        
        with measure_time("agent_manager"):
            from . import agent_manager
            # Create agent manager (this will create storage lazily)
            manager = agent_manager.AgentManager(config)
        
        with measure_time("tools_loading"):
            from . import tools
            # Force tool loading
            _ = tools.get_tool_registry()
        
        # Force garbage collection to clean up any temporary objects
        gc.collect()
    
    return monitor.get_startup_benchmark()


def benchmark_lazy_loading_effectiveness() -> Dict[str, Any]:
    """Benchmark the effectiveness of lazy loading.
    
    Returns:
        Dictionary with before/after metrics comparing eager vs lazy loading
    """
    results = {
        "lazy_loading": {},
        "eager_loading": {},
        "improvement": {}
    }
    
    # Test lazy loading (current implementation)
    monitor = get_performance_monitor()
    monitor.clear_metrics()
    
    with measure_time("lazy_startup", record_metric=False) as lazy_timing:
        # Import modules without forcing lazy loading
        from . import runtime_config
        from .roles import registry  
        from . import agent_manager
        from . import tools
        
        # Just create instances without forcing loading
        config = runtime_config.Config()
        role_registry = registry.RoleRegistry()
        
        gc.collect()
    
    results["lazy_loading"] = {
        "startup_time": lazy_timing["duration"],
        "memory_usage": lazy_timing["memory_end_mb"],
        "memory_delta": lazy_timing["memory_delta_mb"],
        "components_loaded": sum(get_lazy_registry_status().values())
    }
    
    # Test eager loading simulation
    monitor.clear_metrics()
    
    with measure_time("eager_startup", record_metric=False) as eager_timing:
        # Force loading of all components
        benchmark = benchmark_startup()
        gc.collect()
    
    results["eager_loading"] = {
        "startup_time": eager_timing["duration"],
        "memory_usage": eager_timing["memory_end_mb"], 
        "memory_delta": eager_timing["memory_delta_mb"],
        "components_loaded": len(benchmark.lazy_components_loaded)
    }
    
    # Calculate improvements
    lazy_time = results["lazy_loading"]["startup_time"]
    eager_time = results["eager_loading"]["startup_time"]
    lazy_memory = results["lazy_loading"]["memory_usage"]
    eager_memory = results["eager_loading"]["memory_usage"]
    
    results["improvement"] = {
        "startup_time_reduction_percent": ((eager_time - lazy_time) / eager_time * 100) if eager_time > 0 else 0,
        "memory_usage_reduction_percent": ((eager_memory - lazy_memory) / eager_memory * 100) if eager_memory > 0 else 0,
        "startup_time_ratio": lazy_time / eager_time if eager_time > 0 else 1,
        "memory_usage_ratio": lazy_memory / eager_memory if eager_memory > 0 else 1
    }
    
    return results


def performance_decorator(operation_name: str = None):
    """Decorator to automatically measure function performance.
    
    Args:
        operation_name: Name for the operation (defaults to function name)
    """
    def decorator(func: Callable) -> Callable:
        name = operation_name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with measure_time(name):
                return func(*args, **kwargs)
        return wrapper
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            with measure_time(name):
                return await func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


def log_startup_performance():
    """Log startup performance summary."""
    benchmark = _performance_monitor.get_startup_benchmark()
    if not benchmark:
        logger.warning("No startup benchmark data available")
        return
    
    logger.info(f"Startup Performance Summary:")
    logger.info(f"  Total startup time: {benchmark.total_time:.3f}s")
    logger.info(f"  Memory usage: {benchmark.memory_usage:.1f}MB")
    logger.info(f"  Lazy components loaded: {sum(benchmark.lazy_components_loaded.values())}/{len(benchmark.lazy_components_loaded)}")
    
    if benchmark.import_times:
        logger.info("  Import times:")
        for module, time_taken in sorted(benchmark.import_times.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"    {module}: {time_taken:.3f}s")
    
    if benchmark.initialization_times:
        logger.info("  Initialization times:")
        for component, time_taken in sorted(benchmark.initialization_times.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"    {component}: {time_taken:.3f}s")


# Auto-start startup benchmark if environment variable is set
if os.getenv("AGENTSMCP_BENCHMARK_STARTUP", "").lower() in ("1", "true", "yes"):
    _performance_monitor.start_startup_benchmark()
    # Register cleanup to log results at exit
    import atexit
    atexit.register(log_startup_performance)