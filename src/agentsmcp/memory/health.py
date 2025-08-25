"""
Health check utilities for the memory subsystem.

`get_memory_health` returns a JSON‑serialisable snapshot of the current
status of all providers.  The function is intentionally lightweight to
avoid starving request handling during a critical period.

The metrics integration uses the `prometheus_client` library – a
production deployment should expose the `/metrics` endpoint via the
FastAPI application elsewhere in the project.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, List, Any

from .providers import MemoryProvider

logger = logging.getLogger(__name__)

# Optional Prometheus metrics - graceful degradation if not available
try:
    from prometheus_client import Gauge, CollectorRegistry
    HAS_PROMETHEUS = True
    
    # Use a dedicated registry to avoid collisions
    _memory_registry = CollectorRegistry()
    
    # Prometheus gauges - create only if not already registered
    try:
        MEMORY_PROVIDER_HEALTH = Gauge(
            "agentsmcp_memory_provider_health",
            "Health status per memory provider",
            labelnames=["provider"],
            registry=_memory_registry
        )
        MEMORY_CONTEXT_RETRIEVAL_MS = Gauge(
            "agentsmcp_context_retrieval_latency_ms",
            "Latency of the last context retrieval call (ms).",
            registry=_memory_registry
        )
    except ValueError as e:
        # Metrics already registered - get existing ones
        if "already been registered" in str(e):
            # In case of collision, disable metrics to prevent crashes
            logger.warning("Prometheus metrics already registered, disabling memory health metrics")
            HAS_PROMETHEUS = False
            MEMORY_PROVIDER_HEALTH = None
            MEMORY_CONTEXT_RETRIEVAL_MS = None
        else:
            raise
except ImportError:
    HAS_PROMETHEUS = False
    MEMORY_PROVIDER_HEALTH = None
    MEMORY_CONTEXT_RETRIEVAL_MS = None


async def get_memory_health(providers: List[MemoryProvider]) -> Dict[str, Any]:
    """
    Perform a synchronous health check on the supplied providers.

    Each provider's `health_check` method is awaited in parallel to keep
    the overall latency to a few milliseconds.  The result dictionary
    can be consumed by the web UI or a third‑party health monitor.
    
    Returns:
        Dict containing:
        - providers: Dict mapping provider name to health status
        - overall_healthy: Boolean indicating if all providers are healthy
        - check_duration_ms: Time taken to perform all health checks
    """
    start_time = time.time()
    results: Dict[str, bool] = {}
    
    if not providers:
        return {
            "providers": {},
            "overall_healthy": True,
            "check_duration_ms": 0.0,
        }
    
    tasks = {
        provider.__class__.__name__: asyncio.create_task(provider.health_check())
        for provider in providers
    }
    
    for name, task in tasks.items():
        try:
            status = await task
        except Exception as exc:
            logger.exception("Health check failed for provider %s", name)
            status = False

        results[name] = status
        
        # Update Prometheus metrics if available
        if HAS_PROMETHEUS and MEMORY_PROVIDER_HEALTH:
            MEMORY_PROVIDER_HEALTH.labels(provider=name).set(1 if status else 0)

    check_duration = (time.time() - start_time) * 1000  # Convert to milliseconds
    overall_healthy = all(results.values())
    
    return {
        "providers": results,
        "overall_healthy": overall_healthy,
        "check_duration_ms": check_duration,
    }


def record_context_retrieval_latency(latency_ms: float) -> None:
    """
    Record context retrieval latency for monitoring.
    
    Args:
        latency_ms: Latency in milliseconds
    """
    if HAS_PROMETHEUS and MEMORY_CONTEXT_RETRIEVAL_MS:
        MEMORY_CONTEXT_RETRIEVAL_MS.set(latency_ms)
    
    logger.debug("Context retrieval latency: %.2fms", latency_ms)


async def benchmark_context_operations(provider: MemoryProvider, test_data: Dict[str, Any] = None) -> Dict[str, float]:
    """
    Benchmark context operations to measure performance.
    
    Args:
        provider: The memory provider to benchmark
        test_data: Optional test data to use, defaults to a simple test context
        
    Returns:
        Dict containing latency measurements in milliseconds
    """
    if test_data is None:
        test_data = {
            "agent_type": "test",
            "session_id": "benchmark-session",
            "context": {"test": "data", "timestamp": time.time()},
            "metadata": {"benchmarked": True}
        }
    
    test_agent_id = "benchmark-agent"
    results = {}
    
    # Benchmark store operation
    start_time = time.time()
    await provider.store_context(test_agent_id, test_data)
    results["store_ms"] = (time.time() - start_time) * 1000
    
    # Benchmark load operation
    start_time = time.time()
    loaded_context = await provider.load_context(test_agent_id)
    results["load_ms"] = (time.time() - start_time) * 1000
    
    # Verify data integrity
    results["data_integrity"] = loaded_context == test_data
    
    # Benchmark delete operation
    start_time = time.time()
    await provider.delete_context(test_agent_id)
    results["delete_ms"] = (time.time() - start_time) * 1000
    
    # Record metrics
    record_context_retrieval_latency(results["load_ms"])
    
    return results