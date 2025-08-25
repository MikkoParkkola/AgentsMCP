"""Health check system for AgentsMCP discovery service.

Provides comprehensive health monitoring including configuration validation,
dependency checking, circuit breaker status, and service availability.
"""

from __future__ import annotations

import os
import importlib
import time
import threading
import logging
from typing import Any, Dict, List, Optional, Callable, Tuple

from .exceptions import ConfigError, DependencyError, ServiceUnavailableError
from .circuit_breaker import CircuitBreaker
from .cache import cache_stats

logger = logging.getLogger("agentsmcp.health")

# Global registry of circuit breakers for health monitoring
_circuit_breakers: Dict[str, CircuitBreaker] = {}
_health_checks: Dict[str, Callable[[], Tuple[bool, str]]] = {}
_health_lock = threading.RLock()


def register_circuit_breaker(name: str, breaker: CircuitBreaker) -> None:
    """Register a circuit breaker for health monitoring."""
    with _health_lock:
        _circuit_breakers[name] = breaker
        logger.debug("Registered circuit breaker: %s", name)


def register_health_check(name: str, check_func: Callable[[], Tuple[bool, str]]) -> None:
    """Register a custom health check function.
    
    Args:
        name: Unique name for the health check
        check_func: Function that returns (is_healthy, details)
    """
    with _health_lock:
        _health_checks[name] = check_func
        logger.debug("Registered health check: %s", name)


def check_env_var(name: str, required: bool = True) -> Tuple[bool, str]:
    """Check if environment variable is set.
    
    Args:
        name: Environment variable name
        required: Whether the variable is required
        
    Returns:
        (is_healthy, details_message)
    """
    value = os.getenv(name)
    if value:
        # Don't log sensitive values, just confirm presence
        return True, f"present ({'required' if required else 'optional'})"
    elif required:
        return False, "missing (required)"
    else:
        return True, "missing (optional)"


def check_import(module_name: str, required: bool = True) -> Tuple[bool, str]:
    """Check if a Python module can be imported.
    
    Args:
        module_name: Name of module to import
        required: Whether the module is required
        
    Returns:
        (is_healthy, details_message)
    """
    try:
        importlib.import_module(module_name)
        return True, f"imported ({'required' if required else 'optional'})"
    except ImportError as exc:
        if required:
            return False, f"cannot import (required): {exc}"
        else:
            return True, f"cannot import (optional): {exc}"
    except Exception as exc:
        return False, f"import error: {exc}"


def check_file_access(file_path: str, required: bool = True) -> Tuple[bool, str]:
    """Check if file exists and is readable.
    
    Args:
        file_path: Path to file to check
        required: Whether the file is required
        
    Returns:
        (is_healthy, details_message)
    """
    try:
        if os.path.exists(file_path):
            if os.access(file_path, os.R_OK):
                size = os.path.getsize(file_path)
                return True, f"accessible ({size} bytes)"
            else:
                return False, "exists but not readable"
        elif required:
            return False, "missing (required)"
        else:
            return True, "missing (optional)"
    except Exception as exc:
        return False, f"access error: {exc}"


def check_directory_writable(dir_path: str, required: bool = True) -> Tuple[bool, str]:
    """Check if directory exists and is writable.
    
    Args:
        dir_path: Path to directory to check
        required: Whether the directory is required
        
    Returns:
        (is_healthy, details_message)  
    """
    try:
        if os.path.exists(dir_path):
            if os.path.isdir(dir_path):
                if os.access(dir_path, os.W_OK):
                    return True, "writable"
                else:
                    return False, "exists but not writable"
            else:
                return False, "exists but is not a directory"
        elif required:
            return False, "missing (required)"
        else:
            return True, "missing (optional)"
    except Exception as exc:
        return False, f"access error: {exc}"


def circuit_breaker_health(breaker: CircuitBreaker) -> Tuple[bool, Dict[str, Any]]:
    """Get health status of a circuit breaker.
    
    Returns:
        (is_healthy, detailed_status)
    """
    status = breaker.status()
    is_healthy = status["state"] == "CLOSED"
    return is_healthy, status


def run_health_checks() -> Tuple[bool, Dict[str, Any]]:
    """Execute all registered health checks and return overall status.
    
    Returns:
        (overall_healthy, detailed_report)
    """
    start_time = time.time()
    report: Dict[str, Any] = {"checks": {}}
    overall_healthy = True

    # Configuration checks
    try:
        # Check required environment variables
        for env_var, required in [
            ("JWT_SECRET", True),
            ("AGENTSMCP_CONFIG", False),
            ("AGENTSMCP_CACHE_DIR", False)
        ]:
            ok, details = check_env_var(env_var, required)
            report["checks"][f"env:{env_var}"] = {"ok": ok, "details": details}
            if required and not ok:
                overall_healthy = False

        # Check optional dependencies
        for module, required in [
            ("prometheus_client", False),
            ("uvicorn", False), 
            ("fastapi", False),
            ("aiohttp", False),
            ("websockets", False),
            ("cryptography", True),
            ("jose", True)
        ]:
            ok, details = check_import(module, required)
            report["checks"][f"import:{module}"] = {"ok": ok, "details": details}
            if required and not ok:
                overall_healthy = False

        # Check file system access
        cache_dir = os.getenv("AGENTSMCP_CACHE_DIR", "/tmp/agentsmcp-cache")
        ok, details = check_directory_writable(cache_dir, required=False)
        report["checks"]["cache_dir_writable"] = {"ok": ok, "details": details}

    except Exception as exc:
        logger.error("Error during configuration checks: %s", exc)
        report["checks"]["config_check_error"] = {"ok": False, "details": str(exc)}
        overall_healthy = False

    # Circuit breaker status
    try:
        with _health_lock:
            breaker_statuses = {}
            for name, breaker in _circuit_breakers.items():
                is_healthy, status = circuit_breaker_health(breaker)
                breaker_statuses[name] = status
                if not is_healthy:
                    overall_healthy = False

            report["circuit_breakers"] = breaker_statuses
            report["checks"]["circuit_breakers"] = {
                "ok": len([s for s in breaker_statuses.values() if s.get("state") == "CLOSED"]) == len(breaker_statuses),
                "count": len(breaker_statuses)
            }

    except Exception as exc:
        logger.error("Error during circuit breaker checks: %s", exc)
        report["checks"]["circuit_breaker_error"] = {"ok": False, "details": str(exc)}
        overall_healthy = False

    # Custom health checks
    try:
        with _health_lock:
            for name, check_func in _health_checks.items():
                try:
                    ok, details = check_func()
                    report["checks"][f"custom:{name}"] = {"ok": ok, "details": details}
                    if not ok:
                        overall_healthy = False
                except Exception as exc:
                    logger.error("Custom health check '%s' failed: %s", name, exc)
                    report["checks"][f"custom:{name}"] = {"ok": False, "details": f"check failed: {exc}"}
                    overall_healthy = False

    except Exception as exc:
        logger.error("Error during custom health checks: %s", exc)
        report["checks"]["custom_check_error"] = {"ok": False, "details": str(exc)}
        overall_healthy = False

    # Cache statistics
    try:
        stats = cache_stats()
        report["cache_stats"] = stats
        
        # Consider cache healthy if it exists and has reasonable size
        cache_healthy = (
            stats.get("total_files", 0) < 10000 and  # Not too many files
            stats.get("total_size_bytes", 0) < 100 * 1024 * 1024  # Less than 100MB
        )
        report["checks"]["cache_health"] = {"ok": cache_healthy, "details": f"{stats.get('total_files', 0)} files"}

    except Exception as exc:
        logger.error("Error getting cache statistics: %s", exc)
        report["checks"]["cache_stats_error"] = {"ok": False, "details": str(exc)}

    # Add metadata
    report.update({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "service": "agentsmcp-discovery",
        "duration_ms": round((time.time() - start_time) * 1000, 2),
        "overall_healthy": overall_healthy
    })

    return overall_healthy, report


def get_health_summary() -> Dict[str, Any]:
    """Get a simplified health summary suitable for monitoring dashboards."""
    overall_healthy, detailed_report = run_health_checks()
    
    # Count different types of checks
    checks = detailed_report.get("checks", {})
    total_checks = len(checks)
    passing_checks = len([c for c in checks.values() if c.get("ok", False)])
    failing_checks = total_checks - passing_checks
    
    # Circuit breaker summary
    breakers = detailed_report.get("circuit_breakers", {})
    open_breakers = len([b for b in breakers.values() if b.get("state") == "OPEN"])
    
    return {
        "healthy": overall_healthy,
        "timestamp": detailed_report.get("timestamp"),
        "duration_ms": detailed_report.get("duration_ms"),
        "checks": {
            "total": total_checks,
            "passing": passing_checks,
            "failing": failing_checks
        },
        "circuit_breakers": {
            "total": len(breakers),
            "open": open_breakers,
            "closed": len(breakers) - open_breakers
        },
        "cache": detailed_report.get("cache_stats", {})
    }