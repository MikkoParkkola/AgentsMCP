"""Cache implementation for graceful degradation in AgentsMCP discovery system.

Provides file-based caching with TTL support for fallback when services are unavailable.
"""

from __future__ import annotations

import json
import os
import pathlib
import time
import threading
import logging
from typing import Any, Optional, Dict

logger = logging.getLogger("agentsmcp.cache")

# Default cache directory - can be overridden via environment variable
CACHE_ROOT = pathlib.Path(os.getenv("AGENTSMCP_CACHE_DIR", "/tmp/agentsmcp-cache"))
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

# Thread safety for cache operations
_cache_lock = threading.RLock()


def _cache_file(key: str) -> pathlib.Path:
    """Generate safe cache file path for given key."""
    # Replace problematic characters with safe alternatives
    safe_key = key.replace("/", "_").replace(":", "_").replace("?", "_").replace("=", "_")
    # Limit filename length
    if len(safe_key) > 200:
        safe_key = safe_key[:200]
    return CACHE_ROOT / f"{safe_key}.json"


def write_cache(key: str, payload: dict[str, Any], ttl: int = 300) -> None:
    """Write data to cache with TTL.
    
    Args:
        key: Unique cache key
        payload: Data to cache (must be JSON serializable)
        ttl: Time-to-live in seconds
    """
    try:
        with _cache_lock:
            data = {
                "ts": time.time(),
                "ttl": ttl,
                "payload": payload,
                "key": key  # Store original key for debugging
            }
            
            cache_file = _cache_file(key)
            # Ensure parent directory exists
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write atomically by using a temporary file
            temp_file = cache_file.with_suffix('.tmp')
            temp_file.write_text(json.dumps(data, indent=2))
            temp_file.replace(cache_file)
            
            logger.debug("Cached data for key '%s' with TTL %ds", key, ttl)
            
    except Exception as exc:
        logger.warning("Failed to write cache for key '%s': %s", key, exc)


def read_cache(key: str, max_age: Optional[int] = None) -> Optional[dict[str, Any]]:
    """Read data from cache if still valid.
    
    Args:
        key: Cache key to read
        max_age: Override TTL with maximum age in seconds
        
    Returns:
        Cached payload if valid, None if expired or missing
    """
    try:
        with _cache_lock:
            cache_file = _cache_file(key)
            if not cache_file.exists():
                logger.debug("Cache miss for key '%s' - file not found", key)
                return None
            
            try:
                data = json.loads(cache_file.read_text())
            except json.JSONDecodeError as exc:
                logger.warning("Cache corruption for key '%s': %s", key, exc)
                # Remove corrupted cache file
                cache_file.unlink(missing_ok=True)
                return None
            
            # Validate cache structure
            required_fields = {"ts", "ttl", "payload"}
            if not all(field in data for field in required_fields):
                logger.warning("Invalid cache structure for key '%s'", key)
                cache_file.unlink(missing_ok=True)
                return None
            
            # Check if expired
            age = time.time() - data["ts"]
            ttl = min(data["ttl"], max_age) if max_age is not None else data["ttl"]
            
            if age > ttl:
                logger.debug("Cache expired for key '%s' (age=%.1fs, ttl=%ds)", key, age, ttl)
                # Optionally delete expired cache
                cache_file.unlink(missing_ok=True)
                return None
            
            logger.debug("Cache hit for key '%s' (age=%.1fs)", key, age)
            return data["payload"]
            
    except Exception as exc:
        logger.warning("Failed to read cache for key '%s': %s", key, exc)
        return None


def delete_cache(key: str) -> bool:
    """Delete cache entry for given key.
    
    Returns:
        True if cache was deleted, False if it didn't exist
    """
    try:
        with _cache_lock:
            cache_file = _cache_file(key)
            if cache_file.exists():
                cache_file.unlink()
                logger.debug("Deleted cache for key '%s'", key)
                return True
            return False
    except Exception as exc:
        logger.warning("Failed to delete cache for key '%s': %s", key, exc)
        return False


def clear_cache() -> int:
    """Clear all cache files.
    
    Returns:
        Number of files deleted
    """
    try:
        with _cache_lock:
            if not CACHE_ROOT.exists():
                return 0
            
            deleted = 0
            for cache_file in CACHE_ROOT.glob("*.json"):
                try:
                    cache_file.unlink()
                    deleted += 1
                except Exception as exc:
                    logger.warning("Failed to delete cache file %s: %s", cache_file, exc)
            
            logger.info("Cleared %d cache files", deleted)
            return deleted
    except Exception as exc:
        logger.error("Failed to clear cache: %s", exc)
        return 0


def cleanup_expired_cache(max_age: Optional[int] = None) -> int:
    """Remove expired cache entries.
    
    Args:
        max_age: Override TTL - remove entries older than this many seconds
        
    Returns:
        Number of expired entries removed
    """
    try:
        with _cache_lock:
            if not CACHE_ROOT.exists():
                return 0
            
            removed = 0
            current_time = time.time()
            
            for cache_file in CACHE_ROOT.glob("*.json"):
                try:
                    data = json.loads(cache_file.read_text())
                    
                    # Check if cache entry has expired
                    age = current_time - data.get("ts", 0)
                    ttl = min(data.get("ttl", 0), max_age) if max_age is not None else data.get("ttl", 0)
                    
                    if age > ttl:
                        cache_file.unlink()
                        removed += 1
                        
                except (json.JSONDecodeError, KeyError, OSError) as exc:
                    # Remove corrupted or inaccessible cache files
                    logger.debug("Removing corrupted cache file %s: %s", cache_file, exc)
                    try:
                        cache_file.unlink()
                        removed += 1
                    except OSError:
                        pass
            
            if removed > 0:
                logger.info("Cleaned up %d expired cache entries", removed)
            return removed
            
    except Exception as exc:
        logger.error("Failed to cleanup expired cache: %s", exc)
        return 0


def cache_stats() -> dict[str, Any]:
    """Get cache statistics.
    
    Returns:
        Dictionary with cache statistics
    """
    try:
        with _cache_lock:
            if not CACHE_ROOT.exists():
                return {
                    "total_files": 0,
                    "total_size_bytes": 0,
                    "cache_dir": str(CACHE_ROOT),
                    "expired_count": 0
                }
            
            total_files = 0
            total_size = 0
            expired_count = 0
            current_time = time.time()
            
            for cache_file in CACHE_ROOT.glob("*.json"):
                try:
                    total_files += 1
                    total_size += cache_file.stat().st_size
                    
                    # Check if expired
                    data = json.loads(cache_file.read_text())
                    age = current_time - data.get("ts", 0)
                    ttl = data.get("ttl", 0)
                    
                    if age > ttl:
                        expired_count += 1
                        
                except (json.JSONDecodeError, OSError):
                    # Count corrupted files as expired
                    expired_count += 1
            
            return {
                "total_files": total_files,
                "total_size_bytes": total_size,
                "cache_dir": str(CACHE_ROOT),
                "expired_count": expired_count,
                "valid_count": total_files - expired_count
            }
    except Exception as exc:
        logger.error("Failed to get cache stats: %s", exc)
        return {
            "total_files": 0,
            "total_size_bytes": 0,
            "cache_dir": str(CACHE_ROOT),
            "expired_count": 0,
            "error": str(exc)
        }