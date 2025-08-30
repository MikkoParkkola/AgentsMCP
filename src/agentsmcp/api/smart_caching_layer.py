"""
Smart Caching Layer for Sub-100ms Response Times

High-performance caching system with intelligent cache strategies,
automatic invalidation, and predictive prefetching for optimal performance.
"""

import asyncio
import hashlib
import json
import time
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import weakref
from collections import OrderedDict, defaultdict
import pickle
import zlib

from .base import APIBase, APIResponse, APIError


class CacheStrategy(str, Enum):
    """Cache eviction strategies."""
    LRU = "lru"              # Least Recently Used
    LFU = "lfu"              # Least Frequently Used  
    TTL = "ttl"              # Time To Live
    ADAPTIVE = "adaptive"    # Adaptive strategy based on usage patterns
    PRIORITY = "priority"    # Priority-based eviction


class CacheStatus(str, Enum):
    """Cache operation status."""
    HIT = "hit"
    MISS = "miss"
    STALE = "stale"
    EXPIRED = "expired"
    EVICTED = "evicted"
    UPDATED = "updated"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: Optional[float] = None
    priority: int = 0
    size_bytes: int = 0
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl_seconds is None:
            return False
        return (datetime.utcnow() - self.created_at).total_seconds() > self.ttl_seconds
    
    def is_stale(self, staleness_threshold: float = 300) -> bool:
        """Check if cache entry is stale (default 5 minutes)."""
        return (datetime.utcnow() - self.last_accessed).total_seconds() > staleness_threshold
    
    def update_access(self):
        """Update access metadata."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1


@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    hit_count: int = 0
    miss_count: int = 0
    eviction_count: int = 0
    total_size_bytes: int = 0
    entry_count: int = 0
    average_access_time_ms: float = 0.0
    hit_ratio: float = 0.0
    cache_utilization: float = 0.0


class SmartCachingLayer(APIBase):
    """High-performance smart caching layer with sub-100ms response times."""
    
    def __init__(
        self,
        max_size_mb: int = 512,
        default_ttl: int = 3600,
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    ):
        super().__init__("smart_caching_layer")
        
        # Cache configuration
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.strategy = strategy
        
        # Cache storage
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.metrics = CacheMetrics()
        
        # Cache indexes for efficient lookups
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)  # tag -> keys
        self.dependency_index: Dict[str, Set[str]] = defaultdict(set)  # dependency -> keys
        self.priority_queue: List[str] = []  # For priority-based eviction
        
        # Prefetch system
        self.prefetch_patterns: Dict[str, List[str]] = {}  # pattern -> keys to prefetch
        self.access_patterns: Dict[str, List[Tuple[str, datetime]]] = defaultdict(list)
        
        # Performance optimization
        self.compression_enabled = True
        self.compression_threshold = 1024  # Compress entries > 1KB
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        
        # Initialize caching system
        asyncio.create_task(self._initialize_caching_system())
    
    async def _initialize_caching_system(self):
        """Initialize the smart caching system."""
        # Start background tasks
        self.background_tasks.add(
            asyncio.create_task(self._periodic_cleanup())
        )
        self.background_tasks.add(
            asyncio.create_task(self._prefetch_predictor())
        )
        self.background_tasks.add(
            asyncio.create_task(self._metrics_updater())
        )
        self.background_tasks.add(
            asyncio.create_task(self._adaptive_tuning())
        )
        
        self.logger.info(f"Smart caching system initialized with {self.max_size_bytes // (1024*1024)}MB max size")
    
    def _generate_cache_key(self, prefix: str, data: Any) -> str:
        """Generate a consistent cache key from data."""
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
        else:
            data_str = str(data)
        
        hash_value = hashlib.sha256(data_str.encode()).hexdigest()[:16]
        return f"{prefix}:{hash_value}"
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage with optional compression."""
        serialized = pickle.dumps(value)
        
        if self.compression_enabled and len(serialized) > self.compression_threshold:
            compressed = zlib.compress(serialized)
            # Only use compression if it actually reduces size
            if len(compressed) < len(serialized):
                return b'COMPRESSED:' + compressed
        
        return serialized
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage with decompression support."""
        if data.startswith(b'COMPRESSED:'):
            compressed_data = data[11:]  # Remove 'COMPRESSED:' prefix
            serialized = zlib.decompress(compressed_data)
        else:
            serialized = data
        
        return pickle.loads(serialized)
    
    def _calculate_entry_size(self, value: Any) -> int:
        """Calculate approximate size of cache entry in bytes."""
        try:
            serialized = self._serialize_value(value)
            return len(serialized)
        except Exception:
            # Fallback to string representation size
            return len(str(value).encode())
    
    async def get(
        self,
        key: str,
        default: Any = None,
        update_access: bool = True
    ) -> APIResponse:
        """Get value from cache with performance tracking."""
        start_time = time.time()
        
        try:
            entry = self.cache.get(key)
            
            if entry is None:
                # Cache miss
                self.metrics.miss_count += 1
                self._update_hit_ratio()
                
                return APIResponse(
                    status="success",
                    data=default,
                    metadata={
                        "cache_status": CacheStatus.MISS.value,
                        "access_time_ms": (time.time() - start_time) * 1000
                    }
                )
            
            # Check if entry is expired
            if entry.is_expired():
                await self._evict_entry(key, "expired")
                self.metrics.miss_count += 1
                self._update_hit_ratio()
                
                return APIResponse(
                    status="success",
                    data=default,
                    metadata={
                        "cache_status": CacheStatus.EXPIRED.value,
                        "access_time_ms": (time.time() - start_time) * 1000
                    }
                )
            
            # Cache hit - update access metadata
            if update_access:
                entry.update_access()
                # Move to end for LRU strategy
                if self.strategy == CacheStrategy.LRU:
                    self.cache.move_to_end(key)
            
            # Deserialize value
            try:
                value = self._deserialize_value(entry.value)
            except Exception as e:
                self.logger.error(f"Cache deserialization failed for key {key}: {e}")
                await self._evict_entry(key, "corruption")
                self.metrics.miss_count += 1
                self._update_hit_ratio()
                
                return APIResponse(
                    status="success",
                    data=default,
                    metadata={
                        "cache_status": CacheStatus.MISS.value,
                        "error": "deserialization_failed",
                        "access_time_ms": (time.time() - start_time) * 1000
                    }
                )
            
            self.metrics.hit_count += 1
            self._update_hit_ratio()
            
            # Track access pattern for prefetching
            self._track_access_pattern(key)
            
            access_time_ms = (time.time() - start_time) * 1000
            self.metrics.average_access_time_ms = (
                self.metrics.average_access_time_ms * 0.9 + access_time_ms * 0.1
            )
            
            # Check if entry is stale
            status = CacheStatus.STALE.value if entry.is_stale() else CacheStatus.HIT.value
            
            return APIResponse(
                status="success",
                data=value,
                metadata={
                    "cache_status": status,
                    "access_time_ms": access_time_ms,
                    "access_count": entry.access_count,
                    "created_at": entry.created_at.isoformat()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Cache get operation failed: {e}")
            return APIResponse(
                status="error",
                error=str(e),
                metadata={
                    "cache_status": CacheStatus.MISS.value,
                    "access_time_ms": (time.time() - start_time) * 1000
                }
            )
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        priority: int = 0,
        tags: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None
    ) -> APIResponse:
        """Set value in cache with metadata."""
        return await self._execute_with_metrics(
            "set",
            self._set_internal,
            key,
            value,
            ttl,
            priority,
            tags or [],
            dependencies or []
        )
    
    async def _set_internal(
        self,
        key: str,
        value: Any,
        ttl: Optional[int],
        priority: int,
        tags: List[str],
        dependencies: List[str]
    ) -> Dict[str, Any]:
        """Internal cache set operation."""
        # Serialize and calculate size
        try:
            serialized_value = self._serialize_value(value)
            entry_size = len(serialized_value)
        except Exception as e:
            raise APIError(f"Failed to serialize cache value: {e}", "SERIALIZATION_ERROR", 400)
        
        # Check if we need to make space
        await self._ensure_space(entry_size)
        
        # Remove existing entry if it exists
        if key in self.cache:
            await self._remove_from_indexes(key)
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=serialized_value,
            created_at=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
            access_count=0,
            ttl_seconds=ttl or self.default_ttl,
            priority=priority,
            size_bytes=entry_size,
            tags=tags,
            dependencies=dependencies
        )
        
        # Store in cache
        self.cache[key] = entry
        
        # Update indexes
        for tag in tags:
            self.tag_index[tag].add(key)
        
        for dep in dependencies:
            self.dependency_index[dep].add(key)
        
        # Update priority queue if using priority strategy
        if self.strategy == CacheStrategy.PRIORITY:
            self._update_priority_queue(key, priority)
        
        # Update metrics
        self.metrics.total_size_bytes += entry_size
        self.metrics.entry_count += 1
        
        return {
            "key": key,
            "size_bytes": entry_size,
            "ttl_seconds": entry.ttl_seconds,
            "priority": priority,
            "tags": tags,
            "compressed": serialized_value.startswith(b'COMPRESSED:')
        }
    
    async def _ensure_space(self, required_bytes: int):
        """Ensure enough space in cache for new entry."""
        # Check if we need to free up space
        while (self.metrics.total_size_bytes + required_bytes > self.max_size_bytes and 
               len(self.cache) > 0):
            
            # Find entry to evict based on strategy
            key_to_evict = await self._select_eviction_candidate()
            
            if key_to_evict:
                await self._evict_entry(key_to_evict, "space_needed")
            else:
                # No suitable candidate found, can't make space
                raise APIError("Cache is full and no evictable entries found", "CACHE_FULL", 507)
    
    async def _select_eviction_candidate(self) -> Optional[str]:
        """Select cache entry for eviction based on configured strategy."""
        if not self.cache:
            return None
        
        if self.strategy == CacheStrategy.LRU:
            # LRU: Remove least recently used (first item in OrderedDict)
            return next(iter(self.cache))
        
        elif self.strategy == CacheStrategy.LFU:
            # LFU: Remove least frequently used
            return min(self.cache.keys(), key=lambda k: self.cache[k].access_count)
        
        elif self.strategy == CacheStrategy.TTL:
            # TTL: Remove expired entries first, then oldest
            expired_keys = [k for k, entry in self.cache.items() if entry.is_expired()]
            if expired_keys:
                return expired_keys[0]
            return min(self.cache.keys(), key=lambda k: self.cache[k].created_at)
        
        elif self.strategy == CacheStrategy.PRIORITY:
            # Priority: Remove lowest priority entries first
            return min(self.cache.keys(), key=lambda k: (self.cache[k].priority, self.cache[k].created_at))
        
        elif self.strategy == CacheStrategy.ADAPTIVE:
            # Adaptive: Use machine learning-like scoring
            return await self._adaptive_eviction_candidate()
        
        else:
            # Default to LRU
            return next(iter(self.cache))
    
    async def _adaptive_eviction_candidate(self) -> Optional[str]:
        """Select eviction candidate using adaptive scoring."""
        if not self.cache:
            return None
        
        current_time = datetime.utcnow()
        scored_entries = []
        
        for key, entry in self.cache.items():
            # Calculate adaptive score (lower score = better candidate for eviction)
            score = 0.0
            
            # Time-based factors
            age_seconds = (current_time - entry.created_at).total_seconds()
            last_access_seconds = (current_time - entry.last_accessed).total_seconds()
            
            score += age_seconds * 0.1  # Older entries more likely to evict
            score += last_access_seconds * 0.3  # Recently accessed less likely to evict
            
            # Frequency factor
            score -= entry.access_count * 10  # Frequently accessed less likely to evict
            
            # Priority factor
            score -= entry.priority * 100  # High priority less likely to evict
            
            # Size factor (larger entries more expensive to keep)
            score += entry.size_bytes * 0.001
            
            # TTL factor
            if entry.ttl_seconds:
                remaining_ttl = entry.ttl_seconds - (current_time - entry.created_at).total_seconds()
                if remaining_ttl < 0:
                    score += 1000  # Expired entries very likely to evict
                else:
                    score += max(0, 100 - remaining_ttl)  # Soon to expire more likely to evict
            
            scored_entries.append((score, key))
        
        # Return key with highest score (best candidate for eviction)
        scored_entries.sort(reverse=True)
        return scored_entries[0][1] if scored_entries else None
    
    async def _evict_entry(self, key: str, reason: str):
        """Evict a cache entry and update metrics."""
        entry = self.cache.pop(key, None)
        if entry:
            # Remove from indexes
            await self._remove_from_indexes(key)
            
            # Update metrics
            self.metrics.total_size_bytes -= entry.size_bytes
            self.metrics.entry_count -= 1
            self.metrics.eviction_count += 1
            
            self.logger.debug(f"Evicted cache entry {key} (reason: {reason})")
    
    async def _remove_from_indexes(self, key: str):
        """Remove key from all indexes."""
        entry = self.cache.get(key)
        if not entry:
            return
        
        # Remove from tag index
        for tag in entry.tags:
            self.tag_index[tag].discard(key)
            if not self.tag_index[tag]:
                del self.tag_index[tag]
        
        # Remove from dependency index
        for dep in entry.dependencies:
            self.dependency_index[dep].discard(key)
            if not self.dependency_index[dep]:
                del self.dependency_index[dep]
    
    def _update_priority_queue(self, key: str, priority: int):
        """Update priority queue for priority-based eviction."""
        # Remove if exists
        if key in self.priority_queue:
            self.priority_queue.remove(key)
        
        # Insert at correct position
        inserted = False
        for i, existing_key in enumerate(self.priority_queue):
            if self.cache[existing_key].priority < priority:
                self.priority_queue.insert(i, key)
                inserted = True
                break
        
        if not inserted:
            self.priority_queue.append(key)
    
    def _update_hit_ratio(self):
        """Update cache hit ratio."""
        total_requests = self.metrics.hit_count + self.metrics.miss_count
        if total_requests > 0:
            self.metrics.hit_ratio = self.metrics.hit_count / total_requests
    
    def _track_access_pattern(self, key: str):
        """Track access pattern for predictive prefetching."""
        current_time = datetime.utcnow()
        
        # Add to access history
        self.access_patterns[key].append((key, current_time))
        
        # Keep only recent history (last 100 accesses)
        if len(self.access_patterns[key]) > 100:
            self.access_patterns[key] = self.access_patterns[key][-100:]
    
    async def delete(self, key: str) -> APIResponse:
        """Delete entry from cache."""
        return await self._execute_with_metrics(
            "delete",
            self._delete_internal,
            key
        )
    
    async def _delete_internal(self, key: str) -> Dict[str, Any]:
        """Internal cache delete operation."""
        entry = self.cache.get(key)
        if not entry:
            return {"key": key, "found": False}
        
        await self._evict_entry(key, "manual_delete")
        return {"key": key, "found": True, "size_freed": entry.size_bytes}
    
    async def clear(self, pattern: Optional[str] = None, tags: Optional[List[str]] = None) -> APIResponse:
        """Clear cache entries by pattern or tags."""
        return await self._execute_with_metrics(
            "clear",
            self._clear_internal,
            pattern,
            tags
        )
    
    async def _clear_internal(
        self, 
        pattern: Optional[str],
        tags: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Internal cache clear operation."""
        keys_to_delete = set()
        
        if pattern:
            # Simple pattern matching (could be enhanced with regex)
            keys_to_delete.update(k for k in self.cache.keys() if pattern in k)
        
        if tags:
            # Clear by tags
            for tag in tags:
                keys_to_delete.update(self.tag_index.get(tag, set()))
        
        if not pattern and not tags:
            # Clear everything
            keys_to_delete.update(self.cache.keys())
        
        cleared_count = 0
        freed_bytes = 0
        
        for key in keys_to_delete:
            entry = self.cache.get(key)
            if entry:
                freed_bytes += entry.size_bytes
                await self._evict_entry(key, "manual_clear")
                cleared_count += 1
        
        return {
            "cleared_count": cleared_count,
            "freed_bytes": freed_bytes,
            "pattern": pattern,
            "tags": tags
        }
    
    async def invalidate_dependencies(self, dependency: str) -> APIResponse:
        """Invalidate all cache entries that depend on a specific dependency."""
        return await self._execute_with_metrics(
            "invalidate_dependencies",
            self._invalidate_dependencies_internal,
            dependency
        )
    
    async def _invalidate_dependencies_internal(self, dependency: str) -> Dict[str, Any]:
        """Internal dependency invalidation logic."""
        dependent_keys = self.dependency_index.get(dependency, set()).copy()
        
        invalidated_count = 0
        freed_bytes = 0
        
        for key in dependent_keys:
            entry = self.cache.get(key)
            if entry:
                freed_bytes += entry.size_bytes
                await self._evict_entry(key, f"dependency_invalidated:{dependency}")
                invalidated_count += 1
        
        return {
            "dependency": dependency,
            "invalidated_count": invalidated_count,
            "freed_bytes": freed_bytes
        }
    
    async def prefetch(self, keys: List[str], loader: Callable[[str], Any]) -> APIResponse:
        """Prefetch cache entries using provided loader function."""
        return await self._execute_with_metrics(
            "prefetch",
            self._prefetch_internal,
            keys,
            loader
        )
    
    async def _prefetch_internal(
        self, 
        keys: List[str],
        loader: Callable[[str], Any]
    ) -> Dict[str, Any]:
        """Internal prefetch logic."""
        prefetched_count = 0
        failed_keys = []
        
        for key in keys:
            # Skip if already in cache
            if key in self.cache:
                continue
            
            try:
                # Load value
                if asyncio.iscoroutinefunction(loader):
                    value = await loader(key)
                else:
                    value = loader(key)
                
                # Cache the value
                await self._set_internal(key, value, None, 0, [], [])
                prefetched_count += 1
                
            except Exception as e:
                self.logger.warning(f"Prefetch failed for key {key}: {e}")
                failed_keys.append(key)
        
        return {
            "requested_keys": len(keys),
            "prefetched_count": prefetched_count,
            "failed_keys": failed_keys,
            "success_rate": prefetched_count / len(keys) if keys else 0.0
        }
    
    # Background tasks
    async def _periodic_cleanup(self):
        """Periodic cleanup of expired entries."""
        while True:
            try:
                current_time = datetime.utcnow()
                expired_keys = []
                
                # Find expired entries
                for key, entry in self.cache.items():
                    if entry.is_expired():
                        expired_keys.append(key)
                
                # Remove expired entries
                for key in expired_keys:
                    await self._evict_entry(key, "expired_cleanup")
                
                if expired_keys:
                    self.logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
                
                # Sleep for 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                self.logger.error(f"Cache cleanup error: {e}")
                await asyncio.sleep(300)
    
    async def _prefetch_predictor(self):
        """Predictive prefetching based on access patterns."""
        while True:
            try:
                # Analyze access patterns and predict next accesses
                # This is a simplified implementation - production would use ML
                
                for key, accesses in self.access_patterns.items():
                    if len(accesses) >= 5:  # Need some history
                        # Simple time-based prediction
                        recent_accesses = accesses[-5:]
                        time_intervals = []
                        
                        for i in range(1, len(recent_accesses)):
                            interval = (recent_accesses[i][1] - recent_accesses[i-1][1]).total_seconds()
                            time_intervals.append(interval)
                        
                        if time_intervals:
                            avg_interval = sum(time_intervals) / len(time_intervals)
                            
                            # If access pattern is regular, prepare for next access
                            if avg_interval < 3600:  # Less than 1 hour intervals
                                last_access = recent_accesses[-1][1]
                                next_predicted = last_access + timedelta(seconds=avg_interval)
                                
                                # If prediction is soon and entry not in cache, add to prefetch candidates
                                time_to_prediction = (next_predicted - datetime.utcnow()).total_seconds()
                                if 0 < time_to_prediction < 300 and key not in self.cache:  # 5 minutes ahead
                                    # This would trigger prefetch in a real implementation
                                    pass
                
                # Sleep for 10 minutes
                await asyncio.sleep(600)
                
            except Exception as e:
                self.logger.error(f"Prefetch prediction error: {e}")
                await asyncio.sleep(600)
    
    async def _metrics_updater(self):
        """Update cache metrics periodically."""
        while True:
            try:
                # Update utilization
                self.metrics.cache_utilization = (
                    self.metrics.total_size_bytes / self.max_size_bytes
                    if self.max_size_bytes > 0 else 0.0
                )
                
                # Sleep for 30 seconds
                await asyncio.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Metrics update error: {e}")
                await asyncio.sleep(30)
    
    async def _adaptive_tuning(self):
        """Adaptive tuning of cache parameters based on performance."""
        while True:
            try:
                # Analyze performance and adjust parameters
                if self.metrics.hit_ratio < 0.8 and len(self.cache) < self.max_size_bytes // 1024:
                    # Low hit ratio and space available - increase TTL
                    self.default_ttl = min(self.default_ttl * 1.1, 7200)  # Cap at 2 hours
                
                elif self.metrics.hit_ratio > 0.95 and self.metrics.cache_utilization > 0.9:
                    # High hit ratio but cache full - decrease TTL
                    self.default_ttl = max(self.default_ttl * 0.9, 300)  # Min 5 minutes
                
                # Sleep for 30 minutes
                await asyncio.sleep(1800)
                
            except Exception as e:
                self.logger.error(f"Adaptive tuning error: {e}")
                await asyncio.sleep(1800)
    
    async def get_metrics(self) -> APIResponse:
        """Get cache performance metrics."""
        return await self._execute_with_metrics(
            "get_metrics",
            self._get_metrics_internal
        )
    
    async def _get_metrics_internal(self) -> Dict[str, Any]:
        """Internal logic for getting cache metrics."""
        return {
            "performance": {
                "hit_count": self.metrics.hit_count,
                "miss_count": self.metrics.miss_count,
                "hit_ratio": self.metrics.hit_ratio,
                "average_access_time_ms": self.metrics.average_access_time_ms,
                "eviction_count": self.metrics.eviction_count
            },
            "storage": {
                "entry_count": self.metrics.entry_count,
                "total_size_bytes": self.metrics.total_size_bytes,
                "total_size_mb": self.metrics.total_size_bytes / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "cache_utilization": self.metrics.cache_utilization
            },
            "configuration": {
                "strategy": self.strategy.value,
                "default_ttl": self.default_ttl,
                "compression_enabled": self.compression_enabled,
                "compression_threshold": self.compression_threshold
            },
            "indexes": {
                "tag_count": len(self.tag_index),
                "dependency_count": len(self.dependency_index),
                "access_patterns_tracked": len(self.access_patterns)
            }
        }
    
    async def close(self):
        """Clean shutdown of caching layer."""
        self.logger.info("Shutting down caching layer...")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Clear cache
        self.cache.clear()
        self.tag_index.clear()
        self.dependency_index.clear()
        self.access_patterns.clear()
        
        self.logger.info("Caching layer shut down complete")