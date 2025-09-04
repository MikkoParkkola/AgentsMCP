"""Centralized log store for agent execution data.

This module provides a unified interface for storing, retrieving, and managing
agent execution logs with support for multiple storage backends, data retention
policies, and performance monitoring.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, AsyncGenerator, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from ..logging.log_schemas import (
    AgentEvent, EventType, EventSeverity, LoggingConfig, RetentionPolicy
)
from ..logging.storage_adapters import (
    StorageAdapter, FileStorageAdapter, DatabaseStorageAdapter, MemoryStorageAdapter
)
from ..logging.execution_log_capture import ExecutionLogCapture

logger = logging.getLogger(__name__)


class LogStoreStatus(Enum):
    """Status of the log store."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    DEGRADED = "degraded"
    SHUTDOWN = "shutdown"
    ERROR = "error"


@dataclass
class LogStoreStats:
    """Statistics for the log store."""
    
    # Storage statistics
    total_events_stored: int = 0
    total_events_retrieved: int = 0
    storage_errors: int = 0
    retrieval_errors: int = 0
    
    # Performance statistics
    avg_storage_latency_ms: float = 0.0
    avg_retrieval_latency_ms: float = 0.0
    p95_storage_latency_ms: float = 0.0
    p95_retrieval_latency_ms: float = 0.0
    
    # Capacity statistics
    storage_size_bytes: int = 0
    storage_utilization_percent: float = 0.0
    retention_violations: int = 0
    
    # Health statistics
    last_health_check: float = 0.0
    consecutive_failures: int = 0
    uptime_seconds: float = 0.0


class LogStore:
    """Centralized log store with multiple storage backends and advanced features.
    
    Provides a unified interface for all agent execution logging with:
    - Multiple storage backend support (file, database, memory)
    - Automatic failover and redundancy
    - Data retention and cleanup policies
    - Performance monitoring and alerting
    - Query optimization and caching
    """
    
    def __init__(
        self,
        config: LoggingConfig,
        primary_storage: Optional[StorageAdapter] = None,
        backup_storage: Optional[StorageAdapter] = None,
        enable_redundancy: bool = True
    ):
        """Initialize the log store.
        
        Args:
            config: Logging configuration
            primary_storage: Primary storage backend
            backup_storage: Optional backup storage backend
            enable_redundancy: Whether to enable storage redundancy
        """
        self.config = config
        self.enable_redundancy = enable_redundancy
        
        # Initialize storage backends
        self.primary_storage = primary_storage or self._create_default_storage()
        self.backup_storage = backup_storage if enable_redundancy else None
        self.active_storages: List[StorageAdapter] = []
        
        # Initialize execution log capture
        self.log_capture = ExecutionLogCapture(
            config=config,
            storage_adapter=self,  # LogStore acts as a storage adapter
            pii_sanitizer=None  # Will be initialized by ExecutionLogCapture
        )
        
        # State management
        self.status = LogStoreStatus.INITIALIZING
        self.stats = LogStoreStats()
        self.start_time = time.time()
        
        # Background tasks
        self.health_check_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self.metrics_task: Optional[asyncio.Task] = None
        
        # Event listeners and hooks
        self.event_listeners: List[Callable[[AgentEvent], None]] = []
        self.error_handlers: List[Callable[[Exception, str], None]] = []
        
        # Query caching
        self.query_cache: Dict[str, Any] = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Performance tracking
        self.latency_samples = []
        self.error_counts: Dict[str, int] = {}
    
    def _create_default_storage(self) -> StorageAdapter:
        """Create default storage adapter based on configuration."""
        if self.config.storage_backend == "database":
            return DatabaseStorageAdapter(
                db_path=self.config.storage_path or "logs/agent_execution.db",
                encryption_key=None,  # TODO: Add encryption key management
                compression_enabled=self.config.compression_enabled,
                retention_policy=self.config.retention_policy
            )
        elif self.config.storage_backend == "memory":
            return MemoryStorageAdapter(
                max_events=10000,
                retention_policy=self.config.retention_policy
            )
        else:  # Default to file storage
            return FileStorageAdapter(
                storage_path=self.config.storage_path or "logs/agent_execution",
                encryption_key=None,  # TODO: Add encryption key management
                compression_enabled=self.config.compression_enabled,
                retention_policy=self.config.retention_policy
            )
    
    async def initialize(self) -> None:
        """Initialize the log store and all storage backends."""
        logger.info("Initializing LogStore...")
        
        try:
            # Initialize primary storage
            await self.primary_storage.initialize()
            self.active_storages.append(self.primary_storage)
            
            # Initialize backup storage if enabled
            if self.backup_storage:
                try:
                    await self.backup_storage.initialize()
                    self.active_storages.append(self.backup_storage)
                    logger.info("Backup storage initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize backup storage: {e}")
                    if not self.primary_storage:
                        raise
            
            # Initialize execution log capture
            await self.log_capture.start()
            
            # Start background tasks
            self.health_check_task = asyncio.create_task(self._health_check_loop())
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            self.metrics_task = asyncio.create_task(self._metrics_loop())
            
            self.status = LogStoreStatus.ACTIVE
            self.stats.last_health_check = time.time()
            
            logger.info("LogStore initialized successfully")
            
        except Exception as e:
            self.status = LogStoreStatus.ERROR
            logger.error(f"Failed to initialize LogStore: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the log store and cleanup resources."""
        logger.info("Shutting down LogStore...")
        
        self.status = LogStoreStatus.SHUTDOWN
        
        # Cancel background tasks
        for task in [self.health_check_task, self.cleanup_task, self.metrics_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Stop execution log capture
        await self.log_capture.stop()
        
        # Close storage backends
        for storage in self.active_storages:
            try:
                await storage.close()
            except Exception as e:
                logger.error(f"Error closing storage backend: {e}")
        
        logger.info("LogStore shutdown complete")
    
    # Storage Adapter Interface (to integrate with ExecutionLogCapture)
    
    async def store_event(self, event: AgentEvent) -> bool:
        """Store a single event using all active storage backends."""
        if self.status != LogStoreStatus.ACTIVE:
            return False
        
        start_time = time.perf_counter()
        success_count = 0
        
        # Store to all active storages
        for storage in self.active_storages:
            try:
                if await storage.store_event(event):
                    success_count += 1
            except Exception as e:
                logger.error(f"Error storing event in {storage.__class__.__name__}: {e}")
                self._handle_storage_error(storage, e)
        
        # Track performance
        latency_ms = (time.perf_counter() - start_time) * 1000
        self.latency_samples.append(latency_ms)
        
        # Update statistics
        if success_count > 0:
            self.stats.total_events_stored += 1
            # Notify event listeners
            for listener in self.event_listeners:
                try:
                    listener(event)
                except Exception as e:
                    logger.error(f"Error in event listener: {e}")
        else:
            self.stats.storage_errors += 1
        
        return success_count > 0
    
    async def store_batch(self, events: List[AgentEvent]) -> bool:
        """Store a batch of events efficiently."""
        if self.status != LogStoreStatus.ACTIVE or not events:
            return False
        
        start_time = time.perf_counter()
        success_count = 0
        
        # Store to all active storages
        for storage in self.active_storages:
            try:
                if await storage.store_batch(events):
                    success_count += 1
            except Exception as e:
                logger.error(f"Error storing batch in {storage.__class__.__name__}: {e}")
                self._handle_storage_error(storage, e)
        
        # Track performance
        latency_ms = (time.perf_counter() - start_time) * 1000
        self.latency_samples.extend([latency_ms] * len(events))
        
        # Update statistics
        if success_count > 0:
            self.stats.total_events_stored += len(events)
            # Notify event listeners for each event
            for event in events:
                for listener in self.event_listeners:
                    try:
                        listener(event)
                    except Exception as e:
                        logger.error(f"Error in event listener: {e}")
        else:
            self.stats.storage_errors += len(events)
        
        return success_count > 0
    
    async def close(self) -> None:
        """Close the log store (required by StorageAdapter interface)."""
        await self.shutdown()
    
    # Query Interface
    
    async def retrieve_events(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        event_types: Optional[List[EventType]] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: int = 1000,
        use_cache: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Retrieve events from the log store with caching."""
        if self.status not in [LogStoreStatus.ACTIVE, LogStoreStatus.DEGRADED]:
            return
        
        start_perf = time.perf_counter()
        
        # Check cache if enabled
        cache_key = None
        if use_cache:
            cache_key = self._generate_cache_key(
                start_time, end_time, event_types, session_id, agent_id, limit
            )
            cached_result = self.query_cache.get(cache_key)
            if cached_result and time.time() - cached_result['timestamp'] < self.cache_ttl:
                for event_dict in cached_result['events']:
                    yield event_dict
                    self.stats.total_events_retrieved += 1
                return
        
        # Retrieve from primary storage
        retrieved_events = []
        try:
            async for event_dict in self.primary_storage.retrieve_events(
                start_time=start_time,
                end_time=end_time,
                event_types=event_types,
                session_id=session_id,
                agent_id=agent_id,
                limit=limit
            ):
                retrieved_events.append(event_dict)
                yield event_dict
                self.stats.total_events_retrieved += 1
            
            # Cache results if caching is enabled
            if use_cache and cache_key:
                self.query_cache[cache_key] = {
                    'events': retrieved_events,
                    'timestamp': time.time()
                }
        
        except Exception as e:
            logger.error(f"Error retrieving events: {e}")
            self.stats.retrieval_errors += 1
            
            # Try backup storage if primary fails
            if self.backup_storage:
                try:
                    async for event_dict in self.backup_storage.retrieve_events(
                        start_time=start_time,
                        end_time=end_time,
                        event_types=event_types,
                        session_id=session_id,
                        agent_id=agent_id,
                        limit=limit
                    ):
                        yield event_dict
                        self.stats.total_events_retrieved += 1
                except Exception as backup_e:
                    logger.error(f"Error retrieving from backup storage: {backup_e}")
        
        finally:
            # Track retrieval performance
            latency_ms = (time.perf_counter() - start_perf) * 1000
            self.latency_samples.append(latency_ms)
    
    def _generate_cache_key(
        self,
        start_time: Optional[float],
        end_time: Optional[float],
        event_types: Optional[List[EventType]],
        session_id: Optional[str],
        agent_id: Optional[str],
        limit: int
    ) -> str:
        """Generate a cache key for query parameters."""
        import hashlib
        
        key_parts = [
            str(start_time) if start_time else "",
            str(end_time) if end_time else "",
            ",".join(sorted([et.value for et in event_types])) if event_types else "",
            session_id or "",
            agent_id or "",
            str(limit)
        ]
        
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    # High-level query methods
    
    async def get_session_events(
        self, 
        session_id: str, 
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get all events for a specific session."""
        events = []
        async for event in self.retrieve_events(session_id=session_id, limit=limit):
            events.append(event)
        return events
    
    async def get_agent_activity(
        self,
        agent_id: str,
        hours: int = 24,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get recent activity for a specific agent."""
        start_time = time.time() - (hours * 3600)
        events = []
        async for event in self.retrieve_events(
            agent_id=agent_id,
            start_time=start_time,
            limit=limit
        ):
            events.append(event)
        return events
    
    async def get_error_events(
        self,
        hours: int = 24,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get recent error events."""
        start_time = time.time() - (hours * 3600)
        events = []
        async for event in self.retrieve_events(
            event_types=[EventType.ERROR],
            start_time=start_time,
            limit=limit
        ):
            events.append(event)
        return events
    
    async def get_performance_metrics(
        self,
        hours: int = 1,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get recent performance metrics."""
        start_time = time.time() - (hours * 3600)
        events = []
        async for event in self.retrieve_events(
            event_types=[EventType.PERFORMANCE_METRICS],
            start_time=start_time,
            limit=limit
        ):
            events.append(event)
        return events
    
    # Event listener management
    
    def add_event_listener(self, listener: Callable[[AgentEvent], None]) -> None:
        """Add an event listener for real-time event processing."""
        self.event_listeners.append(listener)
        logger.debug(f"Added event listener: {listener.__name__}")
    
    def remove_event_listener(self, listener: Callable[[AgentEvent], None]) -> None:
        """Remove an event listener."""
        if listener in self.event_listeners:
            self.event_listeners.remove(listener)
            logger.debug(f"Removed event listener: {listener.__name__}")
    
    def add_error_handler(self, handler: Callable[[Exception, str], None]) -> None:
        """Add an error handler for storage errors."""
        self.error_handlers.append(handler)
    
    # Background tasks
    
    async def _health_check_loop(self) -> None:
        """Periodic health check for storage backends."""
        while self.status in [LogStoreStatus.ACTIVE, LogStoreStatus.DEGRADED]:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                healthy_storages = []
                for storage in self.active_storages:
                    try:
                        # Simple health check - try to get storage stats
                        stats = await storage.get_storage_stats()
                        if stats:
                            healthy_storages.append(storage)
                    except Exception as e:
                        logger.warning(f"Storage health check failed: {e}")
                        self._handle_storage_error(storage, e)
                
                # Update status based on healthy storages
                if not healthy_storages:
                    self.status = LogStoreStatus.ERROR
                    logger.error("All storage backends are unhealthy")
                elif len(healthy_storages) < len(self.active_storages):
                    self.status = LogStoreStatus.DEGRADED
                    logger.warning(f"Only {len(healthy_storages)}/{len(self.active_storages)} storage backends are healthy")
                else:
                    self.status = LogStoreStatus.ACTIVE
                
                self.stats.last_health_check = time.time()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
    
    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of old data."""
        while self.status in [LogStoreStatus.ACTIVE, LogStoreStatus.DEGRADED]:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Clean up old events from all storages
                for storage in self.active_storages:
                    try:
                        cleaned_count = await storage.cleanup_old_events()
                        if cleaned_count > 0:
                            logger.info(f"Cleaned up {cleaned_count} old events from {storage.__class__.__name__}")
                    except Exception as e:
                        logger.error(f"Error in cleanup for {storage.__class__.__name__}: {e}")
                
                # Clean up query cache
                current_time = time.time()
                expired_keys = [
                    key for key, data in self.query_cache.items()
                    if current_time - data['timestamp'] > self.cache_ttl
                ]
                for key in expired_keys:
                    del self.query_cache[key]
                
                if expired_keys:
                    logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _metrics_loop(self) -> None:
        """Periodic metrics calculation and reporting."""
        while self.status in [LogStoreStatus.ACTIVE, LogStoreStatus.DEGRADED]:
            try:
                await asyncio.sleep(300)  # Update every 5 minutes
                
                # Calculate performance statistics
                if self.latency_samples:
                    sorted_samples = sorted(self.latency_samples)
                    n = len(sorted_samples)
                    
                    self.stats.avg_storage_latency_ms = sum(sorted_samples) / n
                    self.stats.p95_storage_latency_ms = sorted_samples[int(n * 0.95)]
                    
                    # Keep only recent samples
                    if len(self.latency_samples) > 10000:
                        self.latency_samples = self.latency_samples[-5000:]
                
                # Calculate uptime
                self.stats.uptime_seconds = time.time() - self.start_time
                
                # Get storage statistics from primary storage
                try:
                    storage_stats = await self.primary_storage.get_storage_stats()
                    self.stats.storage_size_bytes = storage_stats.get('db_size_bytes', 0) or storage_stats.get('total_size_bytes', 0)
                except Exception as e:
                    logger.error(f"Error getting storage statistics: {e}")
                
                # Log periodic summary
                logger.debug(
                    f"LogStore metrics: {self.stats.total_events_stored} events stored, "
                    f"{self.stats.total_events_retrieved} retrieved, "
                    f"{self.stats.avg_storage_latency_ms:.2f}ms avg latency"
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics loop: {e}")
    
    def _handle_storage_error(self, storage: StorageAdapter, error: Exception) -> None:
        """Handle storage backend errors."""
        storage_name = storage.__class__.__name__
        self.error_counts[storage_name] = self.error_counts.get(storage_name, 0) + 1
        
        # Call error handlers
        for handler in self.error_handlers:
            try:
                handler(error, storage_name)
            except Exception as e:
                logger.error(f"Error in error handler: {e}")
        
        # Consider removing storage if too many consecutive errors
        if self.error_counts[storage_name] > 10:
            logger.error(f"Too many errors from {storage_name}, considering removal")
            # TODO: Implement storage removal logic
    
    def get_stats(self) -> LogStoreStats:
        """Get current log store statistics."""
        return self.stats
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status and health information."""
        return {
            'status': self.status.value,
            'active_storages': len(self.active_storages),
            'stats': self.stats,
            'uptime_seconds': time.time() - self.start_time,
            'cache_entries': len(self.query_cache),
            'error_counts': self.error_counts,
        }