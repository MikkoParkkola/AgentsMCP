"""Storage adapters for agent execution logs.

This module provides pluggable storage backends for agent execution logs,
supporting file-based storage, database storage, and in-memory storage
with encryption, compression, and high-performance batch operations.
"""

import asyncio
import json
import gzip
import os
import sqlite3
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import aiosqlite
import aiofiles
from cryptography.fernet import Fernet
import logging

from .log_schemas import AgentEvent, EventType, EventSeverity, RetentionPolicy

logger = logging.getLogger(__name__)


@dataclass
class StorageStats:
    """Statistics for storage operations."""
    events_stored: int = 0
    events_retrieved: int = 0
    storage_errors: int = 0
    retrieval_errors: int = 0
    total_storage_time_ms: float = 0.0
    total_retrieval_time_ms: float = 0.0
    bytes_written: int = 0
    bytes_read: int = 0
    compression_ratio: float = 0.0


class StorageAdapter(ABC):
    """Abstract base class for storage adapters."""
    
    def __init__(
        self, 
        encryption_key: Optional[bytes] = None,
        compression_enabled: bool = True,
        retention_policy: Optional[RetentionPolicy] = None
    ):
        """Initialize storage adapter.
        
        Args:
            encryption_key: Optional encryption key for data at rest
            compression_enabled: Whether to compress stored data
            retention_policy: Data retention and cleanup policy
        """
        self.encryption_key = encryption_key
        self.compression_enabled = compression_enabled
        self.retention_policy = retention_policy or RetentionPolicy()
        self.stats = StorageStats()
        
        # Initialize encryption
        self._cipher = Fernet(encryption_key) if encryption_key else None
        
        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the storage backend."""
        pass
    
    @abstractmethod
    async def store_event(self, event: AgentEvent) -> bool:
        """Store a single event.
        
        Args:
            event: The event to store
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def store_batch(self, events: List[AgentEvent]) -> bool:
        """Store a batch of events efficiently.
        
        Args:
            events: List of events to store
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def retrieve_events(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        event_types: Optional[List[EventType]] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: int = 1000
    ) -> AsyncGenerator[AgentEvent, None]:
        """Retrieve events based on criteria.
        
        Args:
            start_time: Start timestamp filter
            end_time: End timestamp filter
            event_types: Event type filter
            session_id: Session ID filter
            agent_id: Agent ID filter
            limit: Maximum number of events to return
            
        Yields:
            AgentEvent objects matching the criteria
        """
        pass
    
    @abstractmethod
    async def cleanup_old_events(self) -> int:
        """Clean up old events based on retention policy.
        
        Returns:
            Number of events cleaned up
        """
        pass
    
    @abstractmethod
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage-specific statistics."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the storage backend and cleanup resources."""
        pass
    
    def _encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data if encryption is enabled."""
        if self._cipher:
            return self._cipher.encrypt(data)
        return data
    
    def _decrypt_data(self, data: bytes) -> bytes:
        """Decrypt data if encryption is enabled."""
        if self._cipher:
            return self._cipher.decrypt(data)
        return data
    
    def _compress_data(self, data: bytes) -> bytes:
        """Compress data if compression is enabled."""
        if self.compression_enabled:
            return gzip.compress(data)
        return data
    
    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress data if compression is enabled."""
        if self.compression_enabled:
            try:
                return gzip.decompress(data)
            except gzip.BadGzipFile:
                # Data might not be compressed
                return data
        return data
    
    def _serialize_event(self, event: AgentEvent) -> bytes:
        """Serialize an event to bytes with optional encryption/compression."""
        # Convert event to JSON
        event_dict = event.to_dict()
        json_str = json.dumps(event_dict, separators=(',', ':'))
        json_bytes = json_str.encode('utf-8')
        
        # Apply compression first, then encryption
        if self.compression_enabled:
            json_bytes = self._compress_data(json_bytes)
        
        if self._cipher:
            json_bytes = self._encrypt_data(json_bytes)
        
        return json_bytes
    
    def _deserialize_event(self, data: bytes) -> Dict[str, Any]:
        """Deserialize event data with optional decryption/decompression."""
        # Apply decryption first, then decompression
        if self._cipher:
            data = self._decrypt_data(data)
        
        if self.compression_enabled:
            data = self._decompress_data(data)
        
        # Parse JSON
        json_str = data.decode('utf-8')
        return json.loads(json_str)


class FileStorageAdapter(StorageAdapter):
    """File-based storage adapter with rotation and compression."""
    
    def __init__(
        self,
        storage_path: str = "logs/agent_execution",
        max_file_size_mb: int = 100,
        max_files: int = 100,
        **kwargs
    ):
        """Initialize file storage adapter.
        
        Args:
            storage_path: Directory path for log files
            max_file_size_mb: Maximum size per log file before rotation
            max_files: Maximum number of log files to keep
        """
        super().__init__(**kwargs)
        self.storage_path = Path(storage_path)
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.max_files = max_files
        
        # Current file handle
        self.current_file: Optional[aiofiles.threadpool.text.AsyncTextIOWrapper] = None
        self.current_file_path: Optional[Path] = None
        self.current_file_size = 0
    
    async def initialize(self) -> None:
        """Initialize file storage."""
        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Start background cleanup task
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        
        logger.info(f"FileStorageAdapter initialized at {self.storage_path}")
    
    async def store_event(self, event: AgentEvent) -> bool:
        """Store a single event to file."""
        return await self.store_batch([event])
    
    async def store_batch(self, events: List[AgentEvent]) -> bool:
        """Store a batch of events to file efficiently."""
        if not events:
            return True
        
        start_time = time.perf_counter()
        
        try:
            # Ensure we have a current file
            await self._ensure_current_file()
            
            # Serialize events
            serialized_events = []
            for event in events:
                try:
                    event_data = self._serialize_event(event)
                    serialized_events.append(event_data)
                except Exception as e:
                    logger.error(f"Error serializing event {getattr(event, 'event_id', 'unknown')}: {e}")
                    self.stats.storage_errors += 1
                    continue
            
            if not serialized_events:
                return False
            
            # Write batch to file
            batch_data = b'\n'.join(serialized_events) + b'\n'
            
            async with aiofiles.open(self.current_file_path, 'ab') as f:
                await f.write(batch_data)
                await f.flush()
                os.fsync(f.fileno())  # Ensure data is written to disk
            
            # Update statistics
            self.current_file_size += len(batch_data)
            self.stats.events_stored += len(events)
            self.stats.bytes_written += len(batch_data)
            
            # Check if file rotation is needed
            if self.current_file_size > self.max_file_size:
                await self._rotate_file()
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing batch: {e}")
            self.stats.storage_errors += len(events)
            return False
        
        finally:
            # Track storage time
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.stats.total_storage_time_ms += elapsed_ms
    
    async def retrieve_events(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        event_types: Optional[List[EventType]] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: int = 1000
    ) -> AsyncGenerator[AgentEvent, None]:
        """Retrieve events from files."""
        start_perf = time.perf_counter()
        count = 0
        
        try:
            # Get all log files sorted by creation time (newest first)
            log_files = sorted(
                self.storage_path.glob("*.log"),
                key=lambda p: p.stat().st_ctime,
                reverse=True
            )
            
            for log_file in log_files:
                if count >= limit:
                    break
                
                try:
                    async with aiofiles.open(log_file, 'rb') as f:
                        async for line in f:
                            if count >= limit:
                                break
                            
                            line = line.strip()
                            if not line:
                                continue
                            
                            try:
                                # Deserialize event
                                event_dict = self._deserialize_event(line)
                                
                                # Apply filters
                                if not self._matches_filters(
                                    event_dict, start_time, end_time, 
                                    event_types, session_id, agent_id
                                ):
                                    continue
                                
                                # Convert back to event object (simplified)
                                yield event_dict  # In practice, you'd reconstruct the proper event object
                                count += 1
                                self.stats.events_retrieved += 1
                                
                            except Exception as e:
                                logger.error(f"Error deserializing event: {e}")
                                self.stats.retrieval_errors += 1
                                continue
                
                except Exception as e:
                    logger.error(f"Error reading log file {log_file}: {e}")
                    self.stats.retrieval_errors += 1
                    continue
        
        finally:
            # Track retrieval time
            elapsed_ms = (time.perf_counter() - start_perf) * 1000
            self.stats.total_retrieval_time_ms += elapsed_ms
    
    def _matches_filters(
        self,
        event_dict: Dict[str, Any],
        start_time: Optional[float],
        end_time: Optional[float],
        event_types: Optional[List[EventType]],
        session_id: Optional[str],
        agent_id: Optional[str]
    ) -> bool:
        """Check if event matches the given filters."""
        # Time range filter
        event_time = event_dict.get('timestamp', 0)
        if start_time and event_time < start_time:
            return False
        if end_time and event_time > end_time:
            return False
        
        # Event type filter
        if event_types:
            event_type = event_dict.get('event_type')
            if event_type not in [et.value for et in event_types]:
                return False
        
        # Session ID filter
        if session_id and event_dict.get('session_id') != session_id:
            return False
        
        # Agent ID filter
        if agent_id and event_dict.get('agent_id') != agent_id:
            return False
        
        return True
    
    async def _ensure_current_file(self) -> None:
        """Ensure we have a current file open for writing."""
        if self.current_file_path and self.current_file_path.exists():
            return
        
        # Create new log file
        timestamp = int(time.time())
        filename = f"agent_execution_{timestamp}.log"
        self.current_file_path = self.storage_path / filename
        self.current_file_size = 0
    
    async def _rotate_file(self) -> None:
        """Rotate to a new log file."""
        logger.info(f"Rotating log file {self.current_file_path}")
        
        # Close current file
        self.current_file = None
        self.current_file_path = None
        self.current_file_size = 0
        
        # Clean up old files if needed
        await self._cleanup_old_files()
    
    async def _cleanup_old_files(self) -> None:
        """Clean up old log files based on file count limit."""
        log_files = sorted(
            self.storage_path.glob("*.log"),
            key=lambda p: p.stat().st_ctime,
            reverse=True
        )
        
        # Remove files beyond the limit
        for old_file in log_files[self.max_files:]:
            try:
                old_file.unlink()
                logger.debug(f"Removed old log file {old_file}")
            except Exception as e:
                logger.error(f"Error removing old log file {old_file}: {e}")
    
    async def cleanup_old_events(self) -> int:
        """Clean up events based on retention policy."""
        # This is simplified - in practice you'd need to rewrite files
        # to remove only specific events while keeping others
        logger.info("File-based cleanup not implemented - use database storage for fine-grained cleanup")
        return 0
    
    async def _periodic_cleanup(self) -> None:
        """Periodic cleanup task."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self._cleanup_old_files()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get file storage statistics."""
        try:
            # Count log files and total size
            log_files = list(self.storage_path.glob("*.log"))
            total_size = sum(f.stat().st_size for f in log_files)
            
            return {
                'num_log_files': len(log_files),
                'total_size_bytes': total_size,
                'total_size_mb': total_size / 1024 / 1024,
                'current_file_size_bytes': self.current_file_size,
                'storage_path': str(self.storage_path),
                **asdict(self.stats)
            }
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return asdict(self.stats)
    
    async def close(self) -> None:
        """Close file storage."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("FileStorageAdapter closed")


class DatabaseStorageAdapter(StorageAdapter):
    """SQLite database storage adapter with efficient querying."""
    
    def __init__(
        self,
        db_path: str = "logs/agent_execution.db",
        **kwargs
    ):
        """Initialize database storage adapter.
        
        Args:
            db_path: Path to SQLite database file
        """
        super().__init__(**kwargs)
        self.db_path = db_path
        self.connection_pool: List[aiosqlite.Connection] = []
    
    async def initialize(self) -> None:
        """Initialize database storage."""
        # Create directory for database
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create tables
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS agent_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE,
                    event_type TEXT,
                    timestamp REAL,
                    session_id TEXT,
                    agent_id TEXT,
                    user_id TEXT,
                    severity TEXT,
                    event_data BLOB,
                    created_at REAL DEFAULT (strftime('%s', 'now'))
                )
            """)
            
            # Create indexes for efficient querying
            await db.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON agent_events(timestamp)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON agent_events(session_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_agent_id ON agent_events(agent_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON agent_events(event_type)")
            
            await db.commit()
        
        # Start background cleanup task
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        
        logger.info(f"DatabaseStorageAdapter initialized at {self.db_path}")
    
    async def store_event(self, event: AgentEvent) -> bool:
        """Store a single event to database."""
        return await self.store_batch([event])
    
    async def store_batch(self, events: List[AgentEvent]) -> bool:
        """Store a batch of events to database efficiently."""
        if not events:
            return True
        
        start_time = time.perf_counter()
        
        try:
            # Serialize events
            serialized_events = []
            for event in events:
                try:
                    event_data = self._serialize_event(event)
                    serialized_events.append((
                        getattr(event, 'event_id', ''),
                        getattr(event, 'event_type', EventType.USER_INTERACTION).value,
                        getattr(event, 'timestamp', time.time()),
                        getattr(event, 'session_id', ''),
                        getattr(event, 'agent_id', None),
                        getattr(event, 'user_id', None),
                        getattr(event, 'severity', EventSeverity.INFO).value,
                        event_data
                    ))
                except Exception as e:
                    logger.error(f"Error serializing event: {e}")
                    self.stats.storage_errors += 1
                    continue
            
            if not serialized_events:
                return False
            
            # Batch insert
            async with aiosqlite.connect(self.db_path) as db:
                await db.executemany("""
                    INSERT OR REPLACE INTO agent_events 
                    (event_id, event_type, timestamp, session_id, agent_id, user_id, severity, event_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, serialized_events)
                
                await db.commit()
            
            # Update statistics
            self.stats.events_stored += len(events)
            return True
            
        except Exception as e:
            logger.error(f"Error storing batch to database: {e}")
            self.stats.storage_errors += len(events)
            return False
        
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.stats.total_storage_time_ms += elapsed_ms
    
    async def retrieve_events(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        event_types: Optional[List[EventType]] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: int = 1000
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Retrieve events from database."""
        start_perf = time.perf_counter()
        
        try:
            # Build query
            query = "SELECT event_data FROM agent_events WHERE 1=1"
            params = []
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
            
            if event_types:
                placeholders = ','.join(['?' for _ in event_types])
                query += f" AND event_type IN ({placeholders})"
                params.extend([et.value for et in event_types])
            
            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)
            
            if agent_id:
                query += " AND agent_id = ?"
                params.append(agent_id)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            # Execute query
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(query, params) as cursor:
                    async for row in cursor:
                        try:
                            event_dict = self._deserialize_event(row[0])
                            yield event_dict
                            self.stats.events_retrieved += 1
                        except Exception as e:
                            logger.error(f"Error deserializing event: {e}")
                            self.stats.retrieval_errors += 1
                            continue
        
        except Exception as e:
            logger.error(f"Error retrieving events: {e}")
            self.stats.retrieval_errors += 1
        
        finally:
            elapsed_ms = (time.perf_counter() - start_perf) * 1000
            self.stats.total_retrieval_time_ms += elapsed_ms
    
    async def cleanup_old_events(self) -> int:
        """Clean up old events based on retention policy."""
        try:
            cutoff_time = time.time() - (self.retention_policy.user_interactions_days * 24 * 3600)
            
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    "DELETE FROM agent_events WHERE timestamp < ?",
                    (cutoff_time,)
                )
                deleted_count = cursor.rowcount
                await db.commit()
            
            logger.info(f"Cleaned up {deleted_count} old events")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old events: {e}")
            return 0
    
    async def _periodic_cleanup(self) -> None:
        """Periodic cleanup task."""
        while True:
            try:
                await asyncio.sleep(24 * 3600)  # Run daily
                await self.cleanup_old_events()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get database storage statistics."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Get event counts
                cursor = await db.execute("SELECT COUNT(*) FROM agent_events")
                total_events = (await cursor.fetchone())[0]
                
                # Get database file size
                db_size = Path(self.db_path).stat().st_size
                
                return {
                    'total_events': total_events,
                    'db_size_bytes': db_size,
                    'db_size_mb': db_size / 1024 / 1024,
                    'db_path': self.db_path,
                    **asdict(self.stats)
                }
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return asdict(self.stats)
    
    async def close(self) -> None:
        """Close database storage."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("DatabaseStorageAdapter closed")


class MemoryStorageAdapter(StorageAdapter):
    """In-memory storage adapter for testing and development."""
    
    def __init__(self, max_events: int = 10000, **kwargs):
        """Initialize memory storage adapter.
        
        Args:
            max_events: Maximum number of events to keep in memory
        """
        super().__init__(**kwargs)
        self.max_events = max_events
        self.events: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize memory storage."""
        logger.info("MemoryStorageAdapter initialized")
    
    async def store_event(self, event: AgentEvent) -> bool:
        """Store a single event in memory."""
        return await self.store_batch([event])
    
    async def store_batch(self, events: List[AgentEvent]) -> bool:
        """Store a batch of events in memory."""
        if not events:
            return True
        
        start_time = time.perf_counter()
        
        try:
            async with self._lock:
                for event in events:
                    event_dict = event.to_dict()
                    self.events.append(event_dict)
                    
                    # Remove oldest events if we exceed max_events
                    if len(self.events) > self.max_events:
                        self.events.pop(0)
                
                self.stats.events_stored += len(events)
                return True
                
        except Exception as e:
            logger.error(f"Error storing batch in memory: {e}")
            self.stats.storage_errors += len(events)
            return False
        
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.stats.total_storage_time_ms += elapsed_ms
    
    async def retrieve_events(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        event_types: Optional[List[EventType]] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: int = 1000
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Retrieve events from memory."""
        start_perf = time.perf_counter()
        count = 0
        
        try:
            async with self._lock:
                # Sort events by timestamp (newest first)
                sorted_events = sorted(
                    self.events,
                    key=lambda e: e.get('timestamp', 0),
                    reverse=True
                )
                
                for event_dict in sorted_events:
                    if count >= limit:
                        break
                    
                    # Apply filters
                    if self._matches_filters(
                        event_dict, start_time, end_time,
                        event_types, session_id, agent_id
                    ):
                        yield event_dict
                        count += 1
                        self.stats.events_retrieved += 1
        
        finally:
            elapsed_ms = (time.perf_counter() - start_perf) * 1000
            self.stats.total_retrieval_time_ms += elapsed_ms
    
    def _matches_filters(
        self,
        event_dict: Dict[str, Any],
        start_time: Optional[float],
        end_time: Optional[float],
        event_types: Optional[List[EventType]],
        session_id: Optional[str],
        agent_id: Optional[str]
    ) -> bool:
        """Check if event matches filters."""
        # Time range filter
        event_time = event_dict.get('timestamp', 0)
        if start_time and event_time < start_time:
            return False
        if end_time and event_time > end_time:
            return False
        
        # Event type filter
        if event_types:
            event_type = event_dict.get('event_type')
            if event_type not in [et.value for et in event_types]:
                return False
        
        # Session ID filter
        if session_id and event_dict.get('session_id') != session_id:
            return False
        
        # Agent ID filter
        if agent_id and event_dict.get('agent_id') != agent_id:
            return False
        
        return True
    
    async def cleanup_old_events(self) -> int:
        """Clean up old events from memory."""
        cutoff_time = time.time() - (self.retention_policy.user_interactions_days * 24 * 3600)
        
        async with self._lock:
            original_count = len(self.events)
            self.events = [
                event for event in self.events
                if event.get('timestamp', 0) >= cutoff_time
            ]
            deleted_count = original_count - len(self.events)
        
        return deleted_count
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get memory storage statistics."""
        async with self._lock:
            return {
                'events_in_memory': len(self.events),
                'max_events': self.max_events,
                'memory_utilization_percent': (len(self.events) / self.max_events) * 100,
                **asdict(self.stats)
            }
    
    async def close(self) -> None:
        """Close memory storage."""
        async with self._lock:
            self.events.clear()
        
        logger.info("MemoryStorageAdapter closed")