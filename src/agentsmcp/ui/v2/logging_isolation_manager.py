"""
Logging Isolation Manager - Prevent logging output from contaminating TUI display.

Provides comprehensive logging isolation during TUI operation to prevent debug output,
info messages, warnings, and errors from polluting the terminal display. Captures
all logging output in memory buffers and provides controlled restoration after TUI exit.

ICD Compliance:
- Inputs: tui_active, log_level, buffer_size
- Outputs: isolation_active, buffered_logs, restore_result
- Performance: Isolation/restoration within 50ms
- Key Functions: Capture logs during TUI, restore after TUI exit
"""

import asyncio
import logging
import sys
import threading
import time
import re
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from io import StringIO
from typing import Dict, List, Optional, Any, Set, Deque, TextIO, Callable
from datetime import datetime, timedelta
import weakref


class LogLevel(Enum):
    """Log levels for isolation control."""
    NOTSET = logging.NOTSET
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


# THREAT: Sensitive data exposure in logs
# MITIGATION: Pattern matching for common secrets
SENSITIVE_PATTERNS = [
    re.compile(r'(?i)(api[_-]?key|secret|token|password|credential)["\s:=]+[\w\-./+]{8,}', re.IGNORECASE),
    re.compile(r'(?i)bearer\s+[a-zA-Z0-9\-._~+/]+=*', re.IGNORECASE),
    re.compile(r'(?i)basic\s+[a-zA-Z0-9+/]+=*', re.IGNORECASE),
    re.compile(r'sk-[a-zA-Z0-9]{20,}', re.IGNORECASE),  # OpenAI API keys
    re.compile(r'(?i)(ssh-rsa|ssh-dss|ssh-ed25519)\s+[a-zA-Z0-9+/]+', re.IGNORECASE),
    re.compile(r'(?i)-----BEGIN [A-Z ]+-----[\s\S]*?-----END [A-Z ]+-----', re.IGNORECASE),
    re.compile(r'[0-9a-fA-F]{32}', re.IGNORECASE),  # MD5 hashes that might be secrets
    re.compile(r'[0-9a-fA-F]{64}', re.IGNORECASE),  # SHA256 hashes that might be secrets
]

def sanitize_sensitive_data(message: str) -> str:
    """Sanitize sensitive data from log messages."""
    sanitized = message
    for pattern in SENSITIVE_PATTERNS:
        sanitized = pattern.sub('[REDACTED]', sanitized)
    return sanitized


@dataclass
class BufferedLogEntry:
    """A single buffered log entry."""
    timestamp: datetime
    level: int
    logger_name: str
    message: str
    pathname: str
    lineno: int
    funcName: str
    thread: int
    process: int
    exc_info: Optional[Any] = None
    stack_info: Optional[str] = None
    
    def __post_init__(self):
        """Sanitize sensitive data and enforce message length limits."""
        # THREAT: Memory exhaustion from large log messages
        # MITIGATION: Enforce message size limits
        max_message_size = 10000  # 10KB per message
        if len(self.message) > max_message_size:
            self.message = self.message[:max_message_size] + "... [TRUNCATED]"
        
        # THREAT: Sensitive data exposure in logs
        # MITIGATION: Sanitize sensitive patterns
        self.message = sanitize_sensitive_data(self.message)
        
        # Also sanitize stack info if present
        if self.stack_info and len(self.stack_info) > max_message_size:
            self.stack_info = self.stack_info[:max_message_size] + "... [TRUNCATED]"
        if self.stack_info:
            self.stack_info = sanitize_sensitive_data(self.stack_info)
    
    def to_log_record(self) -> logging.LogRecord:
        """Convert back to a LogRecord for replay."""
        record = logging.LogRecord(
            name=self.logger_name,
            level=self.level,
            pathname=self.pathname,
            lineno=self.lineno,
            msg=self.message,
            args=(),
            exc_info=self.exc_info
        )
        record.funcName = self.funcName
        record.thread = self.thread
        record.process = self.process
        record.stack_info = self.stack_info
        record.created = self.timestamp.timestamp()
        return record


@dataclass
class BufferedLogs:
    """Collection of buffered log entries."""
    entries: List[BufferedLogEntry] = field(default_factory=list)
    total_captured: int = 0
    buffer_overflows: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    max_buffer_size: int = 10000
    
    def add_entry(self, entry: BufferedLogEntry) -> None:
        """Add a log entry to the buffer."""
        # THREAT: Memory exhaustion from unbounded log buffer growth
        # MITIGATION: Enforce strict buffer size limits with memory-aware bounds checking
        if len(self.entries) >= self.max_buffer_size:
            # Remove oldest entry
            self.entries.pop(0)
            self.buffer_overflows += 1
        
        # Additional memory protection: check total memory usage
        estimated_memory_per_entry = 1024  # ~1KB per entry estimate
        estimated_total_memory = len(self.entries) * estimated_memory_per_entry
        max_memory_limit = 50 * 1024 * 1024  # 50MB limit
        
        if estimated_total_memory > max_memory_limit:
            # Emergency cleanup: remove half the oldest entries
            entries_to_remove = len(self.entries) // 2
            for _ in range(entries_to_remove):
                if self.entries:
                    self.entries.pop(0)
                    self.buffer_overflows += 1
        
        self.entries.append(entry)
        self.total_captured += 1
    
    def get_entries_by_level(self, min_level: int) -> List[BufferedLogEntry]:
        """Get entries at or above the specified level."""
        return [entry for entry in self.entries if entry.level >= min_level]
    
    def get_entries_by_logger(self, logger_name: str) -> List[BufferedLogEntry]:
        """Get entries from a specific logger."""
        return [entry for entry in self.entries if entry.logger_name == logger_name]
    
    def clear(self) -> None:
        """Clear all buffered entries."""
        self.entries.clear()
        self.total_captured = 0
        self.buffer_overflows = 0


@dataclass
class RestoreResult:
    """Result of logging restoration operations."""
    success: bool
    handlers_restored: int
    loggers_restored: int
    entries_replayed: int = 0
    errors_encountered: int = 0
    error_message: Optional[str] = None
    operation_time_ms: float = 0.0


class LoggingIsolationHandler(logging.Handler):
    """Custom logging handler that captures log records in memory."""
    
    def __init__(self, buffer: BufferedLogs, min_level: int = logging.NOTSET):
        super().__init__()
        self.buffer = buffer
        self.min_level = min_level
        self.setLevel(min_level)
    
    def emit(self, record: logging.LogRecord) -> None:
        """Capture a log record."""
        try:
            if record.levelno >= self.min_level:
                entry = BufferedLogEntry(
                    timestamp=datetime.fromtimestamp(record.created),
                    level=record.levelno,
                    logger_name=record.name,
                    message=self.format(record),
                    pathname=record.pathname,
                    lineno=record.lineno,
                    funcName=record.funcName,
                    thread=record.thread,
                    process=record.process,
                    exc_info=record.exc_info,
                    stack_info=getattr(record, 'stack_info', None)
                )
                self.buffer.add_entry(entry)
        except Exception:
            # Don't let logging errors break the handler
            pass


class StreamCapture:
    """Capture stdout/stderr streams."""
    
    def __init__(self, original_stream: TextIO):
        self.original_stream = original_stream
        self.captured_output = StringIO()
        self._lock = threading.Lock()
    
    def write(self, text: str) -> int:
        """Write to both captured buffer and optionally suppress."""
        with self._lock:
            # Capture the output
            self.captured_output.write(text)
            # Suppress from original stream during TUI operation
            return len(text)
    
    def flush(self) -> None:
        """Flush the captured buffer."""
        with self._lock:
            self.captured_output.flush()
    
    def getvalue(self) -> str:
        """Get captured output."""
        with self._lock:
            return self.captured_output.getvalue()
    
    def clear(self) -> None:
        """Clear captured output."""
        with self._lock:
            self.captured_output = StringIO()


class LoggingIsolationManager:
    """
    Comprehensive logging isolation manager for TUI applications.
    
    Prevents all logging output from contaminating the TUI display by capturing
    log messages in memory buffers and suppressing console output during TUI operation.
    """
    
    def __init__(self, buffer_size: int = 10000):
        """
        Initialize the logging isolation manager.
        
        Args:
            buffer_size: Maximum number of log entries to buffer
        """
        self._lock = threading.RLock()
        self._initialized = False
        
        # Isolation state
        self._isolation_active = False
        self._tui_active = False
        self._isolation_level = LogLevel.DEBUG
        
        # Buffering
        self._buffered_logs = BufferedLogs(max_buffer_size=buffer_size)
        self._isolation_handler: Optional[LoggingIsolationHandler] = None
        
        # Original handlers and levels
        self._original_handlers: Dict[str, List[logging.Handler]] = {}
        self._original_levels: Dict[str, int] = {}
        self._original_propagate: Dict[str, bool] = {}
        
        # Stream capture
        self._stdout_capture: Optional[StreamCapture] = None
        self._stderr_capture: Optional[StreamCapture] = None
        self._original_stdout: Optional[TextIO] = None
        self._original_stderr: Optional[TextIO] = None
        
        # Logger tracking
        self._monitored_loggers: Set[str] = set()
        self._critical_loggers = {
            'root',
            'agentsmcp',
            'agentsmcp.ui.v2',
            'agentsmcp.conversation.llm_client',
            'agentsmcp.orchestration',
            'agentsmcp.agents'
        }
        
        # Performance tracking
        self._operation_times: Dict[str, float] = {}
        self._start_time: Optional[float] = None
    
    async def initialize(self) -> bool:
        """
        Initialize the logging isolation manager.
        
        Returns:
            True if initialization successful, False otherwise
        """
        async with asyncio.Lock():
            if self._initialized:
                return True
            
            try:
                # Initialize buffering system
                self._buffered_logs = BufferedLogs(max_buffer_size=self._buffered_logs.max_buffer_size)
                
                # Create isolation handler
                self._isolation_handler = LoggingIsolationHandler(
                    buffer=self._buffered_logs,
                    min_level=self._isolation_level.value
                )
                
                # Setup stream captures
                self._setup_stream_captures()
                
                # Discover existing loggers
                self._discover_loggers()
                
                self._initialized = True
                return True
                
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Failed to initialize logging isolation manager: {e}")
                return False
    
    async def activate_isolation(self, 
                                 tui_active: bool = True, 
                                 log_level: LogLevel = LogLevel.DEBUG,
                                 buffer_size: Optional[int] = None) -> bool:
        """
        Activate logging isolation.
        
        Args:
            tui_active: Whether TUI is currently active
            log_level: Minimum log level to capture
            buffer_size: Override buffer size if provided
            
        Returns:
            True if isolation activated successfully, False otherwise
        """
        start_time = time.time()
        
        try:
            with self._lock:
                if self._isolation_active:
                    return True
                
                if not self._initialized:
                    await self.initialize()
                
                # Update configuration
                self._tui_active = tui_active
                self._isolation_level = log_level
                if buffer_size:
                    self._buffered_logs.max_buffer_size = buffer_size
                
                # Update handler level
                if self._isolation_handler:
                    self._isolation_handler.setLevel(log_level.value)
                    self._isolation_handler.min_level = log_level.value
                
                # Capture current state
                await self._capture_original_state()
                
                # Apply isolation
                await self._apply_isolation()
                
                # Activate stream capture
                self._activate_stream_capture()
                
                # Mark as active
                self._isolation_active = True
                self._buffered_logs.start_time = datetime.now()
                self._start_time = start_time
                
                return True
                
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to activate logging isolation: {e}")
            return False
        finally:
            operation_time = (time.time() - start_time) * 1000
            self._operation_times['activate_isolation'] = operation_time
    
    async def deactivate_isolation(self) -> bool:
        """
        Deactivate logging isolation and restore normal logging.
        
        Returns:
            True if isolation deactivated successfully, False otherwise
        """
        start_time = time.time()
        
        try:
            with self._lock:
                if not self._isolation_active:
                    return True
                
                # Mark end time
                self._buffered_logs.end_time = datetime.now()
                
                # Deactivate stream capture
                self._deactivate_stream_capture()
                
                # Restore original logging state
                await self._restore_original_state()
                
                # Mark as inactive
                self._isolation_active = False
                self._tui_active = False
                
                return True
                
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to deactivate logging isolation: {e}")
            return False
        finally:
            operation_time = (time.time() - start_time) * 1000
            self._operation_times['deactivate_isolation'] = operation_time
    
    def is_isolation_active(self) -> bool:
        """Check if logging isolation is currently active."""
        return self._isolation_active
    
    def get_buffered_logs(self) -> BufferedLogs:
        """
        Get the current buffered logs.
        
        Returns:
            BufferedLogs object with captured log entries
        """
        return self._buffered_logs
    
    async def replay_buffered_logs(self, 
                                   min_level: LogLevel = LogLevel.INFO,
                                   target_loggers: Optional[Set[str]] = None) -> RestoreResult:
        """
        Replay buffered logs to their original destinations.
        
        Args:
            min_level: Minimum log level to replay
            target_loggers: Specific loggers to replay (None for all)
            
        Returns:
            RestoreResult with replay statistics
        """
        start_time = time.time()
        replayed = 0
        errors = 0
        
        try:
            entries_to_replay = self._buffered_logs.get_entries_by_level(min_level.value)
            
            if target_loggers:
                entries_to_replay = [
                    entry for entry in entries_to_replay 
                    if entry.logger_name in target_loggers
                ]
            
            # Temporarily restore handlers for replay
            original_isolation_state = self._isolation_active
            if original_isolation_state:
                await self.deactivate_isolation()
            
            try:
                for entry in entries_to_replay:
                    try:
                        logger = logging.getLogger(entry.logger_name)
                        record = entry.to_log_record()
                        logger.handle(record)
                        replayed += 1
                    except Exception:
                        errors += 1
                        
            finally:
                # Restore isolation state if it was active
                if original_isolation_state:
                    await self.activate_isolation(
                        tui_active=self._tui_active,
                        log_level=self._isolation_level
                    )
            
            operation_time = (time.time() - start_time) * 1000
            
            return RestoreResult(
                success=True,
                handlers_restored=0,  # Not applicable for replay
                loggers_restored=0,   # Not applicable for replay
                entries_replayed=replayed,
                errors_encountered=errors,
                operation_time_ms=operation_time
            )
            
        except Exception as e:
            operation_time = (time.time() - start_time) * 1000
            
            return RestoreResult(
                success=False,
                handlers_restored=0,
                loggers_restored=0,
                entries_replayed=replayed,
                errors_encountered=errors + 1,
                error_message=str(e),
                operation_time_ms=operation_time
            )
    
    @contextmanager
    def isolation_context(self, 
                         log_level: LogLevel = LogLevel.DEBUG,
                         buffer_size: Optional[int] = None):
        """
        Context manager for temporary logging isolation.
        
        Args:
            log_level: Minimum log level to capture
            buffer_size: Override buffer size if provided
        """
        activated = False
        
        try:
            # Activate isolation
            task = asyncio.create_task(
                self.activate_isolation(
                    tui_active=True,
                    log_level=log_level,
                    buffer_size=buffer_size
                )
            )
            # Note: In sync context, we can't await, so this is best-effort
            activated = True
            
            yield self._buffered_logs
            
        finally:
            if activated:
                # Deactivate isolation
                try:
                    task = asyncio.create_task(self.deactivate_isolation())
                except:
                    pass  # Best effort cleanup
    
    async def cleanup(self) -> RestoreResult:
        """
        Cleanup the logging isolation manager.
        
        Returns:
            RestoreResult with cleanup status
        """
        start_time = time.time()
        
        try:
            # Deactivate isolation if active
            if self._isolation_active:
                await self.deactivate_isolation()
            
            # Clear buffers
            self._buffered_logs.clear()
            
            # Reset state
            with self._lock:
                self._initialized = False
                self._isolation_handler = None
                self._original_handlers.clear()
                self._original_levels.clear()
                self._original_propagate.clear()
                self._monitored_loggers.clear()
            
            operation_time = (time.time() - start_time) * 1000
            
            return RestoreResult(
                success=True,
                handlers_restored=0,
                loggers_restored=0,
                operation_time_ms=operation_time
            )
            
        except Exception as e:
            operation_time = (time.time() - start_time) * 1000
            
            return RestoreResult(
                success=False,
                handlers_restored=0,
                loggers_restored=0,
                error_message=str(e),
                operation_time_ms=operation_time
            )
    
    def _setup_stream_captures(self) -> None:
        """Setup stdout/stderr capture objects."""
        self._stdout_capture = StreamCapture(sys.stdout)
        self._stderr_capture = StreamCapture(sys.stderr)
    
    def _discover_loggers(self) -> None:
        """Discover existing loggers in the system."""
        # Add critical loggers
        self._monitored_loggers.update(self._critical_loggers)
        
        # Add any existing loggers from logging module
        try:
            # Get all existing loggers
            existing_loggers = [
                logging.getLogger(),  # Root logger
                logging.getLogger('agentsmcp'),
                logging.getLogger('agentsmcp.ui'),
                logging.getLogger('agentsmcp.ui.v2'),
                logging.getLogger('requests'),
                logging.getLogger('urllib3'),
                logging.getLogger('httpx')
            ]
            
            for logger in existing_loggers:
                if logger.name:
                    self._monitored_loggers.add(logger.name)
                else:
                    self._monitored_loggers.add('root')
                    
        except Exception:
            pass  # Continue with critical loggers only
    
    async def _capture_original_state(self) -> None:
        """Capture original logger states before applying isolation."""
        for logger_name in self._monitored_loggers:
            try:
                if logger_name == 'root':
                    logger = logging.getLogger()
                else:
                    logger = logging.getLogger(logger_name)
                
                # Capture handlers
                self._original_handlers[logger_name] = list(logger.handlers)
                
                # Capture level
                self._original_levels[logger_name] = logger.level
                
                # Capture propagate setting
                self._original_propagate[logger_name] = logger.propagate
                
            except Exception:
                continue  # Skip problematic loggers
    
    async def _apply_isolation(self) -> None:
        """Apply logging isolation to monitored loggers."""
        if not self._isolation_handler:
            return
        
        for logger_name in self._monitored_loggers:
            try:
                if logger_name == 'root':
                    logger = logging.getLogger()
                else:
                    logger = logging.getLogger(logger_name)
                
                # Clear existing handlers
                logger.handlers.clear()
                
                # Add our isolation handler
                logger.addHandler(self._isolation_handler)
                
                # Set level to capture everything at or above our threshold
                logger.setLevel(self._isolation_level.value)
                
                # Disable propagation to prevent duplicate captures
                logger.propagate = False
                
            except Exception:
                continue  # Skip problematic loggers
    
    async def _restore_original_state(self) -> None:
        """Restore original logger states."""
        for logger_name in self._monitored_loggers:
            try:
                if logger_name == 'root':
                    logger = logging.getLogger()
                else:
                    logger = logging.getLogger(logger_name)
                
                # Clear isolation handlers
                logger.handlers.clear()
                
                # Restore original handlers
                if logger_name in self._original_handlers:
                    for handler in self._original_handlers[logger_name]:
                        logger.addHandler(handler)
                
                # Restore original level
                if logger_name in self._original_levels:
                    logger.setLevel(self._original_levels[logger_name])
                
                # Restore original propagate setting
                if logger_name in self._original_propagate:
                    logger.propagate = self._original_propagate[logger_name]
                
            except Exception:
                continue  # Skip problematic loggers
    
    def _activate_stream_capture(self) -> None:
        """Activate stdout/stderr stream capture."""
        try:
            if self._stdout_capture:
                self._original_stdout = sys.stdout
                sys.stdout = self._stdout_capture
            
            if self._stderr_capture:
                self._original_stderr = sys.stderr
                sys.stderr = self._stderr_capture
                
        except Exception:
            pass  # Continue without stream capture
    
    def _deactivate_stream_capture(self) -> None:
        """Deactivate stdout/stderr stream capture."""
        try:
            if self._original_stdout:
                sys.stdout = self._original_stdout
                self._original_stdout = None
            
            if self._original_stderr:
                sys.stderr = self._original_stderr
                self._original_stderr = None
                
        except Exception:
            pass  # Best effort restore
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for logging operations."""
        return {
            'operation_times': dict(self._operation_times),
            'isolation_active': self._isolation_active,
            'monitored_loggers': len(self._monitored_loggers),
            'buffered_entries': len(self._buffered_logs.entries),
            'buffer_overflows': self._buffered_logs.buffer_overflows,
            'total_captured': self._buffered_logs.total_captured,
            'isolation_duration_seconds': (
                (time.time() - self._start_time) if self._start_time else 0
            )
        }


# Singleton instance for global access
_logging_isolation_manager: Optional[LoggingIsolationManager] = None


async def get_logging_isolation_manager() -> LoggingIsolationManager:
    """
    Get or create the global logging isolation manager instance.
    
    Returns:
        LoggingIsolationManager instance
    """
    global _logging_isolation_manager
    
    if _logging_isolation_manager is None:
        _logging_isolation_manager = LoggingIsolationManager()
        await _logging_isolation_manager.initialize()
    
    return _logging_isolation_manager


async def cleanup_logging_isolation_manager() -> RestoreResult:
    """
    Cleanup the global logging isolation manager.
    
    Returns:
        RestoreResult with cleanup status
    """
    global _logging_isolation_manager
    
    if _logging_isolation_manager:
        result = await _logging_isolation_manager.cleanup()
        _logging_isolation_manager = None
        return result
    
    return RestoreResult(success=True, handlers_restored=0, loggers_restored=0)