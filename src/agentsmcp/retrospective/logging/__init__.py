"""Execution logging infrastructure for AgentsMCP retrospective system.

This package provides comprehensive agent execution data capture with:
- High-performance logging with minimal overhead (<5ms latency, <2% overhead)
- PII sanitization pipeline for privacy compliance
- Multiple storage backends with encryption support
- Comprehensive activity tracking for retrospective analysis

Main Components:
- ExecutionLogCapture: High-performance asynchronous logging system
- AgentEventSchema: Comprehensive event data models  
- PIISanitizer: Automatic privacy-compliant data sanitization
- StorageAdapters: Pluggable storage backends (file, database)
- LogStore: Centralized log management and retrieval
"""

from .log_schemas import (
    AgentEvent,
    EventType,
    EventSeverity,
    UserInteractionEvent,
    AgentDelegationEvent,
    LLMCallEvent,
    PerformanceMetricsEvent,
    ErrorEvent,
    ContextEvent,
    LoggingConfig,
    RetentionPolicy,
)

from .execution_log_capture import ExecutionLogCapture
from .pii_sanitizer import PIISanitizer, SanitizationLevel
from .storage_adapters import (
    StorageAdapter,
    FileStorageAdapter,
    DatabaseStorageAdapter,
    MemoryStorageAdapter,
)

__all__ = [
    # Core logging components
    "ExecutionLogCapture",
    "PIISanitizer",
    "SanitizationLevel",
    
    # Event schemas
    "AgentEvent",
    "EventType", 
    "EventSeverity",
    "UserInteractionEvent",
    "AgentDelegationEvent",
    "LLMCallEvent",
    "PerformanceMetricsEvent",
    "ErrorEvent",
    "ContextEvent",
    
    # Configuration
    "LoggingConfig",
    "RetentionPolicy",
    
    # Storage adapters
    "StorageAdapter",
    "FileStorageAdapter", 
    "DatabaseStorageAdapter",
    "MemoryStorageAdapter",
]

__version__ = "1.0.0"
__author__ = "AgentsMCP Development Team"
__description__ = "Execution logging infrastructure for retrospective analysis"