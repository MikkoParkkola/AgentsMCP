"""Storage components for AgentsMCP retrospective system.

This package provides centralized storage management for agent execution logs
with support for multiple backends, data retention, and high-performance querying.

Main Components:
- LogStore: Centralized log management and storage coordination
- Storage adapters integration and configuration
"""

from .log_store import LogStore, LogStoreStatus, LogStoreStats

__all__ = [
    "LogStore",
    "LogStoreStatus", 
    "LogStoreStats",
]

__version__ = "1.0.0"
__author__ = "AgentsMCP Development Team"
__description__ = "Storage components for retrospective system"