"""
Adapters for external systems integration.

These adapters implement the ports (repository interfaces)
and provide integration with existing AgentsMCP systems
and external services.
"""

from .config_adapter import AgentsMCPConfigAdapter
from .file_storage_adapter import FileStorageAdapter
from .memory_repositories import (
    InMemorySettingsRepository,
    InMemoryUserRepository,
    InMemoryAgentRepository,
    InMemoryAuditRepository,
    InMemoryCacheRepository,
    InMemorySecretRepository,
)

__all__ = [
    "AgentsMCPConfigAdapter",
    "FileStorageAdapter",
    "InMemorySettingsRepository",
    "InMemoryUserRepository", 
    "InMemoryAgentRepository",
    "InMemoryAuditRepository",
    "InMemoryCacheRepository",
    "InMemorySecretRepository",
]