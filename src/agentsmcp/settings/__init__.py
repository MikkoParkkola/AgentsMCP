"""
AgentsMCP Settings Management System

This module provides a comprehensive backend system for managing product settings 
and specialized agents with a hexagonal architecture design.

Features:
- Hierarchical settings with inheritance (Global → User → Session → Agent)
- Agent lifecycle management with instruction templates
- Real-time validation and conflict detection
- Event-driven design with real-time updates
- Security-first approach with encryption and access control
- Multi-user support with role-based permissions

Architecture:
- Domain: Core business logic and entities
- Ports: Interfaces for external dependencies
- Adapters: Implementations for external systems
- Services: Application orchestration
- API: REST endpoints for UI consumption
- Events: Event publishing and handling
"""

from .domain.entities import (
    SettingsNode,
    AgentDefinition,
    UserProfile,
    SettingsHierarchy,
    AgentInstance,
    AuditEntry,
)
from .domain.value_objects import (
    SettingsLevel,
    SettingValue,
    AgentStatus,
    PermissionLevel,
    ValidationRule,
    ConflictResolution,
)
from .services.settings_service import SettingsService
from .services.agent_service import AgentService
from .services.validation_service import ValidationService
from .api.settings_api import SettingsAPI
from .api.agent_api import AgentAPI

__all__ = [
    # Domain entities
    "SettingsNode",
    "AgentDefinition", 
    "UserProfile",
    "SettingsHierarchy",
    "AgentInstance",
    "AuditEntry",
    
    # Value objects
    "SettingsLevel",
    "SettingValue",
    "AgentStatus",
    "PermissionLevel",
    "ValidationRule",
    "ConflictResolution",
    
    # Services
    "SettingsService",
    "AgentService", 
    "ValidationService",
    
    # API
    "SettingsAPI",
    "AgentAPI",
]