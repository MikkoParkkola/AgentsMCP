"""
Events system for the settings management system.

Provides event-driven architecture with real-time updates
and decoupled communication between components.
"""

from .base_events import DomainEvent, EventHandler
from .settings_events import (
    SettingChangedEvent,
    SettingsNodeCreatedEvent,
    SettingsValidationFailedEvent,
    SettingsConflictDetectedEvent,
)
from .agent_events import (
    AgentDefinitionCreatedEvent,
    AgentDefinitionUpdatedEvent,
    AgentInstanceStartedEvent,
    AgentInstanceStoppedEvent,
    AgentPerformanceUpdatedEvent,
)
from .security_events import (
    SecurityViolationEvent,
    EncryptionKeyRotatedEvent,
    PermissionGrantedEvent,
    PermissionRevokedEvent,
)
from .validation_events import (
    ValidationCompletedEvent,
    ValidationFailedEvent,
    SmartSuggestionGeneratedEvent,
)
from .event_publisher import EventPublisher, InMemoryEventPublisher

__all__ = [
    # Base
    "DomainEvent",
    "EventHandler",
    
    # Settings events
    "SettingChangedEvent",
    "SettingsNodeCreatedEvent",
    "SettingsValidationFailedEvent",
    "SettingsConflictDetectedEvent",
    
    # Agent events
    "AgentDefinitionCreatedEvent",
    "AgentDefinitionUpdatedEvent",
    "AgentInstanceStartedEvent",
    "AgentInstanceStoppedEvent",
    "AgentPerformanceUpdatedEvent",
    
    # Security events
    "SecurityViolationEvent",
    "EncryptionKeyRotatedEvent",
    "PermissionGrantedEvent",
    "PermissionRevokedEvent",
    
    # Validation events
    "ValidationCompletedEvent",
    "ValidationFailedEvent",
    "SmartSuggestionGeneratedEvent",
    
    # Publishers
    "EventPublisher",
    "InMemoryEventPublisher",
]