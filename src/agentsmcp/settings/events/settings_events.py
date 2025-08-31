"""
Settings-related domain events.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .base_events import DomainEvent
from ..domain.value_objects import SettingsLevel


@dataclass(frozen=True)
class SettingChangedEvent(DomainEvent):
    """Event raised when a setting value is changed."""
    
    node_id: str
    hierarchy_id: str
    setting_key: str
    old_value: Any
    new_value: Any
    
    @property
    def event_type(self) -> str:
        return "setting_changed"
    
    @property
    def aggregate_id(self) -> str:
        return self.node_id


@dataclass(frozen=True)
class SettingsNodeCreatedEvent(DomainEvent):
    """Event raised when a new settings node is created."""
    
    hierarchy_id: str
    node_id: str
    level: SettingsLevel
    name: str
    parent_id: Optional[str] = None
    
    @property
    def event_type(self) -> str:
        return "settings_node_created"
    
    @property
    def aggregate_id(self) -> str:
        return self.node_id


@dataclass(frozen=True)
class SettingsNodeUpdatedEvent(DomainEvent):
    """Event raised when a settings node is updated."""
    
    hierarchy_id: str
    node_id: str
    updated_fields: List[str]
    old_values: Dict[str, Any]
    new_values: Dict[str, Any]
    
    @property
    def event_type(self) -> str:
        return "settings_node_updated"
    
    @property
    def aggregate_id(self) -> str:
        return self.node_id


@dataclass(frozen=True)
class SettingsNodeDeletedEvent(DomainEvent):
    """Event raised when a settings node is deleted."""
    
    hierarchy_id: str
    node_id: str
    node_name: str
    level: SettingsLevel
    settings_count: int
    
    @property
    def event_type(self) -> str:
        return "settings_node_deleted"
    
    @property
    def aggregate_id(self) -> str:
        return self.node_id


@dataclass(frozen=True)
class SettingsHierarchyCreatedEvent(DomainEvent):
    """Event raised when a new settings hierarchy is created."""
    
    hierarchy_id: str
    name: str
    root_node_id: str
    
    @property
    def event_type(self) -> str:
        return "settings_hierarchy_created"
    
    @property
    def aggregate_id(self) -> str:
        return self.hierarchy_id


@dataclass(frozen=True)
class SettingsValidationFailedEvent(DomainEvent):
    """Event raised when settings validation fails."""
    
    node_id: str
    setting_key: str
    errors: List[Dict[str, Any]]
    
    @property
    def event_type(self) -> str:
        return "settings_validation_failed"
    
    @property
    def aggregate_id(self) -> str:
        return self.node_id


@dataclass(frozen=True)
class SettingsConflictDetectedEvent(DomainEvent):
    """Event raised when settings conflicts are detected."""
    
    hierarchy_id: str
    node_id: str
    conflicts: List[Dict[str, Any]]
    
    @property
    def event_type(self) -> str:
        return "settings_conflict_detected"
    
    @property
    def aggregate_id(self) -> str:
        return self.node_id


@dataclass(frozen=True)
class SettingsConflictResolvedEvent(DomainEvent):
    """Event raised when settings conflicts are resolved."""
    
    hierarchy_id: str
    node_id: str
    resolved_conflicts: List[Dict[str, Any]]
    resolution_strategy: str
    
    @property
    def event_type(self) -> str:
        return "settings_conflict_resolved"
    
    @property
    def aggregate_id(self) -> str:
        return self.node_id


@dataclass(frozen=True)
class SettingsImportedEvent(DomainEvent):
    """Event raised when settings are imported."""
    
    hierarchy_id: str
    import_source: str
    nodes_created: int
    settings_imported: int
    errors: List[Dict[str, Any]]
    
    @property
    def event_type(self) -> str:
        return "settings_imported"
    
    @property
    def aggregate_id(self) -> str:
        return self.hierarchy_id


@dataclass(frozen=True)
class SettingsExportedEvent(DomainEvent):
    """Event raised when settings are exported."""
    
    hierarchy_id: str
    export_format: str
    include_secrets: bool
    export_size: int
    
    @property
    def event_type(self) -> str:
        return "settings_exported"
    
    @property
    def aggregate_id(self) -> str:
        return self.hierarchy_id


@dataclass(frozen=True)
class SettingsCacheInvalidatedEvent(DomainEvent):
    """Event raised when settings cache is invalidated."""
    
    hierarchy_id: str
    node_id: Optional[str] = None
    reason: str = "manual"
    
    @property
    def event_type(self) -> str:
        return "settings_cache_invalidated"
    
    @property
    def aggregate_id(self) -> str:
        return self.hierarchy_id