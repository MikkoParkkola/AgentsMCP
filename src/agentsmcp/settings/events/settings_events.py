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
    
    node_id: str = None
    hierarchy_id: str = None
    setting_key: str = None
    old_value: Any = None
    new_value: Any = None
    
    @property
    def event_type(self) -> str:
        return "setting_changed"
    
    @property
    def aggregate_id(self) -> str:
        return self.node_id


@dataclass(frozen=True)
class SettingsNodeCreatedEvent(DomainEvent):
    """Event raised when a new settings node is created."""
    
    hierarchy_id: str = None
    node_id: str = None
    level: SettingsLevel = None
    name: str = None
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
    
    hierarchy_id: str = None
    node_id: str = None
    updated_fields: List[str] = None
    old_values: Dict[str, Any] = None
    new_values: Dict[str, Any] = None
    
    @property
    def event_type(self) -> str:
        return "settings_node_updated"
    
    @property
    def aggregate_id(self) -> str:
        return self.node_id


@dataclass(frozen=True)
class SettingsNodeDeletedEvent(DomainEvent):
    """Event raised when a settings node is deleted."""
    
    hierarchy_id: str = None
    node_id: str = None
    node_name: str = None
    level: SettingsLevel = None
    settings_count: int = None
    
    @property
    def event_type(self) -> str:
        return "settings_node_deleted"
    
    @property
    def aggregate_id(self) -> str:
        return self.node_id


@dataclass(frozen=True)
class SettingsHierarchyCreatedEvent(DomainEvent):
    """Event raised when a new settings hierarchy is created."""
    
    hierarchy_id: str = None
    name: str = None
    root_node_id: str = None
    
    @property
    def event_type(self) -> str:
        return "settings_hierarchy_created"
    
    @property
    def aggregate_id(self) -> str:
        return self.hierarchy_id


@dataclass(frozen=True)
class SettingsValidationFailedEvent(DomainEvent):
    """Event raised when settings validation fails."""
    
    node_id: str = None
    setting_key: str = None
    errors: List[Dict[str, Any]] = None
    
    @property
    def event_type(self) -> str:
        return "settings_validation_failed"
    
    @property
    def aggregate_id(self) -> str:
        return self.node_id


@dataclass(frozen=True)
class SettingsConflictDetectedEvent(DomainEvent):
    """Event raised when settings conflicts are detected."""
    
    hierarchy_id: str = None
    node_id: str = None
    conflicts: List[Dict[str, Any]] = None
    
    @property
    def event_type(self) -> str:
        return "settings_conflict_detected"
    
    @property
    def aggregate_id(self) -> str:
        return self.node_id


@dataclass(frozen=True)
class SettingsConflictResolvedEvent(DomainEvent):
    """Event raised when settings conflicts are resolved."""
    
    hierarchy_id: str = None
    node_id: str = None
    resolved_conflicts: List[Dict[str, Any]] = None
    resolution_strategy: str = None
    
    @property
    def event_type(self) -> str:
        return "settings_conflict_resolved"
    
    @property
    def aggregate_id(self) -> str:
        return self.node_id


@dataclass(frozen=True)
class SettingsImportedEvent(DomainEvent):
    """Event raised when settings are imported."""
    
    hierarchy_id: str = None
    import_source: str = None
    nodes_created: int = None
    settings_imported: int = None
    errors: List[Dict[str, Any]] = None
    
    @property
    def event_type(self) -> str:
        return "settings_imported"
    
    @property
    def aggregate_id(self) -> str:
        return self.hierarchy_id


@dataclass(frozen=True)
class SettingsExportedEvent(DomainEvent):
    """Event raised when settings are exported."""
    
    hierarchy_id: str = None
    export_format: str = None
    include_secrets: bool = None
    export_size: int = None
    
    @property
    def event_type(self) -> str:
        return "settings_exported"
    
    @property
    def aggregate_id(self) -> str:
        return self.hierarchy_id


@dataclass(frozen=True)
class SettingsCacheInvalidatedEvent(DomainEvent):
    """Event raised when settings cache is invalidated."""
    
    hierarchy_id: str = None
    node_id: Optional[str] = None
    reason: str = "manual"
    
    @property
    def event_type(self) -> str:
        return "settings_cache_invalidated"
    
    @property
    def aggregate_id(self) -> str:
        return self.hierarchy_id