"""
Domain entities for the settings management system.

Entities are objects with identity that can change over time
while maintaining their identity. They represent the core
business objects in the system.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field

from .value_objects import (
    SettingsLevel,
    SettingValue,
    AgentStatus,
    PermissionLevel,
    PermissionGrant,
    ValidationRule,
    AgentCapability,
    InstructionTemplate,
    ConflictResolution,
    EventMetadata,
    SettingsValidationError,
    SettingsConflictError,
)


@dataclass
class SettingsNode:
    """
    A node in the settings hierarchy tree.
    
    Represents a collection of settings at a specific level
    (global, user, session, or agent) with inheritance support.
    """
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    level: SettingsLevel = SettingsLevel.GLOBAL
    name: str = ""
    parent_id: Optional[str] = None
    settings: Dict[str, SettingValue] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    version: int = 1
    
    def get_setting(self, key: str) -> Optional[SettingValue]:
        """Get a setting value by key."""
        return self.settings.get(key)
    
    def set_setting(self, key: str, value: SettingValue) -> None:
        """Set a setting value."""
        self.settings[key] = value
        self.updated_at = datetime.utcnow()
        self.version += 1
    
    def remove_setting(self, key: str) -> bool:
        """Remove a setting and return True if it existed."""
        if key in self.settings:
            del self.settings[key]
            self.updated_at = datetime.utcnow()
            self.version += 1
            return True
        return False
    
    def get_all_keys(self) -> Set[str]:
        """Get all setting keys in this node."""
        return set(self.settings.keys())
    
    def apply_settings(self, updates: Dict[str, SettingValue]) -> None:
        """Apply multiple setting updates."""
        for key, value in updates.items():
            self.set_setting(key, value)


@dataclass  
class SettingsHierarchy:
    """
    Manages the complete hierarchy of settings nodes with inheritance.
    
    Provides resolution of effective settings by walking up the hierarchy
    and applying inheritance rules.
    """
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "default"
    root_node_id: str = ""
    nodes: Dict[str, SettingsNode] = field(default_factory=dict)
    validation_rules: List[ValidationRule] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def add_node(self, node: SettingsNode) -> None:
        """Add a node to the hierarchy."""
        self.nodes[node.id] = node
        if not self.root_node_id and node.level == SettingsLevel.GLOBAL:
            self.root_node_id = node.id
        self.updated_at = datetime.utcnow()
    
    def get_node(self, node_id: str) -> Optional[SettingsNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def get_nodes_by_level(self, level: SettingsLevel) -> List[SettingsNode]:
        """Get all nodes at a specific level."""
        return [node for node in self.nodes.values() if node.level == level]
    
    def resolve_setting(self, key: str, node_id: str) -> Optional[SettingValue]:
        """
        Resolve a setting value by walking up the hierarchy.
        
        Returns the most specific (lowest level) value for the setting.
        """
        current_node = self.get_node(node_id)
        
        while current_node:
            if key in current_node.settings:
                return current_node.settings[key]
            
            # Move up the hierarchy
            if current_node.parent_id:
                current_node = self.get_node(current_node.parent_id)
            else:
                break
                
        return None
    
    def resolve_all_settings(self, node_id: str) -> Dict[str, SettingValue]:
        """
        Resolve all effective settings for a node by walking up the hierarchy.
        
        Lower-level settings override higher-level ones.
        """
        result: Dict[str, SettingValue] = {}
        path = self._get_hierarchy_path(node_id)
        
        # Apply settings from root to target (higher levels first)
        for node_id in reversed(path):
            node = self.get_node(node_id)
            if node:
                for key, value in node.settings.items():
                    result[key] = value
        
        return result
    
    def detect_conflicts(self, node_id: str) -> List[Dict[str, Any]]:
        """
        Detect settings conflicts in the hierarchy.
        
        Returns list of conflicts with details about conflicting values.
        """
        conflicts = []
        path = self._get_hierarchy_path(node_id)
        
        # Collect all settings in the path
        all_settings: Dict[str, List[tuple[str, SettingValue]]] = {}
        
        for current_node_id in path:
            node = self.get_node(current_node_id)
            if node:
                for key, value in node.settings.items():
                    if key not in all_settings:
                        all_settings[key] = []
                    all_settings[key].append((current_node_id, value))
        
        # Find conflicts (same key with different values)
        for key, values in all_settings.items():
            if len(values) > 1:
                unique_values = set(v[1].value for v in values)
                if len(unique_values) > 1:
                    conflicts.append({
                        "key": key,
                        "values": [{"node_id": nid, "value": val.value, "level": self.get_node(nid).level} 
                                 for nid, val in values],
                        "resolution_suggestion": ConflictResolution.OVERRIDE
                    })
        
        return conflicts
    
    def validate_settings(self, node_id: str) -> List[Dict[str, Any]]:
        """
        Validate all effective settings for a node against validation rules.
        
        Returns list of validation errors/warnings.
        """
        effective_settings = self.resolve_all_settings(node_id)
        errors = []
        
        for rule in self.validation_rules:
            if rule.setting_key in effective_settings:
                setting_value = effective_settings[rule.setting_key]
                is_valid, message = rule.validate(setting_value)
                
                if not is_valid:
                    errors.append({
                        "rule_id": rule.id,
                        "setting_key": rule.setting_key,
                        "message": message,
                        "severity": rule.severity,
                        "value": setting_value.value
                    })
        
        return errors
    
    def _get_hierarchy_path(self, node_id: str) -> List[str]:
        """Get the path from root to the specified node."""
        path = []
        current_node = self.get_node(node_id)
        
        while current_node:
            path.append(current_node.id)
            if current_node.parent_id:
                current_node = self.get_node(current_node.parent_id)
            else:
                break
        
        return path


@dataclass
class UserProfile:
    """
    User profile with preferences and permissions.
    """
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    username: str = ""
    email: str = ""
    full_name: str = ""
    organization_id: Optional[str] = None
    preferences: Dict[str, Any] = field(default_factory=dict)
    permissions: List[PermissionGrant] = field(default_factory=list)
    settings_node_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_active: Optional[datetime] = None
    is_active: bool = True
    
    def has_permission(self, resource_type: str, resource_id: str, 
                      required_level: PermissionLevel) -> bool:
        """Check if user has required permission for a resource."""
        level_hierarchy = {
            PermissionLevel.READ: 1,
            PermissionLevel.WRITE: 2, 
            PermissionLevel.ADMIN: 3,
            PermissionLevel.OWNER: 4
        }
        
        required_value = level_hierarchy[required_level]
        
        for grant in self.permissions:
            if (grant.resource_type == resource_type and 
                grant.resource_id == resource_id and
                (grant.expires_at is None or grant.expires_at > datetime.utcnow())):
                
                granted_value = level_hierarchy[grant.permission_level]
                if granted_value >= required_value:
                    return True
        
        return False
    
    def add_permission(self, grant: PermissionGrant) -> None:
        """Add a permission grant."""
        self.permissions.append(grant)
        self.updated_at = datetime.utcnow()
    
    def remove_permission(self, resource_type: str, resource_id: str) -> bool:
        """Remove permission for a resource."""
        original_count = len(self.permissions)
        self.permissions = [
            p for p in self.permissions 
            if not (p.resource_type == resource_type and p.resource_id == resource_id)
        ]
        
        if len(self.permissions) < original_count:
            self.updated_at = datetime.utcnow()
            return True
        return False


@dataclass
class AgentDefinition:
    """
    Definition of an agent with its configuration and capabilities.
    """
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    status: AgentStatus = AgentStatus.DRAFT
    owner_id: str = ""
    organization_id: Optional[str] = None
    
    # Agent configuration
    base_model: str = ""
    instruction_template: Optional[InstructionTemplate] = None
    capabilities: List[AgentCapability] = field(default_factory=list)
    settings_schema: Dict[str, Any] = field(default_factory=dict)
    default_settings: Dict[str, SettingValue] = field(default_factory=dict)
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    category: str = "general"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    published_at: Optional[datetime] = None
    
    # Usage statistics
    usage_count: int = 0
    success_rate: float = 0.0
    average_response_time: float = 0.0
    
    def add_capability(self, capability: AgentCapability) -> None:
        """Add a capability to the agent."""
        # Remove existing capability with same name
        self.capabilities = [c for c in self.capabilities if c.name != capability.name]
        self.capabilities.append(capability)
        self.updated_at = datetime.utcnow()
    
    def remove_capability(self, capability_name: str) -> bool:
        """Remove a capability by name."""
        original_count = len(self.capabilities)
        self.capabilities = [c for c in self.capabilities if c.name != capability_name]
        
        if len(self.capabilities) < original_count:
            self.updated_at = datetime.utcnow()
            return True
        return False
    
    def update_status(self, new_status: AgentStatus) -> None:
        """Update agent status."""
        if self.status != new_status:
            self.status = new_status
            self.updated_at = datetime.utcnow()
            
            if new_status == AgentStatus.ACTIVE and self.published_at is None:
                self.published_at = datetime.utcnow()
    
    def validate_settings(self, settings: Dict[str, Any]) -> List[str]:
        """Validate settings against the agent's schema."""
        errors = []
        
        # Check required settings
        for key, schema in self.settings_schema.items():
            if schema.get("required", False) and key not in settings:
                errors.append(f"Required setting '{key}' is missing")
        
        # Check setting types and constraints
        for key, value in settings.items():
            if key in self.settings_schema:
                schema = self.settings_schema[key]
                expected_type = schema.get("type")
                
                if expected_type and not isinstance(value, type(expected_type)):
                    errors.append(f"Setting '{key}' must be of type {expected_type}")
        
        return errors


@dataclass
class AgentInstance:
    """
    A running instance of an agent with specific settings.
    """
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_definition_id: str = ""
    user_id: str = ""
    session_id: Optional[str] = None
    name: str = ""
    
    # Configuration
    settings: Dict[str, SettingValue] = field(default_factory=dict)
    effective_settings: Dict[str, SettingValue] = field(default_factory=dict)
    
    # Runtime state
    status: AgentStatus = AgentStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    
    # Performance metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time: float = 0.0
    
    def start(self) -> None:
        """Start the agent instance."""
        self.status = AgentStatus.ACTIVE
        self.started_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
    
    def stop(self) -> None:
        """Stop the agent instance."""
        if self.status == AgentStatus.ACTIVE:
            self.status = AgentStatus.ARCHIVED
            self.stopped_at = datetime.utcnow()
    
    def record_request(self, response_time: float, success: bool) -> None:
        """Record a request and its outcome."""
        self.total_requests += 1
        self.total_response_time += response_time
        self.last_activity = datetime.utcnow()
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    @property
    def average_response_time(self) -> float:
        """Calculate average response time."""
        if self.total_requests == 0:
            return 0.0
        return self.total_response_time / self.total_requests


@dataclass
class AuditEntry:
    """
    Audit log entry for tracking changes and access.
    """
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    user_id: str = ""
    action: str = ""  # "create", "read", "update", "delete", "access"
    resource_type: str = ""  # "settings", "agent", "user"
    resource_id: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    
    # Change tracking
    old_value: Optional[Dict[str, Any]] = None
    new_value: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Ensure required fields are present."""
        if not self.user_id:
            raise ValueError("user_id is required for audit entry")
        if not self.action:
            raise ValueError("action is required for audit entry")
        if not self.resource_type:
            raise ValueError("resource_type is required for audit entry")