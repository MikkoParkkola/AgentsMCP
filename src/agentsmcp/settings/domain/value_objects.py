"""
Value objects for the settings management system.

Value objects are immutable objects that represent concepts
in the domain that are defined by their attributes rather 
than identity.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, ConfigDict, validator


class SettingsLevel(str, Enum):
    """Hierarchy levels for settings inheritance."""
    GLOBAL = "global"
    ORGANIZATION = "organization"  
    USER = "user"
    SESSION = "session"
    AGENT = "agent"


class AgentStatus(str, Enum):
    """Status of an agent definition or instance."""
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    ERROR = "error"


class PermissionLevel(str, Enum):
    """Permission levels for access control."""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    OWNER = "owner"


class ConflictResolution(str, Enum):
    """Strategies for resolving setting conflicts."""
    OVERRIDE = "override"  # Higher level wins
    MERGE = "merge"  # Attempt to merge values
    WARN = "warn"  # Keep current, issue warning
    ERROR = "error"  # Fail with error


class SettingType(str, Enum):
    """Types of settings values."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    SECRET = "secret"  # Encrypted storage required


@dataclass(frozen=True)
class SettingValue:
    """Immutable representation of a setting value."""
    
    value: Any
    type: SettingType
    encrypted: bool = False
    source_level: SettingsLevel = SettingsLevel.GLOBAL
    last_modified: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate setting value matches declared type."""
        if self.type == SettingType.SECRET and not self.encrypted:
            raise ValueError("Secret values must be encrypted")
            
        # Type validation
        expected_types = {
            SettingType.STRING: str,
            SettingType.INTEGER: int,
            SettingType.FLOAT: (int, float),
            SettingType.BOOLEAN: bool,
            SettingType.ARRAY: (list, tuple),
            SettingType.OBJECT: dict,
        }
        
        if self.type in expected_types and self.type != SettingType.SECRET:
            expected = expected_types[self.type]
            if not isinstance(self.value, expected):
                raise ValueError(f"Value {self.value} is not of type {self.type}")


@dataclass(frozen=True)
class ValidationRule:
    """Rule for validating setting values."""
    
    id: str
    setting_key: str
    rule_type: str  # "required", "regex", "range", "enum", "custom"
    parameters: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    severity: str = "error"  # "error", "warning", "info"
    
    def validate(self, value: SettingValue) -> tuple[bool, str]:
        """Validate a setting value against this rule."""
        if self.rule_type == "required":
            is_valid = value.value is not None and value.value != ""
            message = self.error_message or f"Setting '{self.setting_key}' is required"
            
        elif self.rule_type == "regex":
            import re
            pattern = self.parameters.get("pattern", ".*")
            is_valid = bool(re.match(pattern, str(value.value)))
            message = self.error_message or f"Setting '{self.setting_key}' does not match pattern {pattern}"
            
        elif self.rule_type == "range":
            min_val = self.parameters.get("min")
            max_val = self.parameters.get("max")
            val = value.value
            is_valid = True
            
            if min_val is not None and val < min_val:
                is_valid = False
                message = f"Setting '{self.setting_key}' must be >= {min_val}"
            elif max_val is not None and val > max_val:
                is_valid = False
                message = f"Setting '{self.setting_key}' must be <= {max_val}"
            else:
                message = ""
                
        elif self.rule_type == "enum":
            allowed = self.parameters.get("values", [])
            is_valid = value.value in allowed
            message = self.error_message or f"Setting '{self.setting_key}' must be one of {allowed}"
            
        else:
            # Custom validation would be handled by domain service
            is_valid = True
            message = ""
            
        return is_valid, message


@dataclass(frozen=True)
class AgentCapability:
    """Capability that an agent supports."""
    
    name: str
    version: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    required: bool = True


@dataclass(frozen=True)
class InstructionTemplate:
    """Template for agent instructions."""
    
    id: str
    name: str
    version: str
    content: str
    variables: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def render(self, variables: Dict[str, Any]) -> str:
        """Render template with provided variables."""
        result = self.content
        for var_name, value in variables.items():
            if var_name in self.variables:
                result = result.replace(f"{{{{{var_name}}}}}", str(value))
        return result


@dataclass(frozen=True)
class PermissionGrant:
    """Permission granted to a user for a resource."""
    
    user_id: str
    resource_type: str  # "settings", "agent", "organization"
    resource_id: str
    permission_level: PermissionLevel
    granted_by: str
    granted_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None


@dataclass(frozen=True)
class EventMetadata:
    """Metadata for domain events."""
    
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    user_id: Optional[str] = None
    correlation_id: Optional[str] = None
    source: str = "settings_system"
    version: str = "1.0"


class SettingsValidationError(Exception):
    """Raised when settings validation fails."""
    
    def __init__(self, setting_key: str, message: str, severity: str = "error"):
        self.setting_key = setting_key
        self.message = message
        self.severity = severity
        super().__init__(f"Validation failed for '{setting_key}': {message}")


class SettingsConflictError(Exception):
    """Raised when settings conflicts cannot be resolved."""
    
    def __init__(self, conflicts: List[Dict[str, Any]]):
        self.conflicts = conflicts
        super().__init__(f"Settings conflicts detected: {len(conflicts)} conflicts")


class AgentDefinitionError(Exception):
    """Raised when agent definition operations fail."""
    pass


class PermissionDeniedError(Exception):
    """Raised when user lacks required permissions."""
    
    def __init__(self, user_id: str, resource: str, required_permission: PermissionLevel):
        self.user_id = user_id
        self.resource = resource
        self.required_permission = required_permission
        super().__init__(f"User {user_id} lacks {required_permission} permission for {resource}")