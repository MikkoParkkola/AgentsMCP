"""Coordination models for CLI v3 cross-modal interface system.

This module defines Pydantic data structures for managing state synchronization
and capability coordination across CLI, TUI, and WebUI interfaces.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from .command_models import ExecutionMode


class InterfaceMode(str, Enum):
    """Interface modes for cross-modal coordination."""
    CLI = "cli"
    TUI = "tui"
    WEB_UI = "web_ui"
    API = "api"


class SyncStatus(str, Enum):
    """State synchronization status."""
    SYNCED = "synced"
    SYNCING = "syncing"
    CONFLICT = "conflict"
    FAILED = "failed"
    OFFLINE = "offline"


class ConflictResolution(str, Enum):
    """Conflict resolution strategies."""
    LAST_WRITE_WINS = "last_write_wins"
    USER_PROMPT = "user_prompt"
    MERGE = "merge"
    REJECT = "reject"


class CapabilityType(str, Enum):
    """Types of interface capabilities."""
    INPUT = "input"
    OUTPUT = "output"
    INTERACTION = "interaction"
    STORAGE = "storage"
    NETWORK = "network"
    SECURITY = "security"


class Feature(BaseModel):
    """Interface feature definition."""
    
    name: str = Field(..., min_length=1, description="Feature identifier")
    display_name: str = Field(..., min_length=1, description="Human-readable name")
    description: str = Field(..., min_length=1, description="Feature description")
    capability_type: CapabilityType = Field(..., description="Type of capability")
    required_permissions: Set[str] = Field(default_factory=set, description="Required permissions")
    dependencies: Set[str] = Field(default_factory=set, description="Feature dependencies")
    performance_impact: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Performance impact score (0=none, 1=high)"
    )
    availability: Dict[InterfaceMode, bool] = Field(
        default_factory=dict,
        description="Feature availability by interface mode"
    )


class SessionContext(BaseModel):
    """User session context for state synchronization."""
    
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str = Field(..., description="User identifier")
    current_interface: InterfaceMode = Field(..., description="Active interface mode")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    active_commands: List[str] = Field(default_factory=list, description="Currently running commands")
    command_history: List[str] = Field(default_factory=list, max_items=100)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    workspace_state: Dict[str, Any] = Field(default_factory=dict, description="Application state")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StateChange(BaseModel):
    """Individual state change record."""
    
    change_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str = Field(..., description="Session that made the change")
    interface_mode: InterfaceMode = Field(..., description="Interface that made change")
    key_path: str = Field(..., description="Dot-separated path to changed value")
    old_value: Any = Field(None, description="Previous value")
    new_value: Any = Field(..., description="New value")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    user_id: str = Field(..., description="User who made the change")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConflictInfo(BaseModel):
    """State synchronization conflict information."""
    
    conflict_id: str = Field(default_factory=lambda: str(uuid4()))
    key_path: str = Field(..., description="Path where conflict occurred")
    local_change: StateChange = Field(..., description="Local change")
    remote_changes: List[StateChange] = Field(..., description="Conflicting remote changes")
    resolution_strategy: Optional[ConflictResolution] = Field(None)
    resolved_value: Any = Field(None, description="Final resolved value")
    resolution_metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[datetime] = Field(None)


class SharedState(BaseModel):
    """Synchronized state across interfaces."""
    
    session_id: str = Field(..., description="Session identifier")
    version: int = Field(default=1, ge=1, description="State version for optimistic locking")
    last_modified: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sync_status: SyncStatus = Field(default=SyncStatus.SYNCED)
    active_context: SessionContext = Field(..., description="Current session context")
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    interface_states: Dict[InterfaceMode, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Interface-specific state"
    )
    conflicts: List[ConflictInfo] = Field(default_factory=list, description="Unresolved conflicts")
    checksum: Optional[str] = Field(None, description="State integrity checksum")


class ModalSwitchRequest(BaseModel):
    """Request to switch interface modes."""
    
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str = Field(..., description="Session to switch")
    from_mode: InterfaceMode = Field(..., description="Current interface mode")
    to_mode: InterfaceMode = Field(..., description="Target interface mode")
    preserve_state: bool = Field(default=True, description="Whether to preserve session state")
    transfer_context: bool = Field(default=True, description="Whether to transfer command context")
    requested_by: str = Field(..., description="User requesting switch")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('to_mode')
    @classmethod
    def validate_mode_different(cls, v, info):
        """Ensure target mode is different from source."""
        if 'from_mode' in info.data and v == info.data['from_mode']:
            raise ValueError("Target mode must be different from source mode")
        return v


class StateSync(BaseModel):
    """State synchronization request."""
    
    sync_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str = Field(..., description="Session to synchronize")
    interface_mode: InterfaceMode = Field(..., description="Requesting interface")
    current_version: int = Field(ge=1, description="Current state version")
    changes: List[StateChange] = Field(default_factory=list, description="Local changes to sync")
    force_sync: bool = Field(default=False, description="Force sync even with conflicts")
    conflict_resolution: ConflictResolution = Field(
        default=ConflictResolution.USER_PROMPT,
        description="How to handle conflicts"
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CapabilityQuery(BaseModel):
    """Query for interface capabilities."""
    
    query_id: str = Field(default_factory=lambda: str(uuid4()))
    interface: InterfaceMode = Field(..., description="Target interface mode")
    feature: Optional[str] = Field(None, description="Specific feature to check")
    category: Optional[CapabilityType] = Field(None, description="Capability category filter")
    include_dependencies: bool = Field(default=False, description="Include feature dependencies")
    check_permissions: bool = Field(default=True, description="Verify user permissions")
    user_id: Optional[str] = Field(None, description="User for permission checks")


class TransitionResult(BaseModel):
    """Result of interface mode transition."""
    
    request_id: str = Field(..., description="Original request ID")
    success: bool = Field(..., description="Whether transition succeeded")
    from_mode: InterfaceMode = Field(..., description="Source interface mode")
    to_mode: InterfaceMode = Field(..., description="Target interface mode")
    preserved_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Context preserved during transition"
    )
    lost_context: List[str] = Field(
        default_factory=list,
        description="Context that couldn't be preserved"
    )
    warnings: List[str] = Field(default_factory=list, description="Transition warnings")
    errors: List[str] = Field(default_factory=list, description="Transition errors")
    duration_ms: int = Field(ge=0, description="Transition duration in milliseconds")
    new_session_id: Optional[str] = Field(None, description="New session ID if changed")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @model_validator(mode='after')
    def validate_success_consistency(self):
        """Ensure success field matches error state."""
        if not self.success and not self.errors:
            raise ValueError("Failed transitions must have error messages")
        elif self.success and self.errors:
            raise ValueError("Successful transitions cannot have errors")
        return self


class SyncResult(BaseModel):
    """Result of state synchronization operation."""
    
    sync_id: str = Field(..., description="Original sync request ID")
    success: bool = Field(..., description="Whether sync succeeded")
    new_version: int = Field(ge=1, description="New state version after sync")
    conflicts_resolved: int = Field(default=0, ge=0, description="Number of conflicts resolved")
    conflicts_remaining: int = Field(default=0, ge=0, description="Number of unresolved conflicts")
    changes_applied: int = Field(default=0, ge=0, description="Number of changes applied")
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    duration_ms: int = Field(ge=0, description="Sync duration in milliseconds")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CapabilityInfo(BaseModel):
    """Interface capability information."""
    
    interface: InterfaceMode = Field(..., description="Interface mode")
    available_features: List[Feature] = Field(
        default_factory=list,
        description="Features available in this interface"
    )
    supported_capabilities: Dict[CapabilityType, List[str]] = Field(
        default_factory=dict,
        description="Capabilities grouped by type"
    )
    performance_profile: Dict[str, float] = Field(
        default_factory=dict,
        description="Performance characteristics"
    )
    limitations: List[str] = Field(default_factory=list, description="Known limitations")
    recommended_for: List[str] = Field(
        default_factory=list,
        description="Use cases this interface is optimized for"
    )
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# Custom exceptions for coordination system
class CoordinationError(Exception):
    """Base exception for coordination errors."""
    pass


class SyncFailedError(CoordinationError):
    """Raised when state synchronization fails."""
    pass


class ModeNotSupportedError(CoordinationError):
    """Raised when requested interface mode is not supported."""
    pass


class StateLossError(CoordinationError):
    """Raised when state cannot be preserved during transition."""
    pass


class CapabilityMismatchError(CoordinationError):
    """Raised when required capabilities are not available."""
    pass


class ConflictResolutionError(CoordinationError):
    """Raised when conflicts cannot be resolved."""
    pass