"""
Security-related domain events.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .base_events import DomainEvent
from ..domain.value_objects import PermissionLevel


@dataclass(frozen=True)
class SecurityViolationEvent(DomainEvent):
    """Event raised when a security violation is detected."""
    
    violation_type: str
    severity: str  # "low", "medium", "high", "critical"
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    
    @property
    def event_type(self) -> str:
        return "security_violation"


@dataclass(frozen=True)
class EncryptionKeyRotatedEvent(DomainEvent):
    """Event raised when encryption keys are rotated."""
    
    updated_secrets_count: int
    rotation_time: datetime
    
    @property
    def event_type(self) -> str:
        return "encryption_key_rotated"


@dataclass(frozen=True)
class PermissionGrantedEvent(DomainEvent):
    """Event raised when permission is granted to a user."""
    
    granter_id: str
    resource_type: str
    resource_id: str
    permission_level: PermissionLevel
    expires_at: Optional[datetime] = None
    
    @property
    def event_type(self) -> str:
        return "permission_granted"


@dataclass(frozen=True)
class PermissionRevokedEvent(DomainEvent):
    """Event raised when permission is revoked from a user."""
    
    revoker_id: str
    resource_type: str
    resource_id: str
    
    @property
    def event_type(self) -> str:
        return "permission_revoked"