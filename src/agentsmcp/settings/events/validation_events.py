"""
Validation-related domain events.
"""

from __future__ import annotations

from typing import Dict, Any, List
from dataclasses import dataclass

from .base_events import DomainEvent


@dataclass(frozen=True)
class ValidationCompletedEvent(DomainEvent):
    """Event raised when validation completes successfully."""
    
    node_id: str = None
    setting_key: str = None
    validation_result: Dict[str, Any] = None
    
    @property
    def event_type(self) -> str:
        return "validation_completed"
    
    @property
    def aggregate_id(self) -> str:
        return self.node_id


@dataclass(frozen=True)
class ValidationFailedEvent(DomainEvent):
    """Event raised when validation fails."""
    
    node_id: str = None
    setting_key: str = None
    errors: List[Dict[str, Any]] = None
    suggestions: List[Dict[str, Any]] = None
    
    @property
    def event_type(self) -> str:
        return "validation_failed"
    
    @property
    def aggregate_id(self) -> str:
        return self.node_id


@dataclass(frozen=True)
class SmartSuggestionGeneratedEvent(DomainEvent):
    """Event raised when smart suggestions are generated."""
    
    context: Dict[str, Any] = None
    suggestions: List[Dict[str, Any]] = None
    
    @property
    def event_type(self) -> str:
        return "smart_suggestion_generated"