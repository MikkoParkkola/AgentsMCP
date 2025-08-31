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
    
    node_id: str
    setting_key: str
    validation_result: Dict[str, Any]
    
    @property
    def event_type(self) -> str:
        return "validation_completed"
    
    @property
    def aggregate_id(self) -> str:
        return self.node_id


@dataclass(frozen=True)
class ValidationFailedEvent(DomainEvent):
    """Event raised when validation fails."""
    
    node_id: str
    setting_key: str
    errors: List[Dict[str, Any]]
    suggestions: List[Dict[str, Any]]
    
    @property
    def event_type(self) -> str:
        return "validation_failed"
    
    @property
    def aggregate_id(self) -> str:
        return self.node_id


@dataclass(frozen=True)
class SmartSuggestionGeneratedEvent(DomainEvent):
    """Event raised when smart suggestions are generated."""
    
    context: Dict[str, Any]
    suggestions: List[Dict[str, Any]]
    
    @property
    def event_type(self) -> str:
        return "smart_suggestion_generated"