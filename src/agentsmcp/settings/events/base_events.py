"""
Base event classes and interfaces for the event system.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DomainEvent(ABC):
    """
    Base class for all domain events.
    
    Domain events represent something that happened in the domain
    that other parts of the system might be interested in.
    """
    
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    @abstractmethod
    def event_type(self) -> str:
        """Return the event type identifier."""
        pass
    
    @property
    def aggregate_id(self) -> Optional[str]:
        """Return the ID of the aggregate that raised this event."""
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        result = {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "user_id": self.user_id,
            "metadata": self.metadata,
        }
        
        if self.aggregate_id:
            result["aggregate_id"] = self.aggregate_id
        
        # Add event-specific fields
        for key, value in self.__dict__.items():
            if key not in result and not key.startswith("_"):
                if isinstance(value, datetime):
                    result[key] = value.isoformat()
                else:
                    result[key] = value
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DomainEvent:
        """Create event instance from dictionary."""
        # This would need to be implemented by each concrete event type
        raise NotImplementedError("Subclasses must implement from_dict")


class EventHandler(ABC):
    """
    Base class for event handlers.
    
    Event handlers process domain events and can trigger
    side effects or update read models.
    """
    
    @property
    @abstractmethod
    def handled_event_types(self) -> set[str]:
        """Return the set of event types this handler can process."""
        pass
    
    @abstractmethod
    async def handle(self, event: DomainEvent) -> None:
        """Handle a domain event."""
        pass
    
    def can_handle(self, event: DomainEvent) -> bool:
        """Check if this handler can process the given event."""
        return event.event_type in self.handled_event_types