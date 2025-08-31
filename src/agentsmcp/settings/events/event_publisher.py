"""
Event publisher implementations.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable
from collections import defaultdict

from .base_events import DomainEvent, EventHandler


class EventPublisher(ABC):
    """Abstract base class for event publishers."""
    
    @abstractmethod
    async def publish(self, event: DomainEvent) -> None:
        """Publish a domain event."""
        pass
    
    @abstractmethod
    def subscribe(self, handler: EventHandler) -> None:
        """Subscribe an event handler."""
        pass
    
    @abstractmethod
    def unsubscribe(self, handler: EventHandler) -> None:
        """Unsubscribe an event handler."""
        pass


class InMemoryEventPublisher(EventPublisher):
    """In-memory event publisher for local event handling."""
    
    def __init__(self):
        self.handlers: Dict[str, List[EventHandler]] = defaultdict(list)
        self.logger = logging.getLogger(__name__)
    
    async def publish(self, event: DomainEvent) -> None:
        """Publish an event to all registered handlers."""
        event_type = event.event_type
        
        # Get handlers for this specific event type
        handlers = self.handlers.get(event_type, [])
        
        # Also get handlers that handle all events (wildcard)
        handlers.extend(self.handlers.get("*", []))
        
        if not handlers:
            self.logger.debug(f"No handlers registered for event type: {event_type}")
            return
        
        # Execute all handlers concurrently
        tasks = []
        for handler in handlers:
            if handler.can_handle(event):
                tasks.append(self._handle_event_safely(handler, event))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def subscribe(self, handler: EventHandler) -> None:
        """Subscribe an event handler to specific event types."""
        for event_type in handler.handled_event_types:
            if handler not in self.handlers[event_type]:
                self.handlers[event_type].append(handler)
                self.logger.debug(f"Subscribed {handler.__class__.__name__} to {event_type}")
    
    def unsubscribe(self, handler: EventHandler) -> None:
        """Unsubscribe an event handler from all event types."""
        for event_type in handler.handled_event_types:
            if handler in self.handlers[event_type]:
                self.handlers[event_type].remove(handler)
                self.logger.debug(f"Unsubscribed {handler.__class__.__name__} from {event_type}")
    
    async def _handle_event_safely(self, handler: EventHandler, event: DomainEvent) -> None:
        """Handle an event with error handling."""
        try:
            await handler.handle(event)
        except Exception as e:
            self.logger.error(
                f"Error handling event {event.event_type} with {handler.__class__.__name__}: {e}",
                exc_info=True
            )