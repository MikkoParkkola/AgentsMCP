"""
Simple async event handling system.

Provides clean event propagation for keyboard events, resize events, 
and application events without blocking or causing deadlocks.
"""

import asyncio
import weakref
import logging
from typing import Any, Optional, Callable, Dict, List, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import inspect


logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of events that can be handled."""
    KEYBOARD = "keyboard"
    RESIZE = "resize"
    APPLICATION = "application"
    TIMER = "timer"
    CUSTOM = "custom"


@dataclass
class Event:
    """Represents an event in the system."""
    event_type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None
    handled: bool = False
    propagate: bool = True


class EventHandler:
    """Base class for event handlers."""
    
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self.enabled = True
    
    async def handle_event(self, event: Event) -> bool:
        """
        Handle an event.
        
        Args:
            event: Event to handle
            
        Returns:
            True if event was handled and should stop propagation
        """
        return False
    
    def can_handle(self, event: Event) -> bool:
        """Check if this handler can handle the given event."""
        return self.enabled


class AsyncEventSystem:
    """
    Simple async event handling system.
    
    Provides event propagation without blocking or causing deadlocks.
    """
    
    def __init__(self, max_handler_timeout: float = 5.0):
        """
        Initialize the event system.
        
        Args:
            max_handler_timeout: Maximum time to wait for a handler (seconds)
        """
        self._handlers: Dict[EventType, List[weakref.ref]] = {}
        self._running = False
        self._event_queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._max_handler_timeout = max_handler_timeout
        self._stats = {
            'events_processed': 0,
            'events_dropped': 0,
            'handler_timeouts': 0,
            'handler_errors': 0
        }
        
    def add_handler(self, event_type: EventType, handler: EventHandler):
        """
        Add an event handler for a specific event type.
        
        Args:
            event_type: Type of events to handle
            handler: Handler instance
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        
        # Use weak reference to prevent memory leaks
        self._handlers[event_type].append(weakref.ref(handler))
        logger.debug(f"Added handler {handler.name} for {event_type.value}")
    
    def remove_handler(self, event_type: EventType, handler: EventHandler):
        """
        Remove an event handler.
        
        Args:
            event_type: Type of events the handler was registered for
            handler: Handler instance to remove
        """
        if event_type not in self._handlers:
            return
        
        # Find and remove the weak reference
        handlers = self._handlers[event_type]
        for i, weak_handler in enumerate(handlers):
            if weak_handler() is handler:
                handlers.pop(i)
                logger.debug(f"Removed handler {handler.name} for {event_type.value}")
                break
    
    def _cleanup_dead_handlers(self):
        """Remove weak references to dead handlers."""
        for event_type in list(self._handlers.keys()):
            handlers = self._handlers[event_type]
            alive_handlers = [h for h in handlers if h() is not None]
            if len(alive_handlers) != len(handlers):
                logger.debug(f"Cleaned up {len(handlers) - len(alive_handlers)} dead handlers for {event_type.value}")
                self._handlers[event_type] = alive_handlers
    
    async def emit_event(self, event: Event) -> bool:
        """
        Emit an event asynchronously.
        
        Args:
            event: Event to emit
            
        Returns:
            True if event was queued successfully
        """
        if not self._running:
            logger.warning("Event system not running, dropping event")
            self._stats['events_dropped'] += 1
            return False
        
        try:
            # Non-blocking queue put with timeout
            await asyncio.wait_for(
                self._event_queue.put(event), 
                timeout=0.1
            )
            return True
        except asyncio.TimeoutError:
            logger.warning("Event queue full, dropping event")
            self._stats['events_dropped'] += 1
            return False
    
    async def emit_event_sync(self, event: Event) -> bool:
        """
        Emit an event synchronously (process immediately).
        
        Args:
            event: Event to emit
            
        Returns:
            True if event was handled
        """
        return await self._process_event(event)
    
    async def _process_event(self, event: Event) -> bool:
        """
        Process a single event by calling all registered handlers.
        
        Args:
            event: Event to process
            
        Returns:
            True if any handler handled the event
        """
        if event.event_type not in self._handlers:
            return False
        
        # Clean up dead handlers periodically
        self._cleanup_dead_handlers()
        
        handlers = self._handlers[event.event_type]
        handled_by_any = False
        
        for weak_handler in handlers[:]:  # Copy list to avoid modification during iteration
            handler = weak_handler()
            if handler is None:
                continue
                
            if not handler.can_handle(event):
                continue
            
            try:
                # Run handler with timeout to prevent blocking
                handled = await asyncio.wait_for(
                    handler.handle_event(event),
                    timeout=self._max_handler_timeout
                )
                
                if handled:
                    handled_by_any = True
                    event.handled = True
                    
                    # Stop propagation if handler consumed the event
                    if not event.propagate:
                        break
                        
            except asyncio.TimeoutError:
                logger.warning(f"Handler {handler.name} timed out processing {event.event_type.value} event")
                self._stats['handler_timeouts'] += 1
                
            except Exception as e:
                logger.error(f"Handler {handler.name} error processing {event.event_type.value} event: {e}")
                self._stats['handler_errors'] += 1
        
        self._stats['events_processed'] += 1
        return handled_by_any
    
    async def _worker_loop(self):
        """Main event processing worker loop."""
        logger.debug("Event system worker started")
        
        while self._running:
            try:
                # Get event from queue with timeout
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0
                )
                
                await self._process_event(event)
                self._event_queue.task_done()
                
            except asyncio.TimeoutError:
                # Normal timeout, continue loop
                continue
                
            except Exception as e:
                logger.error(f"Error in event worker loop: {e}")
                await asyncio.sleep(0.1)  # Brief pause on error
    
    async def start(self):
        """Start the event system."""
        if self._running:
            return
            
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.debug("Event system started")
    
    async def stop(self):
        """Stop the event system."""
        if not self._running:
            return
            
        self._running = False
        
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        
        # Clear remaining events
        while not self._event_queue.empty():
            try:
                self._event_queue.get_nowait()
                self._event_queue.task_done()
            except asyncio.QueueEmpty:
                break
        
        logger.debug("Event system stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event system statistics."""
        return {
            'running': self._running,
            'queue_size': self._event_queue.qsize(),
            'handler_count': sum(len(handlers) for handlers in self._handlers.values()),
            'events_processed': self._stats['events_processed'],
            'events_dropped': self._stats['events_dropped'],
            'handler_timeouts': self._stats['handler_timeouts'],
            'handler_errors': self._stats['handler_errors']
        }
    
    def __del__(self):
        """Cleanup on deletion."""
        if self._running:
            logger.warning("Event system deleted while running")


class KeyboardEventHandler(EventHandler):
    """Specialized event handler for keyboard events."""
    
    def __init__(self, key_handlers: Optional[Dict[str, Callable]] = None, name: Optional[str] = None):
        """
        Initialize keyboard event handler.
        
        Args:
            key_handlers: Dictionary mapping keys to handler functions
            name: Handler name
        """
        super().__init__(name)
        self._key_handlers = key_handlers or {}
    
    def add_key_handler(self, key: str, handler: Callable):
        """Add a handler for a specific key."""
        self._key_handlers[key] = handler
    
    def remove_key_handler(self, key: str):
        """Remove a handler for a specific key."""
        if key in self._key_handlers:
            del self._key_handlers[key]
    
    async def handle_event(self, event: Event) -> bool:
        """Handle keyboard events."""
        if event.event_type != EventType.KEYBOARD:
            return False
        
        key = event.data.get('key')
        if not key:
            return False
        
        handler = self._key_handlers.get(key)
        if not handler:
            return False
        
        try:
            # Call handler (may be async or sync)
            if inspect.iscoroutinefunction(handler):
                result = await handler(event)
            else:
                result = handler(event)
            
            return bool(result)
        except Exception as e:
            logger.error(f"Error in key handler for '{key}': {e}")
            return False


class TimerEventHandler(EventHandler):
    """Event handler that emits timer events."""
    
    def __init__(self, event_system: AsyncEventSystem, interval: float, name: Optional[str] = None):
        """
        Initialize timer event handler.
        
        Args:
            event_system: Event system to emit to
            interval: Timer interval in seconds
            name: Handler name
        """
        super().__init__(name)
        self._event_system = event_system
        self._interval = interval
        self._task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """Start the timer."""
        if self._running:
            return
            
        self._running = True
        self._task = asyncio.create_task(self._timer_loop())
    
    async def stop(self):
        """Stop the timer."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
    
    async def _timer_loop(self):
        """Timer loop that emits events."""
        while self._running:
            try:
                await asyncio.sleep(self._interval)
                
                if self._running:  # Check again after sleep
                    event = Event(
                        event_type=EventType.TIMER,
                        data={
                            'interval': self._interval,
                            'source': self.name
                        },
                        source=self.name
                    )
                    await self._event_system.emit_event(event)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in timer loop: {e}")
                await asyncio.sleep(1.0)  # Brief pause on error


# Convenience functions
def create_event_system(max_handler_timeout: float = 5.0) -> AsyncEventSystem:
    """Create and return a new AsyncEventSystem instance."""
    return AsyncEventSystem(max_handler_timeout)


def create_keyboard_handler(key_handlers: Optional[Dict[str, Callable]] = None) -> KeyboardEventHandler:
    """Create and return a new KeyboardEventHandler instance."""
    return KeyboardEventHandler(key_handlers)


# Global instance for convenience
_global_event_system: Optional[AsyncEventSystem] = None


def get_event_system() -> AsyncEventSystem:
    """Get or create the global event system instance."""
    global _global_event_system
    if _global_event_system is None:
        _global_event_system = AsyncEventSystem()
    return _global_event_system