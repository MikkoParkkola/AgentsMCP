"""
Status Manager - Centralized system status tracking and display.

Provides clear visual states with icons, descriptions, and context information
for the TUI v2 system. Manages status indicators, error states, and system health
monitoring with real-time updates.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Callable, List, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from .event_system import AsyncEventSystem, Event, EventType

logger = logging.getLogger(__name__)


class SystemState(Enum):
    """System status states with icons and descriptions."""
    STARTING = ("⏳", "Starting", "System is initializing...")
    READY = ("⚡", "Ready", "System ready for input")
    LOADING = ("⏳", "Loading", "Loading resources...")
    PROCESSING = ("🤖", "Processing", "AI processing request...")
    STREAMING = ("📡", "Streaming", "Receiving streaming response...")
    CONNECTING = ("🔌", "Connecting", "Connecting to services...")
    ERROR = ("❌", "Error", "System error occurred")
    WARNING = ("⚠️", "Warning", "System warning")
    OFFLINE = ("🔌", "Offline", "Service unavailable")
    SHUTDOWN = ("🛑", "Shutdown", "System shutting down...")


class StatusPriority(Enum):
    """Priority levels for status messages."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class StatusInfo:
    """Information about current system status."""
    state: SystemState
    message: str = ""
    details: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    priority: StatusPriority = StatusPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def icon(self) -> str:
        """Get status icon."""
        return self.state.value[0]
    
    @property
    def title(self) -> str:
        """Get status title."""
        return self.state.value[1]
    
    @property
    def description(self) -> str:
        """Get status description."""
        return self.state.value[2]
    
    def format_status_line(self, width: int = 80) -> str:
        """Format status for display in status bar."""
        # Icon and state
        status_part = f"{self.icon} {self.title}"
        
        # Add custom message if provided
        if self.message:
            status_part += f": {self.message}"
        elif self.details:
            status_part += f": {self.details}"
        
        # Truncate if too long
        if len(status_part) > width - 20:  # Leave room for context
            status_part = status_part[:width - 23] + "..."
        
        return status_part


@dataclass
class ContextInfo:
    """Context information for status bar."""
    agent_name: Optional[str] = None
    model_name: Optional[str] = None
    connection_status: Optional[str] = None
    active_connections: int = 0
    memory_usage: Optional[str] = None
    
    def format_context_line(self, width: int = 80) -> str:
        """Format context info for display."""
        parts = []
        
        if self.agent_name:
            parts.append(f"Agent: {self.agent_name}")
        
        if self.model_name:
            parts.append(f"Model: {self.model_name}")
        
        if self.connection_status:
            parts.append(f"Conn: {self.connection_status}")
        
        if self.active_connections > 0:
            parts.append(f"Active: {self.active_connections}")
        
        if self.memory_usage:
            parts.append(f"Mem: {self.memory_usage}")
        
        context_line = " | ".join(parts)
        
        # Truncate if too long
        if len(context_line) > width:
            context_line = context_line[:width - 3] + "..."
        
        return context_line


class StatusManager:
    """
    Centralized status management for the TUI v2 system.
    
    Provides:
    - System state tracking and updates
    - Visual status indicators with icons
    - Context information display
    - Error state management
    - Performance monitoring
    """
    
    def __init__(self, event_system: Optional[AsyncEventSystem] = None):
        """Initialize the status manager."""
        self.event_system = event_system
        
        # Current status
        self.current_status = StatusInfo(SystemState.STARTING)
        self.context_info = ContextInfo()
        
        # Status history for debugging
        self.status_history: List[StatusInfo] = []
        self.max_history_size = 100
        
        # Subscribers for status updates
        self._status_subscribers: Set[Callable[[StatusInfo], None]] = set()
        self._context_subscribers: Set[Callable[[ContextInfo], None]] = set()
        
        # Error tracking
        self.error_count = 0
        self.warning_count = 0
        self.last_error: Optional[str] = None
        self.last_warning: Optional[str] = None
        
        # Performance tracking
        self.startup_time: Optional[datetime] = None
        self.last_update_time = datetime.now()
        self.update_count = 0
        
        # Quick commands for status bar
        self.quick_commands = [
            ("🔧/help", "Show help"),
            ("❌/quit", "Exit application"),
            ("📊/status", "System status"),
        ]
        
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the status manager."""
        if self._initialized:
            return True
        
        try:
            self.startup_time = datetime.now()
            
            # Subscribe to application events if event system available
            if self.event_system:
                await self._setup_event_handlers()
            
            # Set initial status
            await self.set_status(SystemState.READY, "System initialized")
            
            self._initialized = True
            logger.info("Status manager initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize status manager: {e}")
            return False
    
    async def _setup_event_handlers(self):
        """Setup event handlers for automatic status updates."""
        if not self.event_system:
            return
        
        # Create status event handler
        from .event_system import EventHandler
        
        class StatusEventHandler(EventHandler):
            def __init__(self, status_manager: 'StatusManager'):
                super().__init__("StatusEventHandler")
                self.status_manager = status_manager
            
            async def handle_event(self, event: Event) -> bool:
                """Handle events that affect status."""
                if event.event_type == EventType.APPLICATION:
                    action = event.data.get('action')
                    
                    if action == 'startup':
                        await self.status_manager.set_status(
                            SystemState.STARTING, 
                            "Application starting..."
                        )
                    elif action == 'shutdown':
                        await self.status_manager.set_status(
                            SystemState.SHUTDOWN,
                            "Application shutting down..."
                        )
                    elif action == 'error_report':
                        error = event.data.get('error', 'Unknown error')
                        await self.status_manager.set_error(str(error))
                    
                    return True
                
                elif event.event_type == EventType.CUSTOM:
                    component = event.data.get('component')
                    action = event.data.get('action')
                    
                    # Chat interface status updates
                    if component == 'chat_interface' and action == 'status_update':
                        message = event.data.get('message', '')
                        state_name = event.data.get('state', 'ready')
                        
                        # Map chat states to system states
                        state_mapping = {
                            'idle': SystemState.READY,
                            'waiting_input': SystemState.READY,
                            'processing': SystemState.PROCESSING,
                            'streaming': SystemState.STREAMING,
                            'error': SystemState.ERROR
                        }
                        
                        system_state = state_mapping.get(state_name, SystemState.READY)
                        await self.status_manager.set_status(system_state, message)
                        return True
                
                return False
        
        # Add the handler
        handler = StatusEventHandler(self)
        self.event_system.add_handler(EventType.APPLICATION, handler)
        self.event_system.add_handler(EventType.CUSTOM, handler)
    
    async def set_status(self, 
                        state: SystemState, 
                        message: str = "",
                        details: str = "",
                        priority: StatusPriority = StatusPriority.NORMAL,
                        **metadata) -> None:
        """Set the current system status."""
        # Create new status info
        new_status = StatusInfo(
            state=state,
            message=message,
            details=details,
            priority=priority,
            metadata=metadata
        )
        
        # Update current status
        old_status = self.current_status
        self.current_status = new_status
        self.last_update_time = datetime.now()
        self.update_count += 1
        
        # Add to history
        self._add_to_history(new_status)
        
        # Track errors and warnings
        if state == SystemState.ERROR:
            self.error_count += 1
            self.last_error = message or details
        elif state == SystemState.WARNING:
            self.warning_count += 1
            self.last_warning = message or details
        
        # Notify subscribers
        await self._notify_status_subscribers(new_status)
        
        # Emit event if event system available
        if self.event_system:
            await self.event_system.emit_event(Event(
                event_type=EventType.CUSTOM,
                data={
                    'component': 'status_manager',
                    'action': 'status_changed',
                    'old_state': old_status.state.name,
                    'new_state': state.name,
                    'message': message,
                    'details': details
                }
            ))
        
        logger.debug(f"Status updated: {state.name} - {message}")
    
    async def set_error(self, error_message: str, details: str = "") -> None:
        """Set error status with message."""
        await self.set_status(
            SystemState.ERROR, 
            error_message, 
            details,
            StatusPriority.CRITICAL
        )
    
    async def set_warning(self, warning_message: str, details: str = "") -> None:
        """Set warning status with message."""
        await self.set_status(
            SystemState.WARNING,
            warning_message,
            details,
            StatusPriority.HIGH
        )
    
    async def clear_error(self) -> None:
        """Clear error status and return to ready state."""
        if self.current_status.state == SystemState.ERROR:
            await self.set_status(SystemState.READY, "Error cleared")
    
    def update_context(self,
                      agent_name: Optional[str] = None,
                      model_name: Optional[str] = None,
                      connection_status: Optional[str] = None,
                      active_connections: Optional[int] = None,
                      memory_usage: Optional[str] = None) -> None:
        """Update context information."""
        old_context = self.context_info
        
        # Update only provided fields
        if agent_name is not None:
            self.context_info.agent_name = agent_name
        if model_name is not None:
            self.context_info.model_name = model_name
        if connection_status is not None:
            self.context_info.connection_status = connection_status
        if active_connections is not None:
            self.context_info.active_connections = active_connections
        if memory_usage is not None:
            self.context_info.memory_usage = memory_usage
        
        # Notify subscribers
        asyncio.create_task(self._notify_context_subscribers(self.context_info))
        
        logger.debug("Context updated")
    
    def _add_to_history(self, status: StatusInfo) -> None:
        """Add status to history with size limit."""
        self.status_history.append(status)
        
        # Maintain history size
        if len(self.status_history) > self.max_history_size:
            self.status_history = self.status_history[-self.max_history_size:]
    
    async def _notify_status_subscribers(self, status: StatusInfo) -> None:
        """Notify status update subscribers."""
        for subscriber in self._status_subscribers.copy():  # Copy to avoid modification during iteration
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    await subscriber(status)
                else:
                    subscriber(status)
            except Exception as e:
                logger.warning(f"Error notifying status subscriber: {e}")
                # Remove broken subscriber
                self._status_subscribers.discard(subscriber)
    
    async def _notify_context_subscribers(self, context: ContextInfo) -> None:
        """Notify context update subscribers."""
        for subscriber in self._context_subscribers.copy():
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    await subscriber(context)
                else:
                    subscriber(context)
            except Exception as e:
                logger.warning(f"Error notifying context subscriber: {e}")
                self._context_subscribers.discard(subscriber)
    
    def subscribe_status_updates(self, callback: Callable[[StatusInfo], None]) -> None:
        """Subscribe to status updates."""
        self._status_subscribers.add(callback)
        logger.debug("Added status update subscriber")
    
    def subscribe_context_updates(self, callback: Callable[[ContextInfo], None]) -> None:
        """Subscribe to context updates."""
        self._context_subscribers.add(callback)
        logger.debug("Added context update subscriber")
    
    def unsubscribe_status_updates(self, callback: Callable[[StatusInfo], None]) -> None:
        """Unsubscribe from status updates."""
        self._status_subscribers.discard(callback)
    
    def unsubscribe_context_updates(self, callback: Callable[[ContextInfo], None]) -> None:
        """Unsubscribe from context updates."""
        self._context_subscribers.discard(callback)
    
    def format_status_bar(self, width: int = 80) -> List[str]:
        """Format complete status bar for display."""
        lines = []
        
        # Top border
        lines.append("┌" + "─" * (width - 2) + "┐")
        
        # Status line
        status_line = self.current_status.format_status_line(width - 4)
        status_padded = f"│ {status_line:<{width - 4}} │"
        lines.append(status_padded)
        
        # Context line if available
        context_line = self.context_info.format_context_line(width - 4)
        if context_line:
            context_padded = f"│ {context_line:<{width - 4}} │"
            lines.append(context_padded)
        
        # Quick commands line
        quick_cmd_text = " | ".join([cmd[0] for cmd in self.quick_commands])
        if len(quick_cmd_text) <= width - 4:
            quick_cmd_padded = f"│ {quick_cmd_text:<{width - 4}} │"
            lines.append(quick_cmd_padded)
        
        # Bottom border
        lines.append("└" + "─" * (width - 2) + "┘")
        
        return lines
    
    def format_compact_status(self, width: int = 40) -> str:
        """Format compact status for narrow displays."""
        return f"{self.current_status.icon} {self.current_status.title}"
    
    def get_uptime(self) -> str:
        """Get system uptime string."""
        if not self.startup_time:
            return "Unknown"
        
        uptime = datetime.now() - self.startup_time
        
        if uptime.days > 0:
            return f"{uptime.days}d {uptime.seconds // 3600}h"
        elif uptime.seconds >= 3600:
            return f"{uptime.seconds // 3600}h {(uptime.seconds % 3600) // 60}m"
        elif uptime.seconds >= 60:
            return f"{uptime.seconds // 60}m"
        else:
            return f"{uptime.seconds}s"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get status manager statistics."""
        return {
            "initialized": self._initialized,
            "current_state": self.current_status.state.name,
            "current_message": self.current_status.message,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "last_error": self.last_error,
            "last_warning": self.last_warning,
            "update_count": self.update_count,
            "uptime": self.get_uptime(),
            "status_subscribers": len(self._status_subscribers),
            "context_subscribers": len(self._context_subscribers),
            "history_size": len(self.status_history)
        }
    
    def get_recent_history(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent status history."""
        recent = self.status_history[-count:] if count > 0 else self.status_history
        
        return [
            {
                "state": status.state.name,
                "message": status.message,
                "details": status.details,
                "timestamp": status.timestamp.isoformat(),
                "priority": status.priority.name
            }
            for status in recent
        ]
    
    async def cleanup(self) -> None:
        """Cleanup the status manager."""
        self._status_subscribers.clear()
        self._context_subscribers.clear()
        self.status_history.clear()
        self._initialized = False
        logger.info("Status manager cleaned up")


# Utility function for easy instantiation
def create_status_manager(event_system: Optional[AsyncEventSystem] = None) -> StatusManager:
    """Create and return a new StatusManager instance."""
    return StatusManager(event_system)