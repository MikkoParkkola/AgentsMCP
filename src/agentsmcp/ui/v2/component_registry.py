"""
Component Registry - UI component lifecycle management.

Manages the registration, focus management, and event routing for UI components
in the TUI interface.
"""

import asyncio
import logging
from typing import Dict, Optional, List, Set, Any, Protocol, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from weakref import WeakSet
from abc import ABC, abstractmethod

from .event_system import AsyncEventSystem, Event, EventType, EventHandler
from .display_renderer import DisplayRenderer

logger = logging.getLogger(__name__)


class ComponentState(Enum):
    """Component lifecycle states."""
    INACTIVE = "inactive"
    ACTIVE = "active"
    FOCUSED = "focused"
    DISABLED = "disabled"
    ERROR = "error"


@dataclass
class ComponentInfo:
    """Information about a registered component."""
    name: str
    component: 'UIComponent'
    state: ComponentState = ComponentState.INACTIVE
    focus_order: int = 0
    region_name: Optional[str] = None
    parent: Optional[str] = None
    children: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class UIComponent(ABC):
    """
    Abstract base class for UI components.
    
    All UI components must implement this interface to be managed
    by the ComponentRegistry.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.state = ComponentState.INACTIVE
        self.focused = False
        self.enabled = True
        self.event_handlers: Dict[EventType, Callable] = {}
    
    @abstractmethod
    async def initialize(self, registry: 'ComponentRegistry') -> bool:
        """Initialize the component. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> bool:
        """Cleanup the component. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def render(self, renderer: DisplayRenderer) -> bool:
        """Render the component. Must be implemented by subclasses."""
        pass
    
    async def on_focus_gained(self) -> bool:
        """Called when component gains focus."""
        self.focused = True
        logger.debug(f"Component {self.name} gained focus")
        return True
    
    async def on_focus_lost(self) -> bool:
        """Called when component loses focus."""
        self.focused = False
        logger.debug(f"Component {self.name} lost focus")
        return True
    
    async def handle_event(self, event: Event) -> bool:
        """Handle events routed to this component."""
        event_type = event.event_type
        if event_type in self.event_handlers:
            try:
                handler = self.event_handlers[event_type]
                if asyncio.iscoroutinefunction(handler):
                    return await handler(event)
                else:
                    return handler(event)
            except Exception as e:
                logger.error(f"Error in component {self.name} event handler: {e}")
                return False
        return False
    
    def set_event_handler(self, event_type: EventType, handler: Callable):
        """Set an event handler for a specific event type."""
        self.event_handlers[event_type] = handler
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get component state information."""
        return {
            'name': self.name,
            'state': self.state.value,
            'focused': self.focused,
            'enabled': self.enabled,
            'event_handlers': list(self.event_handlers.keys())
        }


class InputFieldComponent(UIComponent):
    """Example text input field component."""
    
    def __init__(self, name: str, placeholder: str = ""):
        super().__init__(name)
        self.placeholder = placeholder
        self.text = ""
        self.cursor_position = 0
        self.max_length = 1000
    
    async def initialize(self, registry: 'ComponentRegistry') -> bool:
        """Initialize the input field."""
        # Set up keyboard event handling
        self.set_event_handler(EventType.KEYBOARD, self._handle_keyboard)
        return True
    
    async def cleanup(self) -> bool:
        """Cleanup the input field."""
        return True
    
    async def render(self, renderer: DisplayRenderer) -> bool:
        """Render the input field."""
        display_text = self.text if self.text else self.placeholder
        if self.focused and self.text:
            # Show cursor
            before_cursor = display_text[:self.cursor_position]
            after_cursor = display_text[self.cursor_position:]
            display_text = before_cursor + "│" + after_cursor
        
        # Find our region and render
        region_info = renderer.get_region_info(f"{self.name}_region")
        if region_info:
            return renderer.update_region(f"{self.name}_region", display_text)
        
        return False
    
    async def _handle_keyboard(self, event: Event) -> bool:
        """Handle keyboard input for the text field."""
        if not self.focused:
            return False
        
        key = event.data.get('key')
        character = event.data.get('character')
        
        if character and len(character) == 1 and ord(character) >= 32:
            # Regular character input
            self.text = (self.text[:self.cursor_position] + 
                        character + 
                        self.text[self.cursor_position:])
            self.cursor_position += 1
            return True
        
        elif key == 'backspace' and self.cursor_position > 0:
            self.text = (self.text[:self.cursor_position-1] + 
                        self.text[self.cursor_position:])
            self.cursor_position -= 1
            return True
        
        elif key == 'left' and self.cursor_position > 0:
            self.cursor_position -= 1
            return True
        
        elif key == 'right' and self.cursor_position < len(self.text):
            self.cursor_position += 1
            return True
        
        elif key == 'home':
            self.cursor_position = 0
            return True
        
        elif key == 'end':
            self.cursor_position = len(self.text)
            return True
        
        return False
    
    def get_text(self) -> str:
        """Get current text content."""
        return self.text
    
    def set_text(self, text: str):
        """Set text content."""
        self.text = text[:self.max_length]
        self.cursor_position = min(self.cursor_position, len(self.text))


class ChatHistoryComponent(UIComponent):
    """Chat history display component."""
    
    def __init__(self, name: str, max_entries: int = 100):
        super().__init__(name)
        self.messages: List[Dict[str, Any]] = []
        self.max_entries = max_entries
        self.scroll_position = 0
    
    async def initialize(self, registry: 'ComponentRegistry') -> bool:
        """Initialize chat history."""
        self.set_event_handler(EventType.KEYBOARD, self._handle_keyboard)
        return True
    
    async def cleanup(self) -> bool:
        """Cleanup chat history."""
        return True
    
    async def render(self, renderer: DisplayRenderer) -> bool:
        """Render chat history."""
        if not self.messages:
            content = "[No messages]"
        else:
            # Show recent messages
            visible_messages = self.messages[-20:]  # Show last 20 messages
            lines = []
            for msg in visible_messages:
                timestamp = msg.get('timestamp', '')
                sender = msg.get('sender', 'Unknown')
                content_text = msg.get('content', '')
                lines.append(f"[{timestamp}] {sender}: {content_text}")
            content = '\n'.join(lines)
        
        return renderer.update_region(f"{self.name}_region", content)
    
    async def _handle_keyboard(self, event: Event) -> bool:
        """Handle keyboard input for scrolling."""
        if not self.focused:
            return False
        
        key = event.data.get('key')
        
        if key == 'up':
            self.scroll_position = max(0, self.scroll_position - 1)
            return True
        elif key == 'down':
            max_scroll = max(0, len(self.messages) - 10)
            self.scroll_position = min(max_scroll, self.scroll_position + 1)
            return True
        
        return False
    
    def add_message(self, sender: str, content: str, timestamp: Optional[str] = None):
        """Add a message to the chat history."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%H:%M:%S")
        
        message = {
            'sender': sender,
            'content': content,
            'timestamp': timestamp
        }
        
        self.messages.append(message)
        
        # Keep history size manageable
        if len(self.messages) > self.max_entries:
            self.messages = self.messages[-self.max_entries:]


class ComponentRegistry:
    """
    UI component lifecycle management.
    
    Manages registration, focus, event routing, and cleanup of UI components.
    """
    
    def __init__(self, 
                 event_system: AsyncEventSystem,
                 display_renderer: DisplayRenderer):
        self.event_system = event_system
        self.display_renderer = display_renderer
        
        # Component management
        self._components: Dict[str, ComponentInfo] = {}
        self._focus_order: List[str] = []
        self._current_focus: Optional[str] = None
        self._disabled_components: Set[str] = set()
        
        # Event routing
        self._component_event_handler = ComponentRegistryEventHandler(self)
        self.event_system.add_handler(EventType.KEYBOARD, self._component_event_handler)
        self.event_system.add_handler(EventType.APPLICATION, self._component_event_handler)
        
        # Statistics
        self._stats = {
            'components_registered': 0,
            'components_active': 0,
            'focus_changes': 0,
            'events_routed': 0
        }
    
    async def register_component(self, 
                                component: UIComponent,
                                focus_order: int = 0,
                                region_name: Optional[str] = None,
                                parent: Optional[str] = None) -> bool:
        """
        Register a UI component.
        
        Args:
            component: The component to register
            focus_order: Order in focus chain (lower numbers get focus first)
            region_name: Optional display region name
            parent: Optional parent component name
            
        Returns:
            True if registration successful
        """
        if component.name in self._components:
            logger.warning(f"Component {component.name} already registered")
            return False
        
        try:
            # Initialize the component
            if not await component.initialize(self):
                logger.error(f"Failed to initialize component {component.name}")
                return False
            
            # Create component info
            comp_info = ComponentInfo(
                name=component.name,
                component=component,
                focus_order=focus_order,
                region_name=region_name,
                parent=parent
            )
            
            # Handle parent-child relationships
            if parent and parent in self._components:
                self._components[parent].children.add(component.name)
            
            self._components[component.name] = comp_info
            
            # Update focus order
            self._rebuild_focus_order()
            
            # Set up display region if specified
            if region_name:
                # Define a default region if not already defined
                # This would typically be done by the application or layout engine
                pass
            
            # Set initial state
            component.state = ComponentState.ACTIVE
            comp_info.state = ComponentState.ACTIVE
            
            self._stats['components_registered'] += 1
            self._stats['components_active'] += 1
            
            logger.info(f"Registered component: {component.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering component {component.name}: {e}")
            return False
    
    async def unregister_component(self, name: str) -> bool:
        """
        Unregister and cleanup a component.
        
        Args:
            name: Component name
            
        Returns:
            True if unregistration successful
        """
        if name not in self._components:
            logger.warning(f"Component {name} not registered")
            return False
        
        try:
            comp_info = self._components[name]
            component = comp_info.component
            
            # Remove focus if this component has it
            if self._current_focus == name:
                await self._move_focus_to_next()
            
            # Cleanup child components first
            for child_name in list(comp_info.children):
                await self.unregister_component(child_name)
            
            # Remove from parent's children
            if comp_info.parent and comp_info.parent in self._components:
                parent_info = self._components[comp_info.parent]
                parent_info.children.discard(name)
            
            # Cleanup component
            await component.cleanup()
            
            # Remove from registry
            del self._components[name]
            
            # Remove from focus order
            if name in self._focus_order:
                self._focus_order.remove(name)
            
            self._stats['components_active'] -= 1
            
            logger.info(f"Unregistered component: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error unregistering component {name}: {e}")
            return False
    
    def _rebuild_focus_order(self):
        """Rebuild the focus order list based on focus_order values."""
        components_with_order = [
            (info.focus_order, name) 
            for name, info in self._components.items()
            if info.state not in (ComponentState.DISABLED, ComponentState.ERROR)
        ]
        
        components_with_order.sort(key=lambda x: x[0])
        self._focus_order = [name for _, name in components_with_order]
    
    async def set_focus(self, component_name: str) -> bool:
        """
        Set focus to a specific component.
        
        Args:
            component_name: Name of component to focus
            
        Returns:
            True if focus was set successfully
        """
        if component_name not in self._components:
            logger.warning(f"Cannot focus unknown component: {component_name}")
            return False
        
        comp_info = self._components[component_name]
        if comp_info.state in (ComponentState.DISABLED, ComponentState.ERROR):
            logger.warning(f"Cannot focus disabled component: {component_name}")
            return False
        
        try:
            # Remove focus from current component
            if self._current_focus:
                current_comp = self._components[self._current_focus].component
                await current_comp.on_focus_lost()
                self._components[self._current_focus].state = ComponentState.ACTIVE
            
            # Set focus to new component
            await comp_info.component.on_focus_gained()
            comp_info.state = ComponentState.FOCUSED
            self._current_focus = component_name
            
            self._stats['focus_changes'] += 1
            
            # Emit focus change event
            focus_event = Event(
                event_type=EventType.APPLICATION,
                data={
                    'action': 'focus_change',
                    'component': component_name,
                    'previous_focus': self._current_focus
                }
            )
            await self.event_system.emit_event(focus_event)
            
            logger.debug(f"Focus set to component: {component_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting focus to {component_name}: {e}")
            return False
    
    async def focus_next(self) -> bool:
        """Move focus to the next component in the focus order."""
        if not self._focus_order:
            return False
        
        current_index = 0
        if self._current_focus in self._focus_order:
            current_index = self._focus_order.index(self._current_focus)
        
        next_index = (current_index + 1) % len(self._focus_order)
        next_component = self._focus_order[next_index]
        
        return await self.set_focus(next_component)
    
    async def focus_previous(self) -> bool:
        """Move focus to the previous component in the focus order."""
        if not self._focus_order:
            return False
        
        current_index = 0
        if self._current_focus in self._focus_order:
            current_index = self._focus_order.index(self._current_focus)
        
        prev_index = (current_index - 1) % len(self._focus_order)
        prev_component = self._focus_order[prev_index]
        
        return await self.set_focus(prev_component)
    
    async def _move_focus_to_next(self) -> bool:
        """Internal method to move focus when current component is removed."""
        if not self._focus_order:
            self._current_focus = None
            return True
        
        # Find next available component
        for component_name in self._focus_order:
            if component_name in self._components:
                return await self.set_focus(component_name)
        
        self._current_focus = None
        return True
    
    async def route_event_to_component(self, component_name: str, event: Event) -> bool:
        """
        Route an event to a specific component.
        
        Args:
            component_name: Target component name
            event: Event to route
            
        Returns:
            True if event was handled
        """
        if component_name not in self._components:
            return False
        
        comp_info = self._components[component_name]
        if comp_info.state == ComponentState.DISABLED:
            return False
        
        try:
            handled = await comp_info.component.handle_event(event)
            if handled:
                self._stats['events_routed'] += 1
            return handled
            
        except Exception as e:
            logger.error(f"Error routing event to component {component_name}: {e}")
            # Mark component as error state
            comp_info.state = ComponentState.ERROR
            return False
    
    async def route_event_to_focused(self, event: Event) -> bool:
        """Route an event to the currently focused component."""
        if not self._current_focus:
            return False
        
        return await self.route_event_to_component(self._current_focus, event)
    
    async def render_all_components(self, force: bool = False) -> int:
        """
        Render all active components.
        
        Args:
            force: Force render even if not dirty
            
        Returns:
            Number of components rendered
        """
        rendered_count = 0
        
        for comp_info in self._components.values():
            if comp_info.state in (ComponentState.ACTIVE, ComponentState.FOCUSED):
                try:
                    if await comp_info.component.render(self.display_renderer):
                        rendered_count += 1
                except Exception as e:
                    logger.error(f"Error rendering component {comp_info.name}: {e}")
                    comp_info.state = ComponentState.ERROR
        
        return rendered_count
    
    async def enable_component(self, name: str) -> bool:
        """Enable a disabled component."""
        if name not in self._components:
            return False
        
        comp_info = self._components[name]
        if comp_info.state == ComponentState.DISABLED:
            comp_info.state = ComponentState.ACTIVE
            comp_info.component.enabled = True
            self._rebuild_focus_order()
            logger.debug(f"Enabled component: {name}")
            return True
        
        return False
    
    async def disable_component(self, name: str) -> bool:
        """Disable a component."""
        if name not in self._components:
            return False
        
        comp_info = self._components[name]
        
        # Remove focus if this component has it
        if self._current_focus == name:
            # Clear focus first before trying to move to next
            if name in self._focus_order:
                self._focus_order.remove(name)
            await self._move_focus_to_next()
        
        comp_info.state = ComponentState.DISABLED
        comp_info.component.enabled = False
        
        # Remove from focus order (if not already removed above)
        if name in self._focus_order:
            self._focus_order.remove(name)
        
        logger.debug(f"Disabled component: {name}")
        return True
    
    async def cleanup_all_components(self) -> bool:
        """Cleanup all registered components."""
        try:
            # Unregister all components
            component_names = list(self._components.keys())
            for name in component_names:
                await self.unregister_component(name)
            
            # Remove event handlers
            self.event_system.remove_handler(EventType.KEYBOARD, self._component_event_handler)
            self.event_system.remove_handler(EventType.APPLICATION, self._component_event_handler)
            
            logger.info("Cleaned up all components")
            return True
            
        except Exception as e:
            logger.error(f"Error during component cleanup: {e}")
            return False
    
    def get_registered_components(self) -> Dict[str, ComponentInfo]:
        """Get all registered components."""
        return self._components.copy()
    
    def get_focused_component(self) -> Optional[str]:
        """Get the name of the currently focused component."""
        return self._current_focus
    
    def get_focus_order(self) -> List[str]:
        """Get the current focus order."""
        return self._focus_order.copy()
    
    def get_component_info(self, name: str) -> Optional[ComponentInfo]:
        """Get information about a specific component."""
        return self._components.get(name)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            **self._stats,
            'total_components': len(self._components),
            'focused_component': self._current_focus,
            'focus_order_length': len(self._focus_order),
            'disabled_components': len([c for c in self._components.values() 
                                       if c.state == ComponentState.DISABLED]),
            'error_components': len([c for c in self._components.values() 
                                   if c.state == ComponentState.ERROR])
        }


class ComponentRegistryEventHandler(EventHandler):
    """Event handler for component registry operations."""
    
    def __init__(self, registry: ComponentRegistry):
        super().__init__("ComponentRegistryEventHandler")
        self.registry = registry
    
    async def handle_event(self, event: Event) -> bool:
        """Handle events that affect component management."""
        if event.event_type == EventType.KEYBOARD:
            # Route keyboard events to focused component
            key = event.data.get('key')
            
            # Handle Tab for focus navigation
            if key == 'tab':
                shift = event.data.get('shift', False)
                if shift:
                    await self.registry.focus_previous()
                else:
                    await self.registry.focus_next()
                return True
            
            # Route to focused component
            return await self.registry.route_event_to_focused(event)
        
        elif event.event_type == EventType.APPLICATION:
            action = event.data.get('action')
            
            if action == 'render_components':
                await self.registry.render_all_components()
                return True
            
            elif action == 'focus_component':
                component_name = event.data.get('component')
                if component_name:
                    return await self.registry.set_focus(component_name)
        
        return False