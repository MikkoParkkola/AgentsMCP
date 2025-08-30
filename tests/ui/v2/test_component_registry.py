"""
Tests for ComponentRegistry - UI component lifecycle management.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from agentsmcp.ui.v2.component_registry import (
    ComponentRegistry, UIComponent, ComponentState, ComponentInfo,
    InputFieldComponent, ChatHistoryComponent, ComponentRegistryEventHandler
)
from agentsmcp.ui.v2.event_system import AsyncEventSystem, Event, EventType
from agentsmcp.ui.v2.display_renderer import DisplayRenderer


@pytest.fixture
async def event_system():
    """Create event system for testing."""
    system = AsyncEventSystem()
    await system.start()
    yield system
    await system.stop()


@pytest.fixture
def display_renderer():
    """Create mock display renderer."""
    renderer = Mock(spec=DisplayRenderer)
    renderer.update_region.return_value = True
    renderer.get_region_info.return_value = {'name': 'test_region'}
    return renderer


@pytest.fixture
def registry(event_system, display_renderer):
    """Create component registry for testing."""
    return ComponentRegistry(event_system, display_renderer)


class TestUIComponent(UIComponent):
    """Test implementation of UIComponent."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.initialize_called = False
        self.cleanup_called = False
        self.render_called = False
        self.focus_gained_called = False
        self.focus_lost_called = False
    
    async def initialize(self, registry):
        self.initialize_called = True
        return True
    
    async def cleanup(self):
        self.cleanup_called = True
        return True
    
    async def render(self, renderer):
        self.render_called = True
        return True
    
    async def on_focus_gained(self):
        self.focus_gained_called = True
        return await super().on_focus_gained()
    
    async def on_focus_lost(self):
        self.focus_lost_called = True
        return await super().on_focus_lost()


class TestComponentRegistry:
    """Test ComponentRegistry functionality."""
    
    async def test_register_component(self, registry):
        """Test component registration."""
        component = TestUIComponent("test_component")
        
        result = await registry.register_component(component)
        
        assert result is True
        assert "test_component" in registry.get_registered_components()
        assert component.initialize_called
        
        comp_info = registry.get_component_info("test_component")
        assert comp_info is not None
        assert comp_info.name == "test_component"
        assert comp_info.state == ComponentState.ACTIVE
    
    async def test_register_duplicate_component(self, registry):
        """Test registration of duplicate component names."""
        component1 = TestUIComponent("duplicate")
        component2 = TestUIComponent("duplicate")
        
        assert await registry.register_component(component1)
        assert not await registry.register_component(component2)
        
        # Only first component should be registered
        assert len(registry.get_registered_components()) == 1
    
    async def test_unregister_component(self, registry):
        """Test component unregistration."""
        component = TestUIComponent("test_component")
        
        await registry.register_component(component)
        assert "test_component" in registry.get_registered_components()
        
        result = await registry.unregister_component("test_component")
        
        assert result is True
        assert "test_component" not in registry.get_registered_components()
        assert component.cleanup_called
    
    async def test_unregister_nonexistent_component(self, registry):
        """Test unregistration of non-existent component."""
        result = await registry.unregister_component("nonexistent")
        assert result is False
    
    async def test_focus_management(self, registry):
        """Test component focus management."""
        component1 = TestUIComponent("component1")
        component2 = TestUIComponent("component2")
        
        await registry.register_component(component1, focus_order=1)
        await registry.register_component(component2, focus_order=2)
        
        # Set focus to first component
        result = await registry.set_focus("component1")
        
        assert result is True
        assert registry.get_focused_component() == "component1"
        assert component1.focus_gained_called
        assert component1.focused
        
        # Switch focus
        await registry.set_focus("component2")
        
        assert registry.get_focused_component() == "component2"
        assert component1.focus_lost_called
        assert not component1.focused
        assert component2.focus_gained_called
        assert component2.focused
    
    async def test_focus_order(self, registry):
        """Test focus order management."""
        # Register components in non-sequential order
        component3 = TestUIComponent("component3")
        component1 = TestUIComponent("component1")
        component2 = TestUIComponent("component2")
        
        await registry.register_component(component3, focus_order=3)
        await registry.register_component(component1, focus_order=1)
        await registry.register_component(component2, focus_order=2)
        
        focus_order = registry.get_focus_order()
        
        # Should be sorted by focus_order
        assert focus_order == ["component1", "component2", "component3"]
    
    async def test_focus_navigation(self, registry):
        """Test focus navigation (next/previous)."""
        component1 = TestUIComponent("component1")
        component2 = TestUIComponent("component2")
        component3 = TestUIComponent("component3")
        
        await registry.register_component(component1, focus_order=1)
        await registry.register_component(component2, focus_order=2)
        await registry.register_component(component3, focus_order=3)
        
        # Focus first component
        await registry.set_focus("component1")
        
        # Navigate forward
        await registry.focus_next()
        assert registry.get_focused_component() == "component2"
        
        await registry.focus_next()
        assert registry.get_focused_component() == "component3"
        
        # Wrap around to first
        await registry.focus_next()
        assert registry.get_focused_component() == "component1"
        
        # Navigate backward
        await registry.focus_previous()
        assert registry.get_focused_component() == "component3"
    
    async def test_event_routing(self, registry):
        """Test event routing to components."""
        component = TestUIComponent("test_component")
        
        # Set up event handler
        handled_events = []
        
        async def test_handler(event):
            handled_events.append(event)
            return True
        
        component.set_event_handler(EventType.KEYBOARD, test_handler)
        
        await registry.register_component(component)
        await registry.set_focus("test_component")
        
        # Create test event
        event = Event(
            event_type=EventType.KEYBOARD,
            data={'key': 'a'}
        )
        
        # Route event
        result = await registry.route_event_to_focused(event)
        
        assert result is True
        assert len(handled_events) == 1
        assert handled_events[0] == event
    
    async def test_render_all_components(self, registry, display_renderer):
        """Test rendering all components."""
        component1 = TestUIComponent("component1")
        component2 = TestUIComponent("component2")
        
        await registry.register_component(component1)
        await registry.register_component(component2)
        
        rendered_count = await registry.render_all_components()
        
        assert rendered_count == 2
        assert component1.render_called
        assert component2.render_called
    
    async def test_enable_disable_component(self, registry):
        """Test enabling and disabling components."""
        component = TestUIComponent("test_component")
        
        await registry.register_component(component)
        await registry.set_focus("test_component")
        
        # Disable component
        result = await registry.disable_component("test_component")
        
        assert result is True
        assert registry.get_focused_component() is None  # Focus should be removed
        
        comp_info = registry.get_component_info("test_component")
        assert comp_info.state == ComponentState.DISABLED
        
        # Enable component
        result = await registry.enable_component("test_component")
        
        assert result is True
        comp_info = registry.get_component_info("test_component")
        assert comp_info.state == ComponentState.ACTIVE
    
    async def test_parent_child_relationships(self, registry):
        """Test parent-child component relationships."""
        parent = TestUIComponent("parent")
        child1 = TestUIComponent("child1")
        child2 = TestUIComponent("child2")
        
        await registry.register_component(parent)
        await registry.register_component(child1, parent="parent")
        await registry.register_component(child2, parent="parent")
        
        parent_info = registry.get_component_info("parent")
        assert len(parent_info.children) == 2
        assert "child1" in parent_info.children
        assert "child2" in parent_info.children
        
        child1_info = registry.get_component_info("child1")
        assert child1_info.parent == "parent"
    
    async def test_cleanup_all_components(self, registry):
        """Test cleanup of all components."""
        component1 = TestUIComponent("component1")
        component2 = TestUIComponent("component2")
        
        await registry.register_component(component1)
        await registry.register_component(component2)
        
        result = await registry.cleanup_all_components()
        
        assert result is True
        assert len(registry.get_registered_components()) == 0
        assert component1.cleanup_called
        assert component2.cleanup_called
    
    def test_stats_collection(self, registry):
        """Test statistics collection."""
        stats = registry.get_stats()
        
        assert 'total_components' in stats
        assert 'focused_component' in stats
        assert 'focus_order_length' in stats
        assert 'components_registered' in stats
        assert 'components_active' in stats


class TestInputFieldComponent:
    """Test InputFieldComponent functionality."""
    
    @pytest.fixture
    def input_field(self):
        """Create input field component."""
        return InputFieldComponent("test_input", placeholder="Enter text...")
    
    async def test_initialization(self, input_field, registry):
        """Test input field initialization."""
        result = await input_field.initialize(registry)
        
        assert result is True
        assert EventType.KEYBOARD in input_field.event_handlers
    
    async def test_text_input(self, input_field, registry):
        """Test text input handling."""
        await input_field.initialize(registry)
        input_field.focused = True
        
        # Test character input
        event = Event(
            event_type=EventType.KEYBOARD,
            data={'character': 'a'}
        )
        
        result = await input_field.handle_event(event)
        
        assert result is True
        assert input_field.get_text() == 'a'
        assert input_field.cursor_position == 1
    
    async def test_backspace_handling(self, input_field, registry):
        """Test backspace key handling."""
        await input_field.initialize(registry)
        input_field.focused = True
        input_field.set_text("hello")
        input_field.cursor_position = 5
        
        # Test backspace
        event = Event(
            event_type=EventType.KEYBOARD,
            data={'key': 'backspace'}
        )
        
        result = await input_field.handle_event(event)
        
        assert result is True
        assert input_field.get_text() == 'hell'
        assert input_field.cursor_position == 4
    
    async def test_cursor_navigation(self, input_field, registry):
        """Test cursor navigation keys."""
        await input_field.initialize(registry)
        input_field.focused = True
        input_field.set_text("hello")
        input_field.cursor_position = 2
        
        # Test left arrow
        event = Event(
            event_type=EventType.KEYBOARD,
            data={'key': 'left'}
        )
        
        await input_field.handle_event(event)
        assert input_field.cursor_position == 1
        
        # Test right arrow
        event = Event(
            event_type=EventType.KEYBOARD,
            data={'key': 'right'}
        )
        
        await input_field.handle_event(event)
        assert input_field.cursor_position == 2
    
    async def test_home_end_keys(self, input_field, registry):
        """Test home and end key handling."""
        await input_field.initialize(registry)
        input_field.focused = True
        input_field.set_text("hello world")
        input_field.cursor_position = 5
        
        # Test home key
        event = Event(
            event_type=EventType.KEYBOARD,
            data={'key': 'home'}
        )
        
        await input_field.handle_event(event)
        assert input_field.cursor_position == 0
        
        # Test end key
        event = Event(
            event_type=EventType.KEYBOARD,
            data={'key': 'end'}
        )
        
        await input_field.handle_event(event)
        assert input_field.cursor_position == len("hello world")


class TestChatHistoryComponent:
    """Test ChatHistoryComponent functionality."""
    
    @pytest.fixture
    def chat_history(self):
        """Create chat history component."""
        return ChatHistoryComponent("test_chat", max_entries=50)
    
    async def test_initialization(self, chat_history, registry):
        """Test chat history initialization."""
        result = await chat_history.initialize(registry)
        
        assert result is True
        assert EventType.KEYBOARD in chat_history.event_handlers
        assert len(chat_history.messages) == 0
    
    def test_add_message(self, chat_history):
        """Test adding messages to chat history."""
        chat_history.add_message("user", "Hello world")
        
        assert len(chat_history.messages) == 1
        message = chat_history.messages[0]
        assert message['sender'] == "user"
        assert message['content'] == "Hello world"
        assert 'timestamp' in message
    
    def test_message_limit(self, chat_history):
        """Test message limit enforcement."""
        # Add more messages than the limit
        for i in range(60):
            chat_history.add_message("user", f"Message {i}")
        
        # Should only keep the last max_entries messages
        assert len(chat_history.messages) == 50
        assert chat_history.messages[0]['content'] == "Message 10"  # First kept message
        assert chat_history.messages[-1]['content'] == "Message 59"  # Last message
    
    async def test_scroll_handling(self, chat_history, registry):
        """Test scroll key handling."""
        await chat_history.initialize(registry)
        chat_history.focused = True
        
        # Add some messages
        for i in range(20):
            chat_history.add_message("user", f"Message {i}")
        
        initial_scroll = chat_history.scroll_position
        
        # Test up arrow (scroll up)
        event = Event(
            event_type=EventType.KEYBOARD,
            data={'key': 'up'}
        )
        
        await chat_history.handle_event(event)
        
        # Scroll position should remain at 0 (can't scroll up from top)
        assert chat_history.scroll_position == max(0, initial_scroll - 1)
        
        # Test down arrow
        event = Event(
            event_type=EventType.KEYBOARD,
            data={'key': 'down'}
        )
        
        await chat_history.handle_event(event)
        
        # Should be able to scroll down
        assert chat_history.scroll_position >= initial_scroll


class TestComponentRegistryEventHandler:
    """Test ComponentRegistryEventHandler functionality."""
    
    @pytest.fixture
    def handler(self, registry):
        """Create event handler for testing."""
        return ComponentRegistryEventHandler(registry)
    
    async def test_tab_navigation(self, handler, registry):
        """Test Tab key navigation."""
        # Set up components
        component1 = TestUIComponent("component1")
        component2 = TestUIComponent("component2")
        
        await registry.register_component(component1, focus_order=1)
        await registry.register_component(component2, focus_order=2)
        await registry.set_focus("component1")
        
        # Test Tab key
        event = Event(
            event_type=EventType.KEYBOARD,
            data={'key': 'tab'}
        )
        
        result = await handler.handle_event(event)
        
        assert result is True
        assert registry.get_focused_component() == "component2"
    
    async def test_shift_tab_navigation(self, handler, registry):
        """Test Shift+Tab navigation."""
        # Set up components
        component1 = TestUIComponent("component1")
        component2 = TestUIComponent("component2")
        
        await registry.register_component(component1, focus_order=1)
        await registry.register_component(component2, focus_order=2)
        await registry.set_focus("component2")
        
        # Test Shift+Tab
        event = Event(
            event_type=EventType.KEYBOARD,
            data={'key': 'tab', 'shift': True}
        )
        
        result = await handler.handle_event(event)
        
        assert result is True
        assert registry.get_focused_component() == "component1"
    
    async def test_application_events(self, handler, registry):
        """Test application event handling."""
        component = TestUIComponent("test_component")
        await registry.register_component(component)
        
        # Test render components action
        event = Event(
            event_type=EventType.APPLICATION,
            data={'action': 'render_components'}
        )
        
        result = await handler.handle_event(event)
        
        assert result is True
        assert component.render_called
        
        # Test focus component action
        event = Event(
            event_type=EventType.APPLICATION,
            data={
                'action': 'focus_component',
                'component': 'test_component'
            }
        )
        
        result = await handler.handle_event(event)
        
        assert result is True
        assert registry.get_focused_component() == "test_component"
    
    async def test_event_routing_to_focused(self, handler, registry):
        """Test event routing to focused component."""
        component = TestUIComponent("test_component")
        
        # Mock event handler
        handled_events = []
        
        async def mock_handler(event):
            handled_events.append(event)
            return True
        
        component.set_event_handler(EventType.KEYBOARD, mock_handler)
        
        await registry.register_component(component)
        await registry.set_focus("test_component")
        
        # Send keyboard event
        event = Event(
            event_type=EventType.KEYBOARD,
            data={'key': 'a'}
        )
        
        result = await handler.handle_event(event)
        
        assert result is True
        assert len(handled_events) == 1


@pytest.mark.asyncio
async def test_narrow_terminal_component_behavior():
    """Test component behavior with narrow terminal."""
    event_system = AsyncEventSystem()
    await event_system.start()
    
    # Mock narrow terminal renderer
    renderer = Mock(spec=DisplayRenderer)
    renderer.update_region.return_value = True
    renderer.get_region_info.return_value = {
        'name': 'test_region',
        'width': 30,  # Narrow
        'height': 10
    }
    
    try:
        registry = ComponentRegistry(event_system, renderer)
        
        # Components should still work with narrow terminal
        component = TestUIComponent("narrow_test")
        result = await registry.register_component(component)
        assert result is True
        
        # Rendering should adapt to narrow width
        await registry.render_all_components()
        assert component.render_called
        
    finally:
        await event_system.stop()


@pytest.mark.asyncio  
async def test_wide_terminal_component_behavior():
    """Test component behavior with wide terminal."""
    event_system = AsyncEventSystem()
    await event_system.start()
    
    # Mock wide terminal renderer
    renderer = Mock(spec=DisplayRenderer)
    renderer.update_region.return_value = True
    renderer.get_region_info.return_value = {
        'name': 'test_region', 
        'width': 120,  # Wide
        'height': 40
    }
    
    try:
        registry = ComponentRegistry(event_system, renderer)
        
        # Components should work well with wide terminal
        component = TestUIComponent("wide_test")
        result = await registry.register_component(component)
        assert result is True
        
        # More space should allow better rendering
        await registry.render_all_components()
        assert component.render_called
        
    finally:
        await event_system.stop()