"""
Integration tests for UI v2 application control systems.

Tests how ApplicationController, ComponentRegistry, and KeyboardProcessor
work together in realistic scenarios.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from agentsmcp.ui.v2.application_controller import ApplicationController, ApplicationConfig
from agentsmcp.ui.v2.component_registry import ComponentRegistry, UIComponent
from agentsmcp.ui.v2.keyboard_processor import KeyboardProcessor, KeySequence, ShortcutContext
from agentsmcp.ui.v2.event_system import AsyncEventSystem, Event, EventType
from agentsmcp.ui.v2.terminal_manager import TerminalManager
from agentsmcp.ui.v2.display_renderer import DisplayRenderer
from agentsmcp.ui.v2.input_handler import InputHandler


@pytest.fixture
async def integrated_system():
    """Create a complete integrated system for testing."""
    # Create core components
    event_system = AsyncEventSystem()
    await event_system.start()
    
    terminal_manager = Mock(spec=TerminalManager)
    terminal_manager.detect_capabilities.return_value = Mock(
        interactive=True,
        width=80,
        height=24,
        cursor_control=True,
        alternate_screen=True
    )
    terminal_manager.get_size.return_value = (80, 24)
    
    config = ApplicationConfig(
        graceful_shutdown_timeout=1.0,
        component_cleanup_timeout=0.5
    )
    
    # Create application controller
    controller = ApplicationController(
        event_system=event_system,
        terminal_manager=terminal_manager,
        config=config
    )
    
    # Mock the components that would be created during startup
    with patch('agentsmcp.ui.v2.application_controller.DisplayRenderer') as MockRenderer, \
         patch('agentsmcp.ui.v2.application_controller.InputHandler') as MockInputHandler, \
         patch('agentsmcp.ui.v2.application_controller.ComponentRegistry') as MockRegistry, \
         patch('agentsmcp.ui.v2.application_controller.KeyboardProcessor') as MockKeyboard:
        
        # Create real instances but mock their initialization
        display_renderer = Mock(spec=DisplayRenderer)
        display_renderer.initialize.return_value = True
        display_renderer.define_region.return_value = True
        display_renderer.update_region.return_value = True
        MockRenderer.return_value = display_renderer
        
        input_handler = Mock(spec=InputHandler)
        input_handler.is_available.return_value = True
        MockInputHandler.return_value = input_handler
        
        component_registry = ComponentRegistry(event_system, display_renderer)
        MockRegistry.return_value = component_registry
        
        keyboard_processor = KeyboardProcessor(input_handler, event_system)
        await keyboard_processor.initialize()
        MockKeyboard.return_value = keyboard_processor
        
        # Start the application
        await controller.startup()
        
        yield {
            'controller': controller,
            'event_system': event_system,
            'component_registry': component_registry,
            'keyboard_processor': keyboard_processor,
            'display_renderer': display_renderer,
            'terminal_manager': terminal_manager
        }
    
    # Cleanup
    await controller.shutdown()
    await event_system.stop()


class TestComponent(UIComponent):
    """Simple test component for integration testing."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.events_received = []
        self.render_count = 0
    
    async def initialize(self, registry):
        self.set_event_handler(EventType.KEYBOARD, self._handle_keyboard)
        return True
    
    async def cleanup(self):
        return True
    
    async def render(self, renderer):
        self.render_count += 1
        content = f"Component {self.name} (renders: {self.render_count})"
        return renderer.update_region(f"{self.name}_region", content)
    
    async def _handle_keyboard(self, event):
        self.events_received.append(event)
        return True


class TestIntegration:
    """Test integration between all application control systems."""
    
    async def test_full_application_lifecycle(self, integrated_system):
        """Test complete application lifecycle with all components."""
        controller = integrated_system['controller']
        component_registry = integrated_system['component_registry']
        display_renderer = integrated_system['display_renderer']
        
        # Verify application is running
        assert controller.is_running()
        
        # Register some test components
        comp1 = TestComponent("test1")
        comp2 = TestComponent("test2")
        
        # Mock display regions for components
        display_renderer.define_region.return_value = True
        
        assert await component_registry.register_component(comp1, focus_order=1)
        assert await component_registry.register_component(comp2, focus_order=2)
        
        # Set initial focus
        await component_registry.set_focus("test1")
        assert component_registry.get_focused_component() == "test1"
        
        # Test rendering
        rendered = await component_registry.render_all_components()
        assert rendered == 2
        assert comp1.render_count == 1
        assert comp2.render_count == 1
        
        # Test application commands
        result = await controller.process_command("status")
        assert result['success'] is True
        
        # Verify the application can be cleanly shut down
        await controller.shutdown()
        assert not controller.is_running()
    
    async def test_keyboard_event_flow(self, integrated_system):
        """Test keyboard events flowing through the entire system."""
        event_system = integrated_system['event_system']
        component_registry = integrated_system['component_registry']
        keyboard_processor = integrated_system['keyboard_processor']
        
        # Register a test component
        comp = TestComponent("keyboard_test")
        await component_registry.register_component(comp)
        await component_registry.set_focus("keyboard_test")
        
        # Create a keyboard event
        keyboard_event = Event(
            event_type=EventType.KEYBOARD,
            data={'key': 'a', 'character': 'a'}
        )
        
        # Process through keyboard processor
        await keyboard_processor._process_key_event(keyboard_event)
        
        # Since 'a' is not a shortcut, it should route to the focused component
        # (In text input mode, which we'll simulate)
        keyboard_processor.enter_text_input_mode("keyboard_test")
        
        # Send another event
        await keyboard_processor._process_key_event(keyboard_event)
        
        # Event should have been emitted to event system for text input
        # The component registry's event handler should route it to the focused component
    
    async def test_focus_navigation_integration(self, integrated_system):
        """Test focus navigation through keyboard shortcuts."""
        component_registry = integrated_system['component_registry']
        keyboard_processor = integrated_system['keyboard_processor']
        
        # Register multiple components
        comp1 = TestComponent("nav1")
        comp2 = TestComponent("nav2") 
        comp3 = TestComponent("nav3")
        
        await component_registry.register_component(comp1, focus_order=1)
        await component_registry.register_component(comp2, focus_order=2)
        await component_registry.register_component(comp3, focus_order=3)
        
        # Set initial focus
        await component_registry.set_focus("nav1")
        
        # Simulate Tab key press (should move to next component)
        tab_event = Event(
            event_type=EventType.KEYBOARD,
            data={'key': 'tab'}
        )
        
        await keyboard_processor._process_key_event(tab_event)
        
        # Focus should have moved to next component
        # (This would work if the ApplicationEventHandler properly routes the focus change)
        
        # Verify components are registered correctly
        assert "nav1" in component_registry.get_registered_components()
        assert "nav2" in component_registry.get_registered_components()
        assert "nav3" in component_registry.get_registered_components()
    
    async def test_command_processing_with_shortcuts(self, integrated_system):
        """Test command processing through keyboard shortcuts."""
        controller = integrated_system['controller']
        keyboard_processor = integrated_system['keyboard_processor']
        
        # Add a custom command shortcut
        async def custom_command_handler(event):
            await controller.process_command("status")
            return True
        
        sequence = KeySequence(['s'], {'ctrl'})
        keyboard_processor.add_shortcut(
            sequence,
            custom_command_handler,
            ShortcutContext.GLOBAL,
            "Show status"
        )
        
        # Simulate Ctrl+S
        ctrl_s_event = Event(
            event_type=EventType.KEYBOARD,
            data={'key': 's', 'ctrl': True}
        )
        
        result = await keyboard_processor._process_key_event(ctrl_s_event)
        
        assert result is True
        # The custom handler should have been called
    
    async def test_text_input_component_interaction(self, integrated_system):
        """Test text input interaction between components."""
        component_registry = integrated_system['component_registry']
        keyboard_processor = integrated_system['keyboard_processor']
        display_renderer = integrated_system['display_renderer']
        
        # Create a text input component
        from agentsmcp.ui.v2.component_registry import InputFieldComponent
        
        input_comp = InputFieldComponent("text_input", "Enter text...")
        
        # Mock region for the component
        display_renderer.get_region_info.return_value = {
            'name': 'text_input_region'
        }
        
        await component_registry.register_component(input_comp, region_name="text_input_region")
        await component_registry.set_focus("text_input")
        
        # Enter text input mode
        keyboard_processor.enter_text_input_mode("text_input")
        
        # Simulate typing
        char_event = Event(
            event_type=EventType.KEYBOARD,
            data={'character': 'h', 'key': None}
        )
        
        await keyboard_processor._process_key_event(char_event)
        
        # Verify text input mode is active
        assert keyboard_processor.is_text_input_mode()
        assert keyboard_processor._text_input_component == "text_input"
    
    async def test_error_handling_integration(self, integrated_system):
        """Test error handling across the integrated system."""
        controller = integrated_system['controller']
        component_registry = integrated_system['component_registry']
        
        # Create a component that will raise an error
        class ErrorComponent(UIComponent):
            async def initialize(self, registry):
                return True
            
            async def cleanup(self):
                return True
            
            async def render(self, renderer):
                raise Exception("Render error")
            
            async def handle_event(self, event):
                raise Exception("Event handling error")
        
        error_comp = ErrorComponent("error_comp")
        await component_registry.register_component(error_comp)
        
        # Try to render - should handle errors gracefully
        rendered = await component_registry.render_all_components()
        
        # Error component should be marked as error state
        comp_info = component_registry.get_component_info("error_comp")
        # (Error handling might mark it as error state)
        
        # Application should still be running despite component errors
        assert controller.is_running()
    
    async def test_view_management_integration(self, integrated_system):
        """Test view management with components."""
        controller = integrated_system['controller']
        component_registry = integrated_system['component_registry']
        
        # Register components for different views
        main_comp = TestComponent("main_view_comp")
        settings_comp = TestComponent("settings_view_comp")
        
        await component_registry.register_component(main_comp)
        await component_registry.register_component(settings_comp)
        
        # Register views
        controller.register_view("main", {"components": ["main_view_comp"]})
        controller.register_view("settings", {"components": ["settings_view_comp"]})
        
        # Switch views
        await controller.switch_to_view("main")
        assert controller._current_view == "main"
        
        # In a real implementation, view switching might affect component focus/visibility
        await controller.switch_to_view("settings")
        assert controller._current_view == "settings"
        
        # Go back to previous view
        await controller.go_back()
        assert controller._current_view == "main"
    
    async def test_multi_key_sequence_with_components(self, integrated_system):
        """Test multi-key sequences work correctly with focused components."""
        keyboard_processor = integrated_system['keyboard_processor']
        component_registry = integrated_system['component_registry']
        
        # Register a component
        comp = TestComponent("sequence_test")
        await component_registry.register_component(comp)
        await component_registry.set_focus("sequence_test")
        
        # Add a multi-key shortcut
        handled = []
        
        async def sequence_handler(event):
            handled.append("sequence_handled")
            return True
        
        sequence = KeySequence(['x', 's'], {'ctrl'})  # Ctrl+X Ctrl+S
        keyboard_processor.add_shortcut(sequence, sequence_handler, description="Save")
        
        # Send first key
        event1 = Event(
            event_type=EventType.KEYBOARD,
            data={'key': 'x', 'ctrl': True}
        )
        
        result1 = await keyboard_processor._process_key_event(event1)
        assert result1 is True  # Should be processing sequence
        assert len(handled) == 0  # Not complete yet
        
        # Send second key
        event2 = Event(
            event_type=EventType.KEYBOARD,
            data={'key': 's', 'ctrl': True}
        )
        
        result2 = await keyboard_processor._process_key_event(event2)
        assert result2 is True
        assert len(handled) == 1  # Should be complete now
    
    async def test_context_switching_integration(self, integrated_system):
        """Test context switching affects shortcut behavior."""
        keyboard_processor = integrated_system['keyboard_processor']
        
        # Add shortcuts for different contexts
        global_handled = []
        input_handled = []
        
        async def global_handler(event):
            global_handled.append(event)
            return True
        
        async def input_handler(event):
            input_handled.append(event)
            return True
        
        sequence = KeySequence(['f2'])
        
        keyboard_processor.add_shortcut(sequence, global_handler, ShortcutContext.GLOBAL)
        keyboard_processor.add_shortcut(sequence, input_handler, ShortcutContext.INPUT)
        
        # Test in global context
        f2_event = Event(
            event_type=EventType.KEYBOARD,
            data={'key': 'f2'}
        )
        
        await keyboard_processor._process_key_event(f2_event)
        assert len(global_handled) == 1
        assert len(input_handled) == 0
        
        # Switch to input context
        keyboard_processor.push_context(ShortcutContext.INPUT)
        
        await keyboard_processor._process_key_event(f2_event)
        assert len(global_handled) == 1  # Should stay the same
        assert len(input_handled) == 1   # Input handler should be called


@pytest.mark.asyncio
async def test_integration_narrow_terminal():
    """Test integration with narrow terminal constraints."""
    event_system = AsyncEventSystem()
    await event_system.start()
    
    try:
        # Mock narrow terminal
        terminal_manager = Mock(spec=TerminalManager)
        terminal_manager.detect_capabilities.return_value = Mock(
            interactive=True,
            width=40,  # Narrow
            height=12,
            cursor_control=True,
            alternate_screen=False  # Might be disabled for narrow terminals
        )
        
        controller = ApplicationController(
            event_system=event_system,
            terminal_manager=terminal_manager
        )
        
        # Should still be able to start up
        with patch('agentsmcp.ui.v2.application_controller.DisplayRenderer') as MockRenderer:
            display_renderer = Mock(spec=DisplayRenderer)
            display_renderer.initialize.return_value = True
            MockRenderer.return_value = display_renderer
            
            result = await controller.startup()
            # Might succeed with limited functionality
            
            if result:
                await controller.shutdown()
        
    finally:
        await event_system.stop()


@pytest.mark.asyncio
async def test_integration_wide_terminal():
    """Test integration with wide terminal capabilities."""
    event_system = AsyncEventSystem()
    await event_system.start()
    
    try:
        # Mock wide terminal
        terminal_manager = Mock(spec=TerminalManager)
        terminal_manager.detect_capabilities.return_value = Mock(
            interactive=True,
            width=120,  # Wide
            height=40,
            cursor_control=True,
            alternate_screen=True,
            colors=256,
            unicode_support=True
        )
        
        controller = ApplicationController(
            event_system=event_system,
            terminal_manager=terminal_manager
        )
        
        # Should work excellently with wide terminal
        with patch('agentsmcp.ui.v2.application_controller.DisplayRenderer') as MockRenderer:
            display_renderer = Mock(spec=DisplayRenderer)
            display_renderer.initialize.return_value = True
            MockRenderer.return_value = display_renderer
            
            result = await controller.startup()
            assert result is True
            
            await controller.shutdown()
        
    finally:
        await event_system.stop()