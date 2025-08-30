"""
Tests for ApplicationController - Main application state management and coordination.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from agentsmcp.ui.v2.application_controller import (
    ApplicationController, ApplicationState, ApplicationConfig,
    ViewContext, ApplicationEventHandler, GlobalKeyboardHandler
)
from agentsmcp.ui.v2.event_system import AsyncEventSystem, Event, EventType
from agentsmcp.ui.v2.terminal_manager import TerminalManager


@pytest.fixture
async def event_system():
    """Create event system for testing."""
    system = AsyncEventSystem()
    await system.start()
    yield system
    await system.stop()


@pytest.fixture
def terminal_manager():
    """Create mock terminal manager."""
    manager = Mock(spec=TerminalManager)
    manager.detect_capabilities.return_value = Mock(
        interactive=True,
        width=80,
        height=24
    )
    return manager


@pytest.fixture
def app_config():
    """Create test application config."""
    return ApplicationConfig(
        graceful_shutdown_timeout=1.0,
        component_cleanup_timeout=0.5,
        max_error_count=5
    )


@pytest.fixture
def controller(event_system, terminal_manager, app_config):
    """Create application controller for testing."""
    return ApplicationController(
        event_system=event_system,
        terminal_manager=terminal_manager,
        config=app_config
    )


class TestApplicationController:
    """Test ApplicationController functionality."""
    
    def test_initialization(self, controller):
        """Test controller initialization."""
        assert controller.get_state() == ApplicationState.STOPPED
        assert not controller.is_running()
        assert controller._error_count == 0
        assert len(controller._commands) > 0
        assert 'quit' in controller._commands
        assert 'help' in controller._commands
    
    async def test_startup_success(self, controller):
        """Test successful application startup."""
        with patch.object(controller, 'display_renderer') as mock_renderer, \
             patch.object(controller, 'input_handler') as mock_input, \
             patch.object(controller, 'component_registry') as mock_registry, \
             patch.object(controller, 'keyboard_processor') as mock_keyboard:
            
            # Mock successful initialization
            mock_renderer = Mock()
            mock_renderer.initialize.return_value = True
            controller.display_renderer = mock_renderer
            
            mock_input = Mock()
            mock_input.is_available.return_value = True
            controller.input_handler = mock_input
            
            mock_registry = Mock()
            controller.component_registry = mock_registry
            
            mock_keyboard = Mock()
            mock_keyboard.initialize = AsyncMock(return_value=True)
            controller.keyboard_processor = mock_keyboard
            
            # Test startup
            result = await controller.startup()
            
            assert result is True
            assert controller.get_state() == ApplicationState.RUNNING
            assert controller.is_running()
            assert controller._startup_time is not None
    
    async def test_startup_failure(self, controller):
        """Test startup failure handling."""
        with patch('agentsmcp.ui.v2.application_controller.DisplayRenderer') as MockRenderer:
            # Mock a failing renderer
            mock_renderer = Mock()
            mock_renderer.initialize.return_value = False
            MockRenderer.return_value = mock_renderer
            
            result = await controller.startup()
            
            assert result is False
            assert controller.get_state() == ApplicationState.ERROR
            assert not controller.is_running()
    
    async def test_shutdown_graceful(self, controller):
        """Test graceful shutdown."""
        # Start first
        await controller.startup()
        
        # Mock components for cleanup
        controller.keyboard_processor = Mock()
        controller.keyboard_processor.cleanup = AsyncMock()
        
        controller.component_registry = Mock() 
        controller.component_registry.cleanup_all_components = AsyncMock()
        
        controller.input_handler = Mock()
        controller.input_handler.stop = Mock()
        
        controller.display_renderer = Mock()
        controller.display_renderer.cleanup = Mock()
        
        # Test shutdown
        result = await controller.shutdown(graceful=True)
        
        assert result is True
        assert controller.get_state() == ApplicationState.STOPPED
        assert not controller.is_running()
    
    async def test_command_processing(self, controller):
        """Test command processing."""
        # Test valid command
        result = await controller.process_command('help')
        
        assert result['success'] is True
        assert 'result' in result
        assert result['command'] == 'help'
        
        # Test invalid command
        result = await controller.process_command('invalid_command')
        
        assert result['success'] is False
        assert 'error' in result
        assert 'Unknown command' in result['error']
    
    async def test_command_aliases(self, controller):
        """Test command aliases."""
        # Test alias
        result = await controller.process_command('q')  # Alias for quit
        
        assert result['success'] is True
        # Should have triggered shutdown
        assert controller._shutdown_requested is True
    
    async def test_view_management(self, controller):
        """Test view registration and switching."""
        # Register views
        assert controller.register_view('view1', {'data': 'test1'})
        assert controller.register_view('view2', {'data': 'test2'})
        
        # Test duplicate registration
        assert not controller.register_view('view1')
        
        # Switch views
        assert await controller.switch_to_view('view1')
        assert controller._current_view == 'view1'
        assert controller._views['view1'].active
        
        # Switch with stack
        assert await controller.switch_to_view('view2', push_current=True)
        assert controller._current_view == 'view2'
        assert len(controller._view_stack) == 1
        assert controller._view_stack[0] == 'view1'
        
        # Go back
        assert await controller.go_back()
        assert controller._current_view == 'view1'
        assert len(controller._view_stack) == 0
    
    async def test_error_handling(self, controller):
        """Test error reporting and handling."""
        # Report errors below threshold
        for i in range(3):
            assert controller.report_error(Exception(f"Test error {i}"))
            assert controller.get_state() != ApplicationState.ERROR
        
        # Report errors that exceed threshold
        for i in range(controller.config.max_error_count):
            result = controller.report_error(Exception(f"Critical error {i}"))
        
        assert not result  # Last error should return False
        assert controller.get_state() == ApplicationState.ERROR
    
    async def test_keyboard_shortcuts_setup(self, controller):
        """Test that keyboard shortcuts are properly set up."""
        await controller.startup()
        
        assert controller.keyboard_processor is not None
        
        # Mock the processor and verify shortcuts were added
        mock_processor = Mock()
        mock_processor.add_shortcut = Mock()
        controller.keyboard_processor = mock_processor
        
        controller._setup_keyboard_shortcuts()
        
        # Verify global shortcuts were added
        assert mock_processor.add_shortcut.call_count >= 3
    
    def test_stats_collection(self, controller):
        """Test statistics collection."""
        stats = controller.get_stats()
        
        assert 'state' in stats
        assert 'uptime' in stats
        assert 'views' in stats
        assert 'commands' in stats
        assert 'error_count' in stats
        assert 'running' in stats


class TestApplicationEventHandler:
    """Test ApplicationEventHandler functionality."""
    
    @pytest.fixture
    def handler(self, controller):
        """Create event handler for testing."""
        return ApplicationEventHandler(controller)
    
    async def test_shutdown_request_handling(self, handler, controller):
        """Test shutdown request event handling."""
        event = Event(
            event_type=EventType.APPLICATION,
            data={'action': 'shutdown_request'}
        )
        
        # Mock shutdown method
        controller.shutdown = AsyncMock(return_value=True)
        
        result = await handler.handle_event(event)
        
        assert result is True
        controller.shutdown.assert_called_once_with(graceful=True)
    
    async def test_error_report_handling(self, handler, controller):
        """Test error report event handling."""
        event = Event(
            event_type=EventType.APPLICATION,
            data={
                'action': 'error_report',
                'error': 'Test error',
                'context': 'test context'
            }
        )
        
        # Mock report_error method
        controller.report_error = Mock(return_value=True)
        
        result = await handler.handle_event(event)
        
        assert result is True
        controller.report_error.assert_called_once()
        
        # Verify the exception was created properly
        call_args = controller.report_error.call_args
        assert isinstance(call_args[0][0], Exception)
        assert str(call_args[0][0]) == 'Test error'
        assert call_args[0][1] == 'test context'
    
    async def test_non_application_event(self, handler):
        """Test handling of non-application events."""
        event = Event(
            event_type=EventType.KEYBOARD,
            data={'key': 'a'}
        )
        
        result = await handler.handle_event(event)
        assert result is False


class TestGlobalKeyboardHandler:
    """Test GlobalKeyboardHandler functionality."""
    
    @pytest.fixture
    def handler(self, controller):
        """Create keyboard handler for testing."""
        return GlobalKeyboardHandler(controller)
    
    async def test_keyboard_event_passthrough(self, handler):
        """Test that keyboard events are passed through."""
        event = Event(
            event_type=EventType.KEYBOARD,
            data={'key': 'a'}
        )
        
        # Global handler doesn't process keys directly
        result = await handler.handle_event(event)
        assert result is False
    
    async def test_non_keyboard_event(self, handler):
        """Test handling of non-keyboard events."""
        event = Event(
            event_type=EventType.APPLICATION,
            data={'action': 'test'}
        )
        
        result = await handler.handle_event(event)
        assert result is False


class TestApplicationIntegration:
    """Integration tests for application components."""
    
    async def test_full_lifecycle(self, controller):
        """Test full application lifecycle."""
        # Test startup
        assert await controller.startup()
        assert controller.is_running()
        
        # Test command processing while running
        result = await controller.process_command('status')
        assert result['success'] is True
        
        # Test view operations
        controller.register_view('test_view')
        await controller.switch_to_view('test_view')
        
        # Test shutdown
        assert await controller.shutdown()
        assert not controller.is_running()
        assert controller.get_state() == ApplicationState.STOPPED
    
    async def test_context_manager(self, controller):
        """Test application context manager."""
        async with controller.app_context():
            assert controller.is_running()
            assert controller.get_state() == ApplicationState.RUNNING
        
        # Should be shut down after context exit
        assert not controller.is_running()
        assert controller.get_state() == ApplicationState.STOPPED
    
    async def test_run_method(self, controller):
        """Test the main run method."""
        # Mock to exit quickly
        original_is_running = controller.is_running
        call_count = 0
        
        def mock_is_running():
            nonlocal call_count
            call_count += 1
            return call_count <= 2  # Run a few iterations then stop
        
        controller.is_running = mock_is_running
        
        # Mock startup/shutdown
        controller.startup = AsyncMock(return_value=True)
        controller.shutdown = AsyncMock(return_value=True)
        
        # Run the application
        exit_code = await controller.run()
        
        assert exit_code == 0
        controller.startup.assert_called_once()
        controller.shutdown.assert_called_once()


@pytest.mark.asyncio
async def test_narrow_terminal_compatibility():
    """Test application works with narrow terminal."""
    terminal_manager = Mock(spec=TerminalManager)
    terminal_manager.detect_capabilities.return_value = Mock(
        interactive=True,
        width=40,  # Narrow terminal
        height=12
    )
    
    event_system = AsyncEventSystem()
    await event_system.start()
    
    try:
        controller = ApplicationController(
            event_system=event_system,
            terminal_manager=terminal_manager
        )
        
        # Should still work with narrow terminal
        # (though some components might have limited functionality)
        assert controller.get_state() == ApplicationState.STOPPED
        
    finally:
        await event_system.stop()


@pytest.mark.asyncio
async def test_wide_terminal_compatibility():
    """Test application works with wide terminal."""
    terminal_manager = Mock(spec=TerminalManager)
    terminal_manager.detect_capabilities.return_value = Mock(
        interactive=True,
        width=120,  # Wide terminal
        height=40
    )
    
    event_system = AsyncEventSystem()
    await event_system.start()
    
    try:
        controller = ApplicationController(
            event_system=event_system,
            terminal_manager=terminal_manager
        )
        
        # Should work well with wide terminal
        assert controller.get_state() == ApplicationState.STOPPED
        
    finally:
        await event_system.stop()