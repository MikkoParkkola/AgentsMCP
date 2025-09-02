"""
Unit tests for terminal_controller module.

Tests the ICD-compliant terminal state management functionality including:
- Terminal size detection and monitoring
- Alternate screen buffer management  
- Cursor visibility control
- Performance requirements (operations within 100ms)
- Error handling and cleanup
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.agentsmcp.ui.v2.terminal_controller import (
    TerminalController,
    AlternateScreenMode,
    CursorVisibility,
    TerminalSize,
    TerminalState,
    SizeChangedEvent,
    CleanupResult,
    get_terminal_controller,
    cleanup_terminal_controller
)


class TestTerminalSize:
    """Test TerminalSize dataclass."""
    
    def test_valid_size_creation(self):
        """Test creating valid terminal size."""
        size = TerminalSize(width=80, height=24, timestamp=datetime.now())
        assert size.width == 80
        assert size.height == 24
        assert isinstance(size.timestamp, datetime)
    
    def test_invalid_size_dimensions(self):
        """Test validation of invalid dimensions."""
        with pytest.raises(ValueError):
            TerminalSize(width=0, height=24, timestamp=datetime.now())
        
        with pytest.raises(ValueError):
            TerminalSize(width=80, height=0, timestamp=datetime.now())
        
        with pytest.raises(ValueError):
            TerminalSize(width=-1, height=24, timestamp=datetime.now())


class TestSizeChangedEvent:
    """Test SizeChangedEvent."""
    
    def test_size_change_event_creation(self):
        """Test creating size change event."""
        old_size = TerminalSize(80, 24, datetime.now())
        new_size = TerminalSize(120, 40, datetime.now())
        
        event = SizeChangedEvent(old_size, new_size)
        
        assert event.old_size == old_size
        assert event.new_size == new_size
        assert event.delta_width == 40
        assert event.delta_height == 16
        assert isinstance(event.timestamp, datetime)


class TestTerminalController:
    """Test TerminalController class."""
    
    @pytest.fixture
    def controller(self):
        """Create a fresh terminal controller for each test."""
        return TerminalController()
    
    @pytest.mark.asyncio
    async def test_initialization_success(self, controller):
        """Test successful terminal controller initialization."""
        with patch('sys.stdout.isatty', return_value=True), \
             patch('shutil.get_terminal_size', return_value=Mock(columns=80, lines=24)):
            
            result = await controller.initialize()
            assert result is True
            
            state = await controller.get_terminal_state()
            assert state is not None
            assert state.size.width == 80
            assert state.size.height == 24
            assert state.tty_available is True
    
    @pytest.mark.asyncio
    async def test_initialization_non_tty(self, controller):
        """Test initialization in non-TTY environment."""
        with patch('sys.stdout.isatty', return_value=False), \
             patch('shutil.get_terminal_size', return_value=Mock(columns=80, lines=24)):
            
            result = await controller.initialize()
            assert result is True
            
            state = await controller.get_terminal_state()
            assert state is not None
            assert state.tty_available is False
    
    @pytest.mark.asyncio
    async def test_initialization_failure(self, controller):
        """Test handling of initialization failures."""
        with patch('shutil.get_terminal_size', side_effect=Exception("Terminal error")):
            result = await controller.initialize()
            assert result is False
    
    @pytest.mark.asyncio
    async def test_alternate_screen_operations(self, controller):
        """Test alternate screen buffer management."""
        await controller.initialize()
        
        with patch('sys.stdout.isatty', return_value=True), \
             patch('sys.stdout.write') as mock_write, \
             patch('sys.stdout.flush') as mock_flush:
            
            # Test entering alternate screen
            result = await controller.enter_alternate_screen(AlternateScreenMode.ENABLED)
            assert result is True
            
            mock_write.assert_called_with('\x1b[?1049h')
            mock_flush.assert_called_once()
            
            # Test exiting alternate screen
            mock_write.reset_mock()
            mock_flush.reset_mock()
            
            result = await controller.exit_alternate_screen()
            assert result is True
            
            mock_write.assert_called_with('\x1b[?1049l')
            mock_flush.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_alternate_screen_auto_mode(self, controller):
        """Test auto mode for alternate screen."""
        await controller.initialize()
        
        # Mock capabilities to support alternate screen
        controller._current_state.capabilities['alternate_screen'] = True
        
        with patch('sys.stdout.isatty', return_value=True), \
             patch('sys.stdout.write') as mock_write:
            
            result = await controller.enter_alternate_screen(AlternateScreenMode.AUTO)
            assert result is True
            mock_write.assert_called_with('\x1b[?1049h')
    
    @pytest.mark.asyncio
    async def test_alternate_screen_disabled_mode(self, controller):
        """Test disabled mode for alternate screen."""
        await controller.initialize()
        
        with patch('sys.stdout.isatty', return_value=True), \
             patch('sys.stdout.write') as mock_write:
            
            result = await controller.enter_alternate_screen(AlternateScreenMode.DISABLED)
            assert result is False
            mock_write.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_cursor_visibility_control(self, controller):
        """Test cursor visibility management."""
        await controller.initialize()
        
        with patch('sys.stdout.isatty', return_value=True), \
             patch('sys.stdout.write') as mock_write, \
             patch('sys.stdout.flush'):
            
            # Test hiding cursor
            result = await controller.set_cursor_visibility(CursorVisibility.HIDDEN)
            assert result is True
            mock_write.assert_called_with('\x1b[?25l')
            
            # Test showing cursor
            mock_write.reset_mock()
            result = await controller.set_cursor_visibility(CursorVisibility.VISIBLE)
            assert result is True
            mock_write.assert_called_with('\x1b[?25h')
            
            # Test auto mode
            mock_write.reset_mock()
            result = await controller.set_cursor_visibility(CursorVisibility.AUTO)
            assert result is True
            mock_write.assert_called_with('\x1b[?25h')  # Default to visible
    
    @pytest.mark.asyncio
    async def test_size_change_callbacks(self, controller):
        """Test terminal size change callback system."""
        await controller.initialize()
        
        callback_called = False
        received_event = None
        
        def size_callback(event: SizeChangedEvent):
            nonlocal callback_called, received_event
            callback_called = True
            received_event = event
        
        controller.register_size_change_callback(size_callback)
        
        # Simulate size change
        old_size = TerminalSize(80, 24, datetime.now())
        new_size = TerminalSize(120, 40, datetime.now())
        
        await controller._handle_size_change(new_size)
        
        assert callback_called
        assert received_event is not None
        assert received_event.new_size.width == 120
        assert received_event.new_size.height == 40
    
    @pytest.mark.asyncio
    async def test_size_detection_fallback(self, controller):
        """Test size detection with fallback methods."""
        with patch('shutil.get_terminal_size', side_effect=Exception()), \
             patch('os.getenv') as mock_getenv:
            
            # Test environment variable fallback
            mock_getenv.side_effect = lambda key, default='0': {
                'COLUMNS': '100',
                'LINES': '30'
            }.get(key, default)
            
            size = await controller._detect_terminal_size()
            assert size.width == 100
            assert size.height == 30
    
    @pytest.mark.asyncio
    async def test_size_detection_final_fallback(self, controller):
        """Test final fallback for size detection."""
        with patch('shutil.get_terminal_size', side_effect=Exception()), \
             patch('os.getenv', return_value='0'):
            
            size = await controller._detect_terminal_size()
            assert size.width == 80  # Default fallback
            assert size.height == 24  # Default fallback
    
    @pytest.mark.asyncio
    async def test_cleanup_operations(self, controller):
        """Test cleanup operations."""
        await controller.initialize()
        
        # Setup some state
        await controller.enter_alternate_screen(AlternateScreenMode.ENABLED)
        await controller.set_cursor_visibility(CursorVisibility.HIDDEN)
        
        with patch('sys.stdout.write'), \
             patch('sys.stdout.flush'):
            
            result = await controller.cleanup()
            
            assert result.success is True
            assert result.operations_completed >= 0
            assert result.total_operations > 0
    
    @pytest.mark.asyncio
    async def test_performance_requirements(self, controller):
        """Test that operations complete within 100ms requirement."""
        await controller.initialize()
        
        with patch('sys.stdout.isatty', return_value=True), \
             patch('sys.stdout.write'), \
             patch('sys.stdout.flush'):
            
            # Test alternate screen performance
            start_time = asyncio.get_event_loop().time()
            await controller.enter_alternate_screen(AlternateScreenMode.ENABLED)
            enter_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            start_time = asyncio.get_event_loop().time()
            await controller.exit_alternate_screen()
            exit_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # ICD requirement: operations within 100ms
            assert enter_time < 100, f"Enter alternate screen took {enter_time}ms (> 100ms)"
            assert exit_time < 100, f"Exit alternate screen took {exit_time}ms (> 100ms)"
            
            # Test cursor visibility performance
            start_time = asyncio.get_event_loop().time()
            await controller.set_cursor_visibility(CursorVisibility.HIDDEN)
            cursor_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            assert cursor_time < 100, f"Set cursor visibility took {cursor_time}ms (> 100ms)"
    
    def test_terminal_context_manager(self, controller):
        """Test terminal context manager functionality."""
        # This is a basic test since context manager uses async operations
        with controller.terminal_context() as context:
            assert context is not None
            assert hasattr(context, 'controller')
            assert context.controller is controller
    
    @pytest.mark.asyncio
    async def test_capabilities_detection(self, controller):
        """Test terminal capabilities detection."""
        with patch('os.getenv') as mock_getenv, \
             patch('sys.stdout.isatty', return_value=True):
            
            # Mock environment for color support
            mock_getenv.side_effect = lambda key, default='': {
                'TERM': 'xterm-256color',
                'COLORTERM': 'truecolor'
            }.get(key, default)
            
            capabilities = await controller._detect_capabilities()
            
            assert capabilities['tty'] is True
            assert capabilities['colors'] == 16777216  # True color
            assert capabilities['alternate_screen'] is True
    
    @pytest.mark.asyncio 
    async def test_non_tty_operations(self, controller):
        """Test operations in non-TTY environment."""
        with patch('sys.stdout.isatty', return_value=False):
            await controller.initialize()
            
            # These should fail gracefully in non-TTY
            result = await controller.enter_alternate_screen(AlternateScreenMode.ENABLED)
            assert result is False
            
            result = await controller.set_cursor_visibility(CursorVisibility.HIDDEN)
            assert result is False


class TestGlobalFunctions:
    """Test global utility functions."""
    
    @pytest.mark.asyncio
    async def test_get_terminal_controller_singleton(self):
        """Test global terminal controller singleton."""
        # Clean up any existing instance
        await cleanup_terminal_controller()
        
        controller1 = await get_terminal_controller()
        controller2 = await get_terminal_controller()
        
        assert controller1 is controller2
        
        # Cleanup
        await cleanup_terminal_controller()
    
    @pytest.mark.asyncio
    async def test_cleanup_terminal_controller_no_instance(self):
        """Test cleanup when no controller exists."""
        # Ensure no instance exists
        await cleanup_terminal_controller()
        
        result = await cleanup_terminal_controller()
        assert result.success is True


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def controller(self):
        return TerminalController()
    
    @pytest.mark.asyncio
    async def test_double_initialization(self, controller):
        """Test double initialization handling."""
        result1 = await controller.initialize()
        result2 = await controller.initialize()
        
        assert result1 is True
        assert result2 is True  # Should handle gracefully
    
    @pytest.mark.asyncio
    async def test_operations_before_initialization(self, controller):
        """Test operations called before initialization."""
        state = await controller.get_terminal_state()
        assert state is None
        
        result = await controller.enter_alternate_screen()
        assert result is False
        
        result = await controller.set_cursor_visibility(CursorVisibility.HIDDEN)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_callback_exception_handling(self, controller):
        """Test that callback exceptions don't break size monitoring."""
        await controller.initialize()
        
        def failing_callback(event):
            raise Exception("Callback error")
        
        controller.register_size_change_callback(failing_callback)
        
        # This should not raise an exception
        new_size = TerminalSize(100, 30, datetime.now())
        await controller._handle_size_change(new_size)
        
        # Controller should still be functional
        state = await controller.get_terminal_state()
        assert state is not None
    
    @pytest.mark.asyncio
    async def test_cleanup_with_active_contexts(self, controller):
        """Test cleanup when contexts are still active."""
        await controller.initialize()
        
        # Simulate active context
        import weakref
        mock_context = Mock()
        controller._active_contexts.add(weakref.ref(mock_context))
        
        result = await controller.cleanup()
        assert result.success is True
        assert len(controller._active_contexts) == 0