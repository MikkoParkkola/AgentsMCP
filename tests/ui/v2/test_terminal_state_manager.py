"""
Comprehensive tests for TerminalStateManager - Critical TTY settings management.

This test suite verifies that the TerminalStateManager:
1. Properly initializes and captures terminal state
2. Enters raw/cbreak modes correctly
3. Restores terminal state on exit (CRITICAL for system stability)
4. Handles cleanup gracefully on interrupts and crashes
5. Provides thread-safe operations
"""

import pytest
import os
import sys
import signal
import termios
import threading
import time
from unittest.mock import Mock, patch, MagicMock, call
from contextlib import contextmanager

from agentsmcp.ui.v2.terminal_state_manager import (
    TerminalStateManager, TerminalState, TerminalMode,
    raw_mode, cbreak_mode, get_global_terminal_manager, emergency_terminal_restore
)


class TestTerminalStateManager:
    """Test terminal state manager functionality."""
    
    @pytest.fixture
    def manager(self):
        """Create a fresh terminal state manager."""
        return TerminalStateManager()
    
    @pytest.fixture
    def mock_tty_fd(self):
        """Mock TTY file descriptor."""
        return 3  # Non-standard fd to avoid conflicts
    
    @pytest.fixture
    def mock_terminal_attrs(self):
        """Mock terminal attributes."""
        return [1, 2, 3, 4, 5, 6]  # Simplified termios attributes
    
    def test_initialization_success(self, manager):
        """Test successful terminal state manager initialization."""
        with patch('os.open') as mock_open, \
             patch('os.isatty', return_value=True) as mock_isatty, \
             patch('termios.tcgetattr') as mock_tcgetattr, \
             patch.object(manager, '_register_cleanup_handlers') as mock_register:
            
            mock_open.return_value = 3
            mock_tcgetattr.return_value = [1, 2, 3, 4, 5, 6]
            
            result = manager.initialize()
            
            assert result is True
            assert manager._initialized is True
            assert manager._tty_fd == 3
            assert manager._output_fd == sys.stdout.fileno()
            assert manager._original_state is not None
            assert manager._original_state.fd == 3
            assert manager._original_state.mode == TerminalMode.NORMAL
            mock_register.assert_called_once()
    
    def test_initialization_no_tty(self, manager):
        """Test initialization when no TTY is available."""
        with patch('os.open', side_effect=OSError("No such device")) as mock_open, \
             patch('os.path.exists', return_value=False), \
             patch('os.isatty', return_value=False):
            
            result = manager.initialize()
            
            # Should still succeed but with limited functionality
            assert result is True
            assert manager._initialized is True
            assert manager._tty_fd == sys.stdin.fileno()  # Fallback
    
    def test_initialization_permission_denied(self, manager):
        """Test initialization when TTY access is denied."""
        with patch('os.open', side_effect=PermissionError("Permission denied")) as mock_open, \
             patch('os.path.exists', return_value=True), \
             patch('os.isatty', return_value=False):
            
            result = manager.initialize()
            
            # Should still succeed but with fallback behavior
            assert result is True
    
    def test_enter_raw_mode(self, manager, mock_tty_fd):
        """Test entering raw terminal mode."""
        with patch('tty.setraw') as mock_setraw, \
             patch.object(manager, 'initialize', return_value=True):
            
            manager._initialized = True
            manager._tty_fd = mock_tty_fd
            
            result = manager.enter_raw_mode()
            
            assert result is True
            assert manager._current_mode == TerminalMode.RAW
            assert "raw_mode" in manager._state_changes
            mock_setraw.assert_called_once_with(mock_tty_fd)
    
    def test_enter_raw_mode_already_raw(self, manager, mock_tty_fd):
        """Test entering raw mode when already in raw mode."""
        with patch('tty.setraw') as mock_setraw, \
             patch.object(manager, 'initialize', return_value=True):
            
            manager._initialized = True
            manager._tty_fd = mock_tty_fd
            manager._current_mode = TerminalMode.RAW
            
            result = manager.enter_raw_mode()
            
            assert result is True
            mock_setraw.assert_not_called()
    
    def test_enter_raw_mode_failure(self, manager, mock_tty_fd):
        """Test raw mode entry failure."""
        with patch('tty.setraw', side_effect=termios.error("Invalid argument")) as mock_setraw:
            
            manager._initialized = True
            manager._tty_fd = mock_tty_fd
            
            result = manager.enter_raw_mode()
            
            assert result is False
            assert manager._current_mode == TerminalMode.NORMAL
    
    def test_enter_cbreak_mode(self, manager, mock_tty_fd):
        """Test entering cbreak terminal mode."""
        with patch('tty.setcbreak') as mock_setcbreak:
            
            manager._initialized = True
            manager._tty_fd = mock_tty_fd
            
            result = manager.enter_cbreak_mode()
            
            assert result is True
            assert manager._current_mode == TerminalMode.CBREAK
            assert "cbreak_mode" in manager._state_changes
            mock_setcbreak.assert_called_once_with(mock_tty_fd)
    
    def test_cursor_visibility_control(self, manager):
        """Test cursor show/hide functionality."""
        with patch('os.write') as mock_write:
            
            manager._output_fd = sys.stdout.fileno()
            
            # Test hide cursor
            result = manager.hide_cursor()
            assert result is True
            assert "cursor_hidden" in manager._state_changes
            mock_write.assert_called_with(sys.stdout.fileno(), b'\033[?25l')
            
            # Test show cursor
            result = manager.show_cursor()
            assert result is True
            assert "cursor_hidden" not in manager._state_changes
            mock_write.assert_called_with(sys.stdout.fileno(), b'\033[?25h')
    
    def test_alternate_screen_control(self, manager):
        """Test alternate screen buffer control."""
        with patch('os.write') as mock_write:
            
            manager._output_fd = sys.stdout.fileno()
            
            # Test enter alternate screen
            result = manager.enter_alternate_screen()
            assert result is True
            assert "alternate_screen" in manager._state_changes
            mock_write.assert_called_with(sys.stdout.fileno(), b'\033[?1049h')
            
            # Test exit alternate screen
            result = manager.exit_alternate_screen()
            assert result is True
            assert "alternate_screen" not in manager._state_changes
            mock_write.assert_called_with(sys.stdout.fileno(), b'\033[?1049l')
    
    def test_mouse_reporting_control(self, manager):
        """Test mouse reporting enable/disable."""
        with patch('os.write') as mock_write:
            
            manager._output_fd = sys.stdout.fileno()
            
            # Test enable mouse reporting
            result = manager.enable_mouse_reporting()
            assert result is True
            assert "mouse_enabled" in manager._state_changes
            mock_write.assert_called_with(sys.stdout.fileno(), b'\033[?1000h\033[?1006h')
            
            # Test disable mouse reporting
            result = manager.disable_mouse_reporting()
            assert result is True
            assert "mouse_enabled" not in manager._state_changes
            mock_write.assert_called_with(sys.stdout.fileno(), b'\033[?1006l\033[?1000l')
    
    def test_terminal_state_restoration(self, manager, mock_tty_fd, mock_terminal_attrs):
        """Test complete terminal state restoration."""
        with patch('termios.tcsetattr') as mock_tcsetattr, \
             patch('os.write') as mock_write:
            
            # Set up manager with modified state
            manager._initialized = True
            manager._tty_fd = mock_tty_fd
            manager._output_fd = sys.stdout.fileno()
            manager._original_state = TerminalState(
                fd=mock_tty_fd,
                attrs=mock_terminal_attrs,
                mode=TerminalMode.NORMAL
            )
            manager._state_changes = ["raw_mode", "cursor_hidden", "alternate_screen", "mouse_enabled"]
            
            result = manager.restore_terminal_state()
            
            assert result is True
            assert manager._current_mode == TerminalMode.NORMAL
            assert len(manager._state_changes) == 0
            
            # Verify termios restoration
            mock_tcsetattr.assert_called_once_with(
                mock_tty_fd, termios.TCSADRAIN, mock_terminal_attrs
            )
            
            # Verify visual state restoration
            expected_sequence = (
                b'\033[?1006l\033[?1000l'  # Disable mouse
                b'\033[?1049l'             # Exit alternate screen
                b'\033[?25h'               # Show cursor
                b'\033[0m\033[?25h'        # Reset graphics + show cursor
            )
            mock_write.assert_called_with(sys.stdout.fileno(), expected_sequence)
    
    def test_emergency_restoration(self, manager, mock_tty_fd, mock_terminal_attrs):
        """Test emergency terminal restoration (no exceptions allowed)."""
        manager._tty_fd = mock_tty_fd
        manager._output_fd = sys.stdout.fileno()
        manager._original_state = TerminalState(
            fd=mock_tty_fd,
            attrs=mock_terminal_attrs,
            mode=TerminalMode.NORMAL
        )
        
        # Mock failure scenarios - emergency restore should not raise exceptions
        with patch('termios.tcsetattr', side_effect=Exception("Test error")) as mock_tcsetattr, \
             patch('os.write', side_effect=Exception("Test error")) as mock_write:
            
            # Should not raise any exceptions
            manager._emergency_restore()
            
            # Should have attempted restoration
            mock_tcsetattr.assert_called_once()
            mock_write.assert_called_once()
    
    def test_signal_handler_registration(self, manager):
        """Test that signal handlers are properly registered."""
        with patch('signal.signal') as mock_signal, \
             patch('atexit.register') as mock_atexit:
            
            manager._register_cleanup_handlers()
            
            assert manager._cleanup_registered is True
            
            # Verify atexit handler
            mock_atexit.assert_called_once()
            
            # Verify signal handlers for common signals
            expected_signals = [signal.SIGINT, signal.SIGTERM, signal.SIGQUIT, signal.SIGHUP]
            for sig in expected_signals:
                mock_signal.assert_any_call(sig, manager._signal_handler)
    
    def test_signal_handler_execution(self, manager):
        """Test signal handler executes restoration."""
        with patch.object(manager, '_emergency_restore') as mock_restore, \
             patch('signal.signal') as mock_signal, \
             patch('os.kill') as mock_kill, \
             patch('os.getpid', return_value=12345) as mock_getpid:
            
            # Call signal handler
            manager._signal_handler(signal.SIGINT, None)
            
            # Should restore and re-raise signal
            mock_restore.assert_called_once()
            mock_signal.assert_called_with(signal.SIGINT, signal.SIG_DFL)
            mock_kill.assert_called_with(12345, signal.SIGINT)
    
    def test_thread_safety(self, manager, mock_tty_fd):
        """Test thread safety of terminal operations."""
        manager._initialized = True
        manager._tty_fd = mock_tty_fd
        
        results = []
        errors = []
        
        def concurrent_operation():
            try:
                # Multiple operations to test lock contention
                result1 = manager.enter_raw_mode()
                result2 = manager.hide_cursor()
                result3 = manager.restore_terminal_state()
                results.append((result1, result2, result3))
            except Exception as e:
                errors.append(e)
        
        with patch('tty.setraw'), \
             patch('os.write'), \
             patch('termios.tcsetattr'):
            
            # Run concurrent operations
            threads = [threading.Thread(target=concurrent_operation) for _ in range(5)]
            
            for t in threads:
                t.start()
            
            for t in threads:
                t.join()
            
            # Should complete without errors
            assert len(errors) == 0
            assert len(results) == 5
    
    def test_context_manager_interface(self, manager):
        """Test context manager interface."""
        with patch.object(manager, 'initialize', return_value=True) as mock_init, \
             patch.object(manager, 'cleanup') as mock_cleanup:
            
            with manager:
                assert mock_init.called
            
            mock_cleanup.assert_called_once()
    
    def test_context_manager_initialization_failure(self, manager):
        """Test context manager with initialization failure."""
        with patch.object(manager, 'initialize', return_value=False):
            
            with pytest.raises(RuntimeError, match="Failed to initialize"):
                with manager:
                    pass
    
    def test_cleanup_functionality(self, manager, mock_tty_fd):
        """Test complete cleanup functionality."""
        with patch.object(manager, '_restore_state_locked', return_value=True) as mock_restore, \
             patch('os.close') as mock_close:
            
            # Set up manager state
            manager._initialized = True
            manager._tty_fd = mock_tty_fd
            manager._output_fd = sys.stdout.fileno()
            manager._original_state = Mock()
            
            manager.cleanup()
            
            # Verify cleanup sequence
            mock_restore.assert_called_once()
            mock_close.assert_called_once_with(mock_tty_fd)
            
            # Verify state reset
            assert manager._initialized is False
            assert manager._tty_fd is None
            assert manager._output_fd is None
            assert manager._original_state is None
    
    def test_get_state_info(self, manager, mock_tty_fd):
        """Test state information retrieval."""
        manager._initialized = True
        manager._tty_fd = mock_tty_fd
        manager._output_fd = sys.stdout.fileno()
        manager._original_state = Mock()
        manager._state_changes = ["raw_mode", "cursor_hidden"]
        manager._cleanup_registered = True
        
        state_info = manager.get_state_info()
        
        expected_info = {
            'initialized': True,
            'current_mode': 'normal',
            'tty_fd': mock_tty_fd,
            'output_fd': sys.stdout.fileno(),
            'has_original_state': True,
            'state_changes': ["raw_mode", "cursor_hidden"],
            'cleanup_registered': True
        }
        
        assert state_info == expected_info
    
    def test_is_initialized(self, manager):
        """Test initialization status check."""
        assert manager.is_initialized() is False
        
        manager._initialized = True
        assert manager.is_initialized() is True
    
    def test_get_current_mode(self, manager):
        """Test current mode retrieval."""
        assert manager.get_current_mode() == TerminalMode.NORMAL
        
        manager._current_mode = TerminalMode.RAW
        assert manager.get_current_mode() == TerminalMode.RAW


class TestTerminalModeContextManagers:
    """Test convenience context managers."""
    
    def test_raw_mode_context_manager(self):
        """Test raw mode context manager."""
        with patch('agentsmcp.ui.v2.terminal_state_manager.TerminalStateManager') as MockManager:
            mock_instance = MockManager.return_value
            mock_instance.initialize.return_value = True
            mock_instance.enter_raw_mode.return_value = True
            
            with raw_mode() as manager:
                assert manager == mock_instance
            
            mock_instance.initialize.assert_called_once()
            mock_instance.enter_raw_mode.assert_called_once()
            mock_instance.cleanup.assert_called_once()
    
    def test_raw_mode_initialization_failure(self):
        """Test raw mode context manager initialization failure."""
        with patch('agentsmcp.ui.v2.terminal_state_manager.TerminalStateManager') as MockManager:
            mock_instance = MockManager.return_value
            mock_instance.initialize.return_value = False
            
            with pytest.raises(RuntimeError, match="Failed to initialize"):
                with raw_mode():
                    pass
            
            mock_instance.cleanup.assert_called_once()
    
    def test_raw_mode_enter_failure(self):
        """Test raw mode context manager mode entry failure."""
        with patch('agentsmcp.ui.v2.terminal_state_manager.TerminalStateManager') as MockManager:
            mock_instance = MockManager.return_value
            mock_instance.initialize.return_value = True
            mock_instance.enter_raw_mode.return_value = False
            
            with pytest.raises(RuntimeError, match="Failed to enter raw mode"):
                with raw_mode():
                    pass
            
            mock_instance.cleanup.assert_called_once()
    
    def test_cbreak_mode_context_manager(self):
        """Test cbreak mode context manager."""
        with patch('agentsmcp.ui.v2.terminal_state_manager.TerminalStateManager') as MockManager:
            mock_instance = MockManager.return_value
            mock_instance.initialize.return_value = True
            mock_instance.enter_cbreak_mode.return_value = True
            
            with cbreak_mode() as manager:
                assert manager == mock_instance
            
            mock_instance.initialize.assert_called_once()
            mock_instance.enter_cbreak_mode.assert_called_once()
            mock_instance.cleanup.assert_called_once()


class TestGlobalTerminalManager:
    """Test global terminal manager functionality."""
    
    def test_get_global_manager_singleton(self):
        """Test global manager singleton behavior."""
        # Clear any existing global manager
        import agentsmcp.ui.v2.terminal_state_manager as tsm_module
        tsm_module._global_manager = None
        
        manager1 = get_global_terminal_manager()
        manager2 = get_global_terminal_manager()
        
        assert manager1 is manager2
        assert isinstance(manager1, TerminalStateManager)
    
    def test_emergency_terminal_restore_with_manager(self):
        """Test emergency restore with active global manager."""
        import agentsmcp.ui.v2.terminal_state_manager as tsm_module
        
        mock_manager = Mock()
        tsm_module._global_manager = mock_manager
        
        emergency_terminal_restore()
        
        mock_manager._emergency_restore.assert_called_once()
    
    def test_emergency_terminal_restore_no_manager(self):
        """Test emergency restore with no global manager."""
        import agentsmcp.ui.v2.terminal_state_manager as tsm_module
        
        tsm_module._global_manager = None
        
        # Should not raise exception
        emergency_terminal_restore()


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_multiple_initialization_attempts(self, manager):
        """Test multiple initialization attempts."""
        with patch.object(manager, '_open_tty', return_value=3), \
             patch.object(manager, '_capture_original_state'), \
             patch.object(manager, '_register_cleanup_handlers'):
            
            result1 = manager.initialize()
            result2 = manager.initialize()  # Second attempt
            
            assert result1 is True
            assert result2 is True  # Should return True without re-initializing
    
    def test_operations_without_initialization(self, manager):
        """Test operations without proper initialization."""
        # Operations should fail gracefully
        assert manager.enter_raw_mode() is False
        assert manager.enter_cbreak_mode() is False
        assert manager.restore_terminal_state() is True  # Should succeed (no-op)
    
    def test_restoration_with_partial_state(self, manager):
        """Test restoration with partial state information."""
        manager._initialized = True
        manager._tty_fd = None  # No TTY
        manager._output_fd = sys.stdout.fileno()
        manager._state_changes = ["cursor_hidden"]
        
        with patch('os.write') as mock_write:
            result = manager.restore_terminal_state()
            
            # Should still attempt restoration
            assert result is True
            # Should try to restore visual state
            mock_write.assert_called_once()
    
    def test_tty_open_permission_edge_cases(self, manager):
        """Test TTY opening edge cases."""
        with patch('os.path.exists', return_value=True), \
             patch('os.open', side_effect=[PermissionError(), OSError(), 5]) as mock_open, \
             patch('os.isatty', return_value=True):
            
            fd = manager._open_tty()
            
            # Should try multiple paths and succeed on third attempt
            assert fd == 5
            assert mock_open.call_count >= 3


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    def test_typical_tui_lifecycle(self, manager):
        """Test typical TUI application lifecycle."""
        with patch.object(manager, '_open_tty', return_value=3), \
             patch('termios.tcgetattr', return_value=[1, 2, 3, 4, 5, 6]), \
             patch('tty.setraw'), \
             patch('os.write'), \
             patch('termios.tcsetattr'), \
             patch.object(manager, '_register_cleanup_handlers'):
            
            # 1. Initialize
            assert manager.initialize() is True
            assert manager.is_initialized() is True
            
            # 2. Enter raw mode for TUI
            assert manager.enter_raw_mode() is True
            assert manager.get_current_mode() == TerminalMode.RAW
            
            # 3. Set up visual state
            assert manager.hide_cursor() is True
            assert manager.enter_alternate_screen() is True
            
            # 4. Check state
            state_info = manager.get_state_info()
            assert state_info['initialized'] is True
            assert state_info['current_mode'] == 'raw'
            assert 'cursor_hidden' in state_info['state_changes']
            assert 'alternate_screen' in state_info['state_changes']
            
            # 5. Restore and cleanup
            assert manager.restore_terminal_state() is True
            manager.cleanup()
            
            assert manager.is_initialized() is False
    
    def test_interrupt_during_operation(self, manager):
        """Test handling of interrupts during operation."""
        with patch.object(manager, '_open_tty', return_value=3), \
             patch('termios.tcgetattr', return_value=[1, 2, 3, 4, 5, 6]), \
             patch.object(manager, '_register_cleanup_handlers'), \
             patch.object(manager, '_emergency_restore') as mock_emergency:
            
            manager.initialize()
            
            # Simulate signal reception
            manager._signal_handler(signal.SIGINT, None)
            
            # Should call emergency restore
            mock_emergency.assert_called_once()
    
    def test_concurrent_terminal_usage(self, manager):
        """Test concurrent terminal state changes."""
        with patch('tty.setraw'), \
             patch('tty.setcbreak'), \
             patch('os.write'), \
             patch('termios.tcsetattr'):
            
            manager._initialized = True
            manager._tty_fd = 3
            manager._output_fd = 1
            
            # Simulate concurrent operations from different threads
            results = []
            
            def worker():
                results.append(manager.enter_raw_mode())
                results.append(manager.hide_cursor())
                results.append(manager.enter_cbreak_mode())
                results.append(manager.show_cursor())
            
            threads = [threading.Thread(target=worker) for _ in range(3)]
            
            for t in threads:
                t.start()
            
            for t in threads:
                t.join()
            
            # All operations should succeed
            assert all(results)
            # Final state should be consistent
            assert isinstance(manager._current_mode, TerminalMode)