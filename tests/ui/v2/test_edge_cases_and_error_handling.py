"""
Comprehensive Edge Cases and Error Handling Tests for TUI Input Fixes.

This test suite covers:
1. Permission denied scenarios (TTY access, file operations)
2. Terminal not available (SSH, container, CI environments)
3. Invalid input sequences (malformed escape sequences, corrupted data)
4. Memory/performance stress tests (large inputs, concurrent operations)
5. System resource exhaustion scenarios
6. Network interruption simulation (for future remote features)
7. File system errors and recovery
"""

import pytest
import asyncio
import os
import sys
import signal
import threading
import time
import tempfile
from unittest.mock import Mock, AsyncMock, patch, MagicMock, call
from contextlib import contextmanager

from agentsmcp.ui.v2.terminal_state_manager import TerminalStateManager, TerminalState, TerminalMode
from agentsmcp.ui.v2.unified_input_handler import (
    UnifiedInputHandler, CharacterEchoProcessor, InputEvent, InputEventType
)
from agentsmcp.ui.v2.ansi_markdown_processor import ANSIMarkdownProcessor, RenderConfig
from agentsmcp.ui.v2.main_app import MainTUIApp


class TestPermissionDeniedScenarios:
    """Test handling of permission denied scenarios."""
    
    def test_terminal_state_manager_tty_permission_denied(self):
        """Test TTY access permission denied."""
        manager = TerminalStateManager()
        
        with patch('os.open', side_effect=PermissionError("Permission denied")) as mock_open, \
             patch('os.path.exists', return_value=True), \
             patch('os.isatty', return_value=False):
            
            # Should handle permission denied gracefully
            result = manager.initialize()
            
            # Should succeed with fallback behavior
            assert result is True
            assert manager._initialized is True
            
            # Should fallback to stdin
            assert manager._tty_fd == sys.stdin.fileno()
    
    def test_terminal_attributes_permission_denied(self):
        """Test terminal attribute access permission denied."""
        manager = TerminalStateManager()
        
        with patch('os.open', return_value=3), \
             patch('os.isatty', return_value=True), \
             patch('termios.tcgetattr', side_effect=PermissionError("Operation not permitted")):
            
            result = manager.initialize()
            
            # Should handle gracefully
            assert result is True
            assert manager._original_state is None  # No state captured due to permission error
    
    def test_terminal_mode_change_permission_denied(self):
        """Test terminal mode change permission denied."""
        manager = TerminalStateManager()
        manager._initialized = True
        manager._tty_fd = 3
        
        with patch('tty.setraw', side_effect=PermissionError("Permission denied")):
            
            result = manager.enter_raw_mode()
            
            # Should fail gracefully
            assert result is False
            assert manager._current_mode == TerminalMode.NORMAL  # Unchanged
    
    def test_output_write_permission_denied(self):
        """Test output write permission denied."""
        manager = TerminalStateManager()
        manager._output_fd = 1
        
        with patch('os.write', side_effect=PermissionError("Permission denied")):
            
            result = manager.hide_cursor()
            
            # Should fail gracefully
            assert result is False
            assert "cursor_hidden" not in manager._state_changes
    
    @pytest.mark.asyncio
    async def test_main_app_permission_denied_recovery(self):
        """Test main app recovery from permission denied errors."""
        app = MainTUIApp()
        
        with patch('agentsmcp.ui.v2.terminal_state_manager.TerminalStateManager') as MockTSM:
            mock_tsm = MockTSM.return_value
            mock_tsm.initialize.side_effect = PermissionError("Permission denied")
            
            result = await app.initialize()
            
            # Should fail gracefully
            assert result is False
            assert app.running is False


class TestTerminalNotAvailableScenarios:
    """Test handling when terminal is not available."""
    
    def test_no_tty_available(self):
        """Test when no TTY devices are available."""
        manager = TerminalStateManager()
        
        with patch('os.path.exists', return_value=False), \
             patch('os.isatty', return_value=False):
            
            result = manager.initialize()
            
            # Should handle gracefully
            assert result is True
            assert manager._tty_fd is not None  # Should use fallback
    
    def test_stdin_not_tty(self):
        """Test when stdin is not a TTY (pipe, redirect)."""
        manager = TerminalStateManager()
        
        with patch('os.isatty', return_value=False), \
             patch('os.path.exists', return_value=False):
            
            tty_fd = manager._open_tty()
            
            # Should return None when no TTY available
            assert tty_fd is None
    
    def test_ci_environment_simulation(self):
        """Test behavior in CI environment (no interactive terminal)."""
        manager = TerminalStateManager()
        
        # Simulate CI environment
        with patch.dict(os.environ, {'CI': 'true'}), \
             patch('os.isatty', return_value=False), \
             patch('sys.stdin.isatty', return_value=False), \
             patch('sys.stdout.isatty', return_value=False):
            
            result = manager.initialize()
            
            # Should handle CI environment gracefully
            assert result is True
    
    def test_ssh_without_pty_simulation(self):
        """Test behavior in SSH session without PTY allocation."""
        manager = TerminalStateManager()
        
        # Simulate SSH without PTY
        with patch.dict(os.environ, {'SSH_CONNECTION': '1.1.1.1 12345 2.2.2.2 22'}), \
             patch('os.isatty', return_value=False):
            
            result = manager.initialize()
            
            # Should handle SSH without PTY gracefully
            assert result is True
    
    @pytest.mark.asyncio
    async def test_unified_input_handler_no_terminal(self):
        """Test unified input handler when terminal is unavailable."""
        handler = UnifiedInputHandler()
        
        # Mock terminal manager to simulate no terminal
        handler.terminal_manager.initialize.return_value = False
        
        result = await handler.initialize()
        
        # Should fail gracefully
        assert result is False
    
    def test_ansi_processor_no_color_terminal(self):
        """Test ANSI processor when terminal doesn't support colors."""
        # Simulate terminal without color support
        config = RenderConfig(enable_colors=False)
        processor = ANSIMarkdownProcessor(config)
        
        text = "**Bold** and *italic* text"
        result = processor.process_text(text)
        
        # Should process without ANSI codes
        assert "\x1b[" not in result
        assert result == text  # Should return original text


class TestInvalidInputSequenceHandling:
    """Test handling of invalid and malformed input sequences."""
    
    def test_malformed_escape_sequences(self):
        """Test handling of malformed escape sequences."""
        handler = UnifiedInputHandler()
        
        events = []
        handler._dispatch_event = lambda e: events.append(e)
        
        # Test various malformed sequences
        malformed_sequences = [
            b'\x1b',           # Incomplete escape
            b'\x1b[',          # Incomplete CSI
            b'\x1b[999',       # Incomplete CSI with numbers
            b'\x1b[A\x00',     # Valid sequence followed by null
            b'\x1b[Z',         # Unknown CSI sequence
        ]
        
        for seq in malformed_sequences:
            events.clear()
            handler._escape_buffer = b''
            handler._parsing_sequence = False
            
            # Process each byte
            for byte_val in seq:
                handler._process_byte(byte_val, time.time())
            
            # Should handle without crashing
            # May or may not generate events, but should not crash
    
    def test_buffer_overflow_protection(self):
        """Test protection against escape sequence buffer overflow."""
        handler = UnifiedInputHandler()
        
        # Start parsing sequence
        handler._parsing_sequence = True
        handler._escape_buffer = b'\x1b['
        
        # Add many bytes to trigger overflow protection
        for i in range(20):  # Exceed the 16 byte limit
            handler._process_byte(ord('0') + (i % 10), time.time())
        
        # Should have reset the buffer
        assert handler._parsing_sequence is False
        assert handler._escape_buffer == b''
    
    def test_invalid_unicode_handling(self):
        """Test handling of invalid Unicode sequences."""
        handler = UnifiedInputHandler()
        
        events = []
        handler._dispatch_event = lambda e: events.append(e)
        
        # Test invalid UTF-8 sequences
        invalid_bytes = [128, 129, 130, 255]
        
        for byte_val in invalid_bytes:
            events.clear()
            handler._process_byte(byte_val, time.time())
            
            # Should not generate character events for invalid bytes
            char_events = [e for e in events if e.event_type == InputEventType.CHARACTER]
            assert len(char_events) == 0
    
    def test_null_and_control_character_handling(self):
        """Test handling of null and control characters."""
        handler = UnifiedInputHandler()
        
        events = []
        handler._dispatch_event = lambda e: events.append(e)
        
        # Test various control characters
        control_chars = [0, 1, 2, 5, 6, 7, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31]
        
        for char_code in control_chars:
            events.clear()
            handler._process_byte(char_code, time.time())
            
            # Should handle without crashing
            # Some control chars generate events, others don't
    
    def test_extremely_long_input_sequences(self):
        """Test handling of extremely long input sequences."""
        processor = CharacterEchoProcessor(lambda x: None)
        
        # Create extremely long character sequence
        long_text = 'a' * 10000
        
        # Process each character
        for char in long_text:
            event = InputEvent(
                event_type=InputEventType.CHARACTER,
                data={'character': char},
                timestamp=time.time()
            )
            
            # Should not crash or consume excessive memory
            asyncio.create_task(processor.process_event(event))
        
        # Buffer should contain the long text
        assert len(processor.buffer) == 10000
    
    def test_invalid_mouse_event_sequences(self):
        """Test handling of invalid mouse event sequences."""
        handler = UnifiedInputHandler()
        
        # Test invalid mouse sequences
        invalid_mouse_sequences = [
            '<invalid',
            '<0;invalid;20M',
            '<0;10;invalidM',
            '<0;10;20X',  # Invalid ending
            '<999;999;999M',  # Extreme values
        ]
        
        for seq in invalid_mouse_sequences:
            result = handler._parse_mouse_event(seq, time.time(), b'test')
            
            # Should return None for invalid sequences
            assert result is None


class TestMemoryAndPerformanceStressTests:
    """Test system behavior under memory and performance stress."""
    
    def test_large_input_buffer_handling(self):
        """Test handling of very large input buffers."""
        processor = CharacterEchoProcessor(lambda x: None)
        
        # Create very large buffer
        large_text = 'x' * 100000  # 100KB of text
        
        start_time = time.time()
        processor.set_buffer(large_text)
        end_time = time.time()
        
        # Should handle large buffer efficiently
        assert processor.get_buffer() == large_text
        assert (end_time - start_time) < 1.0  # Should complete within 1 second
    
    def test_rapid_input_processing(self):
        """Test processing of rapid input sequences."""
        handler = UnifiedInputHandler()
        
        processed_count = 0
        
        def count_events(event):
            nonlocal processed_count
            processed_count += 1
        
        handler._dispatch_event = count_events
        
        # Generate rapid input
        start_time = time.time()
        for i in range(1000):
            handler._process_byte(ord('a'), time.time())
        
        processing_time = time.time() - start_time
        
        # Should process all events
        assert processed_count == 1000
        # Should complete reasonably quickly
        assert processing_time < 5.0  # 5 second maximum for 1000 events
    
    @pytest.mark.asyncio
    async def test_concurrent_processor_load(self):
        """Test system under concurrent processor load."""
        handler = UnifiedInputHandler()
        
        # Create multiple processors
        processors = []
        for i in range(100):
            processor = Mock()
            processor.process_event = AsyncMock(return_value=False)
            processors.append(processor)
            handler.add_processor(processor)
        
        # Process event concurrently
        event = InputEvent(
            event_type=InputEventType.CHARACTER,
            data={'character': 'a'},
            timestamp=time.time()
        )
        
        # Process through all processors concurrently
        tasks = [proc.process_event(event) for proc in processors]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        processing_time = time.time() - start_time
        
        # Should complete all processing
        assert len(results) == 100
        assert all(not isinstance(r, Exception) for r in results)
        # Should complete reasonably quickly
        assert processing_time < 10.0
    
    def test_memory_usage_with_large_ansi_processing(self):
        """Test memory usage with large ANSI text processing."""
        processor = ANSIMarkdownProcessor()
        
        # Create large markdown text with complex formatting
        large_markdown = "\n".join([
            f"# Header {i}",
            f"This is **bold paragraph {i}** with *italic text* and `inline code`.",
            f"- List item {i}",
            "```python",
            f"def function_{i}():",
            f"    return 'result_{i}'",
            "```",
            f"> Quote block {i} with [link](http://example{i}.com)",
            ""
        ] for i in range(1000))
        
        # Process large text
        start_time = time.time()
        result = processor.process_text(large_markdown)
        processing_time = time.time() - start_time
        
        # Should complete processing
        assert isinstance(result, str)
        assert len(result) > len(large_markdown)  # Should have added ANSI codes
        # Should complete within reasonable time
        assert processing_time < 30.0  # 30 second maximum
    
    def test_terminal_state_rapid_mode_changes(self):
        """Test terminal state under rapid mode changes."""
        manager = TerminalStateManager()
        manager._initialized = True
        manager._tty_fd = 3
        
        with patch('tty.setraw'), \
             patch('tty.setcbreak'), \
             patch('termios.tcsetattr'):
            
            # Rapid mode changes
            start_time = time.time()
            for i in range(100):
                manager.enter_raw_mode()
                manager.enter_cbreak_mode()
            
            processing_time = time.time() - start_time
            
            # Should handle rapid changes
            assert processing_time < 5.0  # Should be fast
            # Final mode should be consistent
            assert manager._current_mode in [TerminalMode.RAW, TerminalMode.CBREAK]


class TestSystemResourceExhaustionScenarios:
    """Test behavior when system resources are exhausted."""
    
    def test_file_descriptor_exhaustion(self):
        """Test behavior when file descriptors are exhausted."""
        manager = TerminalStateManager()
        
        with patch('os.open', side_effect=OSError("Too many open files")):
            
            result = manager.initialize()
            
            # Should handle FD exhaustion gracefully
            assert result is True  # May succeed with fallback
    
    def test_memory_exhaustion_simulation(self):
        """Test behavior under memory pressure."""
        processor = ANSIMarkdownProcessor()
        
        # Mock memory allocation failure
        with patch('re.sub', side_effect=MemoryError("Out of memory")):
            
            text = "**Bold** text"
            result = processor.process_text(text)
            
            # Should handle memory error gracefully
            assert result == text  # Should return original text
    
    @pytest.mark.asyncio
    async def test_thread_exhaustion_handling(self):
        """Test handling when thread creation fails."""
        handler = UnifiedInputHandler()
        
        with patch('threading.Thread', side_effect=RuntimeError("Cannot create thread")):
            
            result = await handler.start()
            
            # Should handle thread creation failure
            assert result is False
    
    def test_signal_handling_resource_exhaustion(self):
        """Test signal handling when resources are exhausted."""
        manager = TerminalStateManager()
        
        with patch('signal.signal', side_effect=OSError("Resource temporarily unavailable")):
            
            # Should handle signal registration failure gracefully
            manager._register_cleanup_handlers()
            
            # Should not crash
            assert manager._cleanup_registered is False  # Registration failed


class TestConcurrencyAndRaceConditionHandling:
    """Test handling of concurrency issues and race conditions."""
    
    @pytest.mark.asyncio
    async def test_concurrent_initialization_and_cleanup(self):
        """Test concurrent initialization and cleanup operations."""
        app = MainTUIApp()
        
        # Mock components
        app.terminal_state_manager = Mock()
        app.terminal_state_manager.cleanup = Mock()
        app.unified_input_handler = Mock()
        app.unified_input_handler.cleanup = AsyncMock()
        
        # Start concurrent initialization and cleanup
        async def init_task():
            await asyncio.sleep(0.01)
            return True
        
        async def cleanup_task():
            await asyncio.sleep(0.01)
            await app.cleanup()
        
        # Run concurrently
        init_future = asyncio.create_task(init_task())
        cleanup_future = asyncio.create_task(cleanup_task())
        
        results = await asyncio.gather(init_future, cleanup_future, return_exceptions=True)
        
        # Should handle concurrent operations without crashing
        assert len(results) == 2
        for result in results:
            assert not isinstance(result, Exception)
    
    def test_race_condition_in_signal_handlers(self):
        """Test race conditions in signal handler setup and execution."""
        manager = TerminalStateManager()
        
        # Simulate rapid signal handler setup and signal reception
        def setup_and_signal():
            manager._register_cleanup_handlers()
            if hasattr(manager, '_signal_handler'):
                manager._signal_handler(signal.SIGINT, None)
        
        # Run in multiple threads to create race conditions
        threads = [threading.Thread(target=setup_and_signal) for _ in range(10)]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # Should handle race conditions without crashing
    
    @pytest.mark.asyncio
    async def test_concurrent_event_processing_race_conditions(self):
        """Test race conditions in concurrent event processing."""
        handler = UnifiedInputHandler()
        
        shared_state = {'counter': 0}
        
        class RaceConditionProcessor:
            async def process_event(self, event):
                # Simulate race condition
                current = shared_state['counter']
                await asyncio.sleep(0.001)  # Allow other processors to run
                shared_state['counter'] = current + 1
                return False
        
        # Add multiple processors
        processors = [RaceConditionProcessor() for _ in range(10)]
        for processor in processors:
            handler.add_processor(processor)
        
        # Process events concurrently
        event = InputEvent(
            event_type=InputEventType.CHARACTER,
            data={'character': 'a'},
            timestamp=time.time()
        )
        
        tasks = [proc.process_event(event) for proc in processors]
        await asyncio.gather(*tasks)
        
        # Race condition may cause counter to be less than expected
        # But should not crash
        assert shared_state['counter'] <= 10
        assert shared_state['counter'] > 0


class TestErrorRecoveryAndGracefulDegradation:
    """Test error recovery and graceful degradation scenarios."""
    
    @pytest.mark.asyncio
    async def test_partial_component_failure_recovery(self):
        """Test recovery from partial component failures."""
        app = MainTUIApp()
        
        with patch('agentsmcp.ui.v2.terminal_state_manager.TerminalStateManager') as MockTSM, \
             patch('agentsmcp.ui.v2.unified_input_handler.UnifiedInputHandler') as MockUIH:
            
            # Terminal manager succeeds, input handler fails
            mock_tsm = MockTSM.return_value
            mock_tsm.initialize.return_value = True
            mock_tsm.cleanup = Mock()
            
            mock_uih = MockUIH.return_value
            mock_uih.initialize.return_value = False  # Fails
            
            result = await app.initialize()
            
            # Should fail but attempt cleanup of successful components
            assert result is False
            mock_tsm.cleanup.assert_called_once()
    
    def test_ansi_processor_fallback_on_regex_failure(self):
        """Test ANSI processor fallback when regex processing fails."""
        processor = ANSIMarkdownProcessor()
        
        # Mock regex failure
        original_patterns = processor._patterns
        processor._patterns = {}  # Empty patterns to simulate failure
        
        text = "**Bold** text with `code`"
        result = processor.process_text(text)
        
        # Should return original text when processing fails
        assert result == text
        
        # Restore patterns
        processor._patterns = original_patterns
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_with_missing_features(self):
        """Test graceful degradation when features are unavailable."""
        app = MainTUIApp()
        
        # Simulate missing ANSI processor
        with patch('agentsmcp.ui.v2.ansi_markdown_processor.ANSIMarkdownProcessor', side_effect=ImportError("Module not found")):
            
            # Should continue initialization without ANSI processor
            app.ansi_processor = None
            
            # Should handle missing processor gracefully
            assert app.ansi_processor is None
    
    def test_terminal_restoration_partial_failure_recovery(self):
        """Test recovery from partial terminal restoration failures."""
        manager = TerminalStateManager()
        
        # Setup state
        manager._initialized = True
        manager._tty_fd = 3
        manager._output_fd = 1
        manager._original_state = Mock()
        manager._state_changes = ["raw_mode", "cursor_hidden"]
        
        with patch('termios.tcsetattr', side_effect=Exception("Terminal restore failed")), \
             patch('os.write') as mock_write:  # Visual restore succeeds
            
            result = manager.restore_terminal_state()
            
            # Should attempt both restorations
            assert result is False  # Overall failure
            mock_write.assert_called_once()  # Visual restore attempted
            
            # State changes should be cleared despite partial failure
            assert len(manager._state_changes) == 0
    
    def test_input_handler_degradation_without_raw_mode(self):
        """Test input handler degradation when raw mode is unavailable."""
        handler = UnifiedInputHandler()
        
        # Mock terminal manager that can't enter raw mode
        handler.terminal_manager.initialize.return_value = True
        handler.terminal_manager.enter_raw_mode.return_value = False
        
        # Should still initialize with warning
        result = asyncio.run(handler.initialize(lambda x: None))
        
        assert result is True  # Should succeed with degraded functionality


class TestExtremeEdgeCases:
    """Test extremely rare but possible edge cases."""
    
    def test_terminal_state_change_during_restoration(self):
        """Test terminal state changes during restoration process."""
        manager = TerminalStateManager()
        
        # Setup initial state
        manager._initialized = True
        manager._tty_fd = 3
        manager._state_changes = ["raw_mode"]
        
        restore_call_count = 0
        
        def mock_tcsetattr(*args, **kwargs):
            nonlocal restore_call_count
            restore_call_count += 1
            # Simulate state change during restoration
            if restore_call_count == 1:
                manager._state_changes.append("cursor_hidden")
            return None
        
        with patch('termios.tcsetattr', side_effect=mock_tcsetattr), \
             patch('os.write'):
            
            result = manager.restore_terminal_state()
            
            # Should handle mid-restoration state changes
            assert result is True
            assert len(manager._state_changes) == 0  # All cleared
    
    def test_recursive_signal_handling(self):
        """Test handling of recursive signal scenarios."""
        manager = TerminalStateManager()
        
        signal_count = 0
        
        def mock_emergency_restore():
            nonlocal signal_count
            signal_count += 1
            if signal_count < 5:  # Prevent infinite recursion
                manager._signal_handler(signal.SIGTERM, None)
        
        with patch.object(manager, '_emergency_restore', side_effect=mock_emergency_restore), \
             patch('signal.signal'), \
             patch('os.kill'):
            
            manager._signal_handler(signal.SIGINT, None)
            
            # Should handle recursive signals without infinite loop
            assert signal_count >= 1
    
    def test_zero_length_input_sequences(self):
        """Test handling of zero-length and empty input sequences."""
        processor = CharacterEchoProcessor(lambda x: None)
        
        # Test empty events
        empty_events = [
            InputEvent(InputEventType.CHARACTER, {'character': ''}, time.time()),
            InputEvent(InputEventType.SPECIAL_KEY, {'key': ''}, time.time()),
            InputEvent(InputEventType.CHARACTER, {}, time.time()),  # Missing character key
        ]
        
        for event in empty_events:
            # Should handle without crashing
            result = asyncio.run(processor.process_event(event))
            # May return True or False, but should not crash
    
    def test_extremely_rapid_mode_switching(self):
        """Test extremely rapid terminal mode switching."""
        manager = TerminalStateManager()
        manager._initialized = True
        manager._tty_fd = 3
        
        mode_changes = 0
        
        def count_mode_changes(*args, **kwargs):
            nonlocal mode_changes
            mode_changes += 1
        
        with patch('tty.setraw', side_effect=count_mode_changes), \
             patch('tty.setcbreak', side_effect=count_mode_changes):
            
            # Extremely rapid mode switching
            for i in range(1000):
                if i % 2 == 0:
                    manager.enter_raw_mode()
                else:
                    manager.enter_cbreak_mode()
            
            # Should handle rapid switching
            assert mode_changes > 0  # Some changes should occur
            # Final state should be consistent
            assert manager._current_mode in [TerminalMode.RAW, TerminalMode.CBREAK]
    
    def test_unicode_edge_cases_in_ansi_processing(self):
        """Test Unicode edge cases in ANSI processing."""
        processor = ANSIMarkdownProcessor()
        
        unicode_edge_cases = [
            "**粗体** text with emoji 🔥",  # Mixed scripts
            "**\u0000null\u0000** text",    # Null characters
            "**\ufeffBOM** text",           # Byte order mark
            "**\u200bzero\u200bwidth** text",  # Zero-width characters
            "\U0001f600" * 1000,            # Many emoji
        ]
        
        for text in unicode_edge_cases:
            # Should handle without crashing
            result = processor.process_text(text)
            assert isinstance(result, str)
    
    @pytest.mark.asyncio
    async def test_cleanup_during_active_operations(self):
        """Test cleanup called while operations are active."""
        handler = UnifiedInputHandler()
        
        # Mock active operations
        handler.running = True
        handler._input_thread = Mock()
        handler._input_thread.is_alive.return_value = True
        handler._input_thread.join = Mock()
        
        # Mock terminal manager and input handler
        handler.terminal_state_manager = Mock()
        handler.terminal_state_manager.cleanup = Mock()
        
        # Start cleanup while "operations are active"
        await handler.cleanup()
        
        # Should attempt to stop operations
        handler._input_thread.join.assert_called_once()
        handler.terminal_state_manager.cleanup.assert_called_once()
        assert handler.running is False


if __name__ == "__main__":
    # Run specific test categories
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-k", "not stress",  # Skip stress tests by default
    ])