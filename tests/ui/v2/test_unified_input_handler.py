"""
Comprehensive tests for UnifiedInputHandler - Immediate character echo and conflict resolution.

This test suite verifies that the UnifiedInputHandler:
1. Shows typed characters immediately (CRITICAL - fixes main typing issue)
2. Handles raw terminal input correctly
3. Processes special keys and escape sequences
4. Dispatches events to processors correctly
5. Provides thread-safe input processing
6. Handles cleanup gracefully
"""

import pytest
import asyncio
import threading
import time
import os
import select
from unittest.mock import Mock, AsyncMock, patch, MagicMock, call
from contextlib import asynccontextmanager

from agentsmcp.ui.v2.unified_input_handler import (
    UnifiedInputHandler, InputProcessor, CharacterEchoProcessor,
    InputEvent, InputEventType
)
from agentsmcp.ui.v2.terminal_state_manager import TerminalStateManager, TerminalMode


class MockInputProcessor(InputProcessor):
    """Mock input processor for testing."""
    
    def __init__(self):
        self.processed_events = []
        self.should_handle = False
    
    async def process_event(self, event: InputEvent) -> bool:
        self.processed_events.append(event)
        return self.should_handle


class TestCharacterEchoProcessor:
    """Test character echo processor functionality."""
    
    @pytest.fixture
    def output_handler(self):
        """Mock output handler."""
        return Mock()
    
    @pytest.fixture
    def processor(self, output_handler):
        """Create character echo processor."""
        return CharacterEchoProcessor(output_handler)
    
    @pytest.mark.asyncio
    async def test_character_input_immediate_echo(self, processor, output_handler):
        """Test that characters are echoed immediately (CRITICAL TEST)."""
        # Create character event
        event = InputEvent(
            event_type=InputEventType.CHARACTER,
            data={'character': 'a'},
            timestamp=time.time()
        )
        
        result = await processor.process_event(event)
        
        assert result is True
        assert processor.buffer == 'a'
        assert processor.cursor_pos == 1
        # CRITICAL: Output handler should be called immediately for echo
        output_handler.assert_called_once()
        
        # Verify the rendered output includes the cursor
        rendered = output_handler.call_args[0][0]
        assert 'a' in rendered
        assert '█' in rendered  # Cursor indicator
    
    @pytest.mark.asyncio
    async def test_multiple_character_input(self, processor, output_handler):
        """Test multiple character inputs build buffer correctly."""
        characters = ['h', 'e', 'l', 'l', 'o']
        
        for char in characters:
            event = InputEvent(
                event_type=InputEventType.CHARACTER,
                data={'character': char},
                timestamp=time.time()
            )
            await processor.process_event(event)
        
        assert processor.buffer == 'hello'
        assert processor.cursor_pos == 5
        assert output_handler.call_count == 5  # Each character echoed immediately
    
    @pytest.mark.asyncio
    async def test_backspace_handling(self, processor, output_handler):
        """Test backspace removes characters."""
        # Set up buffer with content
        processor.buffer = "hello"
        processor.cursor_pos = 5
        
        # Send backspace event
        event = InputEvent(
            event_type=InputEventType.SPECIAL_KEY,
            data={'key': 'backspace'},
            timestamp=time.time()
        )
        
        result = await processor.process_event(event)
        
        assert result is True
        assert processor.buffer == "hell"
        assert processor.cursor_pos == 4
        output_handler.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_backspace_at_beginning(self, processor, output_handler):
        """Test backspace at buffer beginning does nothing."""
        processor.cursor_pos = 0
        
        event = InputEvent(
            event_type=InputEventType.SPECIAL_KEY,
            data={'key': 'backspace'},
            timestamp=time.time()
        )
        
        result = await processor.process_event(event)
        
        assert result is True
        assert processor.buffer == ""
        assert processor.cursor_pos == 0
        output_handler.assert_called_once()  # Still echoes
    
    @pytest.mark.asyncio
    async def test_delete_key_handling(self, processor, output_handler):
        """Test delete key removes character at cursor."""
        processor.buffer = "hello"
        processor.cursor_pos = 2  # Between 'e' and 'l'
        
        event = InputEvent(
            event_type=InputEventType.SPECIAL_KEY,
            data={'key': 'delete'},
            timestamp=time.time()
        )
        
        result = await processor.process_event(event)
        
        assert result is True
        assert processor.buffer == "helo"  # Third 'l' deleted
        assert processor.cursor_pos == 2
        output_handler.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cursor_movement_left_right(self, processor, output_handler):
        """Test cursor movement with arrow keys."""
        processor.buffer = "hello"
        processor.cursor_pos = 5
        
        # Move left
        left_event = InputEvent(
            event_type=InputEventType.SPECIAL_KEY,
            data={'key': 'left'},
            timestamp=time.time()
        )
        
        result = await processor.process_event(left_event)
        
        assert result is True
        assert processor.cursor_pos == 4
        output_handler.assert_called_with(processor._render_buffer())
        
        # Move right
        right_event = InputEvent(
            event_type=InputEventType.SPECIAL_KEY,
            data={'key': 'right'},
            timestamp=time.time()
        )
        
        result = await processor.process_event(right_event)
        
        assert result is True
        assert processor.cursor_pos == 5
    
    @pytest.mark.asyncio
    async def test_cursor_movement_boundaries(self, processor, output_handler):
        """Test cursor movement at boundaries."""
        processor.buffer = "test"
        processor.cursor_pos = 0
        
        # Try to move left at beginning
        left_event = InputEvent(
            event_type=InputEventType.SPECIAL_KEY,
            data={'key': 'left'},
            timestamp=time.time()
        )
        
        result = await processor.process_event(left_event)
        assert result is False  # Should not handle
        assert processor.cursor_pos == 0
        
        # Move to end
        processor.cursor_pos = 4
        
        # Try to move right at end
        right_event = InputEvent(
            event_type=InputEventType.SPECIAL_KEY,
            data={'key': 'right'},
            timestamp=time.time()
        )
        
        result = await processor.process_event(right_event)
        assert result is False  # Should not handle
        assert processor.cursor_pos == 4
    
    @pytest.mark.asyncio
    async def test_non_printable_characters_ignored(self, processor, output_handler):
        """Test that non-printable characters are ignored."""
        event = InputEvent(
            event_type=InputEventType.CHARACTER,
            data={'character': '\x00'},  # Null character
            timestamp=time.time()
        )
        
        result = await processor.process_event(event)
        
        assert result is False  # Should not handle non-printable
        assert processor.buffer == ""
        output_handler.assert_not_called()
    
    def test_buffer_manipulation(self, processor, output_handler):
        """Test direct buffer manipulation methods."""
        # Set buffer
        processor.set_buffer("test")
        assert processor.buffer == "test"
        assert processor.cursor_pos == 4
        output_handler.assert_called_once()
        
        # Get buffer
        buffer_content = processor.get_buffer()
        assert buffer_content == "test"
        
        # Clear buffer
        processor.clear_buffer()
        assert processor.buffer == ""
        assert processor.cursor_pos == 0
        output_handler.assert_called_with("")  # Called with empty string
    
    def test_cursor_rendering(self, processor, output_handler):
        """Test cursor rendering in different positions."""
        processor.buffer = "hello"
        
        # Cursor at beginning
        processor.cursor_pos = 0
        rendered = processor._render_buffer()
        assert rendered == "█hello"
        
        # Cursor in middle
        processor.cursor_pos = 2
        rendered = processor._render_buffer()
        assert rendered == "he█llo"
        
        # Cursor at end
        processor.cursor_pos = 5
        rendered = processor._render_buffer()
        assert rendered == "hello█"


class TestUnifiedInputHandler:
    """Test unified input handler functionality."""
    
    @pytest.fixture
    def handler(self):
        """Create unified input handler."""
        return UnifiedInputHandler()
    
    @pytest.fixture
    def mock_terminal_manager(self):
        """Mock terminal state manager."""
        manager = Mock(spec=TerminalStateManager)
        manager.initialize.return_value = True
        manager.enter_raw_mode.return_value = True
        manager._tty_fd = 3
        return manager
    
    @pytest.fixture
    def output_handler(self):
        """Mock output handler for immediate echo."""
        return Mock()
    
    @pytest.mark.asyncio
    async def test_initialization_success(self, handler, output_handler):
        """Test successful handler initialization."""
        with patch.object(handler.terminal_manager, 'initialize', return_value=True), \
             patch.object(handler.terminal_manager, 'enter_raw_mode', return_value=True):
            
            result = await handler.initialize(output_handler)
            
            assert result is True
            assert handler.echo_processor is not None
            assert handler.echo_processor in handler.processors
    
    @pytest.mark.asyncio
    async def test_initialization_failure(self, handler, output_handler):
        """Test handler initialization failure."""
        with patch.object(handler.terminal_manager, 'initialize', return_value=False):
            
            result = await handler.initialize(output_handler)
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_initialization_without_output_handler(self, handler):
        """Test initialization without output handler."""
        with patch.object(handler.terminal_manager, 'initialize', return_value=True), \
             patch.object(handler.terminal_manager, 'enter_raw_mode', return_value=True):
            
            result = await handler.initialize()
            
            assert result is True
            assert handler.echo_processor is None
    
    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self, handler):
        """Test handler start/stop lifecycle."""
        with patch.object(handler, 'initialize', return_value=True):
            
            await handler.initialize()
            
            # Start handler
            result = await handler.start()
            assert result is True
            assert handler.running is True
            assert handler._input_thread is not None
            assert handler._input_thread.is_alive()
            
            # Stop handler
            await handler.stop()
            assert handler.running is False
            
            # Thread should terminate
            time.sleep(0.2)  # Give thread time to terminate
            assert not handler._input_thread.is_alive()
    
    def test_processor_management(self, handler):
        """Test input processor add/remove functionality."""
        processor = MockInputProcessor()
        
        # Add processor
        handler.add_processor(processor)
        assert processor in handler.processors
        
        # Don't add duplicate
        handler.add_processor(processor)
        assert handler.processors.count(processor) == 1
        
        # Remove processor
        handler.remove_processor(processor)
        assert processor not in handler.processors
    
    def test_event_handler_management(self, handler):
        """Test event handler add/remove functionality."""
        handler_func = Mock()
        
        # Add event handler
        handler.add_event_handler(InputEventType.CHARACTER, handler_func)
        assert InputEventType.CHARACTER in handler.event_handlers
        assert handler_func in handler.event_handlers[InputEventType.CHARACTER]
        
        # Remove event handler
        handler.remove_event_handler(InputEventType.CHARACTER, handler_func)
        assert handler_func not in handler.event_handlers.get(InputEventType.CHARACTER, [])
    
    def test_byte_processing_printable_characters(self, handler):
        """Test processing of printable character bytes."""
        events = []
        
        def capture_event(event):
            events.append(event)
        
        # Mock event dispatch to capture events
        handler._dispatch_event = capture_event
        
        # Process printable character
        handler._process_byte(ord('a'), time.time())
        
        assert len(events) == 1
        assert events[0].event_type == InputEventType.CHARACTER
        assert events[0].data['character'] == 'a'
    
    def test_byte_processing_control_characters(self, handler):
        """Test processing of control character bytes."""
        events = []
        handler._dispatch_event = lambda e: events.append(e)
        
        test_cases = [
            (3, 'ctrl-c'),      # Ctrl+C
            (4, 'ctrl-d'),      # Ctrl+D
            (8, 'backspace'),   # Backspace
            (127, 'backspace'), # DEL (also backspace)
            (13, 'enter'),      # Enter/Return
            (10, 'ctrl-j'),     # Line feed
            (9, 'tab'),         # Tab
        ]
        
        for byte_val, expected_key in test_cases:
            events.clear()
            handler._process_byte(byte_val, time.time())
            
            assert len(events) == 1
            assert events[0].event_type == InputEventType.SPECIAL_KEY
            assert events[0].data['key'] == expected_key
    
    def test_escape_sequence_parsing(self, handler):
        """Test parsing of escape sequences."""
        events = []
        handler._dispatch_event = lambda e: events.append(e)
        
        # Test arrow key sequence: ESC [ A (Up arrow)
        timestamp = time.time()
        handler._process_byte(27, timestamp)  # ESC
        assert handler._parsing_sequence is True
        
        handler._process_byte(ord('['), timestamp)
        handler._process_byte(ord('A'), timestamp)
        
        assert len(events) == 1
        assert events[0].event_type == InputEventType.SPECIAL_KEY
        assert events[0].data['key'] == 'up'
        assert handler._parsing_sequence is False
        assert handler._escape_buffer == b''
    
    def test_escape_sequence_timeout_protection(self, handler):
        """Test protection against escape sequence buffer overflow."""
        handler._parsing_sequence = True
        handler._escape_buffer = b'x' * 20  # Exceed buffer limit
        
        # Process another byte - should reset buffer
        handler._process_byte(ord('a'), time.time())
        
        assert handler._parsing_sequence is False
        assert handler._escape_buffer == b''
    
    def test_mouse_event_parsing(self, handler):
        """Test mouse event sequence parsing."""
        # SGR mouse format: ESC [ < 0 ; 10 ; 20 M
        sequence = b'\x1b[<0;10;20M'
        timestamp = time.time()
        
        event = handler._parse_escape_sequence(sequence, timestamp)
        
        assert event is not None
        assert event.event_type == InputEventType.MOUSE
        assert event.data['button'] == 'left'
        assert event.data['x'] == 10
        assert event.data['y'] == 20
        assert event.data['pressed'] is True
    
    def test_sequence_completion_detection(self, handler):
        """Test escape sequence completion detection."""
        # CSI sequences
        assert handler._is_sequence_complete(b'\x1b[A') is True    # Arrow key
        assert handler._is_sequence_complete(b'\x1b[3~') is True   # Delete key
        assert handler._is_sequence_complete(b'\x1b[') is False    # Incomplete
        
        # Non-CSI sequences
        assert handler._is_sequence_complete(b'\x1bO') is True     # Complete
        assert handler._is_sequence_complete(b'\x1b') is False     # Incomplete
    
    @pytest.mark.asyncio
    async def test_event_dispatch_to_processors(self, handler):
        """Test event dispatch to processors."""
        processor = MockInputProcessor()
        handler.add_processor(processor)
        
        # Create mock event loop
        loop = asyncio.new_event_loop()
        handler._loop = loop
        
        event = InputEvent(
            event_type=InputEventType.CHARACTER,
            data={'character': 'a'},
            timestamp=time.time()
        )
        
        # Mock asyncio.run_coroutine_threadsafe
        with patch('asyncio.run_coroutine_threadsafe') as mock_run:
            handler._dispatch_event(event)
            
            # Should schedule processor call
            mock_run.assert_called_once()
    
    def test_event_dispatch_to_handlers(self, handler):
        """Test event dispatch to event handlers."""
        sync_handler = Mock()
        async_handler = AsyncMock()
        
        handler.add_event_handler(InputEventType.CHARACTER, sync_handler)
        handler.add_event_handler(InputEventType.CHARACTER, async_handler)
        
        # Create mock event loop
        loop = Mock()
        loop.is_closed.return_value = False
        handler._loop = loop
        
        event = InputEvent(
            event_type=InputEventType.CHARACTER,
            data={'character': 'a'},
            timestamp=time.time()
        )
        
        handler._dispatch_event(event)
        
        # Should schedule both handlers
        loop.call_soon_threadsafe.assert_called_once_with(sync_handler, event)
        
        # Should check if async handler is coroutine function
        with patch('asyncio.iscoroutinefunction', return_value=True), \
             patch('asyncio.run_coroutine_threadsafe') as mock_run:
            
            handler._dispatch_event(event)
            mock_run.assert_called()
    
    def test_input_buffer_interface(self, handler, output_handler):
        """Test input buffer interface methods."""
        # Initialize with echo processor
        echo_processor = CharacterEchoProcessor(output_handler)
        handler.echo_processor = echo_processor
        handler.add_processor(echo_processor)
        
        # Test buffer manipulation
        handler.set_input_buffer("test")
        assert handler.get_input_buffer() == "test"
        
        handler.clear_input_buffer()
        assert handler.get_input_buffer() == ""
        
    def test_input_buffer_interface_without_echo(self, handler):
        """Test input buffer interface without echo processor."""
        # Without echo processor
        assert handler.get_input_buffer() == ""
        
        # These should not raise exceptions
        handler.clear_input_buffer()
        handler.set_input_buffer("test")
    
    @pytest.mark.asyncio
    async def test_cleanup_functionality(self, handler):
        """Test complete cleanup functionality."""
        with patch.object(handler, 'stop') as mock_stop, \
             patch.object(handler.terminal_manager, 'cleanup') as mock_cleanup:
            
            # Add some processors and handlers
            processor = MockInputProcessor()
            event_handler = Mock()
            handler.add_processor(processor)
            handler.add_event_handler(InputEventType.CHARACTER, event_handler)
            
            await handler.cleanup()
            
            mock_stop.assert_called_once()
            mock_cleanup.assert_called_once()
            assert len(handler.processors) == 0
            assert len(handler.event_handlers) == 0
    
    def test_context_manager_interface(self, handler):
        """Test context manager interface."""
        with handler:
            # Should return self
            pass
        
        # Cleanup should be called automatically through __exit__
        # This is tested by ensuring no exceptions are raised
    
    @pytest.mark.asyncio
    async def test_context_manager_cleanup_with_running_loop(self, handler):
        """Test context manager cleanup with running event loop."""
        with patch('asyncio.get_running_loop') as mock_get_loop, \
             patch.object(handler, 'cleanup') as mock_cleanup:
            
            mock_loop = Mock()
            mock_get_loop.return_value = mock_loop
            
            # Simulate __exit__ call
            handler.__exit__(None, None, None)
            
            mock_loop.run_until_complete.assert_called_once()


class TestRawInputProcessing:
    """Test raw input processing scenarios."""
    
    @pytest.fixture
    def handler(self):
        handler = UnifiedInputHandler()
        handler.terminal_manager._tty_fd = 3
        return handler
    
    def test_process_raw_input_single_bytes(self, handler):
        """Test processing single byte inputs."""
        processed_events = []
        handler._dispatch_event = lambda e: processed_events.append(e)
        
        # Process multiple bytes
        raw_data = b'hello'
        handler._process_raw_input(raw_data)
        
        assert len(processed_events) == 5
        for i, event in enumerate(processed_events):
            assert event.event_type == InputEventType.CHARACTER
            assert event.data['character'] == raw_data[i:i+1].decode()
    
    def test_process_raw_input_with_escape_sequence(self, handler):
        """Test processing input with escape sequences."""
        processed_events = []
        handler._dispatch_event = lambda e: processed_events.append(e)
        
        # Send escape sequence for up arrow embedded in text
        raw_data = b'a\x1b[Ab'
        handler._process_raw_input(raw_data)
        
        assert len(processed_events) == 3
        assert processed_events[0].data['character'] == 'a'
        assert processed_events[1].data['key'] == 'up'
        assert processed_events[2].data['character'] == 'b'
    
    def test_input_thread_main_with_select(self, handler):
        """Test input thread main loop with select."""
        handler._stop_event = threading.Event()
        
        with patch('select.select') as mock_select, \
             patch('os.read') as mock_read, \
             patch.object(handler, '_process_raw_input') as mock_process:
            
            # First call returns data, second call times out (stop condition)
            mock_select.side_effect = [
                ([3], [], []),  # Data ready
                ([], [], []),   # Timeout
            ]
            mock_read.return_value = b'a'
            
            # Stop after first iteration
            def stop_after_first_call(*args):
                handler._stop_event.set()
            
            mock_process.side_effect = stop_after_first_call
            
            handler._input_thread_main()
            
            mock_read.assert_called_once_with(3, 64)
            mock_process.assert_called_once_with(b'a')
    
    def test_input_thread_main_with_io_error(self, handler):
        """Test input thread handling IO errors gracefully."""
        handler._stop_event = threading.Event()
        
        with patch('select.select', side_effect=IOError("Interrupted system call")), \
             patch('time.sleep') as mock_sleep:
            
            def stop_after_sleep(*args):
                handler._stop_event.set()
                
            mock_sleep.side_effect = stop_after_sleep
            
            # Should handle error gracefully
            handler._input_thread_main()
            
            mock_sleep.assert_called_once_with(0.1)


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_typing_workflow_immediate_echo(self):
        """Test complete typing workflow with immediate echo (GOLDEN TEST)."""
        output_calls = []
        
        def capture_output(text):
            output_calls.append(text)
        
        handler = UnifiedInputHandler()
        
        with patch.object(handler.terminal_manager, 'initialize', return_value=True), \
             patch.object(handler.terminal_manager, 'enter_raw_mode', return_value=True):
            
            # Initialize with output handler
            await handler.initialize(capture_output)
            
            # Simulate typing "hello"
            for char in "hello":
                event = InputEvent(
                    event_type=InputEventType.CHARACTER,
                    data={'character': char},
                    timestamp=time.time()
                )
                
                # Process through echo processor
                await handler.echo_processor.process_event(event)
            
            # CRITICAL ASSERTION: Each character should have been echoed immediately
            assert len(output_calls) == 5
            
            # Verify buffer builds correctly
            assert handler.get_input_buffer() == "hello"
            
            # Verify last output shows complete word with cursor
            last_output = output_calls[-1]
            assert "hello" in last_output
            assert "█" in last_output  # Cursor
    
    @pytest.mark.asyncio
    async def test_backspace_editing_workflow(self):
        """Test backspace editing workflow."""
        output_calls = []
        
        def capture_output(text):
            output_calls.append(text)
        
        handler = UnifiedInputHandler()
        
        with patch.object(handler.terminal_manager, 'initialize', return_value=True), \
             patch.object(handler.terminal_manager, 'enter_raw_mode', return_value=True):
            
            await handler.initialize(capture_output)
            
            # Type "hello"
            for char in "hello":
                event = InputEvent(
                    event_type=InputEventType.CHARACTER,
                    data={'character': char},
                    timestamp=time.time()
                )
                await handler.echo_processor.process_event(event)
            
            # Backspace twice
            for _ in range(2):
                event = InputEvent(
                    event_type=InputEventType.SPECIAL_KEY,
                    data={'key': 'backspace'},
                    timestamp=time.time()
                )
                await handler.echo_processor.process_event(event)
            
            # Should now have "hel" in buffer
            assert handler.get_input_buffer() == "hel"
            
            # Each operation should have been echoed
            assert len(output_calls) == 7  # 5 chars + 2 backspaces
    
    @pytest.mark.asyncio
    async def test_ctrl_c_handling(self):
        """Test Ctrl+C handling (GOLDEN TEST)."""
        ctrl_c_events = []
        
        def handle_ctrl_c(event):
            ctrl_c_events.append(event)
        
        handler = UnifiedInputHandler()
        handler.add_event_handler(InputEventType.SPECIAL_KEY, handle_ctrl_c)
        
        # Process Ctrl+C byte
        handler._process_byte(3, time.time())  # ASCII 3 = Ctrl+C
        
        # Should have dispatched Ctrl+C event
        assert len(ctrl_c_events) == 1
        assert ctrl_c_events[0].data['key'] == 'ctrl-c'
    
    @pytest.mark.asyncio
    async def test_concurrent_input_processing(self):
        """Test concurrent input processing safety."""
        handler = UnifiedInputHandler()
        processed_events = []
        
        def capture_event(event):
            processed_events.append(event)
        
        handler._dispatch_event = capture_event
        
        # Simulate concurrent byte processing from different threads
        def process_bytes(byte_range):
            for i in byte_range:
                handler._process_byte(ord('a') + i % 26, time.time())
        
        threads = [
            threading.Thread(target=process_bytes, args=(range(0, 5),)),
            threading.Thread(target=process_bytes, args=(range(5, 10),)),
            threading.Thread(target=process_bytes, args=(range(10, 15),)),
        ]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # Should have processed all events without corruption
        assert len(processed_events) == 15
        
        # All events should be character events
        for event in processed_events:
            assert event.event_type == InputEventType.CHARACTER
            assert 'character' in event.data


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""
    
    @pytest.mark.asyncio
    async def test_initialization_with_terminal_failure(self):
        """Test initialization when terminal operations fail."""
        handler = UnifiedInputHandler()
        
        with patch.object(handler.terminal_manager, 'initialize', return_value=False):
            result = await handler.initialize()
            assert result is False
    
    def test_malformed_escape_sequences(self, handler):
        """Test handling of malformed escape sequences."""
        events = []
        handler._dispatch_event = lambda e: events.append(e)
        
        # Malformed mouse sequence
        malformed_sequence = b'\x1b[<invalid'
        
        event = handler._parse_escape_sequence(malformed_sequence, time.time())
        
        # Should return None for malformed sequences
        assert event is None
    
    def test_invalid_mouse_sequence(self, handler):
        """Test handling of invalid mouse sequences."""
        # Invalid mouse sequence format
        invalid_seq = '<invalid;format;here'
        
        event = handler._parse_mouse_event(invalid_seq, time.time(), b'test')
        
        assert event is None
    
    @pytest.mark.asyncio
    async def test_processor_exception_handling(self, handler):
        """Test that processor exceptions don't crash handler."""
        class FaultyProcessor(InputProcessor):
            async def process_event(self, event):
                raise ValueError("Test error")
        
        faulty_processor = FaultyProcessor()
        handler.add_processor(faulty_processor)
        handler._loop = asyncio.new_event_loop()
        
        event = InputEvent(
            event_type=InputEventType.CHARACTER,
            data={'character': 'a'},
            timestamp=time.time()
        )
        
        # Should not raise exception
        with patch('asyncio.run_coroutine_threadsafe') as mock_run:
            handler._dispatch_event(event)
            mock_run.assert_called_once()
    
    def test_event_handler_exception_safety(self, handler):
        """Test that event handler exceptions don't crash dispatcher."""
        def faulty_handler(event):
            raise RuntimeError("Test error")
        
        handler.add_event_handler(InputEventType.CHARACTER, faulty_handler)
        
        event = InputEvent(
            event_type=InputEventType.CHARACTER,
            data={'character': 'a'},
            timestamp=time.time()
        )
        
        # Should not raise exception
        handler._dispatch_event(event)
    
    @pytest.mark.asyncio
    async def test_stop_without_start(self, handler):
        """Test stopping handler without starting."""
        # Should not raise exception
        await handler.stop()
        assert handler.running is False
    
    def test_byte_processing_invalid_utf8(self, handler):
        """Test processing of invalid UTF-8 sequences."""
        events = []
        handler._dispatch_event = lambda e: events.append(e)
        
        # Process invalid UTF-8 byte
        handler._process_byte(255, time.time())
        
        # Should not create any events for invalid bytes
        assert len(events) == 0
    
    @pytest.mark.asyncio
    async def test_cleanup_with_running_thread(self, handler):
        """Test cleanup with input thread still running."""
        # Create a mock thread that's still alive
        mock_thread = Mock()
        mock_thread.is_alive.return_value = True
        mock_thread.join = Mock()
        
        handler._input_thread = mock_thread
        handler.running = True
        
        with patch.object(handler.terminal_manager, 'cleanup'):
            await handler.cleanup()
            
            # Should attempt to join thread with timeout
            mock_thread.join.assert_called_with(timeout=1.0)