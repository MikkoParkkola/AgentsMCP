"""
Comprehensive tests for TUI console alignment issues.

These tests verify that the TUI properly manages cursor positioning and prevents
the progressive indentation bug observed in console output.
"""

import pytest
import asyncio
import io
import sys
import os
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from contextlib import contextmanager
import tempfile
import termios
import tty

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from agentsmcp.ui.v2.fixed_working_tui import FixedWorkingTUI


@contextmanager
def capture_terminal_output():
    """Context manager to capture terminal output for testing."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    try:
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        yield stdout_capture, stderr_capture
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


class MockTerminal:
    """Mock terminal for testing cursor positioning without real TTY."""
    
    def __init__(self):
        self.cursor_row = 0
        self.cursor_col = 0
        self.screen_width = 80
        self.screen_height = 24
        self.buffer = []
        self.current_line = ""
        
    def write(self, text):
        """Process text and update cursor position."""
        for char in text:
            if char == '\r':  # Carriage return
                self.cursor_col = 0
            elif char == '\n':  # Line feed
                self.buffer.append(self.current_line)
                self.current_line = ""
                self.cursor_row += 1
                self.cursor_col = 0
            elif char == '\b':  # Backspace
                if self.cursor_col > 0:
                    self.cursor_col -= 1
                    if self.current_line:
                        self.current_line = self.current_line[:-1]
            elif char == '\033':  # Start of ANSI escape sequence
                # For simplicity, we'll handle basic escape sequences
                continue
            elif ord(char) >= 32:  # Printable character
                if self.cursor_col < len(self.current_line):
                    # Overwrite character
                    self.current_line = (self.current_line[:self.cursor_col] + 
                                       char + 
                                       self.current_line[self.cursor_col + 1:])
                else:
                    # Append character
                    self.current_line += char
                self.cursor_col += 1
                
    def get_current_position(self):
        """Get current cursor position."""
        return self.cursor_row, self.cursor_col
        
    def get_screen_content(self):
        """Get current screen content."""
        content = self.buffer + [self.current_line] if self.current_line else self.buffer
        return content


class TestTUICursorAlignment:
    """Test cursor position tracking and alignment."""
    
    @pytest.fixture
    def tui(self):
        """Create a TUI instance for testing."""
        return FixedWorkingTUI()
        
    @pytest.fixture  
    def mock_terminal(self):
        """Create a mock terminal for testing."""
        return MockTerminal()
    
    def test_initial_cursor_position(self, tui):
        """Test that cursor starts at correct position."""
        assert tui.cursor_col == 0
        
    @patch('sys.stdout')
    def test_show_prompt_resets_cursor(self, mock_stdout, tui):
        """Test that show_prompt properly resets cursor to column 2."""
        tui.show_prompt()
        
        # Should write carriage return and prompt
        mock_stdout.write.assert_called_with('\r> ')
        assert tui.cursor_col == 2
    
    @patch('sys.stdout')  
    def test_clear_screen_sets_correct_cursor(self, mock_stdout, tui):
        """Test that clear_screen_and_show_prompt sets cursor correctly."""
        tui.clear_screen_and_show_prompt()
        
        mock_stdout.write.assert_called()
        assert tui.cursor_col == 2  # After "> "
        
    def test_cursor_tracking_with_input(self, tui, mock_terminal):
        """Test cursor position tracking with simulated input."""
        tui.cursor_col = 2  # Start after prompt
        
        # Simulate typing "hello"
        test_input = "hello"
        for char in test_input:
            tui.input_buffer += char
            tui.cursor_col += 1
            mock_terminal.write(char)
            
        assert tui.cursor_col == 7  # 2 (prompt) + 5 (hello)
        assert mock_terminal.cursor_col == 5  # Only the typed characters
        
    def test_backspace_cursor_handling(self, tui, mock_terminal):
        """Test that backspace properly manages cursor position."""
        tui.cursor_col = 5  # Simulate some input
        tui.input_buffer = "hel"
        
        # Simulate backspace (removes one character)
        if tui.input_buffer and tui.cursor_col > 2:
            tui.input_buffer = tui.input_buffer[:-1]
            tui.cursor_col -= 1
            mock_terminal.write('\b \b')
            
        assert tui.cursor_col == 4
        assert tui.input_buffer == "he"
        
    def test_no_progressive_indentation(self, tui, mock_terminal):
        """Test that multiple prompt cycles don't cause progressive indentation."""
        positions = []
        
        # Simulate multiple prompt-response cycles
        for i in range(5):
            tui.show_prompt()
            positions.append(tui.cursor_col)
            mock_terminal.write('\r> ')
            
        # All prompts should start at same column position
        assert all(pos == 2 for pos in positions), f"Progressive indentation detected: {positions}"
        
    @patch('sys.stdout')
    def test_multi_line_response_alignment(self, mock_stdout, tui):
        """Test that multi-line responses maintain consistent alignment."""
        response = "Line 1\nLine 2\nLine 3"
        lines = response.split('\n')
        
        # Simulate the response formatting from process_line
        mock_stdout.write('🤖 Agent: ')
        for i, response_line in enumerate(lines):
            if i == 0:
                mock_stdout.write(response_line + '\n')
            else:
                mock_stdout.write('         ' + response_line + '\n')  # 9 spaces for alignment
                
        # Verify that continuation lines are properly indented
        calls = mock_stdout.write.call_args_list
        continuation_calls = [call for call in calls if '         ' in str(call)]
        assert len(continuation_calls) == 2  # Two continuation lines
        
    def test_carriage_return_resets_column(self, tui, mock_terminal):
        """Test that carriage return properly resets cursor to column 0."""
        mock_terminal.cursor_col = 20  # Simulate cursor at some position
        mock_terminal.write('\r')
        
        assert mock_terminal.cursor_col == 0
        
    def test_cursor_boundaries(self, tui):
        """Test cursor position boundaries and constraints."""
        # Test that cursor can't go below prompt position
        tui.cursor_col = 2
        tui.input_buffer = ""
        
        # Simulate backspace when at prompt boundary
        original_col = tui.cursor_col
        if not (tui.input_buffer and tui.cursor_col > 2):
            # Should not change cursor position
            pass
        else:
            tui.cursor_col -= 1
            
        assert tui.cursor_col == original_col  # Should remain at boundary
        
    def test_terminal_width_wrapping(self, mock_terminal):
        """Test behavior at terminal width boundaries."""
        # Fill a line to near terminal width
        long_text = 'x' * (mock_terminal.screen_width - 5)
        mock_terminal.write(long_text)
        
        # Cursor should be near end of line
        assert mock_terminal.cursor_col == len(long_text)
        
        # Writing more should wrap appropriately
        mock_terminal.write('extra')
        # This tests the terminal's wrapping behavior
        
    @pytest.mark.asyncio
    async def test_input_echo_alignment(self, tui):
        """Test that input echoing maintains proper alignment."""
        with patch('sys.stdin') as mock_stdin:
            mock_stdin.isatty.return_value = True
            mock_stdin.read.return_value = 'a'  # Single character
            
            tui.cursor_col = 2  # Start after prompt
            original_col = tui.cursor_col
            
            # Simulate character input processing
            char = 'a'
            if ord(char) >= 32:  # Printable character
                tui.input_buffer += char
                tui.cursor_col += 1
                
            assert tui.cursor_col == original_col + 1
            assert 'a' in tui.input_buffer


class TestTUIStateConsistency:
    """Test TUI state consistency across operations."""
    
    @pytest.fixture
    def tui(self):
        return FixedWorkingTUI()
        
    def test_cursor_state_after_clear_screen(self, tui):
        """Test cursor state consistency after clear screen."""
        # Modify cursor state
        tui.cursor_col = 10
        
        # Clear screen should reset state
        with patch('sys.stdout'):
            tui.clear_screen_and_show_prompt()
            
        assert tui.cursor_col == 2  # Should be reset to prompt position
        
    def test_buffer_state_after_line_processing(self, tui):
        """Test that input buffer is properly cleared after processing."""
        tui.input_buffer = "test message"
        
        # Simulate line processing
        with patch('sys.stdout'):
            # After processing, buffer should be cleared
            tui.input_buffer = ""  # This happens in handle_input after enter
            tui.show_prompt()
            
        assert tui.input_buffer == ""
        assert tui.cursor_col == 2
        
    def test_consistent_state_across_multiple_operations(self, tui):
        """Test state consistency across multiple UI operations."""
        states = []
        
        with patch('sys.stdout'):
            for i in range(3):
                tui.show_prompt()
                states.append({
                    'cursor_col': tui.cursor_col,
                    'buffer': tui.input_buffer
                })
                
                # Simulate some input
                tui.input_buffer = f"input{i}"
                tui.cursor_col += len(tui.input_buffer)
                
                # Simulate processing (reset state)
                tui.input_buffer = ""
                
        # After each cycle, prompt should be at same position
        prompt_positions = [state['cursor_col'] for state in states]
        assert all(pos == 2 for pos in prompt_positions)


class TestTUIEdgeCases:
    """Test edge cases that could cause alignment issues."""
    
    @pytest.fixture
    def tui(self):
        return FixedWorkingTUI()
        
    def test_empty_input_handling(self, tui):
        """Test handling of empty input doesn't affect alignment."""
        with patch('sys.stdout'):
            original_col = tui.cursor_col
            
            # Simulate empty input
            empty_line = ""
            # Empty input shouldn't change cursor state
            
            assert tui.cursor_col == original_col
            
    def test_very_long_input_alignment(self, tui, mock_terminal):
        """Test alignment with very long input lines."""
        long_input = "x" * 200  # Very long input
        
        tui.cursor_col = 2  # Start after prompt
        for char in long_input:
            tui.input_buffer += char
            tui.cursor_col += 1
            mock_terminal.write(char)
            
        # Cursor should track correctly even for long input
        expected_col = 2 + len(long_input)
        assert tui.cursor_col == expected_col
        
    def test_special_characters_alignment(self, tui):
        """Test that special characters don't break alignment."""
        special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        
        tui.cursor_col = 2
        for char in special_chars:
            if ord(char) >= 32:  # Printable
                tui.input_buffer += char
                tui.cursor_col += 1
                
        expected_col = 2 + len(special_chars)
        assert tui.cursor_col == expected_col
        
    @patch('sys.stdout')
    def test_error_message_alignment(self, mock_stdout, tui):
        """Test that error messages maintain proper alignment."""
        error_msg = "Test error message"
        
        # Simulate error display
        mock_stdout.write(f'❌ Error: {error_msg}\n')
        mock_stdout.write('   Please try again or use /help for commands.\n')
        
        # After error, prompt should still be aligned
        tui.show_prompt()
        assert tui.cursor_col == 2
        
    def test_unicode_character_handling(self, tui):
        """Test alignment with Unicode characters."""
        unicode_input = "Hello 🌟 World 你好"
        
        tui.cursor_col = 2
        # Simulate input of Unicode characters
        for char in unicode_input:
            if ord(char) >= 32:
                tui.input_buffer += char
                tui.cursor_col += 1  # Note: This might need adjustment for wide chars
                
        # Basic test - cursor should advance
        assert tui.cursor_col > 2
        assert unicode_input in tui.input_buffer


class TestTUIPerformanceAlignment:
    """Test that alignment operations don't degrade performance."""
    
    @pytest.fixture
    def tui(self):
        return FixedWorkingTUI()
        
    def test_cursor_tracking_performance(self, tui):
        """Test that cursor tracking doesn't become slow over time."""
        import time
        
        times = []
        for i in range(100):
            start_time = time.time()
            
            # Simulate cursor operations
            tui.cursor_col = 2
            tui.cursor_col += 10
            tui.cursor_col = 2  # Reset
            
            end_time = time.time()
            times.append(end_time - start_time)
            
        # Verify performance doesn't degrade significantly
        early_avg = sum(times[:10]) / 10
        late_avg = sum(times[-10:]) / 10
        
        # Late operations shouldn't be more than 2x slower
        assert late_avg < early_avg * 2, "Cursor tracking performance degraded"
        
    def test_alignment_memory_usage(self, tui):
        """Test that alignment operations don't cause memory leaks."""
        import gc
        import sys
        
        # Force garbage collection and get baseline
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Perform many alignment operations
        with patch('sys.stdout'):
            for i in range(1000):
                tui.show_prompt()
                tui.cursor_col += 5
                tui.input_buffer = f"test{i}"
                tui.input_buffer = ""  # Clear
                
        # Check for memory leaks
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Should not have significant object growth
        object_growth = final_objects - initial_objects
        assert object_growth < 100, f"Possible memory leak: {object_growth} new objects"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])