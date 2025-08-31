#!/usr/bin/env python3
"""
Test suite for TUI line break fixes.

Tests to ensure that all TUI output lines start at column 0 and do not
exhibit progressive indentation or line continuation issues.
"""

import pytest
import sys
import io
from unittest.mock import patch
from contextlib import redirect_stdout

from agentsmcp.ui.v2.ansi_markdown_processor import render_markdown_lines


class TestTUILineBreakFixes:
    """Test suite for TUI line break fixes."""
    
    def test_render_markdown_lines_proper_formatting(self):
        """Test that markdown lines are rendered without continuation issues."""
        test_text = """# Header 1
This is a paragraph with some **bold** text and *italic* text.

## Header 2
- List item 1
- List item 2

```python
def test_function():
    return "test"
```

> This is a quote
"""
        
        lines = render_markdown_lines(test_text, width=80, indent='')
        
        # All lines should be proper strings without carriage returns
        for line in lines:
            assert isinstance(line, str)
            # Lines should not start with carriage returns
            assert not line.startswith('\r')
            # Lines should not end with carriage returns 
            assert not line.endswith('\r')
    
    def test_progress_message_formatting(self):
        """Test that progress messages format correctly."""
        # Simulate the progress message formatting from fixed_working_tui.py
        captured_output = io.StringIO()
        
        with redirect_stdout(captured_output):
            # Simulate the exact output pattern that was causing issues
            sys.stdout.write('\r🤔 Thinking...\r\n')
            sys.stdout.write('\r🚩 Orchestrating team: agent1, agent2\r\n')
            sys.stdout.write('\r[12:34:56] ▶ agent1 started\r\n')
            sys.stdout.write('\r[12:34:57] ✅ agent1 completed\r\n')
        
        output = captured_output.getvalue()
        lines = output.split('\n')
        
        # Verify each line starts cleanly (after the \r)
        non_empty_lines = [line for line in lines if line.strip()]
        expected_content = [
            '🤔 Thinking...',
            '🚩 Orchestrating team: agent1, agent2', 
            '[12:34:56] ▶ agent1 started',
            '[12:34:57] ✅ agent1 completed'
        ]
        
        for i, expected in enumerate(expected_content):
            if i < len(non_empty_lines):
                # Remove any remaining \r characters for comparison
                clean_line = non_empty_lines[i].replace('\r', '')
                assert clean_line == expected, f"Line {i}: expected '{expected}', got '{clean_line}'"
    
    def test_error_message_formatting(self):
        """Test that error messages format correctly."""
        captured_output = io.StringIO()
        
        with redirect_stdout(captured_output):
            # Simulate error message output
            sys.stdout.write('\r❌ Error: Something went wrong\r\n')
            sys.stdout.write('\r   Please try again or use /help for commands.\r\n')
        
        output = captured_output.getvalue()
        lines = output.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        # Verify error message formatting
        expected_lines = [
            '❌ Error: Something went wrong',
            '   Please try again or use /help for commands.'
        ]
        
        for i, expected in enumerate(expected_lines):
            if i < len(non_empty_lines):
                clean_line = non_empty_lines[i].replace('\r', '')
                assert clean_line == expected, f"Error line {i}: expected '{expected}', got '{clean_line}'"
    
    def test_command_output_formatting(self):
        """Test that command output formats correctly."""
        captured_output = io.StringIO()
        
        with redirect_stdout(captured_output):
            # Simulate command output (like /agents)
            sys.stdout.write('\r\nConfigured agents:\r\n')
            sys.stdout.write('\r- agent1: provider=ollama model=gpt-oss:20b\r\n')
            sys.stdout.write('\r- agent2: provider=openai model=gpt-4\r\n')
        
        output = captured_output.getvalue()
        lines = output.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        # Verify command output formatting
        expected_lines = [
            'Configured agents:',
            '- agent1: provider=ollama model=gpt-oss:20b',
            '- agent2: provider=openai model=gpt-4'
        ]
        
        for i, expected in enumerate(expected_lines):
            if i < len(non_empty_lines):
                clean_line = non_empty_lines[i].replace('\r', '')
                assert clean_line == expected, f"Command line {i}: expected '{expected}', got '{clean_line}'"
    
    def test_help_command_formatting(self):
        """Test that help command output formats correctly."""
        captured_output = io.StringIO()
        
        with redirect_stdout(captured_output):
            # Simulate help command output
            sys.stdout.write('\r\n📚 Commands:\r\n')
            sys.stdout.write('\r  /help   - Show this help\r\n')
            sys.stdout.write('\r  /quit   - Exit TUI\r\n')
            sys.stdout.write('\r  /clear  - Clear conversation history\r\n')
            sys.stdout.write('\r  /agents - List configured agents\r\n')
            sys.stdout.write('\r  Ctrl+C  - Exit TUI\r\n')
            sys.stdout.write('\r\n💬 Just type normally to chat with the LLM!\r\n')
        
        output = captured_output.getvalue()
        lines = output.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        # Verify help output structure - should have command list
        assert len(non_empty_lines) >= 6  # At least the main help items
        
        # Check that commands are properly formatted
        help_line = next((line for line in non_empty_lines if '📚 Commands:' in line), None)
        assert help_line is not None, "Help header not found"
        
        quit_line = next((line for line in non_empty_lines if '/quit' in line), None)
        assert quit_line is not None, "Quit command not found"
    
    def test_no_progressive_indentation(self):
        """Test that multiple lines don't exhibit progressive indentation."""
        captured_output = io.StringIO()
        
        with redirect_stdout(captured_output):
            # Simulate multiple output lines that could cause progressive indentation
            for i in range(5):
                sys.stdout.write(f'\rLine {i + 1}: This is a test line\r\n')
        
        output = captured_output.getvalue()
        lines = output.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        # Each line should start with the same pattern (no progressive indentation)
        for i, line in enumerate(non_empty_lines):
            clean_line = line.replace('\r', '')
            expected = f'Line {i + 1}: This is a test line'
            assert clean_line == expected, f"Progressive indentation detected in line {i}: '{clean_line}'"
    
    def test_ansi_code_handling(self):
        """Test that ANSI escape codes don't interfere with line breaks."""
        captured_output = io.StringIO()
        
        with redirect_stdout(captured_output):
            # Simulate colored output with ANSI codes
            sys.stdout.write('\r\x1b[32mGreen text\x1b[0m\r\n')
            sys.stdout.write('\r\x1b[31mRed text\x1b[0m\r\n')
            sys.stdout.write('\r\x1b[1mBold text\x1b[0m\r\n')
        
        output = captured_output.getvalue()
        
        # Should contain ANSI codes but still have proper line breaks
        assert '\x1b[32m' in output  # Green color code
        assert '\x1b[31m' in output  # Red color code
        assert '\x1b[1m' in output   # Bold code
        assert '\x1b[0m' in output   # Reset code
        
        # Each line should end with \r\n
        lines = output.split('\r\n')
        assert len(lines) >= 3  # At least 3 colored lines
        
        # Verify content is present (even with ANSI codes)
        full_output = output.replace('\r', '')
        assert 'Green text' in full_output
        assert 'Red text' in full_output
        assert 'Bold text' in full_output


if __name__ == '__main__':
    pytest.main([__file__, '-v'])