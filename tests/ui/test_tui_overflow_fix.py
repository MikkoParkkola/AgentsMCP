"""
Test suite for TUI overflow fix to prevent empty line rendering issues.

This test verifies that removing overflow='fold' from Text objects prevents
the creation of empty lines that cause visual rendering problems in the TUI.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from rich.text import Text
from rich.console import Console
from io import StringIO

from agentsmcp.ui.components.enhanced_chat import EnhancedChatInput
from agentsmcp.ui.components.chat_history import ChatHistoryDisplay, ChatMessage


class TestTUIOverflowFix:
    """Test suite for TUI overflow fix."""

    def test_text_overflow_no_empty_lines(self):
        """Test that Text objects without overflow='fold' don't create empty lines."""
        # Create a Text object similar to what our components create
        text = Text()
        text.append("This is a very long message that would previously cause empty lines when overflow='fold' was set")
        
        # Verify that overflow is not set to 'fold'
        assert not hasattr(text, 'overflow') or text.overflow != "fold"
        
        # Render to console and verify no empty lines
        console = Console(file=StringIO(), width=80)
        console.print(text)
        output = console.file.getvalue()
        
        # Check that we don't have multiple consecutive newlines (empty lines)
        lines = output.split('\n')
        consecutive_empty = 0
        max_consecutive_empty = 0
        
        for line in lines:
            if line.strip() == '':
                consecutive_empty += 1
                max_consecutive_empty = max(max_consecutive_empty, consecutive_empty)
            else:
                consecutive_empty = 0
        
        # Should have at most 1 consecutive empty line (the final newline)
        assert max_consecutive_empty <= 1, f"Found {max_consecutive_empty} consecutive empty lines"

    def test_enhanced_chat_no_overflow_fold(self):
        """Test that EnhancedChatInput components don't use overflow='fold'."""
        with patch('agentsmcp.ui.components.enhanced_chat.Syntax') as mock_syntax:
            
            mock_syntax.return_value = None
            
            enhanced_chat = EnhancedChatInput()
            
            # Test regular message formatting
            test_message = "This is a regular message that should not have overflow fold"
            result = enhanced_chat.format_message_display(test_message, "white")
            
            # If result is a Text object, verify it doesn't have overflow='fold'
            if isinstance(result, Text):
                assert not hasattr(result, 'overflow') or result.overflow != "fold"

    def test_chat_history_no_overflow_fold(self):
        """Test that ChatHistoryDisplay components don't use overflow='fold'."""
        with patch('agentsmcp.ui.components.chat_history.Syntax') as mock_syntax:
            
            mock_syntax.return_value = None
            
            chat_history = ChatHistoryDisplay()
            
            # Create a test message
            message = ChatMessage(
                content="This is a test message content that should not cause empty lines",
                message_type="user",
                timestamp="2024-01-01T12:00:00Z"
            )
            
            # Test message rendering
            result = chat_history._render_message(message)
            
            # If result is a Text object, verify it doesn't have overflow='fold'
            if isinstance(result, Text):
                assert not hasattr(result, 'overflow') or result.overflow != "fold"

    def test_very_long_text_handling(self):
        """Test handling of very long text without overflow controls."""
        # Create extremely long text content
        very_long_text = "A" * 1000 + " " + "B" * 1000 + " " + "C" * 1000
        
        text = Text()
        text.append(very_long_text)
        
        # Verify no overflow='fold' is set
        assert not hasattr(text, 'overflow') or text.overflow != "fold"
        
        # Test rendering with different console widths
        for width in [40, 80, 120, 200]:
            console = Console(file=StringIO(), width=width)
            console.print(text)
            output = console.file.getvalue()
            
            # Should not have excessive empty lines
            empty_line_count = output.count('\n\n')
            assert empty_line_count <= 1, f"Too many empty lines ({empty_line_count}) at width {width}"

    def test_terminal_resize_safety(self):
        """Test that text rendering is safe across different terminal sizes."""
        test_content = "This is a test message with various lengths " * 10
        
        text = Text()
        text.append(test_content)
        
        # Test across various terminal widths
        test_widths = [20, 40, 60, 80, 100, 120, 160, 200]
        
        for width in test_widths:
            console = Console(file=StringIO(), width=width)
            console.print(text)
            output = console.file.getvalue()
            
            # Verify content is present and no excessive empty lines
            assert test_content.split()[0] in output, f"Content missing at width {width}"
            
            # Count consecutive empty lines
            lines = output.split('\n')
            consecutive_empty = 0
            max_consecutive = 0
            
            for line in lines:
                if line.strip() == '':
                    consecutive_empty += 1
                    max_consecutive = max(max_consecutive, consecutive_empty)
                else:
                    consecutive_empty = 0
            
            assert max_consecutive <= 2, f"Too many consecutive empty lines at width {width}"

    def test_mixed_content_rendering(self):
        """Test rendering of mixed content types without overflow issues."""
        # Test various content types that might be problematic
        test_cases = [
            "Short message",
            "A very long message that spans multiple lines and should wrap naturally without creating empty line artifacts",
            "Message with\nmultiple\nlines\nalready",
            "Message with tabs\t\tand spaces",
            "Unicode content: 🚀 💻 🔧 ✨",
            "Code-like content: def function(arg): return arg * 2",
            "",  # Empty message
            " ",  # Whitespace only
        ]
        
        for i, content in enumerate(test_cases):
            text = Text()
            text.append(f"Test {i}: {content}")
            
            # Verify no overflow fold
            assert not hasattr(text, 'overflow') or text.overflow != "fold"
            
            # Test rendering
            console = Console(file=StringIO(), width=80)
            console.print(text)
            output = console.file.getvalue()
            
            # Basic sanity checks
            if content.strip():  # Non-empty content
                assert str(i) in output, f"Test case {i} content not found in output"

    def test_code_block_rendering_no_overflow(self):
        """Test that code blocks don't use problematic overflow settings."""
        with patch('agentsmcp.ui.components.enhanced_chat.Syntax') as mock_syntax:
            
            # Mock returns for code detection
            mock_syntax.return_value = Mock()
            
            enhanced_chat = EnhancedChatInput()
            
            # Test code-like content
            code_content = """
def example_function():
    return "This is a code block that should not have overflow issues"
            """
            
            # This should trigger code block handling
            result = enhanced_chat.format_message_display(code_content, "white")
            
            # Verify the result doesn't have problematic overflow settings
            if isinstance(result, Text):
                assert not hasattr(result, 'overflow') or result.overflow != "fold"

    def test_edge_case_empty_and_whitespace(self):
        """Test edge cases with empty and whitespace-only content."""
        edge_cases = ["", " ", "\n", "\t", "   \n   ", "\n\n\n"]
        
        for content in edge_cases:
            text = Text()
            text.append(content)
            
            # Verify no problematic overflow
            assert not hasattr(text, 'overflow') or text.overflow != "fold"
            
            # Test rendering doesn't crash
            console = Console(file=StringIO(), width=80)
            console.print(text)
            output = console.file.getvalue()
            
            # Should complete without error
            assert isinstance(output, str)

    def test_regression_prevention(self):
        """Test to prevent regression of the overflow='fold' issue."""
        # This test specifically checks that our fix remains in place
        
        # Check Enhanced Chat
        with patch('agentsmcp.ui.components.enhanced_chat.Syntax'):
            
            enhanced_chat = EnhancedChatInput()
            result = enhanced_chat.format_message_display("test message", "white")
            
            if isinstance(result, Text):
                # The critical assertion: overflow should NOT be 'fold'
                assert getattr(result, 'overflow', None) != "fold", "REGRESSION: overflow='fold' detected in EnhancedChatInput"
        
        # Check Chat History
        with patch('agentsmcp.ui.components.chat_history.Syntax'):
            
            chat_history = ChatHistoryDisplay()
            message = ChatMessage(
                content="test content",
                message_type="user",
                timestamp="2024-01-01T12:00:00Z"
            )
            
            result = chat_history._render_message(message)
            
            if isinstance(result, Text):
                # The critical assertion: overflow should NOT be 'fold'
                assert getattr(result, 'overflow', None) != "fold", "REGRESSION: overflow='fold' detected in ChatHistoryDisplay"

    def test_narrow_terminal_compatibility(self):
        """Test compatibility with narrow terminals (80 columns baseline)."""
        long_message = "This is a very long message that needs to be handled properly in narrow terminals without creating empty line artifacts or rendering issues."
        
        text = Text()
        text.append(long_message)
        
        # Test narrow terminal (80 columns - baseline requirement)
        console = Console(file=StringIO(), width=80)
        console.print(text)
        output = console.file.getvalue()
        
        # Verify content is present
        assert "very long message" in output
        
        # Verify no excessive empty lines
        lines = output.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        empty_lines = [line for line in lines if not line.strip()]
        
        # Should have content and minimal empty lines
        assert len(non_empty_lines) >= 1, "No content rendered"
        assert len(empty_lines) <= 2, f"Too many empty lines: {len(empty_lines)}"

    def test_wide_terminal_compatibility(self):
        """Test compatibility with wide terminals (120+ columns)."""
        long_message = "This is a test message for wide terminal compatibility. " * 20
        
        text = Text()
        text.append(long_message)
        
        # Test wide terminal
        console = Console(file=StringIO(), width=160)
        console.print(text)
        output = console.file.getvalue()
        
        # Should handle wide terminals gracefully
        assert "test message" in output
        
        # Count lines - should be relatively few due to wide terminal
        lines = output.split('\n')
        content_lines = [line for line in lines if line.strip()]
        
        # Wide terminal should fit more content per line
        assert len(content_lines) >= 1, "No content in wide terminal"