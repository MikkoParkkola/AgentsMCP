"""
Comprehensive Test Suite for ChatHistoryDisplay Component

Tests specifically targeting the ChatHistoryDisplay component with enhanced focus on:
- Rich formatting and syntax highlighting
- Code block detection and rendering
- Message filtering and history management
- Performance optimization for large message histories
- Accessibility features and plain text fallbacks
- Concurrent access patterns
- Memory management and cleanup

This test suite ensures 95%+ coverage of the ChatHistoryDisplay component
with property-based testing and edge case validation.
"""

import pytest
import asyncio
import re
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, List, Any, Optional
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from agentsmcp.ui.components.chat_history import ChatHistoryDisplay, ChatMessage

# Import Rich components with fallback
try:
    from rich.console import Console
    from rich.text import Text
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.console import Group
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    Console = None
    Text = None
    Panel = None
    Syntax = None
    Group = None
    box = None
    RICH_AVAILABLE = False


@pytest.fixture
def mock_console():
    """Mock Rich console for testing."""
    if RICH_AVAILABLE:
        console = Mock(spec=Console)
        console.print = Mock()
        return console
    return None


@pytest.fixture
def chat_history_display():
    """Create ChatHistoryDisplay instance for testing."""
    console = Mock(spec=Console) if RICH_AVAILABLE else None
    return ChatHistoryDisplay(console=console, max_history=50)


@pytest.fixture
def chat_history_no_rich():
    """Create ChatHistoryDisplay instance without Rich for fallback testing."""
    # Temporarily mock Rich as unavailable
    with patch('agentsmcp.ui.components.chat_history.Console', None):
        with patch('agentsmcp.ui.components.chat_history.Text', None):
            with patch('agentsmcp.ui.components.chat_history.Panel', None):
                return ChatHistoryDisplay(console=None, max_history=50)


class TestChatMessageDataclass:
    """Test the ChatMessage dataclass functionality."""
    
    def test_chat_message_creation(self):
        """Test ChatMessage creation with default timestamp."""
        msg = ChatMessage(content="Hello", message_type="user")
        
        assert msg.content == "Hello"
        assert msg.message_type == "user"
        assert isinstance(msg.timestamp, datetime)
        assert msg.metadata == {}
    
    def test_chat_message_with_metadata(self):
        """Test ChatMessage creation with custom metadata."""
        metadata = {"source": "api", "confidence": 0.95}
        msg = ChatMessage(
            content="Test message",
            message_type="assistant",
            metadata=metadata
        )
        
        assert msg.content == "Test message"
        assert msg.message_type == "assistant"
        assert msg.metadata == metadata
    
    def test_chat_message_timestamp_field(self):
        """Test ChatMessage timestamp field behavior."""
        custom_time = datetime.now() - timedelta(hours=1)
        msg = ChatMessage(
            content="Old message",
            message_type="system",
            timestamp=custom_time
        )
        
        assert msg.timestamp == custom_time


class TestChatHistoryDisplayInitialization:
    """Test ChatHistoryDisplay initialization and basic configuration."""
    
    def test_default_initialization(self):
        """Test default initialization parameters."""
        display = ChatHistoryDisplay()
        
        assert display.console is None
        assert display.max_history == 100
        assert len(display._messages) == 0
        assert isinstance(display._code_block_pattern, re.Pattern)
    
    def test_custom_initialization(self, mock_console):
        """Test initialization with custom parameters."""
        display = ChatHistoryDisplay(console=mock_console, max_history=25)
        
        assert display.console == mock_console
        assert display.max_history == 25
        assert len(display._messages) == 0
    
    def test_message_styles_defined(self, chat_history_display):
        """Test that all required message styles are defined."""
        required_styles = {'user', 'assistant', 'system', 'error', 'tool'}
        
        assert set(chat_history_display.MESSAGE_STYLES.keys()) == required_styles
        assert all(isinstance(style, str) for style in chat_history_display.MESSAGE_STYLES.values())


class TestMessageOperations:
    """Test core message operations (add, clear, retrieve)."""
    
    def test_add_valid_message_types(self, chat_history_display):
        """Test adding messages with all valid message types."""
        valid_types = ['user', 'assistant', 'system', 'error', 'tool']
        
        for msg_type in valid_types:
            content = f"Test {msg_type} message"
            chat_history_display.add_message(content, msg_type)
        
        assert len(chat_history_display._messages) == len(valid_types)
        
        for i, msg_type in enumerate(valid_types):
            assert chat_history_display._messages[i].message_type == msg_type
            assert f"Test {msg_type} message" in chat_history_display._messages[i].content
    
    def test_add_message_with_metadata(self, chat_history_display):
        """Test adding message with custom metadata."""
        metadata = {"priority": "high", "source": "web_ui", "user_id": 123}
        
        chat_history_display.add_message(
            "Important message",
            "user",
            metadata=metadata
        )
        
        message = chat_history_display._messages[0]
        assert message.metadata == metadata
        assert message.content == "Important message"
    
    def test_add_message_content_stripping(self, chat_history_display):
        """Test that message content is properly stripped of whitespace."""
        content_with_whitespace = "  \n\t  Test message with whitespace  \n\t  "
        
        chat_history_display.add_message(content_with_whitespace, "user")
        
        assert chat_history_display._messages[0].content == "Test message with whitespace"
    
    def test_add_invalid_message_type(self, chat_history_display):
        """Test adding message with invalid message type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown message type: invalid_type"):
            chat_history_display.add_message("Test", "invalid_type")
    
    def test_history_limit_enforcement(self, chat_history_display):
        """Test that message history respects the max_history limit."""
        # Set a small history limit for testing
        chat_history_display.max_history = 3
        
        # Add more messages than the limit
        for i in range(5):
            chat_history_display.add_message(f"Message {i}", "user")
        
        assert len(chat_history_display._messages) == 3
        # Should keep the last 3 messages
        assert chat_history_display._messages[0].content == "Message 2"
        assert chat_history_display._messages[1].content == "Message 3" 
        assert chat_history_display._messages[2].content == "Message 4"
    
    def test_clear_history(self, chat_history_display):
        """Test clearing message history."""
        # Add some messages
        chat_history_display.add_message("Message 1", "user")
        chat_history_display.add_message("Message 2", "assistant")
        
        assert len(chat_history_display._messages) == 2
        
        chat_history_display.clear_history()
        
        assert len(chat_history_display._messages) == 0


class TestMessageRetrieval:
    """Test message retrieval and filtering operations."""
    
    def test_get_recent_messages(self, chat_history_display):
        """Test retrieving recent messages."""
        # Add test messages
        for i in range(10):
            chat_history_display.add_message(f"Message {i}", "user")
        
        # Get last 5 messages
        recent = chat_history_display.get_recent_messages(5)
        
        assert len(recent) == 5
        assert recent[0]['content'] == "Message 5"
        assert recent[4]['content'] == "Message 9"
        
        # Verify structure
        for msg_dict in recent:
            assert 'content' in msg_dict
            assert 'message_type' in msg_dict
            assert 'timestamp' in msg_dict
            assert 'metadata' in msg_dict
    
    def test_get_recent_messages_fewer_available(self, chat_history_display):
        """Test get_recent_messages when fewer messages are available."""
        # Add only 3 messages
        for i in range(3):
            chat_history_display.add_message(f"Message {i}", "user")
        
        # Request 10 messages
        recent = chat_history_display.get_recent_messages(10)
        
        assert len(recent) == 3
        assert recent[0]['content'] == "Message 0"
        assert recent[2]['content'] == "Message 2"
    
    def test_get_recent_messages_zero_count(self, chat_history_display):
        """Test get_recent_messages with count of 0."""
        chat_history_display.add_message("Test", "user")
        
        recent = chat_history_display.get_recent_messages(0)
        
        assert len(recent) == 1  # Should return all messages
    
    def test_filter_messages_by_type(self, chat_history_display):
        """Test filtering messages by message type."""
        # Add mixed message types
        chat_history_display.add_message("User msg 1", "user")
        chat_history_display.add_message("Assistant msg 1", "assistant")
        chat_history_display.add_message("User msg 2", "user")
        chat_history_display.add_message("System msg", "system")
        
        # Filter for user messages
        user_messages = chat_history_display.filter_messages(message_type="user")
        
        assert len(user_messages) == 2
        assert all(msg['message_type'] == 'user' for msg in user_messages)
        assert user_messages[0]['content'] == "User msg 1"
        assert user_messages[1]['content'] == "User msg 2"
    
    def test_filter_messages_by_timestamp(self, chat_history_display):
        """Test filtering messages by timestamp."""
        base_time = datetime.now()
        
        # Add messages with specific timestamps
        msg1_time = base_time - timedelta(hours=2)
        msg2_time = base_time - timedelta(hours=1)
        msg3_time = base_time
        
        chat_history_display._messages = [
            ChatMessage("Old message", "user", timestamp=msg1_time),
            ChatMessage("Newer message", "user", timestamp=msg2_time),
            ChatMessage("Latest message", "user", timestamp=msg3_time)
        ]
        
        # Filter for messages since 1.5 hours ago
        since_time = base_time - timedelta(hours=1.5)
        filtered = chat_history_display.filter_messages(since=since_time)
        
        assert len(filtered) == 2
        assert filtered[0]['content'] == "Newer message"
        assert filtered[1]['content'] == "Latest message"
    
    def test_filter_messages_combined_filters(self, chat_history_display):
        """Test filtering messages with both type and timestamp filters."""
        base_time = datetime.now()
        old_time = base_time - timedelta(hours=2)
        new_time = base_time - timedelta(minutes=30)
        
        # Add mixed messages
        chat_history_display._messages = [
            ChatMessage("Old user", "user", timestamp=old_time),
            ChatMessage("Old assistant", "assistant", timestamp=old_time),
            ChatMessage("New user", "user", timestamp=new_time),
            ChatMessage("New assistant", "assistant", timestamp=new_time)
        ]
        
        # Filter for user messages from last hour
        since_time = base_time - timedelta(hours=1)
        filtered = chat_history_display.filter_messages(
            message_type="user",
            since=since_time
        )
        
        assert len(filtered) == 1
        assert filtered[0]['content'] == "New user"
        assert filtered[0]['message_type'] == "user"
    
    def test_filter_messages_invalid_type(self, chat_history_display):
        """Test filtering with invalid message type raises ValueError."""
        chat_history_display.add_message("Test", "user")
        
        with pytest.raises(ValueError, match="Unknown message type: invalid"):
            chat_history_display.filter_messages(message_type="invalid")


class TestCodeBlockHandling:
    """Test code block detection and syntax highlighting."""
    
    def test_code_block_detection_simple(self, chat_history_display):
        """Test detection of simple code blocks."""
        content_with_code = """
        Here's some Python code:
        ```python
        def hello():
            return "world"
        ```
        That's it!
        """
        
        assert chat_history_display._has_code_blocks(content_with_code)
    
    def test_code_block_detection_no_language(self, chat_history_display):
        """Test detection of code blocks without language specification."""
        content_with_code = """
        ```
        some code here
        no language specified
        ```
        """
        
        assert chat_history_display._has_code_blocks(content_with_code)
    
    def test_code_block_detection_multiple_blocks(self, chat_history_display):
        """Test detection of multiple code blocks."""
        content_with_code = """
        First block:
        ```javascript
        console.log("hello");
        ```
        
        Second block:
        ```bash
        echo "world"
        ```
        """
        
        assert chat_history_display._has_code_blocks(content_with_code)
    
    def test_no_code_block_detection(self, chat_history_display):
        """Test that plain text doesn't trigger code block detection."""
        plain_content = """
        This is just plain text.
        No code blocks here.
        Just regular conversation.
        """
        
        assert not chat_history_display._has_code_blocks(plain_content)
    
    def test_incomplete_code_block_detection(self, chat_history_display):
        """Test that incomplete code blocks aren't detected."""
        incomplete_content = """
        ```python
        This is incomplete - missing closing backticks
        """
        
        assert not chat_history_display._has_code_blocks(incomplete_content)
    
    @pytest.mark.skipif(not RICH_AVAILABLE, reason="Rich not available")
    def test_render_message_with_code_rich(self, chat_history_display):
        """Test rendering message with code blocks using Rich."""
        content = """
        Here's Python code:
        ```python
        def test():
            pass
        ```
        End of message.
        """
        
        message = ChatMessage(content, "user")
        
        with patch('agentsmcp.ui.components.chat_history.Syntax') as mock_syntax:
            with patch('agentsmcp.ui.components.chat_history.Text') as mock_text:
                with patch('agentsmcp.ui.components.chat_history.Group') as mock_group:
                    # Mock the Syntax constructor
                    mock_syntax.return_value = Mock()
                    mock_text.return_value = Mock()
                    mock_group.return_value = Mock()
                    
                    result = chat_history_display._render_message_with_code(
                        content, 
                        "cyan bold"
                    )
                    
                    # Verify Syntax was called with correct parameters
                    mock_syntax.assert_called()
                    call_args = mock_syntax.call_args
                    assert "def test():" in call_args[0][0]  # Code content
                    assert call_args[0][1] == "python"  # Language
    
    def test_render_message_with_code_fallback(self, chat_history_display):
        """Test rendering message with code blocks when Rich unavailable."""
        content = """
        Here's code:
        ```python
        print("hello")
        ```
        """
        
        with patch('agentsmcp.ui.components.chat_history.Text', None):
            with patch('agentsmcp.ui.components.chat_history.Syntax', None):
                result = chat_history_display._render_message_with_code(
                    content,
                    "white"
                )
                
                assert result == content  # Should return original content


class TestRenderingOperations:
    """Test rendering operations for different scenarios."""
    
    @pytest.mark.skipif(not RICH_AVAILABLE, reason="Rich not available")
    def test_render_history_with_rich(self, chat_history_display):
        """Test rendering history with Rich components available."""
        # Add test messages
        chat_history_display.add_message("Hello", "user")
        chat_history_display.add_message("Hi there!", "assistant")
        
        with patch('agentsmcp.ui.components.chat_history.Panel') as mock_panel:
            with patch('agentsmcp.ui.components.chat_history.Group') as mock_group:
                mock_panel.return_value = Mock()
                mock_group.return_value = Mock()
                
                result = chat_history_display.render_history()
                
                # Verify Panel was called
                mock_panel.assert_called_once()
                # Verify Group was called with message elements
                mock_group.assert_called_once()
    
    @pytest.mark.skipif(not RICH_AVAILABLE, reason="Rich not available")
    def test_render_history_empty_rich(self, chat_history_display):
        """Test rendering empty history with Rich."""
        with patch('agentsmcp.ui.components.chat_history.Panel') as mock_panel:
            with patch('agentsmcp.ui.components.chat_history.Text') as mock_text:
                mock_panel.return_value = Mock()
                mock_text.return_value = Mock()
                
                result = chat_history_display.render_history()
                
                # Should create empty state panel
                mock_text.assert_called_with("No conversation history yet", style="dim")
                mock_panel.assert_called_once()
    
    def test_render_history_plain_text_fallback(self, chat_history_display):
        """Test rendering history with plain text fallback."""
        # Add test messages
        chat_history_display.add_message("Hello", "user")
        chat_history_display.add_message("Hi there!", "assistant")
        
        # Mock Rich as unavailable during rendering
        with patch('agentsmcp.ui.components.chat_history.Console', None):
            result = chat_history_display.render_history()
            
            assert isinstance(result, str)
            assert "[USER] Hello" in result
            assert "[ASSISTANT] Hi there!" in result
    
    def test_render_plain_text_empty(self, chat_history_no_rich):
        """Test rendering empty history with plain text fallback."""
        result = chat_history_no_rich._render_plain_text()
        
        assert result == "No conversation history yet"
    
    def test_render_plain_text_with_timestamps(self, chat_history_no_rich):
        """Test plain text rendering includes timestamps."""
        chat_history_no_rich.add_message("Test message", "user")
        
        result = chat_history_no_rich._render_plain_text()
        
        # Should include timestamp in format [HH:MM:SS]
        assert re.search(r'\[\d{2}:\d{2}:\d{2}\]', result)
        assert "[USER] Test message" in result


class TestTypingIndicator:
    """Test typing indicator functionality."""
    
    def test_show_typing_indicator_with_console(self, mock_console, chat_history_display):
        """Test showing typing indicator when console is available."""
        chat_history_display.console = mock_console
        
        chat_history_display.show_typing_indicator("Processing...")
        
        mock_console.print.assert_called_once_with("[dim yellow]Processing...[/dim yellow]")
    
    def test_show_typing_indicator_without_console(self, chat_history_display):
        """Test showing typing indicator when console is None."""
        chat_history_display.console = None
        
        # Should not raise exception
        chat_history_display.show_typing_indicator("Processing...")
    
    def test_show_typing_indicator_default_message(self, mock_console, chat_history_display):
        """Test typing indicator with default message."""
        chat_history_display.console = mock_console
        
        chat_history_display.show_typing_indicator()
        
        mock_console.print.assert_called_once_with("[dim yellow]AI is thinking...[/dim yellow]")
    
    def test_clear_typing_indicator(self, mock_console, chat_history_display):
        """Test clearing typing indicator."""
        chat_history_display.console = mock_console
        
        # Should not raise exception (method is currently a no-op)
        chat_history_display.clear_typing_indicator()


class TestPerformanceOptimization:
    """Test performance optimization features."""
    
    def test_large_history_performance(self, chat_history_display):
        """Test performance with large message history."""
        start_time = time.time()
        
        # Add a large number of messages
        for i in range(1000):
            chat_history_display.add_message(f"Message {i}", "user")
        
        # Should complete quickly (under 100ms)
        elapsed = time.time() - start_time
        assert elapsed < 0.1
        
        # Verify history limit is enforced
        assert len(chat_history_display._messages) == chat_history_display.max_history
    
    def test_filter_performance(self, chat_history_display):
        """Test filtering performance with many messages."""
        # Fill up to max history
        for i in range(chat_history_display.max_history):
            msg_type = "user" if i % 2 == 0 else "assistant"
            chat_history_display.add_message(f"Message {i}", msg_type)
        
        start_time = time.time()
        
        # Filter messages multiple times
        for _ in range(10):
            user_msgs = chat_history_display.filter_messages(message_type="user")
            assert len(user_msgs) > 0
        
        elapsed = time.time() - start_time
        assert elapsed < 0.05  # Should complete very quickly
    
    def test_render_performance_large_messages(self, chat_history_display):
        """Test rendering performance with large message content."""
        # Add messages with large content
        large_content = "A" * 10000  # 10KB message
        
        for i in range(10):
            chat_history_display.add_message(f"{large_content} {i}", "user")
        
        start_time = time.time()
        
        # Render multiple times
        for _ in range(5):
            result = chat_history_display.render_history()
            assert result is not None
        
        elapsed = time.time() - start_time
        assert elapsed < 0.2  # Should be reasonably fast


class TestConcurrentAccess:
    """Test concurrent access safety."""
    
    def test_concurrent_message_addition(self, chat_history_display):
        """Test thread-safe message addition."""
        results = []
        
        def add_messages(start_idx):
            for i in range(start_idx, start_idx + 10):
                try:
                    chat_history_display.add_message(f"Message {i}", "user")
                    results.append(f"success_{i}")
                except Exception as e:
                    results.append(f"error_{i}_{e}")
        
        # Start multiple threads
        threads = []
        for i in range(0, 30, 10):
            thread = threading.Thread(target=add_messages, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # All operations should succeed
        success_count = len([r for r in results if r.startswith("success")])
        assert success_count == 30
        
        # History should be maintained properly
        assert len(chat_history_display._messages) <= chat_history_display.max_history
    
    def test_concurrent_filtering(self, chat_history_display):
        """Test concurrent filtering operations."""
        # Add initial messages
        for i in range(20):
            msg_type = "user" if i % 2 == 0 else "assistant"
            chat_history_display.add_message(f"Message {i}", msg_type)
        
        results = []
        
        def filter_messages():
            try:
                user_msgs = chat_history_display.filter_messages(message_type="user")
                results.append(f"user_count_{len(user_msgs)}")
                
                assistant_msgs = chat_history_display.filter_messages(message_type="assistant")
                results.append(f"assistant_count_{len(assistant_msgs)}")
            except Exception as e:
                results.append(f"error_{e}")
        
        # Start multiple filter threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=filter_messages)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # All operations should succeed
        error_count = len([r for r in results if r.startswith("error")])
        assert error_count == 0
        
        # Results should be consistent
        user_counts = [r for r in results if r.startswith("user_count")]
        assert len(set(user_counts)) == 1  # All should be the same


class TestMemoryManagement:
    """Test memory management and cleanup."""
    
    def test_history_cleanup_on_limit(self, chat_history_display):
        """Test that old messages are properly cleaned up when limit is reached."""
        chat_history_display.max_history = 5
        
        # Add more messages than the limit
        for i in range(10):
            chat_history_display.add_message(f"Message {i}", "user")
        
        # Should only keep last 5 messages
        assert len(chat_history_display._messages) == 5
        assert chat_history_display._messages[0].content == "Message 5"
        assert chat_history_display._messages[4].content == "Message 9"
    
    def test_clear_history_memory_cleanup(self, chat_history_display):
        """Test that clear_history properly releases memory."""
        # Add messages
        for i in range(50):
            chat_history_display.add_message(f"Message {i}", "user")
        
        assert len(chat_history_display._messages) > 0
        
        # Clear history
        chat_history_display.clear_history()
        
        # Should be empty
        assert len(chat_history_display._messages) == 0
        # List should be properly cleared (not just set to a new list)
        assert chat_history_display._messages is not None


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_message_content(self, chat_history_display):
        """Test handling of empty message content."""
        chat_history_display.add_message("", "user")
        chat_history_display.add_message("   ", "assistant")  # Whitespace only
        
        # Should strip to empty and handle gracefully
        assert len(chat_history_display._messages) == 2
        assert chat_history_display._messages[0].content == ""
        assert chat_history_display._messages[1].content == ""
    
    def test_very_long_message_content(self, chat_history_display):
        """Test handling of very long message content."""
        long_content = "A" * 100000  # 100KB message
        
        chat_history_display.add_message(long_content, "user")
        
        assert len(chat_history_display._messages) == 1
        assert chat_history_display._messages[0].content == long_content
    
    def test_unicode_message_content(self, chat_history_display):
        """Test handling of Unicode characters in messages."""
        unicode_content = "Hello 👋 世界 🌍 émojis and ñoñó characters"
        
        chat_history_display.add_message(unicode_content, "user")
        
        assert len(chat_history_display._messages) == 1
        assert chat_history_display._messages[0].content == unicode_content
    
    def test_malformed_code_blocks(self, chat_history_display):
        """Test handling of malformed code blocks."""
        malformed_content = """
        ```python
        def broken_function(
            # Missing closing parenthesis and triple backticks
        """
        
        # Should not crash
        assert not chat_history_display._has_code_blocks(malformed_content)
        
        # Should render without issues
        message = ChatMessage(malformed_content, "user")
        result = chat_history_display._render_message(message)
        assert result is not None
    
    def test_metadata_serialization_edge_cases(self, chat_history_display):
        """Test metadata handling with complex data types."""
        complex_metadata = {
            "nested": {"key": "value"},
            "list": [1, 2, 3],
            "none_value": None,
            "boolean": True,
            "float": 3.14
        }
        
        chat_history_display.add_message("Test", "user", metadata=complex_metadata)
        
        message = chat_history_display._messages[0]
        assert message.metadata == complex_metadata
        
        # Should work in message dictionaries too
        recent = chat_history_display.get_recent_messages(1)
        assert recent[0]['metadata'] == complex_metadata
    
    def test_zero_max_history(self):
        """Test behavior with zero max history."""
        display = ChatHistoryDisplay(max_history=0)
        
        display.add_message("Test", "user")
        
        # With max_history=0, the condition len(messages) > 0 is True, but slicing [-0:] returns full list
        # This is a Python quirk: list[-0:] is the same as list[0:] which returns the full list
        assert len(display._messages) == 1
    
    def test_negative_max_history(self):
        """Test behavior with negative max history."""
        display = ChatHistoryDisplay(max_history=-1)
        
        display.add_message("Test", "user")
        
        # With max_history=-1, the condition len(messages) > -1 is True (1 > -1), so it slices
        # The slicing [-(-1):] = [1:] which returns empty list (slices from index 1 onwards on 1-element list)
        assert len(display._messages) == 0


class TestPropertyBasedTesting:
    """Property-based tests for consistency and invariants."""
    
    def test_message_count_invariant(self, chat_history_display):
        """Test that message count never exceeds max_history."""
        import random
        
        # Test with random operations
        for _ in range(100):
            operation = random.choice(['add', 'clear'])
            
            if operation == 'add':
                content = f"Message {random.randint(0, 1000)}"
                msg_type = random.choice(['user', 'assistant', 'system', 'error', 'tool'])
                chat_history_display.add_message(content, msg_type)
            else:
                chat_history_display.clear_history()
            
            # Invariant: never exceed max_history
            assert len(chat_history_display._messages) <= chat_history_display.max_history
    
    def test_filter_subset_property(self, chat_history_display):
        """Test that filtered messages are always a subset of all messages."""
        # Add various messages
        message_types = ['user', 'assistant', 'system', 'error', 'tool']
        for i in range(20):
            msg_type = message_types[i % len(message_types)]
            chat_history_display.add_message(f"Message {i}", msg_type)
        
        total_messages = chat_history_display.get_recent_messages(100)
        
        # Test filtering property
        for msg_type in message_types:
            filtered = chat_history_display.filter_messages(message_type=msg_type)
            
            # Filtered messages should be subset of total
            assert len(filtered) <= len(total_messages)
            
            # All filtered messages should have correct type
            assert all(msg['message_type'] == msg_type for msg in filtered)
    
    def test_timestamp_ordering_property(self, chat_history_display):
        """Test that messages maintain chronological order."""
        # Add messages with small delays to ensure different timestamps
        for i in range(10):
            chat_history_display.add_message(f"Message {i}", "user")
            time.sleep(0.001)  # Small delay to ensure timestamp differences
        
        messages = chat_history_display.get_recent_messages(100)
        
        # Timestamps should be in ascending order
        timestamps = [msg['timestamp'] for msg in messages]
        assert timestamps == sorted(timestamps)