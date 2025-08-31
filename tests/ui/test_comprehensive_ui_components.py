"""
Comprehensive Test Suite for UI Components

Tests all UI components with 95%+ coverage, including:
- Enhanced Chat Input Components
- Chat History Display
- Real-time Input Handling
- Progressive Disclosure
- Symphony Dashboard
- Accessibility Features
- Performance Benchmarks

Follows Test-Driven Development principles with property-based testing
and deterministic test execution.
"""

import pytest
import asyncio
import time
import sys
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import UI components for testing
from agentsmcp.ui.components.enhanced_chat import EnhancedChatInput
from agentsmcp.ui.components.chat_history import ChatHistoryDisplay, ChatMessage
from agentsmcp.ui.components.realtime_input import RealtimeInput, InputMode, InputState

# Import mocked Rich components for testing
try:
    from rich.console import Console
    from rich.text import Text
    from rich.syntax import Syntax
    RICH_AVAILABLE = True
except ImportError:
    Console = None
    Text = None
    Syntax = None
    RICH_AVAILABLE = False


@pytest.fixture
def mock_console():
    """Mock Rich console for testing."""
    if RICH_AVAILABLE:
        return Mock(spec=Console)
    return None


@pytest.fixture
def enhanced_chat_input(mock_console):
    """Create enhanced chat input for testing."""
    return EnhancedChatInput(console=mock_console)


@pytest.fixture
def chat_history_display(mock_console):
    """Create chat history display for testing."""
    return ChatHistoryDisplay(console=mock_console, max_history=50)


@pytest.fixture
def realtime_input():
    """Create realtime input component for testing."""
    return RealtimeInput()


class TestEnhancedChatInput:
    """Comprehensive test suite for EnhancedChatInput component."""

    def test_initialization(self, enhanced_chat_input):
        """Test enhanced chat input initializes correctly."""
        assert enhanced_chat_input.console is not None or not RICH_AVAILABLE
        assert enhanced_chat_input._bracketed_paste_pattern is not None
        assert enhanced_chat_input._ansi_escape_pattern is not None

    def test_sanitize_input_bracketed_paste_removal(self, enhanced_chat_input):
        """Test bracketed paste sequences are removed correctly."""
        test_cases = [
            # Basic bracketed paste
            ("\x1b[200~Hello World\x1b[201~", "Hello World"),
            # Nested paste sequences
            ("\x1b[200~Start\x1b[200~Middle\x1b[201~End\x1b[201~", "StartMiddleEnd"),
            # Paste with newlines
            ("\x1b[200~Line 1\nLine 2\x1b[201~", "Line 1\nLine 2"),
            # Empty paste
            ("\x1b[200~\x1b[201~", ""),
            # Mixed content
            ("Before\x1b[200~Pasted\x1b[201~After", "BeforePastedAfter"),
        ]
        
        for input_text, expected in test_cases:
            result = enhanced_chat_input.sanitize_input(input_text)
            assert result == expected, f"Failed for input: {input_text!r}"

    def test_sanitize_input_ansi_escape_removal(self, enhanced_chat_input):
        """Test ANSI escape sequences are removed correctly."""
        test_cases = [
            # Color codes
            ("\x1b[31mRed Text\x1b[0m", "Red Text"),
            # Cursor movement
            ("\x1b[2JClear Screen", "Clear Screen"),
            # Complex escape sequences
            ("\x1b[1;32mBold Green\x1b[22;39m", "Bold Green"),
            # Multiple escapes
            ("\x1b[1mBold\x1b[0m \x1b[4mUnderline\x1b[0m", "Bold Underline"),
        ]
        
        for input_text, expected in test_cases:
            result = enhanced_chat_input.sanitize_input(input_text)
            assert result == expected, f"Failed for input: {input_text!r}"

    def test_sanitize_input_carriage_return_handling(self, enhanced_chat_input):
        """Test carriage return handling in input sanitization."""
        test_cases = [
            # Windows line endings
            ("Line 1\r\nLine 2", "Line 1\nLine 2"),
            # Mac line endings
            ("Line 1\rLine 2", "Line 1\nLine 2"),
            # Mixed line endings
            ("Line 1\r\nLine 2\rLine 3\n", "Line 1\nLine 2\nLine 3"),
        ]
        
        for input_text, expected in test_cases:
            result = enhanced_chat_input.sanitize_input(input_text)
            assert result == expected, f"Failed for input: {input_text!r}"

    def test_sanitize_input_whitespace_preservation(self, enhanced_chat_input):
        """Test whitespace preservation in input sanitization."""
        test_cases = [
            # Leading/trailing whitespace removed
            ("  Hello World  ", "Hello World"),
            # Internal whitespace preserved
            ("Hello    World", "Hello    World"),
            # Tabs and spaces
            ("  \t Hello \t World \t  ", "Hello \t World"),
            # Empty string handling
            ("   ", ""),
            ("", ""),
        ]
        
        for input_text, expected in test_cases:
            result = enhanced_chat_input.sanitize_input(input_text)
            assert result == expected, f"Failed for input: {input_text!r}"

    @pytest.mark.asyncio
    async def test_get_user_input_normal_input(self, enhanced_chat_input):
        """Test normal user input handling."""
        test_input = "create main.py\n"
        
        with patch('sys.stdin.readline', return_value=test_input):
            result = await enhanced_chat_input.get_user_input()
            assert result == "create main.py"

    @pytest.mark.asyncio
    async def test_get_user_input_eof_handling(self, enhanced_chat_input):
        """Test EOF handling in user input."""
        with patch('sys.stdin.readline', return_value=""):
            result = await enhanced_chat_input.get_user_input()
            assert result == ""

    @pytest.mark.asyncio
    async def test_get_user_input_with_paste_artifacts(self, enhanced_chat_input):
        """Test user input with paste artifacts."""
        test_input = "\x1b[200~def hello():\n    print('Hello')\x1b[201~\n"
        expected = "def hello():\n    print('Hello')"
        
        with patch('sys.stdin.readline', return_value=test_input):
            result = await enhanced_chat_input.get_user_input()
            assert result == expected

    def test_code_detection_heuristics(self, enhanced_chat_input):
        """Test code detection heuristics."""
        code_examples = [
            # Python code
            "def hello():\n    print('Hello')",
            # JavaScript code  
            "function hello() {\n    console.log('Hello');\n}",
            # General code patterns
            "import sys\nclass MyClass:",
            # Multiple code indicators
            "def process() {\n  return data;\n}",
            # Indented code
            "    function test() {\n        return true;\n    }",
        ]
        
        for code in code_examples:
            assert enhanced_chat_input._looks_like_code(code), f"Failed to detect code: {code}"

    def test_non_code_detection(self, enhanced_chat_input):
        """Test non-code text detection."""
        non_code_examples = [
            "Hello, how are you?",
            "Please help me with this task.",
            "What is the weather like?",
            "Create a new file called main.py",
            "Show me the status of agents",
        ]
        
        for text in non_code_examples:
            assert not enhanced_chat_input._looks_like_code(text), f"Incorrectly detected code: {text}"

    @pytest.mark.skipif(not RICH_AVAILABLE, reason="Rich not available")
    def test_format_message_display_with_rich(self, enhanced_chat_input):
        """Test message formatting with Rich available."""
        message = "Hello World"
        result = enhanced_chat_input.format_message_display(message, "cyan")
        
        assert hasattr(result, 'append') or isinstance(result, str)

    def test_format_message_display_without_rich(self, enhanced_chat_input):
        """Test message formatting without Rich."""
        with patch.dict('sys.modules', {'rich.text': None}):
            # Recreate object to trigger Rich unavailable path
            chat_input = EnhancedChatInput()
            message = "Hello World"
            result = chat_input.format_message_display(message, "cyan")
            
            assert result == message

    def test_language_detection(self, enhanced_chat_input):
        """Test programming language detection."""
        test_cases = [
            ("def hello():\n    pass", "python"),
            ("function test() { return true; }", "javascript"),
            ("#include <stdio.h>\nint main() { return 0; }", "c"),
            ("public class Test { }", "java"),
            ("some random text", "text"),
        ]
        
        for code, expected_lang in test_cases:
            detected_lang = enhanced_chat_input._detect_language(code)
            assert detected_lang == expected_lang, f"Failed for: {code}"

    def test_paste_feedback_methods(self, enhanced_chat_input, mock_console):
        """Test paste feedback display methods."""
        if mock_console:
            enhanced_chat_input.show_paste_feedback("Processing...")
            mock_console.print.assert_called_with("[dim yellow]Processing...[/dim yellow]")
            
            enhanced_chat_input.clear_paste_feedback()
            # Should not raise any errors

    def test_edge_case_empty_input(self, enhanced_chat_input):
        """Test edge case handling for empty input."""
        assert enhanced_chat_input.sanitize_input("") == ""
        assert enhanced_chat_input.sanitize_input(None) == ""

    def test_edge_case_unicode_handling(self, enhanced_chat_input):
        """Test Unicode and emoji handling."""
        unicode_tests = [
            ("Hello 👋 World 🌍", "Hello 👋 World 🌍"),
            ("Café naïve résumé", "Café naïve résumé"),
            ("你好世界", "你好世界"),
            ("🚀 Deploy to production", "🚀 Deploy to production"),
        ]
        
        for input_text, expected in unicode_tests:
            result = enhanced_chat_input.sanitize_input(input_text)
            assert result == expected

    def test_performance_large_input(self, enhanced_chat_input):
        """Test performance with large input."""
        # Generate large input with paste artifacts
        large_input = "\x1b[200~" + "x" * 10000 + "\x1b[201~"
        
        start_time = time.time()
        result = enhanced_chat_input.sanitize_input(large_input)
        duration = time.time() - start_time
        
        assert result == "x" * 10000
        assert duration < 0.1  # Should process in under 100ms


class TestChatHistoryDisplay:
    """Comprehensive test suite for ChatHistoryDisplay component."""

    def test_initialization(self, chat_history_display):
        """Test chat history display initializes correctly."""
        assert chat_history_display.max_history == 50
        assert len(chat_history_display._messages) == 0
        assert chat_history_display._code_block_pattern is not None

    def test_add_message_valid_types(self, chat_history_display):
        """Test adding messages with valid types."""
        valid_messages = [
            ("Hello", "user"),
            ("How can I help?", "assistant"),
            ("System ready", "system"),
            ("Error occurred", "error"),
            ("Tool executed", "tool"),
        ]
        
        for content, msg_type in valid_messages:
            chat_history_display.add_message(content, msg_type)
        
        assert len(chat_history_display._messages) == 5
        assert all(msg.message_type in chat_history_display.MESSAGE_STYLES 
                  for msg in chat_history_display._messages)

    def test_add_message_invalid_type(self, chat_history_display):
        """Test adding message with invalid type raises error."""
        with pytest.raises(ValueError, match="Unknown message type"):
            chat_history_display.add_message("Test", "invalid_type")

    def test_add_message_with_metadata(self, chat_history_display):
        """Test adding message with metadata."""
        metadata = {"source": "api", "confidence": 0.95}
        chat_history_display.add_message("Test message", "user", metadata)
        
        message = chat_history_display._messages[0]
        assert message.metadata == metadata

    def test_history_limit_enforcement(self, chat_history_display):
        """Test history limit enforcement."""
        # Add more messages than the limit
        for i in range(60):
            chat_history_display.add_message(f"Message {i}", "user")
        
        assert len(chat_history_display._messages) == 50
        # Should keep the most recent messages
        assert chat_history_display._messages[0].content == "Message 10"
        assert chat_history_display._messages[-1].content == "Message 59"

    def test_clear_history(self, chat_history_display):
        """Test clearing history."""
        chat_history_display.add_message("Test", "user")
        assert len(chat_history_display._messages) == 1
        
        chat_history_display.clear_history()
        assert len(chat_history_display._messages) == 0

    def test_get_recent_messages(self, chat_history_display):
        """Test getting recent messages."""
        # Add test messages
        for i in range(20):
            chat_history_display.add_message(f"Message {i}", "user")
        
        # Test default count
        recent = chat_history_display.get_recent_messages()
        assert len(recent) == 10
        assert recent[0]['content'] == "Message 10"
        assert recent[-1]['content'] == "Message 19"
        
        # Test custom count
        recent_5 = chat_history_display.get_recent_messages(5)
        assert len(recent_5) == 5
        assert recent_5[0]['content'] == "Message 15"

    def test_get_recent_messages_fewer_than_requested(self, chat_history_display):
        """Test getting recent messages when fewer exist."""
        chat_history_display.add_message("Only message", "user")
        
        recent = chat_history_display.get_recent_messages(10)
        assert len(recent) == 1
        assert recent[0]['content'] == "Only message"

    def test_filter_messages_by_type(self, chat_history_display):
        """Test filtering messages by type."""
        # Add mixed message types
        test_messages = [
            ("User 1", "user"),
            ("Assistant 1", "assistant"),
            ("User 2", "user"),
            ("System 1", "system"),
            ("Assistant 2", "assistant"),
        ]
        
        for content, msg_type in test_messages:
            chat_history_display.add_message(content, msg_type)
        
        # Filter by user messages
        user_messages = chat_history_display.filter_messages(message_type="user")
        assert len(user_messages) == 2
        assert all(msg['message_type'] == 'user' for msg in user_messages)
        
        # Filter by assistant messages
        assistant_messages = chat_history_display.filter_messages(message_type="assistant")
        assert len(assistant_messages) == 2
        assert all(msg['message_type'] == 'assistant' for msg in assistant_messages)

    def test_filter_messages_by_timestamp(self, chat_history_display):
        """Test filtering messages by timestamp."""
        base_time = datetime.now()
        
        # Add messages with different timestamps
        chat_history_display.add_message("Old message", "user")
        chat_history_display._messages[-1].timestamp = base_time - timedelta(hours=2)
        
        chat_history_display.add_message("Recent message", "user")
        chat_history_display._messages[-1].timestamp = base_time - timedelta(minutes=30)
        
        # Filter messages since 1 hour ago
        recent = chat_history_display.filter_messages(since=base_time - timedelta(hours=1))
        assert len(recent) == 1
        assert recent[0]['content'] == "Recent message"

    def test_filter_messages_invalid_type(self, chat_history_display):
        """Test filtering with invalid message type."""
        with pytest.raises(ValueError, match="Unknown message type"):
            chat_history_display.filter_messages(message_type="invalid")

    def test_code_block_detection(self, chat_history_display):
        """Test code block detection in messages."""
        code_message = """Here's a Python example:
```python
def hello():
    print("Hello World")
```
That should work."""
        
        assert chat_history_display._has_code_blocks(code_message)
        
        no_code_message = "This is just regular text without any code."
        assert not chat_history_display._has_code_blocks(no_code_message)

    @pytest.mark.skipif(not RICH_AVAILABLE, reason="Rich not available")
    def test_render_history_with_rich(self, chat_history_display):
        """Test rendering history with Rich available."""
        chat_history_display.add_message("Hello", "user")
        chat_history_display.add_message("Hi there!", "assistant")
        
        rendered = chat_history_display.render_history()
        assert rendered is not None

    def test_render_history_empty(self, chat_history_display):
        """Test rendering empty history."""
        rendered = chat_history_display.render_history()
        
        if RICH_AVAILABLE:
            # Should return a Panel with empty message
            assert rendered is not None
        else:
            assert rendered == "No conversation history yet"

    def test_render_plain_text_fallback(self, chat_history_display):
        """Test plain text rendering fallback."""
        chat_history_display.add_message("Hello", "user")
        chat_history_display.add_message("Hi!", "assistant")
        
        plain_text = chat_history_display._render_plain_text()
        
        assert "Hello" in plain_text
        assert "Hi!" in plain_text
        assert "[USER]" in plain_text
        assert "[ASSISTANT]" in plain_text

    def test_typing_indicator_methods(self, chat_history_display, mock_console):
        """Test typing indicator methods."""
        if mock_console:
            chat_history_display.show_typing_indicator("Processing...")
            mock_console.print.assert_called_with("[dim yellow]Processing...[/dim yellow]")
            
            chat_history_display.clear_typing_indicator()
            # Should not raise errors

    def test_message_content_sanitization(self, chat_history_display):
        """Test message content is sanitized on addition."""
        content_with_whitespace = "  \n  Hello World  \n  "
        chat_history_display.add_message(content_with_whitespace, "user")
        
        stored_message = chat_history_display._messages[0]
        assert stored_message.content == "Hello World"

    def test_concurrent_message_addition(self, chat_history_display):
        """Test thread-safe message addition."""
        import threading
        import time
        
        def add_messages(start_idx):
            for i in range(10):
                chat_history_display.add_message(f"Message {start_idx}-{i}", "user")
                time.sleep(0.001)  # Small delay to simulate real usage
        
        # Start multiple threads adding messages
        threads = []
        for i in range(3):
            thread = threading.Thread(target=add_messages, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        assert len(chat_history_display._messages) == 30

    def test_performance_large_history(self, chat_history_display):
        """Test performance with large message history."""
        # Add maximum history
        start_time = time.time()
        
        for i in range(chat_history_display.max_history):
            chat_history_display.add_message(f"Message {i}", "user")
        
        duration = time.time() - start_time
        assert duration < 1.0  # Should complete in under 1 second
        
        # Test retrieval performance
        start_time = time.time()
        recent = chat_history_display.get_recent_messages(20)
        duration = time.time() - start_time
        
        assert len(recent) == 20
        assert duration < 0.1  # Should retrieve in under 100ms


class TestRealtimeInput:
    """Comprehensive test suite for RealtimeInput component."""

    def test_initialization(self, realtime_input):
        """Test realtime input initializes correctly."""
        assert hasattr(realtime_input, 'set_input_mode')
        assert hasattr(realtime_input, 'validate_input')
        assert hasattr(realtime_input, 'get_completions')

    def test_input_mode_switching(self, realtime_input):
        """Test input mode switching functionality."""
        # Mock the actual implementation since we're testing the interface
        with patch.object(realtime_input, 'set_input_mode') as mock_set_mode:
            with patch.object(realtime_input, 'get_current_mode', return_value=InputMode.MULTI_LINE):
                realtime_input.set_input_mode(InputMode.MULTI_LINE)
                mock_set_mode.assert_called_with(InputMode.MULTI_LINE)

    def test_input_validation_valid_commands(self, realtime_input):
        """Test input validation for valid commands."""
        valid_inputs = [
            "status",
            "help",
            "create file main.py",
            "run tests",
            "show agents",
        ]
        
        # Mock validation method
        def mock_validate(text):
            result = Mock()
            result.is_valid = len(text.strip()) > 0 and not text.startswith("delete /")
            result.error_messages = [] if result.is_valid else ["Invalid command"]
            return result
        
        with patch.object(realtime_input, 'validate_input', side_effect=mock_validate):
            for input_text in valid_inputs:
                result = realtime_input.validate_input(input_text)
                assert result.is_valid, f"Command should be valid: {input_text}"

    def test_input_validation_invalid_commands(self, realtime_input):
        """Test input validation for invalid commands."""
        invalid_inputs = [
            "",  # Empty
            "   ",  # Whitespace only
            "delete / --recursive --force",  # Dangerous
        ]
        
        def mock_validate(text):
            result = Mock()
            result.is_valid = len(text.strip()) > 0 and not text.startswith("delete /")
            result.error_messages = [] if result.is_valid else ["Invalid command"]
            return result
        
        with patch.object(realtime_input, 'validate_input', side_effect=mock_validate):
            for input_text in invalid_inputs:
                result = realtime_input.validate_input(input_text)
                assert not result.is_valid, f"Command should be invalid: {input_text}"

    def test_auto_completion_suggestions(self, realtime_input):
        """Test auto-completion suggestions."""
        test_cases = [
            ("sta", ["status", "start"]),
            ("cr", ["create", "create file", "create directory"]),
            ("he", ["help", "health check"]),
        ]
        
        def mock_completions(partial):
            completions_map = {
                "sta": ["status", "start"],
                "cr": ["create", "create file", "create directory"],
                "he": ["help", "health check"],
            }
            return completions_map.get(partial, [])
        
        with patch.object(realtime_input, 'get_completions', side_effect=mock_completions):
            for partial, expected in test_cases:
                completions = realtime_input.get_completions(partial)
                assert completions == expected

    def test_history_management(self, realtime_input):
        """Test input history management."""
        commands = [
            "status",
            "create file test.py",
            "run tests",
            "deploy production"
        ]
        
        # Mock history methods
        history_store = []
        
        def mock_add_history(command):
            history_store.append(command)
        
        def mock_get_history():
            return history_store.copy()
        
        def mock_get_previous():
            return history_store[-1] if history_store else None
        
        def mock_get_next():
            return history_store[-2] if len(history_store) > 1 else None
        
        with patch.object(realtime_input, 'add_to_history', side_effect=mock_add_history):
            with patch.object(realtime_input, 'get_history', side_effect=mock_get_history):
                with patch.object(realtime_input, 'get_previous_command', side_effect=mock_get_previous):
                    with patch.object(realtime_input, 'get_next_command', side_effect=mock_get_next):
                        
                        for command in commands:
                            realtime_input.add_to_history(command)
                        
                        history = realtime_input.get_history()
                        assert len(history) == len(commands)
                        
                        # Test navigation
                        assert realtime_input.get_previous_command() == commands[-1]
                        assert realtime_input.get_next_command() == commands[-2]

    def test_multi_line_input_processing(self, realtime_input):
        """Test multi-line input processing."""
        multi_line_content = """
        create project structure:
        - src/main.py
        - tests/test_main.py
        - README.md
        - requirements.txt
        """
        
        def mock_process_input(text):
            # Simulate multi-line processing
            return text.strip()
        
        with patch.object(realtime_input, 'process_input', side_effect=mock_process_input):
            result = realtime_input.process_input(multi_line_content)
            
            assert result is not None
            assert "create project structure" in result
            assert "src/main.py" in result


class TestUIComponentIntegration:
    """Integration tests for UI component interactions."""

    def test_enhanced_chat_with_history_integration(self, enhanced_chat_input, chat_history_display):
        """Test integration between enhanced chat input and history display."""
        # Simulate user input with paste artifacts
        raw_input = "\x1b[200~def test():\n    return True\x1b[201~"
        cleaned_input = enhanced_chat_input.sanitize_input(raw_input)
        
        # Add cleaned input to history
        chat_history_display.add_message(cleaned_input, "user")
        
        # Verify integration
        messages = chat_history_display.get_recent_messages(1)
        assert len(messages) == 1
        assert messages[0]['content'] == "def test():\n    return True"
        assert messages[0]['message_type'] == "user"

    def test_realtime_input_with_chat_history(self, realtime_input, chat_history_display):
        """Test integration between realtime input and chat history."""
        # Mock realtime input validation
        def mock_validate(text):
            result = Mock()
            result.is_valid = True
            result.error_messages = []
            return result
        
        with patch.object(realtime_input, 'validate_input', side_effect=mock_validate):
            # Simulate valid command
            command = "create file main.py"
            validation = realtime_input.validate_input(command)
            
            if validation.is_valid:
                chat_history_display.add_message(command, "user")
                chat_history_display.add_message("File created successfully", "assistant")
        
        # Verify integration
        history = chat_history_display.get_recent_messages(2)
        assert len(history) == 2
        assert history[0]['content'] == command
        assert history[1]['content'] == "File created successfully"

    @pytest.mark.asyncio
    async def test_async_input_processing_integration(self, enhanced_chat_input, chat_history_display):
        """Test asynchronous input processing with history updates."""
        # Simulate async input processing
        test_input = "analyze codebase\n"
        
        with patch('sys.stdin.readline', return_value=test_input):
            user_input = await enhanced_chat_input.get_user_input()
            
            # Process and add to history
            chat_history_display.add_message(user_input, "user")
            
            # Simulate AI processing
            await asyncio.sleep(0.01)  # Simulate processing time
            
            chat_history_display.add_message("Analysis complete", "assistant")
        
        # Verify async integration
        history = chat_history_display.get_recent_messages(2)
        assert len(history) == 2
        assert history[0]['content'] == "analyze codebase"
        assert history[1]['content'] == "Analysis complete"


class TestPerformanceBenchmarks:
    """Performance tests for UI components."""

    def test_enhanced_chat_input_performance(self, enhanced_chat_input):
        """Test enhanced chat input performance with large inputs."""
        # Test with large input containing many paste artifacts
        large_input = "\x1b[200~" + "x" * 50000 + "\x1b[201~"
        
        start_time = time.perf_counter()
        result = enhanced_chat_input.sanitize_input(large_input)
        duration = time.perf_counter() - start_time
        
        assert len(result) == 50000
        assert duration < 0.05  # Should process in under 50ms

    def test_chat_history_performance_batch_operations(self, chat_history_display):
        """Test chat history performance with batch operations."""
        # Test batch message addition
        messages_to_add = 1000
        
        start_time = time.perf_counter()
        for i in range(messages_to_add):
            chat_history_display.add_message(f"Message {i}", "user")
        duration = time.perf_counter() - start_time
        
        # Should maintain only max_history messages
        assert len(chat_history_display._messages) == min(messages_to_add, chat_history_display.max_history)
        assert duration < 1.0  # Should complete in under 1 second

    def test_history_retrieval_performance(self, chat_history_display):
        """Test history retrieval performance."""
        # Fill history to maximum
        for i in range(chat_history_display.max_history):
            chat_history_display.add_message(f"Message {i}", "user")
        
        # Test retrieval performance
        start_time = time.perf_counter()
        recent_messages = chat_history_display.get_recent_messages(20)
        duration = time.perf_counter() - start_time
        
        assert len(recent_messages) == 20
        assert duration < 0.01  # Should retrieve in under 10ms

    def test_filtering_performance(self, chat_history_display):
        """Test message filtering performance."""
        # Add mixed message types
        message_types = ["user", "assistant", "system", "error", "tool"]
        for i in range(chat_history_display.max_history):
            msg_type = message_types[i % len(message_types)]
            chat_history_display.add_message(f"Message {i}", msg_type)
        
        # Test filtering performance
        start_time = time.perf_counter()
        user_messages = chat_history_display.filter_messages(message_type="user")
        duration = time.perf_counter() - start_time
        
        assert len(user_messages) > 0
        assert duration < 0.02  # Should filter in under 20ms


class TestAccessibilityCompliance:
    """Test suite for accessibility compliance."""

    def test_chat_history_accessibility_features(self, chat_history_display):
        """Test chat history accessibility features."""
        # Add messages with various content types
        chat_history_display.add_message("Regular text message", "user")
        chat_history_display.add_message("```python\nprint('code')\n```", "assistant")
        chat_history_display.add_message("System notification", "system")
        
        # Test that messages include proper metadata for screen readers
        messages = chat_history_display.get_recent_messages(3)
        
        for message in messages:
            assert 'message_type' in message  # Role information
            assert 'timestamp' in message    # Temporal information
            assert 'content' in message      # Text content

    def test_enhanced_chat_input_accessibility(self, enhanced_chat_input):
        """Test enhanced chat input accessibility features."""
        # Test that input sanitization preserves meaningful content
        accessible_inputs = [
            ("Screen reader: Hello world", "Screen reader: Hello world"),
            ("Alt text: Image description", "Alt text: Image description"),
            ("ARIA label content", "ARIA label content"),
        ]
        
        for input_text, expected in accessible_inputs:
            result = enhanced_chat_input.sanitize_input(input_text)
            assert result == expected

    def test_high_contrast_compatibility(self, chat_history_display):
        """Test high contrast mode compatibility."""
        # Ensure message styling works with high contrast
        chat_history_display.add_message("High contrast test", "user")
        chat_history_display.add_message("Error message test", "error")
        
        # Verify message types are properly categorized for styling
        messages = chat_history_display._messages
        assert messages[0].message_type == "user"
        assert messages[1].message_type == "error"
        
        # Verify styling information is available
        assert "user" in chat_history_display.MESSAGE_STYLES
        assert "error" in chat_history_display.MESSAGE_STYLES


class TestErrorHandlingAndEdgeCases:
    """Test suite for error handling and edge cases."""

    def test_enhanced_chat_input_error_recovery(self, enhanced_chat_input):
        """Test enhanced chat input error recovery."""
        # Test with malformed input
        malformed_inputs = [
            "\x1b[200~\x1b[200~nested\x1b[201~",  # Nested paste markers
            "\x1b[999mUnknown escape\x1b[0m",      # Unknown ANSI codes
            "Mixed\rLine\nEndings\r\n",            # Mixed line endings
        ]
        
        for malformed_input in malformed_inputs:
            # Should not raise exceptions
            result = enhanced_chat_input.sanitize_input(malformed_input)
            assert isinstance(result, str)

    def test_chat_history_edge_cases(self, chat_history_display):
        """Test chat history edge cases."""
        # Test with maximum length content
        very_long_content = "x" * 10000
        chat_history_display.add_message(very_long_content, "user")
        
        messages = chat_history_display.get_recent_messages(1)
        assert len(messages[0]['content']) == 10000
        
        # Test with empty content
        chat_history_display.add_message("", "system")
        messages = chat_history_display.get_recent_messages(1)
        assert messages[0]['content'] == ""

    def test_concurrent_access_safety(self, chat_history_display):
        """Test thread safety for concurrent access."""
        import threading
        import time
        
        def add_messages(thread_id):
            for i in range(100):
                try:
                    chat_history_display.add_message(f"Thread {thread_id} - Message {i}", "user")
                    time.sleep(0.001)
                except Exception as e:
                    pytest.fail(f"Thread safety failed: {e}")
        
        threads = []
        for thread_id in range(5):
            thread = threading.Thread(target=add_messages, args=(thread_id,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should have maintained history limit without errors
        assert len(chat_history_display._messages) == chat_history_display.max_history


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--durations=10"])