#!/usr/bin/env python3
"""
Unit tests for the feature showcase message display system.

Tests the complete message flow:
TaskTracker -> ChatEngine -> ConsoleMessageFormatter

Verifies that Rich-formatted showcases display correctly without truncation.
"""

import unittest
import time
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

import sys
import os

# Add the source directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from agentsmcp.ui.v3.chat_engine import ChatEngine, ChatMessage, MessageRole
from agentsmcp.ui.v3.console_message_formatter import ConsoleMessageFormatter
from agentsmcp.capabilities.feature_detector import FeatureDetector, FeatureDetectionResult


class TestFeatureShowcaseDisplay(unittest.TestCase):
    """Test the feature showcase display system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.chat_engine = ChatEngine()
        self.captured_messages = []
        self.captured_statuses = []
        self.captured_errors = []
        
        # Mock callbacks
        def mock_message_callback(message: ChatMessage):
            self.captured_messages.append(message)
            
        def mock_status_callback(status: str):
            self.captured_statuses.append(status)
            
        def mock_error_callback(error: str):
            self.captured_errors.append(error)
        
        self.chat_engine.set_callbacks(
            message_callback=mock_message_callback,
            status_callback=mock_status_callback,
            error_callback=mock_error_callback
        )
    
    def test_feature_showcase_message_routing(self):
        """Test that FEATURE_SHOWCASE: messages are routed correctly."""
        showcase_content = "🎯 **Test Feature**\n\n✅ Feature exists"
        status_message = f"FEATURE_SHOWCASE:{showcase_content}"
        
        # Process the status message
        self.chat_engine._notify_status(status_message)
        
        # Verify message was captured, not status
        self.assertEqual(len(self.captured_messages), 1, "Should capture one message")
        self.assertEqual(len(self.captured_statuses), 0, "Should not capture status for showcase")
        
        # Verify message content
        message = self.captured_messages[0]
        self.assertEqual(message.role, MessageRole.SYSTEM)
        expected_content = f"FEATURE_SHOWCASE_FORMAT:{showcase_content}"
        self.assertEqual(message.content, expected_content)
        self.assertIsInstance(message.timestamp, float)
        self.assertGreater(message.timestamp, 0)
    
    def test_feature_showcase_no_truncation(self):
        """Test that long showcase messages are not truncated."""
        # Create a long showcase message
        long_content = "🎯 **Long Feature Description**\n\n" + "x" * 1000
        status_message = f"FEATURE_SHOWCASE:{long_content}"
        
        # Process the message
        self.chat_engine._notify_status(status_message)
        
        # Verify no truncation occurred
        self.assertEqual(len(self.captured_messages), 1)
        message = self.captured_messages[0]
        
        # Should contain the full content plus the prefix
        expected_length = len(f"FEATURE_SHOWCASE_FORMAT:{long_content}")
        self.assertEqual(len(message.content), expected_length)
        
        # Should contain the full original content
        extracted_content = message.content[24:]  # Remove prefix
        self.assertEqual(extracted_content, long_content)
    
    def test_console_formatter_showcase_display(self):
        """Test that ConsoleMessageFormatter handles showcase messages correctly."""
        from rich.console import Console
        
        # Create test showcase message
        showcase_content = "🎯 **Test Feature**\n\n✅ This feature works\n\n**Try it:**\n```bash\n$ test command\n```"
        message = ChatMessage(
            role=MessageRole.SYSTEM,
            content=f"FEATURE_SHOWCASE_FORMAT:{showcase_content}",
            timestamp=time.time()
        )
        
        # Capture console output
        output_capture = StringIO()
        mock_console = Console(file=output_capture, width=80, force_terminal=True)
        formatter = ConsoleMessageFormatter(mock_console)
        
        # Format the message (this calls format_feature_showcase internally)
        with patch.object(formatter, 'format_feature_showcase') as mock_format:
            formatter._format_system_message(message.content, "test timestamp")
            
            # Verify format_feature_showcase was called with correct content
            mock_format.assert_called_once_with(showcase_content)
    
    def test_rich_panel_rendering(self):
        """Test that Rich Panel rendering works correctly."""
        from rich.console import Console
        
        showcase_content = """🎯 **Feature Already Available**

✅ This cli_flag already exists in the codebase

**Try it now:**
```bash
$ ./agentsmcp --version
```

**Related features:**
• `--help`
• `--verbose`

**Detection evidence:**
• Found in help output
• Confirmed via testing"""
        
        # Test direct panel rendering
        output_capture = StringIO()
        console = Console(file=output_capture, width=100, force_terminal=True)
        formatter = ConsoleMessageFormatter(console)
        
        # Call format_feature_showcase directly
        formatter.format_feature_showcase(showcase_content)
        output = output_capture.getvalue()
        
        # Verify output contains expected elements
        self.assertGreater(len(output), 1000, "Should generate substantial Rich output")
        self.assertIn("Feature Already Available", output, "Should contain title")
        self.assertIn("cli_flag already exists", output, "Should contain main message")
        self.assertIn("./agentsmcp --version", output, "Should contain usage example")
        self.assertIn("help", output, "Should contain related features")
        self.assertIn("Found in help output", output, "Should contain evidence")
    
    def test_fallback_behavior(self):
        """Test that fallback still works if showcase display fails."""
        # Mock the _notify_message to raise an exception
        with patch.object(self.chat_engine, '_notify_message', side_effect=Exception("Test error")):
            showcase_content = "Test showcase content"
            status_message = f"FEATURE_SHOWCASE:{showcase_content}"
            
            # Process the message
            self.chat_engine._notify_status(status_message)
            
            # Should fall back to status callback with truncated message
            self.assertEqual(len(self.captured_statuses), 1)
            status = self.captured_statuses[0]
            self.assertTrue(status.startswith("✅ Feature exists:"))
            self.assertIn("Test showcase", status)
    
    def test_prefix_stripping_accuracy(self):
        """Test that prefixes are stripped accurately without data loss."""
        original_content = "🎯 **Test**\n\nMulti-line\ncontent with\nspecial chars: !@#$%"
        
        # Test TaskTracker -> ChatEngine prefix
        tasktracker_msg = f"FEATURE_SHOWCASE:{original_content}"
        self.assertTrue(tasktracker_msg.startswith("FEATURE_SHOWCASE:"))
        extracted1 = tasktracker_msg[17:]  # Remove "FEATURE_SHOWCASE:" 
        self.assertEqual(extracted1, original_content)
        
        # Test ChatEngine -> ConsoleFormatter prefix
        chatengine_msg = f"FEATURE_SHOWCASE_FORMAT:{original_content}"
        self.assertTrue(chatengine_msg.startswith("FEATURE_SHOWCASE_FORMAT:"))
        extracted2 = chatengine_msg[24:]  # Remove "FEATURE_SHOWCASE_FORMAT:"
        self.assertEqual(extracted2, original_content)
        
        # Verify no data corruption through the pipeline
        self.assertEqual(len(extracted2), len(original_content))
        self.assertEqual(extracted2.count('\n'), original_content.count('\n'))
    
    def test_message_role_enum_usage(self):
        """Test that MessageRole enum is used correctly."""
        showcase_content = "Test content"
        status_message = f"FEATURE_SHOWCASE:{showcase_content}"
        
        self.chat_engine._notify_status(status_message)
        
        self.assertEqual(len(self.captured_messages), 1)
        message = self.captured_messages[0]
        
        # Verify role is the enum, not a string
        self.assertIsInstance(message.role, MessageRole)
        self.assertEqual(message.role, MessageRole.SYSTEM)
        self.assertEqual(message.role.value, "system")
    
    def test_timestamp_format(self):
        """Test that timestamp is in correct format."""
        showcase_content = "Test content" 
        status_message = f"FEATURE_SHOWCASE:{showcase_content}"
        
        start_time = time.time()
        self.chat_engine._notify_status(status_message)
        end_time = time.time()
        
        self.assertEqual(len(self.captured_messages), 1)
        message = self.captured_messages[0]
        
        # Verify timestamp is a float in reasonable range
        self.assertIsInstance(message.timestamp, float)
        self.assertGreaterEqual(message.timestamp, start_time)
        self.assertLessEqual(message.timestamp, end_time)


class TestFeatureShowcaseIntegration(unittest.TestCase):
    """Integration tests for feature showcase with FeatureDetector."""
    
    def test_end_to_end_showcase_generation(self):
        """Test complete showcase generation and display pipeline."""
        # Create a FeatureDetectionResult
        result = FeatureDetectionResult(
            exists=True,
            feature_type="cli_flag",
            detection_method="help_analysis",
            evidence=["Found in help output", "Direct testing confirmed"],
            usage_examples=["./agentsmcp --test", "./agentsmcp -t"],
            related_features=["debug", "verbose"], 
            confidence=0.95
        )
        
        # Generate showcase message
        detector = FeatureDetector()
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            showcase_message = loop.run_until_complete(
                detector.generate_feature_showcase(result)
            )
        finally:
            loop.close()
        
        # Verify showcase message structure
        self.assertIn("🎯 **Feature Already Available**", showcase_message)
        self.assertIn("cli_flag already exists", showcase_message)
        self.assertIn("**Try it now:**", showcase_message)
        self.assertIn("./agentsmcp --test", showcase_message)
        self.assertIn("./agentsmcp -t", showcase_message)
        self.assertIn("**Related features you might like:**", showcase_message)
        self.assertIn("--debug", showcase_message)
        self.assertIn("**Detection evidence:**", showcase_message)
        self.assertIn("Found in help output", showcase_message)
        
        # Test that it flows through ChatEngine correctly
        chat_engine = ChatEngine()
        captured_messages = []
        
        def capture_message(msg):
            captured_messages.append(msg)
        
        chat_engine.set_callbacks(message_callback=capture_message)
        
        # Simulate TaskTracker callback
        tasktracker_message = f"FEATURE_SHOWCASE:{showcase_message}"
        chat_engine._notify_status(tasktracker_message)
        
        # Verify end-to-end flow
        self.assertEqual(len(captured_messages), 1)
        final_message = captured_messages[0]
        self.assertEqual(final_message.role, MessageRole.SYSTEM)
        self.assertTrue(final_message.content.startswith("FEATURE_SHOWCASE_FORMAT:"))
        
        # Extract and verify final content
        final_content = final_message.content[24:]
        self.assertEqual(final_content, showcase_message)


if __name__ == '__main__':
    unittest.main()