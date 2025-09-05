#!/usr/bin/env python3
"""
Comprehensive test script to debug the feature showcase message truncation issue.

This script tests the complete message flow:
TaskTracker -> ChatEngine -> ConsoleMessageFormatter

It specifically focuses on:
1. Message prefix routing (FEATURE_SHOWCASE: -> FEATURE_SHOWCASE_FORMAT:)  
2. Rich formatting and Panel display
3. Identifying where message truncation occurs
4. Testing both console and Rich TUI rendering paths
"""

import sys
import os
import io
import time
from unittest.mock import Mock, patch
from contextlib import redirect_stdout, redirect_stderr

# Add the source directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agentsmcp.ui.v3.chat_engine import ChatEngine, ChatMessage, MessageRole
from agentsmcp.ui.v3.console_message_formatter import ConsoleMessageFormatter
from agentsmcp.capabilities.feature_detector import FeatureDetector, FeatureDetectionResult


def create_sample_showcase_message():
    """Create a sample showcase message similar to what FeatureDetector generates."""
    result = FeatureDetectionResult(
        exists=True,
        feature_type="cli_flag",
        detection_method="cli_help_analysis",
        evidence=[
            "Found '--version' in help output",
            "Confirmed via direct testing",
            "Source code contains version handling"
        ],
        usage_examples=[
            "./agentsmcp --version",
            "./agentsmcp -v"
        ],
        related_features=[
            "help", "verbose", "debug"
        ],
        confidence=0.95
    )
    
    detector = FeatureDetector()
    
    # Mock the async method to be sync for testing
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    showcase = loop.run_until_complete(detector.generate_feature_showcase(result))
    loop.close()
    
    return showcase


def test_message_flow_step_by_step():
    """Test each step of the message flow to identify truncation points."""
    print("=" * 80)
    print("TESTING FEATURE SHOWCASE MESSAGE FLOW")
    print("=" * 80)
    
    # Step 1: Generate the showcase message
    print("\n1. GENERATING SHOWCASE MESSAGE")
    print("-" * 40)
    showcase_message = create_sample_showcase_message()
    print(f"Original showcase message length: {len(showcase_message)} characters")
    print("Full showcase message:")
    print(repr(showcase_message))
    print("\nReadable format:")
    print(showcase_message)
    
    # Step 2: TaskTracker sends via callback (simulated)
    print("\n2. TASKTRACKER CALLBACK SIMULATION")
    print("-" * 40)
    tasktracker_message = f"FEATURE_SHOWCASE:{showcase_message}"
    print(f"TaskTracker sends: {len(tasktracker_message)} characters")
    print(f"First 100 chars: {repr(tasktracker_message[:100])}")
    
    # Step 3: ChatEngine processes the status callback
    print("\n3. CHATENGINE STATUS PROCESSING")  
    print("-" * 40)
    
    # Create ChatEngine with mock callbacks
    message_captured = []
    status_captured = []
    error_captured = []
    
    def mock_status_callback(status: str):
        status_captured.append(status)
        print(f"Status callback received: {len(status)} chars")
        print(f"Status preview: {repr(status[:100])}")
    
    def mock_message_callback(message: ChatMessage):
        message_captured.append(message)
        print(f"Message callback received: role={message.role.value}")
        print(f"Message content length: {len(message.content)} chars")
        print(f"Message content preview: {repr(message.content[:100])}")
        
    def mock_error_callback(error: str):
        error_captured.append(error)
        print(f"Error callback: {error}")
    
    chat_engine = ChatEngine()
    chat_engine.set_callbacks(
        status_callback=mock_status_callback,
        message_callback=mock_message_callback,
        error_callback=mock_error_callback
    )
    
    # Trigger the status processing
    chat_engine._notify_status(tasktracker_message)
    
    print(f"Messages captured: {len(message_captured)}")
    if message_captured:
        msg = message_captured[0]
        print(f"Captured message role: {msg.role.value if hasattr(msg.role, 'value') else msg.role}")
        print(f"Captured message content length: {len(msg.content)}")
        print(f"Expected prefix 'FEATURE_SHOWCASE_FORMAT:': {msg.content.startswith('FEATURE_SHOWCASE_FORMAT:')}")
        
    # Step 4: ConsoleMessageFormatter processes the message  
    print("\n4. CONSOLE MESSAGE FORMATTER PROCESSING")
    print("-" * 40)
    
    if message_captured:
        # Test with both a mock console and real Rich console
        from rich.console import Console
        from io import StringIO
        
        # Test 1: Mock console output capture
        mock_output = StringIO()
        mock_console = Console(file=mock_output, width=100, force_terminal=True)
        formatter = ConsoleMessageFormatter(mock_console)
        
        try:
            formatter.format_message(message_captured[0])
            output = mock_output.getvalue()
            print(f"Mock console output length: {len(output)} characters")
            print("Mock console output (first 500 chars):")
            print(repr(output[:500]))
            print("\nMock console output (readable):")
            print(output[:500])
            
        except Exception as e:
            print(f"Error in mock console formatting: {e}")
            import traceback
            traceback.print_exc()
        
        # Test 2: Real console output (to terminal)
        print("\n5. REAL RICH CONSOLE OUTPUT")
        print("-" * 40)
        real_console = Console()
        real_formatter = ConsoleMessageFormatter(real_console)
        
        try:
            print("Rendering with real Rich console:")
            real_formatter.format_message(message_captured[0])
            
        except Exception as e:
            print(f"Error in real console formatting: {e}")
            import traceback
            traceback.print_exc()


def test_direct_showcase_formatting():
    """Test the showcase formatting directly to isolate issues."""
    print("\n" + "=" * 80)
    print("TESTING DIRECT SHOWCASE FORMATTING")
    print("=" * 80)
    
    showcase_message = create_sample_showcase_message()
    
    from rich.console import Console
    from io import StringIO
    
    # Test the format_feature_showcase method directly
    mock_output = StringIO()
    mock_console = Console(file=mock_output, width=100, force_terminal=True)
    formatter = ConsoleMessageFormatter(mock_console)
    
    print(f"Input showcase message length: {len(showcase_message)}")
    print("Testing format_feature_showcase directly...")
    
    try:
        formatter.format_feature_showcase(showcase_message)
        output = mock_output.getvalue()
        print(f"Direct formatting output length: {len(output)} characters")
        print("Output:")
        print(output)
        
    except Exception as e:
        print(f"Error in direct formatting: {e}")
        import traceback
        traceback.print_exc()


def test_rich_markdown_panel():
    """Test Rich markdown and panel rendering specifically."""
    print("\n" + "=" * 80)
    print("TESTING RICH MARKDOWN AND PANEL RENDERING")
    print("=" * 80)
    
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from io import StringIO
    
    showcase_message = create_sample_showcase_message()
    
    # Test 1: Just Markdown rendering
    print("1. Testing Markdown rendering alone:")
    mock_output = StringIO()
    console = Console(file=mock_output, width=100, force_terminal=True)
    
    try:
        markdown = Markdown(showcase_message)
        console.print(markdown)
        output = mock_output.getvalue()
        print(f"Markdown output length: {len(output)}")
        print("Markdown output:")
        print(output)
    except Exception as e:
        print(f"Markdown error: {e}")
    
    # Test 2: Panel with Markdown
    print("\n2. Testing Panel with Markdown:")
    mock_output2 = StringIO()
    console2 = Console(file=mock_output2, width=100, force_terminal=True)
    
    try:
        markdown = Markdown(showcase_message)
        panel = Panel(
            markdown,
            title="🎯 Feature Already Available",
            title_align="left", 
            border_style="green",
            padding=(0, 1)
        )
        console2.print(panel)
        output2 = mock_output2.getvalue()
        print(f"Panel output length: {len(output2)}")
        print("Panel output:")
        print(output2)
    except Exception as e:
        print(f"Panel error: {e}")


def test_prefix_stripping():
    """Test the prefix stripping logic specifically."""
    print("\n" + "=" * 80)
    print("TESTING PREFIX STRIPPING LOGIC")
    print("=" * 80)
    
    showcase_content = create_sample_showcase_message()
    
    # Test ChatEngine prefix stripping
    test_status = f"FEATURE_SHOWCASE:{showcase_content}"
    print(f"Original status length: {len(test_status)}")
    
    # Simulate ChatEngine._notify_status logic
    if test_status.startswith("FEATURE_SHOWCASE:"):
        extracted = test_status[17:]  # Remove "FEATURE_SHOWCASE:" prefix  
        print(f"After ChatEngine extraction: {len(extracted)} chars")
        print(f"Content matches original: {extracted == showcase_content}")
        
        # Simulate the ChatMessage creation
        formatted_content = f"FEATURE_SHOWCASE_FORMAT:{extracted}"
        print(f"ChatMessage content length: {len(formatted_content)}")
        
        # Test ConsoleMessageFormatter prefix stripping
        if formatted_content.startswith("FEATURE_SHOWCASE_FORMAT:"):
            final_content = formatted_content[24:]  # Remove "FEATURE_SHOWCASE_FORMAT:" prefix
            print(f"After ConsoleFormatter extraction: {len(final_content)} chars")
            print(f"Final content matches original: {final_content == showcase_content}")
            print("First 200 chars of final content:")
            print(repr(final_content[:200]))


def test_status_callback_limitations():
    """Test if status callback has length limitations."""
    print("\n" + "=" * 80) 
    print("TESTING STATUS CALLBACK LIMITATIONS")
    print("=" * 80)
    
    # Create very long messages to test limits
    short_msg = "Short message"
    medium_msg = "x" * 500  
    long_msg = "y" * 2000
    very_long_msg = "z" * 10000
    
    captured_statuses = []
    
    def test_callback(status: str):
        captured_statuses.append(status)
        print(f"Callback received: {len(status)} chars")
    
    chat_engine = ChatEngine()
    chat_engine.set_callbacks(status_callback=test_callback, message_callback=None, error_callback=None)
    
    for i, msg in enumerate([short_msg, medium_msg, long_msg, very_long_msg], 1):
        print(f"\nTest {i}: Sending {len(msg)} character message")
        chat_engine._notify_status(f"REGULAR_STATUS:{msg}")
        
        if captured_statuses:
            received = captured_statuses[-1]
            print(f"Received: {len(received)} chars")
            expected_prefix = f"REGULAR_STATUS:{msg}"
            print(f"Complete message received: {received == expected_prefix}")


def main():
    """Run all tests."""
    print("FEATURE SHOWCASE MESSAGE FLOW DEBUG TEST")
    print("This script will test the complete message flow and identify truncation issues.")
    print()
    
    try:
        # Test the complete message flow
        test_message_flow_step_by_step()
        
        # Test direct showcase formatting
        test_direct_showcase_formatting()
        
        # Test Rich components individually
        test_rich_markdown_panel()
        
        # Test prefix stripping logic
        test_prefix_stripping()
        
        # Test status callback limitations
        test_status_callback_limitations()
        
        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED")
        print("=" * 80)
        print("Review the output above to identify where message truncation occurs.")
        print("Key things to check:")
        print("1. Are all message lengths preserved through each step?")
        print("2. Does the Rich formatting work correctly?")
        print("3. Are the prefixes being stripped properly?")
        print("4. Does the Panel display render the complete message?")
        
    except Exception as e:
        print(f"Test suite error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())