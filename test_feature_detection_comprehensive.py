#!/usr/bin/env python3
"""
Comprehensive End-to-End Test for Feature Detection and Showcase System

This test verifies the complete flow:
1. TaskTracker detects existing features 
2. ChatEngine routes feature showcase messages
3. ConsoleMessageFormatter displays Rich formatted panels
4. System terminates cleanly without continued AI processing

Tests the exact user scenarios that were reported as problematic:
- "--help flag" detection and showcase
- "tui command" detection and showcase
"""

import sys
import os
import time
import subprocess
import threading
import io
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import Mock, patch

# Add the src directory to path to import AgentsMCP modules
sys.path.insert(0, '/Users/mikko/github/AgentsMCP/src')

from agentsmcp.orchestration.task_tracker import TaskTracker
from agentsmcp.ui.v3.chat_engine import ChatEngine, ChatState, ChatMessage, MessageRole
from agentsmcp.ui.v3.console_message_formatter import ConsoleMessageFormatter
from agentsmcp.capabilities.feature_detector import FeatureDetector

# Rich imports for capturing output
from rich.console import Console
from rich import get_console

class TestResults:
    def __init__(self):
        self.feature_detection_works = False
        self.showcase_message_complete = False
        self.no_task_execution_failed = False
        self.system_stops_cleanly = False
        self.rich_formatting_works = False
        self.captured_output = ""
        self.exceptions = []
        self.truncation_detected = False

def capture_console_output():
    """Capture all Rich console output"""
    console = get_console()
    output_buffer = io.StringIO()
    
    # Create a new console that writes to our buffer
    test_console = Console(file=output_buffer, force_terminal=True, width=80)
    
    return test_console, output_buffer

def test_feature_detection_direct():
    """Test 1: Direct feature detection functionality"""
    print("\n=== TEST 1: Direct Feature Detection ===")
    
    try:
        detector = FeatureDetector()
        
        # Test help flag detection
        help_result = detector.detect_existing_feature("--help flag")
        print(f"Help detection result: {help_result.exists if help_result else None}")
        
        # Test TUI command detection  
        tui_result = detector.detect_existing_feature("tui command")
        print(f"TUI detection result: {tui_result.exists if tui_result else None}")
        
        return help_result and help_result.exists and tui_result and tui_result.exists
        
    except Exception as e:
        print(f"Feature detection error: {e}")
        return False

def test_task_tracker_showcase_message():
    """Test 2: TaskTracker showcase message generation"""
    print("\n=== TEST 2: TaskTracker Showcase Message Generation ===")
    
    captured_messages = []
    
    def mock_progress_callback(message):
        captured_messages.append(message)
        print(f"Progress callback received: {message[:100]}...")
    
    try:
        # Create TaskTracker with mock callback
        tracker = TaskTracker()
        tracker.progress_update_callback = mock_progress_callback
        
        # Test with help flag request
        task_id = tracker.submit_task("--help flag", "Show help information")
        
        # Check if we got a FEATURE_SHOWCASE message
        showcase_messages = [msg for msg in captured_messages if msg.startswith("FEATURE_SHOWCASE:")]
        
        print(f"Found {len(showcase_messages)} showcase messages")
        for msg in showcase_messages:
            print(f"Showcase message length: {len(msg)} chars")
            if len(msg) < 200:
                print(f"WARNING: Message seems truncated: {msg}")
                
        return len(showcase_messages) > 0 and len(showcase_messages[0]) > 200
        
    except Exception as e:
        print(f"TaskTracker error: {e}")
        return False

def test_chat_engine_routing():
    """Test 3: ChatEngine message routing and formatting"""
    print("\n=== TEST 3: ChatEngine Message Routing ===")
    
    captured_messages = []
    
    def mock_message_callback(message: ChatMessage):
        captured_messages.append(message)
        print(f"Message callback: {message.role} - {message.content[:100]}...")
    
    try:
        # Create ChatEngine with mock callback
        engine = ChatEngine()
        engine.set_callbacks(message_callback=mock_message_callback)
        
        # Simulate the feature showcase flow
        test_showcase = "# TUI Command Available\n\nThe TUI is already implemented! Use:\n\n```bash\n./agentsmcp tui\n```\n\nFeatures:\n- Interactive chat\n- Real-time updates\n- Rich formatting"
        
        # This should trigger the _display_feature_showcase method
        engine._notify_status(f"FEATURE_SHOWCASE:{test_showcase}")
        
        # Check if proper FEATURE_SHOWCASE_FORMAT message was created
        format_messages = [msg for msg in captured_messages 
                          if msg.content.startswith("FEATURE_SHOWCASE_FORMAT:")]
        
        print(f"Found {len(format_messages)} format messages")
        for msg in format_messages:
            print(f"Format message length: {len(msg.content)} chars")
            
        return len(format_messages) > 0
        
    except Exception as e:
        print(f"ChatEngine error: {e}")
        return False

def test_console_formatter_rich_display():
    """Test 4: ConsoleMessageFormatter Rich Panel Display"""
    print("\n=== TEST 4: Console Formatter Rich Display ===")
    
    try:
        # Capture Rich output
        test_console, output_buffer = capture_console_output()
        
        # Create formatter with test console
        formatter = ConsoleMessageFormatter(console=test_console)
        
        # Test showcase message formatting
        test_content = "# TUI Command Available\n\nThe TUI is already implemented!\n\n## Features:\n- Interactive chat\n- Rich formatting\n- Real-time updates\n\n```bash\n./agentsmcp tui\n```"
        
        # Create FEATURE_SHOWCASE_FORMAT message
        showcase_message = ChatMessage(
            role=MessageRole.SYSTEM,
            content=f"FEATURE_SHOWCASE_FORMAT:{test_content}",
            timestamp=time.time()
        )
        
        # Format the message
        formatter.format_message(showcase_message)
        
        # Get captured output
        captured = output_buffer.getvalue()
        print(f"Captured output length: {len(captured)} chars")
        
        # Check for Rich panel indicators
        has_panel = "╭" in captured and "╰" in captured  # Panel borders
        has_title = "Feature Already Available" in captured
        has_content = "TUI Command Available" in captured
        has_markdown = "Interactive chat" in captured
        
        print(f"Rich panel detected: {has_panel}")
        print(f"Panel title detected: {has_title}")  
        print(f"Content rendered: {has_content}")
        print(f"Markdown formatted: {has_markdown}")
        
        if len(captured) < 100:
            print("WARNING: Output seems truncated!")
            print(f"Full output: {repr(captured)}")
        
        return has_panel and has_content and len(captured) > 100
        
    except Exception as e:
        print(f"Console formatter error: {e}")
        return False

def test_end_to_end_user_scenarios():
    """Test 5: End-to-end user scenarios via CLI"""
    print("\n=== TEST 5: End-to-End CLI User Scenarios ===")
    
    test_cases = [
        ("--help flag", "Show help information"),
        ("tui command", "Start the TUI interface")  
    ]
    
    results = []
    
    for query, description in test_cases:
        print(f"\nTesting: '{query}' ({description})")
        
        try:
            # Run AgentsMCP with the query
            process = subprocess.Popen(
                [sys.executable, '-m', 'agentsmcp.cli', query],
                cwd='/Users/mikko/github/AgentsMCP',
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30
            )
            
            stdout, stderr = process.communicate()
            
            print(f"Exit code: {process.returncode}")
            print(f"STDOUT length: {len(stdout)} chars")
            print(f"STDERR length: {len(stderr)} chars")
            
            # Check for key indicators
            has_feature_showcase = "Feature Already Available" in stdout or "🎯" in stdout
            has_truncation = len(stdout) > 0 and len(stdout) < 200 and "..." in stdout  
            has_task_failed = "Task execution failed" in stdout
            has_continued_ai = "I'll help you" in stdout or "Let me" in stdout
            
            print(f"Feature showcase detected: {has_feature_showcase}")
            print(f"Truncation detected: {has_truncation}")
            print(f"Task failed messages: {has_task_failed}")
            print(f"Continued AI processing: {has_continued_ai}")
            
            # Show first 300 chars of output for analysis
            if stdout:
                print(f"First 300 chars: {stdout[:300]}...")
            if stderr:
                print(f"STDERR: {stderr[:200]}...")
            
            results.append({
                'query': query,
                'success': has_feature_showcase and not has_task_failed and not has_continued_ai,
                'has_showcase': has_feature_showcase,
                'truncated': has_truncation,
                'task_failed': has_task_failed,
                'continued_ai': has_continued_ai,
                'output_length': len(stdout)
            })
            
        except subprocess.TimeoutExpired:
            print(f"Test timed out for query: {query}")
            results.append({
                'query': query,
                'success': False,
                'error': 'timeout'
            })
        except Exception as e:
            print(f"Test failed for query '{query}': {e}")
            results.append({
                'query': query,
                'success': False,
                'error': str(e)
            })
    
    return results

def main():
    """Run all comprehensive tests"""
    print("🧪 COMPREHENSIVE FEATURE DETECTION TEST SUITE")
    print("=" * 60)
    
    results = TestResults()
    
    # Test 1: Direct feature detection
    results.feature_detection_works = test_feature_detection_direct()
    
    # Test 2: TaskTracker showcase generation  
    results.showcase_message_complete = test_task_tracker_showcase_message()
    
    # Test 3: ChatEngine routing
    chat_routing_works = test_chat_engine_routing()
    
    # Test 4: Console formatter Rich display
    results.rich_formatting_works = test_console_formatter_rich_display()
    
    # Test 5: End-to-end CLI scenarios
    e2e_results = test_end_to_end_user_scenarios()
    
    # Analyze end-to-end results
    all_e2e_passed = all(r.get('success', False) for r in e2e_results)
    any_truncation = any(r.get('truncated', False) for r in e2e_results)
    any_task_failed = any(r.get('task_failed', False) for r in e2e_results)
    any_continued_ai = any(r.get('continued_ai', False) for r in e2e_results)
    
    results.no_task_execution_failed = not any_task_failed
    results.system_stops_cleanly = not any_continued_ai
    results.truncation_detected = any_truncation
    
    # Print comprehensive results
    print("\n" + "=" * 60)
    print("🎯 COMPREHENSIVE TEST RESULTS")
    print("=" * 60)
    
    print(f"✅ Feature detection works: {results.feature_detection_works}")
    print(f"✅ Showcase message complete: {results.showcase_message_complete}")
    print(f"✅ ChatEngine routing works: {chat_routing_works}")
    print(f"✅ Rich formatting works: {results.rich_formatting_works}")
    print(f"✅ No 'Task execution failed': {results.no_task_execution_failed}")
    print(f"✅ System stops cleanly: {results.system_stops_cleanly}")
    print(f"⚠️  No truncation detected: {not results.truncation_detected}")
    
    print(f"\n📊 End-to-End Results:")
    for result in e2e_results:
        status = "✅" if result.get('success', False) else "❌"
        print(f"{status} {result['query']}: Success={result.get('success', False)}")
        if not result.get('success', False):
            if result.get('truncated'):
                print(f"   ⚠️  Output truncated (length: {result.get('output_length', 0)})")
            if result.get('task_failed'):
                print(f"   ❌ 'Task execution failed' detected")
            if result.get('continued_ai'):
                print(f"   ❌ System continued with AI response")
    
    # Overall assessment
    all_core_tests_pass = (results.feature_detection_works and 
                          results.showcase_message_complete and 
                          chat_routing_works and 
                          results.rich_formatting_works)
    
    all_issues_fixed = (results.no_task_execution_failed and 
                       results.system_stops_cleanly and 
                       not results.truncation_detected)
    
    print(f"\n🎯 FINAL ASSESSMENT:")
    print(f"Core functionality: {'✅ WORKING' if all_core_tests_pass else '❌ BROKEN'}")
    print(f"Reported issues: {'✅ FIXED' if all_issues_fixed else '❌ STILL PRESENT'}")
    print(f"End-to-end flow: {'✅ WORKING' if all_e2e_passed else '❌ BROKEN'}")
    
    if not all_core_tests_pass:
        print("\n🔧 CORE FUNCTIONALITY ISSUES DETECTED")
        print("The basic feature detection and showcase system has problems.")
        
    if not all_issues_fixed:
        print("\n🚨 USER-REPORTED ISSUES STILL PRESENT")
        if results.truncation_detected:
            print("   • Feature showcase messages are still truncated")
        if not results.no_task_execution_failed:
            print("   • 'Task execution failed' messages still appear")
        if not results.system_stops_cleanly:
            print("   • System still continues with AI response after detection")
    
    if all_core_tests_pass and all_issues_fixed:
        print("\n🎉 ALL TESTS PASSED - SYSTEM IS WORKING CORRECTLY!")
    
    return all_core_tests_pass and all_issues_fixed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)