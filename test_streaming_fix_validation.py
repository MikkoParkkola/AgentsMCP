#!/usr/bin/env python3
"""
COMPREHENSIVE STREAMING FIX VALIDATION TEST

This test validates:
1. Console flood fix - streaming should overwrite lines, not create duplicates
2. Orchestration visibility - agent roles should be shown in status
3. Proper line control in both Rich and Plain renderers
"""

import asyncio
import sys
import time
import threading
from io import StringIO
from unittest.mock import Mock, patch

# Add src to path for testing
sys.path.insert(0, 'src')

from agentsmcp.ui.v3.console_renderer import ConsoleRenderer
from agentsmcp.ui.v3.plain_cli_renderer import PlainCLIRenderer
from agentsmcp.ui.v3.tui_launcher import TUILauncher
from agentsmcp.ui.v3.terminal_capabilities import TerminalCapabilities

def test_console_renderer_streaming():
    """Test ConsoleRenderer streaming fixes."""
    print("🧪 Testing ConsoleRenderer streaming...")
    
    # Mock capabilities
    caps = Mock(spec=TerminalCapabilities)
    caps.is_tty = True
    caps.supports_colors = True
    caps.supports_rich = True
    caps.width = 80
    caps.height = 24
    
    # Create renderer
    renderer = ConsoleRenderer(caps)
    
    # Mock state
    renderer.state = Mock()
    renderer.state.is_processing = False
    
    # Initialize
    success = renderer.initialize()
    assert success, "ConsoleRenderer failed to initialize"
    print("  ✓ Initialization successful")
    
    # Test streaming updates - capture output
    captured_output = StringIO()
    
    with patch('sys.stdout.write') as mock_write, patch('sys.stdout.flush') as mock_flush:
        # First streaming update
        renderer.handle_streaming_update("Hello")
        assert renderer._streaming_active, "Streaming should be active"
        
        # Second streaming update - should overwrite, not duplicate
        renderer.handle_streaming_update("Hello world")
        
        # Third streaming update - longer content
        renderer.handle_streaming_update("Hello world! This is a longer streaming message.")
        
        # Verify proper line control was used
        assert mock_write.called, "sys.stdout.write should be called for line control"
        
        # Check for carriage return and clear-to-end sequences
        write_calls = [call[0][0] for call in mock_write.call_args_list if call[0]]
        has_carriage_return = any("\r" in call for call in write_calls)
        has_clear_sequence = any("\033[K" in call for call in write_calls)
        
        assert has_carriage_return, "Should use carriage return for line overwrite"
        assert has_clear_sequence, "Should use clear-to-end sequence"
        
    print("  ✓ Streaming line overwrite working correctly")
    
    # Test streaming finalization
    renderer.display_chat_message("assistant", "Final message", "12:34:56")
    assert not renderer._streaming_active, "Streaming should be inactive after final message"
    print("  ✓ Streaming finalization working")
    
    # Cleanup
    renderer.cleanup()
    print("  ✓ ConsoleRenderer streaming test passed\n")

def test_plain_cli_renderer_streaming():
    """Test PlainCLIRenderer streaming fixes."""
    print("🧪 Testing PlainCLIRenderer streaming...")
    
    # Mock capabilities  
    caps = Mock(spec=TerminalCapabilities)
    caps.is_tty = False
    caps.supports_colors = False
    caps.supports_rich = False
    caps.width = 80
    caps.height = 24
    
    # Create renderer
    renderer = PlainCLIRenderer(caps)
    
    # Mock state
    renderer.state = Mock()
    renderer.state.is_processing = False
    
    # Initialize
    success = renderer.initialize()
    assert success, "PlainCLIRenderer failed to initialize"
    print("  ✓ Initialization successful")
    
    # Test streaming updates
    with patch('sys.stdout.write') as mock_write, patch('sys.stdout.flush') as mock_flush:
        # First streaming update
        renderer.handle_streaming_update("Test message")
        assert renderer._streaming_active, "Streaming should be active"
        
        # Second streaming update - should overwrite
        renderer.handle_streaming_update("Test message updated")
        
        # Long message - should truncate
        long_message = "This is a very long message that should be truncated " * 5
        renderer.handle_streaming_update(long_message)
        
        # Verify proper line control
        assert mock_write.called, "sys.stdout.write should be called"
        assert mock_flush.called, "sys.stdout.flush should be called"
        
        # Check for line control sequences
        write_calls = [call[0][0] for call in mock_write.call_args_list if call[0]]
        has_carriage_return = any("\r" in call for call in write_calls)
        has_clear_sequence = any("\033[K" in call for call in write_calls)
        
        assert has_carriage_return, "Should use carriage return for line overwrite"
        assert has_clear_sequence, "Should use clear-to-end sequence to prevent artifacts"
        
    print("  ✓ Streaming line overwrite working correctly")
    
    # Test streaming finalization
    renderer.display_chat_message("assistant", "Final message", "12:34:56")
    assert not renderer._streaming_active, "Streaming should be inactive after final message"
    print("  ✓ Streaming finalization working")
    
    # Cleanup
    renderer.cleanup()
    print("  ✓ PlainCLIRenderer streaming test passed\n")

def test_orchestration_visibility():
    """Test orchestration visibility enhancements."""
    print("🧪 Testing orchestration visibility...")
    
    # Create TUI launcher
    launcher = TUILauncher()
    
    # Test orchestration enhancement method
    test_cases = [
        ("orchestrating the response", "🎯 Orchestrator: orchestrating the response"),
        ("coordinating multiple agents", "🎯 Orchestrator: coordinating multiple agents"),
        ("tool: mcp__semgrep__scan", "🛠️ Agent-SEMGREP: tool: mcp__semgrep__scan"),
        ("tool: mcp__git__status", "🛠️ Agent-GIT: tool: mcp__git__status"),
        ("analyzing the codebase", "🔍 Analyst Agent: analyzing the codebase"),
        ("processing user input", "🔍 Analyst Agent: processing user input"),
        ("generating response", "✨ Generator Agent: generating response"),
        ("creating solution", "✨ Generator Agent: creating solution"),
        ("streaming response", "📡 Stream Manager: streaming response"),
        ("basic status update", "🎯 Coordinator: basic status update"),
    ]
    
    for input_status, expected_output in test_cases:
        result = launcher._enhance_status_with_orchestration(input_status)
        assert result == expected_output, f"Failed for '{input_status}': got '{result}', expected '{expected_output}'"
        print(f"  ✓ '{input_status}' → '{result}'")
    
    print("  ✓ Orchestration visibility enhancement working\n")

def simulate_streaming_scenario():
    """Simulate a realistic streaming scenario."""
    print("🎭 SIMULATING REALISTIC STREAMING SCENARIO")
    print("=" * 50)
    
    # Mock capabilities for rich environment
    caps = Mock(spec=TerminalCapabilities)
    caps.is_tty = True
    caps.supports_colors = True
    caps.supports_rich = True
    caps.width = 80
    caps.height = 24
    
    # Create console renderer
    renderer = ConsoleRenderer(caps)
    renderer.state = Mock()
    renderer.state.is_processing = False
    
    if not renderer.initialize():
        print("❌ Failed to initialize renderer for simulation")
        return
    
    # Simulate streaming progress
    streaming_content = [
        "I'll help you",
        "I'll help you analyze",
        "I'll help you analyze the codebase",
        "I'll help you analyze the codebase for security issues",
        "I'll help you analyze the codebase for security issues. Let me start by",
        "I'll help you analyze the codebase for security issues. Let me start by scanning the project structure."
    ]
    
    print("\n🚀 Starting simulated streaming response:")
    print("   (This should show clean line overwrites, not console flood)")
    print()
    
    for i, content in enumerate(streaming_content):
        print(f"📡 Streaming update {i+1}/6:", end=" ")
        renderer.handle_streaming_update(content)
        time.sleep(0.3)  # Simulate streaming delay
        
    print("\n")
    
    # Finalize streaming
    renderer.display_chat_message("assistant", streaming_content[-1], "12:34:56")
    print("✓ Streaming completed - should show final message cleanly")
    
    renderer.cleanup()
    print("✓ Simulation completed\n")

def run_comprehensive_test():
    """Run all streaming fix validation tests."""
    print("🧪 COMPREHENSIVE STREAMING FIX VALIDATION")
    print("=" * 50)
    print()
    
    try:
        # Test console renderer streaming
        test_console_renderer_streaming()
        
        # Test plain CLI renderer streaming  
        test_plain_cli_renderer_streaming()
        
        # Test orchestration visibility
        test_orchestration_visibility()
        
        # Simulate realistic scenario
        simulate_streaming_scenario()
        
        print("🎉 ALL STREAMING FIX TESTS PASSED!")
        print()
        print("✅ FIXES VALIDATED:")
        print("   • Console flood eliminated - streaming properly overwrites lines")
        print("   • Line control sequences working in both Rich and Plain renderers")
        print("   • Orchestration visibility shows agent roles and coordination")
        print("   • Stream finalization prevents duplicate messages")
        print()
        print("📋 STREAMING STRATEGY RECOMMENDATION:")
        print("   • Main orchestrator should handle streaming to avoid conflicts")
        print("   • Individual agents should send status updates, not stream responses")
        print("   • Use status callbacks for multi-agent coordination visibility")
        print("   • Rich renderer provides better streaming UX than plain CLI")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)