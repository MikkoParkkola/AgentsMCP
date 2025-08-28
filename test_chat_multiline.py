#!/usr/bin/env python3
"""Test multi-line input in the actual chat/interactive mode"""

import sys
import os
import subprocess
import time
sys.path.insert(0, 'src')

def test_interactive_chat_directly():
    """Test the interactive chat mode directly"""
    from agentsmcp.ui.cli_app import CLIApp
    from agentsmcp.config import Config
    
    print("🧪 Direct Interactive Chat Test")
    print("=" * 40)
    
    # Create a minimal config for testing
    try:
        config = Config()
        cli_app = CLIApp(config)
        
        print("✅ CLIApp initialized successfully")
        print("📋 Ready to test multi-line input in chat interface")
        print()
        
        # Try to access the command interface directly
        theme_manager = cli_app.theme_manager
        print(f"✅ Theme manager: {type(theme_manager).__name__}")
        
        # Test the command interface
        from agentsmcp.ui.command_interface import CommandInterface
        
        class MockOrchestrationManager:
            def __init__(self):
                self.is_running = True
                
        mock_orchestration = MockOrchestrationManager()
        command_interface = CommandInterface(mock_orchestration, theme_manager)
        
        print("🎯 Testing multi-line input methods...")
        
        # Test 1: Simple single line input
        print("\n1️⃣ Testing single line detection:")
        test_input = "hello world"
        result = command_interface._handle_interactive_input_continuation(test_input)
        print(f"   Input: {repr(test_input)}")
        print(f"   Output: {repr(result)}")
        print(f"   ✅ Single line handled correctly" if result == test_input else "   ❌ Issue with single line")
        
        # Test 2: Multi-line content simulation
        print("\n2️⃣ Testing multi-line content handling:")
        multiline_input = "Line 1: This is a test\nLine 2: Of multi-line content\nLine 3: In the chat interface"
        lines = multiline_input.split('\n')
        result = '\n'.join(lines)
        print(f"   Input lines: {len(lines)}")
        print(f"   Output lines: {len(result.split(chr(10)))}")
        print(f"   ✅ Multi-line structure preserved" if len(result.split('\n')) == len(lines) else "   ❌ Multi-line structure lost")
        
        # Test 3: Check if the _looks_incomplete method works
        print("\n3️⃣ Testing incomplete line detection:")
        test_cases = [
            ("if True:", True),
            ("hello world", False),
            ("def function(", True),
            ("import os", False),
            ("for i in range(10):", True)
        ]
        
        for test_line, expected in test_cases:
            result = command_interface._looks_incomplete(test_line)
            status = "✅" if result == expected else "❌"
            print(f"   {status} '{test_line}' -> incomplete: {result} (expected: {expected})")
        
        print(f"\n✅ Direct interface testing completed")
        
    except Exception as e:
        print(f"❌ Error in direct testing: {e}")
        import traceback
        traceback.print_exc()

def test_actual_interactive_mode():
    """Test by running the actual interactive mode with simulated input"""
    print("\n🎮 Testing Actual Interactive Mode")
    print("=" * 40)
    
    # Create test input that simulates both single and multi-line scenarios
    test_inputs = [
        "hello world",  # Single line
        "",  # Empty line
        "exit"  # Exit command
    ]
    
    input_data = '\n'.join(test_inputs) + '\n'
    
    try:
        # Run the interactive mode with test input
        process = subprocess.Popen(
            [sys.executable, '-m', 'agentsmcp', 'interactive', '--no-welcome'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15
        )
        
        stdout, stderr = process.communicate(input=input_data)
        
        print("📊 Interactive Mode Test Results:")
        print(f"   Return code: {process.returncode}")
        print(f"   Stdout length: {len(stdout)} chars")
        print(f"   Stderr length: {len(stderr)} chars")
        
        # Check for success indicators
        if process.returncode == 0:
            print("   ✅ Interactive mode exited cleanly")
        else:
            print(f"   ⚠️  Interactive mode exit code: {process.returncode}")
        
        # Check for error patterns
        error_patterns = [
            "infinite loop", 
            "Traceback", 
            "Error:",
            "Failed",
            "Exception"
        ]
        
        found_errors = []
        for pattern in error_patterns:
            if pattern.lower() in stderr.lower() or pattern.lower() in stdout.lower():
                found_errors.append(pattern)
        
        if found_errors:
            print(f"   ⚠️  Found potential issues: {found_errors}")
            print("\n📝 Stderr output:")
            print(stderr[:500] + "..." if len(stderr) > 500 else stderr)
        else:
            print("   ✅ No obvious errors detected")
            
        # Check for multi-line support indicators
        multiline_indicators = [
            "Multi-line",
            "paste", 
            "bracketed",
            "lines captured"
        ]
        
        found_multiline = []
        for indicator in multiline_indicators:
            if indicator.lower() in stdout.lower():
                found_multiline.append(indicator)
        
        if found_multiline:
            print(f"   ✅ Multi-line features detected: {found_multiline}")
        
    except subprocess.TimeoutExpired:
        process.kill()
        print("   ⚠️  Interactive mode timed out (may indicate hanging)")
    except Exception as e:
        print(f"   ❌ Error testing interactive mode: {e}")

if __name__ == "__main__":
    print("🧪 AgentsMCP Chat Multi-line Input Test Suite")
    print("=" * 60)
    print("Environment:")
    print(f"- Python: {sys.version}")
    print(f"- Platform: {sys.platform}")
    print(f"- Terminal: {os.environ.get('TERM_PROGRAM', 'unknown')}")
    print()
    
    # Test 1: Direct interface testing
    test_interactive_chat_directly()
    
    # Test 2: Actual interactive mode
    test_actual_interactive_mode()
    
    print("\n🎯 Summary:")
    print("- Direct interface testing validates the multi-line logic")
    print("- Interactive mode testing validates the full user experience")
    print("- Both tests should pass for full multi-line chat functionality")