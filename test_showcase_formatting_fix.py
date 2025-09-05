#!/usr/bin/env python3
"""
Critical Test: Verify ConsoleMessageFormatter Feature Showcase Formatting

This test focuses on the root cause identified:
- ConsoleMessageFormatter.format_feature_showcase() method exists
- But the main message formatting flow uses format_and_display_message() instead of format_message()
- This causes FEATURE_SHOWCASE_FORMAT: messages to be processed correctly in ChatEngine
- But never reach the Rich Panel formatting in ConsoleMessageFormatter
"""

import sys
import os
import io
from rich.console import Console

# Add the src directory to path
sys.path.insert(0, '/Users/mikko/github/AgentsMCP/src')

from agentsmcp.ui.v3.console_message_formatter import ConsoleMessageFormatter

def test_console_formatter_methods():
    """Test what methods actually exist on ConsoleMessageFormatter"""
    print("=== CONSOLE FORMATTER METHOD ANALYSIS ===")
    
    # Create a test console that captures output
    output_buffer = io.StringIO()
    test_console = Console(file=output_buffer, force_terminal=True, width=80)
    
    formatter = ConsoleMessageFormatter(console=test_console)
    
    print(f"Available methods:")
    methods = [method for method in dir(formatter) if not method.startswith('_')]
    for method in methods:
        print(f"  - {method}")
    
    print(f"\nKey method checks:")
    print(f"  - Has format_message(): {hasattr(formatter, 'format_message')}")
    print(f"  - Has format_and_display_message(): {hasattr(formatter, 'format_and_display_message')}")
    print(f"  - Has format_feature_showcase(): {hasattr(formatter, 'format_feature_showcase')}")
    
    return formatter, output_buffer

def test_feature_showcase_direct():
    """Test the format_feature_showcase method directly"""
    print("\n=== DIRECT FEATURE SHOWCASE TEST ===")
    
    formatter, output_buffer = test_console_formatter_methods()
    
    if hasattr(formatter, 'format_feature_showcase'):
        test_showcase = """# TUI Command Available

The TUI interface is already implemented and ready to use!

## Quick Start
```bash
./agentsmcp tui
```

## Features
- **Interactive Chat**: Real-time conversation with AI
- **Rich Formatting**: Beautiful text rendering with colors and panels
- **Task Progress**: Visual progress indicators for long-running tasks
- **Command History**: Browse previous commands and responses

## Usage Tips
- Press `Ctrl+C` to exit gracefully
- Use `/help` for available commands
- Try `/status` to see system information

Ready to get started? Just run the command above!"""

        try:
            print("Calling format_feature_showcase() directly...")
            formatter.format_feature_showcase(test_showcase)
            
            output = output_buffer.getvalue()
            print(f"Generated output length: {len(output)} characters")
            print(f"Has panel borders: {'╭' in output and '╰' in output}")
            print(f"Has feature title: {'Feature Already Available' in output}")
            print(f"Has TUI content: {'TUI Command Available' in output}")
            print(f"Has code blocks: {'```bash' in output}")
            
            print(f"\nFirst 500 chars of output:")
            print(repr(output[:500]))
            
            return len(output) > 200 and '╭' in output
            
        except Exception as e:
            print(f"Error calling format_feature_showcase: {e}")
            return False
    else:
        print("format_feature_showcase method not found!")
        return False

def test_system_message_flow():
    """Test how FEATURE_SHOWCASE_FORMAT: messages should be processed"""
    print("\n=== SYSTEM MESSAGE FLOW TEST ===")
    
    formatter, output_buffer = test_console_formatter_methods()
    
    # Test the actual flow that happens in the real system
    test_content = "FEATURE_SHOWCASE_FORMAT:# TUI Available\n\nUse: `./agentsmcp tui`"
    
    try:
        if hasattr(formatter, 'format_and_display_message'):
            print("Testing format_and_display_message() with FEATURE_SHOWCASE_FORMAT...")
            formatter.format_and_display_message("system", test_content)
            
            output = output_buffer.getvalue()
            print(f"Output length: {len(output)}")
            print(f"Has Rich panel: {'╭' in output}")
            print(f"Raw output: {repr(output[:200])}")
            
            return '╭' in output
        else:
            print("format_and_display_message method not found!")
            return False
            
    except Exception as e:
        print(f"Error in system message flow test: {e}")
        return False

def main():
    print("🔍 CRITICAL ISSUE ANALYSIS: Feature Showcase Formatting")
    print("=" * 70)
    
    # Test 1: Check available methods
    formatter_works = test_console_formatter_methods()
    
    # Test 2: Test format_feature_showcase directly
    showcase_works = test_feature_showcase_direct()
    
    # Test 3: Test the actual message flow
    flow_works = test_system_message_flow()
    
    print("\n" + "=" * 70)
    print("🎯 ROOT CAUSE ANALYSIS")
    print("=" * 70)
    
    print(f"✅ ConsoleMessageFormatter exists: {formatter_works is not None}")
    print(f"✅ format_feature_showcase() method works: {showcase_works}")
    print(f"❓ System message flow triggers Rich formatting: {flow_works}")
    
    if showcase_works and not flow_works:
        print("\n🚨 CRITICAL FINDING:")
        print("The format_feature_showcase() method works perfectly when called directly.")
        print("But the system message flow (format_and_display_message) doesn't trigger it!")
        print("\nROOT CAUSE: The ConsoleMessageFormatter.format_and_display_message() method")
        print("needs to check for 'FEATURE_SHOWCASE_FORMAT:' prefix and call format_feature_showcase().")
        print("\nThis explains the user's truncation issue:")
        print("- ChatEngine correctly creates FEATURE_SHOWCASE_FORMAT: messages ✅") 
        print("- But ConsoleMessageFormatter treats them as regular system messages ❌")
        print("- So Rich Panel formatting never happens, causing truncation ❌")
        
        return False
    elif showcase_works and flow_works:
        print("\n🎉 SYSTEM WORKING CORRECTLY:")
        print("Both direct showcase formatting AND system message flow work properly.")
        return True
    else:
        print("\n🚨 MULTIPLE ISSUES DETECTED:")
        print("Both the direct method and system flow have problems.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)