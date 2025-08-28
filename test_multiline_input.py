#!/usr/bin/env python3
"""Quick test script to demonstrate multi-line input functionality."""

import sys
import os
sys.path.insert(0, 'src')

def test_multiline_input():
    """Test the multi-line input system"""
    from agentsmcp.ui.command_interface import CommandInterface
    from agentsmcp.ui.theme_manager import ThemeManager
    
    # Mock orchestration manager for testing
    class MockOrchestrationManager:
        def __init__(self):
            self.is_running = True
    
    # Create components
    theme_manager = ThemeManager()
    orchestration_manager = MockOrchestrationManager()
    command_interface = CommandInterface(orchestration_manager, theme_manager)
    
    print("🧪 Multi-line Input Test")
    print("=" * 50)
    print("Features being tested:")
    print("✅ Multi-line input with prompt_toolkit")
    print("✅ Shift+Enter for new lines")
    print("✅ Enter to submit")
    print("✅ Copy-paste support with line breaks")
    print("✅ Local timezone handling")
    print("✅ Command completion")
    print()
    print("Try pasting this multi-line text:")
    print('''This is line 1
This is line 2
And this is line 3 with more text''')
    print()
    print("Or try typing and using Shift+Enter to create new lines.")
    print("Type 'exit' to quit.")
    print("-" * 50)
    
    while True:
        try:
            # Test the multi-line input
            user_input = command_interface._get_input_with_autocomplete("🎼 test ▶ ")
            
            if user_input.lower() in ['exit', 'quit']:
                break
                
            # Show what we captured
            print(f"\n📝 Captured input ({len(user_input)} chars):")
            print("─" * 30)
            print(repr(user_input))  # Show with escape characters
            print("─" * 30)
            print("Formatted output:")
            print(user_input)
            print("─" * 30)
            print(f"Lines: {len(user_input.split(chr(10)))}")
            print()
            
        except (KeyboardInterrupt, EOFError):
            break
    
    print("\n👋 Multi-line input test completed!")

if __name__ == "__main__":
    test_multiline_input()