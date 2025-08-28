#!/usr/bin/env python3
"""Test script to verify Shift+Enter functionality."""

import sys
import os
sys.path.insert(0, 'src')

def test_shift_enter():
    """Test Shift+Enter detection"""
    from agentsmcp.ui.command_interface import CommandInterface
    from agentsmcp.ui.theme_manager import ThemeManager
    
    print("🧪 Testing Shift+Enter Multi-line Input")
    print("=" * 50)
    
    # Mock orchestration manager
    class MockOrchestrationManager:
        def __init__(self):
            self.is_running = True
    
    theme_manager = ThemeManager()
    orchestration_manager = MockOrchestrationManager()
    command_interface = CommandInterface(orchestration_manager, theme_manager)
    
    print("Instructions for manual testing:")
    print("1. Type some text")
    print("2. Try Shift+Enter - should add new line")
    print("3. Try Ctrl+J - should add new line")  
    print("4. Try Alt+Enter - should add new line")
    print("5. Try Enter on short text - should send")
    print("6. Try Enter on text ending with ':' - should add new line")
    print("7. Type 'exit' to quit")
    print()
    
    while True:
        try:
            # Test the multi-line input system
            user_input = command_interface._get_input_with_autocomplete("🧪 test ▶ ")
            
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
    
    print("\n👋 Shift+Enter test completed!")

if __name__ == "__main__":
    test_shift_enter()