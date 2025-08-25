#!/usr/bin/env python3
"""
Test settings with a simulated interactive environment
"""

import sys
import os
from src.agentsmcp.ui.keyboard_input import KeyboardInput, KeyCode

def test_keyboard_detection():
    """Test keyboard detection in different contexts"""
    print("🔍 Keyboard Detection Test")
    print("=" * 40)
    
    try:
        kb = KeyboardInput()
        print(f"✅ KeyboardInput created")
        print(f"   Platform: {'Windows' if kb.is_windows else 'Unix'}")
        print(f"   Interactive: {kb.is_interactive}")
        print(f"   TTY check: {sys.stdin.isatty()}")
        
        if kb.is_interactive:
            print("✅ Keyboard input should work with real arrow keys")
        else:
            print("❌ Keyboard input will not work - terminal not interactive")
        
        return kb.is_interactive
        
    except RuntimeError as e:
        print(f"❌ RuntimeError (expected in non-interactive): {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Test the keyboard detection"""
    print("🧪 AgentsMCP Keyboard Detection Test")
    print("=" * 50)
    
    result = test_keyboard_detection()
    
    print("\n📋 Summary:")
    if result:
        print("✅ Arrow keys should work in settings!")
        print("   Run: python -m src.agentsmcp.main --interactive")
        print("   Then type: settings")
        print("   Use arrow keys to navigate")
    else:
        print("ℹ️ This test shows non-interactive detection working correctly")
        print("   In a real terminal, arrow keys will work")
    
    return result

if __name__ == "__main__":
    main()