#!/usr/bin/env python3
"""
Test to reproduce and fix the arrow key cancellation issue
"""

import sys
import os
from src.agentsmcp.ui.keyboard_input import KeyboardInput, KeyCode
from src.agentsmcp.ui.modern_settings_ui import ModernSettingsUI
from src.agentsmcp.ui.theme_manager import ThemeManager

def test_keyboard_interactive_fix():
    """Test with forced interactive mode"""
    print("🔧 Testing Keyboard Input Fix")
    print("=" * 50)
    
    keyboard = KeyboardInput()
    
    print(f"Original interactive detection: {keyboard.is_interactive}")
    
    # Force interactive mode for testing
    # In a real fix, we might want to be more careful about this detection
    original_is_interactive = keyboard.is_interactive
    keyboard.is_interactive = True  # Force enable for testing
    
    print(f"Forced interactive mode: {keyboard.is_interactive}")
    print()
    print("Now testing with forced interactive mode...")
    print("Try pressing arrow keys - they should work now!")
    print("Press 'q' to quit")
    print()
    
    count = 0
    while count < 5:
        try:
            key_code, char = keyboard.get_key()
            
            if key_code:
                print(f"✅ Key code detected: {key_code}")
                if key_code == KeyCode.ESCAPE:
                    print("  -> Escape pressed")
                    break
                elif key_code == KeyCode.UP:
                    print("  -> Up arrow pressed - navigation should work!")
                elif key_code == KeyCode.DOWN:
                    print("  -> Down arrow pressed - navigation should work!")
                elif key_code == KeyCode.ENTER:
                    print("  -> Enter pressed")
            
            if char:
                print(f"✅ Character detected: '{char}'")
                if char.lower() == 'q':
                    print("  -> Q pressed - exiting")
                    break
            
            count += 1
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Restore original setting
    keyboard.is_interactive = original_is_interactive
    
    return True

def test_settings_with_better_detection():
    """Test settings with improved terminal detection"""
    print("\n🔧 Testing Settings with Better Terminal Detection")
    print("=" * 60)
    
    theme_manager = ThemeManager()
    settings_ui = ModernSettingsUI(theme_manager)
    
    # Improve the terminal detection
    keyboard = settings_ui.keyboard
    
    # Check if we can access terminal even if isatty() returns False
    # This often happens in containerized environments or certain CLIs
    can_access_terminal = False
    try:
        import termios
        import tty
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        can_access_terminal = True
        print(f"✅ Can access terminal attributes (fd={fd})")
    except Exception as e:
        print(f"❌ Cannot access terminal: {e}")
    
    # Environment checks
    has_term_env = 'TERM' in os.environ and os.environ['TERM'] != 'dumb'
    print(f"Terminal environment: TERM={os.environ.get('TERM', 'not set')}")
    print(f"Has proper TERM env: {has_term_env}")
    
    # Override detection logic
    if can_access_terminal and has_term_env:
        keyboard.is_interactive = True
        print("🔧 Forcing interactive mode due to terminal access capability")
    else:
        print("⚠️  Staying in fallback mode")
    
    print(f"Final interactive state: {keyboard.is_interactive}")
    print()
    
    if keyboard.is_interactive:
        print("✅ Interactive mode enabled - arrow keys should work!")
        print("   Try running the settings now")
    else:
        print("⚠️  Fallback mode - will need to use text commands")
        print("   up/down/enter/q commands")
    
    return keyboard.is_interactive

if __name__ == "__main__":
    print("Testing keyboard input fixes...\n")
    
    # Test 1: Force interactive mode
    test_keyboard_interactive_fix()
    
    # Test 2: Better detection
    interactive = test_settings_with_better_detection()
    
    print(f"\n🎯 Result: Interactive mode {'enabled' if interactive else 'disabled'}")
    print("The issue is that terminal detection is too strict.")
    print("Solution: Improve terminal capability detection logic.")