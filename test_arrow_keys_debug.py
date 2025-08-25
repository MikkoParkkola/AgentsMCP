#!/usr/bin/env python3
"""
Debug arrow key behavior in the settings dialog
"""

import sys
import os
from src.agentsmcp.ui.keyboard_input import KeyboardInput, KeyCode

def debug_keyboard_input():
    """Debug keyboard input behavior"""
    print("🐛 Debugging Keyboard Input")
    print("=" * 50)
    
    keyboard = KeyboardInput()
    
    print(f"Platform: {sys.platform}")
    print(f"Is interactive: {keyboard.is_interactive}")
    print(f"sys.stdin.isatty(): {sys.stdin.isatty()}")
    print(f"sys.stdout.isatty(): {sys.stdout.isatty()}")
    print()
    
    if not keyboard.is_interactive:
        print("❌ Not in interactive mode - arrow keys will fallback to text input")
        print("This explains why arrow keys don't work properly!")
        print()
        print("In fallback mode:")
        print("- Type 'up' or 'u' for up arrow")
        print("- Type 'down' or 'd' for down arrow")
        print("- Press Enter (empty input) to select")
        print("- Type 'q' to quit")
        print()
    
    print("Press arrow keys or type commands. Press 'q' or Ctrl+C to exit:")
    print()
    
    count = 0
    while count < 10:
        try:
            key_code, char = keyboard.get_key()
            
            if key_code:
                print(f"Key code detected: {key_code}")
                if key_code == KeyCode.ESCAPE:
                    print("Escape pressed - this would cancel the dialog")
                    break
                elif key_code == KeyCode.UP:
                    print("Up arrow pressed - this should move selection up")
                elif key_code == KeyCode.DOWN:
                    print("Down arrow pressed - this should move selection down")
                elif key_code == KeyCode.ENTER:
                    print("Enter pressed - this should select current option")
            
            if char:
                print(f"Character detected: '{char}'")
                if char.lower() == 'q':
                    print("Q pressed - exiting")
                    break
            
            if key_code is None and char is None:
                print("No input received (timeout or EOF)")
            
            count += 1
            
        except KeyboardInterrupt:
            print("\nCtrl+C pressed - exiting")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            break

if __name__ == "__main__":
    debug_keyboard_input()