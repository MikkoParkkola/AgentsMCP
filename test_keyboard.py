#!/usr/bin/env python3
"""
Test keyboard input functionality for settings.
"""

import sys
from src.agentsmcp.ui.keyboard_input import KeyboardInput, KeyCode

def test_keyboard_basic():
    """Test if KeyboardInput can be created"""
    try:
        keyboard = KeyboardInput()
        print("✅ KeyboardInput created successfully")
        print(f"   Platform: {'Windows' if keyboard.is_windows else 'Unix'}")
        print(f"   Interactive: {keyboard.is_interactive}")
        return True
    except Exception as e:
        print(f"❌ KeyboardInput creation failed: {e}")
        return False

def test_keyboard_interactive():
    """Test interactive keyboard input"""
    print("\n🧪 Testing Interactive Keyboard Input")
    print("Press arrow keys, ENTER to confirm, or ESC to exit...")
    
    try:
        keyboard = KeyboardInput()
        
        while True:
            try:
                key_code, char = keyboard.get_key()
                
                if key_code == KeyCode.UP:
                    print("↑ UP arrow pressed")
                elif key_code == KeyCode.DOWN:
                    print("↓ DOWN arrow pressed") 
                elif key_code == KeyCode.LEFT:
                    print("← LEFT arrow pressed")
                elif key_code == KeyCode.RIGHT:
                    print("→ RIGHT arrow pressed")
                elif key_code == KeyCode.ENTER:
                    print("✅ ENTER pressed - exiting")
                    break
                elif key_code == KeyCode.ESCAPE:
                    print("🚫 ESCAPE pressed - exiting")
                    break
                elif char:
                    print(f"📝 Character: '{char}'")
                else:
                    print(f"🔍 Key code: {key_code}")
                    
            except KeyboardInterrupt:
                print("\n🛑 Interrupted by user")
                break
                
        return True
        
    except Exception as e:
        print(f"❌ Interactive keyboard test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Keyboard Input System")
    print("=" * 40)
    
    success = test_keyboard_basic()
    
    if success and sys.stdin.isatty():
        try:
            test_keyboard_interactive()
        except KeyboardInterrupt:
            print("\nTest interrupted")
    else:
        print("⚠️ Skipping interactive test (not in terminal)")
    
    print("🎯 Keyboard test completed")