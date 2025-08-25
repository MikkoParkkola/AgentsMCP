#!/usr/bin/env python3
"""
Debug the settings arrow key navigation issue.
"""

import sys
from src.agentsmcp.ui.modern_settings_ui import ModernSettingsUI
from src.agentsmcp.ui.theme_manager import ThemeManager
from src.agentsmcp.ui.keyboard_input import KeyboardInput, KeyCode

def test_keyboard_input_directly():
    """Test keyboard input directly"""
    print("🧪 Testing Keyboard Input Directly")
    print("=" * 50)
    
    try:
        keyboard = KeyboardInput()
        print(f"✅ KeyboardInput created successfully")
        print(f"   Platform: {'Windows' if keyboard.is_windows else 'Unix'}")
        print(f"   Interactive: {keyboard.is_interactive}")
        
        if not keyboard.is_interactive:
            print("⚠️ Not in interactive terminal - this may be the issue!")
            return False
            
        print("\n🎯 Press arrow keys to test (ESC to exit):")
        
        while True:
            key_code, char = keyboard.get_key()
            
            if key_code == KeyCode.UP:
                print("↑ UP arrow detected")
            elif key_code == KeyCode.DOWN:
                print("↓ DOWN arrow detected") 
            elif key_code == KeyCode.LEFT:
                print("← LEFT arrow detected")
            elif key_code == KeyCode.RIGHT:
                print("→ RIGHT arrow detected")
            elif key_code == KeyCode.ENTER:
                print("✅ ENTER detected")
            elif key_code == KeyCode.ESCAPE:
                print("🚫 ESCAPE detected - exiting")
                break
            elif char:
                print(f"📝 Character: '{char}'")
                if char.lower() == 'q':
                    print("Exiting on 'q'")
                    break
            else:
                print(f"🔍 Unknown key code: {key_code}")
                
    except Exception as e:
        print(f"❌ Keyboard input test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_settings_step_by_step():
    """Test each step of settings dialog individually"""
    print("\n🧪 Testing Settings Step-by-Step")
    print("=" * 50)
    
    try:
        theme_manager = ThemeManager()
        settings_ui = ModernSettingsUI(theme_manager)
        
        print(f"✅ Settings UI created successfully")
        print(f"   Keyboard interactive: {settings_ui.keyboard.is_interactive}")
        print(f"   Current settings: {settings_ui.current_settings}")
        
        # Test just the provider selection
        print("\n🎯 Testing provider selection (press arrow keys, ENTER to select, ESC to exit):")
        print("   Available providers:", list(settings_ui.providers.keys()))
        
        result = settings_ui._select_provider()
        print(f"Provider selection result: {result}")
        
        if result:
            print(f"Selected provider: {settings_ui.current_settings['provider']}")
        
        return result
        
    except Exception as e:
        print(f"❌ Settings step-by-step test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run debug tests"""
    print("🐛 Settings Navigation Debug Test")
    print("=" * 60)
    
    if not sys.stdin.isatty():
        print("⚠️ Not running in interactive terminal!")
        print("   Try running directly: python test_settings_debug.py")
        return False
    
    # Test keyboard input first
    keyboard_works = test_keyboard_input_directly()
    
    if keyboard_works:
        # Test settings if keyboard works
        test_settings_step_by_step()
    else:
        print("❌ Keyboard input not working - settings won't work either")
    
    print("\n🎯 Debug test completed")

if __name__ == "__main__":
    main()