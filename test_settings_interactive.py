#!/usr/bin/env python3
"""
Test the settings interactively using the fallback mode.
"""

import sys
from src.agentsmcp.ui.modern_settings_ui import ModernSettingsUI
from src.agentsmcp.ui.theme_manager import ThemeManager

def test_settings_with_fallback():
    """Test settings using keyboard fallback mode"""
    print("🧪 Testing Settings with Keyboard Fallback Mode")
    print("=" * 60)
    
    theme_manager = ThemeManager()
    settings_ui = ModernSettingsUI(theme_manager)
    
    print(f"Keyboard interactive: {settings_ui.keyboard.is_interactive}")
    print(f"Available providers: {list(settings_ui.providers.keys())}")
    print("\nInstructions for fallback mode:")
    print("- Type 'up' or 'u' to move up")
    print("- Type 'down' or 'd' to move down") 
    print("- Press ENTER (empty input) to select")
    print("- Type 'q' to quit")
    print()
    
    try:
        # Test the provider selection
        print("🎯 Starting provider selection...")
        result = settings_ui._select_provider()
        
        if result:
            print(f"✅ Provider selected: {settings_ui.current_settings['provider']}")
            
            # Test model selection
            print("\n🎯 Starting model selection...")
            model_result = settings_ui._select_model()
            
            if model_result:
                print(f"✅ Model selected: {settings_ui.current_settings['model']}")
            else:
                print("❌ Model selection cancelled")
        else:
            print("❌ Provider selection cancelled")
            
    except Exception as e:
        print(f"❌ Settings test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    test_settings_with_fallback()