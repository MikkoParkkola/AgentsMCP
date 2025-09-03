#!/usr/bin/env python3
"""
SIMPLE INPUT PANEL TEST
Test if the _create_input_panel method is working correctly.
"""

import sys
import os
sys.path.insert(0, '/Users/mikko/github/AgentsMCP/src')

def test_input_panel_creation():
    """Test the input panel creation directly."""
    print("🔍 Testing input panel creation...")
    
    try:
        # Import TUI components
        from agentsmcp.ui.cli_app import CLIConfig
        from agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface
        
        # Create a simple config
        class TestConfig:
            debug_mode = True
            verbose = True
        
        config = TestConfig()
        
        # Create TUI instance
        print("✅ Creating TUI instance...")
        tui = RevolutionaryTUIInterface(cli_config=config)
        
        # Test state initialization
        print(f"✅ TUI state exists: {hasattr(tui, 'state')}")
        if hasattr(tui, 'state'):
            print(f"✅ Current input initialized: '{tui.state.current_input}'")
        
        # Test input panel method exists
        print(f"✅ _create_input_panel method exists: {hasattr(tui, '_create_input_panel')}")
        
        # Test setting input
        tui.state.current_input = "test typing"
        print(f"✅ Set test input: '{tui.state.current_input}'")
        
        # Test creating input panel
        print("🔍 Testing _create_input_panel()...")
        try:
            panel_content = tui._create_input_panel()
            print(f"✅ Input panel created successfully")
            print(f"📊 Panel content type: {type(panel_content)}")
            print(f"📊 Panel content preview: {str(panel_content)[:100]}...")
            
            # Check if input text appears in panel
            panel_str = str(panel_content)
            if "test typing" in panel_str:
                print("✅ Input text FOUND in panel content!")
            else:
                print("❌ Input text NOT found in panel content!")
                print(f"🔍 Full panel content: {panel_str}")
                
        except Exception as panel_e:
            print(f"❌ _create_input_panel() failed: {panel_e}")
            import traceback
            traceback.print_exc()
        
        # Test the _sync_refresh_display method
        print("\\n🔍 Testing _sync_refresh_display()...")
        try:
            # This should update the input panel and potentially refresh
            tui._sync_refresh_display()
            print("✅ _sync_refresh_display() completed without error")
        except Exception as refresh_e:
            print(f"❌ _sync_refresh_display() failed: {refresh_e}")
            import traceback
            traceback.print_exc()
        
        print("\\n🎯 Basic functionality test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_input_panel_creation()
    print("\\n🎯 Run this test to see if the core input panel logic works!")
    print("If the input text is found in panel content, the issue is elsewhere.")
    print("If not, there's a problem with the panel creation logic.")