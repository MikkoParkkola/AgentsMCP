#!/usr/bin/env python3
"""
LIVE DISPLAY REFRESH TEST
Test specifically how the Rich Live display handles our refresh strategy.
"""

import sys
import os
sys.path.insert(0, '/Users/mikko/github/AgentsMCP/src')

def test_live_display_refresh_behavior():
    """Test the Live display refresh behavior in isolation."""
    print("🔍 Testing Rich Live display refresh behavior...")
    
    try:
        from agentsmcp.ui.cli_app import CLIConfig
        from agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface
        
        # Create test config with debug
        class DebugConfig:
            debug_mode = True
            verbose = True
        
        config = DebugConfig()
        tui = RevolutionaryTUIInterface(cli_config=config)
        
        print("✅ TUI instance created")
        
        # Test the refresh tracking mechanism
        print("\\n🔍 Testing refresh tracking variables...")
        
        # Check if the tracking variable exists
        has_tracking = hasattr(tui, '_last_input_refresh_content')
        print(f"📊 Tracking variable exists: {has_tracking}")
        
        if not has_tracking:
            print("🔧 Initializing tracking variable...")
            tui._last_input_refresh_content = ""
        
        # Test the refresh condition logic
        print("\\n🔍 Testing refresh conditions...")
        
        # Simulate the condition check from _sync_refresh_display
        tui.state.current_input = "test input 1"
        
        # Mock the live_display object to test condition logic
        class MockLiveDisplay:
            def __init__(self):
                self.refresh_called = False
                
            def refresh(self):
                self.refresh_called = True
                print("🔄 Mock refresh() called!")
        
        tui.live_display = MockLiveDisplay()
        
        # Test condition 1: First refresh (no tracking variable)
        if not hasattr(tui, '_last_input_refresh_content'):
            tui._last_input_refresh_content = tui.state.current_input
            tui.live_display.refresh()
            print("✅ First refresh condition - would trigger refresh")
        
        # Test condition 2: Input changed
        old_input = tui._last_input_refresh_content
        tui.state.current_input = "test input 2"
        
        if (hasattr(tui.live_display, 'refresh') and 
            hasattr(tui, '_last_input_refresh_content') and
            tui._last_input_refresh_content != tui.state.current_input):
            
            tui.live_display.refresh()
            tui._last_input_refresh_content = tui.state.current_input
            print(f"✅ Input changed condition - refresh triggered: '{old_input}' -> '{tui.state.current_input}'")
        
        # Test condition 3: No change (should NOT refresh)
        refresh_before = tui.live_display.refresh_called
        if (hasattr(tui.live_display, 'refresh') and 
            hasattr(tui, '_last_input_refresh_content') and
            tui._last_input_refresh_content != tui.state.current_input):
            
            tui.live_display.refresh()
            print("❌ This should not execute - input hasn't changed")
        else:
            print("✅ No change condition - refresh correctly skipped")
        
        print(f"📊 Total mock refresh calls: {tui.live_display.refresh_called}")
        
        # Now test the real _sync_refresh_display method with mock
        print("\\n🔍 Testing real _sync_refresh_display with mock...")
        
        # Reset mock
        tui.live_display = MockLiveDisplay()
        tui.state.current_input = "real test input"
        
        # Remove old tracking to test fresh start
        if hasattr(tui, '_last_input_refresh_content'):
            delattr(tui, '_last_input_refresh_content')
        
        # Mock sys.stdin.isatty to return True
        import sys
        original_isatty = sys.stdin.isatty
        sys.stdin.isatty = lambda: True
        
        # Mock the layout to exist
        from rich.layout import Layout
        tui.layout = Layout()
        tui.layout.split_column(
            Layout(name="input")
        )
        
        try:
            print("🔄 Calling _sync_refresh_display()...")
            tui._sync_refresh_display()
            print(f"📊 Refresh called on mock: {tui.live_display.refresh_called}")
            
            # Test again with same input (should not refresh)
            tui.live_display.refresh_called = False
            tui._sync_refresh_display()
            print(f"📊 Second call refresh: {tui.live_display.refresh_called} (should be False)")
            
            # Test with changed input (should refresh)
            tui.state.current_input = "changed input"
            tui._sync_refresh_display()
            print(f"📊 After input change refresh: {tui.live_display.refresh_called} (should be True)")
            
        finally:
            # Restore original isatty
            sys.stdin.isatty = original_isatty
        
        print("\\n🎯 Live display refresh test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def test_character_input_flow():
    """Test the complete character input flow."""
    print("\\n\\n🔍 Testing complete character input flow...")
    
    try:
        from agentsmcp.ui.cli_app import CLIConfig
        from agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface
        
        class DebugConfig:
            debug_mode = True
            verbose = True
        
        config = DebugConfig()
        tui = RevolutionaryTUIInterface(cli_config=config)
        
        # Mock the necessary components
        class MockLiveDisplay:
            def __init__(self):
                self.refresh_count = 0
                
            def refresh(self):
                self.refresh_count += 1
                print(f"🔄 Mock refresh #{self.refresh_count}")
        
        tui.live_display = MockLiveDisplay()
        
        # Mock sys.stdin.isatty
        import sys
        original_isatty = sys.stdin.isatty
        sys.stdin.isatty = lambda: True
        
        # Mock the layout
        from rich.layout import Layout
        tui.layout = Layout()
        tui.layout.split_column(Layout(name="input"))
        
        try:
            print("\\n🔤 Simulating character input: 'h', 'e', 'l', 'l', 'o'")
            
            # Simulate typing "hello"
            for i, char in enumerate("hello", 1):
                print(f"\\n--- Character {i}: '{char}' ---")
                
                # This is what happens in _handle_character_input
                old_input = tui.state.current_input
                tui.state.current_input += char
                
                print(f"📝 Input state: '{old_input}' -> '{tui.state.current_input}'")
                
                # Create panel content
                panel_content = tui._create_input_panel()
                print(f"📊 Panel content: {str(panel_content)[:50]}...")
                
                # Call refresh
                refresh_before = tui.live_display.refresh_count
                tui._sync_refresh_display()
                refresh_after = tui.live_display.refresh_count
                
                print(f"🔄 Refresh calls: {refresh_before} -> {refresh_after}")
                
                if refresh_after > refresh_before:
                    print("✅ Refresh was triggered")
                else:
                    print("❌ Refresh was NOT triggered")
            
            print(f"\\n📊 Final state: '{tui.state.current_input}'")
            print(f"📊 Total refreshes: {tui.live_display.refresh_count}")
            
        finally:
            sys.stdin.isatty = original_isatty
        
        print("\\n🎯 Character input flow test completed!")
        
    except Exception as e:
        print(f"❌ Character input flow test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_live_display_refresh_behavior()
    test_character_input_flow()
    print("\\n🎯 NEXT STEPS:")
    print("1. Check if refresh conditions are working correctly")
    print("2. Verify that Live display refresh is actually being called")
    print("3. Test in real TTY environment to see if auto_refresh=False is the issue")