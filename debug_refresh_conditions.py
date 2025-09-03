#!/usr/bin/env python3
"""
DEBUG REFRESH CONDITIONS
Detailed analysis of exactly why the refresh conditions are failing.
"""

import sys
import os
sys.path.insert(0, '/Users/mikko/github/AgentsMCP/src')

def debug_refresh_conditions():
    """Debug exactly what's happening with refresh conditions."""
    print("🔍 Debugging refresh conditions step by step...")
    
    try:
        from agentsmcp.ui.cli_app import CLIConfig
        from agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface
        
        class DebugConfig:
            debug_mode = True
            verbose = True
        
        config = DebugConfig()
        tui = RevolutionaryTUIInterface(cli_config=config)
        
        # Mock components
        class MockLiveDisplay:
            def __init__(self):
                self.refresh_count = 0
                
            def refresh(self):
                self.refresh_count += 1
                print(f"🔄 REFRESH CALLED #{self.refresh_count}")
        
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
            print("\\n🔍 Testing step-by-step conditions...")
            
            # Set up initial state
            tui.state.current_input = "test"
            
            print("\\n--- Step 1: Check individual conditions ---")
            
            # Check hasattr(self, 'live_display')
            has_live_display = hasattr(tui, 'live_display')
            print(f"✅ hasattr(tui, 'live_display'): {has_live_display}")
            
            # Check tui.live_display exists
            live_display_exists = tui.live_display is not None
            print(f"✅ tui.live_display is not None: {live_display_exists}")
            
            # Check sys.stdin.isatty()
            is_tty = sys.stdin.isatty()
            print(f"✅ sys.stdin.isatty(): {is_tty}")
            
            # Check layout exists
            has_layout = hasattr(tui, 'layout') and tui.layout is not None
            print(f"✅ layout exists: {has_layout}")
            
            # Check input key in layout (proper way for Rich Layout)
            input_in_layout = False
            if has_layout:
                try:
                    _ = tui.layout["input"]
                    input_in_layout = True
                except (KeyError, ValueError):
                    input_in_layout = False
            print(f"✅ 'input' in layout: {input_in_layout}")
            
            print("\\n--- Step 2: Check the main condition from _sync_refresh_display ---")
            
            # This is the main condition that should allow us to reach refresh logic
            main_condition = (
                hasattr(tui, 'live_display') and tui.live_display and 
                sys.stdin.isatty()
            )
            print(f"🎯 Main condition (live_display + TTY): {main_condition}")
            
            layout_condition = tui.layout is not None
            print(f"🎯 Layout condition: {layout_condition}")
            
            # Check input key condition properly
            input_key_condition = False
            if tui.layout:
                try:
                    _ = tui.layout["input"]
                    input_key_condition = True
                except (KeyError, ValueError):
                    input_key_condition = False
            print(f"🎯 Input key condition: {input_key_condition}")
            
            print("\\n--- Step 3: Check refresh-specific conditions ---")
            
            # Check hasattr(self.live_display, 'refresh')
            live_has_refresh = hasattr(tui.live_display, 'refresh')
            print(f"✅ hasattr(live_display, 'refresh'): {live_has_refresh}")
            
            # Check tracking variable
            has_tracking = hasattr(tui, '_last_input_refresh_content')
            print(f"📊 hasattr(tui, '_last_input_refresh_content'): {has_tracking}")
            
            if has_tracking:
                tracking_value = tui._last_input_refresh_content
                current_input = tui.state.current_input
                inputs_different = tracking_value != current_input
                print(f"📊 _last_input_refresh_content: '{tracking_value}'")
                print(f"📊 state.current_input: '{current_input}'")
                print(f"📊 inputs_different: {inputs_different}")
                
                # Test the exact condition
                exact_condition = (
                    hasattr(tui.live_display, 'refresh') and 
                    hasattr(tui, '_last_input_refresh_content') and
                    tui._last_input_refresh_content != tui.state.current_input
                )
                print(f"🎯 Exact refresh condition: {exact_condition}")
            else:
                print("📊 No tracking variable - should trigger elif branch")
                
                # Test elif condition
                elif_condition = not hasattr(tui, '_last_input_refresh_content')
                print(f"🎯 Elif condition (no tracking): {elif_condition}")
            
            print("\\n--- Step 4: Manually execute _sync_refresh_display logic ---")
            
            # Simulate the exact logic from _sync_refresh_display
            print("🔄 Simulating _sync_refresh_display logic...")
            
            if not (hasattr(tui, 'live_display') and tui.live_display and 
                    sys.stdin.isatty()):
                print("❌ FAILED: Main condition (live_display + TTY)")
                return
            else:
                print("✅ PASSED: Main condition")
            
            if not tui.layout:
                print("❌ FAILED: Layout condition")
                return
            else:
                print("✅ PASSED: Layout condition")
            
            # Check input key condition properly
            try:
                _ = tui.layout["input"]
                print("✅ PASSED: Input key condition")
            except (KeyError, ValueError):
                print("❌ FAILED: Input key condition")
                return
            
            print("✅ Reached refresh logic section")
            
            # Now test the refresh conditions
            refresh_count_before = tui.live_display.refresh_count
            
            if (hasattr(tui.live_display, 'refresh') and 
                hasattr(tui, '_last_input_refresh_content') and
                tui._last_input_refresh_content != tui.state.current_input):
                
                print("🎯 Taking first refresh branch (input changed)")
                tui.live_display.refresh()
                tui._last_input_refresh_content = tui.state.current_input
                
            elif not hasattr(tui, '_last_input_refresh_content'):
                print("🎯 Taking second refresh branch (no tracking)")
                tui._last_input_refresh_content = tui.state.current_input
                if hasattr(tui.live_display, 'refresh'):
                    tui.live_display.refresh()
                    
            elif (hasattr(tui, '_last_input_refresh_content') and 
                  tui._last_input_refresh_content == tui.state.current_input and
                  tui.state.current_input):
                print("🎯 Taking third refresh branch (forced refresh)")
                if hasattr(tui.live_display, 'refresh'):
                    tui.live_display.refresh()
            else:
                print("❌ No refresh branch taken!")
            
            refresh_count_after = tui.live_display.refresh_count
            print(f"📊 Refresh count: {refresh_count_before} -> {refresh_count_after}")
            
        finally:
            sys.stdin.isatty = original_isatty
            
        print("\\n🎯 Refresh conditions debug completed!")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_refresh_conditions()
    print("\\n🎯 This shows exactly which condition is failing!")