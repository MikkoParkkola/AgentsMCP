#!/usr/bin/env python3
"""
DEBUG LAYOUT CORRUPTION
Investigate what happens when typing starts that causes layout to break.
"""

import sys
import os
sys.path.insert(0, '/Users/mikko/github/AgentsMCP/src')

def debug_layout_corruption_on_typing():
    """Debug the layout corruption that occurs when typing starts."""
    print("🔍 Debugging layout corruption when typing starts...")
    
    try:
        from agentsmcp.ui.cli_app import CLIConfig
        from agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface
        from rich.layout import Layout
        from rich.panel import Panel
        from rich import box
        
        # Create debug config
        class DebugConfig:
            debug_mode = True
            verbose = True
        
        config = DebugConfig()
        tui = RevolutionaryTUIInterface(cli_config=config)
        
        print("✅ TUI instance created")
        
        # Mock components to capture what's happening
        class DiagnosticLiveDisplay:
            def __init__(self):
                self.refresh_count = 0
                self.layout_states = []
                
            def refresh(self):
                self.refresh_count += 1
                # Capture layout state at refresh time
                if hasattr(tui, 'layout') and tui.layout:
                    try:
                        layout_info = {
                            'refresh_num': self.refresh_count,
                            'layout_type': type(tui.layout).__name__,
                            'layout_size': getattr(tui.layout, 'size', 'unknown'),
                            'has_input': True
                        }
                        try:
                            input_panel = tui.layout["input"]
                            layout_info['input_panel_type'] = type(input_panel).__name__
                        except:
                            layout_info['input_panel_type'] = 'ACCESS_ERROR'
                        
                        self.layout_states.append(layout_info)
                    except Exception as e:
                        self.layout_states.append({
                            'refresh_num': self.refresh_count,
                            'error': str(e)
                        })
                
                print(f"🔄 Refresh #{self.refresh_count} - Layout state captured")
        
        tui.live_display = DiagnosticLiveDisplay()
        
        # Mock sys.stdin.isatty
        import sys
        original_isatty = sys.stdin.isatty
        sys.stdin.isatty = lambda: True
        
        # Create initial layout structure
        tui.layout = Layout()
        tui.layout.split_column(Layout(name="input"))
        
        try:
            print("\n🔍 Simulating typing sequence that causes corruption...")
            
            # Step 1: Initial state
            print("\n--- Step 1: Initial state ---")
            tui.state.current_input = ""
            print(f"📊 Initial input: '{tui.state.current_input}'")
            
            # Create initial panel
            initial_panel = tui._create_input_panel()
            print(f"📊 Initial panel created: {type(initial_panel).__name__}")
            
            # Do initial refresh
            tui._sync_refresh_display()
            print(f"📊 Initial refresh done, count: {tui.live_display.refresh_count}")
            
            # Step 2: First character (when corruption typically starts)
            print("\n--- Step 2: First character typed ---")
            tui.state.current_input = "h"
            print(f"📊 After first char: '{tui.state.current_input}'")
            
            # Check layout before refresh
            try:
                pre_refresh_input = tui.layout["input"]
                print(f"📊 Layout input accessible before refresh: {type(pre_refresh_input).__name__}")
            except Exception as e:
                print(f"❌ Layout input NOT accessible before refresh: {e}")
            
            # Do refresh that might cause corruption
            tui._sync_refresh_display()
            print(f"📊 Refresh after first char, count: {tui.live_display.refresh_count}")
            
            # Check layout after refresh
            try:
                post_refresh_input = tui.layout["input"]
                print(f"📊 Layout input accessible after refresh: {type(post_refresh_input).__name__}")
            except Exception as e:
                print(f"❌ Layout input NOT accessible after refresh: {e}")
            
            # Step 3: Multiple rapid characters (simulating real typing)
            print("\n--- Step 3: Rapid character sequence ---")
            for i, char in enumerate("ello world", 1):
                print(f"\n  Character {i+1}: '{char}'")
                
                old_input = tui.state.current_input
                tui.state.current_input += char
                
                # Check layout stability before refresh
                try:
                    _ = tui.layout["input"]
                    layout_stable_before = True
                except:
                    layout_stable_before = False
                    print(f"  ❌ Layout unstable BEFORE refresh for char '{char}'")
                
                # Do refresh
                refresh_before = tui.live_display.refresh_count
                tui._sync_refresh_display()
                refresh_after = tui.live_display.refresh_count
                
                # Check layout stability after refresh
                try:
                    _ = tui.layout["input"]
                    layout_stable_after = True
                except Exception as e:
                    layout_stable_after = False
                    print(f"  ❌ Layout unstable AFTER refresh for char '{char}': {e}")
                
                if layout_stable_before and layout_stable_after:
                    print(f"  ✅ Layout stable for char '{char}'")
                elif not layout_stable_before:
                    print(f"  ⚠️  Layout was already unstable before char '{char}'")
                    break  # Stop if layout is already broken
            
            # Step 4: Analyze what went wrong
            print(f"\n📊 ANALYSIS:")
            print(f"📊 Total refreshes: {tui.live_display.refresh_count}")
            print(f"📊 Layout states captured: {len(tui.live_display.layout_states)}")
            
            for state in tui.live_display.layout_states:
                print(f"  Refresh {state.get('refresh_num', '?')}: {state}")
            
            # Step 5: Test the input panel creation independently
            print(f"\n--- Step 5: Independent panel creation test ---")
            try:
                test_input = "test independent panel"
                tui.state.current_input = test_input
                independent_panel = tui._create_input_panel()
                print(f"✅ Independent panel creation works: {type(independent_panel).__name__}")
                print(f"   Content preview: {str(independent_panel)[:100]}...")
            except Exception as e:
                print(f"❌ Independent panel creation fails: {e}")
            
        finally:
            sys.stdin.isatty = original_isatty
        
        print("\n🎯 Layout corruption debugging completed!")
        
        # Provide diagnosis
        if tui.live_display.refresh_count > 0:
            if any('error' in state for state in tui.live_display.layout_states):
                print("🔍 DIAGNOSIS: Layout errors detected during refresh sequence")
                print("   This suggests the refresh mechanism is corrupting the layout")
            else:
                print("🔍 DIAGNOSIS: Refreshes work but something else might cause corruption")
                print("   The issue may be in timing, threading, or panel updates")
        else:
            print("🔍 DIAGNOSIS: No refreshes occurred - the refresh mechanism itself is broken")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

def debug_enter_key_handling():
    """Debug why Enter key doesn't send messages."""
    print("\n\n🔍 Debugging Enter key / message sending...")
    
    try:
        from agentsmcp.ui.cli_app import CLIConfig
        from agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface
        
        # Create debug config
        class DebugConfig:
            debug_mode = True
            verbose = True
        
        config = DebugConfig()
        tui = RevolutionaryTUIInterface(cli_config=config)
        
        # Check if key handling methods exist
        key_methods = [
            '_handle_character_input',
            '_handle_enter_key', 
            '_handle_key_input',
            '_handle_backspace',
            'send_message',
            '_send_user_message'
        ]
        
        print("📊 Checking key handling methods:")
        for method in key_methods:
            exists = hasattr(tui, method)
            print(f"  {method}: {'✅ EXISTS' if exists else '❌ MISSING'}")
        
        # Check event system
        print("\n📊 Checking event system:")
        if hasattr(tui, 'event_system'):
            print(f"  event_system: ✅ EXISTS ({type(tui.event_system).__name__})")
        else:
            print("  event_system: ❌ MISSING")
        
        # Check input handling setup
        print("\n📊 Checking input setup:")
        if hasattr(tui, 'state'):
            print(f"  state: ✅ EXISTS")
            print(f"  current_input: '{getattr(tui.state, 'current_input', 'NOT SET')}'")
        else:
            print("  state: ❌ MISSING")
            
    except Exception as e:
        print(f"❌ Enter key debug failed: {e}")

if __name__ == "__main__":
    debug_layout_corruption_on_typing()
    debug_enter_key_handling()
    print("\n🎯 Use this output to identify the exact cause of layout corruption!")