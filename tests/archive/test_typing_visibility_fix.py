#!/usr/bin/env python3
"""
TYPING VISIBILITY FIX VERIFICATION
Test that the layout checking bug fix restores typing visibility.
"""

import sys
import os
sys.path.insert(0, '/Users/mikko/github/AgentsMCP/src')

def test_typing_visibility_comprehensive():
    """Comprehensive test of typing visibility after the fix."""
    print("🔍 Testing typing visibility after layout checking fix...")
    
    try:
        from agentsmcp.ui.cli_app import CLIConfig
        from agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface
        
        # Create debug config
        class DebugConfig:
            debug_mode = True
            verbose = True
        
        config = DebugConfig()
        tui = RevolutionaryTUIInterface(cli_config=config)
        
        print("✅ TUI instance created successfully")
        
        # Mock the necessary components for testing
        class MockLiveDisplay:
            def __init__(self):
                self.refresh_count = 0
                self.refresh_calls = []
                
            def refresh(self):
                self.refresh_count += 1
                self.refresh_calls.append(f"Refresh #{self.refresh_count}")
                print(f"🔄 {self.refresh_calls[-1]}")
        
        tui.live_display = MockLiveDisplay()
        
        # Mock sys.stdin.isatty to return True
        import sys
        original_isatty = sys.stdin.isatty
        sys.stdin.isatty = lambda: True
        
        # Create proper layout structure
        from rich.layout import Layout
        tui.layout = Layout()
        tui.layout.split_column(Layout(name="input"))
        
        try:
            print("\n🔍 Testing the fixed _sync_refresh_display method...")
            
            # Test 1: Initial call (should initialize tracking and refresh)
            print("\n--- Test 1: Initial call ---")
            tui.state.current_input = ""
            refresh_before = tui.live_display.refresh_count
            
            tui._sync_refresh_display()
            
            refresh_after = tui.live_display.refresh_count
            print(f"📊 Refresh count: {refresh_before} -> {refresh_after}")
            
            if refresh_after > refresh_before:
                print("✅ PASS: Initial refresh triggered")
            else:
                print("❌ FAIL: Initial refresh not triggered")
            
            # Test 2: Same input (should not refresh)
            print("\n--- Test 2: Same input (no change) ---")
            refresh_before = tui.live_display.refresh_count
            
            tui._sync_refresh_display()
            
            refresh_after = tui.live_display.refresh_count
            print(f"📊 Refresh count: {refresh_before} -> {refresh_after}")
            
            if refresh_after == refresh_before:
                print("✅ PASS: No unnecessary refresh for same input")
            else:
                print("❌ FAIL: Unexpected refresh for same input")
            
            # Test 3: Input change (should refresh)
            print("\n--- Test 3: Input change ---")
            old_input = tui.state.current_input
            tui.state.current_input = "hello"
            refresh_before = tui.live_display.refresh_count
            
            tui._sync_refresh_display()
            
            refresh_after = tui.live_display.refresh_count
            print(f"📊 Input: '{old_input}' -> '{tui.state.current_input}'")
            print(f"📊 Refresh count: {refresh_before} -> {refresh_after}")
            
            if refresh_after > refresh_before:
                print("✅ PASS: Refresh triggered for input change")
            else:
                print("❌ FAIL: Refresh not triggered for input change")
            
            # Test 4: Character-by-character typing simulation
            print("\n--- Test 4: Character-by-character typing ---")
            base_input = "hello"
            for i, char in enumerate(" world!", 1):
                print(f"\n  Typing character {i}: '{char}'")
                
                old_input = tui.state.current_input
                tui.state.current_input += char
                refresh_before = tui.live_display.refresh_count
                
                tui._sync_refresh_display()
                
                refresh_after = tui.live_display.refresh_count
                print(f"  📝 Input: '{old_input}' -> '{tui.state.current_input}'")
                print(f"  📊 Refresh: {refresh_before} -> {refresh_after}")
                
                if refresh_after > refresh_before:
                    print(f"  ✅ Character '{char}' triggered refresh")
                else:
                    print(f"  ❌ Character '{char}' did not trigger refresh")
            
            # Summary
            print(f"\n📊 Final state:")
            print(f"📊 Total refreshes: {tui.live_display.refresh_count}")
            print(f"📊 Final input: '{tui.state.current_input}'")
            print(f"📊 Tracking variable: {getattr(tui, '_last_input_refresh_content', 'NOT SET')}")
            
            # Test 5: Layout checking doesn't crash
            print("\n--- Test 5: Layout checking robustness ---")
            try:
                # This should work now (previously would crash)
                tui._sync_refresh_display()
                print("✅ PASS: Layout checking works without KeyError")
            except KeyError as e:
                print(f"❌ FAIL: Layout checking still has KeyError: {e}")
            except Exception as e:
                print(f"❌ FAIL: Unexpected error in layout checking: {e}")
            
        finally:
            # Restore original isatty
            sys.stdin.isatty = original_isatty
        
        print("\n🎯 Typing visibility test completed!")
        
        # Provide assessment
        total_refreshes = tui.live_display.refresh_count
        if total_refreshes >= 7:  # Initial + 6 character changes + input change
            print("🎉 EXCELLENT: Typing visibility fix appears to be working!")
            print("   Users should now see their typing in real-time.")
            return True
        elif total_refreshes >= 2:
            print("⚠️  PARTIAL: Some refresh calls work, but there may be edge cases.")
            print("   Basic typing might be visible but not optimal.")
            return False
        else:
            print("❌ FAILED: Refresh mechanism still not working properly.")
            print("   Users will still not see their typing.")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_typing_visibility_comprehensive()
    if success:
        print("\n🚀 READY FOR USER TESTING!")
        print("The fix should restore typing visibility in the TUI.")
    else:
        print("\n🔧 NEEDS MORE WORK")
        print("Additional debugging may be required.")