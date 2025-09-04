#!/usr/bin/env python3
"""
THROTTLED REFRESH FIX VERIFICATION
Test that the new throttled refresh strategy prevents layout corruption.
"""

import sys
import os
sys.path.insert(0, '/Users/mikko/github/AgentsMCP/src')

def test_throttled_refresh_fix():
    """Test the throttled refresh fix comprehensively."""
    print("🔍 Testing throttled refresh fix...")
    
    try:
        from agentsmcp.ui.cli_app import CLIConfig
        from agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface
        
        # Create debug config
        class DebugConfig:
            debug_mode = True
            verbose = True
        
        config = DebugConfig()
        tui = RevolutionaryTUIInterface(cli_config=config)
        
        print("✅ TUI instance created")
        
        # Mock Live display with throttling detection
        class ThrottleAwareLiveDisplay:
            def __init__(self):
                self.refresh_count = 0
                self.refresh_calls = []
                self.layout_corruption_detected = False
                
            def refresh(self):
                import time
                current_time = time.time()
                
                self.refresh_count += 1
                
                call_info = {
                    'refresh_num': self.refresh_count,
                    'timestamp': current_time,
                    'input_state': getattr(tui.state, 'current_input', 'N/A')
                }
                
                # Check for rapid succession that could cause corruption
                if len(self.refresh_calls) > 0:
                    time_since_last = current_time - self.refresh_calls[-1]['timestamp']
                    call_info['time_since_last'] = time_since_last
                    
                    if time_since_last < 0.01:  # Less than 10ms
                        self.layout_corruption_detected = True
                        print(f"❌ CORRUPTION RISK: Refresh #{self.refresh_count} too soon (Δt: {time_since_last:.3f}s)")
                    else:
                        print(f"✅ Safe refresh #{self.refresh_count} (Δt: {time_since_last:.3f}s)")
                else:
                    call_info['time_since_last'] = float('inf')
                    print(f"✅ Initial refresh #{self.refresh_count}")
                
                self.refresh_calls.append(call_info)
        
        tui.live_display = ThrottleAwareLiveDisplay()
        
        # Mock TTY
        import sys
        original_isatty = sys.stdin.isatty
        sys.stdin.isatty = lambda: True
        
        # Create layout
        from rich.layout import Layout
        tui.layout = Layout()
        tui.layout.split_column(Layout(name="input"))
        
        try:
            print("\n🔍 Testing throttled refresh behavior...")
            
            # Test 1: Rapid typing (should be throttled)
            print("\n--- Test 1: Rapid typing sequence ---")
            import time
            
            rapid_start = time.time()
            for i, char in enumerate("hello world!", 1):
                old_input = tui.state.current_input
                tui.state.current_input += char
                
                print(f"Character {i}: '{char}' -> '{tui.state.current_input}'")
                tui._sync_refresh_display()
                # No delay - test rapid succession
            
            rapid_duration = time.time() - rapid_start
            print(f"📊 Rapid typing: 12 chars in {rapid_duration:.3f}s")
            
            # Test 2: Normal typing speed
            print("\n--- Test 2: Normal typing speed ---")
            tui.state.current_input = ""  # Reset
            
            for i, char in enumerate("normal", 1):
                old_input = tui.state.current_input  
                tui.state.current_input += char
                
                print(f"Character {i}: '{char}' -> '{tui.state.current_input}'")
                tui._sync_refresh_display()
                
                # Realistic typing delay
                time.sleep(0.1)
            
            # Test 3: Edge case - very long input
            print("\n--- Test 3: Long input test ---")
            tui.state.current_input = ""  # Reset
            
            long_text = "This is a very long input text to test how the throttled refresh handles extended typing sessions without corruption"
            
            batch_start = time.time()
            for i, char in enumerate(long_text[:20], 1):  # First 20 chars
                tui.state.current_input += char
                tui._sync_refresh_display()
            batch_duration = time.time() - batch_start
            
            print(f"📊 Long input batch: 20 chars in {batch_duration:.3f}s")
            
            # Analysis
            print(f"\n📊 COMPREHENSIVE ANALYSIS:")
            print(f"📊 Total refresh calls: {tui.live_display.refresh_count}")
            print(f"📊 Layout corruption detected: {'❌ YES' if tui.live_display.layout_corruption_detected else '✅ NO'}")
            
            # Check throttling effectiveness
            rapid_refreshes = [call for call in tui.live_display.refresh_calls if call.get('time_since_last', float('inf')) < 0.05]
            print(f"📊 Rapid refreshes (< 50ms): {len(rapid_refreshes)}")
            
            if len(rapid_refreshes) > 0:
                print("⚠️  Some refreshes occurred rapidly - throttling may need adjustment")
                for rapid in rapid_refreshes[:3]:
                    print(f"    Refresh #{rapid['refresh_num']}: Δt={rapid.get('time_since_last', 'N/A'):.3f}s")
            else:
                print("✅ All refreshes properly throttled")
            
            # Test throttling mechanism directly
            print(f"\n--- Test 4: Direct throttling test ---")
            tui.state.current_input = "test"
            
            # Call refresh multiple times in rapid succession
            throttle_test_start = time.time()
            for i in range(5):
                tui._sync_refresh_display()
            throttle_test_duration = time.time() - throttle_test_start
            
            # Should only result in 1 actual refresh call due to throttling
            expected_refreshes = tui.live_display.refresh_count
            print(f"📊 Throttle test: 5 calls in {throttle_test_duration:.3f}s")
            print(f"📊 Expected throttling behavior: Multiple calls should result in limited actual refreshes")
            
            # Final assessment
            print(f"\n🎯 FINAL ASSESSMENT:")
            if not tui.live_display.layout_corruption_detected and len(rapid_refreshes) < 3:
                print("🎉 THROTTLED REFRESH FIX: SUCCESS!")
                print("✅ No layout corruption risk detected")
                print("✅ Refresh throttling working effectively")
                print("✅ Ready for user testing")
                return True
            else:
                print("⚠️  THROTTLED REFRESH FIX: NEEDS ADJUSTMENT")
                print(f"❌ Corruption risk: {tui.live_display.layout_corruption_detected}")
                print(f"❌ Rapid refreshes: {len(rapid_refreshes)}")
                return False
                
        finally:
            sys.stdin.isatty = original_isatty
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_throttled_refresh_fix()
    if success:
        print("\n🚀 READY FOR USER TESTING!")
        print("The throttled refresh should prevent layout corruption.")
        print("Both typing visibility AND layout stability should now work.")
    else:
        print("\n🔧 NEEDS FURTHER ADJUSTMENT")
        print("The throttling mechanism may need fine-tuning.")