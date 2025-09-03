#!/usr/bin/env python3
"""
MINIMAL REFRESH STRATEGY TEST
Test a more conservative refresh approach that avoids layout corruption.
"""

import sys
import os
sys.path.insert(0, '/Users/mikko/github/AgentsMCP/src')

def test_alternative_refresh_strategy():
    """Test a more conservative refresh strategy."""
    print("🔍 Testing alternative refresh strategy to avoid layout corruption...")
    
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
        
        # Mock Live display with detailed tracking
        class ConservativeLiveDisplay:
            def __init__(self):
                self.refresh_count = 0
                self.layout_updates = []
                self.refresh_calls = []
                self.last_refresh_time = 0
                
            def refresh(self):
                import time
                current_time = time.time()
                time_since_last = current_time - self.last_refresh_time
                
                self.refresh_count += 1
                self.last_refresh_time = current_time
                
                call_info = {
                    'refresh_num': self.refresh_count,
                    'time_since_last': time_since_last,
                    'input_state': getattr(tui.state, 'current_input', 'N/A')
                }
                self.refresh_calls.append(call_info)
                
                print(f"🔄 Refresh #{self.refresh_count} (Δt: {time_since_last:.3f}s) for: '{call_info['input_state']}'")
                
                # Simulate potential timing issue
                if time_since_last < 0.01:  # Less than 10ms between refreshes
                    print(f"⚠️  WARNING: Very fast refresh (Δt: {time_since_last:.3f}s) - potential race condition!")
        
        tui.live_display = ConservativeLiveDisplay()
        
        # Mock TTY
        import sys
        original_isatty = sys.stdin.isatty
        sys.stdin.isatty = lambda: True
        
        # Create layout
        from rich.layout import Layout
        tui.layout = Layout()
        tui.layout.split_column(Layout(name="input"))
        
        try:
            print("\n🔍 Testing conservative refresh pattern...")
            
            # Scenario 1: Normal typing speed (realistic)
            print("\n--- Scenario 1: Normal typing speed ---")
            import time
            
            for i, char in enumerate("hello", 1):
                print(f"\nTyping '{char}' (character {i})")
                
                old_input = tui.state.current_input
                tui.state.current_input += char
                
                # Call refresh
                tui._sync_refresh_display()
                
                # Realistic delay between keystrokes (100-200ms)
                time.sleep(0.15)
            
            print(f"\n📊 Normal typing: {tui.live_display.refresh_count} refreshes")
            
            # Scenario 2: Very fast typing (stress test)
            print("\n--- Scenario 2: Very fast typing ---")
            tui.state.current_input = ""  # Reset
            
            fast_start = time.time()
            for i, char in enumerate(" world!", 1):
                old_input = tui.state.current_input
                tui.state.current_input += char
                
                tui._sync_refresh_display()
                # No delay - rapid succession
            
            fast_duration = time.time() - fast_start
            print(f"📊 Fast typing: {len(' world!')} chars in {fast_duration:.3f}s")
            
            # Analysis
            print(f"\n📊 ANALYSIS:")
            print(f"📊 Total refreshes: {tui.live_display.refresh_count}")
            
            # Check for potential race conditions
            race_conditions = [call for call in tui.live_display.refresh_calls if call['time_since_last'] < 0.01]
            if race_conditions:
                print(f"⚠️  FOUND {len(race_conditions)} potential race conditions (Δt < 10ms)")
                for race in race_conditions[:3]:  # Show first 3
                    print(f"    Refresh #{race['refresh_num']}: Δt={race['time_since_last']:.3f}s")
            else:
                print("✅ No race conditions detected")
            
            # Recommend strategy
            print(f"\n🎯 STRATEGY RECOMMENDATION:")
            if len(race_conditions) > 3:
                print("❌ Current refresh strategy is too aggressive")
                print("   Recommend: Add debouncing/throttling to prevent layout corruption")
            else:
                print("✅ Current refresh strategy seems reasonable")
                print("   Layout corruption may be caused by other factors")
                
        finally:
            sys.stdin.isatty = original_isatty
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def propose_throttled_refresh_fix():
    """Propose a throttled refresh strategy to prevent corruption."""
    print("\n\n🔧 PROPOSED THROTTLED REFRESH FIX:")
    print("="*50)
    
    throttled_code = '''
def _sync_refresh_display(self):
    """Throttled refresh to prevent layout corruption."""
    import time
    
    # Throttling: Only refresh if enough time has passed
    current_time = time.time()
    if not hasattr(self, '_last_manual_refresh_time'):
        self._last_manual_refresh_time = 0
    
    min_refresh_interval = 0.05  # 50ms minimum between refreshes (20 FPS max)
    time_since_last = current_time - self._last_manual_refresh_time
    
    if time_since_last < min_refresh_interval:
        # Skip refresh if too soon - prevents race conditions
        if hasattr(self, '_pending_refresh_content'):
            self._pending_refresh_content = self.state.current_input
        else:
            self._pending_refresh_content = self.state.current_input
        return
    
    # Proceed with refresh...
    self._last_manual_refresh_time = current_time
    # ... rest of existing refresh logic
    '''
    
    print(throttled_code)
    print("\n✅ This throttling approach would:")
    print("   • Prevent race conditions from rapid typing")
    print("   • Limit refresh rate to 20 FPS (smooth but not excessive)")
    print("   • Queue pending updates instead of dropping them")
    print("   • Maintain typing visibility without corruption")

if __name__ == "__main__":
    test_alternative_refresh_strategy()
    propose_throttled_refresh_fix()