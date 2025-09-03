#!/usr/bin/env python3
"""
LAYOUT CORRUPTION TIMING DIAGNOSTIC
Debug layout corruption that happens when typing starts in real environment.
"""

import sys
import os
import time
sys.path.insert(0, '/Users/mikko/github/AgentsMCP/src')

def debug_layout_corruption_real_timing():
    """Debug layout corruption with real timing analysis."""
    print("🔍 Debugging layout corruption with real timing analysis...")
    
    try:
        from agentsmcp.ui.cli_app import CLIConfig
        from agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface
        from rich.layout import Layout
        from rich.panel import Panel
        from rich.console import Console
        import threading
        import queue
        
        # Create TUI instance
        class DebugConfig:
            debug_mode = True
            verbose = True
        
        config = DebugConfig()
        tui = RevolutionaryTUIInterface(cli_config=config)
        
        print("✅ TUI instance created")
        
        # Create detailed tracking
        class CorruptionTracker:
            def __init__(self):
                self.refresh_calls = []
                self.layout_states = []
                self.corruption_detected = False
                self.lock = threading.Lock()
                
            def track_refresh(self, refresh_num, input_state, timing_info):
                with self.lock:
                    self.refresh_calls.append({
                        'refresh_num': refresh_num,
                        'input_state': input_state,
                        'timestamp': time.time(),
                        **timing_info
                    })
                    
            def track_layout_state(self, state_info):
                with self.lock:
                    self.layout_states.append({
                        'timestamp': time.time(),
                        **state_info
                    })
                    
            def detect_corruption(self, reason):
                with self.lock:
                    self.corruption_detected = True
                    print(f"🚨 CORRUPTION DETECTED: {reason}")
        
        tracker = CorruptionTracker()
        
        # Mock Live display with corruption detection
        class CorruptionDetectingLiveDisplay:
            def __init__(self):
                self.refresh_count = 0
                self.last_refresh_time = 0
                
            def refresh(self):
                current_time = time.time()
                self.refresh_count += 1
                time_since_last = current_time - self.last_refresh_time
                
                # Track this refresh
                tracker.track_refresh(
                    self.refresh_count,
                    getattr(tui.state, 'current_input', 'N/A'),
                    {
                        'time_since_last': time_since_last,
                        'thread_id': threading.get_ident(),
                        'stack_depth': len(traceback.extract_stack()) if 'traceback' in globals() else 'unknown'
                    }
                )
                
                # Check for rapid refreshes (potential corruption cause)
                if time_since_last < 0.01 and self.refresh_count > 1:
                    tracker.detect_corruption(f"Rapid refresh #{self.refresh_count} (Δt: {time_since_last:.3f}s)")
                
                # Check layout accessibility
                try:
                    if hasattr(tui, 'layout') and tui.layout:
                        _ = tui.layout["input"]  # This might fail
                        tracker.track_layout_state({
                            'accessible': True,
                            'refresh_num': self.refresh_count
                        })
                    else:
                        tracker.track_layout_state({
                            'accessible': False,
                            'reason': 'No layout',
                            'refresh_num': self.refresh_count
                        })
                except Exception as e:
                    tracker.detect_corruption(f"Layout access failed at refresh #{self.refresh_count}: {e}")
                    tracker.track_layout_state({
                        'accessible': False,
                        'reason': str(e),
                        'refresh_num': self.refresh_count
                    })
                
                self.last_refresh_time = current_time
                print(f"🔄 Refresh #{self.refresh_count} (Δt: {time_since_last:.3f}s)")
        
        # Set up tracking
        tui.live_display = CorruptionDetectingLiveDisplay()
        
        # Create proper layout
        tui.layout = Layout()
        tui.layout.split_column(Layout(name="input"))
        
        print(f"\n📊 LAYOUT CORRUPTION TIMING TEST:")
        print("=" * 50)
        
        # Test 1: Initial state
        print("\n--- Test 1: Initial state stability ---")
        tui.state.current_input = ""
        
        # Do initial refresh
        tui._sync_refresh_display()
        time.sleep(0.1)  # Allow processing
        
        print(f"Initial refresh done. Corruption detected: {tracker.corruption_detected}")
        
        # Test 2: Single character change
        print("\n--- Test 2: Single character change ---")
        old_corruption = tracker.corruption_detected
        
        tui.state.current_input = "h"
        tui._sync_refresh_display()
        time.sleep(0.05)
        
        if tracker.corruption_detected and not old_corruption:
            print("🚨 Corruption started with first character!")
        else:
            print("✅ Single character OK")
        
        # Test 3: Rapid typing simulation
        print("\n--- Test 3: Rapid typing simulation ---")
        base_corruption = tracker.corruption_detected
        
        rapid_start = time.time()
        for i, char in enumerate("ello world!", 1):
            old_input = tui.state.current_input
            tui.state.current_input += char
            
            # Time the refresh call
            refresh_start = time.time()
            tui._sync_refresh_display()
            refresh_duration = time.time() - refresh_start
            
            print(f"  Char {i} '{char}': refresh took {refresh_duration:.3f}s")
            
            # Very short delay to simulate real typing
            time.sleep(0.001)
            
        rapid_duration = time.time() - rapid_start
        print(f"Rapid typing: {len('ello world!')} chars in {rapid_duration:.3f}s")
        
        if tracker.corruption_detected and not base_corruption:
            print("🚨 Corruption occurred during rapid typing!")
        
        # Test 4: Threading analysis
        print("\n--- Test 4: Threading analysis ---")
        
        thread_ids = set()
        for call in tracker.refresh_calls:
            thread_ids.add(call.get('thread_id', 'unknown'))
        
        print(f"Refresh calls came from {len(thread_ids)} different threads: {thread_ids}")
        
        if len(thread_ids) > 1:
            tracker.detect_corruption("Multiple threads detected - potential race condition!")
        
        # Analysis
        print(f"\n📊 COMPREHENSIVE CORRUPTION ANALYSIS:")
        print("=" * 50)
        
        print(f"Total refresh calls: {tui.live_display.refresh_count}")
        print(f"Corruption detected: {'❌ YES' if tracker.corruption_detected else '✅ NO'}")
        
        # Timing analysis
        rapid_refreshes = [
            call for call in tracker.refresh_calls 
            if call.get('time_since_last', float('inf')) < 0.01
        ]
        
        print(f"Rapid refreshes (< 10ms): {len(rapid_refreshes)}")
        
        # Layout access analysis
        layout_failures = [
            state for state in tracker.layout_states 
            if not state.get('accessible', False)
        ]
        
        print(f"Layout access failures: {len(layout_failures)}")
        
        # Show detailed failure reasons
        if layout_failures:
            print("Layout failure reasons:")
            for failure in layout_failures[:5]:  # First 5
                reason = failure.get('reason', 'unknown')
                refresh_num = failure.get('refresh_num', '?')
                print(f"  Refresh #{refresh_num}: {reason}")
        
        # Timing histogram
        print(f"\nRefresh timing histogram:")
        timing_ranges = [
            (0, 0.001, "< 1ms"),
            (0.001, 0.01, "1-10ms"), 
            (0.01, 0.05, "10-50ms"),
            (0.05, 0.1, "50-100ms"),
            (0.1, float('inf'), "> 100ms")
        ]
        
        for min_time, max_time, label in timing_ranges:
            count = sum(1 for call in tracker.refresh_calls 
                       if min_time <= call.get('time_since_last', float('inf')) < max_time)
            print(f"  {label}: {count} refreshes")
        
        # Recommendations
        print(f"\n🎯 CORRUPTION PREVENTION RECOMMENDATIONS:")
        print("=" * 50)
        
        if rapid_refreshes:
            print("❌ ISSUE: Rapid refresh calls detected")
            print("   → Implement stronger throttling (increase minimum interval)")
            print("   → Consider debouncing instead of simple throttling")
        
        if layout_failures:
            print("❌ ISSUE: Layout access failures detected")
            print("   → Add more robust layout checking")
            print("   → Consider layout recreation on failure")
        
        if len(thread_ids) > 1:
            print("❌ ISSUE: Multi-threading detected")
            print("   → Ensure all UI updates happen on main thread")
            print("   → Add thread-safety locks")
        
        if not tracker.corruption_detected:
            print("✅ NO MAJOR ISSUES: Layout corruption not reproduced in test")
            print("   → Issue might be environment-specific")
            print("   → Try running in actual TTY terminal")
            
    except Exception as e:
        print(f"❌ Layout corruption timing diagnostic failed: {e}")
        import traceback
        traceback.print_exc()

def debug_refresh_method_implementation():
    """Debug the actual refresh method implementation."""
    print(f"\n📊 REFRESH METHOD IMPLEMENTATION:")
    print("=" * 50)
    
    try:
        from agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface
        import inspect
        
        # Get the source code of _sync_refresh_display
        if hasattr(RevolutionaryTUIInterface, '_sync_refresh_display'):
            method = RevolutionaryTUIInterface._sync_refresh_display
            
            try:
                source = inspect.getsource(method)
                lines = source.split('\n')
                
                print("_sync_refresh_display implementation analysis:")
                print(f"  Total lines: {len(lines)}")
                
                # Look for potential issues
                issues = []
                for i, line in enumerate(lines, 1):
                    line_stripped = line.strip()
                    
                    # Check for potential race condition patterns
                    if 'time.time()' in line_stripped:
                        issues.append(f"Line {i}: Time check - {line_stripped}")
                    
                    if 'self.layout[' in line_stripped and 'try:' not in lines[max(0, i-3):i]:
                        issues.append(f"Line {i}: Unchecked layout access - {line_stripped}")
                    
                    if 'refresh()' in line_stripped:
                        issues.append(f"Line {i}: Refresh call - {line_stripped}")
                
                if issues:
                    print("  Potential issues found:")
                    for issue in issues:
                        print(f"    ⚠️  {issue}")
                else:
                    print("  ✅ No obvious issues in implementation")
                    
            except Exception as e:
                print(f"  ❌ Could not analyze source: {e}")
        else:
            print("❌ _sync_refresh_display method not found")
            
    except Exception as e:
        print(f"❌ Refresh method analysis failed: {e}")

if __name__ == "__main__":
    debug_layout_corruption_real_timing()
    debug_refresh_method_implementation()
    print(f"\n🎯 LAYOUT CORRUPTION TIMING DIAGNOSTIC COMPLETE")
    print("=" * 50)
    print("This diagnostic helps identify:")
    print("• Exact timing of layout corruption")
    print("• Race conditions in refresh calls")
    print("• Threading issues")
    print("• Layout access failures")
    print("• Refresh method implementation problems")
    print("Share this output to help fix layout corruption!")