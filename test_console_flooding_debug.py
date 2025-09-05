#!/usr/bin/env python3
"""
DEBUG TEST: Console Status Flooding Issue

This test checks why status updates are printing new lines instead of updating in-place.
We'll examine the Live display state and status routing.
"""

import asyncio
import sys
import os
import time

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agentsmcp.ui.v3.chat_engine import ChatEngine
from agentsmcp.ui.v3.tui_launcher import TUILauncher

async def test_status_display_routing():
    """Debug test to understand status display routing and Live display state."""
    print("🔍 CONSOLE FLOODING DEBUG TEST")
    print("=" * 60)
    print("🎯 Goal: Understand why status updates create new lines instead of updating in-place")
    print("=" * 60)
    
    try:
        # Create TUI launcher
        print("✓ Creating TUI launcher...")
        tui_launcher = TUILauncher()
        
        # Initialize the launcher
        print("✓ Initializing TUI launcher...")
        success = tui_launcher.initialize()
        if not success:
            print("❌ Failed to initialize TUI launcher")
            return False
        
        # Debug: Check initial state
        print(f"\n📊 INITIAL STATE:")
        print(f"   Current renderer: {type(tui_launcher.current_renderer).__name__}")
        
        if tui_launcher.current_renderer:
            renderer = tui_launcher.current_renderer
            print(f"   Renderer has Live: {hasattr(renderer, 'live')}")
            if hasattr(renderer, 'live'):
                print(f"   Live object exists: {renderer.live is not None}")
                if renderer.live:
                    print(f"   Live started: {getattr(renderer.live, '_started', 'unknown')}")
            print(f"   Renderer has layout: {hasattr(renderer, 'layout')}")
            if hasattr(renderer, 'layout'):
                print(f"   Layout exists: {renderer.layout is not None}")
        
        # Debug: Test status updates
        print(f"\n🧪 TESTING STATUS UPDATES:")
        print("-" * 40)
        
        # Create a series of status updates to observe behavior
        test_statuses = [
            "Initializing agent orchestration...",
            "✅ Planning complete - ready for execution",
            "🛠️ Agent-1: Starting product analysis",
            "🔍 Agent-2: Analyzing codebase structure", 
            "Processing agent responses..."
        ]
        
        for i, status in enumerate(test_statuses):
            print(f"\n📤 Sending status {i+1}/5: '{status}'")
            
            # Send status via normal routing
            if tui_launcher.current_renderer:
                print(f"   Routing through: {type(tui_launcher.current_renderer).__name__}.show_status()")
                
                # Debug the Live display state before calling show_status
                renderer = tui_launcher.current_renderer
                if hasattr(renderer, 'live') and hasattr(renderer, 'layout'):
                    live_active = renderer.live is not None
                    layout_active = renderer.layout is not None
                    conditions_met = live_active and layout_active
                    print(f"   Live active: {live_active}, Layout active: {layout_active}")
                    print(f"   Conditions for in-place update: {conditions_met}")
                    
                    if not conditions_met:
                        print(f"   ⚠️  Will fallback to console.print() - creating new line!")
                
                # Call show_status
                tui_launcher.current_renderer.show_status(status)
                
                # Brief pause to see the effect
                await asyncio.sleep(1.0)
        
        # Final state check
        print(f"\n📊 FINAL STATE CHECK:")
        print("-" * 40)
        
        if tui_launcher.current_renderer:
            renderer = tui_launcher.current_renderer
            if hasattr(renderer, '_current_status'):
                print(f"   Last recorded status: '{renderer._current_status}'")
            
            # Check if Live display ever became truly active
            if hasattr(renderer, 'live') and renderer.live:
                started_attr = getattr(renderer.live, '_started', None)
                print(f"   Live display started during test: {started_attr}")
        
        print(f"\n🏆 TEST CONCLUSIONS:")
        print("=" * 60)
        print("1. If 'Will fallback to console.print()' appeared above,")
        print("   then Live display conditions weren't met")
        print("2. If Live display was never started, that's the root cause")
        print("3. Status flooding happens because console.print() creates new lines")
        print("   instead of updating Rich Live display panels in-place")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Debug test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        if 'tui_launcher' in locals():
            try:
                await tui_launcher.cleanup()
            except Exception:
                pass

if __name__ == "__main__":
    print("🚀 Starting Console Flooding Debug Test")
    print("🎯 This test examines the Live display state during status updates")
    print()
    
    success = asyncio.run(test_status_display_routing())
    
    print("\n" + "=" * 60)
    if success:
        print("🔍 DEBUG TEST COMPLETED: Check output above for Live display state issues")
        print("   Look for 'Will fallback to console.print()' warnings")
    else:
        print("❌ DEBUG TEST FAILED: Could not examine status routing properly")
    
    print("=" * 60)
    sys.exit(0 if success else 1)