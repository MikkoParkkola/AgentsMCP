#!/usr/bin/env python3
"""
Debug script that simulates the exact conditions reported by the user:
- Run TUI for 3 seconds (like user report)
- Capture ALL terminal output including control sequences
- Analyze for the specific scrollback flooding pattern
"""

import sys
import asyncio
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def capture_tui_output():
    """Capture actual TUI output as it would appear to a user."""
    print("=== Capturing Actual TUI Terminal Output ===")
    
    from agentsmcp.ui.modern_tui import ModernTUI
    
    class MockConfig:
        interface_mode = "tui"
    
    class MockThemeManager:
        def rich_theme(self): return None
    
    class MockConversationManager:
        def get_history(self): return []
        def process_input(self, msg): return f"Echo: {msg}"
    
    class MockOrchestrationManager:
        def user_settings(self): return {}
    
    # Setup output capture
    import subprocess
    import tempfile
    
    # Create a script that will run the TUI and capture output
    tui_script = f'''
import sys
sys.path.insert(0, "{Path(__file__).parent / "src"}")

import asyncio
from agentsmcp.ui.modern_tui import ModernTUI

class MockConfig:
    interface_mode = "tui"

class MockThemeManager:
    def rich_theme(self): return None

class MockConversationManager:
    def get_history(self): return []
    def process_input(self, msg): return f"Echo: {{msg}}"

class MockOrchestrationManager:
    def user_settings(self): return {{}}

async def run_tui_for_capture():
    tui = ModernTUI(
        config=MockConfig(),
        theme_manager=MockThemeManager(), 
        conversation_manager=MockConversationManager(),
        orchestration_manager=MockOrchestrationManager(),
        no_welcome=True
    )
    
    try:
        # Run TUI but timeout after 2 seconds to capture the flooding
        await asyncio.wait_for(tui.run(), timeout=2.0)
    except asyncio.TimeoutError:
        pass  # Expected

if __name__ == "__main__":
    asyncio.run(run_tui_for_capture())
'''
    
    # Write script to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(tui_script)
        script_path = f.name
    
    try:
        print(f"Running TUI script with output capture...")
        
        # Run the script and capture ALL output including ANSI escape sequences
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        stdout = result.stdout
        stderr = result.stderr
        
        print(f"Captured stdout: {len(stdout)} characters")
        print(f"Captured stderr: {len(stderr)} characters")
        
        # Analyze the stdout for flooding patterns
        if stdout:
            analyze_scrollback_flooding(stdout)
        else:
            print("❌ No stdout captured - TUI may have failed to start")
            if stderr:
                print(f"stderr: {stderr}")
        
        return len(stdout) > 0
        
    except subprocess.TimeoutExpired:
        print("❌ TUI script timed out (may be hanging)")
        return False
    except Exception as e:
        print(f"❌ Failed to run TUI script: {e}")
        return False
    finally:
        # Clean up temp file
        try:
            Path(script_path).unlink()
        except:
            pass

def analyze_scrollback_flooding(output):
    """Analyze captured output for scrollback flooding patterns."""
    print("\n=== Analyzing Scrollback Flooding ===")
    
    # Split by lines and analyze patterns
    lines = output.split('\\n')
    print(f"Total output lines: {len(lines)}")
    
    # Look for repeated identical frames
    frame_hashes = {}
    current_frame = []
    frame_count = 0
    
    for line in lines:
        # Detect frame boundaries (panel borders)
        if line.startswith('╭─') or line.startswith('┌─'):
            # Starting a new frame
            if current_frame:
                # Process previous frame
                frame_str = '\\n'.join(current_frame)
                frame_hash = hash(frame_str)
                
                if frame_hash in frame_hashes:
                    frame_hashes[frame_hash] += 1
                else:
                    frame_hashes[frame_hash] = 1
                
                frame_count += 1
            
            current_frame = [line]
        elif current_frame:
            current_frame.append(line)
    
    # Process last frame
    if current_frame:
        frame_str = '\\n'.join(current_frame)
        frame_hash = hash(frame_str)
        frame_hashes[frame_hash] = frame_hashes.get(frame_hash, 0) + 1
        frame_count += 1
    
    print(f"Detected {frame_count} total frames")
    print(f"Unique frame patterns: {len(frame_hashes)}")
    
    # Find most repeated frames
    repeated_frames = [(count, fhash) for fhash, count in frame_hashes.items() if count > 1]
    repeated_frames.sort(reverse=True)
    
    if repeated_frames:
        print(f"❌ SCROLLBACK FLOODING DETECTED!")
        print(f"Most repeated frame appears {repeated_frames[0][0]} times")
        
        total_repeated = sum(count - 1 for count, _ in repeated_frames)
        print(f"Total redundant frames: {total_repeated}")
        
        # Calculate flooding ratio
        if frame_count > 0:
            flooding_ratio = total_repeated / frame_count
            print(f"Flooding ratio: {flooding_ratio:.2%}")
            
            if flooding_ratio > 0.5:
                print("❌ SEVERE flooding - over 50% redundant frames")
            elif flooding_ratio > 0.2:
                print("⚠️  MODERATE flooding - over 20% redundant frames")  
            else:
                print("✓ Minor flooding - under 20% redundant frames")
        
        return False
    else:
        print("✓ No repeated frames detected")
        return True

def test_with_simulated_input():
    """Test TUI with some simulated user input to trigger more rendering."""
    print("\n=== Testing with Simulated Input ===")
    
    from agentsmcp.ui.modern_tui import ModernTUI
    import threading
    import queue
    
    class MockConfig:
        interface_mode = "tui"
    
    class MockThemeManager:
        def rich_theme(self): return None
    
    class MockConversationManager:
        def get_history(self): return []
        def process_input(self, msg): 
            return f"Mock response to: {msg}"
    
    class MockOrchestrationManager:
        def user_settings(self): return {}
        
        def save_user_settings(self, settings):
            pass
    
    try:
        tui = ModernTUI(
            config=MockConfig(),
            theme_manager=MockThemeManager(),
            conversation_manager=MockConversationManager(), 
            orchestration_manager=MockOrchestrationManager(),
            no_welcome=True
        )
        
        async def run_with_simulated_input():
            # Start TUI in background
            tui_task = asyncio.create_task(tui.run())
            
            # Simulate some user activity after a brief delay
            await asyncio.sleep(0.1)
            
            # Trigger some refresh events manually
            for i in range(5):
                tui.mark_dirty("content") 
                await asyncio.sleep(0.1)
                
            # Trigger sidebar toggle  
            if hasattr(tui, '_toggle_sidebar'):
                tui._toggle_sidebar()
                await asyncio.sleep(0.1)
                tui._toggle_sidebar()
            
            # Wait a bit more then stop
            await asyncio.sleep(0.5)
            tui._running = False
            
            try:
                await asyncio.wait_for(tui_task, timeout=1.0)
            except asyncio.TimeoutError:
                tui_task.cancel()
        
        # Run the test
        asyncio.run(run_with_simulated_input())
        
        print("✓ TUI ran with simulated input without hanging")
        return True
        
    except Exception as e:
        print(f"❌ TUI with simulated input failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run comprehensive scrollback flooding analysis."""
    print("🐛 Scrollback Flooding Analysis")
    print("=" * 50)
    
    results = {
        "Actual TUI Output Capture": capture_tui_output(),
        "Simulated Input Test": test_with_simulated_input(),
    }
    
    print("\\n" + "=" * 50)
    print("📊 Analysis Results:")
    
    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"  
        icon = "✅" if passed else "❌"
        print(f"{icon} {test_name}: {status}")
        if not passed:
            all_passed = False
    
    if not all_passed:
        print("\\n💡 Confirmed Issues - Priority Fixes Needed:")
        print("1. 🚨 CRITICAL: Fix Rich Live event loop to prevent frame flooding")
        print("2. 🔧 Implement proper content change detection before Live.update()")
        print("3. ⚡ Add frame deduplication at the layout level")
        print("4. 🎯 Fix _refresh_event.clear() timing in main loop")
        print("5. 📝 Add comprehensive integration tests for terminal output")
    else:
        print("\\n🎉 No scrollback flooding detected!")
        print("Issues may be environment-specific or timing-related.")

if __name__ == "__main__":
    main()