#!/usr/bin/env python3
"""
Debug script to identify specific TUI rendering issues:
1. Scrollback flooding with identical frames
2. Every other line rendering (missing line breaks)
3. Input system echo problems
"""

import asyncio
import sys
import time
from pathlib import Path
from io import StringIO

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agentsmcp.ui.modern_tui import ModernTUI
from agentsmcp.ui.components.realtime_input import RealTimeInputField
from rich.console import Console

def test_rich_live_rendering():
    """Test Rich Live rendering to identify frame duplication."""
    print("=== Testing Rich Live Rendering Issues ===")
    
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text
    import time
    
    console = Console()
    
    # Create test content that should only render when changed
    test_counter = 0
    
    def generate_content():
        nonlocal test_counter
        test_counter += 1
        return Panel(
            Text(f"Frame {test_counter}\nTime: {time.time():.3f}"),
            title="Test Panel"
        )
    
    # Track render calls
    render_count = 0
    last_content = None
    
    def tracking_render():
        nonlocal render_count, last_content
        render_count += 1
        content = generate_content()
        
        # Check if content actually changed
        content_str = str(content)
        if content_str == last_content:
            print(f"⚠️  Duplicate render #{render_count} - identical content")
        else:
            print(f"✓ Render #{render_count} - content changed")
        
        last_content = content_str
        return content
    
    print("Testing Rich Live with refresh_per_second=3...")
    
    try:
        with Live(tracking_render(), console=console, refresh_per_second=3) as live:
            # Simulate 2 seconds of operation
            start_time = time.time()
            while time.time() - start_time < 2.0:
                time.sleep(0.1)
                
                # Don't trigger any updates - content should be stable
                pass
                
    except Exception as e:
        print(f"❌ Rich Live test failed: {e}")
        return False
        
    print(f"Total renders in 2 seconds: {render_count}")
    if render_count > 8:  # 3 FPS * 2 seconds = ~6, allow some margin
        print("⚠️  Too many renders - possible frame flooding issue")
        return False
    else:
        print("✓ Render frequency looks normal")
        return True

def test_line_break_rendering():
    """Test if line breaks are properly rendered between UI components."""
    print("\n=== Testing Line Break Rendering ===")
    
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.layout import Layout
    
    console = Console(width=80, legacy_windows=False)
    
    # Create layout similar to ModernTUI
    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(name="body", ratio=1),
        Layout(name="footer", size=3),
    )
    
    # Populate with test content
    layout["header"].update(Panel("Header Content", border_style="blue"))
    layout["body"].update(Panel("Body Content\nLine 2\nLine 3", border_style="green"))
    layout["footer"].update(Panel("Footer Content", border_style="red"))
    
    # Capture rendered output
    output = StringIO()
    test_console = Console(file=output, width=80, legacy_windows=False)
    test_console.print(layout)
    
    rendered_output = output.getvalue()
    lines = rendered_output.split('\n')
    
    print(f"Rendered {len(lines)} lines")
    
    # Check for concatenated borders (the reported issue)
    concatenated_borders = []
    for i, line in enumerate(lines):
        if '╰──────╯╭─────────' in line or similar_border_concatenation(line):
            concatenated_borders.append(f"Line {i+1}: {line.strip()}")
    
    if concatenated_borders:
        print("❌ Found concatenated border lines:")
        for border in concatenated_borders:
            print(f"  {border}")
        
        # Show context around problematic lines
        print("\nContext around problematic lines:")
        for border_info in concatenated_borders:
            line_num = int(border_info.split(':')[0].split()[1]) - 1
            start = max(0, line_num - 2)
            end = min(len(lines), line_num + 3)
            
            print(f"\nLines {start+1}-{end}:")
            for j in range(start, end):
                marker = ">>> " if j == line_num else "    "
                print(f"{marker}{j+1:2d}: {repr(lines[j])}")
        
        return False
    else:
        print("✓ No concatenated borders found")
        return True

def similar_border_concatenation(line):
    """Check for various forms of border concatenation."""
    patterns = [
        '╰──────╯╭─────────',  # Bottom + Top
        '─────╯╭─────',       # Partial borders
        '└──────┘┌─────────', # Alternative box chars
        '─────┘┌─────',       # Alternative partial
    ]
    return any(pattern in line for pattern in patterns)

def test_realtime_input_echo():
    """Test RealTimeInputField for input echo issues."""
    print("\n=== Testing RealTimeInput Echo Issues ===")
    
    console = Console(width=80)
    
    try:
        input_field = RealTimeInputField(
            console=console,
            prompt=">>> ",
            max_width=None,
            max_height=3
        )
        
        print("✓ RealTimeInputField created successfully")
        
        # Test first character input
        test_chars = ['h', 'e', 'l', 'l', 'o']
        
        for i, char in enumerate(test_chars):
            # Simulate key input
            success = asyncio.run(input_field.handle_key(char))
            
            if not success:
                print(f"❌ Failed to handle character '{char}' at position {i}")
                return False
                
            # Check if input was properly recorded
            current_input = input_field.get_current_input()
            expected = ''.join(test_chars[:i+1])
            
            if current_input != expected:
                print(f"❌ Input mismatch at position {i}: got '{current_input}', expected '{expected}'")
                return False
        
        print("✓ Character input handling works correctly")
        
        # Test rendering with input content
        try:
            rendered = input_field.render()
            print("✓ Input field renders successfully with content")
        except Exception as e:
            print(f"❌ Input field rendering failed: {e}")
            return False
        
        # Test clear functionality
        input_field.clear_input()
        if input_field.get_current_input() != "":
            print("❌ Clear input functionality failed")
            return False
        
        print("✓ Clear input functionality works")
        return True
        
    except Exception as e:
        print(f"❌ RealTimeInputField test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_frame_change_detection():
    """Test if the TUI properly detects when content actually changes."""
    print("\n=== Testing Frame Change Detection ===")
    
    # Mock the change detection system used in ModernTUI
    render_cache = {}
    
    def simulate_content_hash(content):
        """Simulate the content hashing used in _update_section_if_changed."""
        if hasattr(content, 'renderable') and hasattr(content, 'title'):
            content_str = str(content.renderable) if content.renderable else ""
            title_str = str(content.title) if content.title else ""
            # Clean whitespace like the real implementation
            content_str = ' '.join(content_str.split())
            title_str = ' '.join(title_str.split())
            return hash((content_str, title_str))
        else:
            content_str = str(content)
            content_str = ' '.join(content_str.split())
            return hash(content_str)
    
    from rich.panel import Panel
    from rich.text import Text
    
    # Test 1: Identical content should not trigger update
    content1 = Panel(Text("Hello World"), title="Test")
    content2 = Panel(Text("Hello World"), title="Test")
    
    hash1 = simulate_content_hash(content1)
    hash2 = simulate_content_hash(content2)
    
    if hash1 != hash2:
        print("❌ Identical content produces different hashes")
        print(f"  Hash1: {hash1}")
        print(f"  Hash2: {hash2}")
        return False
    
    print("✓ Identical content produces same hash")
    
    # Test 2: Different content should trigger update
    content3 = Panel(Text("Different Content"), title="Test")
    hash3 = simulate_content_hash(content3)
    
    if hash1 == hash3:
        print("❌ Different content produces same hash")
        return False
    
    print("✓ Different content produces different hash")
    
    # Test 3: Whitespace differences should be normalized
    content4 = Panel(Text("Hello    World"), title="Test")  # Extra spaces
    hash4 = simulate_content_hash(content4)
    
    if hash1 != hash4:
        print("❌ Whitespace normalization not working")
        return False
    
    print("✓ Whitespace normalization works correctly")
    
    return True

def simulate_tui_run_short():
    """Simulate a short TUI run to identify performance issues."""
    print("\n=== Simulating Short TUI Run ===")
    
    class MockConfig:
        interface_mode = "tui"
        
    class MockThemeManager:
        def rich_theme(self): 
            return None
            
    class MockConversationManager:
        def get_history(self):
            return [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        
    class MockOrchestrationManager:
        def user_settings(self): 
            return {}
    
    try:
        tui = ModernTUI(
            config=MockConfig(),
            theme_manager=MockThemeManager(),
            conversation_manager=MockConversationManager(),
            orchestration_manager=MockOrchestrationManager(),
            no_welcome=True
        )
        
        # Track render calls by patching the render method
        original_render = tui._render
        render_call_count = 0
        
        def counting_render():
            nonlocal render_call_count
            render_call_count += 1
            return original_render()
        
        tui._render = counting_render
        
        # Run for a very short time
        async def short_run():
            try:
                # Give it just enough time to initialize and do a few renders
                await asyncio.wait_for(tui.run(), timeout=0.5)
            except asyncio.TimeoutError:
                pass  # Expected - we're testing a short run
            except Exception as e:
                print(f"❌ Short run failed: {e}")
                return False
            return True
        
        success = asyncio.run(short_run())
        
        if success:
            print(f"✓ TUI ran successfully for 0.5 seconds")
            print(f"  Render calls: {render_call_count}")
            
            if render_call_count > 10:  # More than ~20 FPS for 0.5 seconds
                print("⚠️  High render call count - possible flooding")
            else:
                print("✓ Render call count looks reasonable")
                
            return True
        else:
            return False
            
    except Exception as e:
        print(f"❌ TUI simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all rendering debug tests."""
    print("🐛 ModernTUI Rendering Debug Session")
    print("=" * 60)
    
    results = {
        "Rich Live Rendering": test_rich_live_rendering(),
        "Line Break Rendering": test_line_break_rendering(),
        "RealTime Input Echo": test_realtime_input_echo(),
        "Frame Change Detection": test_frame_change_detection(),
        "Short TUI Simulation": simulate_tui_run_short(),
    }
    
    print("\n" + "=" * 60)
    print("📊 Debug Results Summary:")
    
    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        icon = "✅" if passed else "❌"
        print(f"{icon} {test_name}: {status}")
        if not passed:
            all_passed = False
    
    if not all_passed:
        print("\n💡 Issues Found - Recommended Fixes:")
        
        if not results["Rich Live Rendering"]:
            print("1. Reduce Rich Live refresh rate further (try 1-2 FPS)")
            print("2. Implement better frame deduplication")
            
        if not results["Line Break Rendering"]:
            print("3. Check Rich Layout rendering - ensure proper newlines between panels")
            print("4. Verify Panel border styles don't concatenate")
            
        if not results["RealTime Input Echo"]:
            print("5. Fix RealTimeInputField first character handling")
            print("6. Ensure input callbacks are properly connected")
            
        if not results["Frame Change Detection"]:
            print("7. Improve content hashing to prevent false positives")
            print("8. Add more robust change detection")
    
    else:
        print("\n🎉 All rendering tests passed!")
        print("The reported issues may be environmental or integration-related.")

if __name__ == "__main__":
    main()