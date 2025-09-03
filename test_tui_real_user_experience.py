#!/usr/bin/env python3
"""
TUI REAL USER EXPERIENCE TEST
Test TUI behavior as close as possible to real user experience.
"""

import sys
import os
import time
sys.path.insert(0, '/Users/mikko/github/AgentsMCP/src')

def test_real_user_typing_experience():
    """Test typing experience as close to real user as possible."""
    print("🔍 Testing TUI real user typing experience...")
    
    try:
        from agentsmcp.ui.cli_app import CLIConfig
        from agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface
        from rich.layout import Layout
        from rich.panel import Panel
        from rich.console import Console
        
        # Create as close to real config as possible
        class RealUserConfig:
            debug_mode = False  # Real users don't have debug mode
            verbose = False
        
        config = RealUserConfig()
        tui = RevolutionaryTUIInterface(cli_config=config)
        
        print("✅ TUI instance created with production-like config")
        
        # Track real user experience metrics
        class UserExperienceTracker:
            def __init__(self):
                self.typing_visible = []
                self.layout_breaks = []
                self.refresh_calls = 0
                self.enter_key_works = False
                self.startup_layout_ok = True
                
            def check_typing_visible(self, input_text):
                """Check if typing is visible to user."""
                try:
                    panel = tui._create_input_panel()
                    # Panel should contain the input text
                    panel_str = str(panel)
                    is_visible = input_text in panel_str
                    self.typing_visible.append({
                        'input': input_text,
                        'visible': is_visible,
                        'timestamp': time.time()
                    })
                    return is_visible
                except Exception as e:
                    self.typing_visible.append({
                        'input': input_text,
                        'visible': False,
                        'error': str(e),
                        'timestamp': time.time()
                    })
                    return False
            
            def check_layout_integrity(self):
                """Check if layout is intact."""
                try:
                    # Try to access layout components
                    _ = tui.layout["input"]
                    
                    # Try to create and render a panel
                    console = Console()
                    panel = tui._create_input_panel()
                    
                    # Check if panel renders without breaking
                    with console.capture() as capture:
                        console.print(panel)
                    rendered = capture.get()
                    
                    # Check for broken layout indicators
                    is_broken = (
                        len(rendered.strip()) < 10 or  # Too short
                        '?' in rendered or  # Broken characters
                        rendered.count('\n') < 2  # Not enough lines for a panel
                    )
                    
                    if is_broken:
                        self.layout_breaks.append({
                            'timestamp': time.time(),
                            'reason': 'Layout rendering appears broken',
                            'rendered_preview': rendered[:100]
                        })
                    
                    return not is_broken
                    
                except Exception as e:
                    self.layout_breaks.append({
                        'timestamp': time.time(),
                        'reason': str(e)
                    })
                    return False
        
        tracker = UserExperienceTracker()
        
        # Mock Live display that tracks refresh calls
        class UserExperienceLiveDisplay:
            def refresh(self):
                tracker.refresh_calls += 1
                print(f"🔄 Refresh #{tracker.refresh_calls}")
        
        tui.live_display = UserExperienceLiveDisplay()
        
        # Initialize layout like real startup
        print(f"\n--- STARTUP SIMULATION ---")
        tui.layout = Layout()
        tui.layout.split_column(Layout(name="input"))
        
        # Check initial layout
        startup_ok = tracker.check_layout_integrity()
        tracker.startup_layout_ok = startup_ok
        print(f"Startup layout OK: {'✅ YES' if startup_ok else '❌ NO'}")
        
        # Simulate real user typing sequence
        print(f"\n--- REAL USER TYPING SIMULATION ---")
        tui.state.current_input = ""
        
        # User types: "hello, can you help me?"
        user_message = "hello, can you help me?"
        
        print(f"User types: '{user_message}'")
        
        layout_intact = True
        typing_visible = True
        
        for i, char in enumerate(user_message, 1):
            print(f"\n  Character {i}: '{char}'")
            
            # Update input
            old_input = tui.state.current_input
            tui.state.current_input += char
            
            # Check if typing is visible BEFORE refresh
            visible_before = tracker.check_typing_visible(tui.state.current_input)
            
            # Check layout BEFORE refresh
            layout_before = tracker.check_layout_integrity()
            
            # Do the refresh (this is where problems might occur)
            refresh_start = time.time()
            tui._sync_refresh_display()
            refresh_duration = time.time() - refresh_start
            
            # Check if typing is visible AFTER refresh
            visible_after = tracker.check_typing_visible(tui.state.current_input)
            
            # Check layout AFTER refresh
            layout_after = tracker.check_layout_integrity()
            
            # Report results
            if visible_after:
                print(f"  ✅ Typing visible")
            else:
                print(f"  ❌ Typing NOT visible")
                typing_visible = False
            
            if layout_after:
                print(f"  ✅ Layout intact")
            else:
                print(f"  ❌ Layout BROKEN")
                layout_intact = False
                break  # Stop if layout breaks
            
            print(f"  📊 Refresh took {refresh_duration:.3f}s")
            
            # Small delay like real typing
            time.sleep(0.05)
        
        # Test Enter key
        print(f"\n--- ENTER KEY TEST ---")
        if layout_intact:
            print(f"Testing Enter key with input: '{tui.state.current_input}'")
            
            # Try different Enter key methods
            enter_methods_to_test = []
            if hasattr(tui, '_handle_enter_input'):
                enter_methods_to_test.append('_handle_enter_input')
            if hasattr(tui, '_handle_enter_key'):
                enter_methods_to_test.append('_handle_enter_key')
            if hasattr(tui, '_process_user_input'):
                enter_methods_to_test.append('_process_user_input')
            
            enter_worked = False
            for method_name in enter_methods_to_test:
                try:
                    print(f"  Trying {method_name}...")
                    method = getattr(tui, method_name)
                    result = method()
                    print(f"  ✅ {method_name}() -> {result}")
                    enter_worked = True
                    break
                except Exception as e:
                    print(f"  ❌ {method_name}() failed: {e}")
            
            tracker.enter_key_works = enter_worked
        else:
            print("⚠️  Cannot test Enter key - layout is broken")
        
        # Final assessment
        print(f"\n📊 REAL USER EXPERIENCE ASSESSMENT:")
        print("=" * 50)
        
        print(f"✅ Startup layout OK: {'YES' if tracker.startup_layout_ok else 'NO'}")
        print(f"✅ Typing visible: {'YES' if typing_visible else 'NO'}")
        print(f"✅ Layout stays intact: {'YES' if layout_intact else 'NO'}")  
        print(f"✅ Enter key works: {'YES' if tracker.enter_key_works else 'NO'}")
        print(f"📊 Total refresh calls: {tracker.refresh_calls}")
        print(f"📊 Layout breaks detected: {len(tracker.layout_breaks)}")
        
        # Detailed visibility analysis
        visible_count = sum(1 for v in tracker.typing_visible if v.get('visible', False))
        total_checks = len(tracker.typing_visible)
        visibility_percentage = (visible_count / total_checks * 100) if total_checks > 0 else 0
        
        print(f"📊 Typing visibility: {visible_count}/{total_checks} ({visibility_percentage:.1f}%)")
        
        # Show layout break details
        if tracker.layout_breaks:
            print(f"\n🚨 LAYOUT BREAK DETAILS:")
            for i, break_info in enumerate(tracker.layout_breaks[:3], 1):
                reason = break_info.get('reason', 'unknown')
                print(f"  Break {i}: {reason}")
        
        # Overall score
        issues = []
        if not tracker.startup_layout_ok:
            issues.append("Startup layout broken")
        if not typing_visible:
            issues.append("Typing not visible")
        if not layout_intact:
            issues.append("Layout corruption during typing")
        if not tracker.enter_key_works:
            issues.append("Enter key not working")
        
        if not issues:
            print(f"\n🎉 EXCELLENT: TUI works perfectly for real users!")
            return True
        else:
            print(f"\n❌ ISSUES FOUND ({len(issues)}):")
            for issue in issues:
                print(f"  • {issue}")
            print(f"\nThese match the user's reported problems!")
            return False
            
    except Exception as e:
        print(f"❌ Real user experience test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_startup_layout_lines():
    """Test the specific startup layout lines issue."""
    print(f"\n📊 STARTUP LAYOUT LINES TEST:")
    print("=" * 50)
    
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich import box
        
        console = Console()
        terminal_width = console.size.width
        
        print(f"Terminal width: {terminal_width}")
        
        # Test panel at different widths
        test_cases = [
            ("Auto width", None),
            ("Full terminal", terminal_width),
            ("Terminal - 1", terminal_width - 1),
            ("Terminal - 2", terminal_width - 2),
            ("80 chars", 80),
        ]
        
        for name, width in test_cases:
            try:
                panel = Panel(
                    f"Test content for {name} panel width",
                    title=name,
                    width=width,
                    box=box.ROUNDED
                )
                
                with console.capture() as capture:
                    console.print(panel)
                rendered = capture.get()
                
                lines = rendered.split('\n')
                if lines:
                    # Check top border line
                    top_line = lines[0] if lines else ""
                    actual_width = len(top_line)
                    
                    print(f"  {name}: {actual_width} chars")
                    
                    # Check for broken lines (too short)
                    expected_min = width - 5 if width else terminal_width - 10
                    if actual_width < expected_min:
                        print(f"    ⚠️  Line too short! Expected ~{expected_min}, got {actual_width}")
                        print(f"    Preview: '{top_line}'")
                    else:
                        print(f"    ✅ Line length OK")
                
            except Exception as e:
                print(f"  ❌ {name}: {e}")
                
    except Exception as e:
        print(f"❌ Startup layout lines test failed: {e}")

if __name__ == "__main__":
    test_startup_layout_lines()
    success = test_real_user_typing_experience()
    
    print(f"\n🎯 REAL USER EXPERIENCE TEST COMPLETE")
    print("=" * 50)
    
    if success:
        print("🎉 All tests passed - TUI should work for users!")
    else:
        print("❌ Issues found that match user reports")
        print("Use this output to identify and fix the real problems!")
    
    print("\nThis test simulates:")
    print("• Real user typing experience")  
    print("• Layout integrity during typing")
    print("• Enter key functionality")
    print("• Startup layout line issues")
    print("• Exact user-reported problems")