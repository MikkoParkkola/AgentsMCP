#!/usr/bin/env python3
"""
DEBUG LAYOUT CORRUPTION - REAL ISSUE INVESTIGATION
Focus on the actual layout corruption that happens when typing starts.
"""

import sys
import os
import time
sys.path.insert(0, '/Users/mikko/github/AgentsMCP/src')

def debug_layout_corruption():
    """Debug the real layout corruption issue during typing."""
    print("🔍 INVESTIGATING REAL LAYOUT CORRUPTION")
    print("=" * 50)
    
    try:
        from agentsmcp.ui.cli_app import CLIConfig
        from agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface
        from rich.layout import Layout
        from rich.console import Console
        from rich.panel import Panel
        from rich import box
        
        # Create TUI like real usage
        class TestConfig:
            debug_mode = False  # Like real user
            verbose = False
        
        config = TestConfig()
        tui = RevolutionaryTUIInterface(cli_config=config)
        
        console = Console()
        print(f"Terminal size: {console.size}")
        print(f"TTY: stdin={sys.stdin.isatty()}, stdout={sys.stdout.isatty()}")
        
        # Create layout like real TUI startup
        print(f"\n--- LAYOUT CREATION TEST ---")
        tui.layout = Layout()
        
        # Try to replicate the layout structure
        try:
            # Split into main sections like real TUI
            tui.layout.split_column(
                Layout(name="header", size=3),
                Layout(name="main"),
                Layout(name="input", size=5)
            )
            print("✅ Basic layout structure created")
        except Exception as e:
            print(f"❌ Layout structure creation failed: {e}")
            return
        
        # Test initial panel creation and rendering
        print(f"\n--- INITIAL PANEL RENDERING TEST ---")
        
        try:
            # Create input panel
            tui.state.current_input = ""
            input_panel = tui._create_input_panel()
            print(f"Input panel type: {type(input_panel)}")
            
            # Try to render it in a panel container
            panel_container = Panel(
                input_panel,
                title="Input",
                box=box.ROUNDED
            )
            
            print("✅ Initial panel container created")
            
            # Try to render to check for layout issues
            with console.capture() as capture:
                console.print(panel_container)
            initial_render = capture.get()
            
            lines = initial_render.split('\n')
            print(f"Initial render: {len(lines)} lines")
            
            # Check for broken lines
            if lines:
                top_line = lines[0]
                print(f"Top line length: {len(top_line)} chars")
                print(f"Top line preview: '{top_line[:50]}...'")
                
                # Check if line is suspiciously short
                terminal_width = console.size.width
                if len(top_line) < terminal_width - 10:
                    print(f"⚠️  STARTUP LAYOUT ISSUE: Top line too short!")
                    print(f"   Expected ~{terminal_width}, got {len(top_line)}")
                
        except Exception as e:
            print(f"❌ Initial panel rendering failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test what happens when we start typing
        print(f"\n--- TYPING CORRUPTION TEST ---")
        
        # Simulate typing one character at a time and check layout
        test_input = "hello world"
        
        for i, char in enumerate(test_input, 1):
            print(f"\n  Testing character {i}: '{char}'")
            
            # Update input
            tui.state.current_input += char
            print(f"  Current input: '{tui.state.current_input}'")
            
            # Create new panel
            try:
                input_panel = tui._create_input_panel()
                panel_container = Panel(
                    input_panel,
                    title=f"Input (char {i})",
                    box=box.ROUNDED
                )
                
                # Try to render
                with console.capture() as capture:
                    console.print(panel_container)
                render_result = capture.get()
                
                # Check for corruption
                lines = render_result.split('\n')
                if lines:
                    top_line = lines[0]
                    line_len = len(top_line)
                    
                    print(f"  Rendered line length: {line_len}")
                    
                    # Check for broken rendering
                    if line_len < 10:  # Suspiciously short
                        print(f"  🚨 CORRUPTION DETECTED!")
                        print(f"  Line content: '{top_line}'")
                        print(f"  Full render preview:")
                        print(render_result[:200])
                        break
                    elif '?' in top_line or len(lines) < 2:
                        print(f"  ⚠️  POTENTIAL CORRUPTION")
                        print(f"  Line: '{top_line[:30]}...'")
                        
                print(f"  ✅ Character {i} OK")
                
            except Exception as e:
                print(f"  ❌ CORRUPTION at character {i}: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # Test layout object access during typing
        print(f"\n--- LAYOUT ACCESS TEST ---")
        
        try:
            # Try to access layout parts
            header_layout = tui.layout["header"]
            main_layout = tui.layout["main"] 
            input_layout = tui.layout["input"]
            
            print(f"✅ Layout access works:")
            print(f"  Header: {type(header_layout)}")
            print(f"  Main: {type(main_layout)}")
            print(f"  Input: {type(input_layout)}")
            
            # Try to update input layout content
            input_layout.update(Panel("Test content", title="Test"))
            print(f"✅ Layout update works")
            
        except Exception as e:
            print(f"❌ Layout access failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test the actual refresh display method
        print(f"\n--- REFRESH DISPLAY METHOD TEST ---")
        
        # Mock the live_display 
        class TestLiveDisplay:
            def __init__(self):
                self.refresh_count = 0
                self.last_layout = None
            
            def refresh(self):
                self.refresh_count += 1
                # Capture the current layout state
                self.last_layout = str(tui.layout) if tui.layout else "No layout"
                print(f"    Refresh #{self.refresh_count}")
        
        tui.live_display = TestLiveDisplay()
        
        try:
            print("Testing refresh display method...")
            tui._sync_refresh_display()
            print(f"✅ Refresh method works (calls: {tui.live_display.refresh_count})")
        except Exception as e:
            print(f"❌ Refresh method failed: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Layout corruption debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_layout_corruption()