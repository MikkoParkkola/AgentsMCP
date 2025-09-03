#!/usr/bin/env python3
"""
TUI LAYOUT LINES DIAGNOSTIC
Debug why TUI boxes have broken/short lines on startup.
"""

import sys
import os
sys.path.insert(0, '/Users/mikko/github/AgentsMCP/src')

def debug_layout_line_lengths():
    """Debug layout line length issues that cause broken box borders."""
    print("🔍 Debugging TUI layout line length issues...")
    
    try:
        from rich.console import Console
        from rich.layout import Layout
        from rich.panel import Panel
        from rich import box
        import shutil
        
        # Check terminal size detection
        print("📊 TERMINAL SIZE DETECTION:")
        print("=" * 50)
        
        # Method 1: shutil
        try:
            size = shutil.get_terminal_size()
            print(f"shutil.get_terminal_size(): {size.columns}x{size.lines}")
        except Exception as e:
            print(f"❌ shutil.get_terminal_size() failed: {e}")
        
        # Method 2: os
        try:
            size = os.get_terminal_size()
            print(f"os.get_terminal_size(): {size.columns}x{size.lines}")
        except Exception as e:
            print(f"❌ os.get_terminal_size() failed: {e}")
        
        # Method 3: Rich Console
        console = Console()
        print(f"Rich Console size: {console.size}")
        print(f"Rich is_terminal: {console.is_terminal}")
        
        # Test different panel widths
        print(f"\n📊 PANEL WIDTH TESTING:")
        print("=" * 50)
        
        terminal_width = console.size.width
        print(f"Using terminal width: {terminal_width}")
        
        # Test panels at different widths
        test_widths = [
            ("Auto width", None),
            ("Terminal width", terminal_width),
            ("Terminal width - 1", terminal_width - 1),
            ("Terminal width - 2", terminal_width - 2),
            ("Terminal width - 4", terminal_width - 4),
            ("Fixed 80", 80),
            ("Fixed 40", 40),
        ]
        
        for name, width in test_widths:
            try:
                panel = Panel(
                    f"Test content for {name}",
                    title=f"{name} ({width})",
                    width=width,
                    box=box.ROUNDED
                )
                
                # Try to render and measure
                with console.capture() as capture:
                    console.print(panel)
                rendered_output = capture.get()
                
                # Analyze the rendered output
                lines = rendered_output.split('\n')
                if lines:
                    first_line = lines[0]
                    actual_width = len(first_line.rstrip('\n\r'))
                    print(f"✅ {name}: Expected {width}, Actual {actual_width}")
                    
                    # Check for truncation
                    if width and actual_width < width - 2:  # Allow some margin
                        print(f"⚠️  {name}: TRUNCATED (expected ~{width}, got {actual_width})")
                    
                else:
                    print(f"❌ {name}: No output rendered")
                    
            except Exception as e:
                print(f"❌ {name}: Error - {e}")
        
        # Test the actual TUI layout structure
        print(f"\n📊 TUI LAYOUT STRUCTURE TEST:")
        print("=" * 50)
        
        try:
            from agentsmcp.ui.cli_app import CLIConfig
            from agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface
            
            class DiagnosticConfig:
                debug_mode = True
                verbose = True
            
            config = DiagnosticConfig()
            tui = RevolutionaryTUIInterface(cli_config=config)
            
            # Test input panel creation at different terminal sizes
            original_size = console.size
            
            # Simulate different terminal sizes
            test_sizes = [
                ("Current", original_size.columns, original_size.lines),
                ("Narrow", 60, 24),
                ("Wide", 120, 30),
                ("Very narrow", 40, 20),
            ]
            
            for name, cols, lines in test_sizes:
                print(f"\nTesting {name} terminal ({cols}x{lines}):")
                
                # Mock terminal size (this is tricky, but we can test panel creation)
                tui.state.current_input = f"Test input for {name} terminal"
                
                try:
                    panel = tui._create_input_panel()
                    print(f"  ✅ Input panel created: {type(panel).__name__}")
                    
                    # Try to render it
                    with console.capture() as capture:
                        console.print(panel)
                    rendered = capture.get()
                    
                    lines_rendered = rendered.split('\n')
                    if lines_rendered:
                        first_line = lines_rendered[0]
                        panel_width = len(first_line.rstrip())
                        print(f"  📏 Panel rendered width: {panel_width}")
                        
                        # Check if it looks broken
                        if panel_width < 20:  # Suspiciously narrow
                            print(f"  ⚠️  Panel seems too narrow! ({panel_width} chars)")
                        elif panel_width > cols:  # Too wide for terminal
                            print(f"  ⚠️  Panel wider than terminal! ({panel_width} > {cols})")
                        else:
                            print(f"  ✅ Panel width looks reasonable")
                    
                except Exception as e:
                    print(f"  ❌ Panel creation failed: {e}")
                    
        except Exception as e:
            print(f"❌ TUI layout structure test failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Box character testing
        print(f"\n📊 BOX CHARACTER RENDERING TEST:")
        print("=" * 50)
        
        # Test different box styles
        box_styles = [
            ("ROUNDED", box.ROUNDED),
            ("ASCII", box.ASCII),
            ("DOUBLE", box.DOUBLE),
            ("HEAVY", box.HEAVY),
            ("ASCII2", box.ASCII2),
        ]
        
        for name, box_style in box_styles:
            try:
                panel = Panel(
                    f"Testing {name} box style",
                    title=name,
                    box=box_style,
                    width=40
                )
                
                with console.capture() as capture:
                    console.print(panel)
                rendered = capture.get()
                
                print(f"✅ {name} box style renders OK")
                
                # Show first line to check for broken characters
                lines = rendered.split('\n')
                if lines:
                    first_line = lines[0]
                    print(f"  First line: '{first_line[:50]}{'...' if len(first_line) > 50 else ''}'")
                    
                    # Check for common broken character patterns
                    if '?' in first_line or '□' in first_line:
                        print(f"  ⚠️  Possible broken characters detected in {name}")
                    
            except Exception as e:
                print(f"❌ {name} box style failed: {e}")
        
    except Exception as e:
        print(f"❌ Layout line length diagnostic failed: {e}")
        import traceback
        traceback.print_exc()

def debug_terminal_environment_detailed():
    """Detailed terminal environment analysis."""
    print(f"\n📊 DETAILED TERMINAL ENVIRONMENT:")
    print("=" * 50)
    
    import os
    import sys
    
    # Environment variables
    env_vars = [
        'TERM', 'COLORTERM', 'TERM_PROGRAM', 'TERM_PROGRAM_VERSION',
        'COLUMNS', 'LINES', 'SSH_TTY', 'TMUX', 'INSIDE_EMACS'
    ]
    
    print("Environment variables:")
    for var in env_vars:
        value = os.environ.get(var, 'not set')
        print(f"  ${var}: {value}")
    
    # TTY status
    print(f"\nTTY status:")
    print(f"  sys.stdin.isatty(): {sys.stdin.isatty()}")
    print(f"  sys.stdout.isatty(): {sys.stdout.isatty()}")
    print(f"  sys.stderr.isatty(): {sys.stderr.isatty()}")
    
    # File descriptors
    try:
        print(f"  stdin fileno: {sys.stdin.fileno()}")
        print(f"  stdout fileno: {sys.stdout.fileno()}")
        print(f"  stderr fileno: {sys.stderr.fileno()}")
    except:
        print("  ❌ Cannot get file descriptors")

if __name__ == "__main__":
    debug_terminal_environment_detailed()
    debug_layout_line_lengths()
    print(f"\n🎯 LAYOUT LINES DIAGNOSTIC COMPLETE")
    print("=" * 50)
    print("This diagnostic helps identify:")
    print("• Terminal size detection issues")
    print("• Panel width calculation problems")
    print("• Box character rendering issues")
    print("• TUI layout structure problems")
    print("Share this output to help fix the broken box lines!")