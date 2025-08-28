#!/usr/bin/env python3
"""
Debug the specific line concatenation issue seen in the terminal output.
The problem appears to be that consecutive UI frames are being printed 
without proper separation, causing footer/header concatenation.
"""

import sys
from pathlib import Path
from io import StringIO

# Add src to path for imports  
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_rich_live_output_separation():
    """Test if Rich Live properly separates consecutive outputs."""
    print("=== Testing Rich Live Output Separation ===")
    
    from rich.live import Live
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    import time
    import asyncio
    
    # Capture output to analyze
    output_buffer = StringIO()
    console = Console(file=output_buffer, width=80, legacy_windows=False)
    
    frame_count = 0
    
    def generate_frame():
        nonlocal frame_count
        frame_count += 1
        return Panel(
            Text(f"Frame {frame_count}\nTime: {time.time():.3f}"),
            title=f"Test Frame {frame_count}",
            border_style="blue"
        )
    
    print("Generating 3 consecutive frames with Rich Live...")
    
    try:
        # This simulates what happens in the TUI event loop
        with Live(generate_frame(), console=console, refresh_per_second=10) as live:
            time.sleep(0.1)  # Let first frame render
            
            # Force additional updates like the TUI does
            live.update(generate_frame())
            time.sleep(0.1)
            
            live.update(generate_frame()) 
            time.sleep(0.1)
            
        # Analyze the captured output
        output = output_buffer.getvalue()
        
        print(f"Captured {len(output)} characters of output")
        lines = output.split('\n')
        print(f"Split into {len(lines)} lines")
        
        # Look for concatenated borders (the specific issue reported)
        concatenation_issues = []
        
        for i, line in enumerate(lines):
            # Look for footer-header concatenations specifically
            if '╰─' in line and '╭─' in line:
                concatenation_issues.append(f"Line {i+1}: {repr(line)}")
            
            # Also look for multiple panels on same line
            panel_count = line.count('╭─')
            if panel_count > 1:
                concatenation_issues.append(f"Line {i+1}: Multiple panels on same line: {repr(line)}")
        
        if concatenation_issues:
            print("❌ Found line concatenation issues:")
            for issue in concatenation_issues:
                print(f"  {issue}")
            
            # Show some context
            print("\nFirst 20 lines of output:")
            for j, line in enumerate(lines[:20]):
                print(f"{j+1:2d}: {repr(line)}")
            
            return False
        else:
            print("✓ No line concatenation found in Rich Live output")
            
            # But let's check for proper separation
            frame_separations = 0
            for line in lines:
                if line.strip() == '':  # Empty lines separate frames
                    frame_separations += 1
                    
            print(f"Found {frame_separations} empty line separations")
            
            return True
            
    except Exception as e:
        print(f"❌ Rich Live test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_modern_tui_render_output():
    """Test the actual ModernTUI render output for concatenation issues."""
    print("\n=== Testing ModernTUI Render Output ===")
    
    from agentsmcp.ui.modern_tui import ModernTUI
    
    # Create minimal TUI setup
    class MockConfig:
        interface_mode = "tui"
    
    class MockThemeManager:
        def rich_theme(self): return None
    
    class MockConversationManager:
        def get_history(self): return []
    
    class MockOrchestrationManager:
        def user_settings(self): return {}
    
    try:
        tui = ModernTUI(
            config=MockConfig(),
            theme_manager=MockThemeManager(), 
            conversation_manager=MockConversationManager(),
            orchestration_manager=MockOrchestrationManager(),
            no_welcome=True
        )
        
        # Capture the rendered output
        output_buffer = StringIO()
        from rich.console import Console
        test_console = Console(file=output_buffer, width=80, legacy_windows=False)
        
        # Mock the console to capture output
        original_console = tui._console
        tui._console = test_console
        
        # Build layout and render
        tui._layout = tui._build_layout()
        rendered_layout = tui._render()
        
        # Print the layout to capture output
        test_console.print(rendered_layout)
        
        # Restore original console
        tui._console = original_console
        
        # Analyze output
        output = output_buffer.getvalue()
        lines = output.split('\n')
        
        print(f"ModernTUI render produced {len(lines)} lines")
        
        # Check for concatenation issues
        issues = []
        
        for i, line in enumerate(lines):
            # Look for the specific pattern reported: ╰──────╯╭─────────
            if '╰─' in line and '╭─' in line:
                # Check if it's a problematic concatenation (no space between components)
                parts = line.split('╰─')
                if len(parts) > 1:
                    for part in parts[1:]:  # Skip first part before ╰─
                        if part.startswith('─') and '╭─' in part:
                            # This looks like footer-header concatenation
                            issues.append(f"Line {i+1}: Footer-header concatenation: {repr(line)}")
                            break
        
        if issues:
            print("❌ Found concatenation issues in ModernTUI render:")
            for issue in issues:
                print(f"  {issue}")
                
            # Show context around issues
            for issue in issues[:3]:  # Show first 3 issues
                line_num = int(issue.split(':')[0].split()[1]) - 1
                start = max(0, line_num - 2)
                end = min(len(lines), line_num + 3)
                
                print(f"\nContext around line {line_num + 1}:")
                for j in range(start, end):
                    marker = ">>> " if j == line_num else "    "
                    print(f"{marker}{j+1:2d}: {repr(lines[j])}")
            
            return False
        else:
            print("✓ No concatenation issues found in ModernTUI render")
            return True
            
    except Exception as e:
        print(f"❌ ModernTUI render test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_layout_structure():
    """Test if the Layout structure itself is causing issues."""
    print("\n=== Testing Layout Structure ===") 
    
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.console import Console
    from rich.text import Text
    
    # Create the same layout structure as ModernTUI
    layout = Layout(name="root")
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main_area", ratio=1), 
        Layout(name="footer", size=3),
    )
    
    # Populate with content similar to ModernTUI
    layout["header"].update(Panel(Text("Header Content"), border_style="blue"))
    layout["main_area"].update(Panel(Text("Main Content\nLine 2\nLine 3"), border_style="green"))
    layout["footer"].update(Panel(Text("Footer Content"), border_style="red"))
    
    # Render multiple times to simulate the TUI loop
    output_buffer = StringIO()
    console = Console(file=output_buffer, width=80, legacy_windows=False)
    
    print("Rendering layout 3 times consecutively...")
    
    # This simulates what happens in the problematic TUI loop
    console.print(layout)
    console.print(layout)  # Second render without separation
    console.print(layout)  # Third render without separation
    
    output = output_buffer.getvalue()
    lines = output.split('\n')
    
    print(f"3 layout renders produced {len(lines)} lines")
    
    # Count how many complete layouts we see
    header_count = sum(1 for line in lines if "Header Content" in line)
    footer_count = sum(1 for line in lines if "Footer Content" in line)
    
    print(f"Found {header_count} headers and {footer_count} footers")
    
    if header_count != 3 or footer_count != 3:
        print("⚠️  Unexpected number of complete layouts")
    
    # Look for concatenated borders between consecutive layouts
    concatenations = []
    for i, line in enumerate(lines):
        if '╰─' in line and '╭─' in line and line.count('╰─') > 1:
            concatenations.append(f"Line {i+1}: {repr(line)}")
    
    if concatenations:
        print("❌ Found layout concatenations:")
        for concat in concatenations:
            print(f"  {concat}")
        return False
    else:
        print("✓ Layout renders appear properly separated")
        
        # But check if there's adequate spacing
        empty_lines_between = 0
        for i, line in enumerate(lines[:-1]):
            if line.strip() == '' and lines[i+1].strip() != '':
                empty_lines_between += 1
        
        print(f"Found {empty_lines_between} empty lines providing separation")
        return True

def main():
    """Run all line concatenation debug tests."""
    print("🐛 Line Concatenation Debug Session")
    print("=" * 50)
    
    results = {
        "Rich Live Output Separation": test_rich_live_output_separation(),
        "ModernTUI Render Output": test_modern_tui_render_output(),
        "Layout Structure": test_layout_structure(),
    }
    
    print("\n" + "=" * 50)
    print("📊 Results:")
    
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        icon = "✅" if passed else "❌"
        print(f"{icon} {test_name}: {status}")
    
    if not all(results.values()):
        print("\n💡 Potential fixes for line concatenation:")
        print("1. Add explicit newlines between Rich Live updates")
        print("2. Use Live.refresh() sparingly to avoid rapid redraws")
        print("3. Implement proper frame buffering") 
        print("4. Consider using Rich.print() with separators instead of Live updates")
    else:
        print("\n🎉 No line concatenation issues found in isolated tests!")
        print("The problem may occur in the integration or specific terminal environments.")

if __name__ == "__main__":
    main()