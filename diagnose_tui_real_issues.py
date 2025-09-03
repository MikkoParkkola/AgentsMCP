#!/usr/bin/env python3
"""
REAL TUI ISSUES DIAGNOSTIC
Comprehensive diagnostics to identify actual TUI problems in real environment.
Run this to troubleshoot the actual issues you're experiencing.
"""

import sys
import os
import time
import signal
sys.path.insert(0, '/Users/mikko/github/AgentsMCP/src')

def diagnose_terminal_environment():
    """Diagnose terminal environment and capabilities."""
    print("🔍 DIAGNOSING TERMINAL ENVIRONMENT")
    print("=" * 60)
    
    # Basic terminal info
    print(f"📊 Terminal Type: {os.environ.get('TERM', 'unknown')}")
    print(f"📊 TTY: {os.isatty(sys.stdout.fileno())}")
    print(f"📊 Python stdout TTY: {sys.stdout.isatty()}")
    print(f"📊 Python stdin TTY: {sys.stdin.isatty()}")
    
    # Terminal size detection
    try:
        import shutil
        size = shutil.get_terminal_size()
        print(f"📊 Terminal size (shutil): {size.columns}x{size.lines}")
    except:
        print("❌ Failed to get terminal size via shutil")
    
    try:
        import os
        size = os.get_terminal_size()
        print(f"📊 Terminal size (os): {size.columns}x{size.lines}")
    except:
        print("❌ Failed to get terminal size via os")
    
    # Rich terminal detection
    try:
        from rich.console import Console
        console = Console()
        print(f"📊 Rich console size: {console.size}")
        print(f"📊 Rich is terminal: {console.is_terminal}")
        print(f"📊 Rich legacy windows: {console.legacy_windows}")
        print(f"📊 Rich color system: {console.color_system}")
        print(f"📊 Rich encoding: {console.encoding}")
    except Exception as e:
        print(f"❌ Rich console error: {e}")
    
    # Environment variables
    terminal_vars = ['COLUMNS', 'LINES', 'TERM', 'COLORTERM', 'TERM_PROGRAM']
    for var in terminal_vars:
        value = os.environ.get(var, 'not set')
        print(f"📊 ${var}: {value}")

def diagnose_tui_startup():
    """Diagnose TUI startup and initial layout."""
    print(f"\n🔍 DIAGNOSING TUI STARTUP")
    print("=" * 60)
    
    try:
        from agentsmcp.ui.cli_app import CLIConfig
        from agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface
        from rich.console import Console
        from rich.layout import Layout
        from rich.panel import Panel
        from rich import box
        
        # Create TUI
        class DiagnosticConfig:
            debug_mode = True
            verbose = True
        
        config = DiagnosticConfig()
        console = Console()
        
        print(f"✅ Console created: {console.size}")
        
        # Test layout creation
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="input", size=5)
        )
        
        # Test panel creation with different sizes
        terminal_width = console.size.width
        
        print(f"📊 Testing panel creation at width {terminal_width}")
        
        # Create test panels
        test_panels = {
            "Full width": Panel("Test content", width=terminal_width),
            "Width-1": Panel("Test content", width=terminal_width-1),
            "Width-2": Panel("Test content", width=terminal_width-2),
            "Auto width": Panel("Test content"),
        }
        
        for name, panel in test_panels.items():
            try:
                # Try to render panel
                console.print(panel, end="")
                print(f"✅ {name} panel: OK")
            except Exception as e:
                print(f"❌ {name} panel: {e}")
        
        # Test layout rendering
        try:
            layout["header"].update(Panel("Header", title="Header"))
            layout["body"].update(Panel("Body content", title="Body"))  
            layout["input"].update(Panel("Input area", title="Input"))
            
            print(f"✅ Layout structure created successfully")
            
            # Try to render layout
            console.print(layout)
            print(f"✅ Layout renders successfully")
            
        except Exception as e:
            print(f"❌ Layout error: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ TUI startup diagnostic failed: {e}")
        import traceback
        traceback.print_exc()

def diagnose_input_handling():
    """Diagnose input handling and key processing."""
    print(f"\n🔍 DIAGNOSING INPUT HANDLING")
    print("=" * 60)
    
    try:
        from agentsmcp.ui.cli_app import CLIConfig
        from agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface
        
        class DiagnosticConfig:
            debug_mode = True
            verbose = True
        
        config = DiagnosticConfig()
        tui = RevolutionaryTUIInterface(cli_config=config)
        
        # Check input handling methods
        input_methods = [
            '_handle_character_input',
            '_handle_enter_input',
            '_sync_refresh_display',
            '_process_user_input',
            'run'
        ]
        
        print("📊 Input handling methods:")
        for method in input_methods:
            exists = hasattr(tui, method)
            if exists:
                method_obj = getattr(tui, method)
                is_async = hasattr(method_obj, '__await__') or str(type(method_obj)) == "<class 'coroutine'>"
                print(f"  ✅ {method}: {'async' if is_async else 'sync'}")
            else:
                print(f"  ❌ {method}: MISSING")
        
        # Check state initialization
        print("\n📊 State initialization:")
        if hasattr(tui, 'state'):
            state = tui.state
            state_attrs = ['current_input', 'conversation_history', 'is_processing']
            for attr in state_attrs:
                if hasattr(state, attr):
                    value = getattr(state, attr)
                    print(f"  ✅ state.{attr}: {type(value).__name__} = {repr(value)[:50]}")
                else:
                    print(f"  ❌ state.{attr}: MISSING")
        else:
            print("  ❌ No state object")
        
        # Check event system
        print("\n📊 Event system:")
        if hasattr(tui, 'event_system'):
            print(f"  ✅ event_system: {type(tui.event_system).__name__}")
        else:
            print(f"  ❌ event_system: MISSING")
        
        # Check orchestrator
        print("\n📊 Orchestrator:")
        if hasattr(tui, 'orchestrator'):
            if tui.orchestrator:
                print(f"  ✅ orchestrator: {type(tui.orchestrator).__name__}")
            else:
                print(f"  ⚠️  orchestrator: None (will use fallback)")
        else:
            print(f"  ❌ orchestrator: MISSING")
            
    except Exception as e:
        print(f"❌ Input handling diagnostic failed: {e}")
        import traceback
        traceback.print_exc()

def diagnose_key_binding():
    """Diagnose actual key input processing."""
    print(f"\n🔍 DIAGNOSING KEY BINDING")
    print("=" * 60)
    
    # Look for the actual input loop
    try:
        import inspect
        from agentsmcp.ui.cli_app import CLIConfig
        from agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface
        
        class DiagnosticConfig:
            debug_mode = True
            verbose = True
        
        config = DiagnosticConfig()
        tui = RevolutionaryTUIInterface(cli_config=config)
        
        # Find methods that might handle keyboard input
        all_methods = [method for method in dir(tui) if not method.startswith('__')]
        input_related = [m for m in all_methods if any(keyword in m.lower() for keyword in ['input', 'key', 'char', 'enter'])]
        
        print("📊 Input-related methods found:")
        for method in input_related:
            method_obj = getattr(tui, method)
            if callable(method_obj):
                try:
                    sig = inspect.signature(method_obj)
                    print(f"  ✅ {method}{sig}")
                except:
                    print(f"  ✅ {method}(...)")
        
        # Look for keyboard input handling in the main run method
        if hasattr(tui, 'run'):
            run_method = tui.run
            if hasattr(run_method, '__code__'):
                code = run_method.__code__
                print(f"\n📊 run() method analysis:")
                print(f"  Lines of code: {code.co_code.__len__()}")
                print(f"  Variables: {code.co_varnames}")
        
        # Check for terminal input handling
        print(f"\n📊 Looking for terminal input handling...")
        
        # Search for sys.stdin usage
        import re
        tui_file = "/Users/mikko/github/AgentsMCP/src/agentsmcp/ui/v2/revolutionary_tui_interface.py"
        try:
            with open(tui_file, 'r') as f:
                content = f.read()
            
            input_patterns = [
                r'sys\.stdin',
                r'input\(\)',
                r'getch\(\)',
                r'keyboard',
                r'\.read\(1\)',
                r'select\.select',
                r'termios',
                r'tty\.setraw'
            ]
            
            for pattern in input_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    print(f"  ✅ Found {pattern}: {len(matches)} occurrences")
                else:
                    print(f"  ❌ No {pattern} found")
                    
        except Exception as e:
            print(f"❌ File analysis failed: {e}")
            
    except Exception as e:
        print(f"❌ Key binding diagnostic failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all diagnostics."""
    print("🚀 COMPREHENSIVE TUI DIAGNOSTICS")
    print("This will help identify the real issues in your TUI environment")
    print("=" * 60)
    
    try:
        diagnose_terminal_environment()
        diagnose_tui_startup()
        diagnose_input_handling()
        diagnose_key_binding()
        
        print(f"\n🎯 DIAGNOSTIC COMPLETE")
        print("=" * 60)
        print("📧 Please share this output to help identify the exact issues!")
        print("The diagnostic covers:")
        print("  • Terminal environment and size detection")
        print("  • TUI startup and layout creation")
        print("  • Input handling method availability")
        print("  • Key binding and input processing")
        
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()