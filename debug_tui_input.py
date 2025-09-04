#!/usr/bin/env python3
"""Debug version of TUI with extensive input logging."""

import sys
import os
sys.path.insert(0, '.')

def debug_tui_input():
    """Run TUI with extensive input debugging."""
    print("🔍 DEBUG: Starting TUI with input debugging...")
    
    try:
        from src.agentsmcp.ui.v3.tui_launcher import TUILauncher
        from src.agentsmcp.ui.v3.terminal_capabilities import detect_terminal_capabilities
        
        # Check capabilities
        caps = detect_terminal_capabilities()
        print(f"🔍 DEBUG: Terminal capabilities:")
        print(f"   TTY: {caps.is_tty}")
        print(f"   Rich support: {caps.supports_rich}")
        print(f"   Size: {caps.width}x{caps.height}")
        
        # Create launcher
        launcher = TUILauncher()
        print(f"🔍 DEBUG: TUILauncher created")
        
        # Initialize
        if launcher.initialize():
            print(f"🔍 DEBUG: TUILauncher initialized successfully")
            print(f"🔍 DEBUG: Current renderer type: {type(launcher.current_renderer).__name__}")
            
            # Check if it's RichTUIRenderer
            if hasattr(launcher.current_renderer, 'handle_input'):
                print(f"🔍 DEBUG: Renderer has handle_input method")
                
                # Get the method source to verify which version is being used
                import inspect
                try:
                    source = inspect.getsource(launcher.current_renderer.handle_input)
                    if "select.select" in source:
                        print("🔍 DEBUG: ✅ NEW input handling code is being used (select.select found)")
                    elif "input()" in source:
                        print("🔍 DEBUG: ❌ OLD input handling code is being used (input() found)")
                    else:
                        print("🔍 DEBUG: ⚠️  UNKNOWN input handling code")
                        
                    print(f"🔍 DEBUG: First 200 chars of handle_input method:")
                    print(f"   {source[:200]}...")
                except Exception as e:
                    print(f"🔍 DEBUG: Could not inspect handle_input method: {e}")
            
            print(f"🔍 DEBUG: About to start main loop...")
            exit_code = launcher.run()
            print(f"🔍 DEBUG: Main loop completed with exit code: {exit_code}")
            
        else:
            print(f"🔍 DEBUG: ❌ TUILauncher initialization failed")
            
    except Exception as e:
        print(f"🔍 DEBUG: ❌ Exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_tui_input()