#!/usr/bin/env python3
"""
Debug V3 TUI Issue
Trace exact execution flow to find where the TUI exits prematurely
"""

import sys
import os
import asyncio
import traceback

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def debug_print(msg):
    """Print debug message with timestamp and flush immediately."""
    import time
    timestamp = time.strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] DEBUG: {msg}", flush=True)

async def debug_launch_tui():
    """Debug version of launch_tui with detailed tracing."""
    debug_print("Starting debug_launch_tui()")
    
    try:
        from src.agentsmcp.ui.v3.tui_launcher import TUILauncher
        debug_print("TUILauncher imported successfully")
        
        launcher = TUILauncher()
        debug_print("TUILauncher instance created")
        
        debug_print("About to call launcher.run_main_loop()")
        result = await launcher.run_main_loop()
        debug_print(f"launcher.run_main_loop() returned: {result}")
        
        return result
        
    except Exception as e:
        debug_print(f"Exception in debug_launch_tui: {e}")
        debug_print(f"Exception traceback: {traceback.format_exc()}")
        return 1

def main():
    """Main debug function."""
    debug_print("Starting main debug function")
    
    try:
        debug_print("About to call asyncio.run(debug_launch_tui())")
        result = asyncio.run(debug_launch_tui())
        debug_print(f"asyncio.run() completed with result: {result}")
        return result
        
    except KeyboardInterrupt:
        debug_print("Received KeyboardInterrupt")
        print("\n👋 Goodbye!")
        return 0
    except Exception as e:
        debug_print(f"Exception in main: {e}")
        debug_print(f"Exception traceback: {traceback.format_exc()}")
        print(f"❌ Fatal error: {e}")
        return 1
    finally:
        debug_print("main() function finished")

if __name__ == "__main__":
    debug_print("Script started")
    exit_code = main()
    debug_print(f"Script exiting with code: {exit_code}")
    sys.exit(exit_code)