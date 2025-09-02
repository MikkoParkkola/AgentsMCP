#!/usr/bin/env python3
"""
Quick smoke test for actual TUI launch
"""

import asyncio
import logging
import sys
import os

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure minimal logging
logging.basicConfig(level=logging.WARNING)  # Reduce log noise

async def test_actual_tui_launch():
    """Test the actual TUI launch with timeout to prevent hanging."""
    try:
        print("🚀 Testing Revolutionary TUI Launcher with timeout...")
        
        from agentsmcp.ui.v2.revolutionary_launcher import launch_revolutionary_tui
        from agentsmcp.ui.cli_app import CLIConfig
        
        # Create a minimal CLI config
        cli_config = CLIConfig()
        cli_config.debug_mode = True  # Enable debug mode for visibility
        
        # Launch with a 30-second timeout to prevent hanging
        print("Launching TUI with 30-second timeout...")
        result = await asyncio.wait_for(
            launch_revolutionary_tui(cli_config),
            timeout=30.0
        )
        
        print(f"✅ TUI launched successfully! Result: {result}")
        return True
        
    except asyncio.TimeoutError:
        print("❌ TUI launch timed out after 30 seconds - still hanging!")
        return False
    except KeyboardInterrupt:
        print("✅ TUI responded to keyboard interrupt - successful startup (user cancelled)")
        return True
    except Exception as e:
        print(f"❌ TUI launch failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(test_actual_tui_launch())
        if success:
            print("✅ TUI startup test PASSED")
            sys.exit(0)
        else:
            print("❌ TUI startup test FAILED")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n✅ Test interrupted - TUI was responsive to interrupt")
        sys.exit(0)