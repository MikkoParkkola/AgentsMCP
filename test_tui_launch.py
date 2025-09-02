#!/usr/bin/env python3
"""
Test script to demonstrate TUI launch behavior and diagnose issues.
"""
import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, 'src')

from agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface


async def test_tui_launch():
    """Test TUI launch with detailed logging."""
    print("Testing Revolutionary TUI Interface launch...")
    print(f"Python version: {sys.version}")
    print(f"TTY detection - stdin: {sys.stdin.isatty()}, stdout: {sys.stdout.isatty()}")
    print(f"Terminal: {os.environ.get('TERM', 'not set')}")
    print()
    
    # Create interface with debug mode
    class MockCliConfig:
        debug_mode = True
    
    interface = RevolutionaryTUIInterface(cli_config=MockCliConfig())
    
    # Test initialization
    print("Testing interface initialization...")
    init_success = await interface.initialize()
    print(f"Initialization successful: {init_success}")
    
    if not init_success:
        print("❌ Initialization failed - cannot proceed")
        return 1
    
    print("✅ Initialization successful")
    print()
    
    # Test run method (will return immediately in non-TTY)
    print("Testing run method...")
    try:
        exit_code = await interface.run()
        print(f"Run completed with exit code: {exit_code}")
        return exit_code
    except Exception as e:
        print(f"❌ Run failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Cleanup
        await interface._cleanup()


if __name__ == "__main__":
    exit_code = asyncio.run(test_tui_launch())
    print(f"\nTest completed with exit code: {exit_code}")
    sys.exit(exit_code)