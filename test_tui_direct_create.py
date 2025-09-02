#!/usr/bin/env python3
"""
Test direct creation of Revolutionary TUI Interface to isolate the hang
"""

import asyncio
import logging
import sys
import os

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure minimal logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def test_direct_tui_creation():
    """Test direct creation of Revolutionary TUI Interface."""
    try:
        logger.info("Testing direct Revolutionary TUI Interface creation...")
        
        from agentsmcp.ui.v2.revolutionary_tui_interface import create_revolutionary_interface
        from agentsmcp.ui.cli_app import CLIConfig
        
        # Create a minimal CLI config
        cli_config = CLIConfig()
        cli_config.debug_mode = True
        
        logger.info("Calling create_revolutionary_interface with timeout...")
        
        # Test with a 20-second timeout
        result = await asyncio.wait_for(
            create_revolutionary_interface(
                cli_config=cli_config,
                orchestrator_integration=None,
                revolutionary_components={}
            ),
            timeout=20.0
        )
        
        logger.info(f"✅ create_revolutionary_interface succeeded: {type(result)}")
        return True
        
    except asyncio.TimeoutError:
        logger.error("❌ create_revolutionary_interface timed out after 20 seconds")
        return False
    except Exception as e:
        logger.error(f"❌ create_revolutionary_interface failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(test_direct_tui_creation())
        if success:
            print("✅ Direct TUI creation test PASSED")
            sys.exit(0)
        else:
            print("❌ Direct TUI creation test FAILED")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n✅ Test interrupted - responsive to interrupt")
        sys.exit(0)