#!/usr/bin/env python3
"""
Emergency test script to verify Rich Live alternate screen fixes.
This script tests that the TUI properly uses alternate screen buffer
and does NOT pollute terminal scrollback.
"""

import asyncio
import sys
import os
import time
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface

# Configure logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class MockCLIConfig:
    def __init__(self):
        self.debug_mode = True  # Enable debug mode for testing

async def test_emergency_tui_fix():
    """Test the emergency TUI fix for Rich Live alternate screen isolation."""
    
    logger.info("🚨 EMERGENCY TUI TEST: Starting Rich Live alternate screen isolation test")
    
    # Create a mock CLI config
    cli_config = MockCLIConfig()
    
    try:
        # Create the Revolutionary TUI Interface
        logger.info("Creating Revolutionary TUI Interface...")
        tui = RevolutionaryTUIInterface(cli_config=cli_config)
        
        # Set a short test duration
        test_duration = 10  # Run for 10 seconds max
        
        logger.info(f"Starting TUI test for {test_duration} seconds...")
        logger.info("🎯 CRITICAL TEST: Checking if TUI output stays in alternate screen")
        logger.info("⚠️  If you see TUI layout output in this terminal scrollback, the fix FAILED")
        logger.info("✅ If this terminal stays clean and TUI runs in alternate screen, the fix SUCCEEDED")
        
        # Start the TUI with a timeout
        async def run_with_timeout():
            return await tui.run()
        
        async def timeout_handler():
            await asyncio.sleep(test_duration)
            logger.info(f"🕐 Test timeout after {test_duration} seconds - stopping TUI")
            tui.running = False
            return 0
        
        # Run TUI with timeout
        result = await asyncio.wait_for(
            asyncio.gather(run_with_timeout(), timeout_handler(), return_exceptions=True),
            timeout=test_duration + 5  # Extra timeout buffer
        )
        
        logger.info(f"🏁 TUI test completed with result: {result}")
        
    except asyncio.TimeoutError:
        logger.info("⏰ Test timed out - this is expected behavior")
        return 0
    except KeyboardInterrupt:
        logger.info("⚡ Test interrupted by user - this is expected behavior")  
        return 0
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        return 1
    
    logger.info("✅ EMERGENCY TUI TEST COMPLETED")
    logger.info("🔍 Check above: Did you see any TUI layout content in this scrollback?")
    logger.info("   - If YES: Fix FAILED - Rich Live is polluting scrollback")  
    logger.info("   - If NO:  Fix SUCCEEDED - Rich Live is using alternate screen properly")
    
    return 0

if __name__ == "__main__":
    try:
        result = asyncio.run(test_emergency_tui_fix())
        sys.exit(result)
    except KeyboardInterrupt:
        print("\n🚨 Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test script failed: {e}")
        sys.exit(1)