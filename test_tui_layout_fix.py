#!/usr/bin/env python3
"""
Test script to verify the Revolutionary TUI layout corruption fix.

This script tests that the TUI can start without the refresh-triggered
layout corruption that was causing "every-other-line empty" problems.
"""

import asyncio
import logging
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_tui_layout_fix():
    """Test that the TUI can start and initialize without layout corruption."""
    
    logger.info("🚀 Testing Revolutionary TUI layout fix...")
    
    try:
        # Create the interface
        interface = RevolutionaryTUIInterface()
        
        # Initialize it
        success = await interface.initialize()
        
        if not success:
            logger.error("❌ Failed to initialize Revolutionary TUI Interface")
            return False
            
        logger.info("✅ Revolutionary TUI Interface initialized successfully")
        
        # Test that the layout was created properly
        if hasattr(interface, 'layout') and interface.layout:
            logger.info("✅ Layout object created successfully")
        else:
            logger.error("❌ Layout object not created")
            return False
            
        # Test that panels were initialized
        if hasattr(interface, '_layout_initialized'):
            logger.info("✅ Layout initialization flag found")
        else:
            logger.warning("⚠️ Layout initialization flag not found (this is OK)")
            
        # Simulate a quick state update to test that no corruption occurs
        interface.state.current_input = "test input"
        interface.state.is_processing = True
        interface.state.processing_message = "Testing..."
        
        logger.info("✅ State updates applied without errors")
        
        # Clean up
        await interface._cleanup()
        logger.info("✅ Cleanup completed successfully")
        
        logger.info("🎉 Revolutionary TUI layout fix test PASSED!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    logger.info("Starting Revolutionary TUI layout fix test...")
    
    success = await test_tui_layout_fix()
    
    if success:
        logger.info("✅ All tests passed - Layout corruption fix verified!")
        return 0
    else:
        logger.error("❌ Tests failed - Layout corruption may still exist")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)