#!/usr/bin/env python3
"""
Test each manager individually to find which one is hanging
"""

import asyncio
import logging
import sys
import os

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

async def test_terminal_controller():
    """Test terminal controller initialization."""
    try:
        logger.info("Testing terminal controller...")
        from agentsmcp.ui.v2.terminal_controller import get_terminal_controller
        
        controller = await asyncio.wait_for(get_terminal_controller(), timeout=10.0)
        result = await asyncio.wait_for(controller.initialize(), timeout=10.0)
        logger.info(f"✅ Terminal controller: {result}")
        return result
    except asyncio.TimeoutError:
        logger.error("❌ Terminal controller timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Terminal controller failed: {e}")
        return False

async def test_logging_manager():
    """Test logging isolation manager initialization."""
    try:
        logger.info("Testing logging isolation manager...")
        from agentsmcp.ui.v2.logging_isolation_manager import get_logging_isolation_manager
        
        manager = await asyncio.wait_for(get_logging_isolation_manager(), timeout=10.0)
        logger.info(f"✅ Logging manager: {type(manager)}")
        return True
    except asyncio.TimeoutError:
        logger.error("❌ Logging manager timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Logging manager failed: {e}")
        return False

async def test_display_manager():
    """Test display manager initialization."""
    try:
        logger.info("Testing display manager...")
        from agentsmcp.ui.v2.display_manager import get_display_manager
        
        manager = await asyncio.wait_for(get_display_manager(), timeout=10.0)
        logger.info(f"✅ Display manager: {type(manager)}")
        return True
    except asyncio.TimeoutError:
        logger.error("❌ Display manager timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Display manager failed: {e}")
        return False

async def test_managers():
    """Test each manager individually."""
    print("🔍 Testing each manager individually...")
    
    tests = [
        ("Terminal Controller", test_terminal_controller),
        ("Logging Manager", test_logging_manager),
        ("Display Manager", test_display_manager),
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*40}")
        print(f"Testing: {test_name}")
        print(f"{'='*40}")
        
        try:
            result = await asyncio.wait_for(test_func(), timeout=15.0)
            if result:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
                break
        except asyncio.TimeoutError:
            print(f"⏰ {test_name} TIMED OUT - THIS IS THE HANG LOCATION!")
            break
        except Exception as e:
            print(f"💥 {test_name} CRASHED: {e}")
            break

if __name__ == "__main__":
    try:
        asyncio.run(test_managers())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test framework error: {e}")