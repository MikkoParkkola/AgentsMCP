#!/usr/bin/env python3
"""
Simple diagnostic test for TUI startup hang issue
"""

import asyncio
import logging
import sys
import os

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging to see what's happening
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tui_startup_debug.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

async def test_orchestrator_import():
    """Test if orchestrator import is causing the hang."""
    try:
        logger.info("Testing orchestrator import...")
        from agentsmcp.orchestration import Orchestrator, OrchestratorConfig, OrchestratorMode
        logger.info("✅ Orchestrator import successful")
        return True
    except Exception as e:
        logger.error(f"❌ Orchestrator import failed: {e}")
        return False

async def test_orchestrator_creation():
    """Test if orchestrator creation is causing the hang."""
    try:
        logger.info("Testing orchestrator creation...")
        from agentsmcp.orchestration import Orchestrator, OrchestratorConfig, OrchestratorMode
        
        config = OrchestratorConfig(
            mode=OrchestratorMode.STRICT_ISOLATION,
            enable_smart_classification=True,
            fallback_to_simple_response=True,
            max_agent_wait_time_ms=120000,
            synthesis_timeout_ms=5000
        )
        
        logger.info("Creating orchestrator...")
        orchestrator = Orchestrator(config=config)
        logger.info("✅ Orchestrator creation successful")
        return True
    except Exception as e:
        logger.error(f"❌ Orchestrator creation failed: {e}")
        return False

async def test_revolutionary_tui_import():
    """Test if Revolutionary TUI import is causing the hang."""
    try:
        logger.info("Testing Revolutionary TUI import...")
        from agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface
        logger.info("✅ Revolutionary TUI import successful")
        return True
    except Exception as e:
        logger.error(f"❌ Revolutionary TUI import failed: {e}")
        return False

async def test_revolutionary_tui_creation():
    """Test if Revolutionary TUI creation is causing the hang."""
    try:
        logger.info("Testing Revolutionary TUI creation...")
        from agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface
        
        interface = RevolutionaryTUIInterface()
        logger.info("✅ Revolutionary TUI creation successful")
        return True
    except Exception as e:
        logger.error(f"❌ Revolutionary TUI creation failed: {e}")
        return False

async def test_revolutionary_tui_initialize():
    """Test if Revolutionary TUI initialize is causing the hang."""
    try:
        logger.info("Testing Revolutionary TUI initialize...")
        from agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface
        
        interface = RevolutionaryTUIInterface()
        
        # Test initialize with timeout
        logger.info("Calling initialize()...")
        result = await asyncio.wait_for(interface.initialize(), timeout=10.0)
        logger.info(f"✅ Revolutionary TUI initialize successful: {result}")
        return True
    except asyncio.TimeoutError:
        logger.error("❌ Revolutionary TUI initialize timed out after 10 seconds")
        return False
    except Exception as e:
        logger.error(f"❌ Revolutionary TUI initialize failed: {e}")
        return False

async def test_launcher():
    """Test if the launcher is causing the hang."""
    try:
        logger.info("Testing Revolutionary TUI Launcher...")
        from agentsmcp.ui.v2.revolutionary_launcher import RevolutionaryLauncher
        
        launcher = RevolutionaryLauncher()
        logger.info("✅ Revolutionary TUI Launcher creation successful")
        return True
    except Exception as e:
        logger.error(f"❌ Revolutionary TUI Launcher creation failed: {e}")
        return False

async def test_step_by_step():
    """Run step-by-step diagnostics to find the hang location."""
    logger.info("🚀 Starting TUI startup diagnostics...")
    
    tests = [
        ("Orchestrator Import", test_orchestrator_import),
        ("Orchestrator Creation", test_orchestrator_creation),
        ("Revolutionary TUI Import", test_revolutionary_tui_import),
        ("Revolutionary TUI Creation", test_revolutionary_tui_creation),
        ("Revolutionary TUI Initialize", test_revolutionary_tui_initialize),
        ("Revolutionary TUI Launcher", test_launcher),
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = await asyncio.wait_for(test_func(), timeout=15.0)
            if result:
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
                break
        except asyncio.TimeoutError:
            logger.error(f"⏰ {test_name} TIMED OUT after 15 seconds - THIS IS THE HANG LOCATION!")
            break
        except Exception as e:
            logger.error(f"💥 {test_name} CRASHED: {e}")
            break

if __name__ == "__main__":
    try:
        asyncio.run(test_step_by_step())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test framework error: {e}")