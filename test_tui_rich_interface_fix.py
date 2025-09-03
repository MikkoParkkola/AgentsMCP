#!/usr/bin/env python3
"""
Test script to validate Revolutionary TUI Rich interface activation.

This script tests the enhanced TTY detection fix to ensure Claude Code
and similar environments properly activate the Rich interface instead
of falling back to basic mode.
"""

import asyncio
import sys
import os
import logging
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface

# Configure logging to capture debug output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class MockCLIConfig:
    """Mock CLI configuration for testing."""
    def __init__(self):
        self.debug_mode = True
        self.agent_name = "test_agent"
        self.mcp_servers = {}
        

class MockAgentOrchestrator:
    """Mock agent orchestrator for testing."""
    def __init__(self):
        self.agents = []
        
    async def handle_user_input(self, user_input: str):
        """Mock handler for user input."""
        return f"Mock response to: {user_input}"


async def test_enhanced_tty_detection():
    """Test the enhanced TTY detection logic directly."""
    logger.info("=" * 60)
    logger.info("Testing Enhanced TTY Detection")
    logger.info("=" * 60)
    
    # Create TUI instance
    cli_config = MockCLIConfig()
    orchestrator = MockAgentOrchestrator()
    
    tui = RevolutionaryTUIInterface(
        cli_config=cli_config,
        orchestrator_integration=orchestrator,
        revolutionary_components={}
    )
    
    # Test TTY detection
    logger.info("Testing TTY detection in current environment:")
    logger.info(f"  - sys.stdin.isatty(): {sys.stdin.isatty() if hasattr(sys.stdin, 'isatty') else 'N/A'}")
    logger.info(f"  - sys.stdout.isatty(): {sys.stdout.isatty() if hasattr(sys.stdout, 'isatty') else 'N/A'}")
    logger.info(f"  - sys.stderr.isatty(): {sys.stderr.isatty() if hasattr(sys.stderr, 'isatty') else 'N/A'}")
    logger.info(f"  - TERM: {os.environ.get('TERM', 'NOT_SET')}")
    logger.info(f"  - COLORTERM: {os.environ.get('COLORTERM', 'NOT_SET')}")
    logger.info(f"  - Terminal size available: {hasattr(os, 'get_terminal_size')}")
    
    if hasattr(os, 'get_terminal_size'):
        try:
            size = os.get_terminal_size()
            logger.info(f"  - Terminal size: {size.columns}x{size.lines}")
        except (OSError, AttributeError) as e:
            logger.info(f"  - Terminal size error: {e}")
    
    logger.info("\nRunning enhanced TTY detection...")
    
    # Call the enhanced detection method
    result = tui._detect_terminal_capabilities()
    
    logger.info(f"\nEnhanced TTY Detection Result: {result}")
    
    if result:
        logger.info("✅ SUCCESS: Rich interface should be activated!")
    else:
        logger.info("❌ FAILURE: Will fall back to basic mode!")
        
    return result


async def test_tui_startup_simulation():
    """Test TUI startup process to see if Rich interface activates."""
    logger.info("=" * 60)
    logger.info("Testing TUI Startup Simulation")
    logger.info("=" * 60)
    
    # Create TUI instance
    cli_config = MockCLIConfig()
    orchestrator = MockAgentOrchestrator()
    
    tui = RevolutionaryTUIInterface(
        cli_config=cli_config,
        orchestrator_integration=orchestrator,
        revolutionary_components={}
    )
    
    # Initialize the TUI (this calls the enhanced TTY detection)
    logger.info("Initializing TUI components...")
    await tui.initialize()
    
    # Check if Rich is available and TTY detection passed
    from agentsmcp.ui.v2.revolutionary_tui_interface import RICH_AVAILABLE
    
    tty_result = tui._detect_terminal_capabilities()
    
    logger.info(f"TUI Initialization Results:")
    logger.info(f"  - RICH_AVAILABLE: {RICH_AVAILABLE}")
    logger.info(f"  - Enhanced TTY Detection: {tty_result}")
    logger.info(f"  - Should use Rich interface: {RICH_AVAILABLE and tty_result}")
    
    if RICH_AVAILABLE and tty_result:
        logger.info("✅ SUCCESS: TUI will use Rich Live display!")
        return True
    else:
        logger.info("❌ FAILURE: TUI will fall back to basic mode!")
        return False


async def test_environment_variations():
    """Test TTY detection under various environment configurations."""
    logger.info("=" * 60)  
    logger.info("Testing Environment Variations")
    logger.info("=" * 60)
    
    # Save original environment
    original_env = {
        'TERM': os.environ.get('TERM'),
        'COLORTERM': os.environ.get('COLORTERM'),
        'NO_COLOR': os.environ.get('NO_COLOR')
    }
    
    test_configs = [
        {
            'name': 'Default Environment',
            'env': {},
        },
        {
            'name': 'xterm-256color',
            'env': {'TERM': 'xterm-256color'}
        },
        {
            'name': 'With COLORTERM',
            'env': {'COLORTERM': 'truecolor'}
        },
        {
            'name': 'Minimal Terminal',
            'env': {'TERM': 'dumb'}
        },
        {
            'name': 'Color Disabled',
            'env': {'NO_COLOR': '1'}
        }
    ]
    
    results = []
    
    for config in test_configs:
        logger.info(f"\n--- Testing: {config['name']} ---")
        
        # Restore original environment
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        
        # Apply test environment
        for key, value in config['env'].items():
            os.environ[key] = value
            
        # Create fresh TUI instance
        cli_config = MockCLIConfig()
        orchestrator = MockAgentOrchestrator()
        
        tui = RevolutionaryTUIInterface(
            cli_config=cli_config,
            orchestrator_integration=orchestrator,
            revolutionary_components={}
        )
        
        # Test detection
        result = tui._detect_terminal_capabilities()
        results.append((config['name'], result))
        
        logger.info(f"Result for {config['name']}: {'✅ Rich' if result else '❌ Basic'}")
    
    # Restore original environment
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    
    logger.info(f"\n--- Environment Test Summary ---")
    for name, result in results:
        status = "✅ Rich" if result else "❌ Basic"
        logger.info(f"  {name}: {status}")
    
    return results


async def main():
    """Run all TUI Rich interface tests."""
    logger.info("🚀 Revolutionary TUI Rich Interface Fix Validation")
    logger.info("This test validates the enhanced TTY detection fix for Claude Code environments")
    logger.info("")
    
    try:
        # Test 1: Enhanced TTY Detection
        tty_result = await test_enhanced_tty_detection()
        
        # Test 2: TUI Startup Simulation 
        startup_result = await test_tui_startup_simulation()
        
        # Test 3: Environment Variations
        env_results = await test_environment_variations()
        
        # Summary
        logger.info("=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        
        logger.info(f"Enhanced TTY Detection: {'✅ PASS' if tty_result else '❌ FAIL'}")
        logger.info(f"TUI Startup Simulation: {'✅ PASS' if startup_result else '❌ FAIL'}")
        
        env_success_count = sum(1 for _, result in env_results if result)
        logger.info(f"Environment Tests: {env_success_count}/{len(env_results)} passed")
        
        overall_success = tty_result and startup_result and env_success_count > 0
        
        if overall_success:
            logger.info("\n🎉 OVERALL RESULT: ✅ SUCCESS")
            logger.info("The Revolutionary TUI should now properly activate Rich interface in Claude Code!")
            logger.info("You can test with: ./agentsmcp tui")
        else:
            logger.info("\n⚠️  OVERALL RESULT: ❌ NEEDS MORE WORK")
            logger.info("The fix may need additional adjustments for your environment.")
            
        return 0 if overall_success else 1
        
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)