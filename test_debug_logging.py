#!/usr/bin/env python3
"""
Test script to verify emergency debug logging works in Revolutionary TUI.
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agentsmcp.ui.v2.reliability.integration_layer import ReliableTUIInterface


async def test_debug_logging():
    """Test that our debug logging appears correctly."""
    print("=" * 80)
    print("TESTING DEBUG LOGGING IN REVOLUTIONARY TUI")
    print("=" * 80)
    
    # Create mock orchestrator and components
    class MockOrchestrator:
        pass
    
    class MockAgentState:
        pass
    
    class MockCLIConfig:
        debug_mode = True
    
    try:
        print("Creating ReliableTUIInterface...")
        
        # Create TUI with mock components
        tui = ReliableTUIInterface(
            agent_orchestrator=MockOrchestrator(),
            agent_state=MockAgentState(),
            cli_config=MockCLIConfig(),
            revolutionary_components={}
        )
        
        print("Calling TUI.run() to see debug logs...")
        print("-" * 40)
        
        # Call run() - this should show all our debug messages
        result = await tui.run()
        
        print("-" * 40)
        print(f"TUI.run() returned: {result}")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user - this is expected in TUI test")
        return 0
    except Exception as e:
        print(f"Test exception: {e}")
        import traceback
        print(f"Exception traceback:\n{traceback.format_exc()}")
        return 1
    
    print("Debug logging test completed!")
    return 0


if __name__ == "__main__":
    # Run the test
    exit_code = asyncio.run(test_debug_logging())
    sys.exit(exit_code)