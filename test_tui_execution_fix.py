#!/usr/bin/env python3
"""
Test script to validate the Revolutionary TUI execution bypass fix.

This script tests that:
1. Revolutionary TUI Interface can be created successfully
2. ReliableTUIInterface properly calls the Revolutionary TUI's run() method
3. The integration layer no longer bypasses the actual TUI execution
"""

import asyncio
import sys
import os
import logging
from unittest.mock import AsyncMock, MagicMock, patch

# Setup path
sys.path.insert(0, os.path.abspath('.'))

# Set up logging to see what happens
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from src.agentsmcp.ui.v2.reliability.integration_layer import ReliableTUIInterface
from src.agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface


class MockCLIConfig:
    """Mock CLI configuration."""
    def __init__(self):
        self.debug_mode = True
        

class MockOrchestrator:
    """Mock orchestrator."""
    def __init__(self):
        pass


class MockRevolutionaryTUI(RevolutionaryTUIInterface):
    """Mock Revolutionary TUI that tracks if run() was called."""
    
    def __init__(self, *args, **kwargs):
        # Don't call super().__init__ to avoid initialization issues in test
        self.run_called = False
        self.run_call_count = 0
        
    async def run(self) -> int:
        """Mock run method that tracks calls."""
        self.run_called = True
        self.run_call_count += 1
        print(f"🎯 Revolutionary TUI run() method called! (call #{self.run_call_count})")
        return 0
        

async def test_tui_execution_fix():
    """Test that the TUI execution bypass is fixed."""
    
    print("🚀 Testing Revolutionary TUI Execution Fix")
    print("=" * 50)
    
    # Create mock components
    cli_config = MockCLIConfig()
    orchestrator = MockOrchestrator()
    agent_state = {}
    revolutionary_components = {}
    
    print("1. Testing direct Revolutionary TUI creation...")
    try:
        original_tui = RevolutionaryTUIInterface(
            cli_config=cli_config,
            orchestrator_integration=orchestrator,
            revolutionary_components=revolutionary_components
        )
        print(f"   ✅ RevolutionaryTUIInterface created")
        print(f"   ✅ Has run() method: {hasattr(original_tui, 'run')}")
        print(f"   ✅ run() is async: {asyncio.iscoroutinefunction(original_tui.run)}")
    except Exception as e:
        print(f"   ❌ Failed to create RevolutionaryTUIInterface: {e}")
        return False
        
    print("\n2. Testing ReliableTUIInterface with mock Revolutionary TUI...")
    
    # Patch RevolutionaryTUIInterface to use our mock
    with patch('src.agentsmcp.ui.v2.reliability.integration_layer.RevolutionaryTUIInterface', MockRevolutionaryTUI):
        try:
            # Create reliable TUI
            reliable_tui = ReliableTUIInterface(
                agent_orchestrator=orchestrator,
                agent_state=agent_state,
                cli_config=cli_config,
                revolutionary_components=revolutionary_components
            )
            print(f"   ✅ ReliableTUIInterface created")
            
            # Test startup - this should create the TUI instance
            startup_result = await reliable_tui.start()
            print(f"   ✅ Startup completed: {startup_result}")
            print(f"   ✅ Original TUI instance created: {reliable_tui._original_tui is not None}")
            
            # CRITICAL TEST: Call the run() method
            print("\n3. Testing run() method execution...")
            print("   🎯 About to call ReliableTUIInterface.run()...")
            
            # This should call the Revolutionary TUI's run() method
            result = await reliable_tui.run()
            
            print(f"   ✅ ReliableTUIInterface.run() returned: {result}")
            
            # Check if the mock's run() method was actually called
            mock_tui = reliable_tui._original_tui
            if mock_tui.run_called:
                print(f"   🎉 SUCCESS! Revolutionary TUI run() method WAS CALLED!")
                print(f"   🎉 Call count: {mock_tui.run_call_count}")
                return True
            else:
                print(f"   ❌ FAILURE! Revolutionary TUI run() method was NOT called")
                print(f"   ❌ This means the execution bypass bug is still present")
                return False
                
        except Exception as e:
            print(f"   ❌ ReliableTUIInterface test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            

async def test_fallback_mode():
    """Test that fallback mode also calls run() properly."""
    
    print("\n4. Testing fallback mode execution...")
    
    # Mock components
    cli_config = MockCLIConfig()
    orchestrator = MockOrchestrator()
    agent_state = {}
    
    # Create config that forces fallback mode
    from src.agentsmcp.ui.v2.reliability.integration_layer import ReliabilityConfig
    config = ReliabilityConfig(fallback_on_reliability_failure=True)
    
    with patch('src.agentsmcp.ui.v2.reliability.integration_layer.RevolutionaryTUIInterface', MockRevolutionaryTUI):
        try:
            reliable_tui = ReliableTUIInterface(
                agent_orchestrator=orchestrator,
                agent_state=agent_state,
                reliability_config=config,
                cli_config=cli_config,
                revolutionary_components={}
            )
            
            # Force fallback mode by setting flag
            reliable_tui._fallback_mode = True
            reliable_tui._startup_completed = True
            
            # Create mock TUI for fallback
            reliable_tui._original_tui = MockRevolutionaryTUI(
                cli_config=cli_config,
                orchestrator_integration=orchestrator,
                revolutionary_components={}
            )
            
            print("   🎯 Testing fallback mode run()...")
            result = await reliable_tui.run()
            
            print(f"   ✅ Fallback run() returned: {result}")
            
            # Check if run() was called in fallback mode
            mock_tui = reliable_tui._original_tui
            if mock_tui.run_called:
                print(f"   🎉 SUCCESS! Fallback mode also calls Revolutionary TUI run()!")
                return True
            else:
                print(f"   ❌ FAILURE! Fallback mode did not call run()")
                return False
                
        except Exception as e:
            print(f"   ❌ Fallback mode test failed: {e}")
            import traceback
            traceback.print_exc()
            return False


async def main():
    """Run all tests."""
    print("Testing Revolutionary TUI Execution Bypass Fix")
    print("=" * 60)
    
    success1 = await test_tui_execution_fix()
    success2 = await test_fallback_mode()
    
    print("\n" + "=" * 60)
    
    if success1 and success2:
        print("🎉 ALL TESTS PASSED!")
        print("🎉 Revolutionary TUI execution bypass has been FIXED!")
        print("🎉 The TUI's run() method will now be called properly!")
        return 0
    else:
        print("❌ Some tests failed!")
        print("❌ The execution bypass fix may need more work!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)