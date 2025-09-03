#!/usr/bin/env python3
"""
Test script to verify the TUI Guardian shutdown fix.

This script simulates the exact conditions that were causing the 
immediate Guardian shutdown with 0.08s duration.
"""
import asyncio
import logging
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Configure logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from agentsmcp.ui.v2.reliability.integration_layer import ReliableTUIInterface, create_reliable_tui

class MockAgentOrchestrator:
    """Mock orchestrator for testing."""
    def __init__(self):
        self.agents = []

class MockCLIConfig:
    """Mock CLI config for testing."""
    def __init__(self):
        self.debug_mode = True
        self.verbose = True

async def test_tui_fix():
    """Test that the TUI fix resolves the Guardian shutdown issue."""
    print("🧪 Testing TUI Guardian shutdown fix...")
    print("=" * 60)
    
    # Create mock components
    orchestrator = MockAgentOrchestrator()
    cli_config = MockCLIConfig()
    
    print("📋 Test conditions:")
    print(f"   - TTY status: {sys.stdin.isatty()}")
    print(f"   - Environment: {'TTY' if sys.stdin.isatty() else 'Non-TTY (should trigger demo mode)'}")
    print(f"   - Expected: Demo mode messages should appear, not immediate shutdown")
    print()
    
    try:
        print("🚀 Creating ReliableTUIInterface...")
        tui = await create_reliable_tui(
            agent_orchestrator=orchestrator,
            agent_state=None,
            cli_config=cli_config,
            revolutionary_components={}
        )
        print("✅ ReliableTUIInterface created successfully")
        
        print("\n⏱️  Calling tui.run() - this should NOT cause immediate Guardian shutdown...")
        print("   Expected: Demo mode messages should appear within 1 second")
        print("   Previous bug: Immediate 'Guardian shutdown' after 0.08s")
        print()
        
        # Call run() and measure execution time
        import time
        start_time = time.time()
        
        result = await tui.run()
        
        execution_time = time.time() - start_time
        
        print(f"\n📊 Test Results:")
        print(f"   - Execution time: {execution_time:.2f}s")
        print(f"   - Return code: {result}")
        print(f"   - Expected time: >3s (demo mode runs for 3+ seconds)")
        print(f"   - Status: {'✅ PASS' if execution_time > 2.0 else '❌ FAIL'}")
        
        if execution_time < 1.0:
            print(f"\n❌ FAILURE: TUI exited too quickly ({execution_time:.2f}s)")
            print("   This suggests the Guardian shutdown bug is still present")
            return False
        else:
            print(f"\n✅ SUCCESS: TUI ran for expected duration ({execution_time:.2f}s)")
            print("   Demo mode messages should have appeared above")
            return True
            
    except Exception as e:
        print(f"\n❌ ERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("TUI Guardian Shutdown Fix - Test Script")
    print("=====================================")
    
    success = asyncio.run(test_tui_fix())
    
    print("\n" + "=" * 60)
    if success:
        print("✅ TEST PASSED: TUI Guardian shutdown fix is working")
    else:
        print("❌ TEST FAILED: TUI Guardian shutdown bug is still present")
    
    sys.exit(0 if success else 1)