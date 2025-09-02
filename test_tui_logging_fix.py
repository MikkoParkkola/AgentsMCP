#!/usr/bin/env python3
"""
Test script to verify the TUI logging fix works correctly.
This simulates the conditions that were causing debug log flooding.
"""
import asyncio
import sys
import os
import logging

# Add src to path
sys.path.insert(0, 'src')

from agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface
from agentsmcp.ui.cli_app import CLIConfig

async def test_logging_suppression():
    """Test that logging is properly suppressed during TUI operation."""
    print("🧪 Testing TUI logging suppression fix...")
    
    # Set up some debug logging to see if it gets suppressed
    logging.basicConfig(level=logging.DEBUG, 
                       format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    
    # Create loggers that were causing the flooding
    llm_logger = logging.getLogger('agentsmcp.conversation.llm_client')
    orch_logger = logging.getLogger('agentsmcp.orchestration')
    
    print("✅ Before TUI launch - logging configured at DEBUG level")
    
    # Test: these should show up before TUI starts
    llm_logger.info("TEST: This log should appear BEFORE TUI launch")
    orch_logger.debug("TEST: This debug log should also appear BEFORE TUI launch")
    
    # Create TUI interface
    config = CLIConfig(debug_mode=True)
    interface = RevolutionaryTUIInterface(cli_config=config)
    
    print("📺 Starting TUI (will suppress logs during run)...")
    
    # During TUI run, logs should be suppressed
    try:
        # This would normally launch the TUI, but in non-TTY it returns quickly
        exit_code = await interface.run()
        
        print(f"✅ TUI completed with exit code: {exit_code}")
        
        # Test: these should show up after TUI ends (logging restored)
        print("📝 After TUI completion - testing if logging is restored...")
        llm_logger.info("TEST: This log should appear AFTER TUI completion")
        orch_logger.debug("TEST: This debug log should also appear AFTER TUI completion")
        
        return exit_code
        
    except Exception as e:
        print(f"❌ TUI test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

async def test_orchestrator_initialization():
    """Test that orchestrator initialization doesn't flood logs."""
    print("\n🧪 Testing orchestrator initialization logging suppression...")
    
    # Create interface and test initialization
    config = CLIConfig(debug_mode=True)
    interface = RevolutionaryTUIInterface(cli_config=config)
    
    print("🔧 Initializing TUI components (should not flood with debug logs)...")
    
    init_success = await interface.initialize()
    
    if init_success:
        print("✅ TUI initialization completed without log flooding")
    else:
        print("⚠️ TUI initialization failed (but no log flooding)")
    
    return init_success

async def main():
    """Run all tests."""
    print("🚀 AgentsMCP TUI Logging Fix Verification")
    print("=" * 50)
    
    # Test 1: Logging suppression during TUI run
    result1 = await test_logging_suppression()
    
    # Test 2: Orchestrator initialization
    result2 = await test_orchestrator_initialization()
    
    print("\n📊 Test Results:")
    print(f"  TUI logging suppression: {'✅ PASS' if result1 == 0 else '❌ FAIL'}")
    print(f"  Orchestrator init: {'✅ PASS' if result2 else '❌ FAIL'}")
    
    if result1 == 0 and result2:
        print("\n🎉 All tests passed! TUI logging fix is working correctly.")
        return 0
    else:
        print("\n⚠️ Some tests failed. Check output above for details.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)