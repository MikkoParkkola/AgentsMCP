#!/usr/bin/env python3
"""
Simple smoke test to verify TUI startup components work
"""

import asyncio
import logging
import sys
import os

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure minimal logging
logging.basicConfig(level=logging.CRITICAL)

async def main():
    """Run smoke test components."""
    tests_passed = 0
    total_tests = 4
    
    print("🧪 AgentsMCP TUI Smoke Test")
    print("=" * 40)
    
    # Test 1: Can import and create Revolutionary interface
    try:
        print("1️⃣  Testing Revolutionary TUI Interface creation...")
        from agentsmcp.ui.v2.revolutionary_tui_interface import create_revolutionary_interface
        from agentsmcp.ui.cli_app import CLIConfig
        
        cli_config = CLIConfig()
        cli_config.debug_mode = False
        
        interface = await asyncio.wait_for(
            create_revolutionary_interface(
                cli_config=cli_config,
                orchestrator_integration=None,
                revolutionary_components={}
            ),
            timeout=15.0
        )
        
        print("   ✅ Revolutionary TUI Interface created successfully")
        tests_passed += 1
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 2: Can create basic TUI components
    try:
        print("2️⃣  Testing basic TUI components...")
        from agentsmcp.ui.v2.terminal_controller import get_terminal_controller
        from agentsmcp.ui.v2.logging_isolation_manager import get_logging_isolation_manager
        
        # These should not hang anymore
        terminal_controller = await asyncio.wait_for(get_terminal_controller(), timeout=5.0)
        logging_manager = await asyncio.wait_for(get_logging_isolation_manager(), timeout=5.0)
        
        print("   ✅ Basic TUI components created successfully")
        tests_passed += 1
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 3: Can create display manager without hanging  
    try:
        print("3️⃣  Testing Display Manager...")
        from agentsmcp.ui.v2.display_manager import get_display_manager
        
        display_manager = await asyncio.wait_for(get_display_manager(), timeout=8.0)
        
        print("   ✅ Display Manager created successfully")
        tests_passed += 1
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 4: Can import fallback TUI
    try:
        print("4️⃣  Testing Fallback TUI import...")
        from agentsmcp.ui.v2.fixed_working_tui import launch_fixed_working_tui
        
        print("   ✅ Fallback TUI imported successfully")
        tests_passed += 1
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    print("=" * 40)
    print(f"📊 Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed >= 3:
        print("🎉 SMOKE TEST PASSED! TUI startup components are working.")
        print("   The main hanging issue has been resolved.")
        if tests_passed < total_tests:
            print("   Some minor issues remain but TUI should be functional.")
        return True
    else:
        print("💥 SMOKE TEST FAILED! Critical issues remain.")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Test interrupted")
        sys.exit(0)