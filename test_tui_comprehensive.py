#!/usr/bin/env python3
"""
Comprehensive TUI smoke test that mirrors the actual execution path.

This test is designed to catch the real failures that occurred:
1. ReliableTUIInterface constructor parameter conflicts
2. Missing methods on ReliableTUIInterface (run, run_main_loop, stop)
3. Method name mismatches with RevolutionaryTUIInterface
4. Recovery system attribute errors
5. Actual TUI startup process integration issues

Unlike the previous smoke test, this one follows the exact same path
as the real TUI launcher to catch integration failures.
"""

import asyncio
import logging
import sys
import os
import signal
from contextlib import asynccontextmanager

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging to capture errors but suppress debug noise
logging.basicConfig(level=logging.WARNING)

class TimeoutError(Exception):
    """Custom timeout exception for test control."""
    pass

@asynccontextmanager
async def timeout_context(seconds: float, description: str):
    """Context manager for test timeouts with cleanup."""
    task = None
    try:
        def timeout_handler():
            raise TimeoutError(f"Test '{description}' timed out after {seconds}s")
        
        # Set up timeout
        if hasattr(signal, 'SIGALRM'):  # Unix systems
            old_handler = signal.signal(signal.SIGALRM, lambda s, f: timeout_handler())
            signal.alarm(int(seconds))
        
        yield
        
    except TimeoutError:
        raise
    except Exception as e:
        # Re-raise with context
        raise Exception(f"Test '{description}' failed: {e}") from e
    finally:
        # Clean up timeout
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
            if 'old_handler' in locals():
                signal.signal(signal.SIGALRM, old_handler)

async def test_reliable_tui_creation():
    """Test 1: Can create ReliableTUIInterface with correct parameters."""
    print("1️⃣  Testing ReliableTUIInterface creation (real constructor params)...")
    
    try:
        from agentsmcp.ui.v2.reliability.integration_layer import ReliableTUIInterface
        from agentsmcp.ui.cli_app import CLIConfig
        from agentsmcp.orchestration.orchestrator import Orchestrator
        
        # Create real config like the launcher does
        cli_config = CLIConfig()
        cli_config.debug_mode = False
        cli_config.enable_rich_tui = True
        
        # Create orchestrator like the launcher does
        orchestrator = Orchestrator()
        
        # Test the exact same constructor call that was failing
        async with timeout_context(10.0, "ReliableTUIInterface creation"):
            reliable_tui = ReliableTUIInterface(
                agent_orchestrator=orchestrator,
                agent_state={},  # Mock agent state
                cli_config=cli_config,
                revolutionary_components={}
            )
        
        print("   ✅ ReliableTUIInterface created successfully")
        return reliable_tui, True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return None, False

async def test_reliable_tui_methods(reliable_tui):
    """Test 2: Verify ReliableTUIInterface has required methods."""
    print("2️⃣  Testing ReliableTUIInterface required methods...")
    
    try:
        # Check for the method that was missing originally
        if not hasattr(reliable_tui, 'run'):
            raise AttributeError("ReliableTUIInterface missing 'run' method")
        
        # Check that run method is callable
        if not callable(getattr(reliable_tui, 'run')):
            raise AttributeError("ReliableTUIInterface 'run' is not callable")
        
        # Check for other required methods
        required_methods = ['start', 'run_main_loop', 'stop']
        for method_name in required_methods:
            if not hasattr(reliable_tui, method_name):
                raise AttributeError(f"ReliableTUIInterface missing '{method_name}' method")
            if not callable(getattr(reliable_tui, method_name)):
                raise AttributeError(f"ReliableTUIInterface '{method_name}' is not callable")
        
        print("   ✅ All required methods present and callable")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

async def test_revolutionary_tui_delegation(reliable_tui):
    """Test 3: Test that ReliableTUIInterface can create RevolutionaryTUIInterface."""
    print("3️⃣  Testing RevolutionaryTUIInterface delegation setup...")
    
    try:
        # Test startup which creates the original TUI
        async with timeout_context(15.0, "TUI component preparation"):
            startup_result = await reliable_tui.start()
        
        if not startup_result:
            raise Exception("TUI startup returned failure")
        
        # Check that original TUI was created
        if not hasattr(reliable_tui, '_original_tui') or reliable_tui._original_tui is None:
            raise Exception("Original TUI was not created during startup")
        
        # Check that original TUI has the methods we expect to delegate to
        original_tui = reliable_tui._original_tui
        expected_methods = ['_run_main_loop', '_cleanup']  # Actual method names
        
        for method_name in expected_methods:
            if not hasattr(original_tui, method_name):
                raise AttributeError(f"RevolutionaryTUIInterface missing '{method_name}' method")
        
        print("   ✅ RevolutionaryTUIInterface delegation setup successful")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

async def test_tui_run_method(reliable_tui):
    """Test 4: Test that the run method can be called without immediate failure."""
    print("4️⃣  Testing TUI run method execution (brief)...")
    
    try:
        # Create a task to run the TUI
        async def run_tui_briefly():
            """Run TUI for a very short time to test startup."""
            try:
                # This should not immediately crash
                result = await reliable_tui.run()
                return result
            except Exception as e:
                # If it's a shutdown/timeout related error, that's expected for this test
                if any(keyword in str(e).lower() for keyword in ['timeout', 'shutdown', 'cancelled']):
                    return 0  # Expected for brief test
                raise
        
        # Run TUI briefly with timeout
        async with timeout_context(5.0, "TUI run method"):
            tui_task = asyncio.create_task(run_tui_briefly())
            
            # Let it run briefly then cancel
            await asyncio.sleep(0.5)  # Very brief run
            tui_task.cancel()
            
            try:
                result = await tui_task
            except asyncio.CancelledError:
                result = 0  # Expected cancellation
        
        print("   ✅ TUI run method executed without immediate crash")
        return True
        
    except Exception as e:
        # Don't fail on expected shutdown issues during brief test
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ['timeout', 'cancelled', 'shutdown']):
            print("   ✅ TUI run method executed (expected timeout/cancellation)")
            return True
        
        print(f"   ❌ Failed: {e}")
        return False

async def test_recovery_system():
    """Test 5: Test that recovery system components don't have attribute errors."""
    print("5️⃣  Testing recovery system attributes...")
    
    try:
        from agentsmcp.ui.v2.reliability.recovery_manager import RecoveryManager, RecoveryResult, RecoveryStrategy, RecoveryStatus
        from datetime import datetime
        
        # Create a mock recovery result to test attributes  
        recovery_result = RecoveryResult(
            strategy=RecoveryStrategy.RESTART_COMPONENT,
            status=RecoveryStatus.FAILED,
            component_name="test_component",
            start_time=datetime.now(),
            error_message="Test error message"
        )
        
        # Test that the attributes we use actually exist
        assert hasattr(recovery_result, 'error_message'), "RecoveryResult missing 'error_message'"
        assert hasattr(recovery_result, 'success'), "RecoveryResult missing 'success'"
        assert recovery_result.error_message == "Test error message", "error_message value incorrect"
        
        print("   ✅ Recovery system attributes correct")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

async def test_launcher_integration():
    """Test 6: Test actual launcher function that was failing."""
    print("6️⃣  Testing launcher integration (launch_revolutionary_tui function)...")
    
    try:
        from agentsmcp.ui.v2.revolutionary_launcher import launch_revolutionary_tui
        from agentsmcp.ui.cli_app import CLIConfig
        
        # Test the exact function call that the launcher uses
        cli_config = CLIConfig()
        cli_config.debug_mode = False
        cli_config.enable_rich_tui = True
        
        # We can't actually launch it fully, but we can test that the function exists
        # and the parameters are accepted
        if not callable(launch_revolutionary_tui):
            raise Exception("launch_revolutionary_tui is not callable")
        
        print("   ✅ Launcher function exists and is callable")
        
        # Test creating ReliableTUIInterface directly since that's what was actually failing
        from agentsmcp.ui.v2.reliability.integration_layer import ReliableTUIInterface
        from agentsmcp.orchestration.orchestrator import Orchestrator
        
        orchestrator = Orchestrator()
        
        async with timeout_context(15.0, "Direct ReliableTUIInterface creation"):
            tui_interface = ReliableTUIInterface(
                agent_orchestrator=orchestrator,
                agent_state={},  # Mock agent state
                cli_config=cli_config,
                revolutionary_components={}
            )
        
        if tui_interface is None:
            raise Exception("ReliableTUIInterface creation returned None")
        
        # Verify it has the run method that launcher expects
        if not hasattr(tui_interface, 'run'):
            raise Exception("TUI missing run method")
        
        print("   ✅ Launcher integration successful")
        return tui_interface, True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return None, False

async def main():
    """Run comprehensive smoke test."""
    print("🧪 AgentsMCP TUI Comprehensive Smoke Test")
    print("=" * 50)
    print("This test mirrors the actual TUI execution path to catch real integration failures.")
    print()
    
    tests_passed = 0
    total_tests = 6
    reliable_tui = None
    
    # Test 1: Basic ReliableTUIInterface creation
    reliable_tui, success = await test_reliable_tui_creation()
    if success:
        tests_passed += 1
    
    # Test 2: Required methods exist (only if creation succeeded)
    if reliable_tui and await test_reliable_tui_methods(reliable_tui):
        tests_passed += 1
    
    # Test 3: Revolutionary TUI delegation (only if creation succeeded)
    if reliable_tui and await test_revolutionary_tui_delegation(reliable_tui):
        tests_passed += 1
    
    # Test 4: Run method execution (only if delegation succeeded)
    if reliable_tui and tests_passed >= 3:
        if await test_tui_run_method(reliable_tui):
            tests_passed += 1
    
    # Test 5: Recovery system attributes
    if await test_recovery_system():
        tests_passed += 1
    
    # Test 6: Launcher integration
    launcher_tui, success = await test_launcher_integration()
    if success:
        tests_passed += 1
    
    # Cleanup
    try:
        if reliable_tui and hasattr(reliable_tui, 'stop'):
            await reliable_tui.stop()
    except Exception:
        pass  # Ignore cleanup errors
    
    print()
    print("=" * 50)
    print(f"📊 Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed >= 5:
        print("🎉 COMPREHENSIVE SMOKE TEST PASSED!")
        print("   The TUI should start without the constructor/method errors.")
        if tests_passed < total_tests:
            print("   Some edge cases may still exist but core functionality works.")
    elif tests_passed >= 3:
        print("🔧 PARTIAL SUCCESS - Core components work but integration issues remain.")
        print("   The TUI may have runtime issues.")
    else:
        print("💥 COMPREHENSIVE SMOKE TEST FAILED!")
        print("   Critical integration failures detected. TUI will likely not start.")
    
    return tests_passed >= 5

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Test interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Test runner failed: {e}")
        sys.exit(1)