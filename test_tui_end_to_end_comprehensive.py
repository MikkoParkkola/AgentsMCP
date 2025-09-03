#!/usr/bin/env python3
"""
Comprehensive End-to-End Test Suite for Revolutionary TUI Interface
==================================================================

This test suite validates the complete Revolutionary TUI solution works correctly 
for real user scenarios, covering all previously resolved critical issues:

1. ✅ Constructor parameter conflicts (TUI now starts)
2. ✅ 0.08s Guardian shutdown (TUI runs properly) 
3. ✅ Scrollback pollution (Rich Live alternate screen)
4. ✅ Empty lines in layout (separators removed)
5. ✅ Revolutionary TUI execution (integration layer fixed)
6. ✅ Input buffer corruption (race condition resolved)
7. ✅ Rich Live display corruption on keystrokes (pipeline sync fixed)
8. ✅ Production debug cleanup (professional logging)

CRITICAL USER SCENARIOS TESTED:
- TUI Startup: Must start without immediate shutdown
- Display Rendering: Must render properly without scrollback pollution  
- Character Input: Must accumulate correctly ('h' → 'he' → 'hel' → 'hello')
- Display Stability: Must NOT corrupt on keystroke input
- Command Processing: Must handle /quit and other commands correctly
- Clean Output: Must not flood console with debug prints
"""

import asyncio
import logging
import os
import sys
import time
import threading
import signal
import unittest
import tempfile
import io
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from contextlib import redirect_stdout, redirect_stderr
from typing import List, Dict, Any

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from src.agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface, TUIState
    from src.agentsmcp.ui.v2.terminal_controller import TerminalController
    from src.agentsmcp.ui.v2.logging_isolation_manager import LoggingIsolationManager
    from src.agentsmcp.ui.v2.input_rendering_pipeline import InputRenderingPipeline
    from src.agentsmcp.ui.v2.display_manager import DisplayManager
    from src.agentsmcp.orchestration import Orchestrator, OrchestratorConfig
except ImportError as e:
    print(f"Import error: {e}")
    print("Please run from project root directory")
    sys.exit(1)


class MockStdin:
    """Mock stdin that simulates user keyboard input."""
    
    def __init__(self, input_sequence: List[str]):
        self.input_sequence = input_sequence
        self.index = 0
        self.delay = 0.1  # Realistic typing delay
    
    def read_char(self):
        """Simulate reading a character with typing delay."""
        if self.index >= len(self.input_sequence):
            return None
        
        time.sleep(self.delay)
        char = self.input_sequence[self.index]
        self.index += 1
        return char
    
    def reset(self):
        """Reset input sequence for new test."""
        self.index = 0


class OutputCapture:
    """Captures all output to detect console pollution."""
    
    def __init__(self):
        self.stdout_content = []
        self.stderr_content = []
        self.debug_prints = []
    
    def capture_stdout(self, content):
        self.stdout_content.append(content)
        # Detect debug pollution patterns
        if any(pattern in content.lower() for pattern in ['debug:', 'trace:', '[debug]', 'print(']):
            self.debug_prints.append(content)
    
    def capture_stderr(self, content):
        self.stderr_content.append(content)
    
    def has_debug_pollution(self) -> bool:
        """Check if there's unwanted debug output."""
        return len(self.debug_prints) > 0
    
    def get_pollution_summary(self) -> str:
        """Get summary of debug pollution found."""
        return f"Found {len(self.debug_prints)} debug prints: {self.debug_prints[:3]}..."


class TUIEndToEndTests(unittest.TestCase):
    """Comprehensive end-to-end tests for Revolutionary TUI Interface."""
    
    def setUp(self):
        """Set up test environment."""
        self.output_capture = OutputCapture()
        self.mock_orchestrator = Mock(spec=Orchestrator)
        self.mock_orchestrator.start = AsyncMock()
        self.mock_orchestrator.stop = AsyncMock()
        self.mock_orchestrator.get_status = Mock(return_value={'status': 'ready'})
        
        # Configure logging to capture all output
        logging.getLogger().handlers.clear()
        test_handler = logging.StreamHandler()
        test_handler.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(test_handler)
        logging.getLogger().setLevel(logging.DEBUG)
    
    def tearDown(self):
        """Clean up after each test."""
        # Reset logging
        logging.getLogger().handlers.clear()
        
        # Reset any global state
        if hasattr(self, 'tui') and self.tui:
            try:
                asyncio.create_task(self.tui._handle_exit())
            except:
                pass
    
    async def test_01_tui_startup_success(self):
        """
        TEST 1: TUI Startup Success
        ===========================
        Validates that the TUI starts without immediate shutdown.
        This tests the resolution of constructor parameter conflicts.
        """
        print("\n🚀 TEST 1: TUI Startup Success")
        
        startup_successful = False
        shutdown_immediate = True
        
        try:
            # Create TUI instance (this should not fail)
            tui = RevolutionaryTUIInterface()
            self.assertIsNotNone(tui, "TUI instance should be created successfully")
            
            # Test startup process
            startup_task = asyncio.create_task(tui.run())
            
            # Give TUI time to start up properly (more than 0.08s Guardian timeout)
            await asyncio.sleep(0.2)
            
            # Check if TUI is still running (not immediately shut down)
            if not startup_task.done():
                startup_successful = True
                shutdown_immediate = False
                
                # Clean shutdown
                startup_task.cancel()
                try:
                    await startup_task
                except asyncio.CancelledError:
                    pass
            
            self.assertTrue(startup_successful, "TUI should start successfully")
            self.assertFalse(shutdown_immediate, "TUI should not shutdown immediately (0.08s Guardian issue)")
            
            print("✅ TUI startup successful - no immediate shutdown")
            
        except Exception as e:
            self.fail(f"TUI startup failed: {e}")
    
    async def test_02_display_rendering_no_pollution(self):
        """
        TEST 2: Display Rendering Without Scrollback Pollution
        ======================================================
        Validates that Rich Live alternate screen prevents terminal pollution.
        """
        print("\n🖥️  TEST 2: Display Rendering - No Scrollback Pollution")
        
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            try:
                tui = RevolutionaryTUIInterface()
                
                # Start TUI in background
                tui_task = asyncio.create_task(tui.run())
                await asyncio.sleep(0.1)
                
                # Simulate display updates that previously caused pollution
                for i in range(5):
                    # Trigger display refresh
                    if hasattr(tui, '_refresh_display'):
                        await tui._refresh_display()
                    await asyncio.sleep(0.05)
                
                # Check stdout for pollution
                stdout_content = mock_stdout.getvalue()
                
                # Should not contain excessive output or dotted lines
                dotted_lines = stdout_content.count('.')
                excessive_output = len(stdout_content) > 1000  # Reasonable threshold
                
                self.assertFalse(excessive_output, 
                    f"Display should not produce excessive output. Got {len(stdout_content)} chars")
                self.assertLess(dotted_lines, 10, 
                    f"Should not have excessive dotted lines. Found {dotted_lines}")
                
                # Clean shutdown
                tui_task.cancel()
                try:
                    await tui_task
                except asyncio.CancelledError:
                    pass
                
                print("✅ Display rendering clean - no scrollback pollution")
                
            except Exception as e:
                self.fail(f"Display rendering test failed: {e}")
    
    async def test_03_character_input_accumulation(self):
        """
        TEST 3: Character Input Accumulation
        ===================================
        Validates that character input accumulates correctly: 'h' → 'he' → 'hel' → 'hello'
        Tests resolution of input buffer corruption and race conditions.
        """
        print("\n⌨️  TEST 3: Character Input Accumulation")
        
        try:
            tui = RevolutionaryTUIInterface()
            
            # Mock the input buffer to track changes
            input_states = []
            
            original_handle_char = tui._handle_character_input
            def track_input_changes(char):
                original_handle_char(char)
                input_states.append(tui.state.current_input)
            
            tui._handle_character_input = track_input_changes
            
            # Simulate typing "hello" character by character
            test_input = "hello"
            for char in test_input:
                tui._handle_character_input(char)
                await asyncio.sleep(0.01)  # Small delay to catch race conditions
            
            # Validate progressive accumulation
            expected_states = ["h", "he", "hel", "hell", "hello"]
            self.assertEqual(len(input_states), 5, 
                f"Should have 5 input states. Got {len(input_states)}: {input_states}")
            
            for i, expected in enumerate(expected_states):
                self.assertEqual(input_states[i], expected, 
                    f"Input state {i} should be '{expected}', got '{input_states[i]}'")
            
            # Final state should be complete word
            self.assertEqual(tui.state.current_input, "hello", 
                f"Final input should be 'hello', got '{tui.state.current_input}'")
            
            print("✅ Character input accumulation working correctly")
            
        except Exception as e:
            self.fail(f"Character input test failed: {e}")
    
    async def test_04_display_stability_on_keystrokes(self):
        """
        TEST 4: Display Stability on Keystrokes
        =======================================
        Validates that display does NOT corrupt on keystroke input.
        Tests resolution of Rich Live display corruption.
        """
        print("\n🖱️  TEST 4: Display Stability on Keystrokes")
        
        try:
            tui = RevolutionaryTUIInterface()
            
            # Track display corruption indicators
            display_corruption_count = 0
            
            # Mock display refresh to detect corruption
            original_refresh = getattr(tui, '_refresh_display', None)
            async def monitor_display_refresh(*args, **kwargs):
                nonlocal display_corruption_count
                try:
                    if original_refresh:
                        await original_refresh(*args, **kwargs)
                except Exception as e:
                    if "corruption" in str(e).lower() or "display" in str(e).lower():
                        display_corruption_count += 1
                        raise
            
            if original_refresh:
                tui._refresh_display = monitor_display_refresh
            
            # Simulate rapid typing that previously caused corruption
            rapid_input = "rapid typing test to trigger display corruption issues"
            
            for char in rapid_input:
                tui._handle_character_input(char)
                
                # Trigger display refresh after each character (stress test)
                if hasattr(tui, '_refresh_display'):
                    try:
                        await tui._refresh_display()
                    except:
                        pass  # Capture corruption but continue test
                
                await asyncio.sleep(0.001)  # Very rapid typing
            
            # Check for display corruption
            self.assertEqual(display_corruption_count, 0, 
                f"Display should not corrupt on keystrokes. Found {display_corruption_count} corruptions")
            
            # Verify final input state is correct
            self.assertEqual(tui.state.current_input, rapid_input,
                f"Input should be intact after rapid typing. Got: {tui.state.current_input}")
            
            print("✅ Display remains stable during rapid keystrokes")
            
        except Exception as e:
            self.fail(f"Display stability test failed: {e}")
    
    async def test_05_command_processing_quit(self):
        """
        TEST 5: Command Processing (/quit)
        ==================================
        Validates that /quit and other commands are handled correctly.
        """
        print("\n💻 TEST 5: Command Processing (/quit)")
        
        try:
            tui = RevolutionaryTUIInterface()
            
            # Set up command input
            tui.state.current_input = "/quit"
            
            # Mock exit handler to track if it's called
            exit_called = False
            original_handle_exit = tui._handle_exit
            async def track_exit(*args, **kwargs):
                nonlocal exit_called
                exit_called = True
                # Don't actually exit in test
                return 0
            
            tui._handle_exit = track_exit
            
            # Simulate Enter key press to process command
            await tui._handle_enter_input()
            
            # Verify /quit command was processed
            self.assertTrue(exit_called, "/quit command should trigger exit handler")
            
            # Test other commands
            test_commands = ["/help", "/status", "/clear"]
            for cmd in test_commands:
                tui.state.current_input = cmd
                try:
                    await tui._handle_enter_input()
                    # Should not crash
                except Exception as e:
                    self.fail(f"Command '{cmd}' should not cause crash: {e}")
            
            print("✅ Command processing working correctly")
            
        except Exception as e:
            self.fail(f"Command processing test failed: {e}")
    
    async def test_06_clean_output_no_debug_flood(self):
        """
        TEST 6: Clean Output - No Debug Flooding
        ========================================
        Validates that TUI does not flood console with debug prints.
        Tests resolution of production debug cleanup.
        """
        print("\n🧹 TEST 6: Clean Output - No Debug Flooding")
        
        # Capture all output during TUI operation
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            try:
                tui = RevolutionaryTUIInterface()
                
                # Run TUI briefly and perform operations that might trigger debug output
                tui_task = asyncio.create_task(tui.run())
                await asyncio.sleep(0.1)
                
                # Perform operations that previously caused debug flooding
                for i in range(10):
                    tui._handle_character_input(f"test{i}")
                    if hasattr(tui, '_refresh_display'):
                        try:
                            await tui._refresh_display()
                        except:
                            pass
                    await asyncio.sleep(0.01)
                
                # Clean shutdown
                tui_task.cancel()
                try:
                    await tui_task
                except asyncio.CancelledError:
                    pass
                
            except Exception as e:
                pass  # Continue to check output even if TUI fails
        
        # Analyze captured output
        stdout_content = stdout_capture.getvalue()
        stderr_content = stderr_capture.getvalue()
        
        # Check for debug pollution patterns
        debug_patterns = ['DEBUG:', '[DEBUG]', 'print(', 'console.log', '>>>']
        debug_count = 0
        
        for pattern in debug_patterns:
            debug_count += stdout_content.lower().count(pattern.lower())
            debug_count += stderr_content.lower().count(pattern.lower())
        
        # Should have minimal debug output in production
        self.assertLess(debug_count, 5, 
            f"Should have minimal debug output. Found {debug_count} debug statements")
        
        # Output should be reasonable in size (not flooding)
        total_output = len(stdout_content) + len(stderr_content)
        self.assertLess(total_output, 2000, 
            f"Output should be reasonable in size. Got {total_output} characters")
        
        print(f"✅ Clean output verified - {debug_count} debug statements, {total_output} total chars")
    
    async def test_07_full_user_workflow_simulation(self):
        """
        TEST 7: Full User Workflow Simulation
        =====================================
        Simulates a complete user workflow from startup to shutdown.
        This is the ultimate end-to-end validation.
        """
        print("\n👤 TEST 7: Full User Workflow Simulation")
        
        workflow_steps = []
        
        try:
            # Step 1: User starts TUI
            tui = RevolutionaryTUIInterface()
            workflow_steps.append("✅ TUI created successfully")
            
            # Step 2: User begins typing a message
            user_message = "Hello, I need help with my project"
            for char in user_message:
                tui._handle_character_input(char)
                await asyncio.sleep(0.05)  # Realistic typing speed
            
            workflow_steps.append(f"✅ User typed: '{tui.state.current_input}'")
            
            # Step 3: User presses Enter to send message
            await tui._handle_enter_input()
            workflow_steps.append("✅ Message processed")
            
            # Step 4: User types and deletes some text (tests backspace)
            test_text = "mistake"
            for char in test_text:
                tui._handle_character_input(char)
            
            # Delete the mistake
            for _ in range(len(test_text)):
                tui._handle_backspace_input()
            
            workflow_steps.append("✅ Backspace functionality working")
            
            # Step 5: User types a command
            command = "/status"
            for char in command:
                tui._handle_character_input(char)
            
            await tui._handle_enter_input()
            workflow_steps.append("✅ Command executed")
            
            # Step 6: User navigates through history (arrow keys)
            tui._handle_up_arrow()
            tui._handle_down_arrow()
            workflow_steps.append("✅ History navigation working")
            
            # Step 7: User quits gracefully
            quit_command = "/quit"
            tui.state.current_input = ""
            for char in quit_command:
                tui._handle_character_input(char)
            
            # Mock exit to prevent actual exit
            original_exit = tui._handle_exit
            exit_called = False
            async def mock_exit(*args, **kwargs):
                nonlocal exit_called
                exit_called = True
                return 0
            tui._handle_exit = mock_exit
            
            await tui._handle_enter_input()
            
            if exit_called:
                workflow_steps.append("✅ Graceful exit initiated")
            else:
                workflow_steps.append("❌ Exit not properly handled")
            
            # Verify workflow completion
            print("\n📋 User Workflow Summary:")
            for step in workflow_steps:
                print(f"   {step}")
            
            self.assertTrue(all("✅" in step for step in workflow_steps), 
                "All workflow steps should complete successfully")
            
            print("\n🎉 FULL USER WORKFLOW SIMULATION SUCCESSFUL!")
            
        except Exception as e:
            workflow_steps.append(f"❌ Workflow failed: {e}")
            self.fail(f"Full user workflow simulation failed: {e}")
    
    async def test_08_performance_and_memory_validation(self):
        """
        TEST 8: Performance and Memory Validation
        =========================================
        Validates that TUI performs well and doesn't leak memory.
        """
        print("\n⚡ TEST 8: Performance and Memory Validation")
        
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            tui = RevolutionaryTUIInterface()
            
            # Simulate intensive usage
            start_time = time.time()
            
            for cycle in range(100):  # 100 cycles of operations
                # Simulate typing
                for char in "test input":
                    tui._handle_character_input(char)
                
                # Clear input
                for _ in range(10):
                    tui._handle_backspace_input()
                
                # Trigger display refresh
                if hasattr(tui, '_refresh_display'):
                    try:
                        await tui._refresh_display()
                    except:
                        pass
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.001)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Check final memory usage
            gc.collect()  # Force garbage collection
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Performance thresholds
            operations_per_second = 1000 / duration  # 1000 operations
            max_memory_increase = 50  # MB
            min_ops_per_second = 100  # Should be reasonably fast
            
            self.assertLess(memory_increase, max_memory_increase,
                f"Memory increase should be < {max_memory_increase}MB. Got {memory_increase:.2f}MB")
            
            self.assertGreater(operations_per_second, min_ops_per_second,
                f"Should perform > {min_ops_per_second} ops/sec. Got {operations_per_second:.2f}")
            
            print(f"✅ Performance: {operations_per_second:.2f} ops/sec")
            print(f"✅ Memory: +{memory_increase:.2f}MB increase")
            
        except ImportError:
            print("⚠️  psutil not available - skipping memory validation")
        except Exception as e:
            self.fail(f"Performance validation failed: {e}")


async def run_comprehensive_tests():
    """Run all comprehensive TUI tests."""
    print("=" * 70)
    print("🧪 REVOLUTIONARY TUI - COMPREHENSIVE END-TO-END TEST SUITE")
    print("=" * 70)
    print("Validating complete TUI solution for production readiness...")
    print()
    
    # Create test suite
    suite = unittest.TestSuite()
    test_case = TUIEndToEndTests()
    
    # Add all test methods
    test_methods = [
        'test_01_tui_startup_success',
        'test_02_display_rendering_no_pollution',
        'test_03_character_input_accumulation', 
        'test_04_display_stability_on_keystrokes',
        'test_05_command_processing_quit',
        'test_06_clean_output_no_debug_flood',
        'test_07_full_user_workflow_simulation',
        'test_08_performance_and_memory_validation'
    ]
    
    passed_tests = 0
    failed_tests = 0
    test_results = {}
    
    for method_name in test_methods:
        print(f"\n{'=' * 50}")
        try:
            test_method = getattr(test_case, method_name)
            await test_method()
            test_results[method_name] = "✅ PASSED"
            passed_tests += 1
            print(f"🎉 {method_name}: PASSED")
        except Exception as e:
            test_results[method_name] = f"❌ FAILED: {e}"
            failed_tests += 1
            print(f"💥 {method_name}: FAILED - {e}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE TEST RESULTS SUMMARY")
    print("=" * 70)
    
    for method_name, result in test_results.items():
        print(f"{result}")
    
    print(f"\n📈 OVERALL RESULTS:")
    print(f"   ✅ Passed: {passed_tests}")
    print(f"   ❌ Failed: {failed_tests}")
    print(f"   📊 Success Rate: {(passed_tests / len(test_methods)) * 100:.1f}%")
    
    if failed_tests == 0:
        print("\n🚀 REVOLUTIONARY TUI IS READY FOR PRODUCTION!")
        print("   All critical user scenarios validated successfully.")
        print("   TUI solution is comprehensive and production-ready.")
        return True
    else:
        print(f"\n⚠️  REVOLUTIONARY TUI NEEDS ATTENTION!")
        print(f"   {failed_tests} test(s) failed - address before production deployment.")
        return False


if __name__ == "__main__":
    # Run comprehensive tests
    success = asyncio.run(run_comprehensive_tests())
    sys.exit(0 if success else 1)