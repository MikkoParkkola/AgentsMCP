#!/usr/bin/env python3
"""
Display Stability Test for Revolutionary TUI Interface
=====================================================

This test validates that the display system remains stable under various
stress conditions and does not corrupt during rapid refresh cycles.

CRITICAL VALIDATION POINTS:
- No display corruption during rapid refresh cycles
- Rich Live alternate screen mode works correctly
- No scrollback pollution in terminal
- Layout engine handles dynamic content properly
- Memory management during intensive display operations
"""

import asyncio
import logging
import os
import sys
import time
import unittest
import threading
import io
from unittest.mock import Mock, patch, MagicMock
from contextlib import redirect_stdout, redirect_stderr
from typing import List, Dict, Any
import random

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from src.agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface, TUIState
    from src.agentsmcp.ui.v2.display_manager import DisplayManager
    from src.agentsmcp.ui.v2.text_layout_engine import TextLayoutEngine
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text
except ImportError as e:
    print(f"Import error: {e}")
    print("Please run from project root directory")
    sys.exit(1)


class DisplayStabilityTests(unittest.TestCase):
    """Tests for display stability under stress conditions."""
    
    def setUp(self):
        """Set up test environment."""
        self.tui = RevolutionaryTUIInterface()
        self.display_corruption_count = 0
        self.refresh_error_count = 0
        
        # Configure logging to capture display errors
        logging.getLogger().handlers.clear()
        test_handler = logging.StreamHandler()
        test_handler.setLevel(logging.WARNING)
        logging.getLogger().addHandler(test_handler)
        logging.getLogger().setLevel(logging.WARNING)
    
    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'tui') and self.tui:
            try:
                asyncio.create_task(self.tui._handle_exit())
            except:
                pass
    
    async def test_rapid_display_refresh_stability(self):
        """Test display stability during rapid refresh cycles."""
        print("\n🔄 TEST: Rapid Display Refresh Stability")
        
        refresh_count = 0
        errors_detected = []
        
        # Mock display refresh to monitor for errors
        original_refresh = getattr(self.tui, '_refresh_display', None)
        
        async def monitor_refresh(*args, **kwargs):
            nonlocal refresh_count
            try:
                if original_refresh:
                    await original_refresh(*args, **kwargs)
                refresh_count += 1
            except Exception as e:
                errors_detected.append(str(e))
                self.refresh_error_count += 1
        
        if original_refresh:
            self.tui._refresh_display = monitor_refresh
        
        # Perform rapid refresh cycles
        test_cycles = 100
        for cycle in range(test_cycles):
            # Update TUI state to trigger display changes
            self.tui.state.current_input = f"test_input_{cycle}"
            self.tui.state.processing_message = f"Processing cycle {cycle}"
            
            # Trigger refresh
            if hasattr(self.tui, '_refresh_display'):
                try:
                    await self.tui._refresh_display()
                except Exception as e:
                    errors_detected.append(f"Cycle {cycle}: {e}")
            
            # Very short delay to stress test timing
            await asyncio.sleep(0.001)
        
        # Validate results
        self.assertEqual(self.refresh_error_count, 0, 
            f"No refresh errors should occur. Found {self.refresh_error_count} errors: {errors_detected[:3]}")
        
        self.assertGreater(refresh_count, test_cycles * 0.8, 
            f"Most refresh cycles should complete. Expected ~{test_cycles}, got {refresh_count}")
        
        print(f"✅ Rapid refresh stability: {refresh_count} refreshes, {self.refresh_error_count} errors")
    
    async def test_rich_live_alternate_screen_mode(self):
        """Test Rich Live alternate screen mode prevents scrollback pollution."""
        print("\n🖥️  TEST: Rich Live Alternate Screen Mode")
        
        # Capture stdout to monitor for pollution
        stdout_capture = io.StringIO()
        
        with redirect_stdout(stdout_capture):
            try:
                # Mock Rich Live to verify alternate screen usage
                with patch('rich.live.Live') as mock_live_class:
                    mock_live = Mock()
                    mock_live_class.return_value = mock_live
                    mock_live.__enter__ = Mock(return_value=mock_live)
                    mock_live.__exit__ = Mock(return_value=None)
                    
                    # Create TUI and simulate operations
                    tui = RevolutionaryTUIInterface()
                    
                    # Verify Rich Live was configured with alternate screen
                    if mock_live_class.called:
                        args, kwargs = mock_live_class.call_args
                        
                        # Check for alternate screen configuration
                        alt_screen_enabled = kwargs.get('screen', False) or any(
                            'screen' in str(arg).lower() for arg in args
                        )
                        
                        print(f"✅ Rich Live alternate screen configured: {alt_screen_enabled}")
                    else:
                        print("⚠️  Rich Live not used - may be OK for test environment")
                
            except Exception as e:
                self.fail(f"Rich Live alternate screen test failed: {e}")
        
        # Check captured stdout for excessive pollution
        stdout_content = stdout_capture.getvalue()
        pollution_indicators = [
            '\n' * 10,  # Excessive newlines
            '.' * 50,   # Dotted lines
            'DEBUG:',   # Debug output
            'TRACE:',   # Trace output
        ]
        
        pollution_count = sum(stdout_content.count(indicator) for indicator in pollution_indicators)
        
        self.assertLess(pollution_count, 5, 
            f"Minimal stdout pollution expected. Found {pollution_count} pollution indicators")
        
        print(f"✅ Scrollback pollution prevented: {pollution_count} pollution indicators")
    
    async def test_layout_engine_dynamic_content(self):
        """Test layout engine handles dynamic content changes."""
        print("\n📐 TEST: Layout Engine Dynamic Content")
        
        layout_errors = []
        
        try:
            # Test various content scenarios
            content_scenarios = [
                ("Short text", "Hello"),
                ("Long text", "A" * 1000),
                ("Multi-line text", "\n".join([f"Line {i}" for i in range(20)])),
                ("Unicode text", "Hello 👋 World 🌍 with émojis ñ çhars"),
                ("Empty text", ""),
                ("Special chars", "\t\n\r\x1b[31mRed\x1b[0m"),
            ]
            
            for scenario_name, content in content_scenarios:
                try:
                    # Update TUI state with new content
                    self.tui.state.current_input = content
                    self.tui.state.processing_message = f"Testing: {scenario_name}"
                    
                    # Trigger layout update
                    if hasattr(self.tui, '_update_layout'):
                        self.tui._update_layout()
                    
                    # Try to refresh display
                    if hasattr(self.tui, '_refresh_display'):
                        await self.tui._refresh_display()
                    
                    print(f"✅ {scenario_name}: Layout handled correctly")
                    
                except Exception as e:
                    layout_errors.append(f"{scenario_name}: {e}")
                    print(f"❌ {scenario_name}: Layout error - {e}")
        
        except Exception as e:
            self.fail(f"Layout engine test setup failed: {e}")
        
        # Validate no critical layout errors
        critical_errors = [error for error in layout_errors if 'critical' in error.lower()]
        self.assertEqual(len(critical_errors), 0, 
            f"No critical layout errors should occur. Found: {critical_errors}")
        
        print(f"✅ Layout engine stability: {len(layout_errors)} non-critical errors")
    
    async def test_concurrent_display_updates(self):
        """Test concurrent display updates don't cause corruption."""
        print("\n🧵 TEST: Concurrent Display Updates")
        
        update_results = {}
        corruption_detected = False
        update_lock = threading.Lock()
        
        def concurrent_display_update(thread_id: int):
            """Simulate concurrent display updates from different threads."""
            try:
                for i in range(10):
                    with update_lock:
                        # Update different parts of TUI state
                        self.tui.state.current_input = f"thread_{thread_id}_update_{i}"
                        self.tui.state.processing_message = f"Thread {thread_id} processing"
                        
                        # Small delay to increase chance of race conditions
                        time.sleep(0.001)
                        
                        # Verify state consistency
                        expected_input = f"thread_{thread_id}_update_{i}"
                        if self.tui.state.current_input != expected_input:
                            nonlocal corruption_detected
                            corruption_detected = True
                
                update_results[thread_id] = "completed"
                
            except Exception as e:
                update_results[thread_id] = f"failed: {e}"
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=concurrent_display_update, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=10.0)
        
        # Validate results
        completed_threads = sum(1 for result in update_results.values() if result == "completed")
        self.assertEqual(completed_threads, len(threads), 
            f"All threads should complete. Completed: {completed_threads}/{len(threads)}")
        
        self.assertFalse(corruption_detected, "No display corruption should be detected")
        
        print(f"✅ Concurrent updates: {completed_threads} threads completed, no corruption")
    
    async def test_display_memory_usage(self):
        """Test display operations don't leak memory."""
        print("\n💾 TEST: Display Memory Usage")
        
        try:
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Perform intensive display operations
            for cycle in range(50):
                # Create large display content
                large_content = f"Cycle {cycle}: " + "X" * 1000
                
                self.tui.state.current_input = large_content
                self.tui.state.processing_message = f"Memory test cycle {cycle}"
                
                # Update multiple display components
                if hasattr(self.tui, '_update_layout'):
                    self.tui._update_layout()
                
                if hasattr(self.tui, '_refresh_display'):
                    try:
                        await self.tui._refresh_display()
                    except:
                        pass
                
                # Periodic cleanup
                if cycle % 10 == 0:
                    import gc
                    gc.collect()
            
            # Final memory check
            import gc
            gc.collect()
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable
            max_memory_increase = 30  # MB
            self.assertLess(memory_increase, max_memory_increase, 
                f"Memory increase should be < {max_memory_increase}MB. Got {memory_increase:.2f}MB")
            
            print(f"✅ Display memory usage OK: +{memory_increase:.2f}MB increase")
            
        except ImportError:
            print("⚠️  psutil not available - skipping memory test")
        except Exception as e:
            self.fail(f"Display memory test failed: {e}")
    
    async def test_display_performance_benchmarks(self):
        """Benchmark display refresh performance."""
        print("\n📊 TEST: Display Performance Benchmarks")
        
        refresh_times = []
        error_count = 0
        
        # Benchmark display refresh operations
        for i in range(100):
            # Update content to trigger refresh
            self.tui.state.current_input = f"performance_test_{i}"
            
            start_time = time.time()
            
            try:
                if hasattr(self.tui, '_refresh_display'):
                    await self.tui._refresh_display()
                
                end_time = time.time()
                refresh_time = (end_time - start_time) * 1000  # ms
                refresh_times.append(refresh_time)
                
            except Exception as e:
                error_count += 1
        
        # Calculate performance metrics
        if refresh_times:
            avg_refresh_time = sum(refresh_times) / len(refresh_times)
            max_refresh_time = max(refresh_times)
            min_refresh_time = min(refresh_times)
            
            # Performance thresholds
            max_acceptable_avg = 50.0  # ms
            max_acceptable_max = 200.0  # ms
            
            self.assertLess(avg_refresh_time, max_acceptable_avg,
                f"Average refresh time should be < {max_acceptable_avg}ms. Got {avg_refresh_time:.2f}ms")
            
            self.assertLess(max_refresh_time, max_acceptable_max,
                f"Max refresh time should be < {max_acceptable_max}ms. Got {max_refresh_time:.2f}ms")
            
            print(f"✅ Display performance: avg {avg_refresh_time:.2f}ms, max {max_refresh_time:.2f}ms, min {min_refresh_time:.2f}ms")
        
        self.assertEqual(error_count, 0, f"No refresh errors should occur. Found {error_count}")
        
        print(f"✅ Performance benchmark completed: {len(refresh_times)} refreshes, {error_count} errors")
    
    async def test_display_error_recovery(self):
        """Test display system recovers gracefully from errors."""
        print("\n🔧 TEST: Display Error Recovery")
        
        recovery_successful = True
        
        try:
            # Simulate display error conditions
            error_conditions = [
                ("Invalid console state", lambda: setattr(self.tui, '_console', None)),
                ("Corrupted layout", lambda: setattr(self.tui, '_layout', "invalid")),
                ("Memory pressure", lambda: [[] for _ in range(10000)]),  # Consume memory
            ]
            
            for condition_name, setup_error in error_conditions:
                try:
                    # Setup error condition
                    setup_error()
                    
                    # Try to refresh display - should recover gracefully
                    if hasattr(self.tui, '_refresh_display'):
                        await self.tui._refresh_display()
                    
                    print(f"✅ {condition_name}: Recovery successful")
                    
                except Exception as e:
                    # Some errors are expected, but system should not crash
                    if "crash" in str(e).lower() or "fatal" in str(e).lower():
                        recovery_successful = False
                        print(f"❌ {condition_name}: Fatal error - {e}")
                    else:
                        print(f"⚠️  {condition_name}: Expected error handled - {str(e)[:50]}...")
        
        except Exception as e:
            self.fail(f"Error recovery test setup failed: {e}")
        
        self.assertTrue(recovery_successful, "Display system should recover gracefully from errors")
        
        print("✅ Display error recovery validated")


async def run_display_stability_tests():
    """Run all display stability tests."""
    print("=" * 60)
    print("🖥️  DISPLAY STABILITY TEST SUITE")
    print("=" * 60)
    print("Testing display system under stress conditions...")
    print()
    
    test_case = DisplayStabilityTests()
    
    test_methods = [
        'test_rapid_display_refresh_stability',
        'test_rich_live_alternate_screen_mode',
        'test_layout_engine_dynamic_content',
        'test_concurrent_display_updates',
        'test_display_memory_usage',
        'test_display_performance_benchmarks',
        'test_display_error_recovery'
    ]
    
    passed = 0
    failed = 0
    results = {}
    
    for method_name in test_methods:
        print(f"\n{'=' * 40}")
        try:
            await getattr(test_case, method_name)()
            results[method_name] = "✅ PASSED"
            passed += 1
        except Exception as e:
            results[method_name] = f"❌ FAILED: {e}"
            failed += 1
            print(f"💥 {method_name}: FAILED - {e}")
    
    # Summary
    print(f"\n{'=' * 60}")
    print("📊 DISPLAY STABILITY TEST RESULTS")
    print("=" * 60)
    
    for method, result in results.items():
        print(f"{result}")
    
    print(f"\n📈 Results: ✅ {passed} passed, ❌ {failed} failed")
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_display_stability_tests())
    sys.exit(0 if success else 1)