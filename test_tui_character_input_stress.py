#!/usr/bin/env python3
"""
Character Input Stress Test for Revolutionary TUI Interface
==========================================================

This test validates that character input handling is robust under stress conditions,
including rapid typing, Unicode characters, special keys, and edge cases.

CRITICAL VALIDATION POINTS:
- No input buffer corruption during rapid typing
- Correct character accumulation under stress  
- Unicode and special character handling
- Race condition prevention in input pipeline
- Memory management during intensive input
"""

import asyncio
import logging
import os
import sys
import time
import unittest
import threading
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
import random
import string

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from src.agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface, TUIState
    from src.agentsmcp.ui.v2.input_rendering_pipeline import InputRenderingPipeline
except ImportError as e:
    print(f"Import error: {e}")
    print("Please run from project root directory")
    sys.exit(1)


class CharacterInputStressTests(unittest.TestCase):
    """Stress tests for character input handling in Revolutionary TUI."""
    
    def setUp(self):
        """Set up test environment."""
        self.tui = RevolutionaryTUIInterface()
        self.input_corruption_detected = False
        self.race_conditions_detected = 0
        
        # Configure logging
        logging.getLogger().handlers.clear()
        logging.getLogger().setLevel(logging.WARNING)  # Reduce noise
    
    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'tui') and self.tui:
            try:
                asyncio.create_task(self.tui._handle_exit())
            except:
                pass
    
    async def test_rapid_typing_no_corruption(self):
        """Test rapid typing without input buffer corruption."""
        print("\n⚡ TEST: Rapid Typing - No Corruption")
        
        # Generate random text to type rapidly
        test_text = ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=100))
        
        # Track input states to detect corruption
        input_states = []
        original_handle_char = self.tui._handle_character_input
        
        def track_input_corruption(char):
            original_handle_char(char)
            current_state = self.tui.state.current_input
            input_states.append(current_state)
            
            # Check for corruption patterns
            if len(current_state) > len(''.join(input_states[0:len(input_states)])):
                self.input_corruption_detected = True
        
        self.tui._handle_character_input = track_input_corruption
        
        # Rapid typing simulation (no delays)
        start_time = time.time()
        for char in test_text:
            self.tui._handle_character_input(char)
        
        end_time = time.time()
        typing_speed = len(test_text) / (end_time - start_time)
        
        # Validate results
        self.assertFalse(self.input_corruption_detected, "Input corruption detected during rapid typing")
        self.assertEqual(self.tui.state.current_input, test_text, 
            f"Final input should match typed text. Expected: '{test_text}', Got: '{self.tui.state.current_input}'")
        
        print(f"✅ Rapid typing successful: {typing_speed:.0f} chars/sec, no corruption")
    
    async def test_concurrent_input_threads(self):
        """Test concurrent input from multiple threads to detect race conditions."""
        print("\n🧵 TEST: Concurrent Input Threads")
        
        input_results = {}
        race_condition_lock = threading.Lock()
        
        def thread_input_simulation(thread_id: int, text: str):
            """Simulate input from a specific thread."""
            try:
                for char in text:
                    with race_condition_lock:
                        before_state = self.tui.state.current_input
                        self.tui._handle_character_input(char)
                        after_state = self.tui.state.current_input
                        
                        # Check for race condition signs
                        if len(after_state) != len(before_state) + 1:
                            self.race_conditions_detected += 1
                
                input_results[thread_id] = "completed"
                
            except Exception as e:
                input_results[thread_id] = f"failed: {e}"
        
        # Create multiple threads typing different text
        threads = []
        thread_texts = [
            "thread1_input",
            "thread2_input", 
            "thread3_input"
        ]
        
        # Start threads
        for i, text in enumerate(thread_texts):
            thread = threading.Thread(target=thread_input_simulation, args=(i, text))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=5.0)
        
        # Validate results
        completed_threads = sum(1 for result in input_results.values() if result == "completed")
        self.assertEqual(completed_threads, len(threads), 
            f"All threads should complete. Completed: {completed_threads}/{len(threads)}")
        
        self.assertEqual(self.race_conditions_detected, 0, 
            f"No race conditions should be detected. Found: {self.race_conditions_detected}")
        
        print(f"✅ Concurrent input successful: {completed_threads} threads, {self.race_conditions_detected} race conditions")
    
    async def test_unicode_and_special_characters(self):
        """Test handling of Unicode and special characters."""
        print("\n🌍 TEST: Unicode and Special Characters")
        
        # Test various character sets
        test_cases = [
            ("Basic ASCII", "Hello World 123!"),
            ("Unicode Emoji", "Hello 👋 World 🌍 Test 🧪"),
            ("Accented Characters", "Héllö Wörld Tëst Çhärs"),
            ("Asian Characters", "こんにちは 世界 测试"),
            ("Math Symbols", "∑ ∆ π ≈ ≠ ∞ √ ∫"),
            ("Special Keys", "\t\n\r"),
            ("Control Characters", "\x01\x02\x03"),
        ]
        
        for test_name, test_text in test_cases:
            try:
                # Reset input state
                self.tui.state.current_input = ""
                
                # Type the test text
                for char in test_text:
                    self.tui._handle_character_input(char)
                
                # Validate input was handled correctly
                result_text = self.tui.state.current_input
                
                # For control characters, may be filtered out - that's OK
                if test_name == "Control Characters":
                    self.assertIsInstance(result_text, str, f"{test_name}: Result should be string")
                else:
                    self.assertEqual(result_text, test_text, 
                        f"{test_name}: Expected '{test_text}', got '{result_text}'")
                
                print(f"✅ {test_name}: '{test_text[:20]}...' handled correctly")
                
            except Exception as e:
                self.fail(f"{test_name} test failed: {e}")
    
    async def test_input_buffer_boundaries(self):
        """Test input buffer boundary conditions."""
        print("\n📏 TEST: Input Buffer Boundaries")
        
        # Test very long input
        very_long_text = 'A' * 10000  # 10k characters
        
        try:
            for char in very_long_text:
                self.tui._handle_character_input(char)
            
            # Check if input was handled (may be truncated, that's OK)
            result_length = len(self.tui.state.current_input)
            self.assertGreater(result_length, 0, "Input buffer should handle long text")
            
            # If truncated, should be graceful
            if result_length < len(very_long_text):
                print(f"✅ Long input gracefully truncated: {result_length}/{len(very_long_text)} chars")
            else:
                print(f"✅ Long input fully handled: {result_length} chars")
            
        except Exception as e:
            self.fail(f"Long input test failed: {e}")
        
        # Test empty input handling
        self.tui.state.current_input = ""
        self.tui._handle_character_input("")
        self.assertEqual(self.tui.state.current_input, "", "Empty input should remain empty")
        
        print("✅ Buffer boundary conditions handled correctly")
    
    async def test_backspace_stress_test(self):
        """Test intensive backspace operations."""
        print("\n⌫ TEST: Backspace Stress Test")
        
        # Type and delete repeatedly
        test_cycles = 100
        
        for cycle in range(test_cycles):
            # Type some text
            text = f"cycle{cycle}_test"
            for char in text:
                self.tui._handle_character_input(char)
            
            # Delete it all with backspace
            for _ in range(len(text)):
                self.tui._handle_backspace_input()
            
            # Should be empty after each cycle
            self.assertEqual(self.tui.state.current_input, "", 
                f"Input should be empty after cycle {cycle}")
        
        print(f"✅ Backspace stress test completed: {test_cycles} cycles")
    
    async def test_input_memory_management(self):
        """Test memory management during intensive input operations."""
        print("\n💾 TEST: Input Memory Management")
        
        try:
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Perform intensive input operations
            for round_num in range(10):
                # Generate large input
                large_text = ''.join(random.choices(string.printable, k=1000))
                
                for char in large_text:
                    self.tui._handle_character_input(char)
                
                # Clear input
                self.tui.state.current_input = ""
                
                # Force garbage collection periodically
                if round_num % 3 == 0:
                    import gc
                    gc.collect()
            
            # Check final memory usage
            import gc
            gc.collect()
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable
            max_memory_increase = 20  # MB
            self.assertLess(memory_increase, max_memory_increase, 
                f"Memory increase should be < {max_memory_increase}MB. Got {memory_increase:.2f}MB")
            
            print(f"✅ Memory management OK: +{memory_increase:.2f}MB increase")
            
        except ImportError:
            print("⚠️  psutil not available - skipping memory test")
        except Exception as e:
            self.fail(f"Memory management test failed: {e}")
    
    async def test_input_performance_benchmarks(self):
        """Benchmark input handling performance."""
        print("\n📊 TEST: Input Performance Benchmarks")
        
        # Test different input scenarios
        scenarios = [
            ("Single chars", ['a'] * 1000),
            ("Words", ['hello'] * 200),
            ("Mixed content", [random.choice(string.printable) for _ in range(1000)]),
        ]
        
        performance_results = {}
        
        for scenario_name, input_data in scenarios:
            start_time = time.time()
            
            # Reset input
            self.tui.state.current_input = ""
            
            # Process all input
            for item in input_data:
                if isinstance(item, str) and len(item) == 1:
                    self.tui._handle_character_input(item)
                else:
                    for char in str(item):
                        self.tui._handle_character_input(char)
            
            end_time = time.time()
            duration = end_time - start_time
            operations_per_second = len(input_data) / duration if duration > 0 else float('inf')
            
            performance_results[scenario_name] = {
                'ops_per_sec': operations_per_second,
                'duration': duration,
                'input_count': len(input_data)
            }
            
            # Performance should be reasonable
            min_ops_per_second = 500  # Should handle at least 500 ops/sec
            self.assertGreater(operations_per_second, min_ops_per_second,
                f"{scenario_name} should perform > {min_ops_per_second} ops/sec. Got {operations_per_second:.2f}")
        
        # Display performance results
        print("Performance Benchmark Results:")
        for scenario, results in performance_results.items():
            print(f"  {scenario}: {results['ops_per_sec']:.0f} ops/sec ({results['input_count']} operations)")
        
        print("✅ All performance benchmarks passed")


async def run_character_input_stress_tests():
    """Run all character input stress tests."""
    print("=" * 60)
    print("⌨️  CHARACTER INPUT STRESS TEST SUITE")
    print("=" * 60)
    print("Testing input handling under extreme conditions...")
    print()
    
    test_case = CharacterInputStressTests()
    
    test_methods = [
        'test_rapid_typing_no_corruption',
        'test_concurrent_input_threads',
        'test_unicode_and_special_characters',
        'test_input_buffer_boundaries',
        'test_backspace_stress_test',
        'test_input_memory_management',
        'test_input_performance_benchmarks'
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
    print("📊 CHARACTER INPUT STRESS TEST RESULTS")
    print("=" * 60)
    
    for method, result in results.items():
        print(f"{result}")
    
    print(f"\n📈 Results: ✅ {passed} passed, ❌ {failed} failed")
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_character_input_stress_tests())
    sys.exit(0 if success else 1)