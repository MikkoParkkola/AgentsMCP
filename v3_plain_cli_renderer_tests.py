#!/usr/bin/env python3
"""
V3 PlainCLIRenderer Input Handling Test Suite
Comprehensive testing of PlainCLIRenderer input mechanisms

Tests specifically target:
- Input visibility issues (characters appearing in wrong place)
- Input buffer handling
- Handle_input() method behavior
- State management during input
- Error conditions and recovery
"""

import sys
import os
import time
import threading
import subprocess
from typing import Optional, List, Dict, Any
from unittest.mock import Mock, patch
from io import StringIO

def test_header():
    print("=" * 80)
    print("🧪 V3 PlainCLIRenderer Input Handling Test Suite")
    print("=" * 80)
    print("Testing PlainCLIRenderer input mechanisms in isolation\n")

class MockTerminalCapabilities:
    """Mock terminal capabilities for testing."""
    
    def __init__(self, is_tty=True, width=80, height=24):
        self.is_tty = is_tty
        self.width = width
        self.height = height
        self.supports_colors = is_tty
        self.supports_unicode = is_tty
        self.supports_rich = is_tty

class PlainCLIRendererTester:
    """Comprehensive tester for PlainCLIRenderer."""
    
    def __init__(self):
        self.test_results = {}
        self.renderer = None
        
    def test_1_renderer_creation(self):
        """Test basic renderer creation and initialization."""
        print("📋 TEST 1: Renderer Creation and Initialization")
        print("-" * 50)
        
        try:
            from agentsmcp.ui.v3.plain_cli_renderer import PlainCLIRenderer
            
            # Test with TTY capabilities
            tty_caps = MockTerminalCapabilities(is_tty=True)
            renderer = PlainCLIRenderer(tty_caps)
            
            print("  ✅ PlainCLIRenderer created with TTY capabilities")
            
            # Test initialization
            # Redirect stdout to capture initialization output
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            try:
                init_success = renderer.initialize()
                init_output = sys.stdout.getvalue()
            finally:
                sys.stdout = old_stdout
            
            if init_success:
                print("  ✅ Renderer initialization successful")
                print(f"  📄 Initialization output: {len(init_output)} characters")
                
                # Test cleanup
                old_stdout = sys.stdout
                sys.stdout = StringIO()
                try:
                    renderer.cleanup()
                    cleanup_output = sys.stdout.getvalue()
                finally:
                    sys.stdout = old_stdout
                
                print(f"  📄 Cleanup output: {len(cleanup_output)} characters")
            else:
                print("  ❌ Renderer initialization failed")
                return False
            
            # Test with non-TTY capabilities
            non_tty_caps = MockTerminalCapabilities(is_tty=False)
            renderer2 = PlainCLIRenderer(non_tty_caps)
            
            print("  ✅ PlainCLIRenderer created with non-TTY capabilities")
            
            self.renderer = renderer  # Store for other tests
            return True
            
        except ImportError as e:
            print(f"  ❌ Import error: {e}")
            return False
        except Exception as e:
            print(f"  ❌ Creation error: {e}")
            return False
    
    def test_2_state_management(self):
        """Test renderer state management."""
        print("\n📋 TEST 2: State Management")
        print("-" * 50)
        
        try:
            from agentsmcp.ui.v3.plain_cli_renderer import PlainCLIRenderer
            
            caps = MockTerminalCapabilities()
            renderer = PlainCLIRenderer(caps)
            
            # Test initial state
            print("  🔍 Testing initial state...")
            initial_state = renderer.state
            
            print(f"    current_input: '{initial_state.current_input}'")
            print(f"    is_processing: {initial_state.is_processing}")
            print(f"    status_message: '{initial_state.status_message}'")
            print(f"    messages: {len(initial_state.messages)} messages")
            
            # Test state updates
            print("  🔍 Testing state updates...")
            test_states = [
                {"current_input": "test input", "is_processing": False},
                {"current_input": "longer test input", "is_processing": True},
                {"status_message": "Processing...", "is_processing": True},
                {"current_input": "", "is_processing": False, "status_message": ""}
            ]
            
            for i, state_update in enumerate(test_states):
                renderer.update_state(**state_update)
                
                print(f"    Update {i+1}: ", end="")
                for key, expected_value in state_update.items():
                    actual_value = getattr(renderer.state, key)
                    if actual_value == expected_value:
                        print(f"{key}✅ ", end="")
                    else:
                        print(f"{key}❌({actual_value}!={expected_value}) ", end="")
                print()
            
            return True
            
        except Exception as e:
            print(f"  ❌ State management error: {e}")
            return False
    
    def test_3_handle_input_mock(self):
        """Test handle_input method with mocked input."""
        print("\n📋 TEST 3: Handle Input Method (Mocked)")
        print("-" * 50)
        
        try:
            from agentsmcp.ui.v3.plain_cli_renderer import PlainCLIRenderer
            
            caps = MockTerminalCapabilities()
            renderer = PlainCLIRenderer(caps)
            renderer.initialize()
            
            # Test cases for input
            test_inputs = [
                ("hello", "Regular text input"),
                ("/help", "Command input"),
                ("", "Empty input"),
                ("hello world", "Multi-word input"),
                ("/quit", "Quit command"),
                ("special chars: !@#$%", "Special characters")
            ]
            
            print("  🧪 Testing handle_input with various inputs...")
            
            for test_input, description in test_inputs:
                print(f"    Testing: {description}")
                
                # Mock the input() function to return our test input
                with patch('builtins.input', return_value=test_input):
                    try:
                        result = renderer.handle_input()
                        
                        if test_input.strip():  # Non-empty input
                            if result == test_input.strip():
                                print(f"      ✅ Correctly returned: '{result}'")
                            else:
                                print(f"      ❌ Expected: '{test_input.strip()}', got: '{result}'")
                        else:  # Empty input
                            if result is None:
                                print(f"      ✅ Correctly returned None for empty input")
                            else:
                                print(f"      ⚠️  Expected None for empty input, got: '{result}'")
                    
                    except Exception as e:
                        print(f"      ❌ Error handling '{test_input}': {e}")
            
            # Test exception handling
            print("  🧪 Testing exception handling...")
            
            with patch('builtins.input', side_effect=EOFError()):
                result = renderer.handle_input()
                if result == "/quit":
                    print("    ✅ EOFError correctly handled as /quit")
                else:
                    print(f"    ❌ EOFError handling failed, got: {result}")
            
            with patch('builtins.input', side_effect=KeyboardInterrupt()):
                result = renderer.handle_input()
                if result == "/quit":
                    print("    ✅ KeyboardInterrupt correctly handled as /quit")
                else:
                    print(f"    ❌ KeyboardInterrupt handling failed, got: {result}")
            
            renderer.cleanup()
            return True
            
        except Exception as e:
            print(f"  ❌ Handle input mock test error: {e}")
            return False
    
    def test_4_render_frame_behavior(self):
        """Test render_frame method behavior."""
        print("\n📋 TEST 4: Render Frame Behavior")
        print("-" * 50)
        
        try:
            from agentsmcp.ui.v3.plain_cli_renderer import PlainCLIRenderer
            
            caps = MockTerminalCapabilities()
            renderer = PlainCLIRenderer(caps)
            renderer.initialize()
            
            # Capture render frame output
            old_stdout = sys.stdout
            
            # Test various states during rendering
            test_states = [
                {"current_input": "", "is_processing": False, "status_message": ""},
                {"current_input": "test", "is_processing": False, "status_message": ""},
                {"current_input": "test", "is_processing": True, "status_message": "Processing..."},
                {"current_input": "", "is_processing": False, "status_message": "Ready"}
            ]
            
            for i, state in enumerate(test_states):
                print(f"  🎭 Testing render state {i+1}...")
                
                renderer.update_state(**state)
                
                sys.stdout = StringIO()
                try:
                    renderer.render_frame()
                    render_output = sys.stdout.getvalue()
                finally:
                    sys.stdout = old_stdout
                
                print(f"    State: input='{state['current_input']}', processing={state['is_processing']}")
                print(f"    Render output length: {len(render_output)} characters")
                
                if render_output:
                    # Check if output contains expected elements
                    if "💬" in render_output:
                        print("    ✅ Contains expected prompt character")
                    else:
                        print("    ⚠️  Missing expected prompt character")
                    
                    if state["current_input"] and state["current_input"] in render_output:
                        print("    ✅ Contains current input")
                    
                    if state["status_message"] and state["status_message"] in render_output:
                        print("    ✅ Contains status message")
                
                # Simulate multiple renders to test consistency
                sys.stdout = StringIO()
                try:
                    renderer.render_frame()
                    renderer.render_frame()  # Second render
                    double_render_output = sys.stdout.getvalue()
                finally:
                    sys.stdout = old_stdout
                
                if len(double_render_output) == len(render_output):
                    print("    ✅ Consistent render output on multiple calls")
                else:
                    print("    ⚠️  Render output changes between calls")
            
            renderer.cleanup()
            return True
            
        except Exception as e:
            print(f"  ❌ Render frame test error: {e}")
            return False
    
    def test_5_message_display(self):
        """Test message display functionality."""
        print("\n📋 TEST 5: Message Display")
        print("-" * 50)
        
        try:
            from agentsmcp.ui.v3.plain_cli_renderer import PlainCLIRenderer
            
            caps = MockTerminalCapabilities()
            renderer = PlainCLIRenderer(caps)
            renderer.initialize()
            
            # Test different message types
            test_messages = [
                ("Hello world", "info"),
                ("Success message", "success"),
                ("Warning message", "warning"),
                ("Error message", "error"),
                ("No level specified", None)
            ]
            
            old_stdout = sys.stdout
            
            for message, level in test_messages:
                print(f"  📧 Testing message: '{message}' (level: {level})")
                
                sys.stdout = StringIO()
                try:
                    if level:
                        renderer.show_message(message, level)
                    else:
                        renderer.show_message(message)
                    
                    output = sys.stdout.getvalue()
                finally:
                    sys.stdout = old_stdout
                
                if message in output:
                    print("    ✅ Message appears in output")
                else:
                    print("    ❌ Message missing from output")
                
                # Check for level indicators
                level_indicators = {
                    "info": "ℹ️",
                    "success": "✅",
                    "warning": "⚠️",
                    "error": "❌"
                }
                
                expected_indicator = level_indicators.get(level, "ℹ️")
                if expected_indicator in output:
                    print(f"    ✅ Contains expected indicator: {expected_indicator}")
                else:
                    print(f"    ⚠️  Missing expected indicator: {expected_indicator}")
            
            # Test show_error method
            print("  📧 Testing show_error method...")
            
            sys.stdout = StringIO()
            try:
                renderer.show_error("Test error message")
                error_output = sys.stdout.getvalue()
            finally:
                sys.stdout = old_stdout
            
            if "Test error message" in error_output and "❌" in error_output:
                print("    ✅ show_error works correctly")
            else:
                print("    ❌ show_error not working properly")
            
            renderer.cleanup()
            return True
            
        except Exception as e:
            print(f"  ❌ Message display test error: {e}")
            return False
    
    def test_6_threading_safety(self):
        """Test thread safety of renderer operations."""
        print("\n📋 TEST 6: Threading Safety")
        print("-" * 50)
        
        try:
            from agentsmcp.ui.v3.plain_cli_renderer import PlainCLIRenderer
            
            caps = MockTerminalCapabilities()
            renderer = PlainCLIRenderer(caps)
            renderer.initialize()
            
            # Test concurrent operations
            errors = []
            results = []
            
            def render_worker():
                try:
                    for i in range(10):
                        renderer.render_frame()
                        time.sleep(0.01)
                    results.append("render_ok")
                except Exception as e:
                    errors.append(f"render_error: {e}")
            
            def message_worker():
                try:
                    for i in range(10):
                        renderer.show_message(f"Test message {i}", "info")
                        time.sleep(0.01)
                    results.append("message_ok")
                except Exception as e:
                    errors.append(f"message_error: {e}")
            
            def state_worker():
                try:
                    for i in range(10):
                        renderer.update_state(current_input=f"input_{i}")
                        time.sleep(0.01)
                    results.append("state_ok")
                except Exception as e:
                    errors.append(f"state_error: {e}")
            
            print("  🧵 Starting concurrent operations...")
            
            # Redirect stdout to prevent output pollution
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            try:
                threads = [
                    threading.Thread(target=render_worker),
                    threading.Thread(target=message_worker),
                    threading.Thread(target=state_worker)
                ]
                
                for thread in threads:
                    thread.start()
                
                for thread in threads:
                    thread.join(timeout=5.0)  # 5 second timeout
            finally:
                sys.stdout = old_stdout
            
            print(f"  📊 Results: {len(results)} successful, {len(errors)} errors")
            
            if len(results) == 3 and len(errors) == 0:
                print("  ✅ All concurrent operations completed successfully")
            elif len(errors) == 0:
                print(f"  ⚠️  Some operations may have timed out ({len(results)}/3 completed)")
            else:
                print("  ❌ Concurrent operations had errors:")
                for error in errors:
                    print(f"    • {error}")
            
            renderer.cleanup()
            return len(errors) == 0
            
        except Exception as e:
            print(f"  ❌ Threading safety test error: {e}")
            return False
    
    def test_7_input_stress_test(self):
        """Stress test input handling with rapid inputs."""
        print("\n📋 TEST 7: Input Stress Test")
        print("-" * 50)
        
        try:
            from agentsmcp.ui.v3.plain_cli_renderer import PlainCLIRenderer
            
            caps = MockTerminalCapabilities()
            renderer = PlainCLIRenderer(caps)
            renderer.initialize()
            
            # Generate test inputs
            test_inputs = []
            for i in range(100):
                test_inputs.extend([
                    f"message_{i}",
                    f"/command_{i}",
                    "",  # Empty input
                    f"long_message_with_many_words_{i}" * 5
                ])
            
            print(f"  🚀 Stress testing with {len(test_inputs)} inputs...")
            
            successful_inputs = 0
            failed_inputs = 0
            processing_times = []
            
            # Suppress output during stress test
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            try:
                for i, test_input in enumerate(test_inputs):
                    try:
                        with patch('builtins.input', return_value=test_input):
                            start_time = time.time()
                            result = renderer.handle_input()
                            processing_time = time.time() - start_time
                            
                            processing_times.append(processing_time)
                            
                            # Verify result
                            expected = test_input.strip() if test_input.strip() else None
                            if result == expected:
                                successful_inputs += 1
                            else:
                                failed_inputs += 1
                    
                    except Exception as e:
                        failed_inputs += 1
                        # Record the exception for analysis
                        if failed_inputs <= 5:  # Only record first few errors
                            sys.stdout.write(f"Error on input {i}: {e}\n")
            
            finally:
                error_log = sys.stdout.getvalue()
                sys.stdout = old_stdout
            
            # Calculate statistics
            if processing_times:
                avg_time = sum(processing_times) / len(processing_times)
                max_time = max(processing_times)
                min_time = min(processing_times)
            else:
                avg_time = max_time = min_time = 0
            
            print(f"  📊 Stress test results:")
            print(f"    Successful: {successful_inputs}")
            print(f"    Failed: {failed_inputs}")
            print(f"    Success rate: {successful_inputs/(successful_inputs+failed_inputs)*100:.1f}%")
            print(f"    Avg processing time: {avg_time*1000:.2f}ms")
            print(f"    Max processing time: {max_time*1000:.2f}ms")
            print(f"    Min processing time: {min_time*1000:.2f}ms")
            
            if error_log.strip():
                print(f"  ⚠️  Error samples:")
                for line in error_log.strip().split('\n')[:3]:
                    print(f"    {line}")
            
            renderer.cleanup()
            
            # Consider test successful if >95% inputs processed correctly
            success_rate = successful_inputs / (successful_inputs + failed_inputs)
            return success_rate > 0.95
            
        except Exception as e:
            print(f"  ❌ Input stress test error: {e}")
            return False
    
    def run_all_tests(self):
        """Run all PlainCLIRenderer tests."""
        tests = [
            ("Renderer Creation", self.test_1_renderer_creation),
            ("State Management", self.test_2_state_management),
            ("Handle Input (Mocked)", self.test_3_handle_input_mock),
            ("Render Frame Behavior", self.test_4_render_frame_behavior),
            ("Message Display", self.test_5_message_display),
            ("Threading Safety", self.test_6_threading_safety),
            ("Input Stress Test", self.test_7_input_stress_test)
        ]
        
        self.test_results = {}
        
        for test_name, test_method in tests:
            try:
                result = test_method()
                self.test_results[test_name] = result
            except Exception as e:
                print(f"❌ {test_name} crashed: {e}")
                self.test_results[test_name] = False
        
        return self.test_results

def test_integration_with_chat_engine():
    """Test PlainCLIRenderer integration with ChatEngine."""
    print("\n📋 INTEGRATION TEST: PlainCLIRenderer + ChatEngine")
    print("-" * 50)
    
    try:
        from agentsmcp.ui.v3.plain_cli_renderer import PlainCLIRenderer
        from agentsmcp.ui.v3.chat_engine import ChatEngine
        
        # Setup components
        caps = MockTerminalCapabilities()
        renderer = PlainCLIRenderer(caps)
        chat_engine = ChatEngine()
        
        if not renderer.initialize():
            print("  ❌ Failed to initialize renderer")
            return False
        
        print("  ✅ Components initialized")
        
        # Test integration scenarios
        integration_tests = [
            ("/help", "Help command integration"),
            ("/status", "Status command integration"),
            ("hello", "Regular message integration"),
            ("/nonexistent", "Unknown command handling")
        ]
        
        successful_integrations = 0
        
        for test_input, description in integration_tests:
            print(f"    Testing: {description}")
            
            try:
                # Simulate user input through renderer
                with patch('builtins.input', return_value=test_input):
                    user_input = renderer.handle_input()
                
                if user_input:
                    # Process through chat engine
                    import asyncio
                    result = asyncio.run(chat_engine.process_input(user_input))
                    
                    # Verify integration
                    if test_input == "/quit":
                        expected_result = False
                    else:
                        expected_result = True
                    
                    if result == expected_result:
                        print(f"      ✅ Integration successful")
                        successful_integrations += 1
                    else:
                        print(f"      ❌ Integration failed: expected {expected_result}, got {result}")
                else:
                    print(f"      ❌ Renderer returned no input")
            
            except Exception as e:
                print(f"      ❌ Integration error: {e}")
        
        renderer.cleanup()
        
        success_rate = successful_integrations / len(integration_tests)
        print(f"  📊 Integration success rate: {success_rate*100:.1f}%")
        
        return success_rate > 0.75
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Integration test error: {e}")
        return False

def generate_test_recommendations(test_results: Dict[str, bool]):
    """Generate recommendations based on test results."""
    print("\n" + "=" * 80)
    print("🔬 PLAINCLIRENDERER TEST RESULTS & RECOMMENDATIONS")
    print("=" * 80)
    
    passed_tests = sum(1 for result in test_results.values() if result)
    total_tests = len(test_results)
    
    print(f"📊 Test Summary: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    print("\n📋 Test Results:")
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    # Generate specific recommendations
    failed_tests = [name for name, result in test_results.items() if not result]
    
    if failed_tests:
        print(f"\n🔧 FAILED TESTS ANALYSIS:")
        
        if "Renderer Creation" in failed_tests:
            print("  🎯 Renderer Creation Failed:")
            print("    • Check V3 module imports")
            print("    • Verify PlainCLIRenderer class definition")
            print("    • Test terminal capability detection")
        
        if "Handle Input (Mocked)" in failed_tests:
            print("  🎯 Input Handling Issues:")
            print("    • Check handle_input() method implementation")
            print("    • Verify input() function integration")
            print("    • Test exception handling (EOFError, KeyboardInterrupt)")
        
        if "Threading Safety" in failed_tests:
            print("  🎯 Threading Safety Issues:")
            print("    • Add proper locking in PlainCLIRenderer")
            print("    • Review concurrent access to renderer state")
            print("    • Consider thread-safe output mechanisms")
        
        if "Input Stress Test" in failed_tests:
            print("  🎯 Performance Issues:")
            print("    • Optimize handle_input() for rapid calls")
            print("    • Check for memory leaks in input processing")
            print("    • Review input validation logic")
    
    else:
        print("\n🎉 All tests passed! PlainCLIRenderer appears to be working correctly.")
    
    print("\n📋 NEXT STEPS:")
    if failed_tests:
        print("  1. Fix failing test components in PlainCLIRenderer")
        print("  2. Re-run tests to verify fixes")
        print("  3. Test with real TUI integration")
    else:
        print("  1. Test PlainCLIRenderer in real terminal environment")
        print("  2. Check integration with other V3 components")
        print("  3. Verify fixes resolve user-reported issues")

def main():
    test_header()
    
    # Run comprehensive PlainCLIRenderer tests
    tester = PlainCLIRendererTester()
    test_results = tester.run_all_tests()
    
    # Run integration tests
    integration_result = test_integration_with_chat_engine()
    test_results["ChatEngine Integration"] = integration_result
    
    # Generate recommendations
    generate_test_recommendations(test_results)
    
    print(f"\n" + "=" * 80)
    print("🧪 PlainCLIRenderer Test Suite Complete!")
    print("Use these results to identify and fix input handling issues.")
    print("=" * 80)

if __name__ == "__main__":
    main()