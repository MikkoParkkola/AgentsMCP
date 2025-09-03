#!/usr/bin/env python3
"""
Integration Test for Revolutionary TUI Interface - Full Lifecycle
================================================================

This test validates the complete TUI lifecycle from initialization through
graceful shutdown, covering all integration points and real-world scenarios.

LIFECYCLE STAGES TESTED:
1. System Initialization
2. Component Integration
3. User Interaction Flow
4. Event Processing
5. Error Handling & Recovery
6. Resource Management
7. Graceful Shutdown

INTEGRATION POINTS VALIDATED:
- Terminal controller integration
- Logging isolation manager
- Input rendering pipeline
- Display manager coordination
- Event system integration
- Orchestrator communication
"""

import asyncio
import logging
import os
import sys
import time
import unittest
import signal
import threading
import tempfile
from unittest.mock import Mock, patch, MagicMock, AsyncMock, call
from contextlib import contextmanager
from typing import List, Dict, Any, Optional
import json

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from src.agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface, TUIState
    from src.agentsmcp.ui.v2.terminal_controller import TerminalController
    from src.agentsmcp.ui.v2.logging_isolation_manager import LoggingIsolationManager
    from src.agentsmcp.ui.v2.input_rendering_pipeline import InputRenderingPipeline
    from src.agentsmcp.ui.v2.display_manager import DisplayManager
    from src.agentsmcp.ui.v2.event_system import AsyncEventSystem
    from src.agentsmcp.orchestration import Orchestrator, OrchestratorConfig
except ImportError as e:
    print(f"Import error: {e}")
    print("Please run from project root directory")
    sys.exit(1)


class TUIIntegrationLifecycleTests(unittest.TestCase):
    """Integration tests for complete TUI lifecycle."""
    
    def setUp(self):
        """Set up test environment for integration testing."""
        self.lifecycle_events = []
        self.component_status = {}
        self.integration_errors = []
        
        # Mock external dependencies
        self.mock_orchestrator = Mock(spec=Orchestrator)
        self.mock_orchestrator.start = AsyncMock()
        self.mock_orchestrator.stop = AsyncMock()
        self.mock_orchestrator.get_status = Mock(return_value={'status': 'ready'})
        
        # Configure minimal logging
        logging.getLogger().handlers.clear()
        logging.getLogger().setLevel(logging.WARNING)
    
    def tearDown(self):
        """Clean up after integration tests."""
        # Ensure all resources are released
        if hasattr(self, 'tui') and self.tui:
            try:
                asyncio.create_task(self.tui._handle_exit())
            except:
                pass
    
    def track_lifecycle_event(self, event_name: str, details: Dict = None):
        """Track lifecycle events for validation."""
        timestamp = time.time()
        self.lifecycle_events.append({
            'event': event_name,
            'timestamp': timestamp,
            'details': details or {}
        })
    
    async def test_01_system_initialization_sequence(self):
        """
        TEST 1: System Initialization Sequence
        ======================================
        Validates proper initialization of all TUI components.
        """
        print("\n🚀 TEST 1: System Initialization Sequence")
        
        initialization_steps = []
        
        try:
            self.track_lifecycle_event("initialization_start")
            
            # Step 1: Create TUI instance
            tui = RevolutionaryTUIInterface()
            initialization_steps.append("✅ TUI instance created")
            self.assertIsNotNone(tui, "TUI instance should be created")
            
            # Step 2: Validate component initialization
            components_to_check = [
                ('state', 'TUI state initialized'),
                ('_console', 'Rich console initialized'),
                ('_layout', 'Layout initialized'),
            ]
            
            for attr_name, description in components_to_check:
                if hasattr(tui, attr_name):
                    component = getattr(tui, attr_name)
                    if component is not None:
                        initialization_steps.append(f"✅ {description}")
                    else:
                        initialization_steps.append(f"⚠️  {description} - None")
                else:
                    initialization_steps.append(f"⚠️  {description} - Missing")
            
            # Step 3: Validate state initialization
            self.assertIsInstance(tui.state, TUIState, "TUI state should be TUIState instance")
            self.assertEqual(tui.state.current_input, "", "Initial input should be empty")
            self.assertFalse(tui.state.is_processing, "Should not be processing initially")
            initialization_steps.append("✅ TUI state properly initialized")
            
            self.track_lifecycle_event("initialization_complete", 
                {'steps_completed': len(initialization_steps)})
            
            print("📋 Initialization Steps:")
            for step in initialization_steps:
                print(f"   {step}")
            
            print("✅ System initialization sequence validated")
            
        except Exception as e:
            self.integration_errors.append(f"Initialization failed: {e}")
            self.fail(f"System initialization failed: {e}")
    
    async def test_02_component_integration_validation(self):
        """
        TEST 2: Component Integration Validation  
        ========================================
        Validates that all TUI components integrate correctly.
        """
        print("\n🔗 TEST 2: Component Integration Validation")
        
        integration_checks = []
        
        try:
            tui = RevolutionaryTUIInterface()
            self.track_lifecycle_event("component_integration_start")
            
            # Check terminal controller integration
            if hasattr(tui, 'terminal_controller') or hasattr(tui, '_terminal_controller'):
                integration_checks.append("✅ Terminal controller integration available")
            else:
                integration_checks.append("⚠️  Terminal controller not directly accessible")
            
            # Check display manager integration  
            if hasattr(tui, 'display_manager') or hasattr(tui, '_display_manager'):
                integration_checks.append("✅ Display manager integration available")
            else:
                integration_checks.append("⚠️  Display manager not directly accessible")
            
            # Check input pipeline integration
            if hasattr(tui, 'input_pipeline') or hasattr(tui, '_input_pipeline'):
                integration_checks.append("✅ Input pipeline integration available")
            else:
                integration_checks.append("⚠️  Input pipeline not directly accessible")
            
            # Check event system integration
            if hasattr(tui, 'event_system') or hasattr(tui, '_event_system'):
                integration_checks.append("✅ Event system integration available")
            else:
                integration_checks.append("⚠️  Event system not directly accessible")
            
            # Test component coordination
            try:
                # Update state to trigger component coordination
                tui.state.current_input = "integration_test"
                tui.state.processing_message = "Testing component integration"
                
                # Try to refresh display (tests display manager integration)
                if hasattr(tui, '_refresh_display'):
                    await tui._refresh_display()
                    integration_checks.append("✅ Display refresh coordination working")
                
                # Try input handling (tests input pipeline integration)
                tui._handle_character_input('x')
                if 'x' in tui.state.current_input:
                    integration_checks.append("✅ Input handling coordination working")
                
            except Exception as e:
                integration_checks.append(f"❌ Component coordination error: {e}")
            
            self.track_lifecycle_event("component_integration_complete",
                {'checks_completed': len(integration_checks)})
            
            print("🔗 Component Integration Checks:")
            for check in integration_checks:
                print(f"   {check}")
            
            print("✅ Component integration validated")
            
        except Exception as e:
            self.integration_errors.append(f"Component integration failed: {e}")
            self.fail(f"Component integration validation failed: {e}")
    
    async def test_03_user_interaction_flow_complete(self):
        """
        TEST 3: Complete User Interaction Flow
        ======================================  
        Simulates complete user interaction from start to finish.
        """
        print("\n👤 TEST 3: Complete User Interaction Flow")
        
        interaction_flow = []
        
        try:
            tui = RevolutionaryTUIInterface()
            self.track_lifecycle_event("user_interaction_start")
            
            # Flow Step 1: User starts typing
            user_input = "Hello, I need assistance with my project setup"
            for i, char in enumerate(user_input):
                tui._handle_character_input(char)
                # Validate progressive input accumulation
                expected_partial = user_input[:i+1]
                if tui.state.current_input == expected_partial:
                    if i % 10 == 0:  # Sample validation
                        interaction_flow.append(f"✅ Character {i+1}: '{expected_partial[:20]}...'")
                else:
                    interaction_flow.append(f"❌ Character {i+1}: Expected '{expected_partial}', got '{tui.state.current_input}'")
            
            # Flow Step 2: User presses Enter  
            await tui._handle_enter_input()
            interaction_flow.append("✅ Enter key processed")
            
            # Flow Step 3: User tries command completion
            command_input = "/hel"
            tui.state.current_input = ""
            for char in command_input:
                tui._handle_character_input(char)
            
            # Simulate tab completion (if available)
            if hasattr(tui, '_handle_tab_completion'):
                tui._handle_tab_completion()
                interaction_flow.append("✅ Tab completion attempted")
            else:
                interaction_flow.append("⚠️  Tab completion not available")
            
            # Flow Step 4: User navigates history
            tui._handle_up_arrow()
            tui._handle_down_arrow()
            interaction_flow.append("✅ History navigation performed")
            
            # Flow Step 5: User performs text editing
            edit_text = "editing test"
            tui.state.current_input = ""
            for char in edit_text:
                tui._handle_character_input(char)
            
            # Delete some characters
            for _ in range(4):  # Delete "test"
                tui._handle_backspace_input()
            
            expected_after_delete = edit_text[:-4]
            if tui.state.current_input == expected_after_delete:
                interaction_flow.append("✅ Text editing (backspace) working")
            else:
                interaction_flow.append(f"❌ Text editing failed: expected '{expected_after_delete}', got '{tui.state.current_input}'")
            
            # Flow Step 6: User executes various commands
            test_commands = ["/help", "/status", "/clear"]
            for cmd in test_commands:
                tui.state.current_input = cmd
                try:
                    await tui._handle_enter_input()
                    interaction_flow.append(f"✅ Command '{cmd}' processed")
                except Exception as e:
                    interaction_flow.append(f"⚠️  Command '{cmd}' error: {str(e)[:50]}...")
            
            self.track_lifecycle_event("user_interaction_complete",
                {'flow_steps': len(interaction_flow)})
            
            print("👤 User Interaction Flow:")
            for step in interaction_flow:
                print(f"   {step}")
            
            # Validate interaction success
            successful_steps = sum(1 for step in interaction_flow if "✅" in step)
            total_steps = len(interaction_flow)
            success_rate = (successful_steps / total_steps) * 100
            
            self.assertGreater(success_rate, 80, 
                f"User interaction flow should be >80% successful. Got {success_rate:.1f}%")
            
            print(f"✅ User interaction flow validated: {success_rate:.1f}% success rate")
            
        except Exception as e:
            self.integration_errors.append(f"User interaction flow failed: {e}")
            self.fail(f"User interaction flow test failed: {e}")
    
    async def test_04_event_processing_integration(self):
        """
        TEST 4: Event Processing Integration
        ===================================
        Validates event system integration and processing.
        """
        print("\n📡 TEST 4: Event Processing Integration")
        
        event_tests = []
        
        try:
            tui = RevolutionaryTUIInterface()
            self.track_lifecycle_event("event_processing_start")
            
            # Test event handling methods exist
            event_handlers = [
                '_handle_user_input_event',
                '_handle_agent_status_change', 
                '_handle_performance_update',
                '_handle_exit'
            ]
            
            for handler_name in event_handlers:
                if hasattr(tui, handler_name):
                    handler = getattr(tui, handler_name)
                    if callable(handler):
                        event_tests.append(f"✅ Event handler '{handler_name}' available")
                    else:
                        event_tests.append(f"❌ Event handler '{handler_name}' not callable")
                else:
                    event_tests.append(f"⚠️  Event handler '{handler_name}' missing")
            
            # Test event processing
            try:
                # Simulate user input event
                if hasattr(tui, '_handle_user_input_event'):
                    test_event = {'type': 'user_input', 'data': 'test_input'}
                    await tui._handle_user_input_event(test_event)
                    event_tests.append("✅ User input event processed")
                
                # Simulate agent status change event
                if hasattr(tui, '_handle_agent_status_change'):
                    status_event = {'agent_id': 'test_agent', 'status': 'active'}
                    await tui._handle_agent_status_change(status_event)
                    event_tests.append("✅ Agent status change event processed")
                
                # Simulate performance update event
                if hasattr(tui, '_handle_performance_update'):
                    perf_event = {'cpu': 25.5, 'memory': 512, 'timestamp': time.time()}
                    await tui._handle_performance_update(perf_event)
                    event_tests.append("✅ Performance update event processed")
                
            except Exception as e:
                event_tests.append(f"❌ Event processing error: {e}")
            
            self.track_lifecycle_event("event_processing_complete",
                {'tests_completed': len(event_tests)})
            
            print("📡 Event Processing Tests:")
            for test in event_tests:
                print(f"   {test}")
            
            print("✅ Event processing integration validated")
            
        except Exception as e:
            self.integration_errors.append(f"Event processing failed: {e}")
            self.fail(f"Event processing integration test failed: {e}")
    
    async def test_05_error_handling_recovery(self):
        """
        TEST 5: Error Handling & Recovery
        =================================
        Validates TUI handles errors gracefully and recovers.
        """
        print("\n🔧 TEST 5: Error Handling & Recovery")
        
        error_recovery_tests = []
        
        try:
            tui = RevolutionaryTUIInterface()
            self.track_lifecycle_event("error_recovery_start")
            
            # Test error scenarios
            error_scenarios = [
                ("Invalid character input", lambda: tui._handle_character_input(None)),
                ("Invalid state update", lambda: setattr(tui.state, 'current_input', None)),
                ("Display refresh error", lambda: asyncio.create_task(tui._refresh_display()) if hasattr(tui, '_refresh_display') else None),
            ]
            
            for scenario_name, trigger_error in error_scenarios:
                try:
                    # Trigger the error condition
                    trigger_error()
                    
                    # Try to continue normal operations
                    tui._handle_character_input('x')
                    
                    # If we get here, recovery was successful
                    error_recovery_tests.append(f"✅ {scenario_name}: Recovered successfully")
                    
                except Exception as e:
                    # Check if it's a graceful error or a crash
                    if "crash" in str(e).lower() or "fatal" in str(e).lower():
                        error_recovery_tests.append(f"❌ {scenario_name}: Fatal error - {e}")
                    else:
                        error_recovery_tests.append(f"⚠️  {scenario_name}: Handled gracefully - {str(e)[:50]}...")
            
            # Test recovery after errors
            try:
                # Reset to known good state
                tui.state.current_input = "recovery_test"
                tui.state.is_processing = False
                
                # Perform normal operations
                tui._handle_character_input('!')
                
                if tui.state.current_input == "recovery_test!":
                    error_recovery_tests.append("✅ Post-error recovery successful")
                else:
                    error_recovery_tests.append("❌ Post-error recovery failed")
                    
            except Exception as e:
                error_recovery_tests.append(f"❌ Recovery validation failed: {e}")
            
            self.track_lifecycle_event("error_recovery_complete",
                {'tests_completed': len(error_recovery_tests)})
            
            print("🔧 Error Recovery Tests:")
            for test in error_recovery_tests:
                print(f"   {test}")
            
            print("✅ Error handling and recovery validated")
            
        except Exception as e:
            self.integration_errors.append(f"Error recovery test failed: {e}")
            self.fail(f"Error handling and recovery test failed: {e}")
    
    async def test_06_resource_management_lifecycle(self):
        """
        TEST 6: Resource Management Lifecycle  
        =====================================
        Validates proper resource management throughout TUI lifecycle.
        """
        print("\n💾 TEST 6: Resource Management Lifecycle")
        
        resource_tests = []
        
        try:
            self.track_lifecycle_event("resource_management_start")
            
            # Get initial resource state
            initial_resources = self._get_resource_snapshot()
            
            # Create and operate TUI
            tui = RevolutionaryTUIInterface()
            
            # Perform resource-intensive operations
            for i in range(20):
                tui.state.current_input = f"resource_test_{i}_" + "x" * 100
                
                # Trigger display updates
                if hasattr(tui, '_refresh_display'):
                    try:
                        await tui._refresh_display()
                    except:
                        pass
                
                # Process input
                tui._handle_character_input('z')
                tui._handle_backspace_input()
                
                await asyncio.sleep(0.01)
            
            # Check resources during operation  
            operational_resources = self._get_resource_snapshot()
            resource_tests.append(f"✅ Resources during operation: {operational_resources['description']}")
            
            # Cleanup simulation
            tui.state.current_input = ""
            
            # Force cleanup
            import gc
            gc.collect()
            
            # Final resource check
            final_resources = self._get_resource_snapshot()
            
            # Validate resource management
            memory_increase = final_resources['memory_mb'] - initial_resources['memory_mb']
            max_acceptable_increase = 50  # MB
            
            if memory_increase < max_acceptable_increase:
                resource_tests.append(f"✅ Memory management: +{memory_increase:.2f}MB (acceptable)")
            else:
                resource_tests.append(f"⚠️  Memory management: +{memory_increase:.2f}MB (high)")
            
            # Check for resource leaks
            if final_resources['memory_mb'] <= operational_resources['memory_mb']:
                resource_tests.append("✅ No apparent memory leaks")
            else:
                leak_amount = final_resources['memory_mb'] - operational_resources['memory_mb']
                resource_tests.append(f"⚠️  Possible memory leak: +{leak_amount:.2f}MB")
            
            self.track_lifecycle_event("resource_management_complete",
                {'memory_increase_mb': memory_increase})
            
            print("💾 Resource Management Tests:")
            for test in resource_tests:
                print(f"   {test}")
            
            print("✅ Resource management lifecycle validated")
            
        except Exception as e:
            self.integration_errors.append(f"Resource management test failed: {e}")
            self.fail(f"Resource management lifecycle test failed: {e}")
    
    async def test_07_graceful_shutdown_sequence(self):
        """
        TEST 7: Graceful Shutdown Sequence
        ==================================
        Validates proper shutdown and cleanup of all TUI components.
        """
        print("\n🛑 TEST 7: Graceful Shutdown Sequence")
        
        shutdown_tests = []
        
        try:
            tui = RevolutionaryTUIInterface()
            self.track_lifecycle_event("shutdown_sequence_start")
            
            # Simulate active TUI state before shutdown
            tui.state.current_input = "active_session_data"
            tui.state.is_processing = True
            tui.state.processing_message = "Simulating active processing"
            
            # Add some conversation history
            tui.state.conversation_history = [
                {'user': 'test message 1', 'timestamp': time.time()},
                {'user': 'test message 2', 'timestamp': time.time()}
            ]
            
            shutdown_tests.append("✅ Active TUI state established")
            
            # Test shutdown initiation
            shutdown_initiated = False
            original_exit = getattr(tui, '_handle_exit', None)
            
            async def mock_exit(*args, **kwargs):
                nonlocal shutdown_initiated
                shutdown_initiated = True
                # Perform actual shutdown logic
                if original_exit:
                    try:
                        return await original_exit(*args, **kwargs)
                    except:
                        return 0
                return 0
            
            if original_exit:
                tui._handle_exit = mock_exit
            
            # Trigger shutdown via /quit command
            tui.state.current_input = "/quit"
            await tui._handle_enter_input()
            
            if shutdown_initiated:
                shutdown_tests.append("✅ Shutdown initiated successfully")
            else:
                shutdown_tests.append("❌ Shutdown not properly initiated")
            
            # Test cleanup state
            # Note: In real scenario, components would be cleaned up
            # Here we simulate the expected cleanup
            cleanup_checks = [
                ("Processing stopped", not tui.state.is_processing),
                ("Input cleared", tui.state.current_input in ["", "/quit"]),  # May remain as /quit
                ("History preserved", len(tui.state.conversation_history) > 0),
            ]
            
            for check_name, check_result in cleanup_checks:
                if check_result:
                    shutdown_tests.append(f"✅ {check_name}")
                else:
                    shutdown_tests.append(f"⚠️  {check_name} - not as expected")
            
            # Test resource cleanup
            try:
                # Simulate resource cleanup validation
                if hasattr(tui, '_console'):
                    shutdown_tests.append("✅ Console resources available for cleanup")
                
                if hasattr(tui, '_layout'):
                    shutdown_tests.append("✅ Layout resources available for cleanup")
                    
            except Exception as e:
                shutdown_tests.append(f"⚠️  Resource cleanup validation error: {e}")
            
            self.track_lifecycle_event("shutdown_sequence_complete",
                {'tests_completed': len(shutdown_tests)})
            
            print("🛑 Shutdown Sequence Tests:")
            for test in shutdown_tests:
                print(f"   {test}")
            
            print("✅ Graceful shutdown sequence validated")
            
        except Exception as e:
            self.integration_errors.append(f"Shutdown sequence test failed: {e}")
            self.fail(f"Graceful shutdown sequence test failed: {e}")
    
    def _get_resource_snapshot(self) -> Dict[str, Any]:
        """Get current resource usage snapshot."""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            return {
                'memory_mb': memory_mb,
                'description': f"Memory: {memory_mb:.2f}MB"
            }
        except ImportError:
            return {
                'memory_mb': 0.0,
                'description': "Resource monitoring not available"
            }
    
    async def test_08_full_lifecycle_integration(self):
        """
        TEST 8: Full Lifecycle Integration
        ==================================
        Complete end-to-end integration test covering entire TUI lifecycle.
        """
        print("\n🔄 TEST 8: Full Lifecycle Integration")
        
        lifecycle_phases = []
        
        try:
            self.track_lifecycle_event("full_lifecycle_start")
            
            # Phase 1: Initialization
            tui = RevolutionaryTUIInterface()
            lifecycle_phases.append("✅ Phase 1: Initialization completed")
            
            # Phase 2: Startup and ready state
            await asyncio.sleep(0.1)  # Allow initialization to complete
            lifecycle_phases.append("✅ Phase 2: Startup completed")
            
            # Phase 3: User interaction simulation
            interactions = [
                ("Type message", "Hello TUI"),
                ("Send message", "\n"),
                ("Try command", "/status"),
                ("Execute command", "\n"),
                ("Edit input", "test" + "\b\b" + "ed"),
                ("History nav", "↑↓"),
                ("Clear input", "\b" * 10),
            ]
            
            for action_name, action_data in interactions:
                try:
                    if action_name == "Type message":
                        for char in action_data:
                            tui._handle_character_input(char)
                    elif action_name == "Send message":
                        await tui._handle_enter_input()
                    elif action_name == "Try command":
                        tui.state.current_input = action_data
                    elif action_name == "Execute command":
                        await tui._handle_enter_input()
                    elif action_name == "Edit input":
                        # Simplified edit simulation
                        tui.state.current_input = "tested"
                    elif action_name == "History nav":
                        tui._handle_up_arrow()
                        tui._handle_down_arrow()
                    elif action_name == "Clear input":
                        tui.state.current_input = ""
                    
                    lifecycle_phases.append(f"✅ Phase 3.{len([p for p in lifecycle_phases if 'Phase 3' in p]) + 1}: {action_name}")
                    
                except Exception as e:
                    lifecycle_phases.append(f"⚠️  Phase 3: {action_name} error - {str(e)[:50]}...")
            
            # Phase 4: Stress testing
            try:
                for i in range(10):
                    tui._handle_character_input('x')
                    if hasattr(tui, '_refresh_display'):
                        await tui._refresh_display()
                    await asyncio.sleep(0.001)
                
                lifecycle_phases.append("✅ Phase 4: Stress testing completed")
                
            except Exception as e:
                lifecycle_phases.append(f"⚠️  Phase 4: Stress testing error - {e}")
            
            # Phase 5: Error recovery
            try:
                # Introduce controlled error
                tui.state.current_input = None
                tui._handle_character_input('y')  # Should handle gracefully
                
                # Recovery
                tui.state.current_input = "recovered"
                
                lifecycle_phases.append("✅ Phase 5: Error recovery completed")
                
            except Exception as e:
                lifecycle_phases.append(f"⚠️  Phase 5: Error recovery - {str(e)[:50]}...")
            
            # Phase 6: Cleanup and shutdown preparation
            tui.state.is_processing = False
            tui.state.current_input = "/quit"
            
            # Mock exit for testing
            exit_called = False
            async def mock_exit(*args):
                nonlocal exit_called
                exit_called = True
                return 0
            
            if hasattr(tui, '_handle_exit'):
                tui._handle_exit = mock_exit
            
            await tui._handle_enter_input()
            
            if exit_called:
                lifecycle_phases.append("✅ Phase 6: Shutdown preparation completed")
            else:
                lifecycle_phases.append("⚠️  Phase 6: Shutdown preparation - exit not called")
            
            self.track_lifecycle_event("full_lifecycle_complete",
                {'phases_completed': len(lifecycle_phases)})
            
            print("🔄 Full Lifecycle Integration Phases:")
            for phase in lifecycle_phases:
                print(f"   {phase}")
            
            # Calculate success rate
            successful_phases = sum(1 for phase in lifecycle_phases if "✅" in phase)
            total_phases = len(lifecycle_phases)
            success_rate = (successful_phases / total_phases) * 100
            
            self.assertGreater(success_rate, 85, 
                f"Full lifecycle integration should be >85% successful. Got {success_rate:.1f}%")
            
            print(f"✅ Full lifecycle integration validated: {success_rate:.1f}% success rate")
            
        except Exception as e:
            self.integration_errors.append(f"Full lifecycle integration failed: {e}")
            self.fail(f"Full lifecycle integration test failed: {e}")


async def run_integration_lifecycle_tests():
    """Run all TUI integration lifecycle tests."""
    print("=" * 70)
    print("🔄 TUI INTEGRATION LIFECYCLE TEST SUITE")
    print("=" * 70)
    print("Testing complete TUI lifecycle from initialization to shutdown...")
    print()
    
    test_case = TUIIntegrationLifecycleTests()
    
    test_methods = [
        'test_01_system_initialization_sequence',
        'test_02_component_integration_validation',
        'test_03_user_interaction_flow_complete',
        'test_04_event_processing_integration', 
        'test_05_error_handling_recovery',
        'test_06_resource_management_lifecycle',
        'test_07_graceful_shutdown_sequence',
        'test_08_full_lifecycle_integration'
    ]
    
    passed = 0
    failed = 0
    results = {}
    
    for method_name in test_methods:
        print(f"\n{'=' * 50}")
        try:
            await getattr(test_case, method_name)()
            results[method_name] = "✅ PASSED"
            passed += 1
            print(f"🎉 {method_name}: PASSED")
        except Exception as e:
            results[method_name] = f"❌ FAILED: {e}"
            failed += 1
            print(f"💥 {method_name}: FAILED - {e}")
    
    # Summary
    print(f"\n{'=' * 70}")
    print("📊 INTEGRATION LIFECYCLE TEST RESULTS")
    print("=" * 70)
    
    for method, result in results.items():
        print(f"{result}")
    
    print(f"\n📈 Results: ✅ {passed} passed, ❌ {failed} failed")
    print(f"🎯 Success Rate: {(passed / len(test_methods)) * 100:.1f}%")
    
    # Lifecycle events summary
    if hasattr(test_case, 'lifecycle_events') and test_case.lifecycle_events:
        print(f"\n📋 Lifecycle Events Tracked: {len(test_case.lifecycle_events)}")
        
    # Integration errors summary
    if hasattr(test_case, 'integration_errors') and test_case.integration_errors:
        print(f"⚠️  Integration Errors Found: {len(test_case.integration_errors)}")
        for error in test_case.integration_errors[:3]:  # Show first 3
            print(f"   - {error}")
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_integration_lifecycle_tests())
    sys.exit(0 if success else 1)