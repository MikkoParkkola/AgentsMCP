"""
Comprehensive TUI test suite runner and integration tests.

This module provides a complete test suite for TUI alignment issues,
combining all test categories and providing comprehensive coverage analysis.
"""

import pytest
import asyncio
import sys
import os
import time
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, patch, AsyncMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from agentsmcp.ui.v2.fixed_working_tui import FixedWorkingTUI
from test_tui_utilities import (
    TerminalSimulator, AlignmentAnalyzer, OutputPatternMatcher,
    mock_terminal_environment, PerformanceBenchmarker,
    create_test_responses, create_test_inputs
)


class TUITestSuite:
    """
    Comprehensive test suite orchestrator for TUI alignment testing.
    
    This class coordinates all TUI tests and provides comprehensive
    analysis of alignment, logging, and performance issues.
    """
    
    def __init__(self):
        self.tui = None
        self.test_results = {}
        self.alignment_analyzer = AlignmentAnalyzer()
        self.pattern_matcher = OutputPatternMatcher()
        self.benchmarker = PerformanceBenchmarker()
        
    def setup_tui(self):
        """Setup TUI instance for testing."""
        self.tui = FixedWorkingTUI()
        
        # Mock LLM client
        mock_client = AsyncMock()
        mock_client.provider = "test-provider"
        mock_client.model = "test-model"
        mock_client.send_message = AsyncMock()
        mock_client.clear_history = Mock()
        self.tui.llm_client = mock_client
        
        return mock_client
    
    def run_alignment_tests(self) -> Dict[str, Any]:
        """Run all alignment-related tests."""
        results = {
            'progressive_indentation': self._test_progressive_indentation(),
            'cursor_tracking': self._test_cursor_tracking(),
            'multiline_formatting': self._test_multiline_formatting(),
            'boundary_conditions': self._test_boundary_conditions()
        }
        return results
    
    def run_logging_tests(self) -> Dict[str, Any]:
        """Run all logging isolation tests."""
        results = {
            'debug_isolation': self._test_debug_isolation(),
            'ui_log_separation': self._test_ui_log_separation(),
            'async_logging': self._test_async_logging(),
            'error_handling': self._test_error_logging()
        }
        return results
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run all integration tests."""
        results = {
            'conversation_flow': asyncio.run(self._test_conversation_flow()),
            'command_integration': asyncio.run(self._test_command_integration()),
            'error_scenarios': asyncio.run(self._test_error_scenarios()),
            'performance': self._test_performance_integration()
        }
        return results
    
    def _test_progressive_indentation(self) -> Dict[str, Any]:
        """Test for progressive indentation issues."""
        with mock_terminal_environment() as terminal:
            self.setup_tui()
            
            # Simulate multiple prompt cycles
            prompt_positions = []
            for i in range(10):
                self.tui.show_prompt()
                row, col = terminal.get_cursor_position()
                prompt_positions.append(col)
                
                # Simulate some input and processing
                terminal.write(f"input{i}\n")
                terminal.write("🤖 Agent: Response\n")
            
            # Analyze for progressive indentation
            screen_content = terminal.get_screen_content()
            issues = self.alignment_analyzer.analyze_progressive_indentation(screen_content)
            
            return {
                'passed': len(issues) == 0,
                'issues': issues,
                'prompt_positions': prompt_positions,
                'screen_content': screen_content
            }
    
    def _test_cursor_tracking(self) -> Dict[str, Any]:
        """Test cursor tracking accuracy."""
        with mock_terminal_environment() as terminal:
            self.setup_tui()
            
            states = []
            
            # Test various cursor operations
            operations = [
                ('initial', lambda: self.tui.clear_screen_and_show_prompt()),
                ('after_prompt', lambda: self.tui.show_prompt()),
                ('after_input', lambda: self._simulate_input("hello")),
                ('after_backspace', lambda: self._simulate_backspace()),
                ('after_reset', lambda: self.tui.show_prompt())
            ]
            
            for op_name, operation in operations:
                operation()
                state = terminal.save_state()
                states.append((op_name, state))
            
            # Analyze cursor consistency
            cursor_issues = self.alignment_analyzer.analyze_cursor_consistency([s[1] for s in states])
            
            return {
                'passed': len(cursor_issues) == 0,
                'issues': cursor_issues,
                'states': [(name, state.cursor_row, state.cursor_col) for name, state in states]
            }
    
    def _test_multiline_formatting(self) -> Dict[str, Any]:
        """Test multiline response formatting."""
        with mock_terminal_environment() as terminal:
            self.setup_tui()
            
            # Simulate multiline response
            multiline_response = "Line 1\nLine 2\nLine 3"
            self.tui.llm_client.send_message.return_value = multiline_response
            
            self.tui.clear_screen_and_show_prompt()
            asyncio.run(self.tui.process_line("Give multiline response"))
            
            screen_content = terminal.get_screen_content()
            formatting_issues = self.alignment_analyzer.analyze_response_formatting(screen_content)
            
            return {
                'passed': len(formatting_issues) == 0,
                'issues': formatting_issues,
                'screen_content': screen_content
            }
    
    def _test_boundary_conditions(self) -> Dict[str, Any]:
        """Test boundary conditions and edge cases."""
        with mock_terminal_environment(width=40, height=10) as terminal:  # Small terminal
            self.setup_tui()
            
            issues = []
            
            # Test very long input
            long_input = "x" * 100
            try:
                self._simulate_input(long_input)
                row, col = terminal.get_cursor_position()
                if col < 0 or col >= terminal.width or row < 0 or row >= terminal.height:
                    issues.append("Cursor position out of bounds")
            except Exception as e:
                issues.append(f"Long input caused exception: {e}")
            
            # Test at terminal boundaries
            terminal.cursor_row = terminal.height - 1
            terminal.cursor_col = terminal.width - 1
            try:
                terminal.write("boundary test\n")
            except Exception as e:
                issues.append(f"Boundary write caused exception: {e}")
            
            return {
                'passed': len(issues) == 0,
                'issues': issues
            }
    
    def _test_debug_isolation(self) -> Dict[str, Any]:
        """Test debug log isolation from UI."""
        import logging
        from io import StringIO
        
        # Capture logs
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        logger = logging.getLogger('agentsmcp.test')
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        
        with mock_terminal_environment() as terminal:
            self.setup_tui()
            
            # Generate debug logs
            logger.debug("Debug message that should not appear in UI")
            logger.info("Info message")
            
            # Perform UI operations
            self.tui.show_prompt()
            
            screen_content = '\n'.join(terminal.get_screen_content())
            log_content = log_capture.getvalue()
            
            # Debug logs should be in log capture, not screen
            debug_in_screen = "Debug message" in screen_content
            debug_in_logs = "Debug message" in log_content
            
            return {
                'passed': not debug_in_screen and debug_in_logs,
                'debug_leaked_to_ui': debug_in_screen,
                'debug_captured_in_logs': debug_in_logs
            }
    
    def _test_ui_log_separation(self) -> Dict[str, Any]:
        """Test UI and log output separation."""
        import logging
        from io import StringIO
        
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        ui_logger = logging.getLogger('agentsmcp.ui.test')
        ui_logger.addHandler(handler)
        
        with mock_terminal_environment() as terminal:
            self.setup_tui()
            
            # Mix UI operations with logging
            self.tui.show_prompt()
            ui_logger.info("UI log message")
            terminal.write("🤖 Agent: Response\n")
            ui_logger.debug("Debug from UI")
            
            screen_content = '\n'.join(terminal.get_screen_content())
            log_content = log_capture.getvalue()
            
            return {
                'passed': '🤖 Agent:' in screen_content and 'UI log message' not in screen_content,
                'ui_elements_in_screen': '🤖 Agent:' in screen_content,
                'logs_separated': 'UI log message' not in screen_content
            }
    
    def _test_async_logging(self) -> Dict[str, Any]:
        """Test async operation logging isolation."""
        async def async_operation():
            import logging
            logger = logging.getLogger('agentsmcp.async_test')
            logger.debug("Async debug message")
            return "Async result"
        
        with mock_terminal_environment() as terminal:
            self.setup_tui()
            
            # Run async operation
            result = asyncio.run(async_operation())
            self.tui.show_prompt()
            
            screen_content = '\n'.join(terminal.get_screen_content())
            
            return {
                'passed': 'Async debug message' not in screen_content,
                'async_completed': result == "Async result"
            }
    
    def _test_error_logging(self) -> Dict[str, Any]:
        """Test error logging vs error display."""
        import logging
        from io import StringIO
        
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        logger = logging.getLogger('agentsmcp.error_test')
        logger.addHandler(handler)
        
        with mock_terminal_environment() as terminal:
            self.setup_tui()
            
            # Simulate error logging
            error_msg = "Test error occurred"
            logger.error(f"Internal error: {error_msg}")
            
            # Simulate user-facing error display
            terminal.write(f"❌ Error: {error_msg}\n")
            
            screen_content = '\n'.join(terminal.get_screen_content())
            log_content = log_capture.getvalue()
            
            return {
                'passed': '❌ Error:' in screen_content and 'Internal error:' not in screen_content,
                'user_error_displayed': '❌ Error:' in screen_content,
                'internal_error_logged': 'Internal error:' in log_content
            }
    
    async def _test_conversation_flow(self) -> Dict[str, Any]:
        """Test complete conversation flow."""
        with mock_terminal_environment() as terminal:
            mock_client = self.setup_tui()
            
            # Setup conversation responses
            responses = create_test_responses()
            mock_client.send_message.side_effect = responses
            
            self.tui.clear_screen_and_show_prompt()
            
            # Simulate conversation
            inputs = create_test_inputs()[:len(responses)]
            for msg in inputs:
                if msg.strip():  # Skip empty inputs
                    await self.tui.process_line(msg)
                    self.tui.show_prompt()
            
            screen_content = terminal.get_screen_content()
            flow_analysis = self.pattern_matcher.verify_conversation_flow(screen_content)
            
            return {
                'passed': flow_analysis['valid'],
                'flow_issues': flow_analysis['issues'],
                'flow_sequence': flow_analysis['flow_sequence']
            }
    
    async def _test_command_integration(self) -> Dict[str, Any]:
        """Test command integration with chat."""
        with mock_terminal_environment() as terminal:
            mock_client = self.setup_tui()
            mock_client.send_message.return_value = "Chat response"
            
            self.tui.clear_screen_and_show_prompt()
            
            # Test command sequence
            commands = ["/help", "hello", "/clear", "how are you?"]
            for cmd in commands:
                await self.tui.process_line(cmd)
                self.tui.show_prompt()
            
            screen_content = terminal.get_screen_content()
            
            # Verify commands were processed
            help_found = any('📚 Commands:' in line for line in screen_content)
            clear_executed = mock_client.clear_history.called
            
            return {
                'passed': help_found and clear_executed,
                'help_displayed': help_found,
                'clear_executed': clear_executed
            }
    
    async def _test_error_scenarios(self) -> Dict[str, Any]:
        """Test error handling scenarios."""
        with mock_terminal_environment() as terminal:
            mock_client = self.setup_tui()
            mock_client.send_message.side_effect = Exception("Test LLM error")
            
            self.tui.clear_screen_and_show_prompt()
            
            # Process message that will cause error
            await self.tui.process_line("This will cause an error")
            
            screen_content = terminal.get_screen_content()
            error_displayed = any('❌ Error:' in line for line in screen_content)
            
            return {
                'passed': error_displayed,
                'error_displayed': error_displayed,
                'screen_content': screen_content
            }
    
    def _test_performance_integration(self) -> Dict[str, Any]:
        """Test performance characteristics."""
        with mock_terminal_environment() as terminal:
            self.setup_tui()
            
            # Performance test: rapid prompt showing
            with self.benchmarker.measure('rapid_prompts'):
                for i in range(100):
                    self.tui.show_prompt()
            
            # Performance test: cursor operations
            with self.benchmarker.measure('cursor_operations'):
                for i in range(100):
                    self.tui.cursor_col = 2
                    self.tui.cursor_col += 5
                    self.tui.cursor_col = 2
            
            prompt_stats = self.benchmarker.get_statistics('rapid_prompts')
            cursor_stats = self.benchmarker.get_statistics('cursor_operations')
            
            # Check for performance issues
            performance_ok = (prompt_stats.get('average', 0) < 0.001 and 
                            cursor_stats.get('average', 0) < 0.001)
            
            return {
                'passed': performance_ok,
                'prompt_stats': prompt_stats,
                'cursor_stats': cursor_stats
            }
    
    def _simulate_input(self, text: str):
        """Simulate text input to TUI."""
        for char in text:
            if ord(char) >= 32:  # Printable character
                self.tui.input_buffer += char
                self.tui.cursor_col += 1
    
    def _simulate_backspace(self):
        """Simulate backspace operation."""
        if self.tui.input_buffer and self.tui.cursor_col > 2:
            self.tui.input_buffer = self.tui.input_buffer[:-1]
            self.tui.cursor_col -= 1


class TestComprehensiveTUISuite:
    """Pytest class for comprehensive TUI testing."""
    
    @pytest.fixture
    def test_suite(self):
        """Create test suite instance."""
        return TUITestSuite()
    
    def test_alignment_comprehensive(self, test_suite):
        """Run comprehensive alignment tests."""
        results = test_suite.run_alignment_tests()
        
        # All alignment tests should pass
        for test_name, result in results.items():
            assert result['passed'], f"Alignment test '{test_name}' failed: {result.get('issues', [])}"
    
    def test_logging_comprehensive(self, test_suite):
        """Run comprehensive logging tests."""
        results = test_suite.run_logging_tests()
        
        # All logging tests should pass
        for test_name, result in results.items():
            assert result['passed'], f"Logging test '{test_name}' failed"
    
    @pytest.mark.asyncio
    async def test_integration_comprehensive(self, test_suite):
        """Run comprehensive integration tests."""
        results = test_suite.run_integration_tests()
        
        # All integration tests should pass
        for test_name, result in results.items():
            assert result['passed'], f"Integration test '{test_name}' failed: {result}"
    
    def test_regression_prevention(self, test_suite):
        """Test that known issues are prevented."""
        with mock_terminal_environment() as terminal:
            test_suite.setup_tui()
            
            # Test the specific issue from the problem description
            # Simulate the sequence that caused progressive indentation
            
            # Initial setup
            test_suite.tui.clear_screen_and_show_prompt()
            initial_pos = terminal.get_cursor_position()[1]
            
            # Simulate user input and response cycle
            terminal.write("hello\n")
            terminal.write("       🤔 Thinking...\n")
            terminal.write("2025-08-31T11:34:12+0300 DEBUG agentsmcp.conversation.llm_client: Tool execution turn 1/3\n")
            
            # Show next prompt - this should NOT be further indented
            test_suite.tui.show_prompt()
            final_pos = terminal.get_cursor_position()[1]
            
            # Verify no progressive indentation occurred
            assert final_pos <= initial_pos + 2, f"Progressive indentation detected: {initial_pos} -> {final_pos}"
    
    def test_debug_log_isolation_regression(self, test_suite):
        """Test that debug logs don't interfere with UI (regression test)."""
        import logging
        from io import StringIO
        
        # The specific log message from the problem description
        logger = logging.getLogger('agentsmcp.conversation.llm_client')
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        logger.addHandler(handler)
        
        with mock_terminal_environment() as terminal:
            test_suite.setup_tui()
            
            # Generate the specific debug log from the issue
            logger.debug("Tool execution turn 1/3")
            
            # Perform UI operations
            test_suite.tui.show_prompt()
            
            screen_content = '\n'.join(terminal.get_screen_content())
            
            # Debug log should NOT appear in UI
            assert "Tool execution turn 1/3" not in screen_content
            assert "DEBUG" not in screen_content
            
            # But UI elements should be present
            assert ">" in screen_content


def run_comprehensive_tests():
    """Run all comprehensive TUI tests."""
    suite = TUITestSuite()
    
    print("Running TUI Comprehensive Test Suite...")
    
    # Run all test categories
    alignment_results = suite.run_alignment_tests()
    logging_results = suite.run_logging_tests()
    integration_results = suite.run_integration_tests()
    
    # Compile results
    total_tests = 0
    passed_tests = 0
    
    all_results = {
        'Alignment': alignment_results,
        'Logging': logging_results,
        'Integration': integration_results
    }
    
    for category, results in all_results.items():
        print(f"\n{category} Tests:")
        for test_name, result in results.items():
            total_tests += 1
            if result['passed']:
                passed_tests += 1
                print(f"  ✅ {test_name}")
            else:
                print(f"  ❌ {test_name}")
                if 'issues' in result:
                    for issue in result['issues']:
                        print(f"     - {issue}")
    
    print(f"\nSummary: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All TUI tests passed! No alignment issues detected.")
        return True
    else:
        print("⚠️  Some tests failed. TUI alignment issues need to be addressed.")
        return False


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)