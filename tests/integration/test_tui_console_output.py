"""
Integration tests for TUI console output alignment and behavior.

These tests verify complete conversation flows work correctly with proper
alignment and no progressive indentation issues.
"""

import pytest
import asyncio
import io
import sys
import os
import time
from unittest.mock import Mock, patch, MagicMock, AsyncMock, call
from contextlib import contextmanager
import tempfile
import threading

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from agentsmcp.ui.v2.fixed_working_tui import FixedWorkingTUI


class ConsoleOutputCapture:
    """Capture and analyze console output patterns."""
    
    def __init__(self):
        self.output_lines = []
        self.cursor_positions = []
        self.timestamps = []
        
    def write(self, text):
        """Capture text with timestamp and analyze positioning."""
        self.timestamps.append(time.time())
        
        # Split by lines but preserve original structure
        if '\n' in text:
            lines = text.split('\n')
            for i, line in enumerate(lines[:-1]):  # All but last
                self.output_lines.append(line)
            if lines[-1]:  # If last line is not empty
                self.output_lines.append(lines[-1])
        else:
            if text:
                if self.output_lines:
                    self.output_lines[-1] += text
                else:
                    self.output_lines.append(text)
    
    def flush(self):
        """Mock flush method."""
        pass
        
    def get_alignment_issues(self):
        """Analyze output for alignment issues."""
        issues = []
        
        # Check for progressive indentation
        prompt_lines = [i for i, line in enumerate(self.output_lines) if line.strip().startswith('>')]
        
        if len(prompt_lines) > 1:
            first_prompt_pos = self.get_leading_spaces(self.output_lines[prompt_lines[0]])
            for i, prompt_line_idx in enumerate(prompt_lines[1:], 1):
                current_pos = self.get_leading_spaces(self.output_lines[prompt_line_idx])
                if current_pos > first_prompt_pos:
                    issues.append(f"Progressive indentation detected: prompt {i} at position {current_pos}")
                    
        # Check for inconsistent response formatting
        agent_lines = [i for i, line in enumerate(self.output_lines) if '🤖 Agent:' in line]
        for agent_line_idx in agent_lines:
            if agent_line_idx + 1 < len(self.output_lines):
                next_line = self.output_lines[agent_line_idx + 1]
                if next_line.strip() and not next_line.startswith('         '):
                    issues.append(f"Inconsistent response formatting at line {agent_line_idx + 1}")
                    
        return issues
        
    def get_leading_spaces(self, line):
        """Count leading spaces in a line."""
        return len(line) - len(line.lstrip(' '))
        
    def get_conversation_flow(self):
        """Extract conversation flow structure."""
        flow = []
        for i, line in enumerate(self.output_lines):
            if line.strip().startswith('>'):
                flow.append(('prompt', i, line))
            elif '🤖 Agent:' in line:
                flow.append(('response', i, line))
            elif '🤔 Thinking...' in line:
                flow.append(('thinking', i, line))
            elif line.startswith('❌ Error:'):
                flow.append(('error', i, line))
        return flow


class TestTUICompleteConversationFlow:
    """Test complete conversation flows with alignment verification."""
    
    @pytest.fixture
    def tui(self):
        """Create TUI instance for testing."""
        return FixedWorkingTUI()
        
    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client for testing."""
        mock_client = AsyncMock()
        mock_client.provider = "test-provider"
        mock_client.model = "test-model"
        mock_client.send_message = AsyncMock()
        mock_client.clear_history = Mock()
        return mock_client
        
    @pytest.mark.asyncio
    async def test_single_conversation_alignment(self, tui, mock_llm_client):
        """Test that a single conversation maintains proper alignment."""
        tui.llm_client = mock_llm_client
        mock_llm_client.send_message.return_value = "Test response from AI"
        
        capture = ConsoleOutputCapture()
        
        with patch('sys.stdout', capture):
            # Initial setup
            tui.clear_screen_and_show_prompt()
            
            # Process single message
            await tui.process_line("Hello, how are you?")
            
            # Show next prompt
            tui.show_prompt()
            
        # Analyze alignment
        issues = capture.get_alignment_issues()
        assert len(issues) == 0, f"Alignment issues found: {issues}"
        
        # Verify conversation structure
        flow = capture.get_conversation_flow()
        thinking_entries = [entry for entry in flow if entry[0] == 'thinking']
        response_entries = [entry for entry in flow if entry[0] == 'response']
        
        assert len(thinking_entries) >= 1, "Should show thinking indicator"
        assert len(response_entries) >= 1, "Should show response"
        
    @pytest.mark.asyncio
    async def test_multiple_conversation_turns_alignment(self, tui, mock_llm_client):
        """Test that multiple conversation turns don't cause progressive indentation."""
        tui.llm_client = mock_llm_client
        
        # Setup different responses
        responses = [
            "First response",
            "Second response with more text",
            "Third response\nWith multiple lines\nTo test formatting"
        ]
        mock_llm_client.send_message.side_effect = responses
        
        capture = ConsoleOutputCapture()
        
        with patch('sys.stdout', capture):
            # Initial setup
            tui.clear_screen_and_show_prompt()
            
            # Multiple conversation turns
            messages = ["First message", "Second message", "Third message"]
            for i, message in enumerate(messages):
                await tui.process_line(message)
                tui.show_prompt()
                
        # Analyze for progressive indentation
        issues = capture.get_alignment_issues()
        assert len(issues) == 0, f"Progressive indentation detected: {issues}"
        
        # Verify all prompts are at same position
        prompt_lines = [line for line in capture.output_lines if line.strip().startswith('>')]
        if len(prompt_lines) > 1:
            first_indent = capture.get_leading_spaces(prompt_lines[0])
            for i, prompt in enumerate(prompt_lines[1:], 1):
                current_indent = capture.get_leading_spaces(prompt)
                assert current_indent == first_indent, f"Prompt {i} indented differently: {current_indent} vs {first_indent}"
                
    @pytest.mark.asyncio 
    async def test_multiline_response_formatting(self, tui, mock_llm_client):
        """Test that multi-line responses are formatted consistently."""
        tui.llm_client = mock_llm_client
        multiline_response = "Line 1 of response\nLine 2 of response\nLine 3 with more content"
        mock_llm_client.send_message.return_value = multiline_response
        
        capture = ConsoleOutputCapture()
        
        with patch('sys.stdout', capture):
            tui.clear_screen_and_show_prompt()
            await tui.process_line("Give me a multiline response")
            
        # Find response lines
        response_start = None
        for i, line in enumerate(capture.output_lines):
            if '🤖 Agent:' in line:
                response_start = i
                break
                
        assert response_start is not None, "Response not found"
        
        # Check that continuation lines are properly indented
        continuation_lines = []
        for i in range(response_start + 1, len(capture.output_lines)):
            line = capture.output_lines[i]
            if line.strip() and not line.startswith('>'):
                continuation_lines.append(line)
            else:
                break
                
        # All continuation lines should have consistent indentation (9 spaces)
        for line in continuation_lines:
            if line.strip():  # Skip empty lines
                assert line.startswith('         '), f"Continuation line not properly indented: '{line}'"
                
    @pytest.mark.asyncio
    async def test_command_execution_alignment(self, tui):
        """Test that command execution maintains alignment."""
        capture = ConsoleOutputCapture()
        
        with patch('sys.stdout', capture):
            tui.clear_screen_and_show_prompt()
            
            # Test help command
            await tui.process_line("/help")
            tui.show_prompt()
            
            # Test clear command
            await tui.process_line("/clear")
            tui.show_prompt()
            
        issues = capture.get_alignment_issues()
        assert len(issues) == 0, f"Command execution caused alignment issues: {issues}"
        
        # Verify help content appears
        help_found = any('📚 Commands:' in line for line in capture.output_lines)
        assert help_found, "Help command output not found"
        
    @pytest.mark.asyncio
    async def test_error_handling_alignment(self, tui, mock_llm_client):
        """Test that error handling maintains proper alignment."""
        tui.llm_client = mock_llm_client
        # Make LLM client raise an exception
        mock_llm_client.send_message.side_effect = Exception("Test error")
        
        capture = ConsoleOutputCapture()
        
        with patch('sys.stdout', capture):
            tui.clear_screen_and_show_prompt()
            await tui.process_line("This should cause an error")
            tui.show_prompt()
            
        issues = capture.get_alignment_issues()
        assert len(issues) == 0, f"Error handling caused alignment issues: {issues}"
        
        # Verify error message appears
        error_found = any('❌ Error:' in line for line in capture.output_lines)
        assert error_found, "Error message not displayed"
        
    def test_very_long_response_alignment(self, tui):
        """Test alignment with very long responses that might wrap."""
        long_response = "This is a very long response that might wrap across terminal lines. " * 10
        
        capture = ConsoleOutputCapture()
        
        with patch('sys.stdout', capture):
            tui.clear_screen_and_show_prompt()
            
            # Simulate response display
            capture.write('🤖 Agent: ')
            lines = long_response.split('\n')
            for i, response_line in enumerate(lines):
                if i == 0:
                    capture.write(response_line + '\n')
                else:
                    capture.write('         ' + response_line + '\n')
            
            tui.show_prompt()
            
        issues = capture.get_alignment_issues()
        assert len(issues) == 0, f"Long response caused alignment issues: {issues}"


class TestTUIEdgeCaseIntegration:
    """Test integration edge cases that could cause alignment issues."""
    
    @pytest.fixture
    def tui(self):
        return FixedWorkingTUI()
        
    def test_rapid_input_processing(self, tui):
        """Test that rapid input processing doesn't break alignment."""
        capture = ConsoleOutputCapture()
        
        with patch('sys.stdout', capture):
            tui.clear_screen_and_show_prompt()
            
            # Simulate rapid prompt showing
            for i in range(10):
                tui.show_prompt()
                
        issues = capture.get_alignment_issues()
        assert len(issues) == 0, f"Rapid input caused alignment issues: {issues}"
        
    @pytest.mark.asyncio
    async def test_empty_and_whitespace_inputs(self, tui):
        """Test handling of empty and whitespace-only inputs."""
        capture = ConsoleOutputCapture()
        
        with patch('sys.stdout', capture):
            tui.clear_screen_and_show_prompt()
            
            # Test various empty inputs
            test_inputs = ["", "   ", "\t", "\n", "  \t  \n  "]
            for test_input in test_inputs:
                await tui.process_line(test_input)
                tui.show_prompt()
                
        issues = capture.get_alignment_issues()
        assert len(issues) == 0, f"Empty input handling caused alignment issues: {issues}"
        
    def test_special_character_inputs(self, tui):
        """Test alignment with special characters and Unicode."""
        capture = ConsoleOutputCapture()
        
        with patch('sys.stdout', capture):
            tui.clear_screen_and_show_prompt()
            
            # Simulate input with special characters
            special_chars = "Special: !@#$%^&*()_+-=[]{}|;:,.<>? 🌟 你好"
            
            # Simulate character-by-character input
            tui.cursor_col = 2  # Start after prompt
            for char in special_chars:
                if ord(char) >= 32:
                    tui.input_buffer += char
                    tui.cursor_col += 1
                    capture.write(char)
                    
            tui.show_prompt()
            
        issues = capture.get_alignment_issues()
        assert len(issues) == 0, f"Special character input caused alignment issues: {issues}"


class TestTUIPerformanceIntegration:
    """Test TUI performance over extended use."""
    
    @pytest.fixture
    def tui(self):
        return FixedWorkingTUI()
        
    @pytest.mark.asyncio
    async def test_extended_conversation_performance(self, tui):
        """Test that extended conversations don't degrade performance."""
        mock_client = AsyncMock()
        mock_client.send_message.return_value = "Response"
        tui.llm_client = mock_client
        
        capture = ConsoleOutputCapture()
        times = []
        
        with patch('sys.stdout', capture):
            tui.clear_screen_and_show_prompt()
            
            # Simulate extended conversation
            for i in range(50):
                start_time = time.time()
                
                await tui.process_line(f"Message {i}")
                tui.show_prompt()
                
                end_time = time.time()
                times.append(end_time - start_time)
                
        # Performance should not degrade significantly
        early_times = times[:10]
        late_times = times[-10:]
        
        early_avg = sum(early_times) / len(early_times)
        late_avg = sum(late_times) / len(late_times)
        
        assert late_avg < early_avg * 3, f"Performance degraded significantly: {early_avg:.4f}s -> {late_avg:.4f}s"
        
        # Alignment should still be correct
        issues = capture.get_alignment_issues()
        assert len(issues) == 0, f"Extended use caused alignment issues: {issues}"
        
    def test_memory_usage_over_time(self, tui):
        """Test that alignment operations don't cause memory leaks."""
        import gc
        
        # Force garbage collection and measure baseline
        gc.collect()
        initial_count = len(gc.get_objects())
        
        capture = ConsoleOutputCapture()
        
        with patch('sys.stdout', capture):
            # Perform many alignment operations
            for i in range(1000):
                tui.clear_screen_and_show_prompt()
                tui.show_prompt()
                tui.cursor_col = 5
                tui.input_buffer = f"test{i}"
                tui.input_buffer = ""  # Clear
                
        # Check for memory leaks
        gc.collect()
        final_count = len(gc.get_objects())
        growth = final_count - initial_count
        
        assert growth < 200, f"Possible memory leak: {growth} new objects after 1000 operations"


class TestTUIRealWorldScenarios:
    """Test real-world usage scenarios."""
    
    @pytest.fixture
    def tui(self):
        return FixedWorkingTUI()
        
    @pytest.mark.asyncio
    async def test_coding_assistance_scenario(self, tui, mock_llm_client=None):
        """Test a typical coding assistance conversation."""
        if mock_llm_client is None:
            mock_llm_client = AsyncMock()
        tui.llm_client = mock_llm_client
        
        # Define realistic coding conversation
        conversation = [
            ("Write a Python function to calculate fibonacci", 
             "Here's a Python function to calculate Fibonacci numbers:\n\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```"),
            ("Can you optimize this?", 
             "Here's an optimized version using dynamic programming:\n\n```python\ndef fibonacci_optimized(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n```"),
            ("Explain the time complexity", 
             "Time complexity comparison:\n- Original: O(2^n) - exponential\n- Optimized: O(n) - linear")
        ]
        
        mock_llm_client.send_message.side_effect = [response for _, response in conversation]
        
        capture = ConsoleOutputCapture()
        
        with patch('sys.stdout', capture):
            tui.clear_screen_and_show_prompt()
            
            for question, _ in conversation:
                await tui.process_line(question)
                tui.show_prompt()
                
        # Verify no alignment issues in realistic scenario
        issues = capture.get_alignment_issues()
        assert len(issues) == 0, f"Coding assistance scenario caused alignment issues: {issues}"
        
        # Verify conversation structure
        flow = capture.get_conversation_flow()
        response_count = len([entry for entry in flow if entry[0] == 'response'])
        assert response_count >= len(conversation), "Not all responses captured"
        
    @pytest.mark.asyncio
    async def test_mixed_command_and_chat_scenario(self, tui, mock_llm_client=None):
        """Test scenario mixing commands and chat."""
        if mock_llm_client is None:
            mock_llm_client = AsyncMock()
        tui.llm_client = mock_llm_client
        mock_llm_client.send_message.return_value = "Chat response"
        
        capture = ConsoleOutputCapture()
        
        with patch('sys.stdout', capture):
            tui.clear_screen_and_show_prompt()
            
            # Mixed interaction pattern
            await tui.process_line("/help")
            tui.show_prompt()
            
            await tui.process_line("Hello, how are you?")
            tui.show_prompt()
            
            await tui.process_line("/clear")
            tui.show_prompt()
            
            await tui.process_line("What's the weather like?")
            tui.show_prompt()
            
        issues = capture.get_alignment_issues()
        assert len(issues) == 0, f"Mixed command/chat scenario caused alignment issues: {issues}"
        
    def test_terminal_resize_simulation(self, tui):
        """Test behavior when terminal width changes (simulated)."""
        capture = ConsoleOutputCapture()
        
        with patch('sys.stdout', capture):
            # Simulate different terminal widths
            widths = [80, 120, 60, 100]
            
            for width in widths:
                tui.clear_screen_and_show_prompt()
                
                # Simulate content that might be affected by width
                long_line = "x" * (width - 10)
                capture.write(long_line + '\n')
                tui.show_prompt()
                
        issues = capture.get_alignment_issues()
        assert len(issues) == 0, f"Terminal width changes caused alignment issues: {issues}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])