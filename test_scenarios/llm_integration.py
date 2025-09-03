#!/usr/bin/env python3
"""
LLM Integration Test Scenarios for Revolutionary TUI Interface

These test scenarios validate that the TUI properly integrates with LLM backends:
- Basic AI interaction  
- Multi-turn conversations
- Response formatting
- Error handling for LLM failures
"""

import pytest
import subprocess
import time
import re
from pathlib import Path
from typing import List, Dict

# Test configuration
LLM_TEST_TIMEOUT = 30  # Longer timeout for LLM responses
QUICK_LLM_TIMEOUT = 15  # Shorter timeout for simple tests


class LLMIntegrationTests:
    """Test scenarios for LLM integration in TUI."""
    
    @staticmethod
    def get_agentsmcp_cmd():
        """Get the agentsmcp command path."""
        project_root = Path(__file__).parent.parent
        agentsmcp_cmd = project_root / "agentsmcp"
        
        if not agentsmcp_cmd.exists():
            return "agentsmcp"  # Try system installed version
        return str(agentsmcp_cmd)
    
    @pytest.mark.ui
    @pytest.mark.integration
    @pytest.mark.slow
    def test_basic_llm_interaction(self):
        """Test basic LLM interaction works."""
        cmd = [self.get_agentsmcp_cmd(), "tui"]
        
        # Send a simple greeting
        process = subprocess.run(
            cmd,
            input="Hello, can you help me?\nquit\n",
            capture_output=True,
            text=True,
            timeout=LLM_TEST_TIMEOUT,
            cwd=Path(__file__).parent.parent
        )
        
        assert process.returncode in [0, 130], f"LLM interaction failed with code {process.returncode}"
        
        # Should get some kind of response (not just echo)
        output_lower = process.stdout.lower()
        
        # Look for typical AI response patterns
        ai_response_indicators = [
            'hello', 'help', 'assist', 'sure', 'yes', 'how can', 'what can',
            'i can help', 'certainly', 'of course', 'glad to help'
        ]
        
        found_response = any(indicator in output_lower for indicator in ai_response_indicators)
        
        # If no clear AI response, check that at least the input was processed
        if not found_response:
            # Should at least show that input was received and processed
            assert len(process.stdout.strip()) > 20, \
                "No significant output - LLM may not be responding"
    
    @pytest.mark.ui
    @pytest.mark.integration
    @pytest.mark.slow
    def test_python_question_response(self):
        """Test LLM can answer Python-related questions."""
        cmd = [self.get_agentsmcp_cmd(), "tui"]
        
        process = subprocess.run(
            cmd,
            input="What is Python programming language?\nquit\n",
            capture_output=True,
            text=True,
            timeout=LLM_TEST_TIMEOUT,
            cwd=Path(__file__).parent.parent
        )
        
        assert process.returncode in [0, 130], f"Python question test failed with code {process.returncode}"
        
        # Should mention Python and programming concepts
        output_lower = process.stdout.lower()
        python_indicators = [
            'python', 'programming', 'language', 'code', 'script',
            'development', 'syntax', 'interpreter'
        ]
        
        found_indicators = [indicator for indicator in python_indicators if indicator in output_lower]
        assert len(found_indicators) >= 2, \
            f"LLM response should mention Python concepts. Found: {found_indicators}"
    
    @pytest.mark.ui
    @pytest.mark.integration
    @pytest.mark.slow
    def test_multi_turn_conversation(self):
        """Test multi-turn conversation with LLM."""
        cmd = [self.get_agentsmcp_cmd(), "tui"]
        
        # Multiple related questions
        conversation = """What is Python?
Explain Python variables
How do I create a list in Python?
quit
"""
        
        process = subprocess.run(
            cmd,
            input=conversation,
            capture_output=True,
            text=True,
            timeout=LLM_TEST_TIMEOUT * 2,  # Longer timeout for multiple interactions
            cwd=Path(__file__).parent.parent
        )
        
        assert process.returncode in [0, 130], f"Multi-turn conversation failed with code {process.returncode}"
        
        # Should have responses to multiple questions
        output_lower = process.stdout.lower()
        
        # Check for responses to each question
        python_response = any(word in output_lower for word in ['python', 'programming', 'language'])
        variable_response = any(word in output_lower for word in ['variable', 'data', 'value', 'assign'])
        list_response = any(word in output_lower for word in ['list', 'array', '[', ']', 'bracket'])
        
        responses_found = sum([python_response, variable_response, list_response])
        assert responses_found >= 2, \
            f"Should respond to multiple questions. Responses found: {responses_found}/3"
    
    @pytest.mark.ui 
    @pytest.mark.integration
    def test_quick_llm_response(self):
        """Test LLM responds in reasonable time."""
        cmd = [self.get_agentsmcp_cmd(), "tui"]
        
        start_time = time.time()
        process = subprocess.run(
            cmd,
            input="Hi\nquit\n",
            capture_output=True,
            text=True,
            timeout=QUICK_LLM_TIMEOUT,
            cwd=Path(__file__).parent.parent
        )
        end_time = time.time()
        
        assert process.returncode in [0, 130], f"Quick LLM test failed with code {process.returncode}"
        
        # Should complete within reasonable time
        total_time = end_time - start_time
        assert total_time < QUICK_LLM_TIMEOUT * 0.8, \
            f"LLM response took too long: {total_time:.2f}s"
        
        # Should have some output indicating processing
        assert len(process.stdout.strip()) > 5, "No meaningful output from LLM interaction"
    
    @pytest.mark.ui
    @pytest.mark.integration  
    def test_llm_error_handling(self):
        """Test graceful handling of LLM errors."""
        cmd = [self.get_agentsmcp_cmd(), "tui"]
        
        # Send a potentially problematic input
        problematic_input = "A" * 1000 + "\nquit\n"  # Very long input
        
        process = subprocess.run(
            cmd,
            input=problematic_input,
            capture_output=True,
            text=True,
            timeout=LLM_TEST_TIMEOUT,
            cwd=Path(__file__).parent.parent
        )
        
        # Should not crash even with problematic input
        assert process.returncode in [0, 130], f"LLM error handling failed with code {process.returncode}"
        
        # Should not show Python exceptions
        output_combined = (process.stdout + process.stderr).lower()
        assert 'traceback' not in output_combined, "Python traceback visible to user"
        assert 'exception' not in output_combined, "Python exception visible to user"
        
        # Should handle gracefully
        assert len(process.stdout.strip()) > 0, "No output - may have crashed silently"
    
    @pytest.mark.ui
    @pytest.mark.integration
    @pytest.mark.slow
    def test_code_example_request(self):
        """Test LLM can provide code examples."""
        cmd = [self.get_agentsmcp_cmd(), "tui"]
        
        process = subprocess.run(
            cmd,
            input="Show me a simple Python function example\nquit\n",
            capture_output=True,
            text=True,
            timeout=LLM_TEST_TIMEOUT,
            cwd=Path(__file__).parent.parent
        )
        
        assert process.returncode in [0, 130], f"Code example test failed with code {process.returncode}"
        
        # Should provide code-related response
        output_lower = process.stdout.lower()
        code_indicators = [
            'def ', 'function', 'return', 'python', ':', 'example',
            'code', 'print', 'variable'
        ]
        
        found_code_indicators = [indicator for indicator in code_indicators if indicator in output_lower]
        assert len(found_code_indicators) >= 3, \
            f"Should provide code-related response. Found: {found_code_indicators}"
    
    @pytest.mark.ui
    @pytest.mark.integration
    def test_empty_message_handling(self):
        """Test handling of empty messages."""
        cmd = [self.get_agentsmcp_cmd(), "tui"]
        
        process = subprocess.run(
            cmd,
            input="\n\nquit\n",  # Empty messages followed by quit
            capture_output=True,
            text=True,
            timeout=QUICK_LLM_TIMEOUT,
            cwd=Path(__file__).parent.parent
        )
        
        # Should handle empty input gracefully
        assert process.returncode in [0, 130], f"Empty message handling failed with code {process.returncode}"
        
        # Should not crash or show errors
        stderr_lower = process.stderr.lower()
        assert 'error' not in stderr_lower, f"Empty input caused errors: {process.stderr}"


# Standalone execution
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])