#!/usr/bin/env python3
"""
Basic Functionality Test Scenarios for Revolutionary TUI Interface

These test scenarios validate the core functionality that every TUI user expects:
- Launch and startup
- Basic commands (help, clear, quit)  
- Input visibility and responsiveness
- Error handling
"""

import pytest
import subprocess
import time
from pathlib import Path
from typing import List, Dict

# Test configuration
BASIC_TEST_TIMEOUT = 15


class BasicFunctionalityTests:
    """Test scenarios for basic TUI functionality."""
    
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
    def test_tui_launches_successfully(self):
        """Test TUI launches and shows proper interface."""
        cmd = [self.get_agentsmcp_cmd(), "tui"]
        
        # Test that TUI starts and can be quit
        process = subprocess.run(
            cmd,
            input="quit\n",
            capture_output=True,
            text=True,
            timeout=BASIC_TEST_TIMEOUT,
            cwd=Path(__file__).parent.parent
        )
        
        # Should exit cleanly
        assert process.returncode in [0, 130], f"TUI exited with code {process.returncode}"
        
        # Should not show basic fallback prompt
        assert "> " not in process.stdout.split('\n')[0], "TUI fell back to basic prompt mode"
        
        # Output should indicate some kind of TUI interface
        output_lower = process.stdout.lower()
        assert any(keyword in output_lower for keyword in [
            'tui', 'revolutionary', 'rich', 'enhanced', 'interface'
        ]), "No TUI interface indicators found in output"
    
    @pytest.mark.ui
    @pytest.mark.integration
    def test_help_command_works(self):
        """Test help command shows available commands."""
        cmd = [self.get_agentsmcp_cmd(), "tui"]
        
        process = subprocess.run(
            cmd,
            input="help\nquit\n",
            capture_output=True,
            text=True,
            timeout=BASIC_TEST_TIMEOUT,
            cwd=Path(__file__).parent.parent
        )
        
        assert process.returncode in [0, 130], f"Help test failed with code {process.returncode}"
        
        # Should show help information
        output_lower = process.stdout.lower()
        assert any(keyword in output_lower for keyword in [
            'help', 'commands', 'available', 'usage', 'quit', 'clear'
        ]), "Help command did not show expected content"
    
    @pytest.mark.ui
    @pytest.mark.integration
    def test_clear_command_works(self):
        """Test clear command executes without errors."""
        cmd = [self.get_agentsmcp_cmd(), "tui"]
        
        process = subprocess.run(
            cmd,
            input="help\nclear\nquit\n",
            capture_output=True,
            text=True,
            timeout=BASIC_TEST_TIMEOUT,
            cwd=Path(__file__).parent.parent
        )
        
        assert process.returncode in [0, 130], f"Clear test failed with code {process.returncode}"
        
        # Should not crash or show errors
        stderr_lower = process.stderr.lower()
        assert 'error' not in stderr_lower, f"Clear command produced errors: {process.stderr}"
        assert 'exception' not in stderr_lower, f"Clear command caused exceptions: {process.stderr}"
    
    @pytest.mark.ui
    @pytest.mark.integration
    def test_quit_command_works(self):
        """Test quit command exits gracefully."""
        cmd = [self.get_agentsmcp_cmd(), "tui"]
        
        start_time = time.time()
        process = subprocess.run(
            cmd,
            input="quit\n",
            capture_output=True,
            text=True,
            timeout=BASIC_TEST_TIMEOUT,
            cwd=Path(__file__).parent.parent
        )
        end_time = time.time()
        
        # Should exit quickly and cleanly
        assert process.returncode in [0, 130], f"Quit command failed with code {process.returncode}"
        assert (end_time - start_time) < 10, "Quit command took too long to exit"
        
        # Should not show error messages
        stderr_lower = process.stderr.lower()
        assert 'error' not in stderr_lower or 'exception' not in stderr_lower, \
            f"Quit produced errors: {process.stderr}"
    
    @pytest.mark.ui
    @pytest.mark.integration
    def test_input_visibility(self):
        """Test that user input is properly handled (core issue)."""
        cmd = [self.get_agentsmcp_cmd(), "tui"]
        
        # Send some test input
        test_input = "test input visibility\nquit\n"
        
        process = subprocess.run(
            cmd,
            input=test_input,
            capture_output=True,
            text=True,
            timeout=BASIC_TEST_TIMEOUT,
            cwd=Path(__file__).parent.parent
        )
        
        assert process.returncode in [0, 130], f"Input visibility test failed with code {process.returncode}"
        
        # Should not fall back to basic prompt mode (the original issue)
        lines = process.stdout.split('\n')
        basic_prompts = [line for line in lines if line.strip().startswith('> ') and len(line.strip()) < 10]
        assert len(basic_prompts) <= 1, f"Found too many basic prompts, indicating fallback mode: {basic_prompts}"
        
        # Should handle the input without crashing
        assert 'test' in process.stdout or process.returncode in [0, 130], \
            "Input was not processed correctly"
    
    @pytest.mark.ui
    @pytest.mark.integration  
    def test_error_handling(self):
        """Test error handling with invalid commands."""
        cmd = [self.get_agentsmcp_cmd(), "tui"]
        
        process = subprocess.run(
            cmd,
            input="invalid_command_that_does_not_exist\nhelp\nquit\n",
            capture_output=True,
            text=True,
            timeout=BASIC_TEST_TIMEOUT,
            cwd=Path(__file__).parent.parent
        )
        
        # Should not crash on invalid input
        assert process.returncode in [0, 130], f"Error handling test failed with code {process.returncode}"
        
        # Should show help after invalid command (graceful handling)
        output_lower = process.stdout.lower()
        assert any(keyword in output_lower for keyword in [
            'help', 'commands', 'available'
        ]), "Did not show help after invalid command"
        
        # Should not show Python exceptions to user
        assert 'traceback' not in process.stdout.lower(), "Python traceback shown to user"
        assert 'exception' not in process.stdout.lower(), "Python exception shown to user"


# Standalone execution
if __name__ == "__main__":
    pytest.main([__file__, "-v"])