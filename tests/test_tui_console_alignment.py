#!/usr/bin/env python3
"""
Comprehensive test suite for TUI console output alignment.

Tests ensure that all console output starts at column 0 and doesn't continue
from the previous line's position, preventing the progressive indentation bug.
"""

import pytest
import asyncio
import sys
from io import StringIO
from unittest.mock import patch, MagicMock, AsyncMock

# Add src to path to import our modules
sys.path.insert(0, '/Users/mikko/github/AgentsMCP/src')

from agentsmcp.ui.v2.fixed_working_tui import FixedWorkingTUI


class TestTUIConsoleAlignment:
    """Test suite for TUI console output alignment fixes."""

    @pytest.fixture
    def mock_stdout(self):
        """Fixture that captures stdout for analysis."""
        return StringIO()

    @pytest.fixture
    def tui(self):
        """Fixture that creates a FixedWorkingTUI instance."""
        return FixedWorkingTUI()

    def test_clear_screen_and_show_prompt_alignment(self, tui, mock_stdout):
        """Test that clear screen and prompt start at column 0."""
        with patch('sys.stdout', mock_stdout):
            tui.clear_screen_and_show_prompt()
            
        output = mock_stdout.getvalue()
        lines = output.split('\n')
        
        # Check that header lines start with \r (ensuring column 0)
        assert '\r🚀 AgentsMCP - Fixed Working TUI\r\n' in output
        assert '\r' + '─' * 50 + '\r\n' in output
        assert '\rType your message (Ctrl+C to exit, /quit to quit):\r\n' in output

    def test_show_prompt_alignment(self, tui, mock_stdout):
        """Test that prompt starts at column 0."""
        with patch('sys.stdout', mock_stdout):
            tui.show_prompt()
            
        output = mock_stdout.getvalue()
        assert output == '\r> '
        assert tui.cursor_col == 2

    @pytest.mark.asyncio
    async def test_progress_callback_alignment(self, tui, mock_stdout):
        """Test that progress messages start at column 0."""
        # Mock the necessary dependencies
        with patch('sys.stdout', mock_stdout), \
             patch.object(tui, 'llm_client', MagicMock()), \
             patch('agentsmcp.orchestration.team_runner.run_team', new_callable=AsyncMock) as mock_run_team, \
             patch('agentsmcp.orchestration.team_runner.DEFAULT_TEAM', ['test_agent']), \
             patch('shutil.get_terminal_size', return_value=(80, 24)):
            
            # Configure mock to call progress callback
            async def mock_run_team_impl(line, team, progress_callback=None):
                if progress_callback:
                    await progress_callback("job.spawned", {"agent": "test_agent"})
                    await progress_callback("job.completed", {"agent": "test_agent"})
                return {"test_agent": "test output"}
            
            mock_run_team.side_effect = mock_run_team_impl
            
            # Process a test line
            await tui.process_line("test message")
            
        output = mock_stdout.getvalue()
        
        # Check that progress messages start with \r
        assert '\r[] ▶ test_agent started\n' in output
        assert '\r[] ✅ test_agent completed\n' in output

    @pytest.mark.asyncio
    async def test_agent_response_alignment(self, tui, mock_stdout):
        """Test that agent responses start at column 0."""
        # Mock the necessary dependencies
        with patch('sys.stdout', mock_stdout), \
             patch.object(tui, 'llm_client', MagicMock()), \
             patch('agentsmcp.orchestration.team_runner.run_team', new_callable=AsyncMock) as mock_run_team, \
             patch('agentsmcp.orchestration.team_runner.DEFAULT_TEAM', ['test_agent']), \
             patch('shutil.get_terminal_size', return_value=(80, 24)), \
             patch('agentsmcp.ui.v2.ansi_markdown_processor.render_markdown_lines', return_value=['line1', 'line2']):
            
            # Configure mock to return test results
            mock_run_team.return_value = {"test_agent": "test response\nwith multiple lines"}
            
            # Process a test line
            await tui.process_line("test message")
            
        output = mock_stdout.getvalue()
        
        # Check that role headers start with \r
        assert '\r🧩 test_agent:\n' in output
        
        # Check that response lines start with \r
        assert '\rline1\n' in output
        assert '\rline2\n' in output

    @pytest.mark.asyncio
    async def test_error_message_alignment(self, tui, mock_stdout):
        """Test that error messages start at column 0."""
        with patch('sys.stdout', mock_stdout), \
             patch.object(tui, 'llm_client', None):  # No LLM client to trigger error path
            
            await tui.process_line("test message")
            
        output = mock_stdout.getvalue()
        
        # Check that error messages start with \r
        assert '\r⚠️  LLM client unavailable. You said: "test message"\n' in output
        assert '\r   Try restarting the TUI to reconnect.\n' in output

    @pytest.mark.asyncio
    async def test_exception_handling_alignment(self, tui, mock_stdout):
        """Test that exception messages start at column 0."""
        with patch('sys.stdout', mock_stdout), \
             patch.object(tui, 'llm_client', MagicMock()), \
             patch('agentsmcp.orchestration.team_runner.run_team', side_effect=Exception("Test error")):
            
            await tui.process_line("test message")
            
        output = mock_stdout.getvalue()
        
        # Check that exception messages start with \r
        assert '\r❌ Error: Test error\n' in output
        assert '\r   Please try again or use /help for commands.\n' in output

    def test_team_orchestration_alignment(self, mock_stdout):
        """Test that team orchestration message starts at column 0."""
        # Directly test the output line
        with patch('sys.stdout', mock_stdout):
            sys.stdout.write('\r🚩 Orchestrating team: business_analyst, backend_engineer\n')
            
        output = mock_stdout.getvalue()
        assert output == '\r🚩 Orchestrating team: business_analyst, backend_engineer\n'

    def test_cursor_position_tracking(self, tui):
        """Test that cursor position is properly tracked."""
        # Test initial state
        assert tui.cursor_col == 0
        
        # Test after clear screen
        with patch('sys.stdout', StringIO()):
            tui.clear_screen_and_show_prompt()
        assert tui.cursor_col == 0
        
        # Test after showing prompt
        with patch('sys.stdout', StringIO()):
            tui.show_prompt()
        assert tui.cursor_col == 2  # After "> "

    def test_no_progressive_indentation(self, mock_stdout):
        """Test that output doesn't progressively indent."""
        with patch('sys.stdout', mock_stdout):
            # Simulate multiple lines of output
            sys.stdout.write('\r🤔 Thinking...\n')
            sys.stdout.write('\r🚩 Orchestrating team: agent1, agent2\n')
            sys.stdout.write('\r[] ▶ agent1 started\n')
            sys.stdout.write('\r[] ▶ agent2 started\n')
            sys.stdout.write('\r🧩 agent1:\n')
            sys.stdout.write('\rResponse from agent1\n')
            
        output = mock_stdout.getvalue()
        lines = output.split('\n')
        
        # Each line should start with \r (except empty lines)
        content_lines = [line for line in lines if line.strip()]
        for line in content_lines:
            assert line.startswith('\r'), f"Line doesn't start with \\r: {repr(line)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])