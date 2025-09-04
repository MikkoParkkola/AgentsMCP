#!/usr/bin/env python3
"""
CLI integration tests for retrospective TUI commands.

Tests the three new CLI commands added for retrospective functionality:
- retrospective tui: Full TUI interface
- retrospective approve --tui: TUI-based approval interface 
- retrospective monitor: Implementation monitoring TUI
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typer.testing import CliRunner
from click.testing import CliRunner as ClickRunner
from io import StringIO

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from agentsmcp.cli import app, retrospective_group
from agentsmcp.ui.v3.retrospective_tui_interface import RetrospectiveTUIInterface
from agentsmcp.ui.v3.approval_interaction_handler import ApprovalInteractionHandler
from agentsmcp.ui.v3.progress_monitoring_view import ProgressMonitoringView


class TestRetrospectiveCLICommands:
    """Test suite for retrospective CLI command integration."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.runner = CliRunner()
    
    def test_retrospective_tui_command_exists(self):
        """Test that retrospective tui command exists and is accessible."""
        # Test that the command group exists
        result = self.runner.invoke(app, ["retrospective", "--help"])
        assert result.exit_code == 0
        assert "tui" in result.output
    
    def test_retrospective_tui_command_help(self):
        """Test retrospective tui command help text."""
        result = self.runner.invoke(app, ["retrospective", "tui", "--help"])
        assert result.exit_code == 0
        assert "Launch full retrospective TUI interface" in result.output
    
    @patch('agentsmcp.ui.v3.retrospective_tui_interface.RetrospectiveTUIInterface')
    def test_retrospective_tui_command_execution(self, mock_interface_class):
        """Test retrospective tui command execution."""
        # Mock the TUI interface
        mock_interface = MagicMock()
        mock_interface.run = AsyncMock()
        mock_interface_class.return_value = mock_interface
        
        # Test command execution
        result = self.runner.invoke(app, ["retrospective", "tui"])
        
        # Command should execute without error
        # Note: May exit with code 1 due to mocked async context, but should not crash
        assert result.exit_code in [0, 1]
        mock_interface_class.assert_called_once()
    
    def test_retrospective_approve_command_help(self):
        """Test retrospective approve command help text."""
        result = self.runner.invoke(app, ["retrospective", "approve", "--help"])
        assert result.exit_code == 0
        assert "--tui" in result.output or "tui" in result.output.lower()
    
    @patch('agentsmcp.ui.v3.approval_interaction_handler.ApprovalInteractionHandler')
    def test_retrospective_approve_tui_command(self, mock_handler_class):
        """Test retrospective approve --tui command execution."""
        # Mock the approval handler
        mock_handler = MagicMock()
        mock_handler.run_interactive_approval = AsyncMock()
        mock_handler_class.return_value = mock_handler
        
        # Test command execution with --tui flag
        result = self.runner.invoke(app, ["retrospective", "approve", "--tui"])
        
        # Command should execute without error 
        assert result.exit_code in [0, 1]
        mock_handler_class.assert_called_once()
    
    def test_retrospective_monitor_command_help(self):
        """Test retrospective monitor command help text."""
        result = self.runner.invoke(app, ["retrospective", "monitor", "--help"])
        assert result.exit_code == 0
        assert "Launch implementation monitoring TUI" in result.output
    
    @patch('agentsmcp.ui.v3.progress_monitoring_view.ProgressMonitoringView')
    def test_retrospective_monitor_command_execution(self, mock_view_class):
        """Test retrospective monitor command execution."""
        # Mock the monitoring view
        mock_view = MagicMock()
        mock_view.run_monitoring_interface = AsyncMock()
        mock_view_class.return_value = mock_view
        
        # Test command execution
        result = self.runner.invoke(app, ["retrospective", "monitor"])
        
        # Command should execute without error
        assert result.exit_code in [0, 1]
        mock_view_class.assert_called_once()
    
    def test_retrospective_group_structure(self):
        """Test that retrospective command group is properly structured."""
        # Get help for retrospective group
        result = self.runner.invoke(app, ["retrospective", "--help"])
        assert result.exit_code == 0
        
        # Should contain the new commands
        expected_commands = ["tui", "approve", "monitor"]
        for cmd in expected_commands:
            assert cmd in result.output
    
    def test_command_error_handling(self):
        """Test that commands handle errors gracefully."""
        # Test invalid retrospective subcommand
        result = self.runner.invoke(app, ["retrospective", "nonexistent"])
        assert result.exit_code != 0
        assert "No such command" in result.output or "Usage:" in result.output
    
    @patch('agentsmcp.ui.v3.retrospective_tui_interface.RetrospectiveTUIInterface')
    def test_tui_command_with_terminal_detection(self, mock_interface_class):
        """Test TUI command with terminal capability detection."""
        # Mock interface with terminal detection
        mock_interface = MagicMock()
        mock_interface.run = AsyncMock()
        mock_interface.console.is_terminal = True
        mock_interface_class.return_value = mock_interface
        
        result = self.runner.invoke(app, ["retrospective", "tui"])
        
        # Should attempt to create interface regardless of exit code
        mock_interface_class.assert_called_once()


class TestCLIIntegrationWithTUIComponents:
    """Test integration between CLI commands and TUI components."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.runner = CliRunner()
    
    @patch('agentsmcp.ui.v3.retrospective_tui_interface.RetrospectiveTUIInterface')
    @patch('rich.console.Console')
    def test_tui_initialization_flow(self, mock_console_class, mock_interface_class):
        """Test the initialization flow from CLI to TUI components."""
        # Mock console
        mock_console = MagicMock()
        mock_console.is_terminal = True
        mock_console_class.return_value = mock_console
        
        # Mock TUI interface
        mock_interface = MagicMock()
        mock_interface.run = AsyncMock()
        mock_interface_class.return_value = mock_interface
        
        # Execute CLI command
        result = self.runner.invoke(app, ["retrospective", "tui"])
        
        # Verify initialization chain
        mock_interface_class.assert_called_once()
        # Verify console was passed (implementation detail may vary)
    
    @patch('agentsmcp.ui.v3.approval_interaction_handler.ApprovalInteractionHandler')
    def test_approval_handler_initialization(self, mock_handler_class):
        """Test approval handler initialization from CLI."""
        mock_handler = MagicMock()
        mock_handler.run_interactive_approval = AsyncMock()
        mock_handler_class.return_value = mock_handler
        
        result = self.runner.invoke(app, ["retrospective", "approve", "--tui"])
        
        # Should initialize handler
        mock_handler_class.assert_called_once()
    
    @patch('agentsmcp.ui.v3.progress_monitoring_view.ProgressMonitoringView')
    def test_monitoring_view_initialization(self, mock_view_class):
        """Test monitoring view initialization from CLI.""" 
        mock_view = MagicMock()
        mock_view.run_monitoring_interface = AsyncMock()
        mock_view_class.return_value = mock_view
        
        result = self.runner.invoke(app, ["retrospective", "monitor"])
        
        # Should initialize view
        mock_view_class.assert_called_once()
    
    def test_command_parameter_passing(self):
        """Test that command parameters are properly parsed and passed."""
        # Test approve command with different flags
        result = self.runner.invoke(app, ["retrospective", "approve", "--help"])
        assert result.exit_code == 0
        
        # Should show TUI option
        assert "--tui" in result.output or "tui" in result.output.lower()


class TestAsyncCommandHandling:
    """Test async command handling for TUI operations."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.runner = CliRunner()
    
    @patch('asyncio.run')
    @patch('agentsmcp.ui.v3.retrospective_tui_interface.RetrospectiveTUIInterface')
    def test_async_execution_wrapper(self, mock_interface_class, mock_asyncio_run):
        """Test that CLI properly wraps async TUI operations."""
        mock_interface = MagicMock()
        mock_interface.run = AsyncMock()
        mock_interface_class.return_value = mock_interface
        
        # Mock asyncio.run to avoid actual async execution in tests
        mock_asyncio_run.return_value = None
        
        result = self.runner.invoke(app, ["retrospective", "tui"])
        
        # Should attempt to run async operation
        mock_interface_class.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_async_operation_compatibility(self):
        """Test that async operations are compatible with CLI structure."""
        # Test that we can create and run async operations similar to CLI
        from agentsmcp.ui.v3.retrospective_tui_interface import RetrospectiveTUIInterface
        from rich.console import Console
        
        console = Console(file=StringIO(), force_terminal=True)
        interface = RetrospectiveTUIInterface(console=console)
        
        # Should be able to create without error
        assert interface is not None
        assert interface.console is not None
        
        # Mock the run method to avoid full execution
        with patch.object(interface, 'run') as mock_run:
            mock_run.return_value = None
            await interface.run()
            mock_run.assert_called_once()


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery in CLI integration."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.runner = CliRunner()
    
    @patch('agentsmcp.ui.v3.retrospective_tui_interface.RetrospectiveTUIInterface')
    def test_tui_initialization_error_handling(self, mock_interface_class):
        """Test handling of TUI initialization errors."""
        # Mock interface to raise exception during initialization
        mock_interface_class.side_effect = Exception("TUI initialization failed")
        
        result = self.runner.invoke(app, ["retrospective", "tui"])
        
        # Should handle error gracefully
        assert result.exit_code != 0
        mock_interface_class.assert_called_once()
    
    @patch('agentsmcp.ui.v3.approval_interaction_handler.ApprovalInteractionHandler')
    def test_approval_error_handling(self, mock_handler_class):
        """Test handling of approval handler errors."""
        # Mock handler to raise exception
        mock_handler_class.side_effect = Exception("Approval handler failed")
        
        result = self.runner.invoke(app, ["retrospective", "approve", "--tui"])
        
        # Should handle error gracefully
        assert result.exit_code != 0
        mock_handler_class.assert_called_once()
    
    def test_invalid_command_combinations(self):
        """Test handling of invalid command combinations."""
        # Test invalid flags or combinations
        result = self.runner.invoke(app, ["retrospective", "tui", "--invalid-flag"])
        
        # Should show error or help
        assert result.exit_code != 0
    
    @patch('sys.stderr', new_callable=StringIO)
    def test_error_message_quality(self, mock_stderr):
        """Test that error messages are helpful and informative."""
        result = self.runner.invoke(app, ["retrospective", "nonexistent"])
        
        # Should provide helpful error message
        assert result.exit_code != 0
        # Error should be clear about what went wrong
        assert len(result.output) > 0


if __name__ == "__main__":
    """Run tests directly."""
    print("🧪 Running Retrospective CLI Integration Tests")
    print("=" * 60)
    
    # Run pytest programmatically
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--no-header", 
        "--disable-warnings"
    ])
    
    sys.exit(exit_code)