"""
Tests for the revolutionary CLI interface with cost intelligence integration.
"""

import asyncio
import json
import pytest
from unittest.mock import Mock, patch, AsyncMock
from click.testing import CliRunner

from agentsmcp.cli import main, interactive, dashboard, costs, budget, optimize


class TestCLIBasics:
    """Test basic CLI functionality."""
    
    def test_main_help(self):
        """Test that main command shows help."""
        runner = CliRunner()
        result = runner.invoke(main, ['--help'])
        
        assert result.exit_code == 0
        assert "AgentsMCP" in result.output
        assert "Revolutionary Multi-Agent Orchestration" in result.output
        assert "Cost Intelligence" in result.output
    
    def test_main_version(self):
        """Test version flag works."""
        runner = CliRunner()
        result = runner.invoke(main, ['--version'])
        
        assert result.exit_code == 0
        assert result.output.strip().startswith("agentsmcp, version")
    
    def test_commands_available(self):
        """Test that all main commands are available."""
        runner = CliRunner()
        result = runner.invoke(main, ['--help'])
        
        # Revolutionary UI commands
        assert "interactive" in result.output
        assert "dashboard" in result.output
        assert "costs" in result.output
        assert "budget" in result.output
        assert "optimize" in result.output
        
        # Server commands
        assert "server" in result.output


class TestCostCommands:
    """Test cost-related CLI commands."""
    
    @patch('agentsmcp.cli.COST_AVAILABLE', True)
    @patch('agentsmcp.cli.CostTracker')
    @patch('agentsmcp.cli.StatisticsDisplay')
    def test_costs_command_basic(self, mock_stats, mock_tracker_class):
        """Test basic costs command."""
        # Setup mock tracker
        mock_tracker = Mock()
        mock_tracker.total_cost = 0.1234
        mock_tracker.get_daily_cost.return_value = 0.0567
        mock_tracker.get_breakdown.return_value = {}
        mock_tracker_class.return_value = mock_tracker
        
        # Setup mock stats display
        mock_stats_instance = Mock()
        mock_stats_instance.render_async = AsyncMock(return_value="Mock stats output")
        mock_stats.return_value = mock_stats_instance
        
        runner = CliRunner()
        result = runner.invoke(costs)
        
        assert result.exit_code == 0
        mock_tracker_class.assert_called_once()
        mock_stats.assert_called_once()
    
    @patch('agentsmcp.cli.COST_AVAILABLE', False)
    def test_costs_command_unavailable(self):
        """Test costs command when cost tracking unavailable."""
        runner = CliRunner()
        result = runner.invoke(costs)
        
        assert result.exit_code == 1
        assert "Cost tracking not available" in result.output
    
    @patch('agentsmcp.cli.COST_AVAILABLE', True)
    @patch('agentsmcp.cli.CostTracker')
    def test_costs_json_format(self, mock_tracker_class):
        """Test costs command with JSON output."""
        # Setup mock tracker
        mock_tracker = Mock()
        mock_tracker.total_cost = 0.1234
        mock_tracker.get_daily_cost.return_value = 0.0567
        mock_tracker.get_breakdown.return_value = {"openai": {"gpt-3.5": 0.05}}
        mock_tracker_class.return_value = mock_tracker
        
        runner = CliRunner()
        result = runner.invoke(costs, ['--format', 'json', '--breakdown', '--daily'])
        
        assert result.exit_code == 0
        
        # Parse JSON output
        output_data = json.loads(result.output)
        assert output_data['total_cost'] == 0.1234
        assert output_data['daily_cost'] == 0.0567
        assert 'breakdown' in output_data
    
    @patch('agentsmcp.cli.COST_AVAILABLE', True)
    @patch('agentsmcp.cli.CostTracker')
    @patch('agentsmcp.cli.BudgetManager')
    def test_budget_set_command(self, mock_budget_class, mock_tracker_class):
        """Test setting budget."""
        # Setup mocks
        mock_tracker = Mock()
        mock_tracker.total_cost = 25.50
        mock_tracker_class.return_value = mock_tracker
        
        mock_budget = Mock()
        mock_budget.check_budget.return_value = True
        mock_budget.remaining_budget.return_value = 74.50
        mock_budget_class.return_value = mock_budget
        
        runner = CliRunner()
        result = runner.invoke(budget, ['100.0'])
        
        assert result.exit_code == 0
        assert "Monthly budget set to $100.00" in result.output
        assert "Within budget" in result.output
        mock_budget_class.assert_called_with(mock_tracker, 100.0)
    
    @patch('agentsmcp.cli.COST_AVAILABLE', True)
    @patch('agentsmcp.cli.CostTracker')
    @patch('agentsmcp.cli.BudgetManager')  
    def test_budget_check_command(self, mock_budget_class, mock_tracker_class):
        """Test checking budget status."""
        # Setup mocks
        mock_tracker = Mock()
        mock_tracker.total_cost = 150.75
        mock_tracker_class.return_value = mock_tracker
        
        mock_budget = Mock()
        mock_budget.check_budget.return_value = False
        mock_budget_class.return_value = mock_budget
        
        runner = CliRunner()
        result = runner.invoke(budget, ['--check'])
        
        assert result.exit_code == 0
        assert "Budget Status: OVER" in result.output
        assert "Overspent by: $50.75" in result.output
    
    @patch('agentsmcp.cli.COST_AVAILABLE', False)
    def test_budget_unavailable(self):
        """Test budget command when cost tracking unavailable."""
        runner = CliRunner()
        result = runner.invoke(budget, ['100'])
        
        assert result.exit_code == 1
        assert "Budget management not available" in result.output


class TestOptimizeCommand:
    """Test the optimize command."""
    
    @patch('agentsmcp.cli.COST_AVAILABLE', True)
    @patch('agentsmcp.cli.CostTracker')
    @patch('agentsmcp.cli.ModelOptimizer')
    def test_optimize_dry_run(self, mock_optimizer_class, mock_tracker_class):
        """Test optimize command with dry run."""
        runner = CliRunner()
        result = runner.invoke(optimize, ['--mode', 'cost', '--dry-run'])
        
        assert result.exit_code == 0
        assert "Optimizing for: COST" in result.output
        assert "OPTIMIZATION RECOMMENDATIONS (DRY RUN)" in result.output
        assert "Switch to Ollama" in result.output
        assert "Run without --dry-run to apply" in result.output
    
    @patch('agentsmcp.cli.COST_AVAILABLE', True) 
    @patch('agentsmcp.cli.CostTracker')
    @patch('agentsmcp.cli.ModelOptimizer')
    def test_optimize_speed_mode(self, mock_optimizer_class, mock_tracker_class):
        """Test optimize command for speed."""
        runner = CliRunner()
        result = runner.invoke(optimize, ['--mode', 'speed', '--dry-run'])
        
        assert result.exit_code == 0
        assert "Optimizing for: SPEED" in result.output
        assert "faster responses" in result.output
    
    @patch('agentsmcp.cli.COST_AVAILABLE', True)
    @patch('agentsmcp.cli.CostTracker') 
    @patch('agentsmcp.cli.ModelOptimizer')
    def test_optimize_apply(self, mock_optimizer_class, mock_tracker_class):
        """Test applying optimizations."""
        runner = CliRunner()
        result = runner.invoke(optimize, ['--mode', 'balanced'])
        
        assert result.exit_code == 0
        assert "Optimization settings applied!" in result.output
        assert "Future agent spawns will use optimized" in result.output


class TestInteractiveCommands:
    """Test interactive CLI commands."""
    
    @patch('agentsmcp.cli.CLIApp')
    @patch('agentsmcp.cli.asyncio.run')
    def test_interactive_command(self, mock_asyncio_run, mock_cli_app_class):
        """Test interactive command launches properly."""
        mock_app = Mock()
        mock_app.start = AsyncMock()
        mock_cli_app_class.return_value = mock_app
        
        runner = CliRunner()
        result = runner.invoke(interactive, ['--theme', 'dark', '--no-welcome'])
        
        # Verify CLI app was created with correct config
        mock_cli_app_class.assert_called_once()
        config_arg = mock_cli_app_class.call_args[0][0]
        assert config_arg.theme_mode == 'dark'
        assert config_arg.show_welcome == False
        assert config_arg.interface_mode == 'interactive'
        
        # Verify asyncio.run was called
        mock_asyncio_run.assert_called_once()
    
    @patch('agentsmcp.cli.CLIApp')
    @patch('agentsmcp.cli.asyncio.run')
    def test_dashboard_command(self, mock_asyncio_run, mock_cli_app_class):
        """Test dashboard command launches properly."""
        mock_app = Mock()
        mock_app.start = AsyncMock(return_value={"mode": "dashboard"})
        mock_cli_app_class.return_value = mock_app
        
        runner = CliRunner()
        result = runner.invoke(dashboard, ['--refresh-interval', '0.5'])
        
        # Verify CLI app was created with dashboard config
        mock_cli_app_class.assert_called_once()
        config_arg = mock_cli_app_class.call_args[0][0]
        assert config_arg.refresh_interval == 0.5
        assert config_arg.interface_mode == 'dashboard'


class TestServerCommands:
    """Test server management commands."""
    
    @patch('agentsmcp.cli.subprocess.Popen')
    @patch('agentsmcp.cli.PID_FILE')
    def test_server_start_background(self, mock_pid_file, mock_popen):
        """Test starting server in background."""
        mock_process = Mock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process
        
        runner = CliRunner()
        result = runner.invoke(main, ['server', 'start', '--background', '--port', '8001'])
        
        assert result.exit_code == 0
        assert "Server started in background (pid=12345)" in result.output
        assert "Web dashboard:" in result.output
        mock_popen.assert_called_once()
        mock_pid_file.write_text.assert_called_with('12345')
    
    @patch('agentsmcp.cli.PID_FILE')
    @patch('agentsmcp.cli.os.kill')
    def test_server_stop(self, mock_kill, mock_pid_file):
        """Test stopping background server."""
        mock_pid_file.exists.return_value = True
        mock_pid_file.read_text.return_value = '12345'
        mock_pid_file.unlink = Mock()
        
        runner = CliRunner()
        result = runner.invoke(main, ['server', 'stop'])
        
        assert result.exit_code == 0
        assert "Sent SIGTERM to pid 12345" in result.output
        mock_kill.assert_called_with(12345, 15)  # SIGTERM = 15
        mock_pid_file.unlink.assert_called_once()
    
    @patch('agentsmcp.cli.PID_FILE')
    def test_server_stop_no_pid_file(self, mock_pid_file):
        """Test stopping server when no PID file exists."""
        mock_pid_file.exists.return_value = False
        
        runner = CliRunner()
        result = runner.invoke(main, ['server', 'stop'])
        
        assert result.exit_code == 1
        assert "PID file not found" in result.output


class TestCLIIntegration:
    """Integration tests for CLI components."""
    
    def test_all_commands_importable(self):
        """Test that all CLI commands can be imported without errors."""
        # This test ensures no import errors in the CLI module
        from agentsmcp import cli
        
        # Verify main commands exist
        assert hasattr(cli, 'main')
        assert hasattr(cli, 'interactive')
        assert hasattr(cli, 'dashboard')
        assert hasattr(cli, 'costs')
        assert hasattr(cli, 'budget')
        assert hasattr(cli, 'optimize')
    
    def test_cli_entry_point_works(self):
        """Test that CLI can be invoked as entry point."""
        runner = CliRunner()
        result = runner.invoke(main, [])
        
        # Click groups return exit code 2 when no command given (standard behavior)
        assert result.exit_code == 2
        assert "Usage:" in result.output
    
    @patch('agentsmcp.cli._load_config')
    @patch('agentsmcp.cli.configure_logging') 
    def test_config_loading_integration(self, mock_configure_logging, mock_load_config):
        """Test that config loading works properly."""
        mock_config = Mock()
        mock_load_config.return_value = mock_config
        
        runner = CliRunner()
        result = runner.invoke(main, ['--config', 'test.yaml', '--help'])
        
        assert result.exit_code == 0
        mock_load_config.assert_called_once_with('test.yaml')
        mock_configure_logging.assert_called_once()


# Async test utilities
@pytest.mark.asyncio
async def test_async_command_helper():
    """Test the async command helper function."""
    from agentsmcp.cli import _run_async_command
    
    async def test_func(x, y):
        return x + y
    
    result = await _run_async_command(test_func, 2, 3)
    assert result == 5


# Mock fixtures for testing
@pytest.fixture
def mock_cost_tracker():
    """Mock cost tracker for testing."""
    tracker = Mock()
    tracker.total_cost = 1.23
    tracker.get_daily_cost.return_value = 0.45
    tracker.get_monthly_cost.return_value = 12.34
    tracker.get_breakdown.return_value = {
        "openai": {"gpt-3.5-turbo": 0.50, "gpt-4": 0.73}
    }
    return tracker


@pytest.fixture 
def mock_cli_app():
    """Mock CLI app for testing."""
    app = Mock()
    app.start = AsyncMock(return_value={"status": "success"})
    return app


# Performance and stress tests
class TestCLIPerformance:
    """Test CLI performance and error handling."""
    
    def test_help_commands_fast(self):
        """Test that help commands respond quickly."""
        import time
        runner = CliRunner()
        
        start = time.time()
        result = runner.invoke(main, ['--help'])
        duration = time.time() - start
        
        assert result.exit_code == 0
        assert duration < 1.0  # Should respond within 1 second
    
    def test_error_handling_graceful(self):
        """Test that errors are handled gracefully."""
        runner = CliRunner()
        
        # Test invalid command
        result = runner.invoke(main, ['invalid-command'])
        assert result.exit_code != 0
        assert "No such command" in result.output
        
        # Test invalid option
        result = runner.invoke(costs, ['--invalid-option'])
        assert result.exit_code != 0