# tests/test_pipeline/test_ui.py
"""
Tests for the Rich based UI components (initialisation, progress tracking,
error formatting, etc.).
"""

import builtins
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from rich.console import Console
from rich.progress import Progress

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# --------------------------------------------------------------------------- #
#  Helper – create a UI instance that uses a mocked Console.
# --------------------------------------------------------------------------- #
@pytest.fixture
def mocked_console():
    console = MagicMock(spec=Console)
    console.is_terminal = True
    return console


@pytest.fixture
def mock_tracker():
    """Create a mock ExecutionTracker for UI testing."""
    tracker = MagicMock()
    tracker.get_status.return_value = {
        "pipeline_name": "test-pipeline",
        "current_stage": "build",
        "progress": 0.5,
        "status": "running"
    }
    return tracker


@pytest.fixture
def pipeline_monitor(mocked_console, mock_tracker):
    from agentsmcp.pipeline.ui import PipelineMonitor
    
    monitor = PipelineMonitor(
        tracker=mock_tracker,
        pipeline_name="test-pipeline"
    )
    # Replace the console with our mock
    monitor.console = mocked_console
    return monitor


# --------------------------------------------------------------------------- #
#  1️⃣ UI initialisation
# --------------------------------------------------------------------------- #
@pytest.mark.ui
def test_pipeline_monitor_initialises(pipeline_monitor, mock_tracker):
    # The UI should have stored the tracker and pipeline name
    assert pipeline_monitor.tracker is mock_tracker
    assert pipeline_monitor.pipeline_name == "test-pipeline"
    assert pipeline_monitor.console is not None


# --------------------------------------------------------------------------- #
#  2️⃣ Status display and updates
# --------------------------------------------------------------------------- #
@pytest.mark.ui
def test_display_status_calls_console(pipeline_monitor, mocked_console):
    pipeline_monitor.display_status()
    
    # The console should have been called to print status
    assert mocked_console.print.called
    
    # Check that pipeline name appears in the output
    print_calls = mocked_console.print.call_args_list
    output_text = str(print_calls)
    assert "test-pipeline" in output_text


@pytest.mark.ui
def test_update_progress_displays_correctly(pipeline_monitor, mocked_console, mock_tracker):
    # Mock different status scenarios
    mock_tracker.get_status.return_value = {
        "pipeline_name": "test-pipeline",
        "current_stage": "test",
        "progress": 0.75,
        "status": "running",
        "agents": [
            {"name": "agent1", "status": "completed"},
            {"name": "agent2", "status": "running"}
        ]
    }
    
    pipeline_monitor.update()
    
    # Console should be updated with new status
    assert mocked_console.print.called
    
    # Check that stage and progress info appears
    print_calls = mocked_console.print.call_args_list
    output_text = str(print_calls)
    assert "test" in output_text  # stage name
    

# --------------------------------------------------------------------------- #
#  3️⃣ Error formatting and display
# --------------------------------------------------------------------------- #
@pytest.mark.ui
def test_display_error_formats_message(pipeline_monitor, mocked_console):
    err_msg = "Something went wrong!"
    pipeline_monitor.display_error(err_msg)

    # The UI should print the error 
    assert mocked_console.print.called
    
    # Verify the error message appears in output
    print_calls = mocked_console.print.call_args_list
    output_text = str(print_calls)
    assert err_msg in output_text


@pytest.mark.ui
def test_display_error_with_details(pipeline_monitor, mocked_console):
    err_msg = "Build failed"
    details = {"stage": "build", "agent": "agent1", "exit_code": 1}
    
    pipeline_monitor.display_error(err_msg, details)
    
    # Should print both message and details
    assert mocked_console.print.called
    print_calls = mocked_console.print.call_args_list
    output_text = str(print_calls)
    
    assert err_msg in output_text
    assert "build" in output_text
    assert "agent1" in output_text


# --------------------------------------------------------------------------- #
#  4️⃣ Success message display
# --------------------------------------------------------------------------- #
@pytest.mark.ui
def test_display_success_shows_completion(pipeline_monitor, mocked_console):
    pipeline_monitor.display_success("Pipeline completed successfully!")
    
    # Should print success message
    assert mocked_console.print.called
    print_calls = mocked_console.print.call_args_list
    output_text = str(print_calls)
    
    assert "completed successfully" in output_text


# --------------------------------------------------------------------------- #
#  5️⃣ Context manager functionality
# --------------------------------------------------------------------------- #
@pytest.mark.ui
def test_pipeline_monitor_context_manager(mocked_console, mock_tracker):
    from agentsmcp.pipeline.ui import PipelineMonitor
    
    with PipelineMonitor(tracker=mock_tracker, pipeline_name="ctx-test") as monitor:
        monitor.console = mocked_console
        assert monitor is not None
        monitor.display_status()
    
    # Should have displayed status during context
    assert mocked_console.print.called


# --------------------------------------------------------------------------- #
#  6️⃣ Mocking Rich console – ensure no real terminal output while testing.
# --------------------------------------------------------------------------- #
@pytest.mark.ui 
def test_ui_does_not_write_to_real_stdout(pipeline_monitor):
    with patch.object(builtins, "print") as mock_print:
        pipeline_monitor.display_error("test error")
        # Rich should use its own console, not built-in print
        mock_print.assert_not_called()


# --------------------------------------------------------------------------- #
#  7️⃣ Agent status panel tests
# --------------------------------------------------------------------------- #
@pytest.mark.ui
def test_agent_status_display(pipeline_monitor, mocked_console, mock_tracker):
    # Mock agent status data
    mock_tracker.get_status.return_value = {
        "pipeline_name": "test-pipeline", 
        "current_stage": "build",
        "agents": [
            {"name": "builder", "status": "running", "progress": 0.6},
            {"name": "tester", "status": "pending", "progress": 0.0},
            {"name": "deployer", "status": "completed", "progress": 1.0}
        ]
    }
    
    pipeline_monitor.update()
    
    # Console should show agent information
    assert mocked_console.print.called
    print_calls = mocked_console.print.call_args_list
    output_text = str(print_calls)
    
    # Should contain agent names and statuses
    assert "builder" in output_text or "tester" in output_text or "deployer" in output_text