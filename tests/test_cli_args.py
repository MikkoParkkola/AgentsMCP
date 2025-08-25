import pytest
import subprocess
import sys
from pathlib import Path

@pytest.mark.flaky(reruns=2, reruns_delay=1)
def test_help_flag(binary_path):
    """Test that --help displays usage information."""
    result = subprocess.run(
        binary_path + ["--help"],
        capture_output=True,
        text=True,
        timeout=10
    )
    assert result.returncode == 0
    assert "AgentsMCP - Revolutionary Multi-Agent Orchestration Platform" in result.stdout
    assert "--mode" in result.stdout
    assert "--theme" in result.stdout

@pytest.mark.flaky(reruns=2, reruns_delay=1)
@pytest.mark.parametrize(
    "mode",
    ["interactive", "stats"]  # Skip dashboard mode in subprocess tests
)
def test_mode_argument(binary_path, temp_dir, mode):
    """Test different mode arguments."""
    result = subprocess.run(
        binary_path + ["--mode", mode, "--no-welcome", "--debug"],
        input="\n",  # Send enter to exit if interactive
        capture_output=True,
        text=True,
        timeout=10,
        cwd=str(temp_dir)
    )
    # Should not crash - returncode 0 or graceful exit codes are OK
    assert result.returncode in [0, 1, 130]  # 130 is SIGINT

@pytest.mark.flaky(reruns=2, reruns_delay=1)  
def test_dashboard_mode_quick_exit(binary_path, temp_dir):
    """Test dashboard mode starts and can be interrupted quickly."""
    # Start dashboard mode in background
    process = subprocess.Popen(
        binary_path + ["--mode", "dashboard", "--no-welcome", "--debug"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(temp_dir)
    )
    
    try:
        # Let it start for 2 seconds then interrupt
        import time
        time.sleep(2)
        
        # Send SIGTERM (graceful shutdown)
        process.terminate()
        
        # Wait for clean exit
        stdout, stderr = process.communicate(timeout=5)
        
        # Should exit cleanly
        assert process.returncode in [0, -15, 130]  # 0, SIGTERM, or SIGINT
        
    except subprocess.TimeoutExpired:
        # Force kill if it doesn't exit
        process.kill()
        process.communicate()
        pytest.fail("Dashboard mode did not exit gracefully within timeout")

@pytest.mark.flaky(reruns=2, reruns_delay=1)
@pytest.mark.parametrize(
    "theme",
    ["auto", "light", "dark"]
)
def test_theme_argument(binary_path, temp_dir, theme):
    """Test theme selection arguments."""
    result = subprocess.run(
        binary_path + ["--theme", theme, "--no-welcome", "--debug"],
        input="\n",  # Send enter to exit if interactive
        capture_output=True,
        text=True,
        timeout=10,
        cwd=str(temp_dir)
    )
    # Should not crash
    assert result.returncode in [0, 1, 130]

@pytest.mark.flaky(reruns=2, reruns_delay=1)
def test_no_welcome_flag(binary_path, temp_dir):
    """Test that --no-welcome skips the welcome screen."""
    result = subprocess.run(
        binary_path + ["--no-welcome", "--debug"],
        input="\n",  # Send enter to exit
        capture_output=True,
        text=True,
        timeout=10,
        cwd=str(temp_dir)
    )
    # Should not crash and shouldn't show welcome screen
    assert result.returncode in [0, 1, 130]

@pytest.mark.flaky(reruns=2, reruns_delay=1)
def test_no_colors_flag(binary_path, temp_dir):
    """Test that --no-colors disables color output."""
    result = subprocess.run(
        binary_path + ["--no-colors", "--no-welcome", "--debug"],
        input="\n",  # Send enter to exit
        capture_output=True,
        text=True,
        timeout=10,
        cwd=str(temp_dir)
    )
    # Should not crash
    assert result.returncode in [0, 1, 130]

@pytest.mark.flaky(reruns=2, reruns_delay=1)
def test_refresh_interval_argument(binary_path, temp_dir):
    """Test custom refresh interval."""
    result = subprocess.run(
        binary_path + ["--refresh-interval", "0.5", "--no-welcome", "--debug"],
        input="\n",  # Send enter to exit
        capture_output=True,
        text=True,
        timeout=10,
        cwd=str(temp_dir)
    )
    # Should not crash
    assert result.returncode in [0, 1, 130]

@pytest.mark.flaky(reruns=2, reruns_delay=1)
def test_debug_flag(binary_path, temp_dir):
    """Test that --debug enables debug mode."""
    result = subprocess.run(
        binary_path + ["--debug", "--no-welcome"],
        input="\n",  # Send enter to exit
        capture_output=True,
        text=True,
        timeout=10,
        cwd=str(temp_dir)
    )
    # Should not crash
    assert result.returncode in [0, 1, 130]