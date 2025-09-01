import subprocess
import sys

import pytest

@pytest.mark.timeout(10)
def test_cli_help():
    """Ensure the CLI can be invoked with --help without error."""
    result = subprocess.run([sys.executable, "-m", "agentsmcp", "--help"], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, f"agentsmcp --help failed with exit code {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"

@pytest.mark.timeout(10)
def test_tui_start():
    """Attempt to start the interactive TUI and ensure it launches (or exits) cleanly.
    The test runs the command with a short timeout and expects a zero exit code.
    If the process hangs or crashes, the test will fail.
    """
    # Using the module entry point to avoid shell complications
    result = subprocess.run([sys.executable, "-m", "agentsmcp", "run", "interactive"], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, f"agentsmcp run interactive failed with exit code {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
