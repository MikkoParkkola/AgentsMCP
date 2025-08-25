import os
import subprocess
import sys
import tempfile
import pytest
from pathlib import Path
import pexpect
import requests
import time

# ----------------------------------------------------------------------
# General helpers
# ----------------------------------------------------------------------
@pytest.fixture(scope="session")
def binary_path():
    """Return absolute path to the AgentsMCP binary/module."""
    # Since we're using python -m agentsmcp, return the python command with module
    return [sys.executable, "-m", "agentsmcp"]

@pytest.fixture(scope="module")
def api_endpoint():
    """Base URL for the local web server (default port 8000)."""
    return "http://127.0.0.1:8000"

# ----------------------------------------------------------------------
# Temporary working dir (clean sandbox)
# ----------------------------------------------------------------------
@pytest.fixture
def temp_dir():
    """Create an isolated temp directory for each test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)
        # Temp dir automatically removed

# ----------------------------------------------------------------------
# Logging fixture for better test debugging
# ----------------------------------------------------------------------
@pytest.fixture
def log_file(request, temp_dir):
    """Create a log file for each test to capture pexpect interactions."""
    test_name = request.node.name.replace("[", "_").replace("]", "_")
    log_path = temp_dir / f"{test_name}.log"
    return log_path

# ----------------------------------------------------------------------
# The "mcp" fixture – a Pexpect master that works cross‑platform
# ----------------------------------------------------------------------
@pytest.fixture
def mcp(request, binary_path, temp_dir, log_file):
    """
    Start the binary in the requested mode and feed/receive data.

    :param request: special fixture to read keyword arguments
    :param binary_path: fixture
    :param temp_dir: fixture
    :param log_file: fixture for logging
    :return: an object with `send`, `expect`, `close`, etc.
    """
    args = getattr(request, 'param', [])  # e.g. ["--mode", "interactive"]
    env = os.environ.copy()
    # give the binary its own working dir
    env["AGENTSMCP_WORKING_DIR"] = str(temp_dir)

    # Build complete command
    cmd = binary_path + args
    
    # Create log file for debugging
    logfile = open(str(log_file), 'w', encoding='utf-8')
    
    try:
        # On Windows we need the spawn class that works
        if os.name == 'nt':
            child = pexpect.popen_spawn.PopenSpawn(
                ' '.join(cmd), 
                env=env, 
                encoding="utf-8", 
                timeout=15,
                logfile=logfile
            )
        else:
            child = pexpect.spawn(
                cmd[0], 
                args=cmd[1:], 
                env=env, 
                encoding="utf-8", 
                timeout=15,
                logfile=logfile
            )
        
        # Set reasonable defaults for terminal interaction
        child.delaybeforesend = 0.1  # Small delay before sending commands
        
        yield child
        
    finally:
        try:
            child.terminate(force=True)
            child.wait()
        except:
            pass  # Already closed
        
        try:
            logfile.close()
        except:
            pass

# ----------------------------------------------------------------------
# API helper fixture (starts the server mode)
# ----------------------------------------------------------------------
@pytest.fixture
def web_server(binary_path, api_endpoint, temp_dir):
    """
    Launch a background AgentsMCP in interactive mode to test web API.
    """
    env = os.environ.copy()
    env["AGENTSMCP_WORKING_DIR"] = str(temp_dir)
    
    # Start AgentsMCP in interactive mode (web API starts automatically)
    cmd = binary_path + ["--no-welcome", "--debug"]
    child = subprocess.Popen(
        cmd,
        cwd=str(temp_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env
    )
    
    # Give the server time to start
    server_ready = False
    try:
        for i in range(30):  # Wait up to 6 seconds
            try:
                r = requests.get(api_endpoint + "/health", timeout=0.2)
                if r.status_code == 200:
                    server_ready = True
                    break
            except Exception:
                pass
            time.sleep(0.2)
        
        if server_ready:
            yield api_endpoint
        else:
            raise Exception("Web server failed to start")
    finally:
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()