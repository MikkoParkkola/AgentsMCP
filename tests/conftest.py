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

# ----------------------------------------------------------------------
# Multi-turn Tool Execution Test Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def test_llm_client():
    """Create a test LLMClient with consistent configuration."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from agentsmcp.conversation.llm_client import LLMClient
    
    client = LLMClient()
    client.config = {
        "provider": "ollama-turbo",
        "model": "gpt-oss:120b",
        "temperature": 0.1,  # Low temperature for consistent tests
        "max_tokens": 1024
    }
    # Clear any existing conversation history
    client.conversation_history = []
    return client


@pytest.fixture
def mock_tool_responses():
    """Provide realistic mock tool responses."""
    return {
        "list_directory": "Contents of .: DIR src, DIR tests, DIR docs, FILE README.md, FILE setup.py, FILE requirements.txt",
        "read_file": "def main():\n    \"\"\"Main function.\"\"\"\n    print('Hello, World!')\n    return 0\n\nif __name__ == '__main__':\n    main()",
        "search_files": "Found 5 Python files: src/main.py, src/utils.py, tests/test_main.py, tests/test_utils.py, setup.py"
    }


@pytest.fixture
def mock_llm_responses():
    """Provide realistic LLM response structures."""
    return {
        "with_tools": {
            'model': 'gpt-oss:120b',
            'message': {
                'role': 'assistant',
                'content': '',
                'tool_calls': [{
                    'function': {
                        'name': 'list_directory',
                        'arguments': {'path': '.'}
                    }
                }]
            }
        },
        "analysis": {
            'model': 'gpt-oss:120b',
            'message': {
                'role': 'assistant',
                'content': """Based on the tool execution results, I can provide this analysis:

**Project Structure:**
- Well-organized with src/, tests/, and docs/ directories
- Standard Python project layout with setup.py and requirements.txt

**Code Quality Observations:**
- Main functionality is properly organized in src/ directory
- Test suite is present in tests/ directory
- Documentation appears to be available

**Recommendations:**
1. Ensure all code has proper test coverage
2. Keep dependencies in requirements.txt up to date  
3. Consider adding type hints for better code clarity

This appears to be a well-structured Python project following standard conventions.""",
                'tool_calls': None
            }
        },
        "empty": {
            'model': 'gpt-oss:120b',
            'message': {
                'role': 'assistant',
                'content': '',
                'tool_calls': None
            }
        }
    }


# --------------------------------------------------------------------------- #
#  Pipeline Testing Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def temp_template_dir():
    """
    Create a temporary directory that lives for the whole test session.
    The directory is automatically deleted at the end.
    """
    import shutil
    dir_path = Path(tempfile.mkdtemp(prefix="agentsmcp_templates_"))
    yield dir_path
    shutil.rmtree(str(dir_path), ignore_errors=True)


@pytest.fixture(scope="session")
def hello_world_template(temp_template_dir):
    """
    A minimal valid pipeline template that renders to a YAML
    representation of a pipeline with basic stages.
    """
    content = """
name: "{{ pipeline_name | default('hello-world') }}-pipeline"
description: >
  A simple test pipeline with basic stages.
version: "1.0.0"

defaults:
  timeout_seconds: 300
  retries: 1
  on_failure: retry

stages:
  - name: setup
    description: Setup stage
    parallel: false
    agents:
      - type: ollama-turbo
        model: gpt-oss:120b
        task: setup_project
        payload:
          message: "{{ message | default('Hello World') }}"
        timeout_seconds: 300

notifications:
  on_success:
    - type: slack
      channel: "#ci-success"
      message: "✅ Pipeline {{ pipeline_name | default('hello-world') }} completed"
  on_failure:
    - type: slack
      channel: "#ci-failure"
      message: "❌ Pipeline {{ pipeline_name | default('hello-world') }} failed"
    """
    file_path = temp_template_dir / "hello_world.yaml"
    file_path.write_text(content.strip())
    return file_path


@pytest.fixture(scope="session")
def invalid_template(temp_template_dir):
    """
    A template that deliberately produces invalid YAML after rendering.
    This is used to test validation failures.
    """
    content = """
name: "{{ pipeline_name }}-pipeline"
description: "Invalid template"
version: "1.0.0"

stages:
  - name: bad-stage
    description: "This will fail"
    agents:
      - type: ollama-turbo
        model: gpt-oss:120b
        task: {{ undefined_variable }}  # <-- Jinja2 will raise an error
        payload: {}
    """
    file_path = temp_template_dir / "invalid.yaml"
    file_path.write_text(content.strip())
    return file_path


@pytest.fixture
def template_manager(hello_world_template, invalid_template, temp_template_dir):
    """
    Returns a fresh TemplateManager instance pointing at the temporary
    template directory.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from agentsmcp.templates.manager import TemplateManager

    manager = TemplateManager(builtin_dir=temp_template_dir)
    return manager


# Pytest markers for multi-turn test categorization
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests for individual components")
    config.addinivalue_line("markers", "integration: Integration tests with real components") 
    config.addinivalue_line("markers", "behavioral: Behavioral tests for user scenarios")
    config.addinivalue_line("markers", "regression: Regression tests to prevent breaking changes")
    config.addinivalue_line("markers", "multiturn: Multi-turn tool execution tests")
    config.addinivalue_line("markers", "slow: Tests that take longer to run")
    config.addinivalue_line("markers", "pipeline: Pipeline system tests")
    config.addinivalue_line("markers", "template: Template management tests")
    config.addinivalue_line("markers", "ui: User interface tests")
    config.addinivalue_line("markers", "async: Asynchronous tests")