# tests/test_commands/test_pipeline.py
"""
Tests for the CLI command implementation (agentsmcp.commands.pipeline).
We use Click's testing utilities (CliRunner) and mock the underlying engine.
"""

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# --------------------------------------------------------------------------- #
#  Fixture – a patched `PipelineEngine` that the command imports.
# --------------------------------------------------------------------------- #
@pytest.fixture
def patched_engine():
    """
    Replace the real engine with a mock that has an async `run` method.
    """
    with patch("agentsmcp.commands.pipeline.PipelineEngine") as mock_engine_class:
        fake_engine = MagicMock()
        fake_engine.run = AsyncMock(return_value={"status": "completed", "result": "success"})
        mock_engine_class.return_value = fake_engine
        yield fake_engine


@pytest.fixture
def patched_monitor():
    """
    Mock the create_monitor function to avoid UI during tests.
    """
    with patch("agentsmcp.commands.pipeline.create_monitor") as mock_create:
        fake_monitor = MagicMock()
        fake_monitor.__aenter__ = AsyncMock(return_value=fake_monitor)
        fake_monitor.__aexit__ = AsyncMock(return_value=None)
        fake_monitor.update = MagicMock()
        mock_create.return_value = fake_monitor
        yield fake_monitor


@pytest.fixture
def patched_template_manager():
    """
    Mock the TemplateManager to provide controlled template responses.
    """
    with patch("agentsmcp.templates.manager.get_default_template_manager") as mock_manager_func:
        fake_manager = MagicMock()
        fake_manager.list_templates.return_value = ["basic", "python-package", "node-app"]
        fake_manager.get_template_info.return_value = {
            "name": "basic",
            "description": "Basic pipeline template",
            "available": True
        }
        fake_manager.create_pipeline_from_template.return_value = {
            "name": "test-pipeline",
            "description": "Test pipeline",
            "version": "1.0.0",
            "stages": [{"name": "build", "agents": []}]
        }
        mock_manager_func.return_value = fake_manager
        yield fake_manager


# --------------------------------------------------------------------------- #
#  1️⃣ Command parsing and basic execution
# --------------------------------------------------------------------------- #
@pytest.mark.pipeline
def test_pipeline_list_command():
    from agentsmcp.commands.pipeline import list_templates
    
    runner = CliRunner()
    result = runner.invoke(list_templates)
    
    # Should succeed
    assert result.exit_code == 0
    # Should show available templates (from our built-in templates)
    assert "basic" in result.output or "Available templates:" in result.output


@pytest.mark.pipeline
def test_pipeline_info_command(patched_template_manager):
    from agentsmcp.commands.pipeline import template_info
    
    runner = CliRunner()
    result = runner.invoke(template_info, ["basic"])
    
    # Should succeed
    assert result.exit_code == 0
    # Should show template information
    assert "basic" in result.output.lower()


@pytest.mark.pipeline
def test_pipeline_create_command(patched_template_manager, tmp_path):
    from agentsmcp.commands.pipeline import create
    
    output_file = tmp_path / "test-pipeline.yaml"
    
    runner = CliRunner()
    result = runner.invoke(create, [
        "basic", 
        str(output_file),
        "--var", "repository=test-repo",
        "--var", "build_command=make build"
    ])
    
    # Should succeed
    assert result.exit_code == 0
    
    # Template manager should have been called
    patched_template_manager.create_pipeline_from_template.assert_called_once()
    
    # Check the call arguments
    call_args = patched_template_manager.create_pipeline_from_template.call_args
    assert call_args[0][0] == "basic"  # template name
    assert call_args[0][1] == str(output_file)  # output path
    assert "repository" in call_args[1]["variables"]
    assert call_args[1]["variables"]["repository"] == "test-repo"


# --------------------------------------------------------------------------- #
#  2️⃣ Pipeline execution with mocked components
# --------------------------------------------------------------------------- #
@pytest.mark.pipeline
def test_pipeline_run_command(patched_engine, patched_monitor, tmp_path):
    from agentsmcp.commands.pipeline import run
    
    # Create a sample pipeline file
    pipeline_file = tmp_path / "test-pipeline.yaml"
    pipeline_content = """
name: "test-pipeline"
description: "Test pipeline"
version: "1.0.0"
stages:
  - name: build
    agents:
      - type: ollama-turbo
        model: gpt-oss:120b
        task: build_project
        payload: {}
"""
    pipeline_file.write_text(pipeline_content)
    
    runner = CliRunner()
    result = runner.invoke(run, [str(pipeline_file), "--no-ui"])
    
    # Should succeed
    assert result.exit_code == 0
    
    # Engine should have been called
    assert patched_engine.run.called


@pytest.mark.pipeline
def test_pipeline_run_with_stage_filter(patched_engine, patched_monitor, tmp_path):
    from agentsmcp.commands.pipeline import run
    
    # Create a sample pipeline file
    pipeline_file = tmp_path / "test-pipeline.yaml"
    pipeline_content = """
name: "test-pipeline"
description: "Test pipeline"
version: "1.0.0"
stages:
  - name: build
    agents: []
  - name: test
    agents: []
  - name: deploy
    agents: []
"""
    pipeline_file.write_text(pipeline_content)
    
    runner = CliRunner()
    result = runner.invoke(run, [str(pipeline_file), "--stage", "test", "--no-ui"])
    
    # Should succeed
    assert result.exit_code == 0


# --------------------------------------------------------------------------- #
#  3️⃣ Error handling scenarios
# --------------------------------------------------------------------------- #
@pytest.mark.pipeline
def test_pipeline_run_missing_file():
    from agentsmcp.commands.pipeline import run
    
    runner = CliRunner()
    result = runner.invoke(run, ["nonexistent.yaml", "--no-ui"])
    
    # Should fail with error code
    assert result.exit_code != 0
    # Should show helpful error message
    assert "not found" in result.output.lower() or "error" in result.output.lower()


@pytest.mark.pipeline
def test_pipeline_run_invalid_yaml(tmp_path):
    from agentsmcp.commands.pipeline import run
    
    # Create invalid YAML file
    pipeline_file = tmp_path / "invalid.yaml"
    pipeline_file.write_text("invalid: yaml: content: [unclosed")
    
    runner = CliRunner()
    result = runner.invoke(run, [str(pipeline_file), "--no-ui"])
    
    # Should fail
    assert result.exit_code != 0


@pytest.mark.pipeline
def test_pipeline_engine_exception(patched_engine, patched_monitor, tmp_path):
    from agentsmcp.commands.pipeline import run
    
    # Make the engine raise an exception
    patched_engine.run.side_effect = Exception("Pipeline execution failed")
    
    # Create valid pipeline file
    pipeline_file = tmp_path / "test-pipeline.yaml"
    pipeline_content = """
name: "test-pipeline"
stages: []
"""
    pipeline_file.write_text(pipeline_content)
    
    runner = CliRunner()
    result = runner.invoke(run, [str(pipeline_file), "--no-ui"])
    
    # Should handle the error gracefully
    assert result.exit_code != 0
    assert "failed" in result.output.lower() or "error" in result.output.lower()


# --------------------------------------------------------------------------- #
#  4️⃣ Dry run functionality
# --------------------------------------------------------------------------- #
@pytest.mark.pipeline
def test_pipeline_dry_run(patched_engine, patched_monitor, tmp_path):
    from agentsmcp.commands.pipeline import run
    
    # Create a sample pipeline file
    pipeline_file = tmp_path / "test-pipeline.yaml"
    pipeline_content = """
name: "test-pipeline"
stages:
  - name: build
    agents: []
"""
    pipeline_file.write_text(pipeline_content)
    
    runner = CliRunner()
    result = runner.invoke(run, [str(pipeline_file), "--dry-run", "--no-ui"])
    
    # Should succeed
    assert result.exit_code == 0
    
    # Should show dry run output
    assert "dry run" in result.output.lower() or "would execute" in result.output.lower()
    
    # Engine should NOT be called for dry run
    assert not patched_engine.run.called


# --------------------------------------------------------------------------- #
#  5️⃣ Template variable handling
# --------------------------------------------------------------------------- #
@pytest.mark.pipeline
def test_create_command_with_variables(patched_template_manager, tmp_path):
    from agentsmcp.commands.pipeline import create
    
    output_file = tmp_path / "configured-pipeline.yaml"
    
    runner = CliRunner()
    result = runner.invoke(create, [
        "python-package",
        str(output_file),
        "--var", "repository=my-python-project",
        "--var", "python_version=3.11",
        "--var", "test_command=pytest"
    ])
    
    # Should succeed
    assert result.exit_code == 0
    
    # Check that variables were passed correctly
    call_args = patched_template_manager.create_pipeline_from_template.call_args
    variables = call_args[1]["variables"]
    
    assert variables["repository"] == "my-python-project"
    assert variables["python_version"] == "3.11"
    assert variables["test_command"] == "pytest"


# --------------------------------------------------------------------------- #
#  6️⃣ Template validation
# --------------------------------------------------------------------------- #
@pytest.mark.pipeline
def test_validate_command(patched_template_manager):
    from agentsmcp.commands.pipeline import validate
    
    # Mock successful validation
    patched_template_manager.validate_template_file.return_value = {
        "valid": True,
        "message": "Template is valid"
    }
    
    runner = CliRunner()
    result = runner.invoke(validate, ["basic"])
    
    # Should succeed
    assert result.exit_code == 0
    assert "valid" in result.output.lower()


@pytest.mark.pipeline 
def test_validate_command_invalid_template(patched_template_manager):
    from agentsmcp.commands.pipeline import validate
    
    # Mock validation failure
    patched_template_manager.validate_template_file.return_value = {
        "valid": False, 
        "message": "Template has YAML syntax errors"
    }
    
    runner = CliRunner()
    result = runner.invoke(validate, ["invalid-template"])
    
    # Should indicate validation failure
    assert result.exit_code != 0
    assert "error" in result.output.lower() or "invalid" in result.output.lower()