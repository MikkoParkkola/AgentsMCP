# tests/test_templates/test_manager.py
"""
Tests for the TemplateManager (discovery, rendering, validation & error handling).
"""

import json
import os
import sys
from pathlib import Path

import pytest
import yaml
from jinja2 import UndefinedError
from pydantic import ValidationError

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# --------------------------------------------------------------------------- #
#  Helpers – the TemplateManager is deliberately imported inside each test
#  to guarantee a fresh import when monkey‑patching globals.
# --------------------------------------------------------------------------- #
def get_manager(template_dir: Path):
    from agentsmcp.templates.manager import TemplateManager
    return TemplateManager(builtin_dir=template_dir)


# --------------------------------------------------------------------------- #
#  1️⃣ Discovery & caching
# --------------------------------------------------------------------------- #
@pytest.mark.template
def test_discover_templates_creates_cache(template_manager, temp_template_dir):
    # List available templates
    templates = template_manager.list_templates()
    
    # Should find our test templates
    assert "hello_world" in templates
    assert "invalid" in templates
    
    # The manager should have an internal cache
    assert hasattr(template_manager, "_cache")
    assert len(template_manager._cache) >= 2


@pytest.mark.template
def test_get_template_info(template_manager):
    # Get info for existing template
    info = template_manager.get_template_info("hello_world")
    
    assert info["name"] == "hello_world"
    assert info["available"] is True
    assert "description" in info
    assert "stages" in info
    
    # Get info for non-existent template
    info = template_manager.get_template_info("non_existent")
    assert info["available"] is False


# --------------------------------------------------------------------------- #
#  2️⃣ Rendering with variables
# --------------------------------------------------------------------------- #
@pytest.mark.template
def test_render_template_success(template_manager, hello_world_template):
    rendered_dict = template_manager.load_template(
        "hello_world",
        variables={"pipeline_name": "test-pipeline", "message": "Test message"}
    )
    
    assert rendered_dict["name"] == "test-pipeline-pipeline"
    assert rendered_dict["stages"][0]["agents"][0]["payload"]["message"] == "Test message"


@pytest.mark.template
def test_render_template_missing_template_raises(template_manager):
    from agentsmcp.templates.manager import TemplateNotFoundError
    
    with pytest.raises(TemplateNotFoundError):
        template_manager.load_template("non_existent", variables={})


@pytest.mark.template
def test_render_template_jinja_error_propagates(template_manager, invalid_template):
    from agentsmcp.templates.manager import TemplateRenderingError
    
    # The template references an undefined variable – Jinja2 should raise.
    with pytest.raises(TemplateRenderingError):
        template_manager.load_template(
            "invalid",
            variables={"pipeline_name": "bad"}
        )


# --------------------------------------------------------------------------- #
#  3️⃣ Pipeline creation workflow
# --------------------------------------------------------------------------- #
@pytest.mark.template
def test_create_pipeline_from_template(template_manager, temp_dir):
    output_path = temp_dir / "test_pipeline.yaml"
    
    pipeline_dict = template_manager.create_pipeline_from_template(
        "hello_world",
        output_path,
        variables={"pipeline_name": "created-pipeline", "message": "Created!"}
    )
    
    # File should be created
    assert output_path.exists()
    
    # Check the returned dict
    assert pipeline_dict["name"] == "created-pipeline-pipeline"
    
    # Check the written file
    with open(output_path) as f:
        file_content = yaml.safe_load(f)
    
    assert file_content["name"] == "created-pipeline-pipeline"
    assert file_content["stages"][0]["agents"][0]["payload"]["message"] == "Created!"


# --------------------------------------------------------------------------- #
#  4️⃣ Validation
# --------------------------------------------------------------------------- #
@pytest.mark.template
def test_validate_template_file_success(template_manager, hello_world_template):
    result = template_manager.validate_template_file(hello_world_template)
    
    assert result["valid"] is True
    assert "Template syntax is valid" in result["message"]


@pytest.mark.template
def test_validate_template_file_invalid(template_manager, temp_template_dir):
    # Create a file with invalid YAML syntax
    bad_yaml = temp_template_dir / "bad_yaml.yaml"
    bad_yaml.write_text("name: test\nstages:\n  - invalid: yaml: syntax:")
    
    result = template_manager.validate_template_file(bad_yaml)
    
    assert result["valid"] is False
    assert "YAML syntax error" in result["message"]


@pytest.mark.template
def test_validate_template_file_not_found(template_manager, temp_template_dir):
    from agentsmcp.templates.manager import TemplateNotFoundError
    
    non_existent = temp_template_dir / "does_not_exist.yaml"
    
    with pytest.raises(TemplateNotFoundError):
        template_manager.validate_template_file(non_existent)


# --------------------------------------------------------------------------- #
#  5️⃣ File I/O edge cases
# --------------------------------------------------------------------------- #
@pytest.mark.template
def test_render_empty_template_returns_minimal_pipeline(template_manager, temp_template_dir):
    # Create an empty template file
    empty_file = temp_template_dir / "empty.yaml"
    empty_file.write_text("""
name: "empty-pipeline"
description: "Empty pipeline"
version: "1.0.0"
stages: []
    """.strip())
    
    # Refresh cache to pick up new template
    template_manager._discover_templates()
    
    rendered = template_manager.load_template("empty", variables={})
    assert rendered["name"] == "empty-pipeline"
    assert rendered["stages"] == []


@pytest.mark.template
def test_write_pipeline_creates_parent_dirs(template_manager, temp_dir):
    # Create nested path that doesn't exist
    nested_path = temp_dir / "deep" / "nested" / "pipeline.yaml"
    
    pipeline_dict = {
        "name": "test-pipeline",
        "description": "Test",
        "version": "1.0.0",
        "stages": []
    }
    
    template_manager.write_pipeline(pipeline_dict, nested_path)
    
    # File and parent directories should be created
    assert nested_path.exists()
    assert nested_path.parent.exists()
    
    # Content should be correct
    with open(nested_path) as f:
        content = yaml.safe_load(f)
    
    assert content["name"] == "test-pipeline"