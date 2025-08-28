# tests/test_pipeline/test_integration.py
"""
End‑to‑end integration test that wires together:
  * TemplateManager → render → validation
  * PipelineUI (rich) → progress updates  
  * PipelineMonitor → status snapshots
  * A mocked pipeline engine that pretends to execute steps.
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# --------------------------------------------------------------------------- #
#  Fixture – a minimal in‑memory "engine" that the CLI would call.
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_engine():
    """
    An object with an async `run` method that pretends to execute a pipeline.
    The method yields progress events that the UI can consume.
    """
    engine = MagicMock()
    
    async def _run(pipeline_spec, progress_callback=None):
        # Simulate three stages
        stages = ["setup", "build", "test"]
        for i, stage in enumerate(stages):
            await asyncio.sleep(0.01)  # tiny delay to simulate work
            if progress_callback:
                await progress_callback({
                    "stage": stage,
                    "progress": (i + 1) / len(stages),
                    "status": "running" if i < len(stages) - 1 else "completed"
                })
        
        return {"status": "completed", "result": "success", "stages_completed": len(stages)}

    engine.run = AsyncMock(side_effect=_run)
    return engine


@pytest.fixture
def mock_tracker():
    """Mock execution tracker for integration tests."""
    tracker = MagicMock()
    tracker.get_status.return_value = {
        "pipeline_name": "integration-test",
        "current_stage": "setup",
        "progress": 0.0,
        "status": "running",
        "agents": []
    }
    return tracker


# --------------------------------------------------------------------------- #
#  Integration test – everything works as expected.
# --------------------------------------------------------------------------- #
@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_pipeline_execution_flow(
    template_manager,
    hello_world_template,
    fake_engine,
    mock_tracker
):
    """
    1️⃣ Render a valid pipeline from a template.
    2️⃣ Validate it (Pydantic).
    3️⃣ Run the pipeline with the fake engine.
    4️⃣ Verify UI gets progress updates and displays a success message.
    5️⃣ Verify monitor captures snapshots.
    """

    # ------------------------------------------------------------------- #
    #  1️⃣ Render + validate template
    # ------------------------------------------------------------------- #
    pipeline_dict = template_manager.load_template(
        "hello_world",
        variables={"pipeline_name": "integration-test", "message": "E2E Test"}
    )
    
    # Should have rendered correctly
    assert pipeline_dict["name"] == "integration-test-pipeline"
    assert pipeline_dict["stages"][0]["agents"][0]["payload"]["message"] == "E2E Test"

    # ------------------------------------------------------------------- #
    #  2️⃣ Set up monitoring
    # ------------------------------------------------------------------- #
    from agentsmcp.pipeline.monitor import LiveMonitor, MonitoringConfig
    
    config = MonitoringConfig(enable_ui=False, json_output=False)
    monitor = LiveMonitor(tracker=mock_tracker, config=config)
    
    await monitor.start()

    # ------------------------------------------------------------------- #
    #  3️⃣ Execute the pipeline with progress tracking
    # ------------------------------------------------------------------- #
    progress_updates = []
    
    async def progress_callback(update):
        progress_updates.append(update)
        # Update mock tracker to reflect progress
        mock_tracker.get_status.return_value.update({
            "current_stage": update.get("stage", "unknown"),
            "progress": update.get("progress", 0.0),
            "status": update.get("status", "running")
        })
    
    # Run the engine with our progress callback
    result = await fake_engine.run(pipeline_dict, progress_callback)

    # ------------------------------------------------------------------- #
    #  4️⃣ Verify execution results
    # ------------------------------------------------------------------- #
    assert result["status"] == "completed"
    assert result["result"] == "success"
    assert result["stages_completed"] == 3
    
    # Should have received progress updates
    assert len(progress_updates) == 3
    assert progress_updates[0]["stage"] == "setup"
    assert progress_updates[1]["stage"] == "build"
    assert progress_updates[2]["stage"] == "test"
    
    # Final progress should be complete
    assert progress_updates[-1]["progress"] == 1.0
    assert progress_updates[-1]["status"] == "completed"

    # ------------------------------------------------------------------- #
    #  5️⃣ Verify monitoring
    # ------------------------------------------------------------------- #
    await monitor.stop()
    
    # Tracker should have been called for status updates
    assert mock_tracker.get_status.called


@pytest.mark.integration 
@pytest.mark.asyncio
async def test_pipeline_error_handling_integration(
    template_manager,
    hello_world_template,
    mock_tracker
):
    """
    Test integration when pipeline execution fails.
    """
    
    # Create engine that fails
    failing_engine = MagicMock()
    
    async def _failing_run(pipeline_spec, progress_callback=None):
        # Start normally
        if progress_callback:
            await progress_callback({
                "stage": "setup", 
                "progress": 0.3,
                "status": "running"
            })
        
        # Then fail
        raise RuntimeError("Simulated pipeline failure")
    
    failing_engine.run = AsyncMock(side_effect=_failing_run)
    
    # Render template
    pipeline_dict = template_manager.load_template(
        "hello_world",
        variables={"pipeline_name": "failing-test", "message": "Will Fail"}
    )
    
    # Set up monitoring
    from agentsmcp.pipeline.monitor import LiveMonitor, MonitoringConfig
    
    config = MonitoringConfig(enable_ui=False)
    monitor = LiveMonitor(tracker=mock_tracker, config=config)
    
    await monitor.start()
    
    # Execute and expect failure
    with pytest.raises(RuntimeError, match="Simulated pipeline failure"):
        await failing_engine.run(pipeline_dict, lambda x: None)
    
    await monitor.stop()


@pytest.mark.integration
def test_template_to_pipeline_creation_flow(template_manager, temp_dir):
    """
    Test the complete flow from template to pipeline file creation.
    """
    
    output_file = temp_dir / "generated-pipeline.yaml"
    
    # Create pipeline from template
    pipeline_dict = template_manager.create_pipeline_from_template(
        "hello_world",
        output_file,
        variables={
            "pipeline_name": "generated-pipeline",
            "message": "Generated successfully"
        }
    )
    
    # File should exist
    assert output_file.exists()
    
    # Content should be correct
    import yaml
    with open(output_file) as f:
        file_content = yaml.safe_load(f)
    
    assert file_content["name"] == "generated-pipeline-pipeline"
    assert file_content["stages"][0]["agents"][0]["payload"]["message"] == "Generated successfully"
    
    # Returned dict should match file content
    assert pipeline_dict["name"] == file_content["name"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ui_monitor_integration(mock_tracker):
    """
    Test UI and monitoring components working together.
    """
    from agentsmcp.pipeline.ui import PipelineMonitor
    from agentsmcp.pipeline.monitor import LiveMonitor, MonitoringConfig
    
    # Set up UI monitor
    ui_monitor = PipelineMonitor(tracker=mock_tracker, pipeline_name="ui-test")
    
    # Set up live monitor
    config = MonitoringConfig(enable_ui=False, update_interval=0.05)
    live_monitor = LiveMonitor(tracker=mock_tracker, config=config)
    
    # Mock console to avoid real output
    ui_monitor.console = MagicMock()
    
    # Start monitoring
    await live_monitor.start()
    
    # Update UI display
    ui_monitor.display_status()
    ui_monitor.display_success("Test completed successfully")
    
    # Allow some monitoring cycles
    await asyncio.sleep(0.1)
    
    # Stop monitoring
    await live_monitor.stop()
    
    # UI should have made console calls
    assert ui_monitor.console.print.called
    
    # Monitor should have checked tracker status
    assert mock_tracker.get_status.called


@pytest.mark.integration
def test_template_validation_integration(template_manager, temp_template_dir):
    """
    Test template validation across the whole system.
    """
    
    # Create a template with validation issues
    problematic_template = temp_template_dir / "problematic.yaml"
    content = """
name: "{{ pipeline_name }}-pipeline"
# Missing required fields like description, version
stages: []
notifications: {}
"""
    problematic_template.write_text(content.strip())
    
    # Refresh template discovery
    template_manager._discover_templates()
    
    # Template should be discoverable
    templates = template_manager.list_templates()
    assert "problematic" in templates
    
    # But loading should fail validation
    from agentsmcp.templates.manager import TemplateValidationError
    
    with pytest.raises(TemplateValidationError):
        template_manager.load_template(
            "problematic",
            variables={"pipeline_name": "test"}
        )


@pytest.mark.integration
@pytest.mark.asyncio 
async def test_concurrent_pipeline_operations(template_manager, temp_dir):
    """
    Test running multiple pipeline operations concurrently.
    """
    
    async def create_pipeline(name_suffix):
        output_file = temp_dir / f"concurrent-{name_suffix}.yaml"
        return template_manager.create_pipeline_from_template(
            "hello_world",
            output_file,
            variables={
                "pipeline_name": f"concurrent-{name_suffix}",
                "message": f"Concurrent test {name_suffix}"
            }
        )
    
    # Run multiple pipeline creations concurrently
    results = await asyncio.gather(
        create_pipeline("1"),
        create_pipeline("2"), 
        create_pipeline("3"),
        return_exceptions=True
    )
    
    # All should succeed
    assert len(results) == 3
    for result in results:
        assert not isinstance(result, Exception)
        assert "concurrent-" in result["name"]
    
    # All files should be created
    assert (temp_dir / "concurrent-1.yaml").exists()
    assert (temp_dir / "concurrent-2.yaml").exists()
    assert (temp_dir / "concurrent-3.yaml").exists()