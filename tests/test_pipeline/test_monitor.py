# tests/test_pipeline/test_monitor.py
"""
Async tests for the PipelineMonitor (lifecycle, snapshot JSON, thread‑safety).
"""

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# --------------------------------------------------------------------------- #
#  Async fixture that creates a mock status source.
# --------------------------------------------------------------------------- #
@pytest.fixture
def mock_status_source():
    """
    Returns an async callable that yields a synthetic status dict.
    """
    async def _status():
        await asyncio.sleep(0.01)  # simulate I/O latency
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "state": "running",
            "progress": 42,
            "current_stage": "build"
        }

    return AsyncMock(side_effect=_status)


@pytest.fixture
def mock_tracker():
    """Create a mock ExecutionTracker."""
    tracker = MagicMock()
    tracker.get_status.return_value = {
        "pipeline_name": "test-pipeline",
        "current_stage": "build", 
        "progress": 0.5,
        "status": "running",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    return tracker


# --------------------------------------------------------------------------- #
#  Helper – build a fresh monitor.
# --------------------------------------------------------------------------- #
@pytest.fixture
def live_monitor(mock_tracker):
    from agentsmcp.pipeline.monitor import LiveMonitor, MonitoringConfig
    
    config = MonitoringConfig(
        enable_ui=True,
        json_output=False,
        update_interval=0.05
    )
    
    monitor = LiveMonitor(
        tracker=mock_tracker,
        config=config
    )
    return monitor


# --------------------------------------------------------------------------- #
#  1️⃣ Lifecycle – start, collect a few snapshots, stop.
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_live_monitor_lifecycle(live_monitor, mock_tracker):
    # Start monitoring
    await live_monitor.start()
    
    # Allow a couple of update cycles
    await asyncio.sleep(0.2)
    
    # Stop monitoring
    await live_monitor.stop()
    
    # The tracker should have been called for status updates
    assert mock_tracker.get_status.called


@pytest.mark.asyncio
async def test_live_monitor_with_json_output(mock_tracker):
    from agentsmcp.pipeline.monitor import LiveMonitor, MonitoringConfig
    
    config = MonitoringConfig(
        enable_ui=False,
        json_output=True,
        update_interval=0.05
    )
    
    monitor = LiveMonitor(tracker=mock_tracker, config=config)
    
    await monitor.start()
    await asyncio.sleep(0.1)  # Let it collect some data
    await monitor.stop()
    
    # Should have tracked status updates
    assert mock_tracker.get_status.called


# --------------------------------------------------------------------------- #
#  2️⃣ Status collection and snapshots
# --------------------------------------------------------------------------- #
def test_status_collector():
    from agentsmcp.pipeline.monitor import StatusCollector
    
    collector = StatusCollector()
    
    # Add some status snapshots
    status1 = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "state": "running",
        "progress": 0.3
    }
    status2 = {
        "timestamp": datetime.now(timezone.utc).isoformat(), 
        "state": "running",
        "progress": 0.7
    }
    
    collector.add_snapshot(status1)
    collector.add_snapshot(status2)
    
    snapshots = collector.get_snapshots()
    assert len(snapshots) == 2
    assert snapshots[0]["progress"] == 0.3
    assert snapshots[1]["progress"] == 0.7


def test_status_snapshot_json_serialization():
    from agentsmcp.pipeline.monitor import StatusSnapshot
    
    snapshot = StatusSnapshot(
        timestamp=datetime.now(timezone.utc),
        pipeline_name="test-pipeline",
        current_stage="build",
        status="running",
        progress=0.5,
        agents_status=[
            {"name": "agent1", "status": "completed"},
            {"name": "agent2", "status": "running"}
        ]
    )
    
    # Should serialize to JSON
    json_str = snapshot.to_json()
    data = json.loads(json_str)
    
    assert data["pipeline_name"] == "test-pipeline"
    assert data["current_stage"] == "build"
    assert data["progress"] == 0.5
    assert len(data["agents_status"]) == 2


# --------------------------------------------------------------------------- #
#  3️⃣ Monitoring configuration
# --------------------------------------------------------------------------- #
def test_monitoring_config():
    from agentsmcp.pipeline.monitor import MonitoringConfig
    
    # Test default config
    config = MonitoringConfig()
    assert config.enable_ui is True
    assert config.json_output is False
    assert config.update_interval > 0
    
    # Test custom config
    custom_config = MonitoringConfig(
        enable_ui=False,
        json_output=True,
        update_interval=2.0
    )
    assert custom_config.enable_ui is False
    assert custom_config.json_output is True
    assert custom_config.update_interval == 2.0


# --------------------------------------------------------------------------- #
#  4️⃣ Error handling in monitoring
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_live_monitor_handles_tracker_errors(mock_tracker):
    from agentsmcp.pipeline.monitor import LiveMonitor, MonitoringConfig
    
    # Make tracker raise an exception
    mock_tracker.get_status.side_effect = Exception("Tracker error")
    
    config = MonitoringConfig(enable_ui=False, update_interval=0.05)
    monitor = LiveMonitor(tracker=mock_tracker, config=config)
    
    # Should not raise, should handle the error gracefully
    await monitor.start()
    await asyncio.sleep(0.1)
    await monitor.stop()
    
    # The monitor should have attempted to get status
    assert mock_tracker.get_status.called


# --------------------------------------------------------------------------- #
#  5️⃣ Create monitor factory function
# --------------------------------------------------------------------------- #
def test_create_monitor_factory(mock_tracker):
    from agentsmcp.pipeline.monitor import create_monitor, MonitoringConfig
    
    config = MonitoringConfig(enable_ui=True, json_output=False)
    monitor = create_monitor(tracker=mock_tracker, config=config)
    
    assert monitor is not None
    assert monitor.tracker is mock_tracker
    assert monitor.config.enable_ui is True


# --------------------------------------------------------------------------- #
#  6️⃣ Thread safety considerations
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_concurrent_status_updates():
    from agentsmcp.pipeline.monitor import StatusCollector
    
    collector = StatusCollector()
    
    async def add_snapshots(start_idx):
        for i in range(5):
            snapshot = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "state": "running",
                "progress": (start_idx * 5 + i) / 100.0
            }
            collector.add_snapshot(snapshot)
            await asyncio.sleep(0.01)
    
    # Run multiple concurrent tasks
    await asyncio.gather(
        add_snapshots(1),
        add_snapshots(2), 
        add_snapshots(3)
    )
    
    # Should have collected all snapshots
    snapshots = collector.get_snapshots()
    assert len(snapshots) == 15  # 3 tasks * 5 snapshots each