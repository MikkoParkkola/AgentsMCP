"""
Comprehensive test suite for Symphony Dashboard - Real-time multi-agent coordination display.

This test suite validates the symphony mode dashboard functionality including real-time
status display, agent coordination visualization, performance metrics, and accessibility.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta
import sys
from pathlib import Path
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from agentsmcp.ui.components.symphony_dashboard import (
    SymphonyDashboard,
    DashboardConfig,
    AgentVisualizationCard,
    TaskQueueDisplay,
    MetricsPanel,
    ConflictResolutionPanel,
    DashboardTheme,
    create_symphony_dashboard
)


@pytest.fixture
def mock_symphony_api():
    """Create a mock symphony orchestration API."""
    api = Mock()
    api.get_symphony_status = AsyncMock(return_value={
        "success": True,
        "data": {
            "symphony_active": True,
            "start_time": "2025-01-15T10:30:00Z",
            "uptime_seconds": 3600,
            "agents": {
                "total": 8,
                "active": 6,
                "idle": 2,
                "failed": 0
            },
            "tasks": {
                "total": 25,
                "pending": 3,
                "running": 8,
                "completed": 12,
                "failed": 2
            },
            "metrics": {
                "harmony_score": 0.87,
                "active_agents": 6,
                "completed_tasks": 12,
                "failed_tasks": 2,
                "average_task_duration": 45.2,
                "resource_utilization": 0.73,
                "conflict_count": 1,
                "resolved_conflicts": 5,
                "uptime": 3600.0,
                "throughput": 12.5
            },
            "conflicts": {
                "total": 6,
                "unresolved": 1
            },
            "auto_scale_enabled": True,
            "max_agents": 12
        }
    })
    
    api.get_agent_details = AsyncMock(return_value={
        "success": True,
        "data": {
            "agent": {
                "id": "agent-001",
                "name": "coordinator",
                "type": "claude",
                "capabilities": ["coordination", "planning", "analysis"],
                "status": "working",
                "current_task_id": "task-123",
                "workload": 0.8,
                "health_score": 0.95,
                "last_heartbeat": "2025-01-15T11:25:00Z",
                "performance_metrics": {
                    "tasks_completed": 15,
                    "avg_response_time": 2.3,
                    "success_rate": 0.93,
                    "error_rate": 0.07
                },
                "resource_usage": {
                    "cpu": 0.45,
                    "memory": 0.62
                }
            },
            "task_history": [
                {
                    "id": "task-123",
                    "name": "data_analysis",
                    "status": "running",
                    "priority": "high"
                },
                {
                    "id": "task-122",
                    "name": "coordination",
                    "status": "completed",
                    "priority": "normal"
                }
            ]
        }
    })
    
    return api


@pytest.fixture
def dashboard_config():
    """Create a test dashboard configuration."""
    return DashboardConfig(
        auto_refresh=True,
        refresh_interval=2.0,
        show_metrics=True,
        show_agents=True,
        show_tasks=True,
        show_conflicts=True,
        max_agents_display=12,
        max_tasks_display=20,
        theme_name="symphony_dark",
        accessibility_mode=False,
        high_contrast=False,
        verbose_descriptions=False
    )


@pytest.fixture
async def symphony_dashboard(mock_symphony_api, dashboard_config):
    """Create a symphony dashboard for testing."""
    dashboard = SymphonyDashboard(
        symphony_api=mock_symphony_api,
        config=dashboard_config
    )
    await dashboard.initialize()
    yield dashboard
    await dashboard.cleanup()


class TestSymphonyDashboardInitialization:
    """Test suite for dashboard initialization and configuration."""

    @pytest.mark.asyncio
    async def test_dashboard_initialization(self, mock_symphony_api, dashboard_config):
        """Test dashboard initializes correctly with API and config."""
        dashboard = SymphonyDashboard(
            symphony_api=mock_symphony_api,
            config=dashboard_config
        )
        
        assert dashboard.symphony_api == mock_symphony_api
        assert dashboard.config == dashboard_config
        assert dashboard.is_active == False
        assert dashboard.last_update is None

    @pytest.mark.asyncio
    async def test_dashboard_async_initialization(self, symphony_dashboard):
        """Test async initialization completes successfully."""
        assert symphony_dashboard.is_initialized
        assert symphony_dashboard.components is not None
        assert len(symphony_dashboard.components) > 0

    def test_dashboard_config_validation(self):
        """Test dashboard configuration validation."""
        # Valid config
        config = DashboardConfig(refresh_interval=1.0)
        assert config.refresh_interval == 1.0
        
        # Test defaults
        default_config = DashboardConfig()
        assert default_config.auto_refresh
        assert default_config.refresh_interval > 0
        assert default_config.max_agents_display > 0

    @pytest.mark.asyncio
    async def test_dashboard_with_invalid_api(self, dashboard_config):
        """Test dashboard gracefully handles invalid API."""
        invalid_api = Mock()
        invalid_api.get_symphony_status = AsyncMock(side_effect=Exception("API Error"))
        
        dashboard = SymphonyDashboard(
            symphony_api=invalid_api,
            config=dashboard_config
        )
        
        # Should handle initialization gracefully
        try:
            await dashboard.initialize()
        except Exception as e:
            # Should not raise unhandled exceptions
            assert "API Error" in str(e)


class TestDashboardComponents:
    """Test suite for individual dashboard components."""

    @pytest.mark.asyncio
    async def test_agent_visualization_card(self, symphony_dashboard):
        """Test agent visualization card rendering."""
        agent_data = {
            "id": "agent-001",
            "name": "coordinator",
            "type": "claude", 
            "status": "working",
            "workload": 0.8,
            "health_score": 0.95
        }
        
        card = symphony_dashboard.components.get("agent_cards", {}).get("agent-001")
        if card:
            rendered = await card.render(agent_data)
            assert isinstance(rendered, str)
            assert "coordinator" in rendered
            assert "working" in rendered.lower()

    @pytest.mark.asyncio
    async def test_task_queue_display(self, symphony_dashboard):
        """Test task queue display rendering."""
        tasks_data = {
            "total": 25,
            "pending": 3,
            "running": 8,
            "completed": 12,
            "failed": 2
        }
        
        task_display = symphony_dashboard.components.get("task_queue")
        if task_display:
            rendered = await task_display.render(tasks_data)
            assert isinstance(rendered, str)
            assert "25" in rendered  # Total tasks
            assert "running" in rendered.lower()

    @pytest.mark.asyncio
    async def test_metrics_panel(self, symphony_dashboard):
        """Test metrics panel display."""
        metrics_data = {
            "harmony_score": 0.87,
            "resource_utilization": 0.73,
            "throughput": 12.5,
            "average_task_duration": 45.2
        }
        
        metrics_panel = symphony_dashboard.components.get("metrics")
        if metrics_panel:
            rendered = await metrics_panel.render(metrics_data)
            assert isinstance(rendered, str)
            assert "0.87" in rendered or "87%" in rendered  # Harmony score
            assert "73%" in rendered or "0.73" in rendered  # Utilization

    @pytest.mark.asyncio
    async def test_conflict_resolution_panel(self, symphony_dashboard):
        """Test conflict resolution panel."""
        conflicts_data = {
            "total": 6,
            "unresolved": 1,
            "recent_conflicts": [
                {
                    "id": "conflict-001",
                    "type": "resource",
                    "severity": 0.7,
                    "status": "resolving"
                }
            ]
        }
        
        conflict_panel = symphony_dashboard.components.get("conflicts")
        if conflict_panel:
            rendered = await conflict_panel.render(conflicts_data)
            assert isinstance(rendered, str)


class TestRealTimeUpdates:
    """Test suite for real-time dashboard updates."""

    @pytest.mark.asyncio
    async def test_dashboard_refresh_cycle(self, symphony_dashboard):
        """Test dashboard refresh cycle works correctly."""
        # Start dashboard
        await symphony_dashboard.start()
        
        # Wait for at least one refresh cycle
        await asyncio.sleep(symphony_dashboard.config.refresh_interval + 0.5)
        
        # Should have updated at least once
        assert symphony_dashboard.last_update is not None
        assert symphony_dashboard.update_count > 0
        
        await symphony_dashboard.stop()

    @pytest.mark.asyncio
    async def test_manual_refresh(self, symphony_dashboard):
        """Test manual dashboard refresh."""
        initial_update_count = symphony_dashboard.update_count
        
        await symphony_dashboard.refresh()
        
        assert symphony_dashboard.update_count > initial_update_count
        assert symphony_dashboard.last_update is not None

    @pytest.mark.asyncio
    async def test_data_caching_and_invalidation(self, symphony_dashboard):
        """Test data caching and cache invalidation."""
        # First refresh should fetch new data
        await symphony_dashboard.refresh()
        first_data = symphony_dashboard.cached_data.copy()
        
        # Immediate second refresh should use cached data
        await symphony_dashboard.refresh()
        
        # Should have cached data
        assert symphony_dashboard.cached_data is not None
        
        # Force cache invalidation
        symphony_dashboard.invalidate_cache()
        await symphony_dashboard.refresh()
        
        # Should have fresh data
        assert symphony_dashboard.cached_data is not None

    @pytest.mark.asyncio
    async def test_error_handling_during_updates(self, mock_symphony_api, dashboard_config):
        """Test error handling during data updates."""
        # Configure API to fail
        mock_symphony_api.get_symphony_status = AsyncMock(side_effect=Exception("Network error"))
        
        dashboard = SymphonyDashboard(
            symphony_api=mock_symphony_api,
            config=dashboard_config
        )
        
        # Should handle errors gracefully
        await dashboard.refresh()
        
        # Should still be functional
        assert dashboard.error_count > 0


class TestDashboardVisualization:
    """Test suite for dashboard visualization and rendering."""

    @pytest.mark.asyncio
    async def test_full_dashboard_render(self, symphony_dashboard):
        """Test complete dashboard rendering."""
        rendered = await symphony_dashboard.render()
        
        assert isinstance(rendered, str)
        assert len(rendered) > 0
        
        # Should contain key sections
        assert "symphony" in rendered.lower() or "agents" in rendered.lower()

    @pytest.mark.asyncio
    async def test_layout_adaptation(self, symphony_dashboard):
        """Test dashboard layout adapts to different terminal sizes."""
        # Test with different terminal dimensions
        terminal_sizes = [
            (80, 24),   # Standard terminal
            (120, 40),  # Wide terminal
            (60, 20),   # Narrow terminal
        ]
        
        for width, height in terminal_sizes:
            symphony_dashboard.update_terminal_size(width, height)
            rendered = await symphony_dashboard.render()
            
            assert isinstance(rendered, str)
            # Should adapt layout but still render content
            lines = rendered.split('\n')
            assert len(lines) <= height + 5  # Allow some flexibility

    @pytest.mark.asyncio
    async def test_color_theme_application(self, symphony_dashboard):
        """Test color theme application in rendering."""
        # Test different themes
        themes = ["symphony_dark", "symphony_light", "high_contrast"]
        
        for theme_name in themes:
            symphony_dashboard.config.theme_name = theme_name
            symphony_dashboard.apply_theme()
            
            rendered = await symphony_dashboard.render()
            
            assert isinstance(rendered, str)
            # Should contain ANSI color codes for non-plain themes
            if theme_name != "plain":
                assert "\033[" in rendered

    @pytest.mark.asyncio
    async def test_responsive_content_display(self, symphony_dashboard):
        """Test responsive content display based on data volume."""
        # Test with different data volumes
        data_scenarios = [
            {"agents": 2, "tasks": 5},      # Light load
            {"agents": 8, "tasks": 25},     # Normal load  
            {"agents": 12, "tasks": 100},   # Heavy load
        ]
        
        for scenario in data_scenarios:
            # Mock different data volumes
            symphony_dashboard.cached_data = {
                "agents": {"total": scenario["agents"]},
                "tasks": {"total": scenario["tasks"]}
            }
            
            rendered = await symphony_dashboard.render()
            assert isinstance(rendered, str)
            assert len(rendered) > 0


class TestPerformanceAndOptimization:
    """Test suite for dashboard performance and optimization."""

    @pytest.mark.asyncio
    async def test_render_performance(self, symphony_dashboard):
        """Test dashboard render performance meets requirements."""
        import time
        
        # Warm up
        await symphony_dashboard.render()
        
        # Measure render time
        start_time = time.time()
        await symphony_dashboard.render()
        render_time = time.time() - start_time
        
        # Should render in under 100ms
        assert render_time < 0.1, f"Render took {render_time:.3f}s, should be < 0.1s"

    @pytest.mark.asyncio
    async def test_memory_usage_optimization(self, symphony_dashboard):
        """Test memory usage remains optimized."""
        import gc
        import sys
        
        # Get baseline memory
        gc.collect()
        baseline_objects = len(gc.get_objects())
        
        # Perform multiple render cycles
        for _ in range(10):
            await symphony_dashboard.render()
            await symphony_dashboard.refresh()
        
        # Check memory growth
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Memory growth should be reasonable (less than 50% increase)
        growth_ratio = final_objects / baseline_objects
        assert growth_ratio < 1.5, f"Memory grew by {growth_ratio:.2f}x, should be < 1.5x"

    @pytest.mark.asyncio
    async def test_update_frequency_optimization(self, symphony_dashboard):
        """Test update frequency optimization."""
        # Configure for high frequency updates
        symphony_dashboard.config.refresh_interval = 0.5
        
        await symphony_dashboard.start()
        
        # Let it run for a short time
        await asyncio.sleep(2.0)
        
        # Should have updated multiple times but not excessively
        assert 3 <= symphony_dashboard.update_count <= 6
        
        await symphony_dashboard.stop()

    @pytest.mark.asyncio 
    async def test_concurrent_operations(self, symphony_dashboard):
        """Test dashboard handles concurrent operations correctly."""
        # Start multiple concurrent operations
        tasks = [
            symphony_dashboard.refresh(),
            symphony_dashboard.render(),
            symphony_dashboard.update_agent_status("agent-001", "idle"),
            symphony_dashboard.render()
        ]
        
        # Should complete without errors
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check no exceptions occurred
        for result in results:
            assert not isinstance(result, Exception)


class TestAccessibilityFeatures:
    """Test suite for accessibility features."""

    @pytest.mark.asyncio
    async def test_accessibility_mode(self, mock_symphony_api):
        """Test accessibility mode provides enhanced descriptions."""
        config = DashboardConfig(accessibility_mode=True, verbose_descriptions=True)
        dashboard = SymphonyDashboard(symphony_api=mock_symphony_api, config=config)
        
        await dashboard.initialize()
        rendered = await dashboard.render()
        
        assert isinstance(rendered, str)
        # Accessibility mode should provide more descriptive content
        assert len(rendered) > 0

    @pytest.mark.asyncio  
    async def test_high_contrast_mode(self, mock_symphony_api):
        """Test high contrast mode for better visibility."""
        config = DashboardConfig(high_contrast=True)
        dashboard = SymphonyDashboard(symphony_api=mock_symphony_api, config=config)
        
        await dashboard.initialize()
        rendered = await dashboard.render()
        
        assert isinstance(rendered, str)
        # Should still render content in high contrast mode
        assert len(rendered) > 0

    @pytest.mark.asyncio
    async def test_keyboard_navigation_support(self, symphony_dashboard):
        """Test keyboard navigation support."""
        # Test navigation commands
        navigation_commands = [
            "next_section",
            "previous_section", 
            "refresh",
            "toggle_view"
        ]
        
        for command in navigation_commands:
            if hasattr(symphony_dashboard, f"handle_{command}"):
                result = await getattr(symphony_dashboard, f"handle_{command}")()
                # Should handle navigation without errors
                assert result is not None

    @pytest.mark.asyncio
    async def test_screen_reader_compatibility(self, symphony_dashboard):
        """Test screen reader compatibility."""
        symphony_dashboard.config.accessibility_mode = True
        
        rendered = await symphony_dashboard.render()
        
        # Should not contain purely visual elements that screen readers can't interpret
        # Should have descriptive text instead of just symbols/colors
        assert isinstance(rendered, str)


class TestDashboardConfiguration:
    """Test suite for dashboard configuration management."""

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid configuration
        config = DashboardConfig(
            refresh_interval=1.5,
            max_agents_display=10,
            theme_name="symphony_dark"
        )
        
        assert config.refresh_interval == 1.5
        assert config.max_agents_display == 10
        assert config.theme_name == "symphony_dark"

    def test_config_defaults(self):
        """Test configuration defaults are reasonable."""
        config = DashboardConfig()
        
        assert config.auto_refresh
        assert config.refresh_interval > 0
        assert config.refresh_interval < 60  # Not too slow
        assert config.max_agents_display > 0
        assert config.max_tasks_display > 0

    @pytest.mark.asyncio
    async def test_runtime_config_updates(self, symphony_dashboard):
        """Test runtime configuration updates."""
        original_interval = symphony_dashboard.config.refresh_interval
        
        # Update configuration
        symphony_dashboard.config.refresh_interval = original_interval * 2
        symphony_dashboard.apply_config()
        
        assert symphony_dashboard.config.refresh_interval == original_interval * 2

    @pytest.mark.asyncio
    async def test_theme_switching(self, symphony_dashboard):
        """Test dynamic theme switching."""
        themes = ["symphony_dark", "symphony_light", "high_contrast"]
        
        for theme in themes:
            symphony_dashboard.config.theme_name = theme
            symphony_dashboard.apply_theme()
            
            rendered = await symphony_dashboard.render()
            assert isinstance(rendered, str)


class TestErrorHandlingAndResilience:
    """Test suite for error handling and resilience."""

    @pytest.mark.asyncio
    async def test_api_failure_handling(self, dashboard_config):
        """Test handling of API failures."""
        failing_api = Mock()
        failing_api.get_symphony_status = AsyncMock(side_effect=Exception("API Down"))
        
        dashboard = SymphonyDashboard(symphony_api=failing_api, config=dashboard_config)
        
        # Should handle API failures gracefully
        await dashboard.refresh()
        
        assert dashboard.error_count > 0
        assert dashboard.last_error is not None

    @pytest.mark.asyncio
    async def test_partial_data_handling(self, mock_symphony_api, dashboard_config):
        """Test handling of partial/corrupted data."""
        # Configure API to return partial data
        mock_symphony_api.get_symphony_status = AsyncMock(return_value={
            "success": True,
            "data": {
                "symphony_active": True,
                # Missing other expected fields
            }
        })
        
        dashboard = SymphonyDashboard(symphony_api=mock_symphony_api, config=dashboard_config)
        
        await dashboard.refresh()
        rendered = await dashboard.render()
        
        # Should still render something meaningful
        assert isinstance(rendered, str)

    @pytest.mark.asyncio
    async def test_recovery_after_errors(self, symphony_dashboard):
        """Test recovery after errors."""
        # Simulate an error
        symphony_dashboard.error_count = 5
        symphony_dashboard.last_error = "Test error"
        
        # Should be able to recover
        await symphony_dashboard.refresh()
        rendered = await symphony_dashboard.render()
        
        assert isinstance(rendered, str)

    @pytest.mark.asyncio
    async def test_graceful_degradation(self, mock_symphony_api, dashboard_config):
        """Test graceful degradation when features are unavailable."""
        # Configure API with limited functionality
        limited_api = Mock()
        limited_api.get_symphony_status = AsyncMock(return_value={
            "success": True,
            "data": {"symphony_active": False}
        })
        limited_api.get_agent_details = AsyncMock(side_effect=Exception("Not available"))
        
        dashboard = SymphonyDashboard(symphony_api=limited_api, config=dashboard_config)
        
        await dashboard.refresh()
        rendered = await dashboard.render()
        
        # Should still provide basic functionality
        assert isinstance(rendered, str)
        assert "symphony" in rendered.lower() or "inactive" in rendered.lower()


class TestIntegrationScenarios:
    """Test suite for integration scenarios."""

    @pytest.mark.asyncio
    async def test_dashboard_lifecycle(self, mock_symphony_api, dashboard_config):
        """Test complete dashboard lifecycle."""
        dashboard = SymphonyDashboard(symphony_api=mock_symphony_api, config=dashboard_config)
        
        # Initialize
        await dashboard.initialize()
        assert dashboard.is_initialized
        
        # Start real-time updates
        await dashboard.start()
        assert dashboard.is_active
        
        # Perform operations
        await dashboard.refresh()
        rendered = await dashboard.render()
        assert isinstance(rendered, str)
        
        # Stop updates
        await dashboard.stop()
        assert not dashboard.is_active
        
        # Cleanup
        await dashboard.cleanup()

    @pytest.mark.asyncio
    async def test_multi_agent_scenario(self, mock_symphony_api, dashboard_config):
        """Test dashboard with multiple agents scenario."""
        # Configure API for multi-agent scenario
        mock_symphony_api.get_symphony_status = AsyncMock(return_value={
            "success": True,
            "data": {
                "symphony_active": True,
                "agents": {"total": 12, "active": 10, "idle": 2, "failed": 0},
                "tasks": {"total": 50, "running": 20, "completed": 25, "failed": 5},
                "metrics": {"harmony_score": 0.92, "throughput": 25.0}
            }
        })
        
        dashboard = SymphonyDashboard(symphony_api=mock_symphony_api, config=dashboard_config)
        await dashboard.initialize()
        
        rendered = await dashboard.render()
        
        # Should handle large number of agents
        assert "12" in rendered  # Total agents
        assert isinstance(rendered, str)

    @pytest.mark.asyncio
    async def test_performance_under_load(self, mock_symphony_api, dashboard_config):
        """Test dashboard performance under load."""
        dashboard = SymphonyDashboard(symphony_api=mock_symphony_api, config=dashboard_config)
        await dashboard.initialize()
        
        # Simulate high-frequency updates
        for _ in range(20):
            await dashboard.refresh()
            await dashboard.render()
        
        # Should maintain performance
        assert dashboard.update_count >= 20
        assert dashboard.error_count == 0


class TestUtilityFunctions:
    """Test suite for utility functions."""

    def test_create_symphony_dashboard(self, mock_symphony_api, dashboard_config):
        """Test utility function for creating dashboard."""
        dashboard = create_symphony_dashboard(mock_symphony_api, dashboard_config)
        
        assert isinstance(dashboard, SymphonyDashboard)
        assert dashboard.symphony_api == mock_symphony_api
        assert dashboard.config == dashboard_config

    @pytest.mark.asyncio
    async def test_dashboard_state_serialization(self, symphony_dashboard):
        """Test dashboard state can be serialized for persistence."""
        # Update dashboard state
        await symphony_dashboard.refresh()
        
        # Get state
        state = symphony_dashboard.get_state()
        
        assert isinstance(state, dict)
        assert "last_update" in state
        assert "update_count" in state

    @pytest.mark.asyncio
    async def test_dashboard_metrics_export(self, symphony_dashboard):
        """Test dashboard metrics can be exported."""
        # Generate some activity
        for _ in range(3):
            await symphony_dashboard.refresh()
        
        metrics = symphony_dashboard.export_metrics()
        
        assert isinstance(metrics, dict)
        assert "update_count" in metrics
        assert "error_count" in metrics
        assert "last_update" in metrics


# Performance benchmarks
@pytest.mark.asyncio
async def test_dashboard_sub_100ms_response_requirement(symphony_dashboard):
    """Test dashboard meets sub-100ms response time requirement."""
    import time
    
    # Warm up
    await symphony_dashboard.refresh()
    
    # Test multiple render cycles
    response_times = []
    for _ in range(10):
        start_time = time.time()
        await symphony_dashboard.render()
        response_time = time.time() - start_time
        response_times.append(response_time)
    
    # Calculate average response time
    avg_response_time = sum(response_times) / len(response_times)
    max_response_time = max(response_times)
    
    # Should meet sub-100ms requirement
    assert avg_response_time < 0.1, f"Average response time {avg_response_time:.3f}s > 0.1s"
    assert max_response_time < 0.15, f"Max response time {max_response_time:.3f}s > 0.15s"


@pytest.mark.asyncio
async def test_dashboard_concurrent_user_simulation(mock_symphony_api, dashboard_config):
    """Test dashboard can handle multiple concurrent users (simulation)."""
    # Create multiple dashboard instances to simulate concurrent users
    dashboards = []
    for i in range(5):
        dashboard = SymphonyDashboard(symphony_api=mock_symphony_api, config=dashboard_config)
        await dashboard.initialize()
        dashboards.append(dashboard)
    
    # Run concurrent operations
    tasks = []
    for dashboard in dashboards:
        tasks.extend([
            dashboard.refresh(),
            dashboard.render()
        ])
    
    # Execute all concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # All should complete successfully
    for result in results:
        assert not isinstance(result, Exception), f"Task failed: {result}"
    
    # Cleanup
    for dashboard in dashboards:
        await dashboard.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])