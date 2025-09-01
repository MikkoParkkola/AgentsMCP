"""
Test suite for monitoring system integration with TUI components.

This test suite verifies that:
1. Orchestrator properly emits events to monitoring components
2. TUI components can access and display monitoring data
3. Monitoring panels update in real-time
4. Performance metrics are collected accurately
5. Agent tracking works correctly
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import dataclass
from typing import Dict, Any

# Import monitoring components
from agentsmcp.monitoring import MetricsCollector, AgentTracker, PerformanceMonitor
from agentsmcp.orchestration.orchestrator import Orchestrator, OrchestratorConfig
from agentsmcp.ui.v2.orchestrator_integration import OrchestratorTUIIntegration
from agentsmcp.ui.v2.layouts.enhanced_layout import EnhancedLayout, EnhancedLayoutConfig
from agentsmcp.ui.v2.components.agent_status_panel import AgentStatusPanel
from agentsmcp.ui.v2.components.metrics_dashboard import MetricsDashboard
from agentsmcp.ui.v2.components.activity_feed import ActivityFeed
from agentsmcp.ui.v2.components.progress_visualizer import DependencyGraph


class TestMonitoringIntegration:
    """Test monitoring system integration."""
    
    @pytest.fixture
    async def metrics_collector(self):
        """Fixture for metrics collector."""
        collector = MetricsCollector()
        yield collector
        await collector.cleanup()
    
    @pytest.fixture
    async def agent_tracker(self):
        """Fixture for agent tracker."""
        tracker = AgentTracker()
        yield tracker
        await tracker.cleanup()
    
    @pytest.fixture
    async def performance_monitor(self):
        """Fixture for performance monitor."""
        monitor = PerformanceMonitor()
        yield monitor
        await monitor.cleanup()
    
    @pytest.fixture
    async def orchestrator(self, metrics_collector, agent_tracker, performance_monitor):
        """Fixture for orchestrator with monitoring."""
        config = OrchestratorConfig(
            enable_intelligent_delegation=False,  # Simplify for tests
            enable_self_improvement=False
        )
        orchestrator = Orchestrator(config)
        
        # Replace monitoring components with test fixtures
        orchestrator.metrics_collector = metrics_collector
        orchestrator.agent_tracker = agent_tracker
        orchestrator.performance_monitor = performance_monitor
        
        yield orchestrator
        await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_orchestrator_emits_metrics(self, orchestrator):
        """Test that orchestrator emits metrics during operation."""
        # Process a test input
        response = await orchestrator.process_user_input("Hello, how can you help me?")
        
        # Verify metrics were recorded
        metrics = orchestrator.metrics_collector.get_all_metrics()
        assert "orchestrator.requests.total" in metrics
        assert metrics["orchestrator.requests.total"]["value"] >= 1
        
        # Verify response was successful
        assert response.response_type in ["normal", "fallback"]
        assert response.processing_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_agent_tracking_during_delegation(self, orchestrator):
        """Test that agent tracking works during delegation."""
        # Mock agent delegation to simulate real behavior
        original_delegate = orchestrator._delegate_to_agent
        
        async def mock_delegate(agent_id: str, user_input: str, context: Dict) -> str:
            # Simulate tracking calls that would happen in real delegation
            orchestrator.agent_tracker.register_agent(agent_id, "active", {"type": "test_agent"})
            
            task_info = {
                "description": user_input[:50],
                "type": "test_task",
                "priority": "normal"
            }
            task_id = await orchestrator.agent_tracker.start_task(agent_id, task_info)
            
            # Simulate work
            await asyncio.sleep(0.01)
            
            await orchestrator.agent_tracker.complete_task(agent_id, task_id, "completed")
            return f"Mock response from {agent_id}"
        
        orchestrator._delegate_to_agent = mock_delegate
        
        # Process input that requires agent delegation
        response = await orchestrator.process_user_input("Please write a Python function to calculate fibonacci numbers")
        
        # Verify agent tracking occurred
        agents = orchestrator.agent_tracker.get_all_agents()
        assert len(agents) > 0
        
        agent_id = list(agents.keys())[0]
        agent_info = agents[agent_id]
        assert agent_info["status"] in ["active", "idle"]
        
        # Verify task completion
        tasks = orchestrator.agent_tracker.get_agent_tasks(agent_id)
        assert len(tasks) > 0
        assert any(task["status"] == "completed" for task in tasks)
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, orchestrator):
        """Test that performance monitoring captures timing data."""
        # Process multiple requests to generate timing data
        for i in range(3):
            await orchestrator.process_user_input(f"Test request {i}")
            await asyncio.sleep(0.01)  # Small delay between requests
        
        # Verify performance data was collected
        performance_data = orchestrator.performance_monitor.get_summary()
        
        assert "orchestrator.process_user_input" in performance_data
        timing_data = performance_data["orchestrator.process_user_input"]
        
        assert timing_data["count"] >= 3
        assert timing_data["avg_ms"] > 0
        assert timing_data["p95_ms"] > 0
    
    @pytest.mark.asyncio
    async def test_tui_integration_monitoring_access(self):
        """Test that TUI integration can access monitoring components."""
        # Create TUI integration
        tui_integration = OrchestratorTUIIntegration()
        
        # Mock orchestrated conversation with monitoring
        mock_conversation = Mock()
        mock_orchestrator = Mock()
        mock_orchestrator.get_monitoring_components.return_value = {
            "metrics_collector": Mock(spec=MetricsCollector),
            "agent_tracker": Mock(spec=AgentTracker),
            "performance_monitor": Mock(spec=PerformanceMonitor)
        }
        mock_conversation.orchestrator = mock_orchestrator
        tui_integration.orchestrated_conversation = mock_conversation
        
        # Initialize monitoring components
        await tui_integration._initialize_monitoring_components()
        
        # Verify monitoring components are accessible
        assert tui_integration.has_monitoring_support()
        
        components = tui_integration.get_monitoring_components()
        assert components is not None
        assert "metrics_collector" in components
        assert "agent_tracker" in components
        assert "performance_monitor" in components
    
    def test_enhanced_layout_configuration(self):
        """Test enhanced layout with monitoring components."""
        # Create monitoring components
        metrics_collector = MetricsCollector()
        agent_tracker = AgentTracker()
        performance_monitor = PerformanceMonitor()
        
        # Create enhanced layout config
        config = EnhancedLayoutConfig(
            layout_mode="MONITORING",
            show_agent_status=True,
            show_metrics_dashboard=True,
            show_activity_feed=True,
            show_progress_visualization=True
        )
        
        # Create enhanced layout
        layout = EnhancedLayout(
            config=config,
            metrics_collector=metrics_collector,
            agent_tracker=agent_tracker,
            performance_monitor=performance_monitor
        )
        
        # Verify layout configuration
        layout_info = layout.get_layout_info()
        assert layout_info["type"] == "enhanced_layout"
        assert layout_info["config"].layout_mode == "MONITORING"
        assert layout_info["panels"]["agent_status"] is True
        assert layout_info["panels"]["metrics_dashboard"] is True
    
    @pytest.mark.asyncio
    async def test_agent_status_panel_updates(self, agent_tracker):
        """Test that agent status panel receives updates."""
        # Create agent status panel
        panel = AgentStatusPanel(
            agent_tracker=agent_tracker,
            update_interval=0.01,  # Fast updates for testing
            compact_mode=False
        )
        
        # Register test agents
        await agent_tracker.register_agent("test_agent_1", "active", {"type": "backend"})
        await agent_tracker.register_agent("test_agent_2", "idle", {"type": "frontend"})
        
        # Start some tasks
        task1_info = {"description": "Test task 1", "type": "coding", "priority": "high"}
        task1_id = await agent_tracker.start_task("test_agent_1", task1_info)
        
        task2_info = {"description": "Test task 2", "type": "review", "priority": "normal"}
        task2_id = await agent_tracker.start_task("test_agent_2", task2_info)
        
        # Simulate panel refresh
        await panel.refresh_status()
        
        # Verify panel has agent data
        agent_data = panel.get_agent_display_data()
        assert len(agent_data) == 2
        
        agent_1_data = next(a for a in agent_data if a["id"] == "test_agent_1")
        assert agent_1_data["status"] == "active"
        assert len(agent_1_data["active_tasks"]) == 1
        
        # Complete a task and refresh
        await agent_tracker.complete_task("test_agent_1", task1_id, "completed")
        await panel.refresh_status()
        
        # Verify task completion reflected in panel
        updated_data = panel.get_agent_display_data()
        agent_1_updated = next(a for a in updated_data if a["id"] == "test_agent_1")
        assert len(agent_1_updated["active_tasks"]) == 0
    
    @pytest.mark.asyncio
    async def test_metrics_dashboard_data_collection(self, metrics_collector, performance_monitor):
        """Test that metrics dashboard collects and displays data."""
        # Generate test metrics
        for i in range(10):
            metrics_collector.record_counter("test.requests")
            metrics_collector.record_histogram("test.response_time", i * 10)
            metrics_collector.record_gauge("test.active_connections", i)
        
        # Record performance data
        for i in range(5):
            timer = performance_monitor.start_timer("test.operation")
            await asyncio.sleep(0.001)  # Simulate work
            performance_monitor.end_timer(timer)
        
        # Create metrics dashboard
        dashboard = MetricsDashboard(
            metrics_collector=metrics_collector,
            performance_monitor=performance_monitor,
            update_interval=0.01,
            show_alerts=True,
            show_historical=True
        )
        
        # Refresh dashboard
        await dashboard.refresh_metrics()
        
        # Verify dashboard has data
        dashboard_data = dashboard.get_metrics_summary()
        
        assert "counters" in dashboard_data
        assert "histograms" in dashboard_data
        assert "gauges" in dashboard_data
        assert "performance" in dashboard_data
        
        # Verify specific metrics
        assert dashboard_data["counters"]["test.requests"]["value"] == 10
        assert dashboard_data["gauges"]["test.active_connections"]["value"] == 9  # Last recorded value
        assert dashboard_data["performance"]["test.operation"]["count"] == 5
    
    @pytest.mark.asyncio
    async def test_activity_feed_event_tracking(self, agent_tracker):
        """Test that activity feed tracks events correctly."""
        # Create activity feed
        feed = ActivityFeed(
            agent_tracker=agent_tracker,
            max_items=100,
            show_timestamps=True,
            show_severity=True,
            enable_search=True
        )
        
        # Generate test events
        await agent_tracker.register_agent("test_agent", "active", {"type": "test"})
        
        task_info = {"description": "Test task", "type": "test", "priority": "normal"}
        task_id = await agent_tracker.start_task("test_agent", task_info)
        
        await agent_tracker.update_agent_status("test_agent", "working", {"progress": 50})
        await agent_tracker.complete_task("test_agent", task_id, "completed")
        
        # Refresh feed
        await feed.refresh_feed()
        
        # Verify events were captured
        events = feed.get_recent_events()
        assert len(events) >= 3  # Registration, task start, status update, task complete
        
        # Verify event types
        event_types = [event["type"] for event in events]
        assert "agent_registration" in event_types
        assert "task_start" in event_types
        assert "task_complete" in event_types
    
    @pytest.mark.asyncio
    async def test_dependency_graph_visualization(self, agent_tracker):
        """Test dependency graph visualization with agent relationships."""
        # Create dependency graph
        graph = DependencyGraph(
            agent_tracker=agent_tracker,
            update_interval=0.01,
            max_nodes=20
        )
        
        # Create agents with dependencies
        await agent_tracker.register_agent("backend_agent", "active", {"type": "backend"})
        await agent_tracker.register_agent("frontend_agent", "active", {"type": "frontend"})
        await agent_tracker.register_agent("qa_agent", "active", {"type": "qa"})
        
        # Create tasks with dependencies
        backend_task = await agent_tracker.start_task("backend_agent", {
            "description": "Create API",
            "type": "implementation",
            "priority": "high"
        })
        
        frontend_task = await agent_tracker.start_task("frontend_agent", {
            "description": "Create UI",
            "type": "implementation", 
            "priority": "high",
            "dependencies": [backend_task]  # Depends on backend
        })
        
        qa_task = await agent_tracker.start_task("qa_agent", {
            "description": "Test integration",
            "type": "testing",
            "priority": "normal",
            "dependencies": [backend_task, frontend_task]  # Depends on both
        })
        
        # Refresh graph
        await graph.refresh_graph()
        
        # Verify graph structure
        graph_data = graph.get_graph_data()
        
        assert len(graph_data["nodes"]) >= 3  # At least 3 agents
        assert len(graph_data["edges"]) >= 2  # Dependencies between tasks
        
        # Verify dependency relationships
        node_ids = [node["id"] for node in graph_data["nodes"]]
        assert "backend_agent" in node_ids
        assert "frontend_agent" in node_ids
        assert "qa_agent" in node_ids
    
    @pytest.mark.asyncio
    async def test_end_to_end_monitoring_flow(self):
        """Test complete end-to-end monitoring flow."""
        # Create orchestrator with monitoring
        config = OrchestratorConfig(enable_self_improvement=False)
        orchestrator = Orchestrator(config)
        
        try:
            # Process a complex request that would trigger monitoring
            response = await orchestrator.process_user_input(
                "Please analyze this code and suggest improvements: def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"
            )
            
            # Verify response was generated
            assert response.content is not None
            assert len(response.content) > 0
            
            # Get monitoring components
            monitoring = orchestrator.get_monitoring_components()
            assert monitoring is not None
            
            metrics_collector = monitoring["metrics_collector"]
            agent_tracker = monitoring["agent_tracker"]
            performance_monitor = monitoring["performance_monitor"]
            
            # Verify metrics were collected
            metrics = metrics_collector.get_all_metrics()
            assert "orchestrator.requests.total" in metrics
            assert metrics["orchestrator.requests.total"]["value"] >= 1
            
            # Verify performance data
            perf_data = performance_monitor.get_summary()
            assert "orchestrator.process_user_input" in perf_data
            
            # Create TUI components with monitoring data
            enhanced_layout = EnhancedLayout(
                config=EnhancedLayoutConfig(layout_mode="BALANCED"),
                metrics_collector=metrics_collector,
                agent_tracker=agent_tracker,
                performance_monitor=performance_monitor
            )
            
            # Verify layout can access monitoring data
            layout_info = enhanced_layout.get_layout_info()
            assert layout_info["panels"]["metrics_dashboard"] is True
            assert layout_info["panels"]["agent_status"] is True
            
        finally:
            await orchestrator.shutdown()


class TestMonitoringComponents:
    """Test individual monitoring components."""
    
    @pytest.mark.asyncio
    async def test_metrics_collector_operations(self):
        """Test basic metrics collector operations."""
        collector = MetricsCollector()
        
        try:
            # Test counter
            collector.record_counter("test.counter")
            collector.record_counter("test.counter", 5)
            
            # Test gauge
            collector.record_gauge("test.gauge", 42.5)
            collector.record_gauge("test.gauge", 100.0)
            
            # Test histogram
            for value in [10, 20, 30, 40, 50]:
                collector.record_histogram("test.histogram", value)
            
            # Test timer
            timer_id = collector.start_timer("test.timer")
            await asyncio.sleep(0.001)
            collector.end_timer("test.timer", timer_id)
            
            # Verify metrics
            metrics = collector.get_all_metrics()
            
            # Counter should be 6 (1 + 5)
            assert metrics["test.counter"]["value"] == 6
            
            # Gauge should be latest value
            assert metrics["test.gauge"]["value"] == 100.0
            
            # Histogram should have statistics
            histogram = metrics["test.histogram"]
            assert histogram["count"] == 5
            assert histogram["avg"] == 30.0
            assert histogram["min"] == 10
            assert histogram["max"] == 50
            
            # Timer should have timing data
            timer = metrics["test.timer"]
            assert timer["count"] == 1
            assert timer["avg_ms"] > 0
            
        finally:
            await collector.cleanup()
    
    @pytest.mark.asyncio
    async def test_agent_tracker_lifecycle(self):
        """Test agent tracker lifecycle management."""
        tracker = AgentTracker()
        
        try:
            # Register agents
            await tracker.register_agent("agent1", "idle", {"type": "backend"})
            await tracker.register_agent("agent2", "idle", {"type": "frontend"})
            
            # Verify registration
            agents = tracker.get_all_agents()
            assert len(agents) == 2
            assert agents["agent1"]["status"] == "idle"
            assert agents["agent2"]["status"] == "idle"
            
            # Start tasks
            task1_info = {"description": "Backend task", "type": "coding", "priority": "high"}
            task1_id = await tracker.start_task("agent1", task1_info)
            
            task2_info = {"description": "Frontend task", "type": "coding", "priority": "normal"}
            task2_id = await tracker.start_task("agent2", task2_info)
            
            # Update status
            await tracker.update_agent_status("agent1", "working", {"progress": 25})
            await tracker.update_agent_status("agent2", "working", {"progress": 75})
            
            # Verify task tracking
            agent1_tasks = tracker.get_agent_tasks("agent1")
            assert len(agent1_tasks) == 1
            assert agent1_tasks[0]["status"] == "active"
            
            # Complete tasks
            await tracker.complete_task("agent1", task1_id, "completed")
            await tracker.complete_task("agent2", task2_id, "failed", error="Compilation error")
            
            # Verify completion
            agent1_tasks_updated = tracker.get_agent_tasks("agent1")
            completed_task = next(t for t in agent1_tasks_updated if t["id"] == task1_id)
            assert completed_task["status"] == "completed"
            
            agent2_tasks_updated = tracker.get_agent_tasks("agent2")
            failed_task = next(t for t in agent2_tasks_updated if t["id"] == task2_id)
            assert failed_task["status"] == "failed"
            assert failed_task["error"] == "Compilation error"
            
        finally:
            await tracker.cleanup()
    
    @pytest.mark.asyncio
    async def test_performance_monitor_statistics(self):
        """Test performance monitor statistical calculations."""
        monitor = PerformanceMonitor()
        
        try:
            # Generate timing data with known values
            test_times = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # ms
            
            for time_ms in test_times:
                timer = monitor.start_timer("test.operation")
                # Simulate specific timing by manipulating the timer
                monitor.timers["test.operation"][timer]["start_time"] -= time_ms / 1000
                monitor.end_timer(timer)
            
            # Get performance summary
            summary = monitor.get_summary()
            operation_stats = summary["test.operation"]
            
            # Verify statistics
            assert operation_stats["count"] == 10
            assert operation_stats["avg_ms"] == 55.0  # Average of 10-100
            assert operation_stats["min_ms"] >= 10
            assert operation_stats["max_ms"] >= 100
            assert operation_stats["p50_ms"] >= 50  # Median should be around 55
            assert operation_stats["p95_ms"] >= 90  # 95th percentile should be around 95
            assert operation_stats["p99_ms"] >= 100  # 99th percentile should be around 99-100
            
        finally:
            await monitor.cleanup()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])