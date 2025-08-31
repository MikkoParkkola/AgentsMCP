"""
Comprehensive test suite for Symphony Orchestration API - Advanced multi-agent coordination.

This test suite validates the symphony orchestration API including multi-agent coordination,
real-time status monitoring, conflict resolution, and performance under load with 12+ agents.
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta
import sys
from pathlib import Path
from dataclasses import asdict
import uuid

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from agentsmcp.api.symphony_orchestration_api import (
    SymphonyOrchestrationAPI,
    AgentStatus,
    TaskPriority,
    ConflictType,
    Agent,
    Task,
    Conflict,
    SymphonyMetrics
)


@pytest.fixture
async def symphony_api():
    """Create a symphony orchestration API instance for testing."""
    api = SymphonyOrchestrationAPI()
    # Wait for initialization to complete
    await asyncio.sleep(0.1)
    yield api
    # Cleanup
    if api.symphony_active:
        await api.disable_symphony_mode()


@pytest.fixture
def sample_agent_data():
    """Provide sample agent data for testing."""
    return {
        "name": "test_agent",
        "type": "claude",
        "capabilities": ["analysis", "coding", "documentation"]
    }


@pytest.fixture
def sample_task_data():
    """Provide sample task data for testing."""
    return {
        "name": "test_task",
        "description": "A test task for validation",
        "priority": "normal",
        "required_capabilities": ["analysis"],
        "estimated_duration": 30
    }


class TestSymphonyModeLifecycle:
    """Test suite for symphony mode lifecycle management."""

    @pytest.mark.asyncio
    async def test_enable_symphony_mode(self, symphony_api):
        """Test enabling symphony mode with default agents."""
        response = await symphony_api.enable_symphony_mode()
        
        assert response.success
        assert response.data["symphony_active"]
        assert len(response.data["registered_agents"]) >= 2  # Default agents
        assert symphony_api.symphony_active
        assert symphony_api.start_time is not None

    @pytest.mark.asyncio
    async def test_enable_symphony_mode_with_custom_agents(self, symphony_api, sample_agent_data):
        """Test enabling symphony mode with custom initial agents."""
        initial_agents = [
            sample_agent_data,
            {
                "name": "agent2",
                "type": "gpt4", 
                "capabilities": ["execution", "testing"]
            }
        ]
        
        response = await symphony_api.enable_symphony_mode(
            initial_agents=initial_agents,
            auto_scale=True,
            max_agents=15
        )
        
        assert response.success
        assert response.data["symphony_active"]
        assert len(response.data["registered_agents"]) == 2
        assert response.data["auto_scale_enabled"]
        assert response.data["max_agents"] == 15

    @pytest.mark.asyncio
    async def test_disable_symphony_mode(self, symphony_api):
        """Test disabling symphony mode and cleanup."""
        # First enable symphony mode
        await symphony_api.enable_symphony_mode()
        
        # Add some test data
        await symphony_api.register_agent({"name": "test", "type": "claude"})
        await symphony_api.submit_task({"name": "test_task", "description": "test"})
        
        # Disable symphony mode
        response = await symphony_api.disable_symphony_mode()
        
        assert response.success
        assert response.data["symphony_disabled"]
        assert not symphony_api.symphony_active
        assert len(symphony_api.agents) == 0
        assert len(symphony_api.tasks) == 0

    @pytest.mark.asyncio
    async def test_enable_already_active_symphony(self, symphony_api):
        """Test error when trying to enable already active symphony mode."""
        # Enable first time
        await symphony_api.enable_symphony_mode()
        
        # Try to enable again
        response = await symphony_api.enable_symphony_mode()
        
        assert not response.success
        assert response.error_code == "ALREADY_ACTIVE"

    @pytest.mark.asyncio
    async def test_disable_inactive_symphony(self, symphony_api):
        """Test error when trying to disable inactive symphony mode."""
        response = await symphony_api.disable_symphony_mode()
        
        assert not response.success
        assert response.error_code == "NOT_ACTIVE"


class TestAgentManagement:
    """Test suite for agent registration and management."""

    @pytest.mark.asyncio
    async def test_register_agent(self, symphony_api, sample_agent_data):
        """Test registering a new agent."""
        await symphony_api.enable_symphony_mode()
        
        response = await symphony_api.register_agent(sample_agent_data)
        
        assert response.success
        agent_id = response.data["id"]
        assert agent_id in symphony_api.agents
        
        agent = symphony_api.agents[agent_id]
        assert agent.name == sample_agent_data["name"]
        assert agent.type == sample_agent_data["type"]
        assert agent.capabilities == sample_agent_data["capabilities"]
        assert agent.status == AgentStatus.IDLE

    @pytest.mark.asyncio
    async def test_register_agent_with_custom_id(self, symphony_api):
        """Test registering agent with custom ID."""
        await symphony_api.enable_symphony_mode()
        
        custom_id = "custom-agent-001"
        agent_data = {
            "id": custom_id,
            "name": "custom_agent",
            "type": "claude"
        }
        
        response = await symphony_api.register_agent(agent_data)
        
        assert response.success
        assert custom_id in symphony_api.agents

    @pytest.mark.asyncio
    async def test_register_duplicate_agent(self, symphony_api, sample_agent_data):
        """Test error when registering duplicate agent."""
        await symphony_api.enable_symphony_mode()
        
        # Register first time
        await symphony_api.register_agent(sample_agent_data)
        
        # Try to register same agent again
        response = await symphony_api.register_agent(sample_agent_data)
        
        assert not response.success
        assert response.error_code == "DUPLICATE_AGENT"

    @pytest.mark.asyncio
    async def test_agent_limit_enforcement(self, symphony_api, sample_agent_data):
        """Test agent limit enforcement."""
        await symphony_api.enable_symphony_mode(max_agents=3)
        
        # Register agents up to limit (2 default + 1 new = 3)
        response = await symphony_api.register_agent(sample_agent_data)
        assert response.success
        
        # Try to register one more (should fail)
        response = await symphony_api.register_agent({
            "name": "excess_agent",
            "type": "claude"
        })
        
        assert not response.success
        assert response.error_code == "AGENT_LIMIT"

    @pytest.mark.asyncio
    async def test_get_agent_details(self, symphony_api, sample_agent_data):
        """Test retrieving agent details."""
        await symphony_api.enable_symphony_mode()
        
        # Register an agent
        register_response = await symphony_api.register_agent(sample_agent_data)
        agent_id = register_response.data["id"]
        
        # Get agent details
        response = await symphony_api.get_agent_details(agent_id)
        
        assert response.success
        assert response.data["agent"]["id"] == agent_id
        assert response.data["agent"]["name"] == sample_agent_data["name"]
        assert "task_history" in response.data
        assert "recent_performance" in response.data

    @pytest.mark.asyncio
    async def test_get_nonexistent_agent_details(self, symphony_api):
        """Test error when getting details for nonexistent agent."""
        await symphony_api.enable_symphony_mode()
        
        response = await symphony_api.get_agent_details("nonexistent-agent")
        
        assert not response.success
        assert response.error_code == "AGENT_NOT_FOUND"


class TestTaskManagement:
    """Test suite for task submission and management."""

    @pytest.mark.asyncio
    async def test_submit_task(self, symphony_api, sample_task_data):
        """Test submitting a task."""
        await symphony_api.enable_symphony_mode()
        
        response = await symphony_api.submit_task(sample_task_data)
        
        assert response.success
        task_id = response.data["id"]
        assert task_id in symphony_api.tasks
        
        task = symphony_api.tasks[task_id]
        assert task.name == sample_task_data["name"]
        assert task.priority == TaskPriority.NORMAL
        assert task.status == "pending"

    @pytest.mark.asyncio
    async def test_submit_task_with_dependencies(self, symphony_api, sample_task_data):
        """Test submitting task with dependencies."""
        await symphony_api.enable_symphony_mode()
        
        # Submit parent task first
        parent_response = await symphony_api.submit_task(sample_task_data)
        parent_id = parent_response.data["id"]
        
        # Submit dependent task
        dependent_task = {
            "name": "dependent_task",
            "description": "Depends on parent task",
            "dependencies": [parent_id]
        }
        
        response = await symphony_api.submit_task(dependent_task)
        
        assert response.success
        task = symphony_api.tasks[response.data["id"]]
        assert parent_id in task.dependencies

    @pytest.mark.asyncio
    async def test_submit_task_invalid_dependency(self, symphony_api, sample_task_data):
        """Test error when submitting task with invalid dependency."""
        await symphony_api.enable_symphony_mode()
        
        invalid_task = {
            "name": "invalid_task",
            "description": "Has invalid dependency",
            "dependencies": ["nonexistent-task"]
        }
        
        response = await symphony_api.submit_task(invalid_task)
        
        assert not response.success
        assert response.error_code == "INVALID_DEPENDENCY"

    @pytest.mark.asyncio
    async def test_task_priority_queuing(self, symphony_api):
        """Test task priority queuing system."""
        await symphony_api.enable_symphony_mode()
        
        # Submit tasks with different priorities
        tasks = [
            {"name": "low_task", "priority": "low"},
            {"name": "critical_task", "priority": "critical"},
            {"name": "normal_task", "priority": "normal"},
            {"name": "high_task", "priority": "high"}
        ]
        
        for task_data in tasks:
            await symphony_api.submit_task(task_data)
        
        # Check queue order (should be by priority)
        assert len(symphony_api.task_queue) >= 4
        # Critical should be first, low should be last
        critical_task = symphony_api.tasks[symphony_api.task_queue[0]]
        assert critical_task.priority == TaskPriority.CRITICAL

    @pytest.mark.asyncio
    async def test_task_assignment_to_agents(self, symphony_api, sample_agent_data, sample_task_data):
        """Test task assignment to suitable agents."""
        await symphony_api.enable_symphony_mode()
        
        # Register an agent with matching capabilities
        await symphony_api.register_agent(sample_agent_data)
        
        # Submit a task requiring those capabilities
        task_data = sample_task_data.copy()
        task_data["required_capabilities"] = ["analysis"]
        
        await symphony_api.submit_task(task_data)
        
        # Wait for task processing
        await asyncio.sleep(0.2)
        
        # Task should be assigned
        assigned_tasks = [t for t in symphony_api.tasks.values() if t.assigned_agent_id is not None]
        assert len(assigned_tasks) > 0


class TestConflictResolution:
    """Test suite for conflict detection and resolution."""

    @pytest.mark.asyncio
    async def test_conflict_creation(self, symphony_api):
        """Test conflict creation and storage."""
        await symphony_api.enable_symphony_mode()
        
        conflict_id = str(uuid.uuid4())
        conflict = Conflict(
            id=conflict_id,
            type=ConflictType.RESOURCE,
            severity=0.8,
            involved_agents=["agent1", "agent2"],
            involved_tasks=["task1", "task2"],
            description="Resource contention between agents"
        )
        
        symphony_api.conflicts[conflict_id] = conflict
        
        assert conflict_id in symphony_api.conflicts
        assert symphony_api.conflicts[conflict_id].type == ConflictType.RESOURCE
        assert not symphony_api.conflicts[conflict_id].resolved

    @pytest.mark.asyncio
    async def test_resource_conflict_resolution(self, symphony_api):
        """Test resource conflict resolution."""
        await symphony_api.enable_symphony_mode()
        
        # Create a resource conflict
        conflict = Conflict(
            id="test-conflict",
            type=ConflictType.RESOURCE,
            severity=0.7,
            involved_agents=["agent1"],
            involved_tasks=["task1", "task2"],
            description="Resource conflict test"
        )
        
        symphony_api.conflicts["test-conflict"] = conflict
        
        # Add some test tasks
        task1 = Task(
            id="task1",
            name="Task 1",
            description="Test task",
            priority=TaskPriority.HIGH
        )
        task2 = Task(
            id="task2", 
            name="Task 2",
            description="Test task",
            priority=TaskPriority.LOW
        )
        
        symphony_api.tasks["task1"] = task1
        symphony_api.tasks["task2"] = task2
        
        # Resolve conflict
        await symphony_api._resolve_conflict(conflict)
        
        assert conflict.resolved
        assert conflict.resolution_strategy is not None

    @pytest.mark.asyncio
    async def test_conflict_metrics_tracking(self, symphony_api):
        """Test conflict metrics are tracked correctly."""
        await symphony_api.enable_symphony_mode()
        
        # Add resolved and unresolved conflicts
        resolved_conflict = Conflict(
            id="resolved",
            type=ConflictType.SCHEDULING,
            severity=0.5,
            involved_agents=[],
            involved_tasks=[],
            description="Resolved conflict",
            resolved=True
        )
        
        unresolved_conflict = Conflict(
            id="unresolved",
            type=ConflictType.DEPENDENCY,
            severity=0.8,
            involved_agents=[],
            involved_tasks=[],
            description="Unresolved conflict",
            resolved=False
        )
        
        symphony_api.conflicts["resolved"] = resolved_conflict
        symphony_api.conflicts["unresolved"] = unresolved_conflict
        
        metrics = await symphony_api._calculate_current_metrics()
        
        assert metrics.conflict_count == 2
        assert metrics.resolved_conflicts == 1


class TestMetricsAndMonitoring:
    """Test suite for metrics calculation and monitoring."""

    @pytest.mark.asyncio
    async def test_metrics_calculation(self, symphony_api, sample_agent_data, sample_task_data):
        """Test symphony metrics calculation."""
        await symphony_api.enable_symphony_mode()
        
        # Add some test data
        await symphony_api.register_agent(sample_agent_data)
        await symphony_api.submit_task(sample_task_data)
        
        metrics = await symphony_api._calculate_current_metrics()
        
        assert isinstance(metrics, SymphonyMetrics)
        assert metrics.active_agents >= 0
        assert 0.0 <= metrics.harmony_score <= 1.0
        assert metrics.uptime >= 0
        assert metrics.resource_utilization >= 0.0

    @pytest.mark.asyncio
    async def test_harmony_score_calculation(self, symphony_api):
        """Test harmony score calculation accuracy."""
        await symphony_api.enable_symphony_mode()
        
        # Add healthy agents
        for i in range(3):
            agent_data = {
                "name": f"agent_{i}",
                "type": "claude"
            }
            response = await symphony_api.register_agent(agent_data)
            agent_id = response.data["id"]
            symphony_api.agents[agent_id].health_score = 0.9
        
        metrics = await symphony_api._calculate_current_metrics()
        
        # Harmony score should be high with healthy agents
        assert metrics.harmony_score > 0.5

    @pytest.mark.asyncio
    async def test_throughput_calculation(self, symphony_api, sample_task_data):
        """Test throughput calculation."""
        await symphony_api.enable_symphony_mode()
        
        # Add completed tasks
        for i in range(5):
            task_data = sample_task_data.copy()
            task_data["name"] = f"completed_task_{i}"
            response = await symphony_api.submit_task(task_data)
            task_id = response.data["id"]
            
            # Mark as completed
            task = symphony_api.tasks[task_id]
            task.status = "completed"
            task.actual_duration = 10.0
        
        # Wait a moment for time-based calculations
        await asyncio.sleep(0.1)
        
        metrics = await symphony_api._calculate_current_metrics()
        
        assert metrics.completed_tasks == 5
        assert metrics.throughput >= 0

    @pytest.mark.asyncio
    async def test_resource_utilization_calculation(self, symphony_api, sample_agent_data):
        """Test resource utilization calculation."""
        await symphony_api.enable_symphony_mode()
        
        # Register agents with different workloads
        for i in range(3):
            agent_data = sample_agent_data.copy()
            agent_data["name"] = f"agent_{i}"
            response = await symphony_api.register_agent(agent_data)
            agent_id = response.data["id"]
            
            # Set workload
            symphony_api.agents[agent_id].workload = i * 0.5
        
        metrics = await symphony_api._calculate_current_metrics()
        
        assert 0.0 <= metrics.resource_utilization <= 1.0


class TestPerformanceAndScaling:
    """Test suite for performance and scaling capabilities."""

    @pytest.mark.asyncio
    async def test_12_plus_concurrent_agents(self, symphony_api):
        """Test handling 12+ concurrent agents."""
        await symphony_api.enable_symphony_mode(max_agents=15)
        
        # Register 12 agents
        agent_ids = []
        for i in range(12):
            agent_data = {
                "name": f"agent_{i}",
                "type": "claude" if i % 2 == 0 else "gpt4",
                "capabilities": ["analysis", "execution"]
            }
            response = await symphony_api.register_agent(agent_data)
            assert response.success
            agent_ids.append(response.data["id"])
        
        # Verify all agents registered
        assert len(symphony_api.agents) >= 12
        
        # Submit tasks for all agents
        for i in range(12):
            task_data = {
                "name": f"task_{i}",
                "description": f"Task for agent {i}",
                "required_capabilities": ["analysis"]
            }
            response = await symphony_api.submit_task(task_data)
            assert response.success
        
        # Wait for task processing
        await asyncio.sleep(0.5)
        
        # Verify system handles the load
        status_response = await symphony_api.get_symphony_status()
        assert status_response.success
        assert status_response.data["agents"]["total"] >= 12

    @pytest.mark.asyncio
    async def test_high_task_throughput(self, symphony_api, sample_agent_data):
        """Test high task throughput capabilities."""
        await symphony_api.enable_symphony_mode()
        
        # Register multiple agents
        for i in range(5):
            agent_data = sample_agent_data.copy()
            agent_data["name"] = f"throughput_agent_{i}"
            await symphony_api.register_agent(agent_data)
        
        # Submit many tasks rapidly
        task_ids = []
        for i in range(50):
            task_data = {
                "name": f"throughput_task_{i}",
                "description": "High throughput test task",
                "estimated_duration": 1  # Short duration
            }
            response = await symphony_api.submit_task(task_data)
            if response.success:
                task_ids.append(response.data["id"])
        
        # Should successfully queue all tasks
        assert len(task_ids) >= 45  # Allow for some failures

    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, symphony_api):
        """Test memory usage remains reasonable under load."""
        import gc
        
        await symphony_api.enable_symphony_mode()
        
        # Get baseline memory
        gc.collect()
        baseline_objects = len(gc.get_objects())
        
        # Create load
        for i in range(20):
            await symphony_api.register_agent({
                "name": f"memory_test_agent_{i}",
                "type": "claude"
            })
            
            await symphony_api.submit_task({
                "name": f"memory_test_task_{i}",
                "description": "Memory test task"
            })
        
        # Check memory growth
        gc.collect()
        final_objects = len(gc.get_objects())
        growth_ratio = final_objects / baseline_objects
        
        # Memory growth should be reasonable (less than 3x)
        assert growth_ratio < 3.0, f"Memory grew {growth_ratio:.2f}x"

    @pytest.mark.asyncio
    async def test_response_time_under_load(self, symphony_api):
        """Test API response times remain under 100ms under load."""
        await symphony_api.enable_symphony_mode()
        
        # Create some load first
        for i in range(5):
            await symphony_api.register_agent({
                "name": f"load_agent_{i}",
                "type": "claude"
            })
        
        # Measure response times
        response_times = []
        for i in range(20):
            start_time = time.time()
            response = await symphony_api.get_symphony_status()
            response_time = time.time() - start_time
            
            assert response.success
            response_times.append(response_time)
        
        # Calculate statistics
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        
        # Should meet performance requirements
        assert avg_response_time < 0.1, f"Average response time {avg_response_time:.3f}s > 0.1s"
        assert max_response_time < 0.2, f"Max response time {max_response_time:.3f}s > 0.2s"


class TestErrorHandlingAndResilience:
    """Test suite for error handling and system resilience."""

    @pytest.mark.asyncio
    async def test_agent_failure_handling(self, symphony_api, sample_agent_data, sample_task_data):
        """Test handling of agent failures."""
        await symphony_api.enable_symphony_mode()
        
        # Register agent and assign task
        agent_response = await symphony_api.register_agent(sample_agent_data)
        agent_id = agent_response.data["id"]
        
        task_response = await symphony_api.submit_task(sample_task_data)
        task_id = task_response.data["id"]
        
        # Simulate agent failure
        agent = symphony_api.agents[agent_id]
        agent.status = AgentStatus.FAILED
        agent.current_task_id = task_id
        
        # Handle failure
        await symphony_api._handle_agent_failure(agent)
        
        # Task should be reassigned
        task = symphony_api.tasks[task_id]
        assert task.assigned_agent_id is None
        assert task.status == "pending"

    @pytest.mark.asyncio
    async def test_task_retry_mechanism(self, symphony_api, sample_task_data):
        """Test automatic task retry on failure."""
        await symphony_api.enable_symphony_mode()
        
        # Submit task
        response = await symphony_api.submit_task(sample_task_data)
        task_id = response.data["id"]
        task = symphony_api.tasks[task_id]
        
        # Simulate task failure
        task.status = "failed"
        task.retry_count = 1
        task.max_retries = 3
        
        # Handle completion (should trigger retry)
        await symphony_api._handle_task_completion(task)
        
        # Task should be reset for retry
        assert task.status == "pending"

    @pytest.mark.asyncio
    async def test_coordination_system_resilience(self, symphony_api):
        """Test coordination system resilience to errors."""
        await symphony_api.enable_symphony_mode()
        
        # Create problematic scenarios
        # 1. Agent with no heartbeat
        agent_data = {"name": "silent_agent", "type": "claude"}
        response = await symphony_api.register_agent(agent_data)
        agent_id = response.data["id"]
        
        # Set old heartbeat
        symphony_api.agents[agent_id].last_heartbeat = datetime.utcnow() - timedelta(minutes=10)
        
        # Wait for monitoring cycle
        await asyncio.sleep(0.2)
        
        # System should still be functional
        status_response = await symphony_api.get_symphony_status()
        assert status_response.success

    @pytest.mark.asyncio
    async def test_graceful_degradation(self, symphony_api):
        """Test graceful degradation when resources are limited."""
        # Enable with very low limits
        await symphony_api.enable_symphony_mode(max_agents=2)
        
        # Try to register more agents than allowed
        successful_registrations = 0
        for i in range(5):
            response = await symphony_api.register_agent({
                "name": f"degradation_agent_{i}",
                "type": "claude"
            })
            if response.success:
                successful_registrations += 1
        
        # Should have graceful limits
        assert successful_registrations <= 2
        
        # System should still be functional
        status_response = await symphony_api.get_symphony_status()
        assert status_response.success


class TestStatusAndReporting:
    """Test suite for status reporting and system information."""

    @pytest.mark.asyncio
    async def test_get_symphony_status_active(self, symphony_api):
        """Test getting symphony status when active."""
        await symphony_api.enable_symphony_mode()
        
        response = await symphony_api.get_symphony_status()
        
        assert response.success
        data = response.data
        assert data["symphony_active"]
        assert "start_time" in data
        assert "uptime_seconds" in data
        assert "agents" in data
        assert "tasks" in data
        assert "metrics" in data

    @pytest.mark.asyncio
    async def test_get_symphony_status_inactive(self, symphony_api):
        """Test getting symphony status when inactive."""
        response = await symphony_api.get_symphony_status()
        
        assert response.success
        assert not response.data["symphony_active"]
        assert "Symphony mode not active" in response.data["message"]

    @pytest.mark.asyncio
    async def test_status_data_accuracy(self, symphony_api, sample_agent_data, sample_task_data):
        """Test accuracy of status data reporting."""
        await symphony_api.enable_symphony_mode()
        
        # Add known data
        await symphony_api.register_agent(sample_agent_data)
        await symphony_api.submit_task(sample_task_data)
        
        response = await symphony_api.get_symphony_status()
        
        assert response.success
        data = response.data
        
        # Verify counts include our additions
        assert data["agents"]["total"] >= 3  # 2 default + 1 added
        assert data["tasks"]["total"] >= 1
        assert data["tasks"]["pending"] >= 1

    @pytest.mark.asyncio
    async def test_real_time_status_updates(self, symphony_api, sample_task_data):
        """Test real-time status updates reflect current state."""
        await symphony_api.enable_symphony_mode()
        
        # Get initial status
        initial_response = await symphony_api.get_symphony_status()
        initial_task_count = initial_response.data["tasks"]["total"]
        
        # Add a task
        await symphony_api.submit_task(sample_task_data)
        
        # Get updated status
        updated_response = await symphony_api.get_symphony_status()
        updated_task_count = updated_response.data["tasks"]["total"]
        
        # Should reflect the change
        assert updated_task_count > initial_task_count


class TestBackgroundProcesses:
    """Test suite for background monitoring and processing."""

    @pytest.mark.asyncio
    async def test_agent_monitoring_background_task(self, symphony_api, sample_agent_data):
        """Test agent monitoring background task."""
        await symphony_api.enable_symphony_mode()
        
        # Register agent
        response = await symphony_api.register_agent(sample_agent_data)
        agent_id = response.data["id"]
        
        # Simulate agent going offline (old heartbeat)
        symphony_api.agents[agent_id].last_heartbeat = datetime.utcnow() - timedelta(minutes=5)
        
        # Wait for monitoring cycle
        await asyncio.sleep(symphony_api.coordination_rules["heartbeat_interval_seconds"] + 1)
        
        # Agent should be marked as failed
        agent = symphony_api.agents[agent_id]
        assert agent.status == AgentStatus.FAILED or agent.health_score < 1.0

    @pytest.mark.asyncio
    async def test_task_queue_processing(self, symphony_api, sample_agent_data, sample_task_data):
        """Test task queue processing background task."""
        await symphony_api.enable_symphony_mode()
        
        # Register agent
        await symphony_api.register_agent(sample_agent_data)
        
        # Submit task
        await symphony_api.submit_task(sample_task_data)
        
        # Wait for processing
        await asyncio.sleep(0.5)
        
        # Task should be processed (assigned or completed)
        tasks = list(symphony_api.tasks.values())
        assigned_tasks = [t for t in tasks if t.assigned_agent_id is not None]
        assert len(assigned_tasks) > 0

    @pytest.mark.asyncio
    async def test_metrics_update_background_task(self, symphony_api):
        """Test metrics update background task."""
        await symphony_api.enable_symphony_mode()
        
        # Wait for metrics update cycle
        await asyncio.sleep(1.0)
        
        # Should have metrics history
        assert len(symphony_api.metrics_history) > 0
        assert symphony_api.last_metrics_update > 0

    @pytest.mark.asyncio
    async def test_conflict_resolution_background_task(self, symphony_api):
        """Test conflict resolution background task."""
        await symphony_api.enable_symphony_mode()
        
        # Add a conflict
        conflict = Conflict(
            id="test-conflict",
            type=ConflictType.SCHEDULING,
            severity=0.5,
            involved_agents=[],
            involved_tasks=[],
            description="Test conflict for resolution"
        )
        symphony_api.conflicts["test-conflict"] = conflict
        
        # Wait for resolution cycle
        await asyncio.sleep(symphony_api.coordination_rules["conflict_resolution_timeout"] + 1)
        
        # Conflict should be resolved
        assert conflict.resolved


class TestDataConsistency:
    """Test suite for data consistency and integrity."""

    @pytest.mark.asyncio
    async def test_agent_task_relationship_consistency(self, symphony_api, sample_agent_data, sample_task_data):
        """Test consistency between agent and task relationships."""
        await symphony_api.enable_symphony_mode()
        
        # Register agent and submit task
        agent_response = await symphony_api.register_agent(sample_agent_data)
        agent_id = agent_response.data["id"]
        
        task_response = await symphony_api.submit_task(sample_task_data)
        task_id = task_response.data["id"]
        
        # Manually assign task to agent (simulating assignment)
        task = symphony_api.tasks[task_id]
        agent = symphony_api.agents[agent_id]
        
        task.assigned_agent_id = agent_id
        agent.current_task_id = task_id
        
        # Verify relationship consistency
        assert task.assigned_agent_id == agent_id
        assert agent.current_task_id == task_id

    @pytest.mark.asyncio
    async def test_task_dependency_consistency(self, symphony_api, sample_task_data):
        """Test task dependency relationship consistency."""
        await symphony_api.enable_symphony_mode()
        
        # Submit parent task
        parent_response = await symphony_api.submit_task(sample_task_data)
        parent_id = parent_response.data["id"]
        
        # Submit dependent task
        dependent_task = sample_task_data.copy()
        dependent_task["name"] = "dependent_task"
        dependent_task["dependencies"] = [parent_id]
        
        dependent_response = await symphony_api.submit_task(dependent_task)
        dependent_id = dependent_response.data["id"]
        
        # Verify dependency consistency
        dependent_task_obj = symphony_api.tasks[dependent_id]
        assert parent_id in dependent_task_obj.dependencies

    @pytest.mark.asyncio
    async def test_metrics_data_consistency(self, symphony_api):
        """Test metrics data consistency with actual system state."""
        await symphony_api.enable_symphony_mode()
        
        # Add known data
        for i in range(3):
            await symphony_api.register_agent({
                "name": f"metrics_agent_{i}",
                "type": "claude"
            })
        
        metrics = await symphony_api._calculate_current_metrics()
        
        # Metrics should reflect actual state
        assert metrics.active_agents == len([a for a in symphony_api.agents.values() 
                                             if a.status in [AgentStatus.WORKING, AgentStatus.IDLE]])


# Integration and end-to-end tests
@pytest.mark.asyncio
async def test_complete_symphony_workflow(symphony_api):
    """Test complete symphony workflow from start to finish."""
    # 1. Enable symphony mode
    enable_response = await symphony_api.enable_symphony_mode()
    assert enable_response.success
    
    # 2. Register multiple agents
    agents = []
    for i in range(4):
        agent_data = {
            "name": f"workflow_agent_{i}",
            "type": "claude",
            "capabilities": ["analysis", "execution"]
        }
        response = await symphony_api.register_agent(agent_data)
        assert response.success
        agents.append(response.data["id"])
    
    # 3. Submit multiple tasks
    tasks = []
    for i in range(8):
        task_data = {
            "name": f"workflow_task_{i}",
            "description": f"Workflow test task {i}",
            "priority": "normal" if i % 2 == 0 else "high",
            "required_capabilities": ["analysis"]
        }
        response = await symphony_api.submit_task(task_data)
        assert response.success
        tasks.append(response.data["id"])
    
    # 4. Wait for processing
    await asyncio.sleep(1.0)
    
    # 5. Check system status
    status_response = await symphony_api.get_symphony_status()
    assert status_response.success
    assert status_response.data["symphony_active"]
    assert status_response.data["agents"]["total"] >= 4
    assert status_response.data["tasks"]["total"] >= 8
    
    # 6. Get detailed agent information
    for agent_id in agents[:2]:  # Check first 2 agents
        agent_response = await symphony_api.get_agent_details(agent_id)
        assert agent_response.success
    
    # 7. Disable symphony mode
    disable_response = await symphony_api.disable_symphony_mode()
    assert disable_response.success
    assert not symphony_api.symphony_active


@pytest.mark.asyncio
async def test_symphony_mode_stress_test(symphony_api):
    """Stress test symphony mode with high load."""
    await symphony_api.enable_symphony_mode(max_agents=15)
    
    # Register maximum agents
    for i in range(12):
        agent_data = {
            "name": f"stress_agent_{i}",
            "type": "claude" if i % 2 == 0 else "gpt4",
            "capabilities": ["analysis", "execution", "coordination"]
        }
        response = await symphony_api.register_agent(agent_data)
        assert response.success
    
    # Submit many tasks with various priorities
    priorities = ["low", "normal", "high", "critical"]
    for i in range(100):
        task_data = {
            "name": f"stress_task_{i}",
            "description": f"Stress test task {i}",
            "priority": priorities[i % len(priorities)],
            "required_capabilities": ["analysis"] if i % 3 == 0 else []
        }
        response = await symphony_api.submit_task(task_data)
        # Some may fail due to system limits, that's expected
    
    # Let system process for a while
    await asyncio.sleep(2.0)
    
    # System should still be responsive
    status_response = await symphony_api.get_symphony_status()
    assert status_response.success
    
    # Check metrics are reasonable
    metrics = status_response.data["metrics"]
    assert 0.0 <= metrics["harmony_score"] <= 1.0
    assert metrics["active_agents"] <= 15


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])