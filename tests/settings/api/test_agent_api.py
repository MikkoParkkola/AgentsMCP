"""
API tests for agent REST endpoints.
"""

import pytest
import json
import uuid
from datetime import datetime
from fastapi.testclient import TestClient
from fastapi import FastAPI

from agentsmcp.settings.api.agent_api import AgentAPI
from agentsmcp.settings.services.agent_service import AgentService
from agentsmcp.settings.adapters.memory_repositories import (
    InMemoryAgentRepository,
    InMemoryUserRepository,
    InMemoryAuditRepository,
)
from agentsmcp.settings.domain.entities import UserProfile
from agentsmcp.settings.domain.value_objects import (
    AgentStatus,
    PermissionLevel,
    ValidationRule,
)
from agentsmcp.settings.events.event_publisher import EventPublisher


@pytest.fixture
def repositories():
    """Create repository fixtures."""
    return {
        "agent": InMemoryAgentRepository(),
        "user": InMemoryUserRepository(),
        "audit": InMemoryAuditRepository(),
    }


@pytest.fixture
def event_publisher():
    """Create event publisher fixture."""
    return EventPublisher()


@pytest.fixture
def agent_service(repositories, event_publisher):
    """Create agent service fixture."""
    return AgentService(
        agent_repository=repositories["agent"],
        user_repository=repositories["user"],
        audit_repository=repositories["audit"],
        event_publisher=event_publisher
    )


@pytest.fixture
def sample_user():
    """Create sample user fixture."""
    return UserProfile(
        user_id="test_user_123",
        username="testuser",
        email="test@example.com",
        permissions=[PermissionLevel.READ, PermissionLevel.WRITE]
    )


@pytest.fixture
def app(agent_service):
    """Create FastAPI test app."""
    app = FastAPI()
    agent_api = AgentAPI(agent_service)
    app.include_router(agent_api.router)
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


class TestAgentAPI:
    """Test Agent API endpoints."""
    
    @pytest.mark.asyncio
    async def test_create_agent_endpoint(self, client, repositories, sample_user):
        """Test POST /api/v1/agents endpoint."""
        await repositories["user"].save_user(sample_user)
        
        request_data = {
            "name": "Test Agent",
            "description": "A test agent for API testing",
            "configuration": {
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 1000
            }
        }
        
        response = client.post(
            "/api/v1/agents",
            json=request_data,
            headers={"X-User-ID": sample_user.user_id}
        )
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert "agent_id" in response_data
        assert isinstance(response_data["agent_id"], str)
        assert "message" in response_data
        
        # Verify agent was created
        agent = await repositories["agent"].get_agent_definition(response_data["agent_id"])
        assert agent is not None
        assert agent.name == "Test Agent"
        assert agent.configuration["model"] == "gpt-4"
    
    @pytest.mark.asyncio
    async def test_get_agent_endpoint(self, client, repositories, agent_service, sample_user):
        """Test GET /api/v1/agents/{agent_id} endpoint."""
        await repositories["user"].save_user(sample_user)
        
        # Create agent first
        agent_id = await agent_service.create_agent_definition(
            user_id=sample_user.user_id,
            name="Get Test Agent",
            description="Agent for testing GET endpoint",
            configuration={"model": "claude-3"}
        )
        
        response = client.get(
            f"/api/v1/agents/{agent_id}",
            headers={"X-User-ID": sample_user.user_id}
        )
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert response_data["id"] == agent_id
        assert response_data["name"] == "Get Test Agent"
        assert response_data["configuration"]["model"] == "claude-3"
        assert response_data["status"] == "draft"
        assert "created_at" in response_data
        assert "updated_at" in response_data
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_agent(self, client, sample_user):
        """Test GET endpoint with non-existent agent ID."""
        fake_id = str(uuid.uuid4())
        
        response = client.get(
            f"/api/v1/agents/{fake_id}",
            headers={"X-User-ID": sample_user.user_id}
        )
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    @pytest.mark.asyncio
    async def test_update_agent_endpoint(self, client, repositories, agent_service, sample_user):
        """Test PUT /api/v1/agents/{agent_id} endpoint."""
        await repositories["user"].save_user(sample_user)
        
        # Create agent first
        agent_id = await agent_service.create_agent_definition(
            user_id=sample_user.user_id,
            name="Original Agent",
            description="Original description"
        )
        
        update_data = {
            "name": "Updated Agent",
            "description": "Updated description",
            "configuration": {
                "model": "gpt-4",
                "temperature": 0.8,
                "max_tokens": 2000
            }
        }
        
        response = client.put(
            f"/api/v1/agents/{agent_id}",
            json=update_data,
            headers={"X-User-ID": sample_user.user_id}
        )
        
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        
        # Verify agent was updated
        agent = await repositories["agent"].get_agent_definition(agent_id)
        assert agent.name == "Updated Agent"
        assert agent.description == "Updated description"
        assert agent.configuration["temperature"] == 0.8
    
    @pytest.mark.asyncio
    async def test_delete_agent_endpoint(self, client, repositories, agent_service, sample_user):
        """Test DELETE /api/v1/agents/{agent_id} endpoint."""
        await repositories["user"].save_user(sample_user)
        
        # Create agent first
        agent_id = await agent_service.create_agent_definition(
            user_id=sample_user.user_id,
            name="Delete Test Agent",
            description="Agent for testing deletion"
        )
        
        # Verify agent exists
        agent = await repositories["agent"].get_agent_definition(agent_id)
        assert agent is not None
        
        # Delete agent
        response = client.delete(
            f"/api/v1/agents/{agent_id}",
            headers={"X-User-ID": sample_user.user_id}
        )
        
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        
        # Verify agent was deleted
        agent = await repositories["agent"].get_agent_definition(agent_id)
        assert agent is None
    
    @pytest.mark.asyncio
    async def test_change_agent_status_endpoint(self, client, repositories, agent_service, sample_user):
        """Test PATCH /api/v1/agents/{agent_id}/status endpoint."""
        await repositories["user"].save_user(sample_user)
        
        # Create agent first
        agent_id = await agent_service.create_agent_definition(
            user_id=sample_user.user_id,
            name="Status Test Agent",
            description="Agent for testing status changes"
        )
        
        request_data = {"status": "testing"}
        
        response = client.patch(
            f"/api/v1/agents/{agent_id}/status",
            json=request_data,
            headers={"X-User-ID": sample_user.user_id}
        )
        
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        
        # Verify status changed
        agent = await repositories["agent"].get_agent_definition(agent_id)
        assert agent.status == AgentStatus.TESTING
    
    @pytest.mark.asyncio
    async def test_invalid_status_change(self, client, repositories, agent_service, sample_user):
        """Test status change with invalid status value."""
        await repositories["user"].save_user(sample_user)
        
        # Create agent first
        agent_id = await agent_service.create_agent_definition(
            user_id=sample_user.user_id,
            name="Invalid Status Test",
            description="Test invalid status"
        )
        
        request_data = {"status": "invalid_status"}
        
        response = client.patch(
            f"/api/v1/agents/{agent_id}/status",
            json=request_data,
            headers={"X-User-ID": sample_user.user_id}
        )
        
        assert response.status_code == 422
        error_details = response.json()["detail"]
        assert any("status" in str(error) for error in error_details)
    
    @pytest.mark.asyncio
    async def test_get_user_agents_endpoint(self, client, repositories, agent_service, sample_user):
        """Test GET /api/v1/agents endpoint."""
        await repositories["user"].save_user(sample_user)
        
        # Create multiple agents
        agent_names = ["Agent 1", "Agent 2", "Agent 3"]
        agent_ids = []
        
        for name in agent_names:
            agent_id = await agent_service.create_agent_definition(
                user_id=sample_user.user_id,
                name=name,
                description=f"Description for {name}"
            )
            agent_ids.append(agent_id)
        
        response = client.get(
            "/api/v1/agents",
            headers={"X-User-ID": sample_user.user_id}
        )
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert "agents" in response_data
        agents = response_data["agents"]
        assert len(agents) == 3
        
        retrieved_names = [agent["name"] for agent in agents]
        for name in agent_names:
            assert name in retrieved_names
    
    @pytest.mark.asyncio
    async def test_create_agent_instance_endpoint(self, client, repositories, agent_service, sample_user):
        """Test POST /api/v1/agents/{agent_id}/instances endpoint."""
        await repositories["user"].save_user(sample_user)
        
        # Create agent definition first
        agent_id = await agent_service.create_agent_definition(
            user_id=sample_user.user_id,
            name="Instance Test Agent",
            description="Agent for testing instances"
        )
        
        request_data = {
            "session_id": "test_session_123",
            "runtime_config": {
                "current_task": "analysis",
                "context": "document_review"
            }
        }
        
        response = client.post(
            f"/api/v1/agents/{agent_id}/instances",
            json=request_data,
            headers={"X-User-ID": sample_user.user_id}
        )
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert "instance_id" in response_data
        assert isinstance(response_data["instance_id"], str)
        
        # Verify instance was created
        instance = await repositories["agent"].get_agent_instance(response_data["instance_id"])
        assert instance is not None
        assert instance.session_id == "test_session_123"
        assert instance.runtime_config["current_task"] == "analysis"
    
    @pytest.mark.asyncio
    async def test_get_agent_instances_endpoint(self, client, repositories, agent_service, sample_user):
        """Test GET /api/v1/agents/{agent_id}/instances endpoint."""
        await repositories["user"].save_user(sample_user)
        
        # Create agent definition
        agent_id = await agent_service.create_agent_definition(
            user_id=sample_user.user_id,
            name="Multi-Instance Agent",
            description="Agent with multiple instances"
        )
        
        # Create multiple instances
        instance_ids = []
        for i in range(3):
            instance_id = await agent_service.create_agent_instance(
                user_id=sample_user.user_id,
                agent_definition_id=agent_id,
                session_id=f"session_{i}"
            )
            instance_ids.append(instance_id)
        
        response = client.get(
            f"/api/v1/agents/{agent_id}/instances",
            headers={"X-User-ID": sample_user.user_id}
        )
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert "instances" in response_data
        instances = response_data["instances"]
        assert len(instances) == 3
        
        retrieved_ids = [instance["id"] for instance in instances]
        for instance_id in instance_ids:
            assert instance_id in retrieved_ids
    
    @pytest.mark.asyncio
    async def test_update_instance_config_endpoint(self, client, repositories, agent_service, sample_user):
        """Test PUT /api/v1/agents/instances/{instance_id}/config endpoint."""
        await repositories["user"].save_user(sample_user)
        
        # Create agent and instance
        agent_id = await agent_service.create_agent_definition(
            user_id=sample_user.user_id,
            name="Config Update Agent",
            description="Agent for testing config updates"
        )
        
        instance_id = await agent_service.create_agent_instance(
            user_id=sample_user.user_id,
            agent_definition_id=agent_id,
            session_id="config_test_session"
        )
        
        request_data = {
            "runtime_config": {
                "current_task": "code_review",
                "context": "python_file",
                "settings": {"strict_mode": True}
            }
        }
        
        response = client.put(
            f"/api/v1/agents/instances/{instance_id}/config",
            json=request_data,
            headers={"X-User-ID": sample_user.user_id}
        )
        
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        
        # Verify config was updated
        instance = await repositories["agent"].get_agent_instance(instance_id)
        assert instance.runtime_config["current_task"] == "code_review"
        assert instance.runtime_config["settings"]["strict_mode"] is True
    
    @pytest.mark.asyncio
    async def test_update_performance_metrics_endpoint(self, client, repositories, agent_service, sample_user):
        """Test PUT /api/v1/agents/instances/{instance_id}/metrics endpoint."""
        await repositories["user"].save_user(sample_user)
        
        # Create agent and instance
        agent_id = await agent_service.create_agent_definition(
            user_id=sample_user.user_id,
            name="Metrics Test Agent",
            description="Agent for testing metrics"
        )
        
        instance_id = await agent_service.create_agent_instance(
            user_id=sample_user.user_id,
            agent_definition_id=agent_id,
            session_id="metrics_test_session"
        )
        
        request_data = {
            "metrics": {
                "response_time_ms": 1500,
                "accuracy_score": 0.95,
                "tasks_completed": 42
            }
        }
        
        response = client.put(
            f"/api/v1/agents/instances/{instance_id}/metrics",
            json=request_data,
            headers={"X-User-ID": sample_user.user_id}
        )
        
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["status"] == "success"
        
        # Verify metrics were updated
        instance = await repositories["agent"].get_agent_instance(instance_id)
        assert instance.performance_metrics["response_time_ms"] == 1500
        assert instance.performance_metrics["accuracy_score"] == 0.95
    
    @pytest.mark.asyncio
    async def test_unauthorized_agent_access(self, client, repositories, agent_service, sample_user):
        """Test unauthorized access to other users' agents."""
        # Create authorized user and agent
        await repositories["user"].save_user(sample_user)
        
        agent_id = await agent_service.create_agent_definition(
            user_id=sample_user.user_id,
            name="Private Agent",
            description="Should not be accessible to others"
        )
        
        # Try to access with different user ID
        response = client.get(
            f"/api/v1/agents/{agent_id}",
            headers={"X-User-ID": "unauthorized_user"}
        )
        
        # Should return 403 or 404 depending on implementation
        assert response.status_code in [403, 404]
    
    @pytest.mark.asyncio
    async def test_invalid_request_data(self, client, sample_user):
        """Test API endpoints with invalid request data."""
        # Try to create agent with missing required fields
        response = client.post(
            "/api/v1/agents",
            json={},  # Missing required fields
            headers={"X-User-ID": sample_user.user_id}
        )
        
        assert response.status_code == 422
        error_details = response.json()["detail"]
        # Should have errors for missing name and description
        error_fields = [str(error) for error in error_details]
        assert any("name" in error for error in error_fields)
        assert any("description" in error for error in error_fields)
    
    @pytest.mark.asyncio
    async def test_agent_lifecycle_via_api(self, client, repositories, sample_user):
        """Test complete agent lifecycle through API endpoints."""
        await repositories["user"].save_user(sample_user)
        
        # 1. Create agent
        create_response = client.post(
            "/api/v1/agents",
            json={
                "name": "Lifecycle Test Agent",
                "description": "Agent for testing full lifecycle",
                "configuration": {"model": "gpt-4"}
            },
            headers={"X-User-ID": sample_user.user_id}
        )
        agent_id = create_response.json()["agent_id"]
        
        # 2. Update agent
        client.put(
            f"/api/v1/agents/{agent_id}",
            json={
                "name": "Updated Lifecycle Agent",
                "description": "Updated description",
                "configuration": {"model": "gpt-4", "temperature": 0.8}
            },
            headers={"X-User-ID": sample_user.user_id}
        )
        
        # 3. Change status to testing
        client.patch(
            f"/api/v1/agents/{agent_id}/status",
            json={"status": "testing"},
            headers={"X-User-ID": sample_user.user_id}
        )
        
        # 4. Create instance
        instance_response = client.post(
            f"/api/v1/agents/{agent_id}/instances",
            json={"session_id": "lifecycle_session"},
            headers={"X-User-ID": sample_user.user_id}
        )
        instance_id = instance_response.json()["instance_id"]
        
        # 5. Update metrics
        client.put(
            f"/api/v1/agents/instances/{instance_id}/metrics",
            json={"metrics": {"test_score": 0.9}},
            headers={"X-User-ID": sample_user.user_id}
        )
        
        # 6. Change status to active
        client.patch(
            f"/api/v1/agents/{agent_id}/status",
            json={"status": "active"},
            headers={"X-User-ID": sample_user.user_id}
        )
        
        # Verify final state
        get_response = client.get(
            f"/api/v1/agents/{agent_id}",
            headers={"X-User-ID": sample_user.user_id}
        )
        
        agent_data = get_response.json()
        assert agent_data["name"] == "Updated Lifecycle Agent"
        assert agent_data["status"] == "active"
        assert agent_data["configuration"]["temperature"] == 0.8
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self, client, sample_user):
        """Test API error handling."""
        # Test with invalid UUID format
        response = client.get(
            "/api/v1/agents/invalid-uuid",
            headers={"X-User-ID": sample_user.user_id}
        )
        
        # Should handle gracefully
        assert response.status_code in [400, 404, 422]
        
        # Test without authentication
        response = client.get("/api/v1/agents")
        assert response.status_code == 422  # Missing header
    
    @pytest.mark.asyncio
    async def test_concurrent_instance_operations(self, client, repositories, agent_service, sample_user):
        """Test concurrent operations on agent instances."""
        await repositories["user"].save_user(sample_user)
        
        # Create agent
        agent_id = await agent_service.create_agent_definition(
            user_id=sample_user.user_id,
            name="Concurrent Test Agent",
            description="Agent for testing concurrent operations"
        )
        
        # Create instance
        instance_response = client.post(
            f"/api/v1/agents/{agent_id}/instances",
            json={"session_id": "concurrent_session"},
            headers={"X-User-ID": sample_user.user_id}
        )
        instance_id = instance_response.json()["instance_id"]
        
        # Simulate concurrent config and metrics updates
        config_response = client.put(
            f"/api/v1/agents/instances/{instance_id}/config",
            json={"runtime_config": {"task": "concurrent_test"}},
            headers={"X-User-ID": sample_user.user_id}
        )
        
        metrics_response = client.put(
            f"/api/v1/agents/instances/{instance_id}/metrics",
            json={"metrics": {"concurrent_score": 1.0}},
            headers={"X-User-ID": sample_user.user_id}
        )
        
        # Both should succeed
        assert config_response.status_code == 200
        assert metrics_response.status_code == 200
        
        # Verify both updates were applied
        instance = await repositories["agent"].get_agent_instance(instance_id)
        assert instance.runtime_config["task"] == "concurrent_test"
        assert instance.performance_metrics["concurrent_score"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__])