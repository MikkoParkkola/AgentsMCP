from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from agentsmcp.config import AgentConfig, Config
from agentsmcp.server import AgentServer


@pytest.fixture
def config():
    """Return a default config."""
    cfg = Config()
    cfg.agents['test'] = AgentConfig(type="test", model="test-model")
    return cfg


@pytest.fixture
def agent_server(config):
    """Return an AgentServer instance."""
    return AgentServer(config)


@pytest.fixture
def client(agent_server):
    """Return a TestClient for the server."""
    return TestClient(agent_server.app)


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "AgentsMCP"


def test_health_endpoint(client):
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@patch('agentsmcp.server.AgentManager')
def test_spawn_endpoint(mock_agent_manager_class, client):
    """Test spawn endpoint."""
    # Mock agent manager
    mock_agent_manager = AsyncMock()
    mock_agent_manager.spawn_agent.return_value = "job-123"
    mock_agent_manager_class.return_value = mock_agent_manager

    response = client.post("/spawn", json={
        "agent_type": "test",
        "task": "test task",
        "timeout": 300
    })

    assert response.status_code == 200
    assert response.json()["job_id"] == "job-123"


def test_spawn_endpoint_unknown_agent(client):
    """Test spawn endpoint with unknown agent type."""
    response = client.post("/spawn", json={
        "agent_type": "unknown",
        "task": "test task"
    })

    assert response.status_code == 400


@patch('agentsmcp.server.AgentManager')
def test_status_endpoint(mock_agent_manager_class, client):
    """Test status endpoint."""
    from datetime import datetime

    from agentsmcp.models import JobState, JobStatus

    # Mock agent manager
    mock_agent_manager = AsyncMock()
    mock_status = JobStatus(
        job_id="job-123",
        state=JobState.RUNNING,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    mock_agent_manager.get_job_status.return_value = mock_status
    mock_agent_manager_class.return_value = mock_agent_manager

    response = client.get("/status/job-123")

    assert response.status_code == 200
    assert response.json()["state"] == "RUNNING"


@patch('agentsmcp.server.AgentManager')
def test_cancel_job_endpoint(mock_agent_manager_class, client):
    """Test cancel job endpoint."""
    # Mock agent manager
    mock_agent_manager = AsyncMock()
    mock_agent_manager.cancel_job.return_value = True
    mock_agent_manager_class.return_value = mock_agent_manager

    response = client.delete("/jobs/job-123")

    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"


def test_list_agents_endpoint(client):
    """Test list agents endpoint."""
    response = client.get("/agents")
    assert response.status_code == 200
    assert "agents" in response.json()
    assert "codex" in response.json()["agents"]


def test_cors_headers(agent_server):
    """Test CORS configuration."""
    # Check that CORS middleware is configured
    middlewares = agent_server.app.user_middleware
    cors_found = any(
        hasattr(middleware, 'cls') and
        'cors' in str(middleware.cls).lower()
        for middleware in middlewares
    )
    assert cors_found
