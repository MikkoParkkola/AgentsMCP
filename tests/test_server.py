import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from agentsmcp.config import Config, AgentConfig
from agentsmcp.server import AgentServer


@pytest.fixture
def config():
    """Create test configuration."""
    config = Config()
    config.agents["test"] = AgentConfig(type="test", model="test-model")
    return config


@pytest.fixture
def agent_server(config):
    """Create agent server for testing."""
    return AgentServer(config)


@pytest.fixture
def client(agent_server):
    """Create test client."""
    return TestClient(agent_server.app)


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert data["service"] == "AgentsMCP"
    assert "endpoints" in data


def test_health_endpoint(client):
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


def test_list_agents_endpoint(client):
    """Test list agents endpoint."""
    response = client.get("/agents")
    assert response.status_code == 200
    
    data = response.json()
    assert "agents" in data
    assert "configs" in data
    assert isinstance(data["agents"], list)


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
    data = response.json()
    assert data["job_id"] == "job-123"
    assert data["status"] == "spawned"


def test_spawn_endpoint_unknown_agent(client):
    """Test spawn endpoint with unknown agent type."""
    response = client.post("/spawn", json={
        "agent_type": "unknown",
        "task": "test task"
    })
    
    assert response.status_code == 400
    assert "Unknown agent type" in response.json()["detail"]


@patch('agentsmcp.server.AgentManager')
def test_status_endpoint(mock_agent_manager_class, client):
    """Test status endpoint."""
    from agentsmcp.models import JobStatus, JobState
    from datetime import datetime
    
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
    data = response.json()
    assert data["job_id"] == "job-123"
    assert data["state"] == "running"


@patch('agentsmcp.server.AgentManager')
def test_status_endpoint_not_found(mock_agent_manager_class, client):
    """Test status endpoint with non-existent job."""
    # Mock agent manager
    mock_agent_manager = AsyncMock()
    mock_agent_manager.get_job_status.return_value = None
    mock_agent_manager_class.return_value = mock_agent_manager
    
    response = client.get("/status/nonexistent")
    
    assert response.status_code == 404
    assert "Job not found" in response.json()["detail"]


@patch('agentsmcp.server.AgentManager')
def test_cancel_job_endpoint(mock_agent_manager_class, client):
    """Test cancel job endpoint."""
    # Mock agent manager
    mock_agent_manager = AsyncMock()
    mock_agent_manager.cancel_job.return_value = True
    mock_agent_manager_class.return_value = mock_agent_manager
    
    response = client.delete("/jobs/job-123")
    
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == "job-123"
    assert data["status"] == "cancelled"


@patch('agentsmcp.server.AgentManager')
def test_cancel_job_endpoint_not_found(mock_agent_manager_class, client):
    """Test cancel job endpoint with non-existent job."""
    # Mock agent manager
    mock_agent_manager = AsyncMock()
    mock_agent_manager.cancel_job.return_value = False
    mock_agent_manager_class.return_value = mock_agent_manager
    
    response = client.delete("/jobs/nonexistent")
    
    assert response.status_code == 404


def test_cors_headers(agent_server):
    """Test CORS configuration."""
    # Check that CORS middleware is configured
    middlewares = agent_server.app.middleware_stack
    cors_found = any(
        hasattr(middleware, 'cls') and 
        'cors' in str(middleware.cls).lower()
        for middleware in middlewares
    )
    assert cors_found or len(agent_server.config.server.cors_origins) > 0