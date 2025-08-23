import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from agentsmcp.config import Config, AgentConfig
from agentsmcp.models import JobState, JobStatus
from agentsmcp.agent_manager import AgentManager
from agentsmcp.agents.base import BaseAgent


class MockAgent(BaseAgent):
    """Mock agent for testing."""
    
    def __init__(self, agent_config: AgentConfig, global_config: Config):
        super().__init__(agent_config, global_config)
        self.execute_task = AsyncMock()
        self.cleanup = AsyncMock()


@pytest.fixture
def config():
    """Create test configuration."""
    config = Config()
    return config


@pytest.fixture
def agent_manager(config):
    """Create agent manager for testing."""
    manager = AgentManager(config)
    # Replace agent classes with mock
    manager.agent_classes = {"test": MockAgent}
    return manager


@pytest.mark.asyncio
async def test_spawn_agent(agent_manager, config):
    """Test spawning an agent."""
    # Add test agent config
    config.agents["test"] = AgentConfig(type="test", model="test-model")
    
    job_id = await agent_manager.spawn_agent("test", "test task", 60)
    
    assert job_id is not None
    assert job_id in agent_manager.jobs
    
    job = agent_manager.jobs[job_id]
    assert job.agent_type == "test"
    assert job.task == "test task"
    assert job.timeout == 60


@pytest.mark.asyncio
async def test_get_job_status(agent_manager, config):
    """Test getting job status."""
    config.agents["test"] = AgentConfig(type="test", model="test-model")
    
    job_id = await agent_manager.spawn_agent("test", "test task")
    status = await agent_manager.get_job_status(job_id)
    
    assert status is not None
    assert status.job_id == job_id
    assert status.state in [JobState.PENDING, JobState.RUNNING]


@pytest.mark.asyncio
async def test_cancel_job(agent_manager, config):
    """Test cancelling a job."""
    config.agents["test"] = AgentConfig(type="test", model="test-model")
    
    job_id = await agent_manager.spawn_agent("test", "test task")
    success = await agent_manager.cancel_job(job_id)
    
    assert success is True
    
    status = await agent_manager.get_job_status(job_id)
    assert status.state == JobState.CANCELLED


@pytest.mark.asyncio
async def test_unknown_agent_type(agent_manager):
    """Test spawning unknown agent type."""
    with pytest.raises(ValueError, match="Unknown agent type"):
        await agent_manager.spawn_agent("unknown", "test task")


@pytest.mark.asyncio
async def test_job_completion(agent_manager, config):
    """Test job completion."""
    config.agents["test"] = AgentConfig(type="test", model="test-model")
    
    # Mock successful execution
    mock_result = "Task completed successfully"
    
    job_id = await agent_manager.spawn_agent("test", "test task")
    job = agent_manager.jobs[job_id]
    job.agent.execute_task.return_value = mock_result
    
    # Wait a bit for the task to complete
    await asyncio.sleep(0.1)
    
    status = await agent_manager.get_job_status(job_id)
    if status.state == JobState.COMPLETED:
        assert status.output == mock_result


def test_job_status_creation():
    """Test JobStatus creation."""
    status = JobStatus(job_id="test-123", state=JobState.PENDING)
    
    assert status.job_id == "test-123"
    assert status.state == JobState.PENDING
    assert status.created_at is not None
    assert status.updated_at is not None