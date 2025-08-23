import asyncio
from unittest.mock import AsyncMock

import pytest

from agentsmcp.agent_manager import AgentManager
from agentsmcp.agents.base import BaseAgent
from agentsmcp.config import AgentConfig, Config
from agentsmcp.models import JobState


class MockAgent(BaseAgent):
    async def execute_task(self, task: str) -> str:
        return "mock result"


@pytest.fixture
def config():
    """Return a default config."""
    return Config()


@pytest.fixture
async def agent_manager(config):
    """Return an AgentManager instance."""
    manager = AgentManager(config)
    # Replace agent classes with mock
    manager.agent_classes["test"] = MockAgent
    return manager


@pytest.mark.asyncio
async def test_spawn_agent(agent_manager, config):
    """Test spawning an agent."""
    # Add test agent config
    config.agents["test"] = AgentConfig(type="test", model="test-model")

    manager = await agent_manager
    job_id = await manager.spawn_agent("test", "test task", 60)
    assert isinstance(job_id, str)
    
    status = await manager.get_job_status(job_id)
    assert status.state == JobState.PENDING


@pytest.mark.asyncio
async def test_get_job_status(agent_manager, config):
    """Test getting job status."""
    config.agents["test"] = AgentConfig(type="test", model="test-model")

    manager = await agent_manager
    job_id = await manager.spawn_agent("test", "test task")
    status = await manager.get_job_status(job_id)
    
    assert status is not None
    assert status.job_id == job_id


@pytest.mark.asyncio
async def test_cancel_job(agent_manager, config):
    """Test cancelling a job."""
    config.agents["test"] = AgentConfig(type="test", model="test-model")

    manager = await agent_manager
    job_id = await manager.spawn_agent("test", "test task")
    
    # Give the task a moment to start
    await asyncio.sleep(0.1)
    
    success = await manager.cancel_job(job_id)
    assert success
    
    status = await manager.get_job_status(job_id)
    assert status.state == JobState.CANCELLED


@pytest.mark.asyncio
async def test_job_completion(agent_manager, config):
    """Test job completion."""
    config.agents["test"] = AgentConfig(type="test", model="test-model")

    # Mock successful execution
    mock_result = "Task completed successfully"
    
    manager = await agent_manager
    # Replace the agent's execute_task with a mock
    manager.agent_classes["test"].execute_task = AsyncMock(return_value=mock_result)

    job_id = await manager.spawn_agent("test", "test task")
    
    # Wait for completion
    status = await manager.wait_for_completion(job_id)
    
    assert status.state == JobState.COMPLETED
    assert status.output == mock_result
