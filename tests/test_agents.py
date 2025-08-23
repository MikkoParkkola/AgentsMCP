
import pytest

from agentsmcp.agents.base import BaseAgent
from agentsmcp.agents.claude_agent import ClaudeAgent
from agentsmcp.agents.codex_agent import CodexAgent
from agentsmcp.agents.ollama_agent import OllamaAgent
from agentsmcp.config import AgentConfig, Config


@pytest.fixture
def config():
    """Create test configuration."""
    return Config()


@pytest.fixture
def agent_config():
    """Create test agent configuration."""
    return AgentConfig(
        type="test",
        model="test-model",
        max_tokens=1000,
        temperature=0.7,
        tools=["filesystem", "git"]
    )


def test_base_agent_initialization(agent_config, config):
    """Test base agent initialization."""
    class TestAgent(BaseAgent):
        async def execute_task(self, task: str) -> str:
            return "test result"
    
    agent = TestAgent(agent_config, config)
    
    assert agent.agent_config == agent_config
    assert agent.global_config == config
    assert agent.get_model() == "test-model"
    assert agent.get_max_tokens() == 1000
    assert agent.get_temperature() == 0.7


@pytest.mark.asyncio
async def test_codex_agent_fallback(agent_config, config):
    """Test CodexAgent fallback when MCP is not available."""
    agent = CodexAgent(agent_config, config)
    
    result = await agent.execute_task("Write a hello world function")
    
    assert "Codex Simulation" in result
    assert "hello world function" in result


@pytest.mark.asyncio
async def test_claude_agent_simulation(agent_config, config):
    """Test ClaudeAgent simulation."""
    agent = ClaudeAgent(agent_config, config)
    
    result = await agent.execute_task("Analyze this code")
    
    assert "Claude Agent Analysis" in result
    assert "Analyze this code" in result
    assert "test-model" in result


@pytest.mark.asyncio
async def test_ollama_agent_fallback(agent_config, config):
    """Test OllamaAgent fallback when MCP is not available."""
    agent = OllamaAgent(agent_config, config)
    
    result = await agent.execute_task("Explain Python")
    
    assert "Ollama Agent" in result
    assert "Explain Python" in result
    assert "test-model" in result


def test_agent_system_prompt(agent_config, config):
    """Test system prompt handling."""
    agent_config.system_prompt = "You are a test assistant"
    
    class TestAgent(BaseAgent):
        async def execute_task(self, task: str) -> str:
            return "test"
    
    agent = TestAgent(agent_config, config)
    assert agent.get_system_prompt() == "You are a test assistant"
    
    # Test default system prompt
    agent_config.system_prompt = None
    agent = TestAgent(agent_config, config)
    assert "helpful AI assistant" in agent.get_system_prompt()


def test_agent_tools_initialization(agent_config, config):
    """Test that agent tools are properly initialized."""
    class TestAgent(BaseAgent):
        async def execute_task(self, task: str) -> str:
            return "test"
    
    agent = TestAgent(agent_config, config)
    
    # Should have filesystem and git tools (from agent_config.tools)
    tool_names = [tool.name for tool in agent.tools]
    assert "filesystem" in tool_names


@pytest.mark.asyncio
async def test_agent_cleanup():
    """Test agent cleanup."""
    class TestAgent(BaseAgent):
        async def execute_task(self, task: str) -> str:
            return "test"
        
        async def cleanup(self):
            self.cleaned_up = True
    
    agent = TestAgent(AgentConfig(type="test"), Config())
    await agent.cleanup()
    assert hasattr(agent, 'cleaned_up') and agent.cleaned_up


@pytest.mark.asyncio 
async def test_codex_agent_cleanup(agent_config, config):
    """Test CodexAgent cleanup."""
    agent = CodexAgent(agent_config, config)
    agent.session_id = "test-session"
    
    await agent.cleanup()
    assert agent.session_id is None