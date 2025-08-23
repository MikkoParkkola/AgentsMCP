import pytest
from pathlib import Path
import tempfile
import yaml

from agentsmcp.config import Config, ServerConfig, AgentConfig, RAGConfig


def test_default_config():
    """Test creating a default configuration."""
    config = Config()
    
    assert config.server.host == "localhost"
    assert config.server.port == 8000
    assert len(config.agents) == 3
    assert "codex" in config.agents
    assert "claude" in config.agents
    assert "ollama" in config.agents


def test_server_config_validation():
    """Test server configuration validation."""
    # Valid port
    server_config = ServerConfig(port=8080)
    assert server_config.port == 8080
    
    # Invalid port
    with pytest.raises(ValueError):
        ServerConfig(port=70000)


def test_agent_config_validation():
    """Test agent configuration validation."""
    # Valid temperature
    agent_config = AgentConfig(type="test", temperature=0.5)
    assert agent_config.temperature == 0.5
    
    # Invalid temperature
    with pytest.raises(ValueError):
        AgentConfig(type="test", temperature=3.0)


def test_rag_config_validation():
    """Test RAG configuration validation."""
    # Valid similarity threshold
    rag_config = RAGConfig(similarity_threshold=0.8)
    assert rag_config.similarity_threshold == 0.8
    
    # Invalid similarity threshold
    with pytest.raises(ValueError):
        RAGConfig(similarity_threshold=1.5)


def test_config_from_file():
    """Test loading configuration from a file."""
    config_data = {
        "server": {"host": "0.0.0.0", "port": 9000},
        "agents": {
            "test": {
                "type": "test",
                "model": "gpt-4",
                "max_tokens": 2000
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = Path(f.name)
    
    try:
        config = Config.from_file(temp_path)
        assert config.server.host == "0.0.0.0"
        assert config.server.port == 9000
        assert "test" in config.agents
    finally:
        temp_path.unlink()


def test_config_save_to_file():
    """Test saving configuration to a file."""
    config = Config()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        temp_path = Path(f.name)
    
    try:
        config.save_to_file(temp_path)
        assert temp_path.exists()
        
        # Load it back and verify
        loaded_config = Config.from_file(temp_path)
        assert loaded_config.server.host == config.server.host
    finally:
        temp_path.unlink()


def test_get_agent_config():
    """Test getting agent configuration."""
    config = Config()
    
    codex_config = config.get_agent_config("codex")
    assert codex_config is not None
    assert codex_config.type == "codex"
    
    missing_config = config.get_agent_config("nonexistent")
    assert missing_config is None


def test_get_tool_config():
    """Test getting tool configuration."""
    config = Config()
    
    fs_config = config.get_tool_config("filesystem")
    assert fs_config is not None
    assert fs_config.name == "filesystem"
    
    missing_config = config.get_tool_config("nonexistent")
    assert missing_config is None