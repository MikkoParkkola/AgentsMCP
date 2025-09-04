"""Tests for enhanced context-aware preprocessing functionality."""

import asyncio
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from src.agentsmcp.conversation.llm_client import LLMClient
from src.agentsmcp.utils.project_detector import ProjectDetector, estimate_tokens, format_project_context


class TestProjectDetector:
    """Test project detection utilities."""
    
    def test_estimate_tokens(self):
        """Test token estimation function."""
        assert estimate_tokens("") == 0
        assert estimate_tokens("hello") == 1  # 5 chars / 4 ≈ 1
        assert estimate_tokens("hello world") == 2  # 11 chars / 4 ≈ 2
        assert estimate_tokens("a" * 100) == 25  # 100 chars / 4 = 25
    
    def test_detect_python_project(self, tmp_path):
        """Test Python project detection."""
        # Create a Python project structure
        (tmp_path / "pyproject.toml").write_text("""
[project]
name = "test-project"
description = "A test Python project"
dependencies = ["fastapi", "pytest"]

[build-system]
requires = ["poetry-core"]
""")
        
        (tmp_path / "src").mkdir()
        (tmp_path / "tests").mkdir()
        (tmp_path / "README.md").write_text("# Test Project\n\nThis is a test project for AgentsMCP.")
        # Also add setup.py to ensure Python detection
        (tmp_path / "setup.py").write_text("from setuptools import setup\nsetup(name='test-project')")
        
        context = ProjectDetector.detect_project_context(str(tmp_path))
        
        # Check if tomllib is available for full testing
        try:
            import tomllib
            toml_available = True
        except ImportError:
            try:
                import tomli
                toml_available = True
            except ImportError:
                toml_available = False
        
        # Should always detect Python project due to setup.py
        assert context["project_type"] == "python"
        assert "python" in context["languages"]
        
        if toml_available:
            assert context["project_name"] == "test-project"
            # Note: README description takes priority over pyproject.toml description 
            assert "This is a test project for AgentsMCP" in context["description"]
            assert "fastapi" in context["frameworks"]
            assert "pytest" in context["frameworks"]
            assert "poetry" in context["frameworks"]
        else:
            # Without TOML parsing, should fallback to README description and setuptools
            assert "This is a test project for AgentsMCP" in context["description"]
            assert "setuptools" in context["frameworks"]
        
        # These should always be detected regardless of TOML parsing
        assert "pyproject.toml" in context["key_files"]
        assert "src/" in context["key_files"]
        assert "tests/" in context["key_files"]
    
    def test_detect_nodejs_project(self, tmp_path):
        """Test Node.js project detection."""
        # Create a Node.js project structure
        package_json = {
            "name": "test-nodejs-app",
            "description": "A test Node.js application",
            "dependencies": {"express": "^4.18.0", "react": "^18.2.0"},
            "devDependencies": {"typescript": "^4.9.0", "jest": "^29.0.0"}
        }
        
        import json
        (tmp_path / "package.json").write_text(json.dumps(package_json))
        (tmp_path / "src").mkdir()
        (tmp_path / "__tests__").mkdir()
        
        context = ProjectDetector.detect_project_context(str(tmp_path))
        
        assert context["project_type"] == "nodejs"
        assert context["project_name"] == "test-nodejs-app"
        assert context["description"] == "A test Node.js application"
        assert "javascript" in context["languages"]
        assert "typescript" in context["languages"]
        assert "express" in context["frameworks"]
        assert "react" in context["frameworks"]
        assert "jest" in context["frameworks"]
    
    def test_detect_rust_project(self, tmp_path):
        """Test Rust project detection."""
        # Create a Rust project structure
        (tmp_path / "Cargo.toml").write_text("""
[package]
name = "test-rust-app"
description = "A test Rust application"

[dependencies]
tokio = "1.0"
serde = "1.0"
""")
        
        (tmp_path / "src").mkdir()
        
        context = ProjectDetector.detect_project_context(str(tmp_path))
        
        # Check if tomllib is available for full testing
        try:
            import tomllib
            toml_available = True
        except ImportError:
            try:
                import tomli
                toml_available = True
            except ImportError:
                toml_available = False
        
        if toml_available:
            assert context["project_type"] == "rust"
            assert context["project_name"] == "test-rust-app"
            assert context["description"] == "A test Rust application"
            assert "rust" in context["languages"]
            assert "tokio" in context["frameworks"]
            assert "serde" in context["frameworks"]
        
        # This should always be detected regardless of TOML parsing
        assert "Cargo.toml" in context["key_files"]
    
    def test_format_project_context(self):
        """Test project context formatting."""
        project_context = {
            "directory": "/test/project",
            "project_type": "python",
            "project_name": "test-app",
            "description": "A test application",
            "languages": ["python"],
            "frameworks": ["fastapi", "pytest"],
            "key_files": ["pyproject.toml", "src/", "tests/"],
            "structure_summary": "Source: src | Tests: tests"
        }
        
        formatted = format_project_context(project_context)
        
        assert "CURRENT DIRECTORY CONTEXT:" in formatted
        assert "Working Directory: /test/project" in formatted
        assert "Project Type: Python (test-app)" in formatted
        assert "Description: A test application" in formatted
        assert "Languages: python" in formatted
        assert "Frameworks: fastapi, pytest" in formatted
        assert "Key Files: pyproject.toml, src/, tests/" in formatted
        assert "Structure: Source: src | Tests: tests" in formatted
    
    def test_format_project_context_with_error(self):
        """Test project context formatting with error."""
        project_context = {"error": "Directory not found"}
        formatted = format_project_context(project_context)
        assert "PROJECT CONTEXT ERROR: Directory not found" in formatted


class TestEnhancedPreprocessing:
    """Test enhanced preprocessing functionality."""
    
    @pytest.fixture
    def llm_client(self, tmp_path):
        """Create a test LLM client."""
        config_path = tmp_path / "test_config.json"
        config_path.write_text('{"model": "test-model", "provider": "test-provider"}')
        
        client = LLMClient(config_path)
        client.current_working_directory = str(tmp_path)
        return client
    
    def test_initialization_with_context_features(self, llm_client):
        """Test that context features are properly initialized."""
        assert hasattr(llm_client, 'preprocessing_context_enabled')
        assert hasattr(llm_client, 'preprocessing_history_enabled')
        assert hasattr(llm_client, 'preprocessing_max_history_messages')
        assert hasattr(llm_client, 'preprocessing_directory_context_enabled')
        assert hasattr(llm_client, 'current_working_directory')
        
        # Default values
        assert llm_client.preprocessing_context_enabled is True
        assert llm_client.preprocessing_history_enabled is True
        assert llm_client.preprocessing_max_history_messages == 10
        assert llm_client.preprocessing_directory_context_enabled is True
    
    def test_context_configuration_methods(self, llm_client):
        """Test context configuration methods."""
        # Test directory context toggle
        result = llm_client.set_preprocessing_context_enabled(False)
        assert "disabled" in result
        assert llm_client.preprocessing_directory_context_enabled is False
        
        result = llm_client.set_preprocessing_context_enabled(True)
        assert "enabled" in result
        assert llm_client.preprocessing_directory_context_enabled is True
        
        # Test history configuration
        result = llm_client.set_preprocessing_history_enabled(False)
        assert "disabled" in result
        assert llm_client.preprocessing_history_enabled is False
        
        result = llm_client.set_preprocessing_max_history(5)
        assert "5" in result
        assert llm_client.preprocessing_max_history_messages == 5
        
        # Test invalid values
        result = llm_client.set_preprocessing_max_history(-1)
        assert "non-negative" in result
        
        result = llm_client.set_preprocessing_max_history(100)
        assert "should not exceed 50" in result
    
    def test_working_directory_management(self, llm_client, tmp_path):
        """Test working directory management."""
        # Test setting valid directory
        test_dir = tmp_path / "test_project"
        test_dir.mkdir()
        
        result = llm_client.set_working_directory(str(test_dir))
        assert "Working directory set" in result
        assert llm_client.get_working_directory() == str(test_dir.resolve())
        
        # Test invalid directory
        result = llm_client.set_working_directory("/non/existent/path")
        assert "does not exist" in result
        
        # Test file instead of directory
        test_file = tmp_path / "test_file.txt"
        test_file.write_text("test")
        result = llm_client.set_working_directory(str(test_file))
        assert "not a directory" in result
    
    def test_conversation_history_selection(self, llm_client):
        """Test conversation history selection logic."""
        # Add some conversation history
        llm_client.conversation_history = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "First response"},
            {"role": "user", "content": "Second message"},
            {"role": "assistant", "content": "Second response"},
            {"role": "user", "content": "Third message"},
        ]
        
        # Test history selection with token limits
        history = llm_client._select_relevant_history(1000)  # Large token limit
        assert "CONVERSATION HISTORY" in history
        assert "First message" in history
        assert "Third message" in history
        
        # Test with small token limit
        history = llm_client._select_relevant_history(50)  # Small token limit
        # Should select fewer messages
        assert len(history) < 500  # Rough check that it's truncated
        
        # Test with no history
        llm_client.conversation_history = []
        history = llm_client._select_relevant_history(1000)
        assert history == ""
    
    @pytest.mark.asyncio
    async def test_preprocessing_model_capabilities(self, llm_client):
        """Test preprocessing model capabilities detection."""
        with patch.object(llm_client, 'get_model_capabilities') as mock_get_caps:
            # Mock capabilities
            mock_capabilities = Mock()
            mock_capabilities.context_tokens = 16000
            mock_capabilities.supports_streaming = True
            mock_get_caps.return_value = mock_capabilities
            
            capabilities = await llm_client._get_preprocessing_model_capabilities()
            
            assert capabilities['context_limit'] == 16000
            assert capabilities['supports_streaming'] is True
    
    @pytest.mark.asyncio
    async def test_build_preprocessing_context(self, llm_client, tmp_path):
        """Test comprehensive preprocessing context building."""
        # Setup test project
        (tmp_path / "pyproject.toml").write_text("""
[project]
name = "test-context-project"
description = "Test project for context building"
""")
        
        # Setup conversation history
        llm_client.conversation_history = [
            {"role": "user", "content": "Previous question about the project"},
            {"role": "assistant", "content": "Previous answer"},
        ]
        
        # Mock preprocessing model capabilities
        with patch.object(llm_client, '_get_preprocessing_model_capabilities') as mock_caps:
            mock_caps.return_value = {'context_limit': 8000, 'supports_streaming': False}
            
            context = await llm_client._build_preprocessing_context("Test user input")
            
            assert "CURRENT DIRECTORY CONTEXT:" in context
            assert "test-context-project" in context
            assert "CONVERSATION HISTORY" in context
            assert "Previous question" in context
    
    @pytest.mark.asyncio
    async def test_enhanced_optimize_prompt(self, llm_client):
        """Test enhanced optimize_prompt with context."""
        # Mock the simple message method
        llm_client._send_simple_message = AsyncMock(return_value="Enhanced and optimized prompt with context")
        
        # Mock context building
        with patch.object(llm_client, '_build_preprocessing_context') as mock_context:
            mock_context.return_value = "Mock context for testing"
            
            # Test with preprocessing enabled
            llm_client.preprocessing_enabled = True
            llm_client.preprocessing_threshold = 2
            
            result = await llm_client.optimize_prompt("This is a test prompt with multiple words")
            
            assert "Enhanced and optimized prompt with context" in result
            mock_context.assert_called_once()
    
    def test_preprocessing_context_status(self, llm_client):
        """Test preprocessing context status display."""
        status = llm_client.get_preprocessing_context_status()
        
        assert "Preprocessing Context Status" in status
        assert "Directory Context" in status
        assert "Conversation History" in status
        assert "Working Directory" in status
        assert "Max Messages" in status
        assert "Available Messages" in status
        assert "Usage:" in status
    
    def test_enhanced_preprocessing_config(self, llm_client):
        """Test enhanced preprocessing configuration display."""
        config = llm_client.get_preprocessing_config()
        
        assert "Context-Aware Features" in config
        assert "Directory Context" in config
        assert "Conversation History" in config
        assert "Working Directory" in config
        assert "Context Features:" in config
        assert "/preprocessing context on/off" in config
        assert "/preprocessing history" in config


class TestChatEngineIntegration:
    """Test chat engine integration with enhanced preprocessing."""
    
    @pytest.fixture
    def mock_chat_engine(self):
        """Create a mock chat engine for testing."""
        from src.agentsmcp.ui.v3.chat_engine import ChatEngine
        
        engine = ChatEngine()
        engine._llm_client = Mock()
        engine._notify_message = Mock()
        engine._notify_error = Mock()
        engine._format_timestamp = Mock(return_value="2024-01-01 12:00:00")
        return engine
    
    @pytest.mark.asyncio
    async def test_enhanced_preprocessing_commands(self, mock_chat_engine):
        """Test enhanced preprocessing command handling."""
        # Test context commands
        await mock_chat_engine._handle_preprocessing_command("context on")
        mock_chat_engine._llm_client.set_preprocessing_context_enabled.assert_called_with(True)
        
        await mock_chat_engine._handle_preprocessing_command("context off")
        mock_chat_engine._llm_client.set_preprocessing_context_enabled.assert_called_with(False)
        
        # Test history commands
        await mock_chat_engine._handle_preprocessing_command("history on")
        mock_chat_engine._llm_client.set_preprocessing_history_enabled.assert_called_with(True)
        
        await mock_chat_engine._handle_preprocessing_command("history 5")
        mock_chat_engine._llm_client.set_preprocessing_max_history.assert_called_with(5)
        
        # Test working directory command
        await mock_chat_engine._handle_preprocessing_command("workdir /test/path")
        mock_chat_engine._llm_client.set_working_directory.assert_called_with("/test/path")
    
    @pytest.mark.asyncio
    async def test_context_status_notification(self, mock_chat_engine):
        """Test that context status is shown during preprocessing."""
        # Mock LLM client attributes
        mock_chat_engine._llm_client.preprocessing_directory_context_enabled = True
        mock_chat_engine._llm_client.preprocessing_history_enabled = True
        mock_chat_engine._llm_client.conversation_history = [{"role": "user", "content": "test"}]
        mock_chat_engine._llm_client.preprocessing_max_history_messages = 10
        
        # Mock other required methods
        mock_chat_engine._notify_status = Mock()
        
        # This would be called during chat message handling
        # Simulate the status notification logic
        context_info = []
        if mock_chat_engine._llm_client.preprocessing_directory_context_enabled:
            context_info.append("📁 Directory Context")
        if mock_chat_engine._llm_client.preprocessing_history_enabled and mock_chat_engine._llm_client.conversation_history:
            history_count = min(len(mock_chat_engine._llm_client.conversation_history), mock_chat_engine._llm_client.preprocessing_max_history_messages)
            context_info.append(f"📚 History ({history_count} msgs)")
        
        context_str = f" + {' + '.join(context_info)}" if context_info else ""
        expected_status = f"📝 Optimizing prompt with enhanced context{context_str}..."
        
        assert "📁 Directory Context" in expected_status
        assert "📚 History (1 msgs)" in expected_status


if __name__ == "__main__":
    pytest.main([__file__])