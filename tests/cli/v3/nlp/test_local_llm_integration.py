"""Tests for local LLM integration."""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone

from src.agentsmcp.cli.v3.nlp.local_llm_integration import LocalLLMIntegration
from src.agentsmcp.cli.v3.models.nlp_models import (
    LLMConfig,
    LLMResponse,
    TokenUsage,
    ParsedCommand,
    ParsingMethod,
    LLMUnavailableError,
    ContextTooLargeError,
    ParsingFailedError
)


class TestLocalLLMIntegration:
    """Test local LLM integration functionality."""

    def setup_method(self):
        """Setup test instance."""
        self.config = LLMConfig(
            model_name="test-model",
            max_tokens=1024,
            temperature=0.1,
            timeout_seconds=30.0,
            context_window=8000
        )
        self.integration = LocalLLMIntegration(self.config)

    def test_initialization_default_config(self):
        """Test initialization with default config."""
        integration = LocalLLMIntegration()
        assert integration.config.model_name == "gpt-oss:20b"
        assert len(integration.base_urls) == 2
        assert "127.0.0.1:11435" in integration.base_urls[0]
        assert "localhost:11434" in integration.base_urls[1]

    def test_initialization_custom_config(self):
        """Test initialization with custom config."""
        assert self.integration.config == self.config
        assert "AgentsMCP CLI" in self.integration.system_prompt
        assert "JSON response" in self.integration.system_prompt

    def test_system_prompt_structure(self):
        """Test system prompt contains required elements."""
        prompt = self.integration.system_prompt
        
        # Check for key instructions
        assert "natural language command parser" in prompt.lower()
        assert "json" in prompt.lower()
        assert "action" in prompt.lower()
        assert "parameters" in prompt.lower()
        assert "confidence" in prompt.lower()
        assert "explanation" in prompt.lower()
        
        # Check for supported actions
        supported_actions = ["analyze", "help", "status", "tui", "init", "optimize", "run", "file", "settings", "dashboard"]
        for action in supported_actions:
            assert action in prompt.lower()
        
        # Check for examples
        assert "examples:" in prompt.lower()

    @pytest.mark.asyncio
    async def test_parse_command_no_httpx(self):
        """Test parse_command when httpx is not available."""
        with patch('src.agentsmcp.cli.v3.nlp.local_llm_integration.httpx', None):
            integration = LocalLLMIntegration(self.config)
            
            with pytest.raises(LLMUnavailableError, match="httpx library not available"):
                await integration.parse_command("analyze my code")

    @pytest.mark.asyncio
    async def test_parse_command_context_too_large(self):
        """Test parse_command with context too large."""
        # Create very large input
        large_input = "analyze " + "x" * 10000
        
        with pytest.raises(ContextTooLargeError):
            await self.integration.parse_command(large_input)

    def test_prepare_prompt_basic(self):
        """Test basic prompt preparation."""
        prompt = self.integration._prepare_prompt("analyze my code")
        
        assert "analyze my code" in prompt
        assert "Parse this natural language command" in prompt
        assert "JSON only" in prompt

    def test_prepare_prompt_with_context(self):
        """Test prompt preparation with context."""
        context = {
            "command_history": ["help", "status", "analyze", "tui", "init"],
            "current_directory": "/home/user/project",
            "recent_files": ["main.py", "utils.py", "config.json"]
        }
        
        prompt = self.integration._prepare_prompt("analyze my code", context)
        
        assert "analyze my code" in prompt
        assert "Recent commands:" in prompt
        assert "init" in prompt  # Should include last 5 commands
        assert "/home/user/project" in prompt
        assert "Recent files:" in prompt
        assert "config.json" in prompt  # Should include last 3 files

    def test_prepare_prompt_context_limits(self):
        """Test prompt preparation respects context limits."""
        context = {
            "command_history": [f"command{i}" for i in range(10)],
            "recent_files": [f"file{i}.py" for i in range(10)]
        }
        
        prompt = self.integration._prepare_prompt("test", context)
        
        # Should only include last 5 commands
        assert "command9" in prompt
        assert "command5" in prompt
        assert "command4" not in prompt
        
        # Should only include last 3 files
        assert "file9.py" in prompt
        assert "file7.py" in prompt
        assert "file6.py" not in prompt

    @pytest.mark.asyncio
    async def test_make_ollama_request_success(self):
        """Test successful Ollama API request."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {
                "content": '{"action": "analyze", "parameters": {"target": "code"}, "confidence": 0.85, "explanation": "I understood this as a request to analyze code."}'
            }
        }
        
        messages = [
            {"role": "system", "content": self.integration.system_prompt},
            {"role": "user", "content": "analyze my code"}
        ]
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            
            result = await self.integration._make_ollama_request("http://localhost:11434", messages)
            
            assert result is not None
            assert isinstance(result, LLMResponse)
            assert "analyze" in result.content
            assert result.model_name == "test-model"
            assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_make_ollama_request_failure(self):
        """Test failed Ollama API request."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        
        messages = [{"role": "user", "content": "test"}]
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            
            result = await self.integration._make_ollama_request("http://localhost:11434", messages)
            
            assert result is None

    @pytest.mark.asyncio
    async def test_make_ollama_request_timeout(self):
        """Test Ollama API request timeout."""
        import httpx
        
        messages = [{"role": "user", "content": "test"}]
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.TimeoutException("Request timed out")
            )
            
            result = await self.integration._make_ollama_request("http://localhost:11434", messages)
            
            assert result is None

    @pytest.mark.asyncio
    async def test_parse_llm_response_valid_json(self):
        """Test parsing valid LLM response."""
        llm_response = LLMResponse(
            content='{"action": "analyze", "parameters": {"target": "code"}, "confidence": 0.85, "explanation": "I understood this as analyzing code."}',
            finish_reason="stop",
            model_name="test-model"
        )
        
        command, explanation = await self.integration._parse_llm_response(llm_response)
        
        assert command is not None
        assert isinstance(command, ParsedCommand)
        assert command.action == "analyze"
        assert command.parameters == {"target": "code"}
        assert command.confidence == 0.85
        assert command.method == ParsingMethod.LLM
        assert explanation == "I understood this as analyzing code."

    @pytest.mark.asyncio
    async def test_parse_llm_response_json_in_text(self):
        """Test parsing LLM response with JSON embedded in text."""
        llm_response = LLMResponse(
            content='Here is the parsed command: {"action": "help", "parameters": {}, "confidence": 0.95, "explanation": "User wants help."} Hope this helps!',
            finish_reason="stop",
            model_name="test-model"
        )
        
        command, explanation = await self.integration._parse_llm_response(llm_response)
        
        assert command is not None
        assert command.action == "help"
        assert command.confidence == 0.95

    @pytest.mark.asyncio
    async def test_parse_llm_response_markdown_json(self):
        """Test parsing LLM response with JSON in markdown code block."""
        llm_response = LLMResponse(
            content='```json\n{"action": "status", "parameters": {}, "confidence": 0.9, "explanation": "Check status"}\n```',
            finish_reason="stop",
            model_name="test-model"
        )
        
        command, explanation = await self.integration._parse_llm_response(llm_response)
        
        assert command is not None
        assert command.action == "status"
        assert command.confidence == 0.9

    @pytest.mark.asyncio
    async def test_parse_llm_response_invalid_json(self):
        """Test parsing invalid JSON response."""
        llm_response = LLMResponse(
            content='{"action": "analyze", invalid json',
            finish_reason="stop",
            model_name="test-model"
        )
        
        command, explanation = await self.integration._parse_llm_response(llm_response)
        
        assert command is None
        assert "Failed to parse" in explanation

    @pytest.mark.asyncio
    async def test_parse_llm_response_missing_required_fields(self):
        """Test parsing response missing required fields."""
        llm_response = LLMResponse(
            content='{"parameters": {"target": "code"}, "confidence": 0.85}',  # Missing action
            finish_reason="stop",
            model_name="test-model"
        )
        
        command, explanation = await self.integration._parse_llm_response(llm_response)
        
        assert command is None
        assert "Missing or invalid 'action' field" in explanation

    @pytest.mark.asyncio
    async def test_parse_llm_response_validation_and_defaults(self):
        """Test response validation and default values."""
        llm_response = LLMResponse(
            content='{"action": "  analyze  ", "confidence": 1.5, "explanation": 123}',  # Invalid values
            finish_reason="stop",
            model_name="test-model"
        )
        
        command, explanation = await self.integration._parse_llm_response(llm_response)
        
        if command:
            # Should clean and validate values
            assert command.action.strip() == command.action
            assert 0.0 <= command.confidence <= 1.0
            assert isinstance(command.parameters, dict)

    def test_clean_json_string(self):
        """Test JSON string cleaning."""
        # Test markdown removal
        json_str = '```json\n{"action": "test"}\n```'
        cleaned = self.integration._clean_json_string(json_str)
        assert '```' not in cleaned
        assert 'json' not in cleaned or cleaned.count('json') <= 1
        
        # Test whitespace removal
        json_str = '   {"action": "test"}   '
        cleaned = self.integration._clean_json_string(json_str)
        assert cleaned == '{"action": "test"}'

    @pytest.mark.asyncio
    async def test_check_availability_success(self):
        """Test availability check success."""
        mock_response = Mock()
        mock_response.status_code = 200
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
            
            available = await self.integration.check_availability()
            assert available is True

    @pytest.mark.asyncio
    async def test_check_availability_failure(self):
        """Test availability check failure."""
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=Exception("Connection failed")
            )
            
            available = await self.integration.check_availability()
            assert available is False

    @pytest.mark.asyncio
    async def test_check_availability_no_httpx(self):
        """Test availability check without httpx."""
        with patch('src.agentsmcp.cli.v3.nlp.local_llm_integration.httpx', None):
            integration = LocalLLMIntegration(self.config)
            available = await integration.check_availability()
            assert available is False

    @pytest.mark.asyncio
    async def test_get_model_info_success(self):
        """Test getting model info successfully."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "name": "test-model",
            "size": "7B",
            "family": "llama"
        }
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            
            info = await self.integration.get_model_info()
            assert "name" in info
            assert info["name"] == "test-model"

    @pytest.mark.asyncio
    async def test_get_model_info_failure(self):
        """Test getting model info failure."""
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=Exception("Connection failed")
            )
            
            info = await self.integration.get_model_info()
            assert "error" in info

    def test_update_config(self):
        """Test updating LLM configuration."""
        new_config = LLMConfig(
            model_name="new-model",
            temperature=0.5
        )
        
        self.integration.update_config(new_config)
        assert self.integration.config == new_config
        assert self.integration.config.model_name == "new-model"
        assert self.integration.config.temperature == 0.5

    def test_estimate_tokens(self):
        """Test token estimation."""
        # Test simple text
        text = "analyze my code"
        tokens = self.integration.estimate_tokens(text)
        assert tokens > 0
        assert tokens > len(text.split())  # Should be slightly more than word count
        
        # Test empty text
        assert self.integration.estimate_tokens("") >= 1
        
        # Test longer text
        long_text = " ".join(["word"] * 100)
        long_tokens = self.integration.estimate_tokens(long_text)
        short_tokens = self.integration.estimate_tokens("word word")
        assert long_tokens > short_tokens

    def test_is_context_too_large(self):
        """Test context size checking."""
        # Small text should be fine
        small_text = "analyze my code"
        assert not self.integration.is_context_too_large(small_text)
        
        # Very large text should exceed limit
        large_text = " ".join(["word"] * 10000)
        assert self.integration.is_context_too_large(large_text)

    @pytest.mark.asyncio
    async def test_test_connection_success(self):
        """Test connection testing success."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"version": "0.1.0"}
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
            
            result = await self.integration.test_connection()
            
            assert result["available"] is True
            assert "model" in result
            assert "endpoints" in result
            assert "config" in result
            assert len(result["endpoints"]) > 0

    @pytest.mark.asyncio
    async def test_test_connection_failure(self):
        """Test connection testing failure."""
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=Exception("Connection failed")
            )
            
            result = await self.integration.test_connection()
            
            assert result["available"] is False
            assert len(result["endpoints"]) > 0
            # All endpoints should show as unavailable
            assert all(not ep.get("available", False) for ep in result["endpoints"])

    @pytest.mark.asyncio
    async def test_test_connection_no_httpx(self):
        """Test connection testing without httpx."""
        with patch('src.agentsmcp.cli.v3.nlp.local_llm_integration.httpx', None):
            integration = LocalLLMIntegration(self.config)
            result = await integration.test_connection()
            
            assert result["available"] is False
            assert "httpx library not installed" in result["error"]
            assert result["endpoints"] == []

    @pytest.mark.asyncio
    async def test_full_parse_command_integration(self):
        """Test full parse_command integration."""
        # Mock successful LLM response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {
                "content": '{"action": "analyze", "parameters": {"target": "code"}, "confidence": 0.85, "explanation": "I understood this as a request to analyze code."}'
            }
        }
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            
            command, explanation = await self.integration.parse_command("analyze my code")
            
            assert command is not None
            assert isinstance(command, ParsedCommand)
            assert command.action == "analyze"
            assert command.parameters == {"target": "code"}
            assert command.confidence == 0.85
            assert command.method == ParsingMethod.LLM
            assert "analyze code" in explanation

    @pytest.mark.asyncio
    async def test_parse_command_with_context(self):
        """Test parse_command with conversation context."""
        context = {
            "command_history": ["help", "status"],
            "current_directory": "/project",
            "recent_files": ["main.py"]
        }
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {
                "content": '{"action": "analyze", "parameters": {"target": "main.py"}, "confidence": 0.9, "explanation": "Analyzing the recent file main.py"}'
            }
        }
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            
            command, explanation = await self.integration.parse_command("analyze it", context)
            
            assert command is not None
            assert command.action == "analyze"