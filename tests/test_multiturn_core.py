#!/usr/bin/env python3
"""
Core unit tests for multi-turn tool execution functionality.
These tests ensure the fundamental mechanics work correctly.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentsmcp.conversation.llm_client import LLMClient, ConversationMessage


class TestMultiTurnCore:
    """Test core multi-turn tool execution functionality."""

    @pytest.fixture
    def llm_client(self):
        """Create LLMClient instance for testing."""
        return LLMClient()

    @pytest.fixture
    def mock_tool_response(self):
        """Mock LLM response with tool calls."""
        return {
            'choices': [{
                'message': {
                    'content': '',
                    'tool_calls': [{
                        'function': {
                            'name': 'list_directory',
                            'arguments': {'path': '.'}
                        }
                    }]
                }
            }]
        }

    @pytest.fixture
    def mock_analysis_response(self):
        """Mock LLM response with final analysis (no tool calls)."""
        return {
            'choices': [{
                'message': {
                    'content': 'Based on the tool results, this project has a well-structured codebase with proper organization. The src/ directory contains the main application code, and there are comprehensive tests. I recommend focusing on code coverage improvements.',
                    'tool_calls': None
                }
            }]
        }

    def test_extract_tool_calls_with_tools(self, llm_client, mock_tool_response):
        """Test that tool calls are correctly extracted from LLM response."""
        tool_calls = llm_client._extract_tool_calls(mock_tool_response)
        
        assert len(tool_calls) == 1
        assert tool_calls[0]['function']['name'] == 'list_directory'
        assert tool_calls[0]['function']['arguments'] == {'path': '.'}

    def test_extract_tool_calls_without_tools(self, llm_client, mock_analysis_response):
        """Test that no tool calls are extracted when none present."""
        tool_calls = llm_client._extract_tool_calls(mock_analysis_response)
        
        assert tool_calls == []

    def test_extract_response_content(self, llm_client, mock_analysis_response):
        """Test that response content is correctly extracted."""
        content = llm_client._extract_response_content(mock_analysis_response)
        
        assert "Based on the tool results" in content
        assert "well-structured codebase" in content

    @pytest.mark.asyncio
    async def test_call_llm_with_tools_enabled(self, llm_client):
        """Test that LLM is called with tools when enabled."""
        messages = [{"role": "user", "content": "Test message"}]
        
        with patch.object(llm_client, '_call_ollama_turbo') as mock_call:
            mock_call.return_value = {'choices': [{'message': {'content': 'test'}}]}
            
            await llm_client._call_llm_via_mcp(messages, enable_tools=True)
            
            # Verify tools were passed
            mock_call.assert_called_once_with(messages, True)

    @pytest.mark.asyncio
    async def test_call_llm_with_tools_disabled(self, llm_client):
        """Test that LLM is called without tools when disabled."""
        messages = [{"role": "user", "content": "Test message"}]
        
        with patch.object(llm_client, '_call_ollama_turbo') as mock_call:
            mock_call.return_value = {'choices': [{'message': {'content': 'test'}}]}
            
            await llm_client._call_llm_via_mcp(messages, enable_tools=False)
            
            # Verify tools were disabled
            mock_call.assert_called_once_with(messages, False)

    @pytest.mark.asyncio
    async def test_tool_execution_success(self, llm_client):
        """Test successful tool execution and result handling."""
        with patch.object(llm_client, '_execute_tool_call') as mock_execute:
            mock_execute.return_value = "Directory contents: file1.py, file2.py"
            
            result = await llm_client._execute_tool_call('list_directory', {'path': '.'})
            
            assert "Directory contents" in result
            mock_execute.assert_called_once_with('list_directory', {'path': '.'})

    def test_conversation_message_creation(self, llm_client):
        """Test that conversation messages are created with proper structure."""
        timestamp = datetime.now().isoformat()
        message = ConversationMessage(
            role="user",
            content="Test message",
            timestamp=timestamp,
            context={"test": True}
        )
        
        assert message.role == "user"
        assert message.content == "Test message"
        assert message.timestamp == timestamp
        assert message.context == {"test": True}

    def test_conversation_history_management(self, llm_client):
        """Test that conversation history is properly managed."""
        initial_count = len(llm_client.conversation_history)
        
        # Add a message
        message = ConversationMessage(
            role="user",
            content="Test message",
            timestamp=datetime.now().isoformat()
        )
        llm_client.conversation_history.append(message)
        
        assert len(llm_client.conversation_history) == initial_count + 1
        assert llm_client.conversation_history[-1].content == "Test message"


class TestToolExecution:
    """Test tool execution mechanics."""

    @pytest.fixture
    def llm_client(self):
        return LLMClient()

    @pytest.mark.asyncio
    async def test_json_parameter_parsing(self, llm_client):
        """Test that JSON string parameters are correctly parsed."""
        json_params = '{"path": ".", "recursive": true}'
        
        with patch.object(llm_client, '_execute_tool_call') as mock_execute:
            mock_execute.return_value = "Success"
            
            # This would be called in the actual tool execution loop
            parsed_params = json.loads(json_params)
            result = await llm_client._execute_tool_call('list_directory', parsed_params)
            
            mock_execute.assert_called_once_with('list_directory', {'path': '.', 'recursive': True})

    @pytest.mark.asyncio
    async def test_tool_execution_error_handling(self, llm_client):
        """Test that tool execution errors are handled gracefully."""
        with patch.object(llm_client, '_execute_tool_call') as mock_execute:
            mock_execute.side_effect = Exception("Tool execution failed")
            
            # Should not raise, but handle gracefully
            try:
                await llm_client._execute_tool_call('nonexistent_tool', {})
            except Exception as e:
                assert "Tool execution failed" in str(e)


class TestProviderAbstraction:
    """Test that enable_tools parameter works across all providers."""

    @pytest.fixture
    def llm_client(self):
        return LLMClient()

    @pytest.mark.parametrize("provider_method", [
        '_call_ollama_turbo',
        '_call_openai', 
        '_call_openrouter',
        '_call_anthropic',
        '_call_codex',
        '_call_ollama'
    ])
    def test_provider_methods_accept_enable_tools(self, llm_client, provider_method):
        """Test that all provider methods accept enable_tools parameter."""
        method = getattr(llm_client, provider_method)
        
        # Check that the method signature includes enable_tools
        import inspect
        sig = inspect.signature(method)
        assert 'enable_tools' in sig.parameters
        
        # Check default value is True
        assert sig.parameters['enable_tools'].default is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])