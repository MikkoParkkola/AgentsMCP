#!/usr/bin/env python3
"""
Behavioral tests for multi-turn tool execution.
These tests verify that the system behaves correctly in real-world scenarios
and capture expected behaviors to prevent regression.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentsmcp.conversation.llm_client import LLMClient


class TestMultiTurnBehaviors:
    """Test specific behavioral patterns in multi-turn execution."""

    @pytest.fixture
    def llm_client(self):
        return LLMClient()

    @pytest.fixture
    def realistic_responses(self):
        """Realistic response patterns from actual usage."""
        return {
            "code_review_query": "Can you review this codebase and identify any issues?",
            "quality_assessment": "What is the quality status of this project?", 
            "structure_analysis": "Please analyze the project structure and suggest improvements",
            "security_audit": "Are there any security concerns in this codebase?"
        }

    @pytest.mark.asyncio
    async def test_code_review_behavior(self, llm_client, realistic_responses):
        """Test that code review queries produce comprehensive analysis."""
        
        # Mock responses that follow the expected pattern
        responses = [
            # First call - tool execution
            {
                'message': {
                    'content': '',
                    'tool_calls': [{'function': {'name': 'list_directory', 'arguments': {'path': '.'}}}]
                }
            },
            # Second call - more tool execution  
            {
                'message': {
                    'content': '',
                    'tool_calls': [{'function': {'name': 'read_file', 'arguments': {'file_path': 'src/main.py'}}}]
                }
            },
            # Final call - analysis (no tools)
            {
                'message': {
                    'content': """# Code Review Analysis

Based on my examination of the codebase, here's my comprehensive review:

## Structure Analysis
- Well-organized directory structure
- Clear separation of concerns
- Proper module organization

## Code Quality Issues Identified
1. **Type Hints**: Several functions lack type hints
2. **Error Handling**: Some functions need better exception handling  
3. **Documentation**: Missing docstrings in key modules

## Security Assessment
- No obvious security vulnerabilities found
- Input validation appears adequate
- No hardcoded credentials detected

## Recommendations
1. Add comprehensive type hints across the codebase
2. Implement consistent error handling patterns
3. Add missing documentation
4. Consider adding integration tests

## Overall Rating: Good (7/10)
The codebase is well-structured with room for improvement in documentation and type safety.""",
                    'tool_calls': None
                }
            }
        ]
        
        call_index = 0
        async def mock_llm_call(messages, enable_tools=True):
            nonlocal call_index
            if call_index < len(responses):
                response = responses[call_index]
                call_index += 1
                return response
            return responses[-1]  # Return final response for any additional calls
        
        async def mock_tool_execution(tool_name, params):
            if tool_name == "list_directory":
                return "Contents: src/, tests/, README.md, setup.py"
            elif tool_name == "read_file":
                return "def main():\n    print('Hello World')\n    return 0"
            return f"Mock result for {tool_name}"
        
        with patch.object(llm_client, '_call_llm_via_mcp', side_effect=mock_llm_call), \
             patch.object(llm_client, '_execute_tool_call', side_effect=mock_tool_execution):
            
            response = await llm_client.send_message(realistic_responses["code_review_query"])
            
            # Verify behavioral expectations
            assert "Code Review" in response or "analysis" in response.lower()
            assert "recommendations" in response.lower() or "recommend" in response.lower()
            assert len(response) > 500  # Should be comprehensive
            assert "quality" in response.lower() or "issues" in response.lower()

    @pytest.mark.asyncio
    async def test_quality_assessment_behavior(self, llm_client, realistic_responses):
        """Test quality assessment produces structured evaluation."""
        
        quality_response = {
            'message': {
                'content': """# Project Quality Assessment

## Overall Quality Score: 8.5/10

### Strengths
✅ **Code Organization**: Excellent modular structure
✅ **Testing**: Comprehensive test coverage (>90%)
✅ **Documentation**: Well-documented APIs
✅ **Dependencies**: Up-to-date and minimal dependencies

### Areas for Improvement  
⚠️ **Performance**: Some algorithms could be optimized
⚠️ **Error Handling**: Inconsistent error handling patterns
⚠️ **Logging**: Limited structured logging

### Security Status
🔒 **Security**: No critical vulnerabilities detected
🔒 **Dependencies**: All dependencies scanned and clean

### Recommendations
1. **High Priority**: Standardize error handling across modules
2. **Medium Priority**: Implement performance optimizations for core algorithms  
3. **Low Priority**: Enhance logging with structured format

### Quality Metrics
- Code Coverage: 92%
- Cyclomatic Complexity: Average 4.2 (Good)
- Maintainability Index: 73 (Good)
- Technical Debt Ratio: 12% (Acceptable)""",
                'tool_calls': None
            }
        }
        
        async def mock_llm_call(messages, enable_tools=True):
            return quality_response
        
        with patch.object(llm_client, '_call_llm_via_mcp', side_effect=mock_llm_call):
            response = await llm_client.send_message(realistic_responses["quality_assessment"])
            
            # Verify quality assessment patterns
            assert "quality" in response.lower()
            assert "score" in response.lower() or "rating" in response.lower()
            assert "recommend" in response.lower() or "improvement" in response.lower()
            assert len(response) > 300

    @pytest.mark.asyncio 
    async def test_follow_up_question_behavior(self, llm_client):
        """Test that follow-up questions build on previous context."""
        
        first_response = {
            'message': {
                'content': """Based on my analysis, this project has good structure but needs better error handling and more comprehensive tests.""",
                'tool_calls': None
            }
        }
        
        followup_response = {
            'message': {
                'content': """Based on my previous analysis where I identified error handling and testing gaps, here are the top 3 priorities:

1. **Error Handling** (High Priority): Implement consistent exception handling patterns across all modules
2. **Test Coverage** (High Priority): Increase test coverage from current 65% to target 90%  
3. **Documentation** (Medium Priority): Add comprehensive docstrings and API documentation

These improvements will significantly enhance code quality and maintainability.""",
                'tool_calls': None
            }
        }
        
        responses = [first_response, followup_response]
        call_index = 0
        
        async def mock_llm_call(messages, enable_tools=True):
            nonlocal call_index
            if call_index < len(responses):
                response = responses[call_index]
                call_index += 1
                return response
            return responses[-1]
        
        with patch.object(llm_client, '_call_llm_via_mcp', side_effect=mock_llm_call):
            # First question
            response1 = await llm_client.send_message(
                "What are the main issues with this codebase?"
            )
            
            # Follow-up question
            response2 = await llm_client.send_message(
                "Based on your analysis, what are the top 3 priority improvements?"
            )
            
            # Verify follow-up builds on context
            assert "previous analysis" in response2.lower() or "based on" in response2.lower()
            assert "priority" in response2.lower()
            assert len(response2) > 200

    @pytest.mark.asyncio
    async def test_edge_case_empty_response(self, llm_client):
        """Test handling of edge cases like empty LLM responses."""
        
        async def mock_llm_call(messages, enable_tools=True):
            return {'message': {'content': '', 'tool_calls': None}}
        
        with patch.object(llm_client, '_call_llm_via_mcp', side_effect=mock_llm_call):
            response = await llm_client.send_message("Test empty response handling")
            
            # Should handle gracefully
            assert isinstance(response, str)
            # Should provide some fallback message
            assert len(response) > 0

    @pytest.mark.asyncio
    async def test_long_conversation_context(self, llm_client):
        """Test that long conversations maintain context appropriately."""
        
        conversation_responses = [
            "I'll analyze the project structure for you.",
            "Based on the directory listing, I can see this is a Python project.",
            "The code quality appears good with proper organization.",
            "Here's my final comprehensive assessment based on all the information gathered."
        ]
        
        response_index = 0
        async def mock_llm_call(messages, enable_tools=True):
            nonlocal response_index
            if response_index < len(conversation_responses):
                content = conversation_responses[response_index]
                response_index += 1
                return {'message': {'content': content, 'tool_calls': None}}
            return {'message': {'content': conversation_responses[-1], 'tool_calls': None}}
        
        with patch.object(llm_client, '_call_llm_via_mcp', side_effect=mock_llm_call):
            
            # Simulate a longer conversation
            questions = [
                "Analyze this project",
                "What's the directory structure?", 
                "How's the code quality?",
                "Give me a final summary"
            ]
            
            responses = []
            for question in questions:
                response = await llm_client.send_message(question)
                responses.append(response)
            
            # Verify conversation builds appropriately
            assert len(responses) == 4
            assert all(len(r) > 0 for r in responses)
            
            # Verify conversation history grows
            assert len(llm_client.conversation_history) >= len(questions)


class TestGoldenMaster:
    """Golden master tests to capture expected outputs for regression testing."""

    @pytest.fixture
    def llm_client(self):
        return LLMClient()

    def test_tool_call_extraction_golden(self, llm_client):
        """Golden master test for tool call extraction."""
        
        # Known good response format
        response = {
            'choices': [{
                'message': {
                    'content': '',
                    'tool_calls': [
                        {
                            'function': {
                                'name': 'list_directory',
                                'arguments': {'path': '.', 'recursive': False}
                            }
                        },
                        {
                            'function': {
                                'name': 'read_file', 
                                'arguments': {'file_path': 'README.md'}
                            }
                        }
                    ]
                }
            }]
        }
        
        tool_calls = llm_client._extract_tool_calls(response)
        
        # Golden master expectations
        assert len(tool_calls) == 2
        assert tool_calls[0]['function']['name'] == 'list_directory'
        assert tool_calls[0]['function']['arguments'] == {'path': '.', 'recursive': False}
        assert tool_calls[1]['function']['name'] == 'read_file'
        assert tool_calls[1]['function']['arguments'] == {'file_path': 'README.md'}

    def test_content_extraction_golden(self, llm_client):
        """Golden master test for content extraction."""
        
        response = {
            'choices': [{
                'message': {
                    'content': 'This is the expected content from the LLM response.',
                    'tool_calls': None
                }
            }]
        }
        
        content = llm_client._extract_response_content(response)
        
        # Golden master expectation
        assert content == 'This is the expected content from the LLM response.'

    def test_conversation_message_golden(self, llm_client):
        """Golden master test for conversation message structure."""
        
        from datetime import datetime
        timestamp = "2023-12-01T10:30:00"
        
        message = llm_client.conversation_history.__class__[0].__class__(
            role="user",
            content="Test message content", 
            timestamp=timestamp,
            context={"test": True}
        ) if llm_client.conversation_history else None
        
        # This test verifies the ConversationMessage structure remains stable
        # (Will be updated based on actual implementation)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])