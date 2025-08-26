"""Test multi-turn tool execution in LLM client.

This test ensures that when the LLM returns tool calls, the system:
1. Executes the tools
2. Sends results back to the LLM
3. Gets a final analysis from the LLM
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

# Import the LLM client
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from agentsmcp.conversation.llm_client import LLMClient


@pytest.mark.asyncio
async def test_multi_turn_tool_execution():
    """Test that multi-turn tool execution works correctly."""
    
    # Create LLM client
    client = LLMClient()
    
    # Mock the LLM API calls  
    with patch.object(client, '_call_llm_via_mcp') as mock_call:
        # First call: LLM returns tool calls without content
        first_response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",  # No content, just tool calls
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "list_directory",
                            "arguments": {"path": "."}
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }
        
        # Second call: LLM provides analysis of tool results
        second_response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Based on the directory listing, I can see the following files and folders in the current directory: [analysis of results]",
                    "tool_calls": []  # No more tool calls
                },
                "finish_reason": "stop"
            }]
        }
        
        # Configure mock to return different responses
        mock_call.side_effect = [first_response, second_response]
        
        # Mock tool execution
        with patch.object(client, '_execute_tool_call') as mock_execute:
            mock_execute.return_value = "Contents of .:\nfile1.txt (100 bytes)\nfile2.py (200 bytes)\nfolder1/"
            
            # Send a message that should trigger tool use
            response = await client.send_message("Please list the files in the current directory")
            
            # Verify the response contains analysis, not just tool results
            assert response is not None
            assert "Based on the directory listing" in response
            assert "analysis of results" in response
            
            # Verify tool was executed
            mock_execute.assert_called_once_with("list_directory", {"path": "."})
            
            # Verify LLM was called twice (multi-turn)
            assert mock_call.call_count == 2


@pytest.mark.asyncio
async def test_tool_results_not_returned_raw():
    """Test that raw tool results are not returned without LLM analysis."""
    
    client = LLMClient()
    
    with patch.object(client, '_call_llm_via_mcp') as mock_call:
        # LLM returns only tool calls, no content
        response_with_tools = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",  # Empty content
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": {"file_path": "test.txt"}
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }
        
        # After tool execution, LLM provides analysis
        final_response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "The file contains important configuration settings for the application.",
                    "tool_calls": []
                },
                "finish_reason": "stop"
            }]
        }
        
        mock_call.side_effect = [response_with_tools, final_response]
        
        with patch.object(client, '_execute_tool_call') as mock_execute:
            mock_execute.return_value = "Contents of test.txt:\nconfig_key=value\nother_setting=123"
            
            response = await client.send_message("What's in test.txt?")
            
            # Response should be the LLM's analysis, not raw tool output
            assert "Tool execution result for" not in response
            assert "Contents of test.txt:" not in response
            assert "important configuration settings" in response


@pytest.mark.asyncio
async def test_multiple_tool_calls_in_single_turn():
    """Test handling multiple tool calls in a single LLM response."""
    
    client = LLMClient()
    
    with patch.object(client, '_call_llm_via_mcp') as mock_call:
        # LLM returns multiple tool calls
        response_with_tools = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "list_directory",
                                "arguments": {"path": "."}
                            }
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": {"file_path": "README.md"}
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }]
        }
        
        final_response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "I found the README.md file in the directory and it contains project documentation.",
                    "tool_calls": []
                },
                "finish_reason": "stop"
            }]
        }
        
        mock_call.side_effect = [response_with_tools, final_response]
        
        with patch.object(client, '_execute_tool_call') as mock_execute:
            mock_execute.side_effect = [
                "Contents of .:\nREADME.md (500 bytes)\nsrc/",
                "Contents of README.md:\n# Project Title\nThis is a test project."
            ]
            
            response = await client.send_message("List files and read the README")
            
            # Verify both tools were executed
            assert mock_execute.call_count == 2
            
            # Verify final response is analysis, not raw results
            assert "found the README.md" in response
            assert "project documentation" in response


@pytest.mark.asyncio 
async def test_max_tool_turns_limit():
    """Test that tool execution stops after max turns to prevent infinite loops."""
    
    client = LLMClient()
    
    with patch.object(client, '_call_llm_via_mcp') as mock_call:
        # Always return tool calls (simulate potential infinite loop)
        always_tools_response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_x",
                        "type": "function",
                        "function": {
                            "name": "list_directory",
                            "arguments": {"path": "."}
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }
        
        # Final response when forced to not use tools
        final_response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Based on the tool results, here is my analysis...",
                    "tool_calls": []
                },
                "finish_reason": "stop"
            }]
        }
        
        # Return tool calls 3 times, then final response
        mock_call.side_effect = [
            always_tools_response,  # Turn 1
            always_tools_response,  # Turn 2
            always_tools_response,  # Turn 3
            final_response         # Forced final analysis
        ]
        
        with patch.object(client, '_execute_tool_call') as mock_execute:
            mock_execute.return_value = "Tool result"
            
            response = await client.send_message("Do something requiring tools")
            
            # Should execute tools 3 times (max_tool_turns=3)
            assert mock_execute.call_count == 3
            
            # Should call LLM 4 times (3 tool turns + 1 final)
            assert mock_call.call_count == 4
            
            # Should get final analysis
            assert "Based on the tool results" in response


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_multi_turn_tool_execution())
    asyncio.run(test_tool_results_not_returned_raw())
    asyncio.run(test_multiple_tool_calls_in_single_turn())
    asyncio.run(test_max_tool_turns_limit())
    print("All tests passed!")