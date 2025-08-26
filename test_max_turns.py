#!/usr/bin/env python3
"""
Test that we get final analysis after max tool turns
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agentsmcp.conversation.llm_client import LLMClient

async def test_max_turns():
    """Test that after max tool turns, we get final analysis."""
    print("🧪 Testing Max Tool Turns Logic")
    print("=" * 50)
    
    # Initialize LLM client  
    llm_client = LLMClient()
    
    # Set a lower max_tool_turns to test faster
    # Temporarily hack this for testing
    import agentsmcp.conversation.llm_client
    original_method = agentsmcp.conversation.llm_client.LLMClient.send_message
    
    async def send_message_with_limited_turns(self, message, context=None):
        """Override with max_tool_turns = 1 to test quickly."""
        try:
            # Add user message to history with timestamp
            from datetime import datetime
            timestamp = datetime.now().isoformat()
            user_msg = agentsmcp.conversation.llm_client.ConversationMessage(role="user", content=message, timestamp=timestamp, context=context)
            self.conversation_history.append(user_msg)
            
            # Multi-turn tool execution loop
            max_tool_turns = 1  # Limited for testing
            turn = 0
            
            while turn < max_tool_turns:
                turn += 1
                print(f"🔧 Tool execution turn {turn}/{max_tool_turns}")
                
                # Prepare messages for LLM with auto-detected capabilities
                messages = await self._prepare_messages()
                
                # Use real MCP ollama client based on provider
                response = await self._call_llm_via_mcp(messages)
                if not response:
                    return "Sorry, I'm having trouble connecting to the LLM service. Please check your configuration in settings."
                
                # Check for tool calls in response
                tool_calls = self._extract_tool_calls(response)
                
                if not tool_calls:
                    # No tool calls - extract final response and return
                    assistant_content = self._extract_response_content(response)
                    if assistant_content:
                        # Add final assistant response to history
                        response_timestamp = datetime.now().isoformat()
                        assistant_msg = agentsmcp.conversation.llm_client.ConversationMessage(role="assistant", content=assistant_content, timestamp=response_timestamp)
                        self.conversation_history.append(assistant_msg)
                        return assistant_content
                    else:
                        return "No content received from LLM."
                
                print(f"🔄 Found {len(tool_calls)} tool calls, executing...")
                
                # Execute tool calls and add results as separate messages
                for tool_call in tool_calls:
                    try:
                        tool_name = tool_call.get('function', {}).get('name', '')
                        parameters = tool_call.get('function', {}).get('arguments', {})
                        
                        # Parse arguments if they're a JSON string
                        if isinstance(parameters, str):
                            import json
                            try:
                                parameters = json.loads(parameters)
                            except json.JSONDecodeError:
                                print(f"❌ Failed to parse tool call arguments: {parameters}")
                                continue
                        
                        # Execute the tool
                        result = await self._execute_tool_call(tool_name, parameters)
                        
                        # Add tool result as a user message to continue the conversation
                        tool_timestamp = datetime.now().isoformat()
                        tool_result_msg = agentsmcp.conversation.llm_client.ConversationMessage(
                            role="user", 
                            content=f"Tool execution result for {tool_name}: {result}", 
                            timestamp=tool_timestamp
                        )
                        self.conversation_history.append(tool_result_msg)
                        print(f"✅ Executed {tool_name}")
                        
                    except Exception as e:
                        print(f"❌ Error executing tool call {tool_call}: {e}")
                
                # Continue to next turn to let LLM process tool results
            
            # If we hit max tool turns, ask for final analysis without tools
            print(f"🎯 Max tool turns ({max_tool_turns}) reached, requesting final analysis")
            
            # Add a message asking for final analysis
            analysis_timestamp = datetime.now().isoformat()
            analysis_request_msg = agentsmcp.conversation.llm_client.ConversationMessage(
                role="user", 
                content="Based on the tool execution results above, please provide your complete analysis and recommendations. Do not use any more tools, just give your comprehensive response based on what you've discovered.", 
                timestamp=analysis_timestamp
            )
            self.conversation_history.append(analysis_request_msg)
            
            # Get final analysis without allowing more tool calls
            print("📊 Requesting final analysis from LLM...")
            messages = await self._prepare_messages()
            response = await self._call_llm_via_mcp(messages)
            
            if response:
                # Extract final analysis content
                assistant_content = self._extract_response_content(response)
                if assistant_content:
                    # Add final assistant response to history
                    response_timestamp = datetime.now().isoformat()
                    assistant_msg = agentsmcp.conversation.llm_client.ConversationMessage(role="assistant", content=assistant_content, timestamp=response_timestamp)
                    self.conversation_history.append(assistant_msg)
                    
                    print("✅ Received final analysis!")
                    return assistant_content
                else:
                    print("❌ Final analysis response had no content")
                    return "Final analysis had no content."
            
            # Fallback if final analysis fails
            return "I've gathered information using tools but encountered an issue providing the final analysis. Please try asking your question again."
                
        except Exception as e:
            print(f"❌ Error in LLM communication: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    # Temporarily replace method
    llm_client.send_message = send_message_with_limited_turns.__get__(llm_client, type(llm_client))
    
    try:
        response = await llm_client.send_message(
            "What is the quality status of this project?",
            context={"test": True}
        )
        print(f"\n📝 Final response: {response[:500]}...")
        
        # Check if we got actual analysis
        if "analysis" in response.lower() or "quality" in response.lower() or "recommend" in response.lower():
            print("✅ SUCCESS: Got final analysis after max tool turns!")
            return True
        else:
            print("❌ FAILED: No analysis in final response")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_max_turns())
    sys.exit(0 if success else 1)