#!/usr/bin/env python3
"""
Simple test to check if final analysis works at all
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agentsmcp.conversation.llm_client import LLMClient, ConversationMessage

async def test_final_analysis():
    """Test final analysis directly."""
    print("🧪 Testing Final Analysis Logic Directly")
    print("=" * 50)
    
    # Initialize LLM client  
    llm_client = LLMClient()
    
    # Manually add some conversation history to simulate tool results
    from datetime import datetime
    
    user_msg = ConversationMessage(
        role="user", 
        content="What is the quality status of this project?", 
        timestamp=datetime.now().isoformat()
    )
    llm_client.conversation_history.append(user_msg)
    
    tool_result_msg = ConversationMessage(
        role="user", 
        content="Tool execution result for list_directory: Contents of .: DIR src, DIR tests, FILE README.md, FILE requirements.txt", 
        timestamp=datetime.now().isoformat()
    )
    llm_client.conversation_history.append(tool_result_msg)
    
    # Now add final analysis request
    analysis_request_msg = ConversationMessage(
        role="user", 
        content="Based on the tool execution results above, please provide your complete analysis and recommendations. Do not use any more tools, just give your comprehensive response based on what you've discovered.", 
        timestamp=datetime.now().isoformat()
    )
    llm_client.conversation_history.append(analysis_request_msg)
    
    try:
        print("📊 Requesting final analysis from LLM...")
        
        # Get final analysis
        messages = await llm_client._prepare_messages()
        print(f"📝 Prepared {len(messages)} messages for LLM")
        
        response = await llm_client._call_llm_via_mcp(messages)
        print(f"📨 Got response: {response is not None}")
        
        if response:
            # Extract final analysis content
            assistant_content = llm_client._extract_response_content(response)
            print(f"💬 Extracted content length: {len(assistant_content) if assistant_content else 0}")
            
            if assistant_content:
                print(f"✅ Final analysis: {assistant_content[:300]}...")
                
                # Check if we got actual analysis
                if any(word in assistant_content.lower() for word in ["analysis", "quality", "recommend", "status", "project"]):
                    print("✅ SUCCESS: Got meaningful final analysis!")
                    return True
                else:
                    print("❌ Content exists but doesn't seem to be analysis")
                    return False
            else:
                print("❌ No content in response")
                
                # Check what's actually in the response
                print(f"🔍 Response keys: {list(response.keys()) if response else 'None'}")
                if response and 'message' in response:
                    print(f"🔍 Message keys: {list(response['message'].keys())}")
                    if 'thinking' in response['message']:
                        print(f"🧠 Thinking: {response['message']['thinking'][:200]}...")
                    if 'tool_calls' in response['message']:
                        print(f"🛠️ Tool calls: {len(response['message']['tool_calls'])}")
                
                return False
        else:
            print("❌ No response from LLM")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_final_analysis())
    sys.exit(0 if success else 1)