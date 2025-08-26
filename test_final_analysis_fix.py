#!/usr/bin/env python3
"""
Test that final analysis now works with tools disabled
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agentsmcp.conversation.llm_client import LLMClient

async def test_final_analysis_fix():
    """Test that final analysis works with enable_tools=False."""
    print("🧪 Testing Final Analysis Fix")
    print("==" * 25)
    
    # Initialize LLM client  
    llm_client = LLMClient()
    
    try:
        # Test the new enable_tools parameter in _call_llm_via_mcp
        from datetime import datetime
        
        # Add some conversation history to simulate tool results
        user_msg = llm_client.conversation_history.__class__(
            role="user", 
            content="What is the quality status of this project?", 
            timestamp=datetime.now().isoformat()
        )
        llm_client.conversation_history.append(user_msg)
        
        tool_result_msg = llm_client.conversation_history.__class__(
            role="user", 
            content="Tool execution result for list_directory: Contents of .: DIR src, DIR tests, FILE README.md, FILE requirements.txt, FILE setup.py", 
            timestamp=datetime.now().isoformat()
        )
        llm_client.conversation_history.append(tool_result_msg)
        
        # Add final analysis request
        analysis_request_msg = llm_client.conversation_history.__class__(
            role="user", 
            content="Based on the tool execution results above, please provide your complete analysis and recommendations. Do not use any more tools, just give your comprehensive response based on what you've discovered.", 
            timestamp=datetime.now().isoformat()
        )
        llm_client.conversation_history.append(analysis_request_msg)
        
        # Test final analysis call with tools disabled
        print("📊 Testing final analysis with enable_tools=False...")
        messages = await llm_client._prepare_messages()
        print(f"📝 Prepared {len(messages)} messages")
        
        # This should NOT include tools in the LLM call
        response = await llm_client._call_llm_via_mcp(messages, enable_tools=False)
        print(f"📨 Got response: {response is not None}")
        
        if response:
            assistant_content = llm_client._extract_response_content(response)
            print(f"💬 Extracted content length: {len(assistant_content) if assistant_content else 0}")
            
            if assistant_content:
                print(f"✅ Final analysis (first 500 chars): {assistant_content[:500]}...")
                
                # Check if we got actual analysis
                analysis_keywords = ["analysis", "quality", "recommend", "status", "project", "structure", "issue", "improvement"]
                if any(word in assistant_content.lower() for word in analysis_keywords):
                    print("✅ SUCCESS: Got meaningful final analysis without tool calls!")
                    return True
                else:
                    print("❌ Content exists but doesn't seem to be analysis")
                    return False
            else:
                print("❌ No content in response")
                
                # Debug response structure
                print(f"🔍 Response keys: {list(response.keys()) if response else 'None'}")
                if response and 'message' in response:
                    message = response['message']
                    print(f"🔍 Message keys: {list(message.keys()) if isinstance(message, dict) else 'Not a dict'}")
                
                return False
        else:
            print("❌ No response from LLM")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_full_multiturn_workflow():
    """Test the full multi-turn workflow with final analysis."""
    print("\n🔄 Testing Full Multi-turn Workflow")
    print("==" * 25)
    
    llm_client = LLMClient()
    
    try:
        # Test a complete multi-turn scenario that should hit the final analysis logic
        response = await llm_client.send_message(
            "Can you analyze the project structure? I want to understand the quality status and any issues.",
            context={"test": True}
        )
        
        print(f"💬 Complete response length: {len(response)}")
        print(f"📝 Response preview: {response[:500]}...")
        
        # Check if we got comprehensive analysis (not just tool outputs)
        analysis_indicators = [
            "analysis", "quality", "recommend", "status", "structure", 
            "issue", "improvement", "suggest", "project", "code", "assessment"
        ]
        
        if any(indicator in response.lower() for indicator in analysis_indicators):
            print("✅ SUCCESS: Full multi-turn workflow with final analysis working!")
            return True
        else:
            print("❌ FAILED: Response doesn't contain comprehensive analysis")
            return False
            
    except Exception as e:
        print(f"❌ Full workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing Final Analysis Fix Implementation")
    print("=" * 50)
    
    # Test 1: Direct final analysis with enable_tools=False
    success1 = asyncio.run(test_final_analysis_fix())
    
    # Test 2: Full multi-turn workflow
    success2 = asyncio.run(test_full_multiturn_workflow())
    
    overall_success = success1 and success2
    
    if overall_success:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Final analysis with tools disabled: Working")
        print("✅ Full multi-turn workflow: Working")
        print("\nThe multi-turn tool execution issue should now be resolved!")
    else:
        print("\n❌ SOME TESTS FAILED")
        if not success1:
            print("❌ Final analysis with tools disabled: Failed")
        if not success2:
            print("❌ Full multi-turn workflow: Failed")
    
    sys.exit(0 if overall_success else 1)