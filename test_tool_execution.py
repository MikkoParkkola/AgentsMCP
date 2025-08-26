#!/usr/bin/env python3
"""
Quick test to verify tool execution is working
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agentsmcp.conversation.llm_client import LLMClient
from agentsmcp.tools.base_tools import tool_registry

async def test_tool_execution():
    """Test that tool execution is working"""
    print("🧪 Testing Tool Execution in LLM Client")
    print("=" * 50)
    
    # Initialize LLM client
    llm_client = LLMClient()
    
    # Check that tools are registered
    all_tools = tool_registry.get_all_tools()
    print(f"✅ Tools registered: {[tool.name for tool in all_tools]}")
    
    # Test tool execution directly
    try:
        print("\n🔧 Testing direct tool execution:")
        result = await llm_client._execute_tool("list_directory", {"path": "."})
        print(f"✅ list_directory: {result[:100]}..." if len(result) > 100 else result)
    except Exception as e:
        print(f"❌ Direct tool execution failed: {e}")
        return False
    
    # Test a simple query that should trigger tool usage
    print("\n🤖 Testing LLM with tool integration:")
    try:
        response = await llm_client.send_message(
            "List the files in the current directory",
            context={"test": True}
        )
        print(f"✅ LLM response with tool execution: {response[:200]}..." if len(response) > 200 else response)
        
        # Check if response contains tool results
        if "✅" in response and ("list_directory" in response or "Contents of" in response):
            print("✅ Tool execution successful - response contains tool results!")
            return True
        else:
            print("❌ Tool execution may have failed - no tool results in response")
            print(f"Full response: {response}")
            return False
            
    except Exception as e:
        print(f"❌ LLM tool integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_tool_execution())
    if success:
        print("\n🎉 Tool execution test passed!")
        print("The fix is working - tools are now being executed instead of returning placeholder messages.")
    else:
        print("\n❌ Tool execution test failed!")
        print("The fix may need additional work.")
    sys.exit(0 if success else 1)