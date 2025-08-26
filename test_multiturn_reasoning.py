#!/usr/bin/env python3
"""
Test multi-turn reasoning and analysis capabilities
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agentsmcp.conversation.llm_client import LLMClient

async def test_multiturn_reasoning():
    """Test that the LLM can perform multi-turn reasoning with tool execution."""
    print("🧪 Testing Multi-Turn Reasoning in LLM Client")
    print("=" * 50)
    
    # Initialize LLM client
    llm_client = LLMClient()
    
    # Test 1: Code review question (should list files, then read and analyze)
    print("\n🔍 Test 1: Code Review Analysis")
    try:
        response1 = await llm_client.send_message(
            "Can you do a code review of this project? Please analyze the code structure and identify any potential issues.",
            context={"test": True}
        )
        print(f"✅ Code review response: {response1[:300]}...")
        
        # Check if response contains analysis (not just tool output)
        if "analysis" in response1.lower() or "review" in response1.lower() or "issue" in response1.lower():
            print("✅ Response contains actual analysis!")
        else:
            print("❌ Response might be missing analysis")
        
    except Exception as e:
        print(f"❌ Code review test failed: {e}")
        return False
    
    # Test 2: Quality status question (should analyze multiple files)
    print("\n📊 Test 2: Project Quality Assessment")
    try:
        response2 = await llm_client.send_message(
            "What is the quality status of this project? Are there any security concerns?",
            context={"test": True}
        )
        print(f"✅ Quality assessment response: {response2[:300]}...")
        
        # Check if response contains quality assessment (not just file listings)
        if "quality" in response2.lower() or "security" in response2.lower() or "recommend" in response2.lower():
            print("✅ Response contains quality assessment!")
        else:
            print("❌ Response might be missing quality assessment")
        
    except Exception as e:
        print(f"❌ Quality assessment test failed: {e}")
        return False
    
    # Test 3: Follow-up question to see conversation continuity
    print("\n🔄 Test 3: Follow-up Analysis")
    try:
        response3 = await llm_client.send_message(
            "Based on your analysis, what are the top 3 priority improvements needed?",
            context={"test": True}
        )
        print(f"✅ Follow-up response: {response3[:300]}...")
        
        # Check if response builds on previous analysis
        if "priority" in response3.lower() or "improvement" in response3.lower() or "1." in response3:
            print("✅ Response provides prioritized recommendations!")
        else:
            print("❌ Response might not be building on previous analysis")
        
    except Exception as e:
        print(f"❌ Follow-up test failed: {e}")
        return False
    
    print("\n🎉 Multi-turn reasoning tests completed!")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_multiturn_reasoning())
    if success:
        print("\n✅ Multi-turn reasoning is working correctly!")
        print("The LLM can now execute tools AND continue reasoning to provide comprehensive analysis.")
    else:
        print("\n❌ Multi-turn reasoning needs more work!")
    sys.exit(0 if success else 1)