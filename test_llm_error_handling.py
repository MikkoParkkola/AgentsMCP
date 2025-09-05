#!/usr/bin/env python3
"""
Test script to verify LLM error handling improvements.
Tests that the system shows proper error messages instead of generic responses.
"""

import asyncio
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agentsmcp.conversation.llm_client import LLMClient

async def test_llm_error_handling():
    """Test that LLM failures result in error messages, not generic responses."""
    print("🧪 Testing LLM Error Handling Fix")
    print("=" * 50)
    
    # Create LLM client (will likely fail to connect properly)
    try:
        client = LLMClient()
        print(f"✓ LLM client created (provider: {client.provider}, model: {client.model})")
    except Exception as e:
        print(f"❌ Failed to create LLM client: {e}")
        return
    
    # Test 1: Simple message that would trigger generic response before fix
    test_cases = [
        "analyze the project in current directory",
        "what kind of issues do you see here?",
        "help me understand this codebase",
        "Hello, how can you help me?",
    ]
    
    for i, test_message in enumerate(test_cases, 1):
        print(f"\n🔍 Test {i}: '{test_message[:50]}{'...' if len(test_message) > 50 else ''}'")
        print("-" * 40)
        
        try:
            response = await client.send_message(test_message)
            
            # Check if response contains error message markers
            if any(marker in response for marker in ["❌", "LLM Connection Error", "Failed to connect", "not running", "not configured"]):
                print("✅ PASS: Got proper error message instead of generic response")
                print(f"   Error message: {response[:100]}{'...' if len(response) > 100 else ''}")
            elif any(generic in response.lower() for generic in [
                "hello! how can i assist you today",
                "i understand you're asking", 
                "i can help with basic commands",
                "i'll analyze the project structure",
                "what would you like me to do"
            ]):
                print("❌ FAIL: Still getting generic response")
                print(f"   Generic response: {response[:100]}{'...' if len(response) > 100 else ''}")
            else:
                print("📝 INFO: Got unexpected response (may be valid LLM response)")
                print(f"   Response: {response[:100]}{'...' if len(response) > 100 else ''}")
                
        except Exception as e:
            print(f"⚠️  Exception during test: {e}")
    
    print(f"\n📊 Summary:")
    print("The fix replaces generic 'intelligent' responses with clear error messages")
    print("when LLM connection fails. Users should now see:")
    print("• Specific error messages about missing API keys") 
    print("• Instructions on how to fix connection issues")
    print("• Clear indication that the LLM is not reachable")
    print("• NO generic responses like 'Hello! How can I assist you today?'")

if __name__ == "__main__":
    asyncio.run(test_llm_error_handling())