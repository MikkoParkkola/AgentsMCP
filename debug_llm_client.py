#!/usr/bin/env python3
"""
Debug LLM client to understand what's happening
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agentsmcp.conversation.llm_client import LLMClient

async def debug_llm():
    """Debug LLM client functionality"""
    
    llm_client = LLMClient()
    
    print("🔍 Debugging LLM Client")
    print(f"   Provider: {llm_client.provider}")
    print(f"   Model: {llm_client.model}")
    print(f"   Config: {llm_client.config}")
    print()
    
    # Test the _generate_intelligent_response method directly
    test_inputs = [
        "hello there",
        "show me the status", 
        "I need help with settings"
    ]
    
    print("=" * 60)
    print("TESTING INTELLIGENT RESPONSE GENERATION")
    print("=" * 60)
    
    for user_input in test_inputs:
        print(f"\n👤 User: {user_input}")
        
        # Test intelligent response
        messages = [{"role": "user", "content": user_input}]
        response = llm_client._generate_intelligent_response(user_input, messages)
        print(f"🤖 Intelligent Response: {response}")
        
        # Test actual send_message
        try:
            full_response = await llm_client.send_message(user_input)
            print(f"📤 Full Response: {full_response}")
        except Exception as e:
            print(f"❌ Send message error: {e}")
        
        print("-" * 40)

if __name__ == "__main__":
    asyncio.run(debug_llm())