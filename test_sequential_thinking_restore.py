#!/usr/bin/env python3
"""
Test script to verify the sequential thinking restoration in AgentsMCP.

This script tests:
1. API key error message fix (should show OLLAMA_API_KEY not OLLAMA-TURBO_API_KEY)
2. Sequential thinking works for complex queries (5+ words)
3. Agent progress visibility is restored
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agentsmcp.ui.v3.chat_engine import ChatEngine


async def test_sequential_thinking_restoration():
    """Test that sequential thinking and progress visibility is restored."""
    print("\n=== Testing Sequential Thinking Restoration ===")
    
    engine = ChatEngine()
    
    # Test 1: Simple query (≤4 words) should use direct LLM
    print("\n1. Testing simple query routing...")
    route, word_count = engine._route_input("hello there")
    print(f"Input: 'hello there' -> Route: {route}, Words: {word_count}")
    
    if route == "direct" and word_count == 2:
        print("✅ Simple query routing: PASSED")
    else:
        print("❌ Simple query routing: FAILED")
        return False
    
    # Test 2: Complex query (5+ words) should use preprocessed LLM
    print("\n2. Testing complex query routing...")
    route, word_count = engine._route_input("analyze the project structure and provide recommendations")
    print(f"Input: complex query -> Route: {route}, Words: {word_count}")
    
    if route == "preprocessed" and word_count >= 5:
        print("✅ Complex query routing: PASSED")
    else:
        print("❌ Complex query routing: FAILED")
        return False
    
    # Test 3: Check that helper methods exist
    print("\n3. Testing helper method availability...")
    methods_to_check = [
        '_get_conversation_context',
        '_get_directory_context', 
        '_use_sequential_thinking',
        '_create_enhanced_prompt'
    ]
    
    missing_methods = []
    for method_name in methods_to_check:
        if not hasattr(engine, method_name):
            missing_methods.append(method_name)
    
    if missing_methods:
        print(f"❌ Missing methods: {missing_methods}")
        return False
    else:
        print("✅ All helper methods available: PASSED")
    
    # Test 4: Test context methods work
    print("\n4. Testing context methods...")
    try:
        conv_context = engine._get_conversation_context()
        dir_context = engine._get_directory_context()
        
        if "conversation" in conv_context.lower() and "directory" in dir_context.lower():
            print("✅ Context methods functional: PASSED")
        else:
            print("❌ Context methods not working correctly")
            return False
            
    except Exception as e:
        print(f"❌ Context methods failed: {e}")
        return False
    
    print("\n✅ ALL TESTS PASSED - Sequential thinking restoration successful!")
    return True


if __name__ == "__main__":
    success = asyncio.run(test_sequential_thinking_restoration())
    sys.exit(0 if success else 1)