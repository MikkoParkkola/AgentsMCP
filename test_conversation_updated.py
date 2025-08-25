#!/usr/bin/env python3
"""
Test the updated conversational interface with real configured model
"""

import asyncio
from src.agentsmcp.conversation.conversation import ConversationManager
from src.agentsmcp.ui.theme_manager import ThemeManager

async def test_conversation():
    """Test conversational interface with real configured model"""
    theme_manager = ThemeManager()
    
    # Create conversation manager (without command interface for this test)
    conversation_manager = ConversationManager(None, theme_manager)
    
    print("🤖 Testing conversational interface with configured model...")
    print(f"   Provider: {conversation_manager.llm_client.provider}")
    print(f"   Model: {conversation_manager.llm_client.model}")
    print(f"   Config: {conversation_manager.llm_client.config}")
    print()
    
    # Test some conversational inputs
    test_inputs = [
        "hello",
        "show me the system status",
        "I want to check what agents are running",
        "change theme to dark",
        "help me with something"
    ]
    
    for user_input in test_inputs:
        print(f"👤 User: {user_input}")
        response = await conversation_manager.process_input(user_input)
        print(f"🤖 Assistant: {response}")
        print("-" * 80)

if __name__ == "__main__":
    asyncio.run(test_conversation())