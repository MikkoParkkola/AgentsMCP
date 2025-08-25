#!/usr/bin/env python3
"""
Test the chat functionality to see if it's working properly
"""

import asyncio
import json
from pathlib import Path

# Add the src directory to Python path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agentsmcp.conversation.conversation import ConversationManager
from agentsmcp.conversation.fake_command_interface import FakeCommandInterface
from agentsmcp.ui.theme_manager import ThemeManager

async def test_chat():
    """Test chat functionality with configured model"""
    
    # Setup with fake command interface for testing
    theme_manager = ThemeManager()
    fake_command_interface = FakeCommandInterface()
    conversation_manager = ConversationManager(fake_command_interface, theme_manager)
    
    print("🤖 Testing Chat with Configured Model")
    print(f"   Provider: {conversation_manager.llm_client.provider}")
    print(f"   Model: {conversation_manager.llm_client.model}")
    print(f"   Config: {conversation_manager.llm_client.config}")
    print()
    
    # Test basic chat
    test_inputs = [
        "hello there",
        "show me the status", 
        "I need help with settings",
        "change theme to dark"
    ]
    
    print("=" * 60)
    print("TESTING CHAT RESPONSES")
    print("=" * 60)
    
    for i, user_input in enumerate(test_inputs, 1):
        print(f"\n[Test {i}/4] User: {user_input}")
        try:
            response = await conversation_manager.process_input(user_input)
            print(f"🤖 Assistant: {response}")
            
            # Check if command was extracted
            if hasattr(conversation_manager, '_last_command_intent'):
                intent = conversation_manager._last_command_intent
                if intent:
                    print(f"   → Extracted Command: {intent.command} {intent.params or ''}")
                else:
                    print("   → No command extracted")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        print("-" * 40)
    
    print("\n✅ Chat functionality test completed")

if __name__ == "__main__":
    asyncio.run(test_chat())