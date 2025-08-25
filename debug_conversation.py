#!/usr/bin/env python3
"""
Debug conversation manager to understand the issue
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agentsmcp.conversation.conversation import ConversationManager
from agentsmcp.conversation.fake_command_interface import FakeCommandInterface
from agentsmcp.ui.theme_manager import ThemeManager

async def debug_conversation():
    """Debug conversation manager functionality"""
    
    theme_manager = ThemeManager()
    fake_command_interface = FakeCommandInterface()
    conversation_manager = ConversationManager(fake_command_interface, theme_manager)
    
    print("🔍 Debugging Conversation Manager")
    print(f"   Command Interface: {conversation_manager.command_interface}")
    print(f"   Theme Manager: {conversation_manager.theme_manager}")
    print(f"   LLM Client: {conversation_manager.llm_client}")
    print()
    
    # Test various inputs step by step
    test_inputs = [
        "hello there",
        "show me the status"
    ]
    
    for i, user_input in enumerate(test_inputs, 1):
        print(f"=" * 60)
        print(f"TEST {i}: {user_input}")
        print("=" * 60)
        
        try:
            # Step 1: Check command intent extraction
            command_intent = conversation_manager._extract_command_intent(user_input)
            print(f"🎯 Command Intent: {command_intent}")
            if command_intent:
                print(f"   Command: {command_intent.command}")
                print(f"   Confidence: {command_intent.confidence}")
            
            # Step 2: Get LLM response directly
            llm_response = await conversation_manager.llm_client.send_message(user_input, {
                "available_commands": list(conversation_manager.command_patterns.keys()),
                "current_theme": 'auto'
            })
            print(f"💬 LLM Response: {llm_response}")
            
            # Step 3: Check command extraction from LLM response
            llm_command = conversation_manager._extract_command_from_llm_response(llm_response)
            print(f"⚡ LLM Command: {llm_command}")
            if llm_command:
                print(f"   Command: {llm_command.command}")
                print(f"   Parameters: {llm_command.parameters}")
            
            # Step 4: Full process
            full_response = await conversation_manager.process_input(user_input)
            print(f"🎭 Full Response: {full_response}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        print()

if __name__ == "__main__":
    asyncio.run(debug_conversation())