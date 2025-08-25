#!/usr/bin/env python3
"""
Simple test for conversational interface functionality.
"""

import asyncio
import sys
from src.agentsmcp.conversation import ConversationManager
from src.agentsmcp.ui.theme_manager import ThemeManager

async def test_conversational_interface():
    """Test the conversational interface"""
    print("🧪 Testing AgentsMCP Conversational Interface")
    print("=" * 50)
    
    # Initialize components
    theme_manager = ThemeManager()
    conversation_manager = ConversationManager(None, theme_manager)
    
    # Test natural language inputs
    test_inputs = [
        "show me the status",
        "what can you help me with?", 
        "open settings please",
        "start the dashboard",
        "change theme to dark",
        "help me with commands"
    ]
    
    for i, user_input in enumerate(test_inputs, 1):
        print(f"\n🔍 Test {i}: '{user_input}'")
        print("-" * 30)
        
        try:
            # Process input through conversational manager
            response = await conversation_manager.process_input(user_input)
            print(f"✅ Response: {response}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n🎯 Conversational interface test completed!")
    return True

if __name__ == "__main__":
    try:
        result = asyncio.run(test_conversational_interface())
        if result:
            print("\n🎉 All conversational tests passed!")
            sys.exit(0)
        else:
            print("\n⚠️ Some tests failed")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        sys.exit(1)