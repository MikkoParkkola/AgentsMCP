#!/usr/bin/env python3
"""
Test AgentsMCP with coding tasks using the new command system
"""

import sys
import asyncio
sys.path.insert(0, 'src')

from agentsmcp.config import Config
from agentsmcp.ui.command_interface import CommandInterface

async def test_coding_scenario():
    """Test a realistic coding scenario"""
    print("🧪 Testing AgentsMCP Coding Scenario")
    print("=" * 50)
    
    # Create command interface
    config = Config()
    cmd_interface = CommandInterface(config)
    
    print("1. Testing /help command...")
    try:
        success, result = await cmd_interface._process_command("/help")
        print(f"   Result: {'✅ Success' if success else '❌ Failed'}")
        if result:
            print(f"   Output preview: {str(result)[:100]}...")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n2. Testing /agents command...")
    try:
        success, result = await cmd_interface._process_command("/agents")
        print(f"   Result: {'✅ Success' if success else '❌ Failed'}")
        if result:
            print(f"   Output preview: {str(result)[:100]}...")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n3. Testing /status command...")
    try:
        success, result = await cmd_interface._process_command("/status")
        print(f"   Result: {'✅ Success' if success else '❌ Failed'}")
        if result:
            print(f"   Output preview: {str(result)[:100]}...")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n4. Testing conversation processing...")
    try:
        # Test if conversational input is properly handled
        test_conversation = "Write a simple Python function to calculate factorial"
        is_command = cmd_interface._is_direct_command(test_conversation)
        print(f"   Conversation detection: {'✅ Correct (not a command)' if not is_command else '❌ Wrong (detected as command)'}")
        
        # Process the conversation (this will use the conversation manager)
        success, result = await cmd_interface._process_input(test_conversation)
        print(f"   Conversation processing: {'✅ Success' if success else '❌ Failed'}")
        if result:
            print(f"   Response preview: {str(result)[:150]}...")
            
    except Exception as e:
        print(f"   ❌ Conversation error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Coding scenario test completed!")
    
    return True

if __name__ == "__main__":
    asyncio.run(test_coding_scenario())