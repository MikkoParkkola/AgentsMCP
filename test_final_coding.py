#!/usr/bin/env python3
"""
Final test of coding task with the new command system
"""

import sys
import asyncio
sys.path.insert(0, 'src')

from agentsmcp.config import Config
from agentsmcp.ui.command_interface import CommandInterface

async def test_final_coding_task():
    """Test a comprehensive coding task"""
    print("🎯 Final Coding Task Test")
    print("=" * 40)
    
    # Create command interface
    config = Config()
    cmd_interface = CommandInterface(config)
    
    coding_tasks = [
        "Create a Python class for a simple calculator with add, subtract, multiply, and divide methods",
        "Add input validation and error handling to the calculator", 
        "Write unit tests for the calculator class",
    ]
    
    print("Testing conversational coding tasks:")
    print()
    
    for i, task in enumerate(coding_tasks, 1):
        print(f"Task {i}: {task}")
        print("-" * 40)
        
        try:
            # Verify it's treated as conversation
            is_command = cmd_interface._is_direct_command(task)
            print(f"Command detection: {'❌ Wrong' if is_command else '✅ Correct (conversation)'}")
            
            # Process the task
            success, result = await cmd_interface._process_input(task)
            
            if success and result:
                print("✅ Task processed successfully")
                # Show first few lines of response
                response_lines = str(result).split('\n')[:5]
                for line in response_lines:
                    print(f"  {line}")
                if len(str(result).split('\n')) > 5:
                    print("  ...")
            else:
                print("❌ Task processing failed")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()
    
    print("=" * 40)
    print("✅ Final coding test completed!")

if __name__ == "__main__":
    asyncio.run(test_final_coding_task())