#!/usr/bin/env python3
"""
Test more CLI commands
"""
import asyncio
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, 'src')

from agentsmcp.ui.cli_app import CLIApp

async def test_more_commands():
    """Test additional CLI commands"""
    
    # Create CLI app
    cli_app = CLIApp()
    
    print("Testing additional CLI commands...")
    
    # Test commands directly
    command_interface = cli_app.command_interface
    
    if command_interface:
        print("\n=== Testing agents list command ===")
        success, result = await command_interface.execute_single_command("agents list")
        print(f"Agents list command result: success={success}")
        
        print("\n=== Testing symphony start command ===")
        success, result = await command_interface.execute_single_command("symphony start")
        print(f"Symphony start command result: success={success}")
        
        print("\n=== Testing theme info command ===")
        success, result = await command_interface.execute_single_command("theme info")
        print(f"Theme info command result: success={success}")
        
        print("\n=== Testing config show command ===")
        success, result = await command_interface.execute_single_command("config show")
        print(f"Config show command result: success={success}")
        
        print("\n=== Testing help execute command ===")
        success, result = await command_interface.execute_single_command("help execute")
        print(f"Help execute command result: success={success}")
        
    else:
        print("Command interface not available")

if __name__ == "__main__":
    asyncio.run(test_more_commands())