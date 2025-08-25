#!/usr/bin/env python3
"""
Test specific CLI commands
"""
import asyncio
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, 'src')

from agentsmcp.ui.cli_app import CLIApp

async def test_commands():
    """Test specific CLI commands"""
    
    # Create CLI app
    cli_app = CLIApp()
    
    print("Testing CLI commands...")
    
    # Test commands directly
    command_interface = cli_app.command_interface
    
    if command_interface:
        print("\n=== Testing status command ===")
        success, result = await command_interface.execute_single_command("status")
        print(f"Status command result: success={success}")
        
        print("\n=== Testing execute command with self-improvement task ===")
        task = "Analyze the AgentsMCP codebase and suggest 3 concrete improvements to enhance performance, usability, or code quality"
        success, result = await command_interface.execute_single_command(f'execute "{task}"')
        print(f"Execute command result: success={success}")
        
    else:
        print("Command interface not available")

if __name__ == "__main__":
    asyncio.run(test_commands())