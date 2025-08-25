#!/usr/bin/env python3
"""
Test script to interact with AgentsMCP CLI
"""
import asyncio
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, 'src')

from agentsmcp.ui.cli_app import CLIApp
from agentsmcp.ui.command_interface import CommandInterface

async def test_cli():
    """Test basic CLI functionality"""
    
    # Create CLI app
    cli_app = CLIApp()
    
    print("Testing CLI command execution...")
    
    # Test help command directly
    command_interface = cli_app.command_interface
    
    if command_interface:
        print("\n=== Testing help command ===")
        success, result = await command_interface.execute_single_command("help")
        print(f"Help command result: success={success}")
        
        print("\n=== Testing settings command ===")  
        success, result = await command_interface.execute_single_command("settings")
        print(f"Settings command result: success={success}")
        
        print("\n=== Testing generate-config command ===")
        success, result = await command_interface.execute_single_command("generate-config")
        print(f"Generate-config command result: success={success}")
        
        print("\n=== Testing status command ===")
        success, result = await command_interface.execute_single_command("status")
        print(f"Status command result: success={success}")
        
    else:
        print("Command interface not available")

if __name__ == "__main__":
    asyncio.run(test_cli())