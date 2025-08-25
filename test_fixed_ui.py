#!/usr/bin/env python3
"""
Test fixed UI alignment and rendering
"""
import asyncio
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, 'src')

from agentsmcp.ui.cli_app import CLIApp

async def test_fixed_ui():
    """Test fixed UI rendering"""
    
    # Create CLI app
    cli_app = CLIApp()
    
    print("Testing fixed UI alignment and rendering...")
    
    # Test commands directly
    command_interface = cli_app.command_interface
    
    if command_interface:
        print("\n=== Testing fixed help command ===")
        success, result = await command_interface.execute_single_command("help")
        print(f"Help command result: success={success}")
        
        print("\n=== Testing fixed status command ===")
        success, result = await command_interface.execute_single_command("status")
        print(f"Status command result: success={success}")
        
        print("\n=== Testing fixed generate-config command ===")
        success, result = await command_interface.execute_single_command("generate-config")
        print(f"Generate-config command result: success={success}")
        
        print("\n=== Testing fixed execute command ===")
        task = "Test the UI alignment and rendering quality"
        success, result = await command_interface.execute_single_command(f'execute "{task}"')
        print(f"Execute command result: success={success}")
        
    else:
        print("Command interface not available")

if __name__ == "__main__":
    asyncio.run(test_fixed_ui())