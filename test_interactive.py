#!/usr/bin/env python3
"""
Test interactive UI mode with simulated user input
"""
import asyncio
import sys
import os
from unittest.mock import patch
from io import StringIO

# Add the src directory to Python path
sys.path.insert(0, 'src')

async def test_interactive_ui():
    """Test interactive UI mode"""
    print("Testing AgentsMCP interactive UI mode...")
    
    # Simulate user inputs
    test_inputs = [
        'help',
        'status', 
        'execute "Create a simple Python script that prints Hello World"',
        'agents list',
        'exit'
    ]
    
    # Mock input to simulate user interaction
    input_iter = iter(test_inputs)
    
    def mock_input(prompt):
        try:
            cmd = next(input_iter)
            print(f"{prompt}{cmd}")  # Show what we're "typing"
            return cmd
        except StopIteration:
            return 'exit'
    
    try:
        from agentsmcp.ui.cli_app import CLIApp
        
        # Create and configure CLI app
        cli_app = CLIApp()
        
        # Test interactive mode with mocked input
        with patch('builtins.input', side_effect=mock_input):
            print("Starting interactive mode test...")
            await cli_app._run_interactive_mode()
        
        print("Interactive mode test completed successfully!")
        
    except KeyboardInterrupt:
        print("\nInteractive mode test interrupted - this is expected behavior")
    except Exception as e:
        print(f"Interactive mode test encountered error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_interactive_ui())