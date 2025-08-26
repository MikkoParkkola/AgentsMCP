#!/usr/bin/env python3
"""
Test script for AgentsMCP interactive mode with new command prefix system
"""

import subprocess
import sys
import time
import os

def test_interactive_mode():
    """Test the interactive mode with various inputs"""
    print("🧪 Testing AgentsMCP Interactive Mode with / Command Prefix")
    print("=" * 60)
    
    # Prepare test commands and conversations
    test_inputs = [
        "/help",  # Should show help
        "/agents",  # Should list agents  
        "/status",  # Should show system status
        "Hello, can you help me with a Python coding task?",  # Conversational
        "Write a simple function to calculate fibonacci numbers",  # Conversational  
        "/exit"  # Should exit
    ]
    
    print("Test inputs prepared:")
    for i, inp in enumerate(test_inputs, 1):
        input_type = "COMMAND" if inp.startswith('/') else "CONVERSATION"
        print(f"  {i}. [{input_type}] {inp}")
    
    print("\n" + "=" * 60)
    print("Starting interactive session...")
    print("Note: This will run the actual AgentsMCP binary")
    print("=" * 60)
    
    return test_inputs

if __name__ == "__main__":
    test_inputs = test_interactive_mode()
    
    print("\nTo test manually, run:")
    print("PYTHONPATH=src python -m agentsmcp --mode interactive")
    print("\nThen try these inputs:")
    for inp in test_inputs:
        print(f"  {inp}")