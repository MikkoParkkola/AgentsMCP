#!/usr/bin/env python3
"""
Test the command prefix logic directly
"""

import sys
sys.path.insert(0, 'src')

from agentsmcp.config import Config
from agentsmcp.ui.command_interface import CommandInterface

async def test_command_prefix():
    """Test command prefix detection"""
    print("🧪 Testing Command Prefix Logic")
    print("=" * 40)
    
    # Create command interface
    config = Config()
    cmd_interface = CommandInterface(config)
    
    # Test cases
    test_cases = [
        ("/help", True, "Should be detected as command"),
        ("/agents", True, "Should be detected as command"),
        ("/status", True, "Should be detected as command"),
        ("help", False, "Should NOT be detected as command (missing /)"),
        ("Hello, can you help me?", False, "Should be treated as conversation"),
        ("Write a Python function", False, "Should be treated as conversation"),
        ("/", False, "Just slash should not be valid"),
        ("", False, "Empty input should not be valid"),
    ]
    
    print("Testing command detection:")
    all_passed = True
    
    for input_text, expected, description in test_cases:
        result = cmd_interface._is_direct_command(input_text)
        status = "✅ PASS" if result == expected else "❌ FAIL"
        print(f"  {status} '{input_text}' -> {result} ({description})")
        if result != expected:
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("✅ All command prefix tests passed!")
    else:
        print("❌ Some tests failed!")
    
    return all_passed

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_command_prefix())