#!/usr/bin/env python3
"""
Test the fixed conversational interface detection.
"""

import asyncio
import sys
from src.agentsmcp.ui.command_interface import CommandInterface
from src.agentsmcp.ui.theme_manager import ThemeManager
from src.agentsmcp.orchestration.orchestration_manager import OrchestrationManager

def test_command_detection():
    """Test the _is_direct_command method"""
    print("🧪 Testing Command Detection Logic")
    print("=" * 50)
    
    # Create minimal command interface for testing
    theme_manager = ThemeManager()
    # Create a mock orchestration manager to avoid numpy dependency issues
    class MockOrchestrationManager:
        def __init__(self):
            pass
    
    orchestration_manager = MockOrchestrationManager()
    cmd_interface = CommandInterface(orchestration_manager, theme_manager)
    
    # Test cases: (input, expected_is_direct_command, explanation)
    test_cases = [
        # Direct commands (should be True)
        ("help", True, "single command word"),
        ("status", True, "single command word"),
        ("theme dark", True, "command with parameter"),
        ("execute --mode hybrid 'task'", True, "command with --parameters"),
        
        # Conversational inputs (should be False)  
        ("help me", False, "command + conversational indicator"),
        ("show me the status", False, "conversational request"),
        ("what is the status", False, "question form"),
        ("can you help me", False, "polite request"),
        ("please show status", False, "polite command"),
        ("tell me about the system", False, "conversational request"),
        ("how do I check status", False, "question"),
        
        # Edge cases
        ("", False, "empty input"),
        ("unknown command", False, "non-existent command"),
    ]
    
    results = []
    for input_text, expected, explanation in test_cases:
        actual = cmd_interface._is_direct_command(input_text)
        status = "✅ PASS" if actual == expected else "❌ FAIL"
        results.append((status, input_text, actual, expected, explanation))
        
        print(f"{status} '{input_text}' → {actual} (expected {expected}) - {explanation}")
    
    # Summary
    passed = sum(1 for result in results if result[0] == "✅ PASS")
    total = len(results)
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All command detection tests passed!")
        return True
    else:
        print("⚠️ Some command detection tests failed")
        return False

if __name__ == "__main__":
    success = test_command_detection()
    sys.exit(0 if success else 1)