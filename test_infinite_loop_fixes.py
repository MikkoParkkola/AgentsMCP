#!/usr/bin/env python3
"""
Test script to verify the three critical infinite loop fixes:
1. Timeout protection in MCP calls
2. Preprocessing bypass for simple inputs
3. Enhanced simple input detection

This script tests the fixes without actually running the full TUI.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agentsmcp.ui.v3.chat_engine import ChatEngine
from agentsmcp.conversation.llm_client import LLMClient


def test_simple_input_detection():
    """Test Fix 3: Enhanced Simple Input Detection"""
    print("Testing Fix 3: Enhanced Simple Input Detection")
    print("=" * 50)
    
    engine = ChatEngine()
    
    # Test cases that should be detected as simple
    simple_inputs = [
        "hello",
        "hi",
        "how are you?",
        "how are you",
        "what's up",
        "thanks",
        "bye",
        "who?",
        "what?",
        "how?",
        "good morning",
        "who are you",
        "are you there",
        "hi there"  # This should be detected as simple (15 chars or less)
    ]
    
    # Test cases that should NOT be detected as simple  
    complex_inputs = [
        "Can you help me write a Python script for data analysis?",
        "I need to implement a REST API with authentication",
        "What are the best practices for database optimization?",
        "Create a comprehensive test suite",
        "this is a longer input that should not be considered simple"
    ]
    
    print("Simple inputs (should return True):")
    for inp in simple_inputs:
        result = engine._is_simple_input(inp)
        status = "✓" if result else "✗"
        print(f"  {status} '{inp}' -> {result}")
    
    print("\nComplex inputs (should return False):")
    for inp in complex_inputs:
        result = engine._is_simple_input(inp)
        status = "✓" if not result else "✗"
        print(f"  {status} '{inp}' -> {result}")
    
    # Summary
    simple_results = [engine._is_simple_input(inp) for inp in simple_inputs]
    complex_results = [engine._is_simple_input(inp) for inp in complex_inputs]
    
    simple_correct = sum(simple_results)
    complex_correct = sum(not r for r in complex_results)
    
    print(f"\nResults:")
    print(f"Simple inputs correctly detected: {simple_correct}/{len(simple_inputs)}")
    print(f"Complex inputs correctly detected: {complex_correct}/{len(complex_inputs)}")
    
    if simple_correct == len(simple_inputs) and complex_correct == len(complex_inputs):
        print("✓ Fix 3: Enhanced Simple Input Detection - PASSED")
        return True
    else:
        print("✗ Fix 3: Enhanced Simple Input Detection - FAILED")
        return False


def test_timeout_protection_logic():
    """Test Fix 1: Timeout Protection Logic (without actual MCP calls)"""
    print("\nTesting Fix 1: Timeout Protection Logic")
    print("=" * 50)
    
    # We can't test the actual timeout without making real MCP calls,
    # but we can verify the logic is in place
    try:
        # Check if the timeout logic exists in the code
        llm_client_path = Path(__file__).parent / "src" / "agentsmcp" / "conversation" / "llm_client.py"
        with open(llm_client_path, 'r') as f:
            content = f.read()
            
        # Look for timeout protection patterns
        has_asyncio_wait_for = "asyncio.wait_for(" in content
        has_timeout_exception = "asyncio.TimeoutError" in content
        has_timeout_value = "timeout=30.0" in content
        
        print(f"asyncio.wait_for found: {has_asyncio_wait_for}")
        print(f"TimeoutError handling found: {has_timeout_exception}")
        print(f"30-second timeout found: {has_timeout_value}")
        
        if has_asyncio_wait_for and has_timeout_exception and has_timeout_value:
            print("✓ Fix 1: Timeout Protection Logic - PASSED (code inspection)")
            return True
        else:
            print("✗ Fix 1: Timeout Protection Logic - FAILED (code inspection)")
            return False
            
    except Exception as e:
        print(f"✗ Fix 1: Timeout Protection Logic - ERROR: {e}")
        return False


def test_preprocessing_bypass_logic():
    """Test Fix 2: Preprocessing Bypass Logic (without full initialization)"""
    print("\nTesting Fix 2: Preprocessing Bypass Logic")
    print("=" * 50)
    
    try:
        # Check if the bypass logic exists in the code
        chat_engine_path = Path(__file__).parent / "src" / "agentsmcp" / "ui" / "v3" / "chat_engine.py"
        with open(chat_engine_path, 'r') as f:
            content = f.read()
            
        # Look for preprocessing bypass patterns
        has_original_preprocessing = "original_preprocessing" in content
        has_preprocessing_disabled = "preprocessing_enabled = False" in content
        has_finally_restore = "finally:" in content and "original_preprocessing is not None" in content
        has_simple_input_check = "is_simple and self._llm_client" in content
        
        print(f"Original preprocessing backup found: {has_original_preprocessing}")
        print(f"Preprocessing disable logic found: {has_preprocessing_disabled}")
        print(f"Finally restore logic found: {has_finally_restore}")
        print(f"Simple input check found: {has_simple_input_check}")
        
        if has_original_preprocessing and has_preprocessing_disabled and has_finally_restore and has_simple_input_check:
            print("✓ Fix 2: Preprocessing Bypass Logic - PASSED (code inspection)")
            return True
        else:
            print("✗ Fix 2: Preprocessing Bypass Logic - FAILED (code inspection)")
            return False
            
    except Exception as e:
        print(f"✗ Fix 2: Preprocessing Bypass Logic - ERROR: {e}")
        return False


def main():
    """Run all fix tests"""
    print("AgentsMCP Infinite Loop Fixes Verification")
    print("=" * 60)
    
    results = []
    
    # Test each fix
    results.append(test_simple_input_detection())
    results.append(test_timeout_protection_logic())
    results.append(test_preprocessing_bypass_logic())
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL FIXES VERIFIED - Ready for testing!")
        print("\nTo test manually:")
        print("1. Run: ./agentsmcp tui")
        print("2. Type: how are you?")
        print("3. Should respond quickly without infinite loop")
        return 0
    else:
        print("❌ SOME FIXES FAILED - Review implementation")
        return 1


if __name__ == "__main__":
    sys.exit(main())