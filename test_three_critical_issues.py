#!/usr/bin/env python3
"""
Systematic reproduction of three critical AgentsMCP issues.

Test Cases:
1. "hello" → Routes correctly but returns EMPTY response
2. "how are you?" → WORKS correctly 
3. "why don't you answer to my hello?" → INFINITE LOOP in sequential thinking

This script will reproduce the issues systematically to trace execution paths.
"""

import asyncio
import sys
import os
import logging
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set TUI mode to prevent console contamination
os.environ['AGENTSMCP_TUI_MODE'] = '1'

# Set up detailed logging 
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(name)s:%(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)

# Create debug logger
debug_logger = logging.getLogger("DEBUG_REPRODUCTION")

async def test_single_input(test_case_name: str, user_input: str, expected_behavior: str):
    """Test a single input case with detailed tracing."""
    print(f"\n{'='*80}")
    print(f"TEST CASE: {test_case_name}")
    print(f"INPUT: '{user_input}'")
    print(f"EXPECTED: {expected_behavior}")
    print(f"{'='*80}")
    
    try:
        # Import and create chat engine
        from agentsmcp.ui.v3.chat_engine import ChatEngine
        
        # Create fresh chat engine for each test
        chat_engine = ChatEngine()
        
        # Track execution times and status
        start_time = time.time()
        response_received = False
        empty_response = False
        timeout_triggered = False
        
        # Set up callbacks to track behavior
        def status_callback(status: str):
            elapsed = time.time() - start_time
            print(f"[{elapsed:.1f}s] STATUS: {status}")
            
        def message_callback(message):
            nonlocal response_received, empty_response
            elapsed = time.time() - start_time
            print(f"[{elapsed:.1f}s] MESSAGE [{message.role.value}]: {repr(message.content)}")
            if message.role.value == "assistant":
                response_received = True
                if not message.content.strip():
                    empty_response = True
        
        def error_callback(error: str):
            elapsed = time.time() - start_time
            print(f"[{elapsed:.1f}s] ERROR: {error}")
        
        chat_engine.set_callbacks(status_callback, message_callback, error_callback)
        
        # Process the input with a timeout
        print(f"[0.0s] PROCESSING INPUT: '{user_input}'")
        
        try:
            # Set a reasonable timeout for the test
            result = await asyncio.wait_for(
                chat_engine.process_input(user_input),
                timeout=15.0  # 15 second timeout
            )
            
            elapsed = time.time() - start_time
            print(f"[{elapsed:.1f}s] COMPLETED: result={result}")
            
        except asyncio.TimeoutError:
            timeout_triggered = True
            elapsed = time.time() - start_time
            print(f"[{elapsed:.1f}s] TIMEOUT: Process took longer than 15 seconds")
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[{elapsed:.1f}s] EXCEPTION: {type(e).__name__}: {e}")
            
        # Analyze results
        print(f"\n--- ANALYSIS ---")
        print(f"Response received: {response_received}")
        print(f"Empty response: {empty_response}")
        print(f"Timeout triggered: {timeout_triggered}")
        print(f"Total time: {time.time() - start_time:.1f}s")
        
        # Cleanup
        await chat_engine.cleanup()
        
        return {
            "response_received": response_received,
            "empty_response": empty_response, 
            "timeout_triggered": timeout_triggered,
            "total_time": time.time() - start_time
        }
        
    except Exception as e:
        print(f"TEST SETUP ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

async def main():
    """Run all three test cases systematically."""
    print("🔍 SYSTEMATIC REPRODUCTION OF THREE CRITICAL ISSUES")
    print("=" * 80)
    
    # Test Case 1: Empty response issue
    result1 = await test_single_input(
        test_case_name="1. EMPTY RESPONSE BUG",
        user_input="hello",
        expected_behavior="Simple input → direct response BUT returns EMPTY"
    )
    
    # Wait between tests
    await asyncio.sleep(1)
    
    # Test Case 2: Working case (control)
    result2 = await test_single_input(
        test_case_name="2. WORKING CASE (CONTROL)",
        user_input="how are you?", 
        expected_behavior="Simple input → direct response AND works correctly"
    )
    
    # Wait between tests  
    await asyncio.sleep(1)
    
    # Test Case 3: Infinite loop issue
    result3 = await test_single_input(
        test_case_name="3. INFINITE LOOP BUG",
        user_input="why don't you answer to my hello?",
        expected_behavior="Complex input → preprocessing → planning → execution BUT gets stuck"
    )
    
    # Summary analysis
    print(f"\n{'='*80}")
    print("SUMMARY ANALYSIS")
    print(f"{'='*80}")
    
    print("\nTest 1 (hello):", result1)
    print("Test 2 (how are you?):", result2)  
    print("Test 3 (complex input):", result3)
    
    # Identify patterns
    print(f"\n{'='*80}")
    print("PATTERNS IDENTIFIED")
    print(f"{'='*80}")
    
    if result1.get("empty_response"):
        print("❌ ISSUE 1: Empty response confirmed for 'hello'")
    else:
        print("✅ Issue 1: Not reproduced")
        
    if result2.get("response_received") and not result2.get("empty_response"):
        print("✅ CONTROL: 'how are you?' works correctly") 
    else:
        print("❌ CONTROL: Unexpected behavior in working case")
        
    if result3.get("timeout_triggered"):
        print("❌ ISSUE 3: Infinite loop confirmed for complex input")
    else:
        print("✅ Issue 3: Not reproduced")
        
    print(f"\n{'='*80}")
    print("READY FOR ROOT CAUSE ANALYSIS")
    print(f"{'='*80}")

if __name__ == "__main__":
    asyncio.run(main())