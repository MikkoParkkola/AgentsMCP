#!/usr/bin/env python3
"""
Test script to verify the TUI infinite loop fix.

This script tests:
1. Simple inputs don't trigger infinite progress loops
2. Complex inputs still work with task tracking  
3. All background threads are properly cleaned up
4. No infinite status updates flood the console
"""

import subprocess
import time
import signal
import os
import threading


def test_simple_input_no_infinite_loop():
    """Test that simple inputs like 'hello' don't cause infinite loops."""
    print("🧪 Testing simple input handling...")
    
    cmd = ["./agentsmcp", "tui"]
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd="/Users/mikko/github/AgentsMCP"
    )
    
    # Send simple input
    process.stdin.write("hello\n")
    process.stdin.flush()
    
    # Wait for initial processing
    time.sleep(3)
    
    # Send quit command
    process.stdin.write("/quit\n")
    process.stdin.flush()
    
    # Wait for clean shutdown
    try:
        stdout, stderr = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        print("❌ FAIL: Process didn't exit cleanly (infinite loop suspected)")
        return False
    
    # Check output for infinite loops
    lines = stdout.split('\n')
    status_lines = [line for line in lines if '⏸️ General_Agent' in line or '░░░░░░' in line]
    
    if len(status_lines) > 5:  # More than expected suggests infinite loop
        print(f"❌ FAIL: Too many progress status lines ({len(status_lines)}) - infinite loop detected")
        print("Sample status lines:")
        for line in status_lines[:3]:
            print(f"  {line}")
        return False
    
    print("✅ PASS: Simple input processed without infinite loops")
    return True


def test_complex_input_with_task_tracking():
    """Test that complex inputs still trigger proper task tracking."""
    print("\n🧪 Testing complex input with task tracking...")
    
    cmd = ["./agentsmcp", "tui"]
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd="/Users/mikko/github/AgentsMCP"
    )
    
    # Send complex input
    process.stdin.write("What is the capital of France?\n")
    process.stdin.flush()
    
    # Wait for processing
    time.sleep(5)
    
    # Send quit
    process.stdin.write("/quit\n") 
    process.stdin.flush()
    
    try:
        stdout, stderr = process.communicate(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        print("❌ FAIL: Complex input process didn't exit cleanly")
        return False
    
    # Check for enhanced preprocessing (indicates task tracking was triggered)
    if "Optimizing prompt with enhanced context" in stdout or "📝" in stdout:
        print("✅ PASS: Complex input triggered enhanced task tracking")
        return True
    else:
        print("❌ FAIL: Complex input didn't trigger expected task tracking")
        print("Sample output:")
        print(stdout[:500])
        return False


def test_no_infinite_status_flooding():
    """Test that status updates don't flood the console infinitely."""
    print("\n🧪 Testing for status flooding prevention...")
    
    cmd = ["./agentsmcp", "tui"]
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd="/Users/mikko/github/AgentsMCP"
    )
    
    # Send input and let it run for a bit
    process.stdin.write("hi\n")
    process.stdin.flush()
    
    # Collect output for 4 seconds
    output_lines = []
    start_time = time.time()
    
    def collect_output():
        while time.time() - start_time < 4:
            try:
                line = process.stdout.readline()
                if line:
                    output_lines.append(line.strip())
            except:
                break
    
    collector_thread = threading.Thread(target=collect_output)
    collector_thread.start()
    
    time.sleep(4)
    
    # Quit
    process.stdin.write("/quit\n")
    process.stdin.flush()
    
    try:
        process.communicate(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
    
    collector_thread.join(timeout=1)
    
    # Check for excessive status flooding
    status_flood_lines = [line for line in output_lines if "⏱️" in line and "Planning complete" in line]
    
    if len(status_flood_lines) > 3:
        print(f"❌ FAIL: Status flooding detected ({len(status_flood_lines)} repeated status lines)")
        print("Sample repeated lines:")
        for line in status_flood_lines[:2]:
            print(f"  {line}")
        return False
    
    print("✅ PASS: No status flooding detected")
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("🚀 Running TUI Infinite Loop Fix Tests")
    print("=" * 50)
    
    tests = [
        test_simple_input_no_infinite_loop,
        test_complex_input_with_task_tracking,
        test_no_infinite_status_flooding,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ FAIL: {test_func.__name__} threw exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Infinite loop fix is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Issues may still exist.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)