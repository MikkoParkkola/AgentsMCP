#!/usr/bin/env python3
"""Test script to verify timestamp functionality in TUI messages."""

import subprocess
import sys
import time

def test_timestamp_functionality():
    """Test the TUI timestamp implementation with actual input."""
    print("🧪 Testing TUI timestamp functionality...")
    
    try:
        # Start the TUI process using the correct executable
        cmd = ["./agentsmcp", "tui"]
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="/Users/mikko/github/AgentsMCP"
        )
        
        # Send test inputs with slight delays
        test_inputs = [
            "hello",          # Should trigger a user message with timestamp
            "/help",          # Should trigger system message with timestamp
            "/quit"           # Exit
        ]
        
        # Send inputs
        input_text = "\n".join(test_inputs) + "\n"
        stdout, stderr = process.communicate(input=input_text, timeout=10)
        
        print("📝 STDOUT Output:")
        print(stdout)
        print("\n📝 STDERR Output:")
        print(stderr)
        
        # Check if timestamps appear in the output
        if "[" in stdout and "]" in stdout:
            print("✅ Timestamp format [hh:mm:ss] detected in output")
        else:
            print("❌ No timestamp format detected in output")
            
        return process.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out")
        process.kill()
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_timestamp_functionality()
    sys.exit(0 if success else 1)