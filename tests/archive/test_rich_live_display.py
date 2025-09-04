#!/usr/bin/env python3
"""Test script to verify Rich Live display panels functionality."""

import subprocess
import sys
import time
import os

def test_rich_live_display():
    """Test the Rich Live display implementation."""
    print("🧪 Testing Rich Live display panels...")
    
    try:
        # Set environment to force TTY detection for Rich
        env = os.environ.copy()
        env['TERM'] = 'xterm-256color'  # Force color support
        env['FORCE_COLOR'] = '1'        # Force color output
        
        # Start the TUI process
        cmd = ["./agentsmcp", "tui"]
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="/Users/mikko/github/AgentsMCP",
            env=env
        )
        
        # Send test inputs
        test_inputs = [
            "test message",   # Should trigger user/AI messages with timestamps
            "/help",          # Should trigger system message
            "/quit"           # Exit
        ]
        
        # Send inputs
        input_text = "\n".join(test_inputs) + "\n"
        stdout, stderr = process.communicate(input=input_text, timeout=15)
        
        print("📝 STDOUT Output:")
        print(stdout)
        print("\n📝 STDERR Output:")
        print(stderr)
        
        # Check for Rich Live display indicators
        success_indicators = [
            "[" in stdout and "]" in stdout,  # Timestamps
            "PHASE 3" in stdout,              # Phase 3 renderer
            "Rich" in stdout                  # Rich renderer active
        ]
        
        if all(success_indicators):
            print("✅ Rich Live display appears to be working")
        else:
            print("⚠️ Rich Live display may not be fully active")
            print(f"   - Timestamps: {success_indicators[0]}")
            print(f"   - Phase 3: {success_indicators[1]}")
            print(f"   - Rich active: {success_indicators[2]}")
            
        return process.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out")
        process.kill()
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_rich_live_display()
    sys.exit(0 if success else 1)