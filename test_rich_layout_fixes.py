#!/usr/bin/env python3
"""Test script to verify Rich TUI layout fixes."""

import sys
import subprocess
import time
import os

def test_rich_layout_fixes():
    """Test the Rich TUI layout improvements."""
    print("🧪 Testing Rich TUI layout fixes...")
    
    try:
        # Set environment for Rich support 
        env = os.environ.copy()
        env['TERM'] = 'xterm-256color'
        env['FORCE_COLOR'] = '1'
        
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
            "hello",          # Should show user message + AI response
            "/help",          # Should show system help message
            "/quit"           # Exit
        ]
        
        # Send inputs
        input_text = "\n".join(test_inputs) + "\n"
        stdout, stderr = process.communicate(input=input_text, timeout=15)
        
        print("📝 Layout Test Results:")
        print("=" * 60)
        print(stdout)
        print("=" * 60)
        
        if stderr.strip():
            print("🔧 STDERR:")
            print(stderr)
            print("=" * 60)
        
        # Check for layout improvements
        success_indicators = []
        
        # Check if we got clean output without excessive boxes
        if "╭" in stdout and "╰" in stdout:
            box_count = stdout.count("╭")
            if box_count < 20:  # Should have reasonable number of boxes, not excessive
                success_indicators.append(("Reasonable box count", True))
            else:
                success_indicators.append(("Reasonable box count", False))
        else:
            success_indicators.append(("Rich boxes present", False))
        
        # Check for timestamps
        success_indicators.append(("Timestamps", "[" in stdout and "]" in stdout))
        
        # Check for single goodbye message
        goodbye_count = stdout.count("Goodbye")
        success_indicators.append(("Single goodbye", goodbye_count <= 1))
        
        # Display results
        print("\n🔍 Success Indicators:")
        for indicator, status in success_indicators:
            status_symbol = "✅" if status else "❌"
            print(f"   {status_symbol} {indicator}")
        
        # Overall assessment
        passed = all(status for _, status in success_indicators)
        
        if passed:
            print("\n✅ Rich TUI layout fixes appear to be working correctly!")
        else:
            print("\n⚠️ Some issues may remain with the Rich TUI layout.")
            
        return process.returncode == 0 and passed
        
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out")
        process.kill()
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_rich_layout_fixes()
    sys.exit(0 if success else 1)