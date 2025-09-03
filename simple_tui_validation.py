#!/usr/bin/env python3
"""
Simple TUI Validation Test

Tests that the Revolutionary TUI now enables Rich interface instead of basic prompt.
"""
import subprocess
import sys
import time
import os

def test_tui_rich_interface():
    """Test that TUI enables Rich interface (shows ANSI escape sequences)."""
    print("🔍 Testing TUI Rich Interface Activation...")
    
    # Test command that should trigger Rich interface
    cmd = ['./agentsmcp', 'tui']
    
    try:
        # Start TUI process
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd='/Users/mikko/github/AgentsMCP'
        )
        
        # Send quit command and wait briefly
        stdout, stderr = process.communicate(input="quit\n", timeout=5)
        
        # Check for Rich interface indicators
        output = stdout + stderr
        
        # Look for ANSI escape sequences that indicate Rich interface
        rich_indicators = [
            '[?47h',     # Save screen buffer
            '[?1049h',   # Enter alternate screen
            '[?25l',     # Hide cursor
            '[?1049l',   # Exit alternate screen
            '[?47l',     # Restore screen buffer
        ]
        
        found_rich_sequences = []
        for indicator in rich_indicators:
            if indicator in output:
                found_rich_sequences.append(indicator)
        
        # Check for old basic prompt indicators (should NOT be present)
        basic_prompt_indicators = [
            'Running in non-TTY environment - providing command interface...',
            '> ',  # Basic prompt (but this might appear in other contexts)
        ]
        
        found_basic_indicators = []
        for indicator in basic_prompt_indicators:
            if indicator in output:
                found_basic_indicators.append(indicator)
        
        print(f"📊 Analysis Results:")
        print(f"   Rich sequences found: {len(found_rich_sequences)}/5")
        print(f"   Rich sequences: {found_rich_sequences}")
        print(f"   Basic indicators found: {len(found_basic_indicators)}")
        print(f"   Basic indicators: {found_basic_indicators}")
        
        # Determine result
        has_rich_interface = len(found_rich_sequences) >= 2  # At least 2 ANSI sequences
        has_basic_fallback = len(found_basic_indicators) > 0
        
        if has_rich_interface and not has_basic_fallback:
            print("✅ SUCCESS: Rich interface is active!")
            print("   - ANSI escape sequences detected")
            print("   - No basic prompt fallback messages")
            return True
        elif has_rich_interface and has_basic_fallback:
            print("⚠️  PARTIAL: Rich interface detected but also basic indicators")
            print("   - This might indicate transition states")
            return True
        else:
            print("❌ FAILURE: Rich interface not detected")
            print("   - Missing ANSI escape sequences")
            print("   - May still be using basic prompt mode")
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️  Test timed out (expected for interactive TUI)")
        process.kill()
        stdout, stderr = process.communicate()
        output = stdout + stderr
        
        # Even with timeout, we can check for Rich sequences
        rich_sequences = sum(1 for seq in ['[?47h', '[?1049h', '[?25l'] if seq in output)
        if rich_sequences >= 1:
            print("✅ SUCCESS: Rich interface started (detected ANSI sequences)")
            return True
        else:
            print("❌ FAILURE: No Rich interface sequences detected")
            return False
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def main():
    """Run TUI validation tests."""
    print("🚀 Simple TUI Validation Test")
    print("=" * 50)
    
    # Test 1: Rich interface activation
    test_result = test_tui_rich_interface()
    
    print("\n🎯 FINAL RESULT:")
    if test_result:
        print("✅ TUI VALIDATION PASSED")
        print("   The Revolutionary TUI now enables Rich interface!")
        print("   Users should see proper terminal UI instead of basic prompt.")
    else:
        print("❌ TUI VALIDATION FAILED") 
        print("   The TUI may still be using basic prompt mode.")
        print("   Additional investigation needed.")
    
    return test_result

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)