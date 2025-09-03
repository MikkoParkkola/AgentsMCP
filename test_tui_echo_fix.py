#!/usr/bin/env python3
"""
Test script to verify TUI input echo functionality.

This script tests that characters typed by the user are visible on screen
when using the Revolutionary TUI interface.
"""

import sys
import os
import subprocess
import time
import threading
import signal
from pathlib import Path

def test_tui_echo_manually():
    """
    Manual test that launches TUI and provides instructions for human testing.
    """
    print("🧪 TUI Echo Test - Manual Verification")
    print("=" * 50)
    print()
    print("This test will launch the TUI in a subprocess.")
    print("You should be able to see characters as you type them.")
    print()
    print("TEST PROCEDURE:")
    print("1. TUI will start and show: 💬 > ")
    print("2. Type: 'hello world'")
    print("3. You SHOULD see: 💬 > hello world")
    print("4. Press Enter to send the message")
    print("5. Type 'quit' and press Enter to exit")
    print()
    print("❌ FAILURE: If you see 💬 >  but can't see what you type")
    print("✅ SUCCESS: If you can see 💬 > hello world as you type")
    print()
    
    response = input("Ready to start test? (y/n): ").strip().lower()
    if response != 'y':
        print("Test cancelled.")
        return
    
    print("\n🚀 Starting TUI in 3 seconds...")
    time.sleep(3)
    
    try:
        # Launch TUI as subprocess in current terminal
        result = subprocess.run([
            sys.executable, "-m", "agentsmcp.cli", "tui"
        ], cwd=Path(__file__).parent)
        
        print(f"\n📊 TUI exited with code: {result.returncode}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")

def test_tui_echo_programmatically():
    """
    Programmatic test that verifies terminal state after Rich Live display.
    """
    print("🔬 TUI Echo Test - Programmatic Verification")
    print("=" * 50)
    
    try:
        import termios
        import tty
        
        if not sys.stdin.isatty():
            print("❌ Not running in a TTY - cannot test terminal echo")
            return False
            
        print("✅ TTY detected")
        
        # Save original terminal settings
        fd = sys.stdin.fileno()
        original_settings = termios.tcgetattr(fd)
        
        print(f"📋 Original terminal settings (lflag): {original_settings[3]:08b}")
        
        # Check if ECHO flag is set (should be for normal terminals)
        echo_enabled = bool(original_settings[3] & termios.ECHO)
        canonical_enabled = bool(original_settings[3] & termios.ICANON)
        
        print(f"🔊 ECHO flag enabled: {echo_enabled}")
        print(f"📝 ICANON flag enabled: {canonical_enabled}")
        
        if echo_enabled and canonical_enabled:
            print("✅ Terminal is in correct state for input echo")
            return True
        else:
            print("❌ Terminal is NOT in correct state for input echo")
            return False
            
    except Exception as e:
        print(f"❌ Programmatic test failed: {e}")
        return False

def main():
    """Run the echo test suite."""
    print("🔧 TUI Input Echo Fix Verification")
    print("=" * 60)
    print()
    
    # Run programmatic test first
    programmatic_success = test_tui_echo_programmatically()
    
    print("\n" + "=" * 60)
    
    if programmatic_success:
        print("📋 Programmatic test passed - terminal is in correct state")
        print("🎯 Proceeding to manual verification test...")
        print()
        test_tui_echo_manually()
    else:
        print("❌ Programmatic test failed - terminal state issues detected")
        print("🔧 Manual test may not work correctly")
        
        response = input("\nRun manual test anyway? (y/n): ").strip().lower()
        if response == 'y':
            test_tui_echo_manually()

if __name__ == "__main__":
    main()