#!/usr/bin/env python3
"""
Quick test script for TUI functionality.
Tests both typing echo and scrollback behavior.
"""

import subprocess
import sys
import time

print("🧪 Testing AgentsMCP TUI Functionality")
print("=" * 50)
print()
print("This will launch the TUI for manual testing:")
print("1. Check if you can see characters as you type them")
print("2. Test that keyboard shortcuts work (arrows, etc.)")
print("3. Try typing slash commands like '/help' or '/quit'")
print("4. Exit with '/quit' and check if scrollback flooding occurred")
print()
print("Key things to verify:")
print("✓ Characters appear immediately as you type")
print("✓ Cursor is visible and moves properly")
print("✓ Keyboard shortcuts work (arrows, tab, backspace)")
print("✓ No duplicate TUI frames in terminal history after exit")
print()

input("Press Enter to launch TUI (or Ctrl+C to cancel)...")

try:
    # Launch the TUI in interactive mode
    subprocess.run([
        sys.executable, '-m', 'agentsmcp.cli', 'interactive', '--no-welcome'
    ], check=False)
    
    print()
    print("TUI test completed!")
    print("Check the terminal scrollback above - there should be minimal/no frame duplication.")
    
except KeyboardInterrupt:
    print("\nTest cancelled.")
except Exception as e:
    print(f"\nError running TUI: {e}")