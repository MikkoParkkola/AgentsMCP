#!/usr/bin/env python3

"""
Test console renderer streaming behavior in different environments.
This test helps us understand what's really happening with the streaming display.
"""

import sys
import os

def test_console_behavior():
    """Test console behavior for streaming updates."""
    
    print("🧪 Console Streaming Behavior Test")
    print("=" * 50)
    
    # Test environment detection
    is_tty = sys.stdout.isatty()
    print(f"Environment: TTY={is_tty}")
    
    # Test basic carriage return behavior
    print("\n1. Testing carriage return behavior:")
    
    if is_tty:
        # In a real terminal, this should overwrite
        for i in range(5):
            sys.stdout.write(f"\rTesting... {i}")
            sys.stdout.flush()
            import time
            time.sleep(0.5)
        print()  # Newline to finish
    else:
        print("  Not in TTY - carriage return won't work properly")
        print("  This explains why streaming output is duplicating")
    
    # Test escape sequences
    print("\n2. Testing escape sequences:")
    
    if is_tty:
        sys.stdout.write("Before escape")
        sys.stdout.write("\033[K")  # Clear to end of line
        sys.stdout.write("After clear")
        sys.stdout.flush()
        print()
    else:
        print("  Escape sequences don't work in non-TTY environment")
    
    # Recommendations
    print("\n🎯 Recommendations:")
    if is_tty:
        print("  ✅ TTY environment - streaming should work with \\r")
        print("  ✅ Use carriage return and escape sequences")
    else:
        print("  ❌ Non-TTY environment - different approach needed")
        print("  🔧 For non-TTY: Use dots or progress indicators instead")
        print("  🔧 Don't attempt line overwriting")
    
    print("\n3. Alternative streaming approach for non-TTY:")
    print("🤖 AI: ", end="", flush=True)
    for i in range(10):
        if i % 3 == 0:  # Show progress every few updates
            print(".", end="", flush=True)
        import time
        time.sleep(0.1)
    print(" [Response complete]")
    
    print("\n✅ Console streaming test completed")

if __name__ == "__main__":
    test_console_behavior()