#!/usr/bin/env python3
"""
TTY Detection Diagnosis Tool
This helps diagnose why the TUI might be running in demo mode instead of Rich mode.
"""

import sys
import os

def diagnose_tty():
    print("🔍 TTY Detection Diagnosis")
    print("=" * 50)
    
    # Check basic TTY status
    stdin_tty = sys.stdin.isatty() if hasattr(sys.stdin, 'isatty') else False
    stdout_tty = sys.stdout.isatty() if hasattr(sys.stdout, 'isatty') else False  
    stderr_tty = sys.stderr.isatty() if hasattr(sys.stderr, 'isatty') else False
    
    print(f"📟 stdin.isatty():  {stdin_tty}")
    print(f"📟 stdout.isatty(): {stdout_tty}")
    print(f"📟 stderr.isatty(): {stderr_tty}")
    
    # Check environment variables that might affect TTY detection
    print(f"\n🌍 Environment Variables:")
    tty_vars = ['TERM', 'COLORTERM', 'CI', 'GITHUB_ACTIONS', 'TRAVIS', 'JENKINS', 'BUILD']
    for var in tty_vars:
        value = os.environ.get(var, 'not set')
        print(f"   {var}: {value}")
    
    # Check file descriptors
    print(f"\n📂 File Descriptors:")
    try:
        print(f"   stdin fd:  {sys.stdin.fileno()}")
        print(f"   stdout fd: {sys.stdout.fileno()}")  
        print(f"   stderr fd: {sys.stderr.fileno()}")
    except Exception as e:
        print(f"   Error getting file descriptors: {e}")
    
    # Check terminal capabilities
    print(f"\n⚙️  Terminal Capabilities:")
    try:
        import termios
        import tty
        print("   ✅ termios and tty modules available")
        
        if stdin_tty:
            try:
                fd = sys.stdin.fileno()
                attrs = termios.tcgetattr(fd)
                print("   ✅ Can read terminal attributes from stdin")
            except Exception as e:
                print(f"   ❌ Cannot read terminal attributes from stdin: {e}")
                
                # Try /dev/tty fallback
                try:
                    test_fd = os.open('/dev/tty', os.O_RDONLY)
                    attrs = termios.tcgetattr(test_fd) 
                    os.close(test_fd)
                    print("   ✅ Can read terminal attributes from /dev/tty")
                except Exception as e2:
                    print(f"   ❌ Cannot read terminal attributes from /dev/tty: {e2}")
        else:
            print("   ⚠️  stdin is not a TTY")
            
    except ImportError:
        print("   ❌ termios/tty modules not available")
    
    # Check Rich availability
    print(f"\n🎨 Rich Library:")
    try:
        import rich
        from rich.console import Console
        from rich.live import Live
        print(f"   ✅ Rich available, version: {rich.__version__}")
        
        console = Console()
        print(f"   ✅ Console created, size: {console.size}")
        print(f"   ✅ Console is_terminal: {console.is_terminal}")
        print(f"   ✅ Console legacy_windows: {console.legacy_windows}")
    except ImportError as e:
        print(f"   ❌ Rich not available: {e}")
    except Exception as e:
        print(f"   ❌ Rich error: {e}")
    
    # Final assessment
    print(f"\n🎯 Assessment:")
    if stdin_tty and stdout_tty:
        print("   ✅ Should use Rich TUI mode")
    elif stdin_tty:
        print("   ⚠️  Mixed TTY state - stdin TTY but stdout not")
    else:
        print("   ❌ Should use demo mode")
    
    print(f"\n💡 If you see a Rich TUI but this shows demo mode, there may be")
    print(f"   a TTY detection inconsistency in the application.")

if __name__ == "__main__":
    diagnose_tty()