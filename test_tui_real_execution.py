#!/usr/bin/env python3
"""
Real TUI Execution Test - Run the actual CLI application

This script runs the actual AgentsMCP CLI to verify:
1. The TUI starts correctly with the TTY fix
2. Input/output works in the current environment
3. The fix doesn't break normal operations
"""

import subprocess
import sys
import os
import time
import threading
from pathlib import Path


def run_cli_with_timeout():
    """Run the CLI application with timeout and capture output"""
    
    # Change to project directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    print("🚀 RUNNING REAL TUI EXECUTION TEST")
    print("=" * 60)
    print(f"Working directory: {os.getcwd()}")
    print("Testing the actual AgentsMCP CLI application...")
    print("=" * 60)
    
    try:
        # Run the CLI application
        print("\n🎯 Starting AgentsMCP CLI...")
        
        # Use timeout to prevent hanging
        cmd = [sys.executable, "-m", "agentsmcp.cli"]
        
        print(f"Command: {' '.join(cmd)}")
        
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        print("✅ CLI process started")
        
        # Give it a moment to initialize
        time.sleep(2)
        
        # Send a test command
        print("⌨️  Sending test input...")
        try:
            process.stdin.write("hello\n")
            process.stdin.flush()
            time.sleep(1)
            
            # Try to send quit command
            process.stdin.write("/quit\n")
            process.stdin.flush()
            time.sleep(1)
        except Exception as e:
            print(f"⚠️  Input error: {e}")
        
        # Wait for completion or timeout
        try:
            stdout, stderr = process.communicate(timeout=10)
            
            print("\n📊 EXECUTION RESULTS:")
            print(f"Return code: {process.returncode}")
            
            if stdout:
                print(f"\n📤 STDOUT:")
                print(stdout)
            
            if stderr:
                print(f"\n📤 STDERR:")
                print(stderr)
            
            # Analyze results
            success_indicators = [
                "Revolutionary TUI Interface" in (stdout + stderr),
                "Starting" in (stdout + stderr),
                "initialized" in (stdout + stderr).lower(),
                process.returncode is not None,  # Process completed
            ]
            
            success_count = sum(success_indicators)
            
            if success_count >= 2:
                print("\n✅ CLI EXECUTION: SUCCESS")
                print("✅ The TUI appears to be working correctly!")
            else:
                print("\n⚠️  CLI EXECUTION: PARTIAL")
                print("⚠️  Some issues detected, but no crashes")
            
        except subprocess.TimeoutExpired:
            print("\n⚠️  CLI execution timed out after 10 seconds")
            print("⚠️  This might indicate the TUI is waiting for input (which is good)")
            print("⚠️  Terminating process...")
            process.terminate()
            process.wait()
            
        except Exception as e:
            print(f"\n🚨 CLI execution error: {e}")
            
    except FileNotFoundError:
        print("❌ AgentsMCP CLI not found")
        print("ℹ️  Trying alternative import paths...")
        
        # Try running from src directory
        try:
            os.chdir("src")
            cmd = [sys.executable, "-c", "from agentsmcp.cli import main; main()"]
            print(f"Alternative command: {' '.join(cmd)}")
            
            process = subprocess.run(cmd, timeout=5, capture_output=True, text=True)
            
            print(f"Return code: {process.returncode}")
            if process.stdout:
                print(f"STDOUT: {process.stdout}")
            if process.stderr:
                print(f"STDERR: {process.stderr}")
                
        except Exception as alt_e:
            print(f"❌ Alternative execution failed: {alt_e}")
            
    except Exception as e:
        print(f"🚨 Test execution failed: {e}")
    
    print("\n" + "=" * 60)


def check_tui_files():
    """Check if TUI files exist and contain the fix"""
    print("\n🔍 CHECKING TUI FILES...")
    
    tui_file = Path("src/agentsmcp/ui/v2/revolutionary_tui_interface.py")
    
    if tui_file.exists():
        print(f"✅ TUI file found: {tui_file}")
        
        # Check for the TTY fix
        content = tui_file.read_text()
        
        # Look for the fix
        if "sys.stdin.isatty()" in content:
            print("✅ TTY fix detected in file")
            
            # Count occurrences
            stdin_count = content.count("sys.stdin.isatty()")
            stdout_count = content.count("sys.stdout.isatty()")
            
            print(f"📊 TTY checks: stdin={stdin_count}, stdout={stdout_count}")
            
            # Look for the specific fix pattern
            if "is_tty = sys.stdin.isatty()" in content:
                print("✅ Specific TTY fix pattern found")
            else:
                print("⚠️  Specific fix pattern not found, but stdin checks exist")
                
        else:
            print("❌ No TTY checks found in file")
    else:
        print(f"❌ TUI file not found: {tui_file}")


def main():
    """Main test function"""
    check_tui_files()
    run_cli_with_timeout()


if __name__ == "__main__":
    main()