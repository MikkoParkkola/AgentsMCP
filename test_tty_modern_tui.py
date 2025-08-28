#!/usr/bin/env python3
"""
Test ModernTUI in a proper TTY environment using pty.
"""

import pty
import os
import sys
import subprocess
from pathlib import Path

def test_in_pty():
    """Test the ModernTUI in a pseudo-TTY environment."""
    
    print("🔍 Testing ModernTUI in pseudo-TTY environment...")
    
    # Create a test script that will run in the pty
    test_script = """
import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_tui():
    from agentsmcp.ui.modern_tui import ModernTUI
    from agentsmcp.ui.theme_manager import ThemeManager
    from agentsmcp.ui.cli_app import CLIConfig
    from agentsmcp.ui.command_interface import CommandInterface
    
    print("📊 TTY Status in pseudo-TTY:")
    print(f"stdin.isatty(): {sys.stdin.isatty()}")
    print(f"stdout.isatty(): {sys.stdout.isatty()}")
    
    if not sys.stdin.isatty():
        print("❌ Still not a TTY!")
        return False
        
    print("✅ We have a TTY! Testing ModernTUI...")
    
    # Create dependencies
    theme_manager = ThemeManager()
    config = CLIConfig(show_welcome=False)
    
    # Mock orchestration manager
    class MockOrchestrationManager:
        async def get_system_status(self):
            return {"status": "mock"}
        async def initialize(self, mode="hybrid"):
            return {"status": "initialized"}
    
    orchestration_manager = MockOrchestrationManager()
    command_interface = CommandInterface(
        orchestration_manager=orchestration_manager,
        theme_manager=theme_manager,
        agent_manager=None,
        app_config=None
    )
    
    tui = ModernTUI(
        config=config,
        theme_manager=theme_manager,
        conversation_manager=command_interface.conversation_manager,
        orchestration_manager=orchestration_manager,
        theme="auto",
        no_welcome=True,
    )
    
    print("🚀 Starting ModernTUI.run()...")
    
    # Run for a short time then exit
    try:
        # We'll timeout quickly since this is a test
        await asyncio.wait_for(tui.run(), timeout=2.0)
    except asyncio.TimeoutError:
        print("⏰ Timeout (normal for this test)")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

asyncio.run(test_tui())
"""
    
    # Write the test script
    script_path = Path("temp_tty_test.py")
    script_path.write_text(test_script)
    
    try:
        # Create a pseudo-TTY and run the test
        master_fd, slave_fd = pty.openpty()
        
        # Fork a child process
        pid = os.fork()
        
        if pid == 0:  # Child process
            # Close master, make slave our controlling terminal
            os.close(master_fd)
            os.dup2(slave_fd, sys.stdin.fileno())
            os.dup2(slave_fd, sys.stdout.fileno())
            os.dup2(slave_fd, sys.stderr.fileno())
            os.close(slave_fd)
            
            # Execute our test script
            os.execv(sys.executable, [sys.executable, str(script_path)])
        
        else:  # Parent process
            os.close(slave_fd)
            
            # Send some input after a brief delay (simulating user typing '/quit')
            import time
            time.sleep(1)
            os.write(master_fd, b"/quit\n")
            
            # Read output
            try:
                output = os.read(master_fd, 4096).decode('utf-8', errors='ignore')
                print("📋 Output from TTY test:")
                print(output)
                
                # Check if it ran the rich TUI (not fallback)
                if "=== AgentsMCP (fallback CLI) ===" in output:
                    print("❌ Still falling back to basic CLI even in TTY!")
                elif "TTY Status" in output and "stdin.isatty(): True" in output:
                    print("✅ Successfully detected TTY and may have run rich TUI!")
                else:
                    print("⚠️  Unclear result - check output above")
                    
            except OSError:
                pass
            
            # Wait for child
            os.waitpid(pid, 0)
            os.close(master_fd)
            
    except Exception as e:
        print(f"❌ PTY test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if script_path.exists():
            script_path.unlink()

def test_actual_cli_command():
    """Test the actual CLI command with timeout."""
    
    print("\n🔍 Testing actual ./agentsmcp interactive command...")
    
    try:
        # Test with very short timeout
        result = subprocess.run(
            ["./agentsmcp", "interactive", "--no-welcome"],
            timeout=3.0,  # 3 second timeout
            capture_output=True,
            text=True,
            input="/quit\n"  # Send quit command
        )
        
        print("📋 Command output:")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print("Return code:", result.returncode)
        
        if "Failed to start ModernTUI, falling back to legacy mode" in result.stderr:
            print("❌ Found the exact error message!")
            print("💡 This confirms ModernTUI is failing during startup")
        elif "=== AgentsMCP (fallback CLI) ===" in result.stdout:
            print("⚠️  ModernTUI is immediately falling back due to TTY detection")
        else:
            print("✅ Command ran without obvious errors")
            
    except subprocess.TimeoutExpired:
        print("⏰ Command timed out (expected for interactive mode)")
    except Exception as e:
        print(f"❌ Command test failed: {e}")

if __name__ == "__main__":
    print("🧪 ModernTUI TTY Testing")
    print("=" * 40)
    
    test_in_pty()
    test_actual_cli_command()