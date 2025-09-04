#!/usr/bin/env python3
"""
TUI User Interaction Scenario Testing

This test suite validates the gap between expected and actual TUI behavior,
focusing on interactive scenarios that should work but don't.
"""

import pytest
import subprocess
import time
import os
import sys
import pty
import select
import signal
import threading
from pathlib import Path

class TUITestResult:
    """Container for TUI test results"""
    def __init__(self):
        self.stdout = ""
        self.stderr = ""
        self.exit_code = None
        self.runtime_seconds = 0
        self.timed_out = False
        self.interactive_detected = False
        self.visual_interface_detected = False
        self.demo_mode_detected = False

@pytest.fixture
def project_root():
    """Get project root directory"""
    return Path(__file__).parent

class TUIInteractionTester:
    """Test TUI interaction scenarios"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.agentsmcp_path = project_root / "agentsmcp"
    
    def run_tui_with_timeout(self, timeout_seconds=10, use_pty=False, input_sequence=None):
        """
        Run TUI with timeout and optional input sequence
        
        Args:
            timeout_seconds: How long to wait before killing
            use_pty: Use pseudo-terminal for TTY simulation
            input_sequence: List of strings to send as input
        """
        result = TUITestResult()
        start_time = time.time()
        
        if use_pty:
            return self._run_with_pty(timeout_seconds, input_sequence, result)
        else:
            return self._run_without_pty(timeout_seconds, result)
    
    def _run_without_pty(self, timeout_seconds, result):
        """Run TUI without PTY (current failing scenario)"""
        try:
            process = subprocess.Popen(
                [sys.executable, "-m", "agentsmcp", "tui"],
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            start_time = time.time()
            
            try:
                stdout, stderr = process.communicate(timeout=timeout_seconds)
                result.stdout = stdout
                result.stderr = stderr
                result.exit_code = process.returncode
                result.runtime_seconds = time.time() - start_time
                
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                result.stdout = stdout
                result.stderr = stderr
                result.timed_out = True
                result.runtime_seconds = time.time() - start_time
                
        except Exception as e:
            result.stderr = f"Exception running TUI: {e}"
            result.exit_code = -1
        
        # Analyze output
        self._analyze_output(result)
        return result
    
    def _run_with_pty(self, timeout_seconds, input_sequence, result):
        """Run TUI with PTY to simulate real terminal"""
        master_fd = None
        pid = None
        
        try:
            # Create PTY
            master_fd, slave_fd = pty.openpty()
            
            # Fork process
            pid = os.fork()
            
            if pid == 0:  # Child process
                os.close(master_fd)
                os.dup2(slave_fd, 0)  # stdin
                os.dup2(slave_fd, 1)  # stdout  
                os.dup2(slave_fd, 2)  # stderr
                os.close(slave_fd)
                
                # Execute TUI
                os.execvp(sys.executable, [sys.executable, "-m", "agentsmcp", "tui"])
            
            else:  # Parent process
                os.close(slave_fd)
                
                # Collect output with timeout
                output_parts = []
                start_time = time.time()
                
                # Send input sequence if provided
                if input_sequence:
                    def send_inputs():
                        time.sleep(0.5)  # Wait for TUI to start
                        for input_str in input_sequence:
                            time.sleep(0.2)
                            os.write(master_fd, input_str.encode())
                    
                    input_thread = threading.Thread(target=send_inputs)
                    input_thread.start()
                
                while time.time() - start_time < timeout_seconds:
                    ready, _, _ = select.select([master_fd], [], [], 0.1)
                    
                    if ready:
                        try:
                            data = os.read(master_fd, 1024)
                            if data:
                                output_parts.append(data.decode('utf-8', errors='ignore'))
                        except OSError:
                            break
                    
                    # Check if child process is still alive
                    try:
                        pid_status, _ = os.waitpid(pid, os.WNOHANG)
                        if pid_status != 0:
                            break
                    except OSError:
                        break
                
                # Cleanup
                try:
                    os.kill(pid, signal.SIGTERM)
                    os.waitpid(pid, 0)
                except OSError:
                    pass
                
                result.stdout = ''.join(output_parts)
                result.runtime_seconds = time.time() - start_time
                
        except Exception as e:
            result.stderr = f"PTY execution error: {e}"
            result.exit_code = -1
        
        finally:
            if master_fd:
                os.close(master_fd)
        
        self._analyze_output(result)
        return result
    
    def _analyze_output(self, result):
        """Analyze TUI output to detect modes and behaviors"""
        stdout_lower = result.stdout.lower()
        stderr_lower = result.stderr.lower()
        
        # Detect demo mode
        result.demo_mode_detected = any(phrase in stdout_lower for phrase in [
            "demo mode", "non-tty environment", "demo countdown"
        ])
        
        # Detect interactive features
        result.interactive_detected = any(phrase in stdout_lower for phrase in [
            "waiting for input", "type your message", "interactive mode",
            "press enter", "user input", "awaiting command"
        ])
        
        # Detect visual interface
        result.visual_interface_detected = any(phrase in stdout_lower for phrase in [
            "revolutionary tui interface", "rich", "panels", "layout",
            "symphony dashboard", "ai command composer"
        ])

# Test Classes

class TestTUIBasicLifecycle:
    """Test basic TUI startup, runtime, and shutdown"""
    
    def test_tui_startup_without_pty(self, project_root):
        """SCENARIO 1: Basic TUI startup without PTY - Current failing scenario"""
        tester = TUIInteractionTester(project_root)
        result = tester.run_tui_with_timeout(timeout_seconds=8, use_pty=False)
        
        print(f"\n=== TUI STARTUP TEST (No PTY) ===")
        print(f"Runtime: {result.runtime_seconds:.2f}s")
        print(f"Exit code: {result.exit_code}")
        print(f"Demo mode detected: {result.demo_mode_detected}")
        print(f"Interactive mode detected: {result.interactive_detected}")
        print(f"Visual interface detected: {result.visual_interface_detected}")
        
        print(f"\nSTDOUT:\n{result.stdout}")
        print(f"\nSTDERR:\n{result.stderr}")
        
        # EXPECTED vs ACTUAL analysis
        print(f"\n=== EXPECTED vs ACTUAL ===")
        print(f"EXPECTED: TUI should wait for user input indefinitely")
        print(f"ACTUAL: TUI runs for {result.runtime_seconds:.2f}s and exits")
        
        print(f"EXPECTED: Interactive interface with Rich components")  
        print(f"ACTUAL: Demo mode detected = {result.demo_mode_detected}")
        
        # This test documents the current broken behavior
        assert result.demo_mode_detected, "TUI should be in demo mode (current broken state)"
        assert not result.interactive_detected, "TUI should NOT be interactive (current broken state)"
        assert result.runtime_seconds < 10, "TUI should exit quickly in demo mode"
    
    def test_tui_startup_with_pty(self, project_root):
        """SCENARIO 1B: Basic TUI startup with PTY - What SHOULD happen"""
        if os.name == 'nt':  # Skip on Windows
            pytest.skip("PTY not available on Windows")
            
        tester = TUIInteractionTester(project_root)
        result = tester.run_tui_with_timeout(timeout_seconds=8, use_pty=True)
        
        print(f"\n=== TUI STARTUP TEST (With PTY) ===")
        print(f"Runtime: {result.runtime_seconds:.2f}s")
        print(f"Exit code: {result.exit_code}")  
        print(f"Demo mode detected: {result.demo_mode_detected}")
        print(f"Interactive mode detected: {result.interactive_detected}")
        print(f"Visual interface detected: {result.visual_interface_detected}")
        print(f"Timed out (good): {result.timed_out}")
        
        print(f"\nSTDOUT:\n{result.stdout}")
        print(f"\nSTDERR:\n{result.stderr}")
        
        # EXPECTED behavior with proper TTY
        print(f"\n=== EXPECTED BEHAVIOR WITH TTY ===")
        print(f"EXPECTED: TUI should stay running (timeout expected)")
        print(f"EXPECTED: Interactive interface, NOT demo mode")
        print(f"EXPECTED: Rich visual components")
        
        # This test shows what should happen with proper TTY
        # We expect it to timeout because TUI should keep running
        if result.timed_out:
            print("✅ GOOD: TUI stayed running (had to timeout)")
        else:
            print("❌ BAD: TUI exited instead of staying interactive")

class TestTUIInteractiveInput:
    """Test interactive input scenarios"""
    
    def test_user_input_simulation(self, project_root):
        """SCENARIO 2: Interactive input testing with simulated user input"""
        if os.name == 'nt':
            pytest.skip("PTY not available on Windows")
            
        tester = TUIInteractionTester(project_root)
        
        # Simulate user typing commands
        input_sequence = [
            "hello\n",           # Basic message
            "/help\n",           # Help command  
            "/clear\n",          # Clear command
            "/quit\n"            # Quit command
        ]
        
        result = tester.run_tui_with_timeout(
            timeout_seconds=10, 
            use_pty=True, 
            input_sequence=input_sequence
        )
        
        print(f"\n=== INTERACTIVE INPUT TEST ===")
        print(f"Runtime: {result.runtime_seconds:.2f}s")
        print(f"Demo mode: {result.demo_mode_detected}")
        print(f"Interactive: {result.interactive_detected}")
        
        print(f"\nSTDOUT:\n{result.stdout}")
        print(f"\nSTDERR:\n{result.stderr}")
        
        # Check if input was processed
        stdout_lower = result.stdout.lower()
        input_processed = any(cmd in stdout_lower for cmd in ["hello", "/help", "/clear", "/quit"])
        
        print(f"\n=== INPUT PROCESSING ANALYSIS ===")
        print(f"Input commands detected in output: {input_processed}")
        print(f"EXPECTED: User input should be visible and processed")
        print(f"ACTUAL: Input processed = {input_processed}")

class TestTUIVisualInterface:
    """Test visual interface components"""
    
    def test_rich_components_initialization(self, project_root):
        """SCENARIO 4: Visual interface testing - Rich components"""
        tester = TUIInteractionTester(project_root)
        result = tester.run_tui_with_timeout(timeout_seconds=8, use_pty=False)
        
        print(f"\n=== VISUAL INTERFACE TEST ===")
        
        # Check for specific Rich/TUI components in output
        component_indicators = {
            "Revolutionary TUI Interface": "revolutionary tui interface" in result.stdout.lower(),
            "Symphony Dashboard": "symphony dashboard" in result.stdout.lower(), 
            "AI Command Composer": "ai command composer" in result.stdout.lower(),
            "Rich panels": any(word in result.stdout.lower() for word in ["panel", "layout", "rich"]),
            "Status updates": any(word in result.stdout.lower() for word in ["status", "operational", "ready"]),
        }
        
        print(f"Component Detection:")
        for component, detected in component_indicators.items():
            print(f"  {component}: {detected}")
        
        print(f"\nSTDOUT:\n{result.stdout}")
        
        print(f"\n=== VISUAL INTERFACE ANALYSIS ===")
        print(f"EXPECTED: Rich visual components should initialize and display")
        print(f"ACTUAL: Components detected = {any(component_indicators.values())}")
        
        if result.demo_mode_detected:
            print("NOTE: Running in demo mode - visual components limited")

class TestTUILifecycleManagement:
    """Test TUI lifecycle - startup, runtime, shutdown"""
    
    def test_tui_should_stay_running(self, project_root):
        """SCENARIO 3: TUI lifecycle - should stay running until user exits"""
        tester = TUIInteractionTester(project_root)
        result = tester.run_tui_with_timeout(timeout_seconds=5, use_pty=False)
        
        print(f"\n=== TUI LIFECYCLE TEST ===")
        print(f"Runtime: {result.runtime_seconds:.2f}s")
        print(f"Exit code: {result.exit_code}")
        print(f"Demo mode: {result.demo_mode_detected}")
        
        print(f"\nSTDOUT:\n{result.stdout}")
        
        print(f"\n=== LIFECYCLE ANALYSIS ===")
        print(f"EXPECTED: TUI should stay running indefinitely waiting for user input")
        print(f"ACTUAL: TUI ran for {result.runtime_seconds:.2f}s and exited with code {result.exit_code}")
        
        if result.demo_mode_detected:
            print("ISSUE: TUI entered demo mode instead of interactive mode")
            print("CAUSE: Not running in TTY environment")
        
        # Document the broken behavior
        assert result.runtime_seconds < 10, "TUI exits quickly instead of staying interactive"
        assert result.demo_mode_detected, "TUI enters demo mode instead of interactive mode"

if __name__ == "__main__":
    # Run tests directly
    project_root = Path(__file__).parent
    
    print("🧪 TUI USER INTERACTION SCENARIO TESTING")
    print("=" * 60)
    print("Testing the gap between expected vs actual TUI behavior")
    print()
    
    # Initialize tester
    tester = TUIInteractionTester(project_root)
    
    # Test 1: Basic startup (current failing scenario)
    print("🔍 TEST 1: Basic TUI Startup (No TTY)")
    result1 = tester.run_tui_with_timeout(timeout_seconds=8, use_pty=False)
    print(f"Result: Demo mode={result1.demo_mode_detected}, Runtime={result1.runtime_seconds:.2f}s")
    
    # Test 2: With PTY (what should happen)
    if os.name != 'nt':
        print("\n🔍 TEST 2: TUI Startup with PTY")  
        result2 = tester.run_tui_with_timeout(timeout_seconds=5, use_pty=True)
        print(f"Result: Demo mode={result2.demo_mode_detected}, Timed out={result2.timed_out}")
    
    print(f"\n📋 SUMMARY")
    print(f"ISSUE IDENTIFIED: TUI detects non-TTY environment and enters demo mode")
    print(f"EXPECTED: Interactive TUI that stays running")
    print(f"ACTUAL: Demo mode that runs for ~3 seconds and exits")
    print(f"SOLUTION NEEDED: Proper TTY handling or interactive mode fallback")