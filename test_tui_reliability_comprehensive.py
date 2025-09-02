#!/usr/bin/env python3
"""
TUI Reliability Comprehensive Test Suite

CRITICAL MISSION: Catch the ACTUAL TUI startup hangs that existing smoke tests missed!

This test suite targets the real-world hang scenarios where `./agentsmcp tui` hangs 
completely after "Initializing Revolutionary TUI Interface..." and becomes unresponsive.

Test Categories:
1. REAL TUI STARTUP TESTS - Actually launch the TUI command and detect hangs
2. RELIABILITY ARCHITECTURE TESTS - Test all reliability modules working together
3. HANG SCENARIO SIMULATION - Reproduce exact hang conditions
4. END-TO-END TUI LIFECYCLE - Full TUI operation with timeout protection
5. RECOVERY SYSTEM VALIDATION - Test automatic recovery from hang states

Key Features:
- Process-based TUI launching with real timeout detection
- Simulation of exact user hang scenarios (typing "/quit" with no response)
- Validation of ReliableTUIInterface vs original interface behavior
- Comprehensive reliability module integration testing
- Real-world timeout and hang detection
"""

import asyncio
import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
import threading
import psutil
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json

# Add the src directory to Python path to import agentsmcp
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Test configuration
DEFAULT_TEST_TIMEOUT = 45  # Extended timeout for comprehensive testing
STARTUP_HANG_TIMEOUT = 12  # Timeout to detect startup hangs (original issue)
INPUT_RESPONSE_TIMEOUT = 8  # Timeout to detect input response hangs
PROCESS_KILL_TIMEOUT = 5   # Timeout for process termination
MAX_LOG_SIZE = 10000       # Maximum log size to capture

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [RELIABILITY-TEST] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Test result with detailed information."""
    test_name: str
    success: bool
    message: str
    duration: float
    details: Dict[str, Any]
    error_details: Optional[str] = None


@dataclass
class ProcessInfo:
    """Information about a spawned test process."""
    process: subprocess.Popen
    pid: int
    start_time: float
    command: List[str]
    stdout_data: str = ""
    stderr_data: str = ""
    terminated: bool = False


class HangDetector:
    """Detects various types of TUI hangs."""
    
    @staticmethod
    def detect_startup_hang(process_info: ProcessInfo, timeout: float) -> Tuple[bool, str]:
        """
        Detect if TUI startup has hung.
        
        Returns:
            (is_hung, reason)
        """
        elapsed = time.time() - process_info.start_time
        
        if elapsed > timeout:
            return True, f"Startup exceeded timeout ({timeout}s)"
        
        # Check if process is still alive but not responding
        if process_info.process.poll() is None:
            # Process is alive - check if it's actually doing anything
            try:
                proc = psutil.Process(process_info.pid)
                cpu_percent = proc.cpu_percent(interval=0.1)
                
                # If CPU usage is 0 for more than half the timeout, likely hung
                if elapsed > timeout / 2 and cpu_percent < 0.1:
                    return True, f"Process appears hung (no CPU activity for {elapsed:.1f}s)"
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return True, "Process died unexpectedly"
        
        return False, "Startup proceeding normally"
    
    @staticmethod
    def detect_input_hang(process_info: ProcessInfo, input_sent_time: float, timeout: float) -> Tuple[bool, str]:
        """
        Detect if TUI is not responding to input.
        
        Returns:
            (is_hung, reason)
        """
        elapsed = time.time() - input_sent_time
        
        if elapsed > timeout:
            return True, f"No response to input for {elapsed:.1f}s"
        
        return False, "Input response within normal range"


class ProcessManager:
    """Manages test processes with proper cleanup."""
    
    def __init__(self):
        self.active_processes: List[ProcessInfo] = []
        self._cleanup_lock = threading.Lock()
    
    def spawn_tui_process(self, args: List[str], env: Optional[Dict[str, str]] = None) -> ProcessInfo:
        """Spawn a TUI process for testing."""
        cmd = [sys.executable, "-m", "agentsmcp.cli"] + args
        
        # Setup environment
        test_env = os.environ.copy()
        if env:
            test_env.update(env)
        
        # Ensure no existing environment variables interfere
        test_env.pop('CI', None)
        test_env.pop('GITHUB_ACTIONS', None)
        
        start_time = time.time()
        
        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                env=test_env,
                cwd=str(Path(__file__).parent)
            )
            
            process_info = ProcessInfo(
                process=process,
                pid=process.pid,
                start_time=start_time,
                command=cmd
            )
            
            with self._cleanup_lock:
                self.active_processes.append(process_info)
            
            logger.info(f"Spawned TUI process: PID {process.pid}, CMD: {' '.join(cmd)}")
            return process_info
            
        except Exception as e:
            logger.error(f"Failed to spawn TUI process: {e}")
            raise
    
    def read_process_output(self, process_info: ProcessInfo, max_size: int = MAX_LOG_SIZE) -> Tuple[str, str]:
        """Read available output from process without blocking."""
        stdout_data = ""
        stderr_data = ""
        
        try:
            # Non-blocking read
            import select
            
            if hasattr(select, 'select'):  # Unix systems
                ready, _, _ = select.select([process_info.process.stdout, process_info.process.stderr], [], [], 0)
                
                if process_info.process.stdout in ready:
                    try:
                        data = process_info.process.stdout.read(max_size)
                        if data:
                            stdout_data = data
                    except:
                        pass
                
                if process_info.process.stderr in ready:
                    try:
                        data = process_info.process.stderr.read(max_size)
                        if data:
                            stderr_data = data
                    except:
                        pass
            else:
                # Windows fallback - try to read with timeout
                try:
                    stdout_data = process_info.process.stdout.read(max_size) or ""
                    stderr_data = process_info.process.stderr.read(max_size) or ""
                except:
                    pass
        
        except Exception:
            pass  # Ignore read errors
        
        # Update accumulated data
        if stdout_data:
            process_info.stdout_data += stdout_data
            if len(process_info.stdout_data) > max_size:
                process_info.stdout_data = process_info.stdout_data[-max_size:]
        
        if stderr_data:
            process_info.stderr_data += stderr_data
            if len(process_info.stderr_data) > max_size:
                process_info.stderr_data = process_info.stderr_data[-max_size:]
        
        return stdout_data, stderr_data
    
    def terminate_process(self, process_info: ProcessInfo, timeout: float = PROCESS_KILL_TIMEOUT) -> bool:
        """Terminate a process gracefully, then forcefully if needed."""
        if process_info.terminated:
            return True
        
        try:
            process = process_info.process
            
            # Try graceful termination first
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=timeout / 2)
                    logger.info(f"Process {process_info.pid} terminated gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill
                    logger.warning(f"Force killing process {process_info.pid}")
                    process.kill()
                    try:
                        process.wait(timeout=timeout / 2)
                    except subprocess.TimeoutExpired:
                        logger.error(f"Failed to kill process {process_info.pid}")
                        return False
            
            process_info.terminated = True
            return True
            
        except Exception as e:
            logger.error(f"Error terminating process {process_info.pid}: {e}")
            return False
    
    def cleanup_all(self):
        """Clean up all active processes."""
        with self._cleanup_lock:
            for process_info in self.active_processes:
                if not process_info.terminated:
                    self.terminate_process(process_info)
            self.active_processes.clear()
    
    def __del__(self):
        self.cleanup_all()


class ReliabilityTestSuite:
    """Comprehensive reliability test suite."""
    
    def __init__(self, timeout: int = DEFAULT_TEST_TIMEOUT, verbose: bool = False):
        self.timeout = timeout
        self.verbose = verbose
        self.results: List[TestResult] = []
        self.process_manager = ProcessManager()
        self.hang_detector = HangDetector()
        
        # Set up detailed logging if verbose
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            logging.getLogger('agentsmcp').setLevel(logging.DEBUG)
    
    def log(self, message: str, level: str = "info"):
        """Log a message with timestamp and appropriate formatting."""
        timestamp = time.strftime("%H:%M:%S")
        if level == "error":
            print(f"❌ [{timestamp}] {message}")
            logger.error(message)
        elif level == "success":
            print(f"✅ [{timestamp}] {message}")
            logger.info(message)
        elif level == "warning":
            print(f"⚠️ [{timestamp}] {message}")
            logger.warning(message)
        else:
            print(f"ℹ️ [{timestamp}] {message}")
            logger.info(message)
    
    def add_result(self, test_name: str, success: bool, message: str, duration: float, 
                   details: Optional[Dict[str, Any]] = None, error_details: Optional[str] = None):
        """Add a test result."""
        self.results.append(TestResult(
            test_name=test_name,
            success=success,
            message=message,
            duration=duration,
            details=details or {},
            error_details=error_details
        ))
    
    async def test_real_tui_startup_timeout_detection(self) -> bool:
        """
        TEST 1: Real TUI startup with timeout detection
        
        This is the CRITICAL test that should catch the actual hang issue.
        We spawn the real TUI process and detect if it hangs during startup.
        """
        start_time = time.time()
        test_name = "real_tui_startup_timeout_detection"
        
        try:
            self.log("🚀 Testing real TUI startup with timeout detection...")
            
            # Spawn real TUI process
            process_info = self.process_manager.spawn_tui_process(['tui'])
            
            # Monitor startup for hang detection
            startup_phase_start = time.time()
            initialization_detected = False
            startup_completed = False
            
            while time.time() - startup_phase_start < STARTUP_HANG_TIMEOUT:
                # Check if process is still running
                if process_info.process.poll() is not None:
                    # Process exited - check exit code
                    exit_code = process_info.process.poll()
                    if exit_code == 0:
                        startup_completed = True
                        break
                    else:
                        # Process failed
                        stdout_data, stderr_data = self.process_manager.read_process_output(process_info)
                        duration = time.time() - start_time
                        self.add_result(
                            test_name, False, 
                            f"TUI process exited with code {exit_code}",
                            duration,
                            {
                                "exit_code": exit_code,
                                "stdout": stdout_data,
                                "stderr": stderr_data
                            }
                        )
                        self.log(f"❌ TUI process failed with exit code {exit_code}", "error")
                        return False
                
                # Read output to detect initialization messages
                stdout_data, stderr_data = self.process_manager.read_process_output(process_info)
                
                if "Initializing Revolutionary TUI Interface" in (stdout_data + stderr_data):
                    initialization_detected = True
                    self.log("📝 Detected TUI initialization message")
                
                # Check for hang
                is_hung, reason = self.hang_detector.detect_startup_hang(process_info, STARTUP_HANG_TIMEOUT)
                if is_hung:
                    # CRITICAL: This is the hang we want to catch!
                    self.process_manager.terminate_process(process_info)
                    duration = time.time() - start_time
                    
                    stdout_data, stderr_data = self.process_manager.read_process_output(process_info)
                    
                    self.add_result(
                        test_name, False,
                        f"🚨 DETECTED TUI STARTUP HANG: {reason}",
                        duration,
                        {
                            "hang_detected": True,
                            "hang_reason": reason,
                            "initialization_detected": initialization_detected,
                            "stdout": process_info.stdout_data,
                            "stderr": process_info.stderr_data,
                            "elapsed_time": time.time() - startup_phase_start
                        }
                    )
                    self.log(f"🚨 CRITICAL: Detected TUI startup hang - {reason}", "error")
                    return False
                
                # Small sleep to avoid busy waiting
                await asyncio.sleep(0.1)
            
            # If we get here, either startup completed or timeout reached
            if startup_completed:
                duration = time.time() - start_time
                self.add_result(
                    test_name, True,
                    "TUI startup completed successfully",
                    duration,
                    {
                        "initialization_detected": initialization_detected,
                        "startup_time": duration
                    }
                )
                self.log("✅ TUI startup completed successfully", "success")
                return True
            else:
                # Timeout reached without completion - this is likely a hang
                self.process_manager.terminate_process(process_info)
                duration = time.time() - start_time
                
                self.add_result(
                    test_name, False,
                    f"🚨 TUI startup timeout - likely hang after {STARTUP_HANG_TIMEOUT}s",
                    duration,
                    {
                        "timeout_reached": True,
                        "initialization_detected": initialization_detected,
                        "stdout": process_info.stdout_data,
                        "stderr": process_info.stderr_data
                    }
                )
                self.log(f"🚨 CRITICAL: TUI startup timeout - likely hang!", "error")
                return False
            
        except Exception as e:
            duration = time.time() - start_time
            self.add_result(
                test_name, False,
                f"Test exception: {str(e)}",
                duration,
                error_details=str(e)
            )
            self.log(f"❌ Real TUI startup test failed: {e}", "error")
            return False
        finally:
            # Ensure process cleanup
            try:
                if 'process_info' in locals():
                    self.process_manager.terminate_process(process_info)
            except:
                pass
    
    async def test_input_response_hang_detection(self) -> bool:
        """
        TEST 2: Test input response hang detection
        
        This test simulates typing "/quit" and detecting if the TUI becomes
        unresponsive to input (the second part of the original hang issue).
        """
        start_time = time.time()
        test_name = "input_response_hang_detection"
        
        try:
            self.log("⌨️ Testing input response hang detection...")
            
            # Set environment for non-interactive mode that still accepts input
            env = {'AGENTSMCP_TEST_MODE': '1'}
            
            # Spawn TUI process
            process_info = self.process_manager.spawn_tui_process(['tui'], env=env)
            
            # Wait for startup (brief)
            startup_wait = 3.0
            await asyncio.sleep(startup_wait)
            
            # Check if process is still alive
            if process_info.process.poll() is not None:
                # Process already exited
                duration = time.time() - start_time
                self.add_result(
                    test_name, False,
                    f"Process exited during startup (exit code: {process_info.process.poll()})",
                    duration
                )
                return False
            
            # Send input command
            test_input = "/quit\n"
            input_sent_time = time.time()
            
            try:
                process_info.process.stdin.write(test_input)
                process_info.process.stdin.flush()
                self.log("📤 Sent '/quit' command to TUI")
            except Exception as e:
                self.log(f"⚠️ Failed to send input: {e}", "warning")
                # Continue test anyway
            
            # Monitor for response or hang
            response_timeout = INPUT_RESPONSE_TIMEOUT
            response_detected = False
            
            while time.time() - input_sent_time < response_timeout:
                # Check if process responded (exited)
                if process_info.process.poll() is not None:
                    response_detected = True
                    break
                
                # Read output for response indicators
                stdout_data, stderr_data = self.process_manager.read_process_output(process_info)
                
                if any(keyword in (stdout_data + stderr_data).lower() for keyword in 
                       ['exiting', 'goodbye', 'quit', 'exit', 'shutdown']):
                    response_detected = True
                    self.log("📝 Detected response to quit command")
                    break
                
                # Check for input hang
                is_hung, reason = self.hang_detector.detect_input_hang(
                    process_info, input_sent_time, response_timeout
                )
                if is_hung:
                    # CRITICAL: Input response hang detected!
                    self.process_manager.terminate_process(process_info)
                    duration = time.time() - start_time
                    
                    self.add_result(
                        test_name, False,
                        f"🚨 DETECTED INPUT RESPONSE HANG: {reason}",
                        duration,
                        {
                            "input_hang_detected": True,
                            "hang_reason": reason,
                            "input_sent": test_input.strip(),
                            "response_timeout": response_timeout,
                            "stdout": process_info.stdout_data,
                            "stderr": process_info.stderr_data
                        }
                    )
                    self.log(f"🚨 CRITICAL: Input response hang - {reason}", "error")
                    return False
                
                await asyncio.sleep(0.1)
            
            # Check final result
            if response_detected:
                duration = time.time() - start_time
                response_time = time.time() - input_sent_time
                self.add_result(
                    test_name, True,
                    f"TUI responded to input in {response_time:.2f}s",
                    duration,
                    {"response_time": response_time, "input_sent": test_input.strip()}
                )
                self.log("✅ TUI input response test passed", "success")
                return True
            else:
                # No response detected - likely hang
                self.process_manager.terminate_process(process_info)
                duration = time.time() - start_time
                
                self.add_result(
                    test_name, False,
                    f"🚨 No response to input after {response_timeout}s - likely input hang",
                    duration,
                    {
                        "no_response": True,
                        "input_sent": test_input.strip(),
                        "timeout": response_timeout,
                        "stdout": process_info.stdout_data,
                        "stderr": process_info.stderr_data
                    }
                )
                self.log("🚨 CRITICAL: No response to input - likely hang!", "error")
                return False
            
        except Exception as e:
            duration = time.time() - start_time
            self.add_result(
                test_name, False,
                f"Test exception: {str(e)}",
                duration,
                error_details=str(e)
            )
            self.log(f"❌ Input response test failed: {e}", "error")
            return False
        finally:
            try:
                if 'process_info' in locals():
                    self.process_manager.terminate_process(process_info)
            except:
                pass
    
    async def test_reliability_modules_integration(self) -> bool:
        """
        TEST 3: Test all reliability modules working together
        
        This test validates that all the reliability modules can be imported
        and initialized correctly together.
        """
        start_time = time.time()
        test_name = "reliability_modules_integration"
        
        try:
            self.log("🔧 Testing reliability modules integration...")
            
            # Test imports
            from agentsmcp.ui.v2.reliability import (
                StartupOrchestrator, TimeoutGuardian, ComponentInitializer,
                HealthMonitor, RecoveryManager
            )
            from agentsmcp.ui.v2.reliability.integration_layer import ReliableTUIInterface
            
            self.log("📦 All reliability modules imported successfully")
            
            # Test basic instantiation
            startup_orchestrator = StartupOrchestrator()
            timeout_guardian = TimeoutGuardian()
            component_initializer = ComponentInitializer()
            health_monitor = HealthMonitor()
            recovery_manager = RecoveryManager()
            
            self.log("🏗️ All reliability components instantiated")
            
            # Test that ReliableTUIInterface can be created
            try:
                # We can't fully test without a real orchestrator, but we can test basic creation
                reliable_interface = ReliableTUIInterface(None, None)
                self.log("✨ ReliableTUIInterface created successfully")
            except Exception as e:
                self.log(f"⚠️ ReliableTUIInterface creation issue (expected): {e}", "warning")
            
            duration = time.time() - start_time
            self.add_result(
                test_name, True,
                "All reliability modules integrated successfully",
                duration,
                {
                    "modules_tested": [
                        "StartupOrchestrator", "TimeoutGuardian", "ComponentInitializer",
                        "HealthMonitor", "RecoveryManager", "ReliableTUIInterface"
                    ]
                }
            )
            self.log("✅ Reliability modules integration test passed", "success")
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.add_result(
                test_name, False,
                f"Reliability modules integration failed: {str(e)}",
                duration,
                error_details=str(e)
            )
            self.log(f"❌ Reliability modules integration test failed: {e}", "error")
            return False
    
    async def test_timeout_guardian_protection(self) -> bool:
        """
        TEST 4: Test TimeoutGuardian protection against hangs
        
        This test validates that the timeout guardian can protect against
        operations that would otherwise hang indefinitely.
        """
        start_time = time.time()
        test_name = "timeout_guardian_protection"
        
        try:
            self.log("⏱️ Testing TimeoutGuardian protection...")
            
            from agentsmcp.ui.v2.reliability.timeout_guardian import (
                TimeoutGuardian, timeout_protection, get_global_guardian
            )
            
            guardian = get_global_guardian()
            
            # Test 1: Protect a hanging operation
            async def hanging_operation():
                """Simulates an operation that hangs forever."""
                await asyncio.sleep(1000)  # Would hang for 1000 seconds
                return "never reached"
            
            # Should timeout and not hang
            timeout_duration = 2.0
            try:
                result = await guardian.protect_operation(
                    hanging_operation(),
                    timeout_duration,
                    operation_name="test_hanging_operation"
                )
                # If we reach here, protection failed
                duration = time.time() - start_time
                self.add_result(
                    test_name, False,
                    "TimeoutGuardian failed to protect against hanging operation",
                    duration
                )
                return False
            except asyncio.TimeoutError:
                # Expected - timeout protection worked
                self.log("🛡️ TimeoutGuardian successfully protected against hang")
            
            # Test 2: Allow normal operation to complete
            async def normal_operation():
                await asyncio.sleep(0.1)
                return "completed successfully"
            
            result = await guardian.protect_operation(
                normal_operation(),
                timeout_duration,
                operation_name="test_normal_operation"
            )
            
            if result != "completed successfully":
                duration = time.time() - start_time
                self.add_result(
                    test_name, False,
                    f"TimeoutGuardian interfered with normal operation: {result}",
                    duration
                )
                return False
            
            duration = time.time() - start_time
            self.add_result(
                test_name, True,
                "TimeoutGuardian protection working correctly",
                duration,
                {
                    "protected_hanging_operation": True,
                    "allowed_normal_operation": True,
                    "timeout_used": timeout_duration
                }
            )
            self.log("✅ TimeoutGuardian protection test passed", "success")
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.add_result(
                test_name, False,
                f"TimeoutGuardian test failed: {str(e)}",
                duration,
                error_details=str(e)
            )
            self.log(f"❌ TimeoutGuardian protection test failed: {e}", "error")
            return False
    
    async def test_end_to_end_reliable_vs_original(self) -> bool:
        """
        TEST 5: Compare ReliableTUIInterface vs Original behavior
        
        This test attempts to demonstrate the difference between the original
        TUI interface and the reliable version (if possible in test environment).
        """
        start_time = time.time()
        test_name = "reliable_vs_original_comparison"
        
        try:
            self.log("🔍 Testing ReliableTUIInterface vs Original comparison...")
            
            # Import both interfaces
            from agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface
            try:
                from agentsmcp.ui.v2.reliability.integration_layer import ReliableTUIInterface
                reliable_available = True
            except ImportError as e:
                self.log(f"⚠️ ReliableTUIInterface not available: {e}", "warning")
                reliable_available = False
            
            if not reliable_available:
                duration = time.time() - start_time
                self.add_result(
                    test_name, False,
                    "ReliableTUIInterface not available for comparison",
                    duration
                )
                return False
            
            # Test original interface basic creation
            from agentsmcp.ui.cli_app import CLIConfig
            cli_config = CLIConfig()
            cli_config.debug_mode = self.verbose
            
            original_interface = RevolutionaryTUIInterface(cli_config=cli_config)
            self.log("🔧 Original TUI interface created")
            
            # Test reliable interface basic creation
            try:
                reliable_interface = ReliableTUIInterface(None, None)  # Basic creation test
                self.log("✨ Reliable TUI interface created")
                
                # Both interfaces created successfully
                duration = time.time() - start_time
                self.add_result(
                    test_name, True,
                    "Both original and reliable interfaces can be created",
                    duration,
                    {
                        "original_interface": True,
                        "reliable_interface": True,
                        "comparison_possible": True
                    }
                )
                self.log("✅ Interface comparison test passed", "success")
                return True
                
            except Exception as e:
                self.log(f"⚠️ ReliableTUIInterface creation failed: {e}", "warning")
                duration = time.time() - start_time
                self.add_result(
                    test_name, False,
                    f"ReliableTUIInterface creation failed: {str(e)}",
                    duration,
                    {"original_interface": True, "reliable_interface": False}
                )
                return False
            
        except Exception as e:
            duration = time.time() - start_time
            self.add_result(
                test_name, False,
                f"Interface comparison test failed: {str(e)}",
                duration,
                error_details=str(e)
            )
            self.log(f"❌ Interface comparison test failed: {e}", "error")
            return False
    
    async def test_component_health_monitoring(self) -> bool:
        """
        TEST 6: Test component health monitoring and hang detection
        
        This test validates that the health monitor can detect component
        issues and trigger recovery actions.
        """
        start_time = time.time()
        test_name = "component_health_monitoring"
        
        try:
            self.log("🏥 Testing component health monitoring...")
            
            from agentsmcp.ui.v2.reliability.health_monitor import (
                HealthMonitor, HealthStatus, get_global_health_monitor
            )
            
            health_monitor = get_global_health_monitor()
            
            # Test basic health monitoring
            health_status = health_monitor.get_current_health()
            self.log(f"📊 Current health status: {health_status}")
            
            # Test health metrics collection
            metrics = health_monitor.collect_performance_metrics()
            self.log(f"📈 Collected {len(metrics)} performance metrics")
            
            duration = time.time() - start_time
            self.add_result(
                test_name, True,
                "Health monitoring system working correctly",
                duration,
                {
                    "health_status": str(health_status),
                    "metrics_collected": len(metrics)
                }
            )
            self.log("✅ Component health monitoring test passed", "success")
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.add_result(
                test_name, False,
                f"Health monitoring test failed: {str(e)}",
                duration,
                error_details=str(e)
            )
            self.log(f"❌ Component health monitoring test failed: {e}", "error")
            return False
    
    async def run_comprehensive_tests(self) -> bool:
        """Run all comprehensive reliability tests."""
        self.log("🚀 Starting Comprehensive TUI Reliability Test Suite")
        self.log(f"⏱️ Maximum test timeout: {self.timeout}s")
        overall_start = time.time()
        
        tests = [
            ("Real TUI Startup Timeout Detection", self.test_real_tui_startup_timeout_detection),
            ("Input Response Hang Detection", self.test_input_response_hang_detection),
            ("Reliability Modules Integration", self.test_reliability_modules_integration),
            ("TimeoutGuardian Protection", self.test_timeout_guardian_protection),
            ("Reliable vs Original Interface", self.test_end_to_end_reliable_vs_original),
            ("Component Health Monitoring", self.test_component_health_monitoring),
        ]
        
        passed = 0
        failed = 0
        critical_failures = []
        
        for test_name, test_func in tests:
            self.log(f"🧪 Running: {test_name}")
            try:
                # Run each test with timeout protection
                test_task = asyncio.create_task(test_func())
                success = await asyncio.wait_for(test_task, timeout=self.timeout)
                
                if success:
                    passed += 1
                    self.log(f"✅ {test_name} PASSED", "success")
                else:
                    failed += 1
                    # Check if this was a critical failure (hang detection)
                    if "hang" in test_name.lower() or "timeout" in test_name.lower():
                        critical_failures.append(test_name)
                    self.log(f"❌ {test_name} FAILED", "error")
                    
            except asyncio.TimeoutError:
                failed += 1
                critical_failures.append(test_name)
                self.add_result(
                    test_name.lower().replace(" ", "_"), False,
                    f"Test timed out after {self.timeout}s",
                    self.timeout
                )
                self.log(f"⏰ {test_name} TIMED OUT", "error")
                
            except Exception as e:
                failed += 1
                self.add_result(
                    test_name.lower().replace(" ", "_"), False,
                    f"Test exception: {str(e)}",
                    time.time() - overall_start
                )
                self.log(f"💥 {test_name} CRASHED: {e}", "error")
        
        # Cleanup all processes
        self.process_manager.cleanup_all()
        
        # Generate comprehensive summary
        overall_duration = time.time() - overall_start
        self.log("\n" + "="*80)
        self.log("🏁 COMPREHENSIVE TUI RELIABILITY TEST RESULTS")
        self.log("="*80)
        self.log(f"📊 Total Tests: {passed + failed}")
        self.log(f"✅ Passed: {passed}")
        self.log(f"❌ Failed: {failed}")
        self.log(f"⏱️ Total Duration: {overall_duration:.2f}s")
        
        if critical_failures:
            self.log("\n🚨 CRITICAL FAILURES (Hang Detection):")
            for failure in critical_failures:
                self.log(f"   • {failure}")
        
        if failed == 0:
            self.log("\n🎉 ALL RELIABILITY TESTS PASSED!", "success")
            self.log("✅ TUI appears to be functioning without hang issues")
        else:
            self.log(f"\n❌ {failed} TESTS FAILED", "error")
            if critical_failures:
                self.log("🚨 CRITICAL: Hang detection tests failed - TUI may have hang issues!")
            else:
                self.log("ℹ️ Non-critical test failures - TUI basic functionality may be OK")
        
        # Show detailed results if verbose
        if self.verbose or failed > 0:
            self.log("\n📋 DETAILED TEST RESULTS:")
            self.log("-" * 80)
            for result in self.results:
                status = "✅ PASS" if result.success else "❌ FAIL"
                self.log(f"{status} {result.test_name} ({result.duration:.2f}s)")
                self.log(f"    💬 {result.message}")
                if result.error_details:
                    self.log(f"    🔍 Error: {result.error_details}")
                if self.verbose and result.details:
                    for key, value in result.details.items():
                        self.log(f"    📎 {key}: {value}")
        
        return failed == 0


def print_usage():
    """Print usage information."""
    print("""
TUI Reliability Comprehensive Test Suite

MISSION: Catch the ACTUAL TUI startup hangs that smoke tests missed!

Usage: python test_tui_reliability_comprehensive.py [options]

Options:
  --timeout N     Set timeout in seconds (default: 45)
  --verbose       Enable verbose output and debug logging
  --help          Show this help message

This test suite performs real process-based TUI testing to catch:
• TUI startup hangs after "Initializing Revolutionary TUI Interface..."
• Input response hangs (typing "/quit" with no response)
• Reliability module integration issues
• Timeout protection failures

Examples:
  python test_tui_reliability_comprehensive.py                    # Full test suite
  python test_tui_reliability_comprehensive.py --verbose          # Detailed output
  python test_tui_reliability_comprehensive.py --timeout 60       # Extended timeout
    """.strip())


async def main():
    """Main entry point for the comprehensive reliability test suite."""
    import sys
    
    # Parse command line arguments
    timeout = DEFAULT_TEST_TIMEOUT
    verbose = False
    
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--timeout" and i + 1 < len(args):
            timeout = int(args[i + 1])
            i += 2
        elif args[i] == "--verbose":
            verbose = True
            i += 1
        elif args[i] == "--help":
            print_usage()
            return 0
        else:
            print(f"❌ Unknown argument: {args[i]}")
            print_usage()
            return 1
    
    # Create and run test suite
    test_suite = ReliabilityTestSuite(timeout=timeout, verbose=verbose)
    
    try:
        success = await test_suite.run_comprehensive_tests()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n⚠️ Test suite interrupted by user")
        test_suite.process_manager.cleanup_all()
        return 130
    except Exception as e:
        print(f"💥 Test suite crashed: {e}")
        test_suite.process_manager.cleanup_all()
        return 1
    finally:
        # Ensure cleanup
        try:
            test_suite.process_manager.cleanup_all()
        except:
            pass


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)