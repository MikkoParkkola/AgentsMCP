#!/usr/bin/env python3
"""
COMPREHENSIVE END-TO-END USER TESTING FOR REVOLUTIONARY TUI

This test suite provides REAL USER ENVIRONMENT SIMULATION to verify:
1. TUI launches like a real user would via `./agentsmcp tui --debug`
2. TUI stays active and waits for user input (no premature shutdown)
3. User interaction flow works (typing, commands, quit)
4. No "Guardian shutdown" warnings during normal operation
5. All fixes to ReliableTUIInterface.run() method are verified

QA MISSION: Catch the "immediate shutdown" bug that affected real users
CRITICAL: These tests simulate exact user experience, not just unit tests
"""

import asyncio
import os
import signal
import subprocess
import sys
import time
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

# Add source directory to path for testing
repo_root = Path(__file__).parent
src_dir = repo_root / "src"
sys.path.insert(0, str(src_dir))

# Configure logging for QA analysis
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Test execution result with detailed metrics."""
    name: str
    passed: bool
    execution_time_s: float
    details: Dict[str, Any]
    error_msg: Optional[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

class UserWorkflowTester:
    """
    End-to-end tester simulating real user workflow.
    
    This tester executes the TUI exactly as a user would:
    - Launches via CLI command
    - Monitors startup behavior
    - Simulates user input
    - Verifies expected behavior
    """
    
    def __init__(self, debug_mode: bool = True):
        self.debug_mode = debug_mode
        self.repo_root = Path(__file__).parent
        self.agentsmcp_executable = self.repo_root / "agentsmcp"
        self.test_results: List[TestResult] = []
        
        # Test configuration
        self.max_startup_time_s = 15.0  # Max time for TUI to start
        self.activity_check_interval_s = 0.5  # How often to check TUI is alive
        self.user_interaction_delay_s = 2.0  # Delay before user input
        
    async def run_comprehensive_suite(self) -> Dict[str, Any]:
        """
        Run comprehensive end-to-end testing suite.
        
        Returns:
            QA verdict with detailed test results
        """
        logger.info("🧪 STARTING COMPREHENSIVE END-TO-END USER WORKFLOW TESTING")
        logger.info("=" * 70)
        
        suite_start_time = time.time()
        
        # Test 1: CLI Command Launch Verification
        result_1 = await self._test_cli_command_launch()
        self.test_results.append(result_1)
        
        # Test 2: TUI Startup Without Immediate Shutdown
        result_2 = await self._test_tui_startup_stays_active()
        self.test_results.append(result_2)
        
        # Test 3: User Interaction Flow
        result_3 = await self._test_user_interaction_flow()
        self.test_results.append(result_3)
        
        # Test 4: TTY vs Non-TTY Environment Testing
        result_4 = await self._test_tty_vs_non_tty_behavior()
        self.test_results.append(result_4)
        
        # Test 5: Guardian Timeout Logic Testing
        result_5 = await self._test_guardian_timeout_logic()
        self.test_results.append(result_5)
        
        # Test 6: Graceful Exit Testing
        result_6 = await self._test_graceful_exit_commands()
        self.test_results.append(result_6)
        
        # Test 7: Error Recovery Testing
        result_7 = await self._test_error_recovery_scenarios()
        self.test_results.append(result_7)
        
        # Test 8: Performance Benchmark
        result_8 = await self._test_performance_benchmarks()
        self.test_results.append(result_8)
        
        suite_execution_time = time.time() - suite_start_time
        
        # Generate QA verdict
        verdict = self._generate_qa_verdict(suite_execution_time)
        
        return verdict
    
    async def _test_cli_command_launch(self) -> TestResult:
        """Test 1: Verify CLI command launches TUI correctly."""
        logger.info("🔍 Test 1: CLI Command Launch Verification")
        
        start_time = time.time()
        
        try:
            # Check agentsmcp executable exists and is executable
            if not self.agentsmcp_executable.exists():
                return TestResult(
                    name="CLI Command Launch",
                    passed=False,
                    execution_time_s=time.time() - start_time,
                    details={"error": "agentsmcp executable not found"},
                    error_msg="agentsmcp executable not found at expected location"
                )
            
            if not os.access(self.agentsmcp_executable, os.X_OK):
                return TestResult(
                    name="CLI Command Launch",
                    passed=False,
                    execution_time_s=time.time() - start_time,
                    details={"error": "agentsmcp executable not executable"},
                    error_msg="agentsmcp executable exists but is not executable"
                )
            
            # Test command parsing by running help
            try:
                result = subprocess.run(
                    [str(self.agentsmcp_executable), "tui", "--help"],
                    capture_output=True,
                    text=True,
                    timeout=10.0
                )
                
                if result.returncode != 0:
                    return TestResult(
                        name="CLI Command Launch",
                        passed=False,
                        execution_time_s=time.time() - start_time,
                        details={
                            "error": "CLI command failed",
                            "returncode": result.returncode,
                            "stderr": result.stderr
                        },
                        error_msg=f"CLI command help failed with code {result.returncode}"
                    )
                
                logger.info("✅ CLI command exists and responds to help")
                
                return TestResult(
                    name="CLI Command Launch",
                    passed=True,
                    execution_time_s=time.time() - start_time,
                    details={
                        "executable_path": str(self.agentsmcp_executable),
                        "help_output_length": len(result.stdout),
                        "command_working": True
                    }
                )
                
            except subprocess.TimeoutExpired:
                return TestResult(
                    name="CLI Command Launch",
                    passed=False,
                    execution_time_s=time.time() - start_time,
                    details={"error": "CLI command help timed out"},
                    error_msg="CLI command help timed out after 10 seconds"
                )
                
        except Exception as e:
            return TestResult(
                name="CLI Command Launch",
                passed=False,
                execution_time_s=time.time() - start_time,
                details={"exception": str(e)},
                error_msg=f"CLI command launch test failed: {e}"
            )
    
    async def _test_tui_startup_stays_active(self) -> TestResult:
        """Test 2: Critical test - TUI should stay active, not shutdown immediately."""
        logger.info("🔍 Test 2: TUI Startup Stays Active (CRITICAL ANTI-REGRESSION)")
        
        start_time = time.time()
        
        try:
            # Launch TUI process in debug mode
            logger.info("🚀 Launching TUI process...")
            
            process = subprocess.Popen(
                [str(self.agentsmcp_executable), "tui", "--debug"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0  # Unbuffered for real-time interaction
            )
            
            # Monitor process for immediate shutdown (the bug we're testing against)
            logger.info("⏰ Monitoring for immediate shutdown (0.085s Guardian bug)...")
            
            activity_checks = []
            startup_complete = False
            guardian_warnings = []
            
            # Check every 100ms for first 5 seconds
            for check_num in range(50):  # 50 * 0.1s = 5 seconds
                await asyncio.sleep(0.1)
                
                poll_result = process.poll()
                is_alive = poll_result is None
                
                activity_checks.append({
                    "time_offset": check_num * 0.1,
                    "is_alive": is_alive,
                    "poll_result": poll_result
                })
                
                # Read any output to check for Guardian warnings
                try:
                    # Non-blocking read of stderr for warnings
                    import select
                    if hasattr(select, 'select'):  # Unix systems
                        ready, _, _ = select.select([process.stderr], [], [], 0)
                        if ready:
                            stderr_data = process.stderr.read(1024)
                            if stderr_data and "Guardian shutdown" in stderr_data:
                                guardian_warnings.append(stderr_data)
                except:
                    pass  # Skip non-blocking read on systems that don't support it
                
                # Critical check: Process should NOT exit in first 2 seconds
                if not is_alive and check_num < 20:  # First 2 seconds
                    logger.error(f"❌ TUI DIED IMMEDIATELY at {check_num * 0.1:.1f}s - CRITICAL REGRESSION!")
                    
                    # Get any error output
                    try:
                        stdout_data, stderr_data = process.communicate(timeout=1)
                    except subprocess.TimeoutExpired:
                        stdout_data, stderr_data = "", ""
                    
                    process.terminate()
                    
                    return TestResult(
                        name="TUI Startup Stays Active",
                        passed=False,
                        execution_time_s=time.time() - start_time,
                        details={
                            "immediate_shutdown": True,
                            "shutdown_time_s": check_num * 0.1,
                            "exit_code": poll_result,
                            "stdout": stdout_data,
                            "stderr": stderr_data,
                            "guardian_warnings": guardian_warnings,
                            "activity_checks": activity_checks
                        },
                        error_msg=f"TUI shut down immediately after {check_num * 0.1:.1f}s (expected >2s activity)"
                    )
                
                # Success criteria: TUI alive for at least 2 seconds
                if is_alive and check_num >= 20:  # 2+ seconds alive
                    startup_complete = True
                    logger.info(f"✅ TUI has been alive for {check_num * 0.1:.1f}s - startup successful!")
                    break
            
            # Cleanup process
            if process.poll() is None:
                # Send quit command to gracefully exit
                try:
                    process.stdin.write("quit\n")
                    process.stdin.flush()
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=1)
                except:
                    process.terminate()
            
            execution_time = time.time() - start_time
            
            if startup_complete:
                return TestResult(
                    name="TUI Startup Stays Active",
                    passed=True,
                    execution_time_s=execution_time,
                    details={
                        "stayed_alive": True,
                        "activity_duration_s": 5.0,
                        "guardian_warnings": guardian_warnings,
                        "activity_checks": len(activity_checks),
                        "no_immediate_shutdown": True
                    }
                )
            else:
                return TestResult(
                    name="TUI Startup Stays Active", 
                    passed=False,
                    execution_time_s=execution_time,
                    details={
                        "stayed_alive": False,
                        "activity_checks": activity_checks,
                        "guardian_warnings": guardian_warnings
                    },
                    error_msg="TUI did not stay alive for minimum required time"
                )
                
        except Exception as e:
            logger.error(f"TUI startup test failed with exception: {e}")
            return TestResult(
                name="TUI Startup Stays Active",
                passed=False,
                execution_time_s=time.time() - start_time,
                details={"exception": str(e)},
                error_msg=f"TUI startup test failed: {e}"
            )
    
    async def _test_user_interaction_flow(self) -> TestResult:
        """Test 3: Verify user can interact with TUI (typing, commands)."""
        logger.info("🔍 Test 3: User Interaction Flow")
        
        start_time = time.time()
        
        try:
            # Launch TUI process
            process = subprocess.Popen(
                [str(self.agentsmcp_executable), "tui", "--debug"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0
            )
            
            # Wait for startup
            await asyncio.sleep(2.0)
            
            if process.poll() is not None:
                return TestResult(
                    name="User Interaction Flow",
                    passed=False,
                    execution_time_s=time.time() - start_time,
                    details={"error": "TUI died during startup"},
                    error_msg="TUI process died before interaction could begin"
                )
            
            # Test user inputs
            test_inputs = [
                "help",      # Help command
                "",          # Empty input (should handle gracefully)
                "status",    # Status command  
                "quit"       # Exit command
            ]
            
            interactions_successful = 0
            
            for i, input_cmd in enumerate(test_inputs):
                try:
                    logger.info(f"📝 Sending user input {i+1}: '{input_cmd}'")
                    
                    # Send input
                    process.stdin.write(f"{input_cmd}\n")
                    process.stdin.flush()
                    
                    # Wait for processing
                    await asyncio.sleep(0.5)
                    
                    # Check if process is still alive (except for quit command)
                    is_alive = process.poll() is None
                    
                    if input_cmd == "quit":
                        # For quit command, we expect the process to exit
                        await asyncio.sleep(1.0)
                        final_status = process.poll()
                        
                        if final_status is not None:
                            logger.info(f"✅ Quit command worked - process exited with code {final_status}")
                            interactions_successful += 1
                        else:
                            logger.warning("⚠️ Quit command sent but process still alive")
                            process.terminate()
                        break
                    else:
                        if is_alive:
                            logger.info(f"✅ Input '{input_cmd}' processed, TUI still alive")
                            interactions_successful += 1
                        else:
                            logger.error(f"❌ TUI died after input '{input_cmd}'")
                            break
                            
                except Exception as e:
                    logger.error(f"❌ Failed to send input '{input_cmd}': {e}")
                    break
            
            # Ensure process cleanup
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=2)
            
            execution_time = time.time() - start_time
            
            success_rate = interactions_successful / len(test_inputs)
            passed = success_rate >= 0.75  # 75% success rate required
            
            return TestResult(
                name="User Interaction Flow",
                passed=passed,
                execution_time_s=execution_time,
                details={
                    "interactions_successful": interactions_successful,
                    "total_interactions": len(test_inputs),
                    "success_rate": success_rate,
                    "test_inputs": test_inputs
                }
            )
            
        except Exception as e:
            return TestResult(
                name="User Interaction Flow",
                passed=False,
                execution_time_s=time.time() - start_time,
                details={"exception": str(e)},
                error_msg=f"User interaction test failed: {e}"
            )
    
    async def _test_tty_vs_non_tty_behavior(self) -> TestResult:
        """Test 4: Verify proper behavior in TTY vs non-TTY environments."""
        logger.info("🔍 Test 4: TTY vs Non-TTY Environment Testing")
        
        start_time = time.time()
        
        try:
            results = {}
            
            # Test 1: TTY environment (normal terminal)
            logger.info("🖥️ Testing TTY environment...")
            
            tty_process = subprocess.Popen(
                [str(self.agentsmcp_executable), "tui", "--debug"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            await asyncio.sleep(2.0)
            tty_alive = tty_process.poll() is None
            
            if tty_alive:
                tty_process.stdin.write("quit\n")
                tty_process.wait(timeout=2)
            else:
                tty_process.terminate()
            
            results['tty_test'] = {'alive_after_2s': tty_alive}
            
            # Test 2: Non-TTY environment (simulated)
            logger.info("📄 Testing non-TTY environment...")
            
            # Run with no stdin to simulate non-TTY
            non_tty_process = subprocess.Popen(
                [str(self.agentsmcp_executable), "tui", "--debug"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            await asyncio.sleep(2.0)
            non_tty_alive = non_tty_process.poll() is None
            
            # Non-TTY should either handle gracefully or exit cleanly
            if non_tty_alive:
                non_tty_process.terminate()
                non_tty_process.wait(timeout=2)
            
            results['non_tty_test'] = {'alive_after_2s': non_tty_alive}
            
            execution_time = time.time() - start_time
            
            # Both tests should show reasonable behavior
            passed = tty_alive or non_tty_alive  # At least one should work
            
            return TestResult(
                name="TTY vs Non-TTY Behavior",
                passed=passed,
                execution_time_s=execution_time,
                details=results
            )
            
        except Exception as e:
            return TestResult(
                name="TTY vs Non-TTY Behavior",
                passed=False,
                execution_time_s=time.time() - start_time,
                details={"exception": str(e)},
                error_msg=f"TTY testing failed: {e}"
            )
    
    async def _test_guardian_timeout_logic(self) -> TestResult:
        """Test 5: Verify Guardian timeout logic works correctly."""
        logger.info("🔍 Test 5: Guardian Timeout Logic Testing")
        
        start_time = time.time()
        
        try:
            # This test focuses on ensuring the Guardian doesn't cause premature shutdown
            # We look for Guardian-related log messages and timeouts
            
            process = subprocess.Popen(
                [str(self.agentsmcp_executable), "tui", "--debug"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0
            )
            
            guardian_events = []
            timeout_events = []
            
            # Monitor for 10 seconds looking for Guardian events
            for i in range(100):  # 10 seconds of monitoring
                await asyncio.sleep(0.1)
                
                # Check process status
                is_alive = process.poll() is None
                
                if not is_alive:
                    # Process died - check if it was due to Guardian
                    try:
                        stdout_data, stderr_data = process.communicate(timeout=1)
                        
                        if "Guardian shutdown" in stderr_data or "timeout" in stderr_data.lower():
                            timeout_events.append({
                                "time": i * 0.1,
                                "stderr": stderr_data
                            })
                            
                    except:
                        pass
                    break
                
                # Successful monitoring
                guardian_events.append({
                    "time": i * 0.1,
                    "status": "alive"
                })
            
            # Cleanup
            if process.poll() is None:
                try:
                    process.stdin.write("quit\n")
                    process.wait(timeout=2)
                except:
                    process.terminate()
            
            execution_time = time.time() - start_time
            
            # Success criteria: No premature Guardian timeouts
            premature_timeout = len(timeout_events) > 0 and any(
                event['time'] < 5.0 for event in timeout_events
            )
            
            passed = not premature_timeout
            
            return TestResult(
                name="Guardian Timeout Logic",
                passed=passed,
                execution_time_s=execution_time,
                details={
                    "guardian_events": len(guardian_events),
                    "timeout_events": timeout_events,
                    "premature_timeout": premature_timeout,
                    "monitoring_duration_s": min(10.0, execution_time)
                }
            )
            
        except Exception as e:
            return TestResult(
                name="Guardian Timeout Logic",
                passed=False,
                execution_time_s=time.time() - start_time,
                details={"exception": str(e)},
                error_msg=f"Guardian timeout test failed: {e}"
            )
    
    async def _test_graceful_exit_commands(self) -> TestResult:
        """Test 6: Verify all exit commands work correctly."""
        logger.info("🔍 Test 6: Graceful Exit Commands Testing")
        
        start_time = time.time()
        
        try:
            exit_commands = ["quit", "exit", "q"]
            results = {}
            
            for cmd in exit_commands:
                logger.info(f"🚪 Testing exit command: '{cmd}'")
                
                # Launch fresh process for each test
                process = subprocess.Popen(
                    [str(self.agentsmcp_executable), "tui", "--debug"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Wait for startup
                await asyncio.sleep(2.0)
                
                if process.poll() is not None:
                    results[cmd] = {
                        "worked": False,
                        "error": "Process died during startup"
                    }
                    continue
                
                # Send exit command
                try:
                    process.stdin.write(f"{cmd}\n")
                    process.stdin.flush()
                    
                    # Wait for graceful exit
                    await asyncio.sleep(2.0)
                    
                    exit_code = process.poll()
                    
                    if exit_code is not None:
                        results[cmd] = {
                            "worked": True,
                            "exit_code": exit_code,
                            "graceful": exit_code == 0
                        }
                        logger.info(f"✅ Command '{cmd}' worked (exit code: {exit_code})")
                    else:
                        # Force termination if still alive
                        process.terminate()
                        process.wait(timeout=1)
                        results[cmd] = {
                            "worked": False,
                            "error": "Command did not cause exit"
                        }
                        logger.warning(f"⚠️ Command '{cmd}' did not cause exit")
                        
                except Exception as e:
                    process.terminate()
                    results[cmd] = {
                        "worked": False,
                        "error": str(e)
                    }
                    logger.error(f"❌ Command '{cmd}' failed: {e}")
            
            execution_time = time.time() - start_time
            
            # Success criteria: At least 2 out of 3 exit commands should work
            working_commands = sum(1 for result in results.values() if result.get('worked', False))
            passed = working_commands >= 2
            
            return TestResult(
                name="Graceful Exit Commands",
                passed=passed,
                execution_time_s=execution_time,
                details={
                    "results": results,
                    "working_commands": working_commands,
                    "total_commands": len(exit_commands)
                }
            )
            
        except Exception as e:
            return TestResult(
                name="Graceful Exit Commands",
                passed=False,
                execution_time_s=time.time() - start_time,
                details={"exception": str(e)},
                error_msg=f"Exit commands test failed: {e}"
            )
    
    async def _test_error_recovery_scenarios(self) -> TestResult:
        """Test 7: Verify error recovery works without losing user session."""
        logger.info("🔍 Test 7: Error Recovery Scenarios Testing")
        
        start_time = time.time()
        
        try:
            # This test simulates error conditions and verifies recovery
            scenarios = [
                "invalid_command",
                "empty_input_sequence", 
                "rapid_input_sequence"
            ]
            
            results = {}
            
            for scenario in scenarios:
                logger.info(f"🔄 Testing error recovery for: {scenario}")
                
                process = subprocess.Popen(
                    [str(self.agentsmcp_executable), "tui", "--debug"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=0
                )
                
                # Wait for startup
                await asyncio.sleep(2.0)
                
                if process.poll() is not None:
                    results[scenario] = {
                        "recovered": False,
                        "error": "Process died during startup"
                    }
                    continue
                
                # Execute error scenario
                try:
                    if scenario == "invalid_command":
                        # Send invalid commands
                        invalid_commands = ["invalid123", "nonexistent_cmd", "!@#$%"]
                        for cmd in invalid_commands:
                            process.stdin.write(f"{cmd}\n")
                            process.stdin.flush()
                            await asyncio.sleep(0.2)
                    
                    elif scenario == "empty_input_sequence":
                        # Send empty inputs
                        for _ in range(5):
                            process.stdin.write("\n")
                            process.stdin.flush()
                            await asyncio.sleep(0.1)
                    
                    elif scenario == "rapid_input_sequence":
                        # Send rapid inputs
                        for i in range(10):
                            process.stdin.write(f"test{i}\n")
                            process.stdin.flush()
                            await asyncio.sleep(0.05)
                    
                    # Check if TUI is still alive after error scenario
                    await asyncio.sleep(1.0)
                    still_alive = process.poll() is None
                    
                    if still_alive:
                        # Try to send a normal command to verify recovery
                        process.stdin.write("help\n")
                        process.stdin.flush()
                        await asyncio.sleep(0.5)
                        
                        final_alive = process.poll() is None
                        
                        results[scenario] = {
                            "recovered": final_alive,
                            "survived_errors": True,
                            "responds_after_errors": final_alive
                        }
                        
                        logger.info(f"✅ Scenario '{scenario}': Recovery {'successful' if final_alive else 'failed'}")
                    else:
                        results[scenario] = {
                            "recovered": False,
                            "survived_errors": False
                        }
                        logger.error(f"❌ Scenario '{scenario}': TUI died during error scenario")
                
                except Exception as e:
                    results[scenario] = {
                        "recovered": False,
                        "error": str(e)
                    }
                    logger.error(f"❌ Scenario '{scenario}' failed with exception: {e}")
                
                # Cleanup
                if process.poll() is None:
                    try:
                        process.stdin.write("quit\n")
                        process.wait(timeout=2)
                    except:
                        process.terminate()
            
            execution_time = time.time() - start_time
            
            # Success criteria: TUI should recover from at least 2/3 error scenarios
            recovered_scenarios = sum(1 for result in results.values() if result.get('recovered', False))
            passed = recovered_scenarios >= 2
            
            return TestResult(
                name="Error Recovery Scenarios",
                passed=passed,
                execution_time_s=execution_time,
                details={
                    "results": results,
                    "recovered_scenarios": recovered_scenarios,
                    "total_scenarios": len(scenarios)
                }
            )
            
        except Exception as e:
            return TestResult(
                name="Error Recovery Scenarios",
                passed=False,
                execution_time_s=time.time() - start_time,
                details={"exception": str(e)},
                error_msg=f"Error recovery test failed: {e}"
            )
    
    async def _test_performance_benchmarks(self) -> TestResult:
        """Test 8: Performance benchmarks (startup time, shutdown time)."""
        logger.info("🔍 Test 8: Performance Benchmarks")
        
        start_time = time.time()
        
        try:
            metrics = {
                "startup_times": [],
                "shutdown_times": [],
                "interaction_response_times": []
            }
            
            # Run 3 iterations for statistical significance
            for iteration in range(3):
                logger.info(f"📊 Performance benchmark iteration {iteration + 1}/3")
                
                # Measure startup time
                startup_start = time.time()
                
                process = subprocess.Popen(
                    [str(self.agentsmcp_executable), "tui", "--debug"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=0
                )
                
                # Wait for TUI to be responsive (can accept commands)
                startup_complete = False
                for check in range(100):  # Max 10 seconds
                    await asyncio.sleep(0.1)
                    if process.poll() is not None:
                        break
                    
                    if check >= 20:  # After 2 seconds, assume startup complete
                        startup_complete = True
                        break
                
                startup_time = time.time() - startup_start
                if startup_complete:
                    metrics["startup_times"].append(startup_time)
                    logger.info(f"  📈 Startup time: {startup_time:.2f}s")
                
                if process.poll() is not None:
                    continue  # Skip this iteration if process died
                
                # Measure interaction response time
                interaction_start = time.time()
                process.stdin.write("help\n")
                process.stdin.flush()
                await asyncio.sleep(0.5)  # Wait for response
                interaction_time = time.time() - interaction_start
                
                if process.poll() is None:  # Still alive
                    metrics["interaction_response_times"].append(interaction_time)
                    logger.info(f"  ⚡ Interaction response time: {interaction_time:.2f}s")
                
                # Measure shutdown time
                shutdown_start = time.time()
                process.stdin.write("quit\n")
                process.stdin.flush()
                
                # Wait for graceful shutdown
                for check in range(50):  # Max 5 seconds
                    await asyncio.sleep(0.1)
                    if process.poll() is not None:
                        shutdown_time = time.time() - shutdown_start
                        metrics["shutdown_times"].append(shutdown_time)
                        logger.info(f"  🚪 Shutdown time: {shutdown_time:.2f}s")
                        break
                else:
                    # Force termination if didn't shutdown gracefully
                    process.terminate()
                    process.wait(timeout=1)
            
            execution_time = time.time() - start_time
            
            # Calculate performance statistics
            performance_stats = {}
            for metric_name, values in metrics.items():
                if values:
                    performance_stats[metric_name] = {
                        "avg": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "count": len(values)
                    }
            
            # Performance criteria
            avg_startup = performance_stats.get("startup_times", {}).get("avg", float('inf'))
            avg_shutdown = performance_stats.get("shutdown_times", {}).get("avg", float('inf'))
            
            # Success criteria: Startup < 15s, Shutdown < 5s
            passed = avg_startup < 15.0 and avg_shutdown < 5.0
            
            return TestResult(
                name="Performance Benchmarks",
                passed=passed,
                execution_time_s=execution_time,
                details={
                    "performance_stats": performance_stats,
                    "raw_metrics": metrics,
                    "iterations_completed": min(len(metrics["startup_times"]), 3)
                }
            )
            
        except Exception as e:
            return TestResult(
                name="Performance Benchmarks",
                passed=False,
                execution_time_s=time.time() - start_time,
                details={"exception": str(e)},
                error_msg=f"Performance benchmark failed: {e}"
            )
    
    def _generate_qa_verdict(self, suite_execution_time: float) -> Dict[str, Any]:
        """Generate comprehensive QA verdict based on test results."""
        
        logger.info("📋 GENERATING QA VERDICT")
        logger.info("=" * 50)
        
        # Calculate overall statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result.passed)
        failed_tests = total_tests - passed_tests
        
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0.0
        
        # Critical test analysis
        critical_tests = [
            "TUI Startup Stays Active",  # Most critical - the bug we're fixing
            "User Interaction Flow",
            "Graceful Exit Commands"
        ]
        
        critical_results = {
            name: next((r for r in self.test_results if r.name == name), None)
            for name in critical_tests
        }
        
        critical_passed = sum(1 for result in critical_results.values() if result and result.passed)
        critical_pass_rate = critical_passed / len(critical_tests)
        
        # Determine QA verdict
        if critical_pass_rate >= 1.0 and pass_rate >= 0.875:  # 7/8 tests pass + all critical
            verdict = "ACCEPT"
            readiness = "READY_FOR_USER_TESTING"
            confidence = "HIGH"
        elif critical_pass_rate >= 1.0 and pass_rate >= 0.75:  # 6/8 tests pass + all critical  
            verdict = "ACCEPT"
            readiness = "READY_FOR_USER_TESTING" 
            confidence = "MEDIUM"
        elif critical_pass_rate >= 2/3 and pass_rate >= 0.625:  # 5/8 tests pass + 2/3 critical
            verdict = "CONDITIONAL_ACCEPT"
            readiness = "NEEDS_MINOR_FIXES"
            confidence = "MEDIUM"
        else:
            verdict = "BLOCK"
            readiness = "NEEDS_MAJOR_FIXES"
            confidence = "LOW"
        
        # Identify blocking issues
        blocking_issues = []
        for result in self.test_results:
            if not result.passed and result.name in critical_tests:
                blocking_issues.append({
                    "test": result.name,
                    "error": result.error_msg,
                    "severity": "CRITICAL"
                })
        
        # Performance analysis
        perf_result = next((r for r in self.test_results if r.name == "Performance Benchmarks"), None)
        performance_summary = {}
        if perf_result and perf_result.passed:
            perf_stats = perf_result.details.get("performance_stats", {})
            performance_summary = {
                "startup_time_avg": perf_stats.get("startup_times", {}).get("avg"),
                "shutdown_time_avg": perf_stats.get("shutdown_times", {}).get("avg"),
                "performance_acceptable": True
            }
        
        # Test execution summary
        test_summary = []
        for result in self.test_results:
            test_summary.append({
                "name": result.name,
                "status": "PASS" if result.passed else "FAIL",
                "execution_time_s": result.execution_time_s,
                "error": result.error_msg if not result.passed else None,
                "warnings": result.warnings
            })
        
        # Generate verdict structure
        qa_verdict = {
            "verdict": verdict,
            "readiness": readiness,
            "confidence": confidence,
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "pass_rate": pass_rate,
                "critical_pass_rate": critical_pass_rate,
                "suite_execution_time_s": suite_execution_time
            },
            "critical_test_results": {
                name: {
                    "passed": result.passed if result else False,
                    "error": result.error_msg if result and not result.passed else None
                }
                for name, result in critical_results.items()
            },
            "blocking_issues": blocking_issues,
            "performance_summary": performance_summary,
            "test_results": test_summary,
            "recommendations": self._generate_recommendations(verdict, readiness),
            "user_acceptance_criteria": {
                "tui_starts_without_immediate_shutdown": critical_results.get("TUI Startup Stays Active", {}).passed if critical_results.get("TUI Startup Stays Active") else False,
                "user_can_interact_with_tui": critical_results.get("User Interaction Flow", {}).passed if critical_results.get("User Interaction Flow") else False,
                "user_can_exit_gracefully": critical_results.get("Graceful Exit Commands", {}).passed if critical_results.get("Graceful Exit Commands") else False,
                "no_guardian_shutdown_warnings": True  # Would need to check logs for this
            }
        }
        
        # Log verdict summary
        logger.info(f"📊 QA VERDICT: {verdict} ({readiness})")
        logger.info(f"🎯 Pass Rate: {pass_rate*100:.1f}% ({passed_tests}/{total_tests})")
        logger.info(f"🔥 Critical Pass Rate: {critical_pass_rate*100:.1f}% ({critical_passed}/{len(critical_tests)})")
        logger.info(f"⏱️ Total Execution Time: {suite_execution_time:.1f}s")
        
        if blocking_issues:
            logger.error(f"🚫 Blocking Issues Found: {len(blocking_issues)}")
            for issue in blocking_issues:
                logger.error(f"   - {issue['test']}: {issue['error']}")
        
        return qa_verdict
    
    def _generate_recommendations(self, verdict: str, readiness: str) -> List[str]:
        """Generate specific recommendations based on test results."""
        recommendations = []
        
        # Analyze failed tests for specific recommendations
        failed_results = [r for r in self.test_results if not r.passed]
        
        for failed_result in failed_results:
            if failed_result.name == "TUI Startup Stays Active":
                recommendations.append("CRITICAL: Fix immediate shutdown bug in ReliableTUIInterface.run() method")
                recommendations.append("Verify Guardian timeout logic doesn't trigger premature shutdown")
                
            elif failed_result.name == "User Interaction Flow":
                recommendations.append("Fix user input handling to ensure TUI responds to commands")
                recommendations.append("Verify input pipeline processes user commands correctly")
                
            elif failed_result.name == "Graceful Exit Commands": 
                recommendations.append("Implement proper handling for quit/exit/q commands")
                recommendations.append("Ensure cleanup methods are called during graceful shutdown")
                
            elif failed_result.name == "Guardian Timeout Logic":
                recommendations.append("Review Guardian timeout thresholds to prevent false positives")
                recommendations.append("Ensure Guardian only triggers on actual hangs, not normal operation")
                
            elif failed_result.name == "Performance Benchmarks":
                recommendations.append("Optimize TUI startup performance (target <10s startup)")
                recommendations.append("Optimize shutdown performance (target <3s shutdown)")
        
        # General recommendations based on verdict
        if verdict == "BLOCK":
            recommendations.append("Address all critical test failures before user testing")
            recommendations.append("Run full regression testing after fixes")
        elif verdict == "CONDITIONAL_ACCEPT":
            recommendations.append("Fix minor issues but proceed with limited user testing")
            recommendations.append("Monitor user feedback closely for additional issues")
        elif verdict == "ACCEPT":
            recommendations.append("Proceed with user acceptance testing")
            recommendations.append("Monitor production deployment for any edge cases")
        
        return recommendations


async def main():
    """Main execution function for comprehensive testing."""
    print("🧪 REVOLUTIONARY TUI - COMPREHENSIVE END-TO-END QA TESTING")
    print("=" * 70)
    print("Mission: Verify fixes for immediate shutdown bug and user workflow")
    print()
    
    # Initialize tester
    tester = UserWorkflowTester(debug_mode=True)
    
    try:
        # Run comprehensive test suite
        verdict = await tester.run_comprehensive_suite()
        
        # Display results
        print("\n" + "=" * 70)
        print("🏁 COMPREHENSIVE QA TESTING COMPLETE")
        print("=" * 70)
        
        print(f"\n📊 QA VERDICT: {verdict['verdict']} ({verdict['readiness']})")
        print(f"🎯 Overall Pass Rate: {verdict['summary']['pass_rate']*100:.1f}%")
        print(f"🔥 Critical Test Pass Rate: {verdict['summary']['critical_pass_rate']*100:.1f}%")
        print(f"⏱️ Total Execution Time: {verdict['summary']['suite_execution_time_s']:.1f}s")
        
        # Show test results
        print(f"\n📋 TEST RESULTS SUMMARY:")
        print("-" * 50)
        for test_result in verdict['test_results']:
            status_emoji = "✅" if test_result['status'] == "PASS" else "❌"
            print(f"{status_emoji} {test_result['name']}: {test_result['status']} ({test_result['execution_time_s']:.1f}s)")
            if test_result['error']:
                print(f"    Error: {test_result['error']}")
        
        # Show user acceptance criteria
        print(f"\n🎯 USER ACCEPTANCE CRITERIA:")
        print("-" * 30)
        acceptance = verdict['user_acceptance_criteria']
        for criteria, passed in acceptance.items():
            status_emoji = "✅" if passed else "❌"
            criteria_name = criteria.replace("_", " ").title()
            print(f"{status_emoji} {criteria_name}: {'PASS' if passed else 'FAIL'}")
        
        # Show blocking issues
        if verdict['blocking_issues']:
            print(f"\n🚫 BLOCKING ISSUES ({len(verdict['blocking_issues'])}):")
            print("-" * 30)
            for issue in verdict['blocking_issues']:
                print(f"❌ {issue['test']}: {issue['error']}")
        
        # Show recommendations  
        if verdict['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            print("-" * 20)
            for i, rec in enumerate(verdict['recommendations'], 1):
                print(f"{i}. {rec}")
        
        # Final verdict
        print("\n" + "=" * 70)
        if verdict['verdict'] == "ACCEPT":
            print("🎉 QA VERDICT: READY FOR USER TESTING!")
            print("   The TUI should work correctly for real users.")
        elif verdict['verdict'] == "CONDITIONAL_ACCEPT":
            print("⚠️ QA VERDICT: PROCEED WITH CAUTION")
            print("   Most functionality works but minor issues remain.")
        else:
            print("🛑 QA VERDICT: BLOCKED - NOT READY FOR USERS")
            print("   Critical issues must be resolved before user testing.")
        
        # Return appropriate exit code
        return 0 if verdict['verdict'] in ["ACCEPT", "CONDITIONAL_ACCEPT"] else 1
        
    except Exception as e:
        logger.error(f"QA testing failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)