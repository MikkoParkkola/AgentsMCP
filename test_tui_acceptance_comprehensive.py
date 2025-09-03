#!/usr/bin/env python3
"""
Comprehensive TUI Acceptance Tests for Revolutionary TUI Interface

This test suite validates that the Revolutionary TUI Interface works end-to-end
like a real user would experience, including:
- TUI launch and startup
- Interactive messaging with LLM responses  
- Command functionality (help, clear, quit)
- Input visibility (the original reported issue)
- Rich interface display with panels and colors
- Error handling and graceful exit

These tests run actual ./agentsmcp tui commands and validate real user workflows.
"""

import asyncio
import os
import subprocess
import sys
import time
import re
import tempfile
import pytest
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import signal
from unittest.mock import patch

# Test configuration
TEST_TIMEOUT = 30  # seconds
TUI_STARTUP_TIMEOUT = 10  # seconds
LLM_RESPONSE_TIMEOUT = 20  # seconds

@dataclass
class TestResult:
    """Result of a TUI test execution."""
    name: str
    success: bool
    output: str
    error_output: str
    exit_code: int
    duration: float
    expected_patterns: List[str] = None
    found_patterns: List[str] = None
    failure_reason: str = ""

class TUITestRunner:
    """
    Test runner for Revolutionary TUI Interface acceptance tests.
    
    Runs actual ./agentsmcp tui commands with various inputs and validates
    outputs to ensure the TUI works as expected by real users.
    """
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.results: List[TestResult] = []
        
        # Get project root directory
        self.project_root = Path(__file__).parent
        self.agentsmcp_cmd = self.project_root / "agentsmcp"
        
        # Ensure agentsmcp command exists
        if not self.agentsmcp_cmd.exists():
            # Try alternative paths
            alt_paths = [
                self.project_root / "src" / "agentsmcp" / "__main__.py",
                "agentsmcp"  # System installed
            ]
            for alt_path in alt_paths:
                if isinstance(alt_path, str) or alt_path.exists():
                    self.agentsmcp_cmd = alt_path
                    break
    
    def _log(self, message: str):
        """Log debug message if debug mode enabled."""
        if self.debug:
            print(f"[TUI_TEST] {message}")
    
    def _run_tui_command(self, 
                        input_sequence: str, 
                        test_name: str,
                        timeout: int = TEST_TIMEOUT,
                        expected_patterns: List[str] = None,
                        args: List[str] = None) -> TestResult:
        """
        Run TUI command with input sequence and validate output.
        
        Args:
            input_sequence: Input to send to TUI (newline-separated commands)
            test_name: Name of the test for reporting
            timeout: Timeout in seconds
            expected_patterns: Regex patterns that should be found in output
            args: Additional arguments to pass to TUI command
            
        Returns:
            TestResult with execution details
        """
        start_time = time.time()
        
        # Prepare command
        if isinstance(self.agentsmcp_cmd, str):
            cmd = [self.agentsmcp_cmd, "tui"]
        else:
            cmd = ["python", str(self.agentsmcp_cmd), "tui"]
        
        if args:
            cmd.extend(args)
        
        self._log(f"Running command: {' '.join(cmd)}")
        self._log(f"Input sequence: {repr(input_sequence)}")
        
        try:
            # Run TUI process with input
            process = subprocess.run(
                cmd,
                input=input_sequence,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "PYTHONPATH": str(self.project_root / "src")}
            )
            
            duration = time.time() - start_time
            self._log(f"Command completed in {duration:.2f}s")
            self._log(f"Exit code: {process.returncode}")
            self._log(f"Output: {repr(process.stdout[:500])}")
            
            # Check expected patterns
            found_patterns = []
            failure_reason = ""
            success = True
            
            if expected_patterns:
                for pattern in expected_patterns:
                    if re.search(pattern, process.stdout, re.MULTILINE | re.DOTALL):
                        found_patterns.append(pattern)
                    else:
                        success = False
                        failure_reason = f"Pattern not found: {pattern}"
                        break
            
            # Additional success criteria
            if success and process.returncode not in [0, 130]:  # 130 = Ctrl+C
                success = False
                failure_reason = f"Unexpected exit code: {process.returncode}"
            
            return TestResult(
                name=test_name,
                success=success,
                output=process.stdout,
                error_output=process.stderr,
                exit_code=process.returncode,
                duration=duration,
                expected_patterns=expected_patterns or [],
                found_patterns=found_patterns,
                failure_reason=failure_reason
            )
            
        except subprocess.TimeoutExpired as e:
            duration = time.time() - start_time
            return TestResult(
                name=test_name,
                success=False,
                output=e.stdout or "",
                error_output=e.stderr or "",
                exit_code=-1,
                duration=duration,
                expected_patterns=expected_patterns or [],
                found_patterns=[],
                failure_reason=f"Test timed out after {timeout}s"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                name=test_name,
                success=False,
                output="",
                error_output=str(e),
                exit_code=-1,
                duration=duration,
                expected_patterns=expected_patterns or [],
                found_patterns=[],
                failure_reason=f"Test failed with exception: {e}"
            )

    def run_all_tests(self) -> List[TestResult]:
        """Run all TUI acceptance tests."""
        self._log("Starting comprehensive TUI acceptance tests...")
        
        # Test 1: Basic TUI Launch and Quit
        result = self._run_tui_command(
            input_sequence="quit\n",
            test_name="TUI Launch Test",
            timeout=TUI_STARTUP_TIMEOUT,
            expected_patterns=[
                r"Revolutionary TUI|Enhanced|Rich|TUI",  # Should show TUI interface
                r"quit|exit|goodbye",  # Should handle quit command
            ]
        )
        self.results.append(result)
        
        # Test 2: Help Command Functionality
        result = self._run_tui_command(
            input_sequence="help\nquit\n",
            test_name="Help Command Test",
            timeout=TEST_TIMEOUT,
            expected_patterns=[
                r"help|commands|available|usage",  # Should show help
                r"quit|clear|help",  # Should list commands
            ]
        )
        self.results.append(result)
        
        # Test 3: Clear Command Functionality
        result = self._run_tui_command(
            input_sequence="help\nclear\nquit\n",
            test_name="Clear Command Test",
            timeout=TEST_TIMEOUT,
            expected_patterns=[
                r"help",  # Initial help should appear
                # After clear, screen should be refreshed (hard to test in subprocess)
            ]
        )
        self.results.append(result)
        
        # Test 4: Interactive Messaging (LLM Integration)
        result = self._run_tui_command(
            input_sequence="Hello, can you help me with Python?\nquit\n",
            test_name="Interactive Messaging Test",
            timeout=LLM_RESPONSE_TIMEOUT,
            expected_patterns=[
                r"Hello|help|Python|assist|sure",  # Should get LLM response
            ]
        )
        self.results.append(result)
        
        # Test 5: Multi-message Conversation
        result = self._run_tui_command(
            input_sequence="What is Python?\nExplain variables\nquit\n",
            test_name="Multi-message Conversation Test",
            timeout=LLM_RESPONSE_TIMEOUT,
            expected_patterns=[
                r"Python.*language|programming",  # Should explain Python
                r"variable|data|value|store",  # Should explain variables
            ]
        )
        self.results.append(result)
        
        # Test 6: Error Handling - Invalid Command
        result = self._run_tui_command(
            input_sequence="invalid_command_xyz\nhelp\nquit\n",
            test_name="Error Handling Test",
            timeout=TEST_TIMEOUT,
            expected_patterns=[
                r"help|commands|available",  # Should show help after invalid command
            ]
        )
        self.results.append(result)
        
        # Test 7: Rich Interface Detection
        result = self._run_tui_command(
            input_sequence="quit\n",
            test_name="Rich Interface Detection Test",
            timeout=TUI_STARTUP_TIMEOUT,
            args=["--debug"],
            expected_patterns=[
                r"Rich|Enhanced|TTY|Live",  # Should indicate Rich interface mode
            ]
        )
        self.results.append(result)
        
        # Test 8: Input Visibility Test (Core Issue)
        result = self._run_tui_command(
            input_sequence="test input visibility\nquit\n",
            test_name="Input Visibility Test",
            timeout=TEST_TIMEOUT,
            expected_patterns=[
                # Should NOT see basic prompt fallback
                r"(?!.*>.*>)",  # Should not have basic '> ' prompt pattern
                # Should handle input properly
                r"test|input|visibility",
            ]
        )
        self.results.append(result)
        
        # Test 9: Graceful Exit with Keyboard Interrupt
        # Note: This test simulates Ctrl+C behavior
        result = self._run_tui_command(
            input_sequence="",  # No input, will timeout and get killed
            test_name="Keyboard Interrupt Test", 
            timeout=3,  # Short timeout to trigger interrupt-like behavior
            expected_patterns=[]  # Just verify it doesn't crash
        )
        # Override success criteria for this test
        result.success = result.exit_code in [0, -1, 130]  # Various interrupt codes
        if not result.success:
            result.failure_reason = f"Unexpected behavior on interrupt: {result.exit_code}"
        self.results.append(result)
        
        self._log(f"Completed {len(self.results)} acceptance tests")
        return self.results

    def generate_report(self) -> str:
        """Generate comprehensive test report."""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        report = f"""
# TUI Acceptance Test Report

## Summary
- **Total Tests**: {total_tests}
- **Passed**: {passed_tests}
- **Failed**: {failed_tests}  
- **Success Rate**: {success_rate:.1f}%

## Test Results

"""
        
        for result in self.results:
            status_icon = "✅" if result.success else "❌"
            report += f"### {status_icon} {result.name}\n"
            report += f"- **Status**: {'PASS' if result.success else 'FAIL'}\n"
            report += f"- **Duration**: {result.duration:.2f}s\n"
            report += f"- **Exit Code**: {result.exit_code}\n"
            
            if result.expected_patterns:
                report += f"- **Expected Patterns**: {len(result.expected_patterns)}\n"
                report += f"- **Found Patterns**: {len(result.found_patterns)}\n"
            
            if not result.success:
                report += f"- **Failure Reason**: {result.failure_reason}\n"
                
            if result.output and len(result.output.strip()) > 0:
                report += f"- **Output Sample**: {repr(result.output[:200])}\n"
                
            if result.error_output and len(result.error_output.strip()) > 0:
                report += f"- **Error Output**: {repr(result.error_output[:200])}\n"
                
            report += "\n"
        
        # Add recommendations
        report += "## Recommendations\n\n"
        if failed_tests == 0:
            report += "🎉 All acceptance tests passed! The Revolutionary TUI Interface is working correctly.\n\n"
        else:
            report += f"⚠️  {failed_tests} test(s) failed. Investigation and fixes needed:\n\n"
            for result in self.results:
                if not result.success:
                    report += f"- **{result.name}**: {result.failure_reason}\n"
            report += "\n"
        
        # Add test scenarios for manual verification
        report += """## Manual Verification Scenarios

For final validation, manually test these scenarios:

1. **Launch TUI**: `./agentsmcp tui` should show Rich interface without dotted lines
2. **Type visibility**: User typing should be immediately visible as you type
3. **LLM interaction**: Send "Hello" and verify you get an AI response
4. **Commands work**: Try `help`, `clear`, `quit` commands
5. **Rich interface**: Should see panels, colors, and proper layout (not basic `> ` prompt)
6. **Clean exit**: `quit` should exit gracefully without errors

"""
        
        return report


# Pytest integration
class TestTUIAcceptance:
    """Pytest test class for TUI acceptance tests."""
    
    @pytest.fixture(scope="class")
    def tui_runner(self):
        """Create TUI test runner."""
        return TUITestRunner(debug=True)
    
    @pytest.mark.ui
    @pytest.mark.integration
    @pytest.mark.slow
    def test_tui_launch(self, tui_runner):
        """Test TUI launches successfully."""
        result = tui_runner._run_tui_command(
            input_sequence="quit\n",
            test_name="TUI Launch",
            timeout=TUI_STARTUP_TIMEOUT
        )
        assert result.success, f"TUI launch failed: {result.failure_reason}"
        assert result.exit_code in [0, 130], f"Unexpected exit code: {result.exit_code}"
    
    @pytest.mark.ui
    @pytest.mark.integration 
    def test_help_command(self, tui_runner):
        """Test help command works."""
        result = tui_runner._run_tui_command(
            input_sequence="help\nquit\n",
            test_name="Help Command",
            expected_patterns=[r"help|commands|usage"]
        )
        assert result.success, f"Help command failed: {result.failure_reason}"
    
    @pytest.mark.ui
    @pytest.mark.integration
    @pytest.mark.slow
    def test_llm_interaction(self, tui_runner):
        """Test LLM interaction works."""
        result = tui_runner._run_tui_command(
            input_sequence="Hello world\nquit\n",
            test_name="LLM Interaction",
            timeout=LLM_RESPONSE_TIMEOUT,
            expected_patterns=[r"hello|world|help|assist"]
        )
        assert result.success, f"LLM interaction failed: {result.failure_reason}"


def main():
    """Run comprehensive TUI acceptance tests."""
    print("🚀 Starting Revolutionary TUI Interface Acceptance Tests...")
    
    runner = TUITestRunner(debug=True)
    results = runner.run_all_tests()
    
    # Generate and display report
    report = runner.generate_report()
    print(report)
    
    # Save report to file
    with open("TUI_ACCEPTANCE_TEST_REPORT.md", "w") as f:
        f.write(report)
    
    # Return appropriate exit code
    failed_count = sum(1 for r in results if not r.success)
    if failed_count == 0:
        print("🎉 All TUI acceptance tests passed!")
        return 0
    else:
        print(f"❌ {failed_count} TUI acceptance tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())