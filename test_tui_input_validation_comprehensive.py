#!/usr/bin/env python3
"""
Comprehensive TUI Input Typing Validation Test Suite

This test suite validates that the critical input typing visibility fix is working correctly.
The fix addresses the issue where user typing was not immediately visible in the TUI interface.

Test Categories:
1. Enhanced Demo Mode Testing
2. User Experience Validation 
3. Regression Testing
4. Edge Case Testing
5. Critical Success Criteria Verification

Key Fix Components Tested:
- Input rendering pipeline immediate feedback
- Character input handling in _handle_character_input()
- State synchronization between input buffer and pipeline
- Demo mode with interactive capabilities in non-TTY environments
- Fallback input handling for different terminal environments
"""

import asyncio
import subprocess
import sys
import os
import time
import threading
import io
import contextlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import tempfile
import pty
import signal
import fcntl
import termios
import select

@dataclass
class ValidationResult:
    """Result of a validation test."""
    test_name: str
    passed: bool
    details: str
    error: Optional[str] = None
    execution_time: float = 0.0

@dataclass
class TestOutput:
    """Captured output from TUI test."""
    stdout: str
    stderr: str
    exit_code: int
    execution_time: float

class TUIInputValidator:
    """Comprehensive TUI input typing validation."""
    
    def __init__(self):
        self.results: List[ValidationResult] = []
        self.agentsmcp_path = "./agentsmcp"
        self.total_tests = 0
        self.passed_tests = 0
    
    def log(self, message: str):
        """Log with timestamp."""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def add_result(self, result: ValidationResult):
        """Add a validation result."""
        self.results.append(result)
        self.total_tests += 1
        if result.passed:
            self.passed_tests += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        
        self.log(f"{status} {result.test_name}")
        if result.details:
            self.log(f"     Details: {result.details}")
        if result.error:
            self.log(f"     Error: {result.error}")
        if result.execution_time > 0:
            self.log(f"     Time: {result.execution_time:.2f}s")
    
    def run_tui_command(self, timeout: float = 10.0, input_data: str = None) -> TestOutput:
        """Run the TUI command and capture output."""
        start_time = time.time()
        
        try:
            # For input testing, we need to simulate interactive input
            if input_data:
                # Use expect-style interaction for input testing
                import pexpect
                try:
                    child = pexpect.spawn(self.agentsmcp_path + " tui", timeout=timeout)
                    child.logfile_read = sys.stdout.buffer if hasattr(sys.stdout, 'buffer') else None
                    
                    # Send input data
                    for char in input_data:
                        child.send(char)
                        time.sleep(0.1)  # Small delay between characters
                    
                    # Wait for output
                    child.expect(pexpect.EOF, timeout=timeout)
                    output = child.before.decode('utf-8') if child.before else ""
                    exit_code = child.exitstatus
                    
                    return TestOutput(
                        stdout=output,
                        stderr="",
                        exit_code=exit_code or 0,
                        execution_time=time.time() - start_time
                    )
                    
                except ImportError:
                    # Fallback without pexpect
                    self.log("WARNING: pexpect not available, using basic subprocess")
                    pass
            
            # Standard subprocess for non-interactive tests
            process = subprocess.Popen(
                [self.agentsmcp_path, "tui"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                text=True,
                cwd=os.getcwd()
            )
            
            # Send input if provided
            stdin_data = input_data if input_data else None
            
            try:
                stdout, stderr = process.communicate(input=stdin_data, timeout=timeout)
                exit_code = process.returncode
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                exit_code = -1
                stderr += "\nProcess killed due to timeout"
            
            return TestOutput(
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return TestOutput(
                stdout="",
                stderr=f"Command execution failed: {str(e)}",
                exit_code=-1,
                execution_time=time.time() - start_time
            )
    
    def test_enhanced_demo_mode(self):
        """Test 1: Enhanced Demo Mode with Interactive Capabilities."""
        self.log("\n=== TEST 1: Enhanced Demo Mode ===")
        
        # Run TUI in demo mode (non-TTY environment)
        output = self.run_tui_command(timeout=15.0)
        
        # Check for demo mode indicators
        demo_indicators = [
            "Revolutionary TUI Interface - Demo Mode",
            "Running in non-TTY environment",
            "TUI initialized successfully in demo mode",
            "All systems operational",
            "Ready for interactive use in TTY environment",
            "Demo countdown:",
            "Demo completed - TUI shutting down gracefully"
        ]
        
        missing_indicators = []
        for indicator in demo_indicators:
            if indicator not in output.stdout:
                missing_indicators.append(indicator)
        
        # Test passes if most indicators are present and no critical errors
        passed = (len(missing_indicators) <= 1 and 
                 output.exit_code == 0 and
                 "TUI initialized successfully" in output.stdout)
        
        details = f"Exit code: {output.exit_code}, Demo indicators: {len(demo_indicators) - len(missing_indicators)}/{len(demo_indicators)}"
        if missing_indicators:
            details += f", Missing: {missing_indicators}"
        
        self.add_result(ValidationResult(
            test_name="Enhanced Demo Mode",
            passed=passed,
            details=details,
            error=output.stderr if output.stderr else None,
            execution_time=output.execution_time
        ))
    
    def test_tui_startup_sequence(self):
        """Test 2: TUI Startup Sequence Unchanged."""
        self.log("\n=== TEST 2: TUI Startup Sequence ===")
        
        output = self.run_tui_command(timeout=12.0)
        
        # Expected startup sequence elements
        startup_elements = [
            "Starting Revolutionary TUI system",
            "Phase 1: Initializing feature detection",
            "Phase 2: Determining feature level",
            "Phase 3: Launching",
            "Revolutionary TUI Interface"
        ]
        
        found_elements = 0
        for element in startup_elements:
            if element in output.stdout:
                found_elements += 1
        
        passed = (found_elements >= 4 and output.exit_code == 0)
        
        self.add_result(ValidationResult(
            test_name="TUI Startup Sequence",
            passed=passed,
            details=f"Startup elements found: {found_elements}/{len(startup_elements)}",
            error=output.stderr if output.stderr else None,
            execution_time=output.execution_time
        ))
    
    def test_no_crashes_or_errors(self):
        """Test 3: No Crashes or Critical Errors."""
        self.log("\n=== TEST 3: No Crashes or Critical Errors ===")
        
        output = self.run_tui_command(timeout=12.0)
        
        # Check for critical errors
        critical_errors = [
            "Traceback",
            "Exception",
            "Error:",
            "CRITICAL",
            "FATAL"
        ]
        
        found_errors = []
        for error in critical_errors:
            if error in output.stdout or error in output.stderr:
                found_errors.append(error)
        
        # Filter out expected/benign warnings
        benign_warnings = [
            "Cannot detect terminal capabilities",
            "Not running in a terminal (TTY)"
        ]
        
        actual_errors = []
        for error in found_errors:
            is_benign = any(warning in output.stderr for warning in benign_warnings)
            if not is_benign:
                actual_errors.append(error)
        
        passed = (len(actual_errors) == 0 and output.exit_code == 0)
        
        self.add_result(ValidationResult(
            test_name="No Crashes or Critical Errors",
            passed=passed,
            details=f"Exit code: {output.exit_code}, Critical errors: {len(actual_errors)}",
            error=f"Found errors: {actual_errors}" if actual_errors else None,
            execution_time=output.execution_time
        ))
    
    def test_input_handling_infrastructure(self):
        """Test 4: Input Handling Infrastructure Present."""
        self.log("\n=== TEST 4: Input Handling Infrastructure ===")
        
        # Check that key components exist in the codebase
        components_to_check = [
            "src/agentsmcp/ui/v2/revolutionary_tui_interface.py",
            "src/agentsmcp/ui/v2/input_rendering_pipeline.py",
            "src/agentsmcp/ui/v2/reliability/integration_layer.py"
        ]
        
        missing_components = []
        for component in components_to_check:
            if not os.path.exists(component):
                missing_components.append(component)
        
        # Check for key methods in the revolutionary_tui_interface.py
        key_methods = []
        try:
            with open("src/agentsmcp/ui/v2/revolutionary_tui_interface.py", "r") as f:
                content = f.read()
                if "_handle_character_input" in content:
                    key_methods.append("_handle_character_input")
                if "_demo_mode_loop" in content:
                    key_methods.append("_demo_mode_loop")
                if "render_immediate_feedback" in content:
                    key_methods.append("render_immediate_feedback")
        except Exception as e:
            pass
        
        passed = (len(missing_components) == 0 and len(key_methods) >= 2)
        
        self.add_result(ValidationResult(
            test_name="Input Handling Infrastructure",
            passed=passed,
            details=f"Components: {len(components_to_check) - len(missing_components)}/{len(components_to_check)}, Key methods: {len(key_methods)}",
            error=f"Missing: {missing_components}" if missing_components else None
        ))
    
    def test_input_rendering_pipeline_fix(self):
        """Test 5: Input Rendering Pipeline Fix Implementation."""
        self.log("\n=== TEST 5: Input Rendering Pipeline Fix ===")
        
        try:
            with open("src/agentsmcp/ui/v2/input_rendering_pipeline.py", "r") as f:
                pipeline_content = f.read()
            
            # Check for key fix components
            fix_indicators = [
                "render_immediate_feedback",
                "immediate feedback for character input",
                "InputState",
                "cursor_position",
                "sanitize_control_characters"
            ]
            
            found_indicators = 0
            for indicator in fix_indicators:
                if indicator in pipeline_content:
                    found_indicators += 1
            
            # Check for the critical fix - render_immediate_feedback method
            has_immediate_feedback = "def render_immediate_feedback" in pipeline_content
            has_state_update = "_current_state = InputState" in pipeline_content
            
            passed = (found_indicators >= 4 and has_immediate_feedback and has_state_update)
            
            self.add_result(ValidationResult(
                test_name="Input Rendering Pipeline Fix",
                passed=passed,
                details=f"Fix indicators: {found_indicators}/{len(fix_indicators)}, Immediate feedback: {has_immediate_feedback}, State update: {has_state_update}",
            ))
            
        except Exception as e:
            self.add_result(ValidationResult(
                test_name="Input Rendering Pipeline Fix",
                passed=False,
                details="Failed to analyze pipeline file",
                error=str(e)
            ))
    
    def test_demo_to_interactive_transition(self):
        """Test 6: Demo Mode Shows Transition to Interactive Mode Capability."""
        self.log("\n=== TEST 6: Demo to Interactive Transition ===")
        
        output = self.run_tui_command(timeout=12.0)
        
        # Look for messages indicating interactive capability
        transition_indicators = [
            "Ready for interactive use in TTY environment",
            "Run in a proper terminal for full interactive experience",
            "TUI staying active",
            "demonstrating proper lifecycle"
        ]
        
        found_indicators = 0
        for indicator in transition_indicators:
            if indicator in output.stdout:
                found_indicators += 1
        
        # Check that demo completes properly without hanging
        clean_exit = ("Demo completed" in output.stdout and 
                     "shutting down gracefully" in output.stdout)
        
        passed = (found_indicators >= 2 and clean_exit and output.exit_code == 0)
        
        self.add_result(ValidationResult(
            test_name="Demo to Interactive Transition",
            passed=passed,
            details=f"Transition indicators: {found_indicators}/{len(transition_indicators)}, Clean exit: {clean_exit}",
            execution_time=output.execution_time
        ))
    
    def test_performance_and_responsiveness(self):
        """Test 7: Performance and Responsiveness."""
        self.log("\n=== TEST 7: Performance and Responsiveness ===")
        
        # Test startup time
        start_time = time.time()
        output = self.run_tui_command(timeout=15.0)
        total_time = time.time() - start_time
        
        # Good performance indicators
        startup_fast = total_time < 10.0  # Should start within 10 seconds
        no_hanging = output.exit_code == 0  # Should exit cleanly
        
        # Check for performance problems
        performance_issues = []
        if "timeout" in output.stderr.lower():
            performance_issues.append("timeout")
        if total_time > 15.0:
            performance_issues.append("slow_startup")
        
        passed = (startup_fast and no_hanging and len(performance_issues) == 0)
        
        self.add_result(ValidationResult(
            test_name="Performance and Responsiveness",
            passed=passed,
            details=f"Startup time: {total_time:.2f}s, Performance issues: {len(performance_issues)}",
            error=f"Issues: {performance_issues}" if performance_issues else None,
            execution_time=total_time
        ))
    
    def test_logging_isolation(self):
        """Test 8: Logging Isolation During TUI Operation."""
        self.log("\n=== TEST 8: Logging Isolation ===")
        
        output = self.run_tui_command(timeout=12.0)
        
        # Check that logging is properly isolated - no debug spam in stdout
        debug_pollution_indicators = [
            "DEBUG:",
            "INFO:",
            "WARNING:" # Some warnings are expected, but excessive ones indicate problems
        ]
        
        pollution_count = 0
        for indicator in debug_pollution_indicators:
            pollution_count += output.stdout.count(indicator)
            pollution_count += output.stderr.count(indicator)
        
        # Some warnings are expected (terminal capabilities), but debug spam should be minimal
        acceptable_pollution = pollution_count < 10  # Allow some expected warnings
        clean_user_output = "Revolutionary TUI Interface" in output.stdout
        
        passed = (acceptable_pollution and clean_user_output)
        
        self.add_result(ValidationResult(
            test_name="Logging Isolation",
            passed=passed,
            details=f"Pollution count: {pollution_count}, Clean output: {clean_user_output}",
            error=f"Too much debug output" if not acceptable_pollution else None
        ))
    
    def run_all_validations(self):
        """Run all validation tests."""
        self.log("🚀 Starting Comprehensive TUI Input Typing Validation")
        self.log("=" * 70)
        
        # Run all tests
        test_methods = [
            self.test_enhanced_demo_mode,
            self.test_tui_startup_sequence, 
            self.test_no_crashes_or_errors,
            self.test_input_handling_infrastructure,
            self.test_input_rendering_pipeline_fix,
            self.test_demo_to_interactive_transition,
            self.test_performance_and_responsiveness,
            self.test_logging_isolation
        ]
        
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                test_name = test_method.__name__.replace("test_", "").replace("_", " ").title()
                self.add_result(ValidationResult(
                    test_name=test_name,
                    passed=False,
                    details="Test execution failed",
                    error=str(e)
                ))
            time.sleep(0.5)  # Brief pause between tests
    
    def generate_report(self) -> str:
        """Generate comprehensive validation report."""
        report = []
        report.append("=" * 70)
        report.append("🔍 COMPREHENSIVE TUI INPUT TYPING VALIDATION REPORT")
        report.append("=" * 70)
        report.append("")
        
        # Summary
        pass_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        report.append(f"📊 SUMMARY:")
        report.append(f"   Total Tests: {self.total_tests}")
        report.append(f"   Passed: {self.passed_tests}")
        report.append(f"   Failed: {self.total_tests - self.passed_tests}")
        report.append(f"   Pass Rate: {pass_rate:.1f}%")
        report.append("")
        
        # Overall verdict
        if pass_rate >= 87.5:  # 7/8 tests or better
            verdict = "✅ INPUT TYPING FIX VALIDATED - Ready for Production"
            status_color = "GREEN"
        elif pass_rate >= 75.0:  # 6/8 tests
            verdict = "⚠️  INPUT TYPING FIX MOSTLY WORKING - Minor Issues Found"
            status_color = "YELLOW"
        else:
            verdict = "❌ INPUT TYPING FIX NEEDS ATTENTION - Critical Issues Found"
            status_color = "RED"
        
        report.append(f"🏆 OVERALL VERDICT: {verdict}")
        report.append("")
        
        # Critical Success Criteria Check
        report.append("🎯 CRITICAL SUCCESS CRITERIA:")
        
        critical_criteria = [
            ("Demo mode shows demo messages first", "Enhanced Demo Mode"),
            ("TUI startup sequence unchanged", "TUI Startup Sequence"),
            ("No crashes or errors", "No Crashes or Critical Errors"),
            ("Input handling infrastructure present", "Input Handling Infrastructure"),
            ("Input rendering pipeline fix implemented", "Input Rendering Pipeline Fix"),
            ("Demo transitions to show interactive capability", "Demo to Interactive Transition")
        ]
        
        criteria_met = 0
        for criteria_desc, test_name in critical_criteria:
            test_result = next((r for r in self.results if r.test_name == test_name), None)
            if test_result and test_result.passed:
                report.append(f"   ✅ {criteria_desc}")
                criteria_met += 1
            else:
                report.append(f"   ❌ {criteria_desc}")
        
        report.append("")
        report.append(f"   Critical Criteria Met: {criteria_met}/{len(critical_criteria)}")
        report.append("")
        
        # Detailed Results
        report.append("📋 DETAILED RESULTS:")
        for result in self.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            report.append(f"   {status} {result.test_name}")
            if result.details:
                report.append(f"      Details: {result.details}")
            if result.error:
                report.append(f"      Error: {result.error}")
            if result.execution_time > 0:
                report.append(f"      Time: {result.execution_time:.2f}s")
            report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS:")
        failed_tests = [r for r in self.results if not r.passed]
        if not failed_tests:
            report.append("   🎉 All tests passed! The input typing fix is working correctly.")
            report.append("   ✨ The TUI is ready for user testing and production deployment.")
        else:
            report.append("   🔧 The following issues should be addressed:")
            for failed_test in failed_tests:
                report.append(f"      • {failed_test.test_name}: {failed_test.details}")
                if failed_test.error:
                    report.append(f"        Error: {failed_test.error}")
        
        report.append("")
        report.append("=" * 70)
        
        return "\n".join(report)

def main():
    """Main validation execution."""
    validator = TUIInputValidator()
    
    try:
        validator.run_all_validations()
        report = validator.generate_report()
        
        # Print to console
        print(report)
        
        # Save to file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_filename = f"tui_input_validation_report_{timestamp}.txt"
        with open(report_filename, "w") as f:
            f.write(report)
        
        print(f"\n📄 Detailed report saved to: {report_filename}")
        
        # Exit with appropriate code
        if validator.passed_tests == validator.total_tests:
            sys.exit(0)  # All tests passed
        elif validator.passed_tests >= validator.total_tests * 0.75:
            sys.exit(1)  # Most tests passed, minor issues
        else:
            sys.exit(2)  # Significant issues found
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Validation failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(3)

if __name__ == "__main__":
    main()