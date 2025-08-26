#!/usr/bin/env python3
"""
Comprehensive Test Suite for AgentsMCP
Tests all use cases, UX, and UI using the compiled binary (agentsmcp command)
"""

import asyncio
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import tempfile
import os

class AgentsMCPTester:
    def __init__(self):
        self.test_results = []
        self.binary_path = "agentsmcp"  # Use compiled binary
        self.test_dir = Path.cwd()
        
    def log_test(self, test_name: str, status: str, details: str = "", issues: List[str] = None):
        """Log test results"""
        result = {
            "test_name": test_name,
            "status": status,  # PASS, FAIL, ISSUES
            "details": details,
            "issues": issues or [],
            "timestamp": time.time()
        }
        self.test_results.append(result)
        
        status_icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_icon} {test_name}: {status}")
        if details:
            print(f"   {details}")
        for issue in (issues or []):
            print(f"   🐛 ISSUE: {issue}")
        print()

    def run_command(self, args: List[str], input_text: str = "", timeout: int = 10) -> Tuple[int, str, str]:
        """Run agentsmcp command and return (exit_code, stdout, stderr)"""
        try:
            cmd = [self.binary_path] + args
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.test_dir
            )
            
            stdout, stderr = process.communicate(input=input_text, timeout=timeout)
            return process.returncode, stdout, stderr
            
        except subprocess.TimeoutExpired:
            process.kill()
            return -1, "", "Command timed out"
        except FileNotFoundError:
            return -1, "", f"Binary '{self.binary_path}' not found"
        except Exception as e:
            return -1, "", str(e)

    def test_binary_availability(self):
        """Test 1: Check if the binary is available and responds"""
        print("🔧 Testing binary availability...")
        
        exit_code, stdout, stderr = self.run_command(["--help"], timeout=5)
        
        if exit_code != 0:
            self.log_test("Binary Availability", "FAIL", 
                         f"Binary not found or failed to run. Exit code: {exit_code}",
                         [f"stderr: {stderr}", "Make sure agentsmcp is installed and in PATH"])
            return False
        
        # Check for expected help content
        expected_sections = ["usage:", "options:", "Examples:"]
        missing_sections = [section for section in expected_sections if section not in stdout.lower()]
        
        if missing_sections:
            self.log_test("Binary Availability", "ISSUES",
                         "Binary runs but help output seems incomplete",
                         [f"Missing help sections: {missing_sections}"])
        else:
            self.log_test("Binary Availability", "PASS", 
                         "Binary available and help works correctly")
        
        return True

    def test_startup_modes(self):
        """Test 2: Test all startup modes"""
        print("🔧 Testing startup modes...")
        
        modes = [
            ("interactive", "Interactive Command Mode"),
            ("dashboard", "dashboard"),  # Expected content varies
            ("stats", "statistics")      # Expected content varies
        ]
        
        issues = []
        
        for mode, expected_indicator in modes:
            print(f"  Testing {mode} mode...")
            exit_code, stdout, stderr = self.run_command(
                ["--mode", mode, "--no-welcome"], 
                input_text="\n",  # Send enter to exit
                timeout=8
            )
            
            if exit_code == -1:  # Timeout is expected for interactive modes
                if expected_indicator.lower() in stdout.lower():
                    print(f"    ✅ {mode} mode starts correctly")
                else:
                    issues.append(f"{mode} mode doesn't show expected content: '{expected_indicator}'")
                    print(f"    ❌ {mode} mode missing expected content")
            else:
                # Some modes might exit cleanly
                if stderr:
                    issues.append(f"{mode} mode has stderr output: {stderr}")
        
        if issues:
            self.log_test("Startup Modes", "ISSUES", 
                         f"Tested {len(modes)} modes, found issues",
                         issues)
        else:
            self.log_test("Startup Modes", "PASS", 
                         f"All {len(modes)} modes start correctly")

    def test_command_line_options(self):
        """Test 3: Test command line options"""
        print("🔧 Testing command line options...")
        
        issues = []
        
        # Test theme options
        themes = ["auto", "light", "dark"]
        for theme in themes:
            exit_code, stdout, stderr = self.run_command(
                ["--theme", theme, "--mode", "interactive", "--no-welcome"],
                input_text="\n",
                timeout=5
            )
            
            if "error:" in stderr.lower() or "unrecognized" in stderr.lower():
                issues.append(f"Theme '{theme}' not recognized or causes error")

        # Test other options
        test_options = [
            ["--no-welcome", "--mode", "interactive"],
            ["--no-colors", "--mode", "interactive"],  
            ["--debug", "--mode", "interactive"],
            ["--refresh-interval", "1.0", "--mode", "interactive"]
        ]
        
        for options in test_options:
            exit_code, stdout, stderr = self.run_command(options, input_text="\n", timeout=5)
            if "error:" in stderr.lower():
                issues.append(f"Options {options} cause error: {stderr}")

        if issues:
            self.log_test("Command Line Options", "ISSUES",
                         "Some options have problems", issues)
        else:
            self.log_test("Command Line Options", "PASS",
                         "All tested options work correctly")

    def test_interactive_mode_ux(self):
        """Test 4: Test interactive mode user experience"""
        print("🔧 Testing interactive mode UX...")
        
        issues = []
        
        # Test basic interactive startup
        exit_code, stdout, stderr = self.run_command(
            ["--mode", "interactive"],
            input_text="help\nexit\n",
            timeout=10
        )
        
        # Check for UX issues in output
        lines = stdout.split('\n')
        
        # Check for duplicate content
        daily_wisdom_count = sum(1 for line in lines if "daily wisdom" in line.lower())
        if daily_wisdom_count > 1:
            issues.append(f"Daily Wisdom appears {daily_wisdom_count} times (should be 1)")
        
        # Check for truncation (lines ending with ...)
        truncated_lines = [line for line in lines if line.strip().endswith("...") and len(line.strip()) > 10]
        if truncated_lines:
            issues.append(f"Found {len(truncated_lines)} truncated lines in output")
            for line in truncated_lines[:3]:  # Show first 3 examples
                issues.append(f"Truncated: {line.strip()}")
        
        # Check for error messages in normal flow
        error_indicators = ["error:", "failed:", "❌", "exception", "traceback"]
        error_lines = []
        for line in lines:
            line_lower = line.lower()
            if any(indicator in line_lower for indicator in error_indicators):
                error_lines.append(line.strip())
        
        if error_lines:
            issues.append("Found error messages during normal startup")
            for error in error_lines[:3]:  # Show first 3 examples
                issues.append(f"Error: {error}")
        
        # Check welcome screen completeness
        expected_welcome_elements = ["agentsmcp", "interactive", "help"]
        missing_elements = []
        output_lower = stdout.lower()
        for element in expected_welcome_elements:
            if element not in output_lower:
                missing_elements.append(element)
        
        if missing_elements:
            issues.append(f"Welcome screen missing elements: {missing_elements}")
        
        if issues:
            self.log_test("Interactive Mode UX", "ISSUES",
                         "Found UX issues in interactive mode", issues)
        else:
            self.log_test("Interactive Mode UX", "PASS",
                         "Interactive mode UX looks good")

    def test_conversational_interface(self):
        """Test 5: Test conversational interface functionality"""
        print("🔧 Testing conversational interface...")
        
        issues = []
        
        # Test basic conversation
        conversation_inputs = [
            "hello",
            "what can you do?", 
            "show me the status",
            "help me test this system",
            "exit"
        ]
        
        input_text = "\n".join(conversation_inputs) + "\n"
        
        exit_code, stdout, stderr = self.run_command(
            ["--mode", "interactive", "--no-welcome"],
            input_text=input_text,
            timeout=15
        )
        
        # Analyze responses
        lines = stdout.split('\n')
        
        # Check if conversational responses are present
        response_indicators = ["🤖", "agentsmcp:", "response", "i can", "i'll", "let me"]
        found_responses = False
        for line in lines:
            line_lower = line.lower()
            if any(indicator in line_lower for indicator in response_indicators):
                found_responses = True
                break
        
        if not found_responses:
            issues.append("No conversational responses detected")
        
        # Check for broken agent delegation
        if "delegate-to-" in stdout.lower() and "not yet implemented" in stdout.lower():
            issues.append("Agent delegation shows 'not yet implemented' warnings")
        
        # Check for incomplete responses (agent tasks that don't complete)
        if "⚠️" in stdout and "fallback" in stdout.lower():
            issues.append("System falling back to warnings instead of completing tasks")
        
        if issues:
            self.log_test("Conversational Interface", "ISSUES",
                         "Found issues in conversation handling", issues)
        else:
            self.log_test("Conversational Interface", "PASS",
                         "Conversational interface works well")

    def test_command_execution(self):
        """Test 6: Test direct command execution"""
        print("🔧 Testing command execution...")
        
        issues = []
        
        # Test direct commands
        commands = [
            "help",
            "status", 
            "dashboard",
            "settings"
        ]
        
        for command in commands:
            exit_code, stdout, stderr = self.run_command(
                ["--mode", "interactive", "--no-welcome"],
                input_text=f"{command}\nexit\n",
                timeout=8
            )
            
            # Check if command was recognized and executed
            if "unrecognized" in stdout.lower() or "unknown" in stdout.lower():
                issues.append(f"Command '{command}' not recognized")
            elif len(stdout.strip()) < 50:  # Very short output might indicate no response
                issues.append(f"Command '{command}' gave minimal or no output")

        if issues:
            self.log_test("Command Execution", "ISSUES",
                         "Found issues with direct commands", issues)
        else:
            self.log_test("Command Execution", "PASS",
                         "Direct commands execute correctly")

    def test_agent_orchestration(self):
        """Test 7: Test agent orchestration functionality"""
        print("🔧 Testing agent orchestration...")
        
        issues = []
        
        # Test orchestration requests
        orchestration_prompts = [
            "delegate to codex: analyze this repository",
            "use claude to summarize the documentation", 
            "→→ DELEGATE-TO-ollama: simple test task"
        ]
        
        for prompt in orchestration_prompts:
            exit_code, stdout, stderr = self.run_command(
                ["--mode", "interactive", "--no-welcome"],
                input_text=f"{prompt}\nexit\n",
                timeout=10
            )
            
            # Check for orchestration issues
            if "not yet implemented" in stdout.lower():
                issues.append(f"Orchestration shows 'not implemented' for: {prompt}")
            elif "fallback" in stdout.lower() and "failed" in stdout.lower():
                issues.append(f"Orchestration failing with fallbacks for: {prompt}")
            elif "⚠️" in stdout and len([line for line in stdout.split('\n') if line.strip()]) < 5:
                issues.append(f"Orchestration gives minimal response for: {prompt}")

        if issues:
            self.log_test("Agent Orchestration", "ISSUES",
                         "Found issues with agent orchestration", issues)
        else:
            self.log_test("Agent Orchestration", "PASS",
                         "Agent orchestration working correctly")

    def test_error_handling(self):
        """Test 8: Test error handling and recovery"""
        print("🔧 Testing error handling...")
        
        issues = []
        
        # Test invalid commands and inputs
        invalid_inputs = [
            "///invalid///command///",
            "?!@#$%^&*()",
            "very long command that should not exist and might cause issues if not handled properly",
            ""  # Empty input
        ]
        
        for invalid_input in invalid_inputs:
            exit_code, stdout, stderr = self.run_command(
                ["--mode", "interactive", "--no-welcome"],
                input_text=f"{invalid_input}\nexit\n",
                timeout=8
            )
            
            # Check that system handles it gracefully (no crashes, helpful responses)
            if "traceback" in stdout.lower() or "exception" in stdout.lower():
                issues.append(f"System crashed or showed traceback for input: '{invalid_input}'")
            elif stderr and "error" in stderr.lower():
                issues.append(f"System showed stderr error for input: '{invalid_input}': {stderr}")

        # Test invalid command line options
        invalid_options = [
            ["--invalid-option"],
            ["--mode", "invalid_mode"],
            ["--theme", "invalid_theme"]
        ]
        
        for invalid_option in invalid_options:
            exit_code, stdout, stderr = self.run_command(invalid_option, timeout=5)
            
            if exit_code != 2:  # argparse typically returns 2 for invalid args
                # Check that error message is helpful
                if not stderr or "usage:" not in stderr:
                    issues.append(f"Invalid option {invalid_option} doesn't show helpful error")

        if issues:
            self.log_test("Error Handling", "ISSUES",
                         "Found issues with error handling", issues)
        else:
            self.log_test("Error Handling", "PASS",
                         "Error handling works correctly")

    def test_performance_and_responsiveness(self):
        """Test 9: Test performance and responsiveness"""
        print("🔧 Testing performance and responsiveness...")
        
        issues = []
        
        # Test startup time
        start_time = time.time()
        exit_code, stdout, stderr = self.run_command(
            ["--mode", "interactive", "--no-welcome"],
            input_text="exit\n",
            timeout=10
        )
        startup_time = time.time() - start_time
        
        if startup_time > 5:
            issues.append(f"Slow startup: {startup_time:.1f} seconds (should be under 5s)")
        
        # Test response time to simple commands
        simple_commands = ["help", "status"]
        for command in simple_commands:
            start_time = time.time()
            exit_code, stdout, stderr = self.run_command(
                ["--mode", "interactive", "--no-welcome"],
                input_text=f"{command}\nexit\n",
                timeout=8
            )
            response_time = time.time() - start_time
            
            if response_time > 4:
                issues.append(f"Slow response to '{command}': {response_time:.1f}s (should be under 4s)")
        
        if issues:
            self.log_test("Performance", "ISSUES",
                         "Found performance issues", issues)
        else:
            self.log_test("Performance", "PASS",
                         f"Good performance - startup: {startup_time:.1f}s")

    def test_documentation_and_help(self):
        """Test 10: Test documentation and help completeness"""
        print("🔧 Testing documentation and help...")
        
        issues = []
        
        # Test --help output
        exit_code, stdout, stderr = self.run_command(["--help"], timeout=5)
        
        expected_help_content = [
            "usage:",
            "options:",
            "--mode",
            "--theme", 
            "examples:",
            "interactive",
            "dashboard"
        ]
        
        missing_content = []
        help_output_lower = stdout.lower()
        for expected in expected_help_content:
            if expected not in help_output_lower:
                missing_content.append(expected)
        
        if missing_content:
            issues.append(f"Help output missing content: {missing_content}")
        
        # Test in-app help
        exit_code, stdout, stderr = self.run_command(
            ["--mode", "interactive", "--no-welcome"],
            input_text="help\nexit\n",
            timeout=8
        )
        
        # Check if help command provides useful information
        if len(stdout.split('\n')) < 10:
            issues.append("In-app help command provides very little information")
        
        if "command" not in stdout.lower() and "help" not in stdout.lower():
            issues.append("In-app help doesn't explain available commands")
        
        if issues:
            self.log_test("Documentation", "ISSUES",
                         "Found documentation issues", issues)
        else:
            self.log_test("Documentation", "PASS",
                         "Documentation is complete and helpful")

    async def run_all_tests(self):
        """Run all tests in sequence"""
        print("🚀 Starting Comprehensive AgentsMCP Test Suite")
        print("=" * 60)
        print()
        
        # Test sequence - order matters (basic functionality first)
        tests = [
            self.test_binary_availability,
            self.test_startup_modes,
            self.test_command_line_options,
            self.test_interactive_mode_ux,
            self.test_conversational_interface,
            self.test_command_execution,
            self.test_agent_orchestration,
            self.test_error_handling,
            self.test_performance_and_responsiveness,
            self.test_documentation_and_help
        ]
        
        # Run tests
        for i, test_func in enumerate(tests, 1):
            print(f"[{i}/{len(tests)}] {test_func.__doc__.split(':')[0].replace('Test ', '').strip()}")
            print("-" * 40)
            
            try:
                if test_func == self.test_binary_availability:
                    # If binary test fails, stop testing
                    if not test_func():
                        print("❌ Cannot continue without working binary")
                        break
                else:
                    test_func()
                    
            except Exception as e:
                self.log_test(test_func.__name__, "FAIL", f"Test crashed: {str(e)}")
            
            print()
        
        # Summary
        self.print_summary()
        return self.generate_fix_recommendations()

    def print_summary(self):
        """Print test summary"""
        print("=" * 60)
        print("🎯 TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for result in self.test_results if result["status"] == "PASS")
        issues = sum(1 for result in self.test_results if result["status"] == "ISSUES") 
        failed = sum(1 for result in self.test_results if result["status"] == "FAIL")
        total = len(self.test_results)
        
        print(f"Total Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"⚠️  Issues Found: {issues}")
        print(f"❌ Failed: {failed}")
        print()
        
        if failed > 0:
            print("🔥 CRITICAL FAILURES:")
            for result in self.test_results:
                if result["status"] == "FAIL":
                    print(f"  • {result['test_name']}: {result['details']}")
            print()
        
        if issues > 0:
            print("🐛 ISSUES FOUND:")
            issue_count = 0
            for result in self.test_results:
                if result["status"] == "ISSUES":
                    for issue in result["issues"]:
                        issue_count += 1
                        print(f"  {issue_count}. {issue}")
            print()

    def generate_fix_recommendations(self) -> List[str]:
        """Generate prioritized fix recommendations"""
        fixes = []
        
        # Collect all issues
        all_issues = []
        for result in self.test_results:
            if result["status"] in ["FAIL", "ISSUES"]:
                all_issues.extend(result["issues"])
        
        # Categorize and prioritize fixes
        ui_issues = [issue for issue in all_issues if any(keyword in issue.lower() 
                    for keyword in ["truncated", "duplicate", "display", "ui", "ux"])]
        
        functionality_issues = [issue for issue in all_issues if any(keyword in issue.lower()
                              for keyword in ["not implemented", "failed", "error", "command"])]
        
        performance_issues = [issue for issue in all_issues if any(keyword in issue.lower()
                            for keyword in ["slow", "timeout", "performance"])]
        
        if ui_issues:
            fixes.append(("UI/UX Issues", ui_issues))
        if functionality_issues:
            fixes.append(("Functionality Issues", functionality_issues))
        if performance_issues:
            fixes.append(("Performance Issues", performance_issues))
        
        return fixes

async def main():
    """Main test function"""
    tester = AgentsMCPTester()
    
    print("🧪 AgentsMCP Comprehensive Test Suite")
    print("Testing the compiled binary (agentsmcp command)")
    print("Evaluating all use cases, UX, and UI")
    print()
    
    fix_recommendations = await tester.run_all_tests()
    
    print("🔧 FIX RECOMMENDATIONS")
    print("=" * 60)
    
    if not fix_recommendations:
        print("🎉 No issues found! AgentsMCP is working perfectly.")
        return 0
    
    priority = 1
    for category, issues in fix_recommendations:
        print(f"\n{priority}. {category} (Priority: {'HIGH' if priority == 1 else 'MEDIUM'})")
        for issue in issues:
            print(f"   • {issue}")
        priority += 1
    
    print(f"\n📊 Total Issues to Fix: {sum(len(issues) for _, issues in fix_recommendations)}")
    
    return 1 if fix_recommendations else 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)