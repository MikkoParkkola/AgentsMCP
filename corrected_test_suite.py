#!/usr/bin/env python3
"""
Corrected AgentsMCP Comprehensive Test Suite

This test suite properly evaluates the compiled binary using the correct subcommand structure.
Tests all use cases, UX, and UI to identify remaining issues.
"""

import subprocess
import sys
import time
import os
from typing import Dict, List, Tuple, Optional


class CorrectedAgentsMCPTester:
    def __init__(self):
        self.issues = []
        self.test_results = {}
        self.passed_tests = 0
        self.failed_tests = 0
        
    def log_issue(self, issue: str):
        """Log an issue found during testing"""
        self.issues.append(issue)
        print(f"   🐛 ISSUE: {issue}")
    
    def log_success(self, message: str):
        """Log a successful test result"""
        print(f"   ✅ SUCCESS: {message}")
    
    def run_command(self, cmd: List[str], timeout: int = 5) -> Tuple[str, str, int]:
        """Run a command and return stdout, stderr, return code"""
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=timeout
            )
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            return "", "TIMEOUT", 1
        except Exception as e:
            return "", f"ERROR: {e}", 1
    
    def test_basic_functionality(self):
        """Test basic binary functionality"""
        print("🔧 Testing basic functionality...")
        
        # Test binary exists and runs
        stdout, stderr, code = self.run_command(["agentsmcp", "--version"])
        if code != 0:
            self.log_issue(f"Binary not working: {stderr}")
            return False
        else:
            self.log_success("Binary runs and shows version")
        
        # Test help output
        stdout, stderr, code = self.run_command(["agentsmcp", "--help"])
        if code != 0:
            self.log_issue(f"Help command failed: {stderr}")
        else:
            # Check for essential help content
            required_commands = ["interactive", "dashboard", "budget", "costs"]
            missing_commands = [cmd for cmd in required_commands if cmd not in stdout.lower()]
            if missing_commands:
                self.log_issue(f"Help missing commands: {missing_commands}")
            else:
                self.log_success("Help shows all main commands")
        
        return True
    
    def test_subcommands(self):
        """Test all subcommands with proper structure"""
        print("🔧 Testing subcommands...")
        
        subcommands = [
            "interactive",
            "dashboard", 
            "budget",
            "costs",
            "discovery",
            "mcp",
            "models",
            "optimize",
            "server"
        ]
        
        for cmd in subcommands:
            stdout, stderr, code = self.run_command(["agentsmcp", cmd, "--help"])
            if code != 0:
                self.log_issue(f"Subcommand '{cmd}' help failed: {stderr}")
            else:
                self.log_success(f"Subcommand '{cmd}' help works")
    
    def test_interactive_mode_startup(self):
        """Test interactive mode startup and early interface"""
        print("🔧 Testing interactive mode startup...")
        
        # Test interactive mode starts properly (very short timeout)
        stdout, stderr, code = self.run_command(["agentsmcp", "interactive"], timeout=3)
        
        if "TIMEOUT" in stderr:
            # This is expected - interactive mode should keep running
            # Check if we got the expected startup output
            if "AgentsMCP" in stdout and "Interactive Mode" in stdout:
                self.log_success("Interactive mode starts with proper branding")
            else:
                self.log_issue("Interactive mode startup missing expected content")
        elif code != 0:
            self.log_issue(f"Interactive mode failed to start: {stderr}")
        
        # Check for UI elements in the output we did capture
        expected_ui_elements = [
            "AgentsMCP",
            "Interactive",
            "Daily Wisdom",
            "agentsmcp ▶"
        ]
        
        missing_elements = [elem for elem in expected_ui_elements if elem not in stdout]
        if missing_elements:
            self.log_issue(f"Interactive mode missing UI elements: {missing_elements}")
        else:
            self.log_success("Interactive mode has all expected UI elements")
    
    def test_dashboard_startup(self):
        """Test dashboard mode startup"""
        print("🔧 Testing dashboard startup...")
        
        stdout, stderr, code = self.run_command(["agentsmcp", "dashboard"], timeout=3)
        
        if "TIMEOUT" in stderr:
            # Expected for dashboard mode
            expected_dashboard_elements = [
                "Dashboard Mode",
                "System Overview",
                "Performance Metrics"
            ]
            
            missing_elements = [elem for elem in expected_dashboard_elements if elem not in stdout]
            if missing_elements:
                self.log_issue(f"Dashboard missing elements: {missing_elements}")
            else:
                self.log_success("Dashboard shows all expected elements")
        elif code != 0:
            self.log_issue(f"Dashboard failed to start: {stderr}")
    
    def test_theme_options(self):
        """Test theme options work"""
        print("🔧 Testing theme options...")
        
        themes = ["auto", "light", "dark"]
        for theme in themes:
            stdout, stderr, code = self.run_command(
                ["agentsmcp", "dashboard", "--theme", theme], 
                timeout=2
            )
            
            if code != 0 and "TIMEOUT" not in stderr:
                self.log_issue(f"Theme '{theme}' causes error: {stderr}")
            else:
                self.log_success(f"Theme '{theme}' works")
    
    def test_budget_commands(self):
        """Test budget and cost commands"""
        print("🔧 Testing budget commands...")
        
        # Test budget command
        stdout, stderr, code = self.run_command(["agentsmcp", "budget", "--help"])
        if code != 0:
            self.log_issue(f"Budget help failed: {stderr}")
        else:
            self.log_success("Budget command help works")
        
        # Test costs command  
        stdout, stderr, code = self.run_command(["agentsmcp", "costs", "--help"])
        if code != 0:
            self.log_issue(f"Costs help failed: {stderr}")
        else:
            self.log_success("Costs command help works")
    
    def test_mcp_management(self):
        """Test MCP server management"""
        print("🔧 Testing MCP management...")
        
        stdout, stderr, code = self.run_command(["agentsmcp", "mcp", "--help"])
        if code != 0:
            self.log_issue(f"MCP help failed: {stderr}")
        else:
            self.log_success("MCP management help works")
    
    def test_discovery_features(self):
        """Test agent discovery features"""
        print("🔧 Testing discovery features...")
        
        stdout, stderr, code = self.run_command(["agentsmcp", "discovery", "--help"])
        if code != 0:
            self.log_issue(f"Discovery help failed: {stderr}")
        else:
            self.log_success("Discovery command help works")
    
    def evaluate_ui_quality(self):
        """Evaluate UI/UX quality from captured output"""
        print("🔧 Evaluating UI/UX quality...")
        
        # Test interactive mode UI quality
        stdout, stderr, code = self.run_command(["agentsmcp", "interactive"], timeout=3)
        
        ui_quality_checks = {
            "Has consistent branding": "AgentsMCP" in stdout,
            "Shows proper mode indicator": "Interactive Mode" in stdout,
            "Has clear navigation hints": any(hint in stdout for hint in ["help", "commands", "Type"]),
            "No duplicate welcome": stdout.count("Daily Wisdom") <= 1,
            "Clean interface": "ERROR" not in stdout.upper() and "WARN" not in stdout.upper(),
            "Proper prompt indicator": "▶" in stdout
        }
        
        for check, passed in ui_quality_checks.items():
            if passed:
                self.log_success(f"UI Quality: {check}")
            else:
                self.log_issue(f"UI Quality issue: {check}")
    
    def run_comprehensive_test(self):
        """Run all tests and generate report"""
        print("🚀 Starting Corrected AgentsMCP Test Suite")
        print("=" * 60)
        
        # Run all test categories
        test_categories = [
            ("Basic Functionality", self.test_basic_functionality),
            ("Subcommands", self.test_subcommands),
            ("Interactive Mode", self.test_interactive_mode_startup),
            ("Dashboard Mode", self.test_dashboard_startup),
            ("Theme Options", self.test_theme_options),
            ("Budget Commands", self.test_budget_commands),
            ("MCP Management", self.test_mcp_management),
            ("Discovery Features", self.test_discovery_features),
            ("UI/UX Quality", self.evaluate_ui_quality)
        ]
        
        for i, (name, test_func) in enumerate(test_categories, 1):
            print(f"\n[{i}/{len(test_categories)}] {name}")
            print("-" * 40)
            
            try:
                result = test_func()
                if result is not False:
                    self.test_results[name] = "PASS"
                    self.passed_tests += 1
                else:
                    self.test_results[name] = "ISSUES"
            except Exception as e:
                print(f"   ❌ Test failed with exception: {e}")
                self.test_results[name] = "FAILED"
                self.failed_tests += 1
        
        # Generate final report
        self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 60)
        print("🎯 CORRECTED TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        issues_found = len([r for r in self.test_results.values() if r == "ISSUES"])
        
        print(f"Total Test Categories: {total_tests}")
        print(f"✅ Passed: {self.passed_tests}")
        print(f"⚠️ Issues Found: {issues_found}")
        print(f"❌ Failed: {self.failed_tests}")
        
        if self.issues:
            print(f"\n🐛 ISSUES IDENTIFIED:")
            for i, issue in enumerate(self.issues, 1):
                print(f"  {i}. {issue}")
        
        if issues_found == 0 and self.failed_tests == 0:
            print("\n🎉 ALL TESTS PASSED! AgentsMCP is working correctly.")
        else:
            print(f"\n🔧 PRIORITY FIXES NEEDED:")
            critical_issues = [i for i in self.issues if any(word in i.lower() for word in ["failed", "error", "missing"])]
            for issue in critical_issues[:5]:  # Top 5 critical issues
                print(f"  • {issue}")


if __name__ == "__main__":
    tester = CorrectedAgentsMCPTester()
    tester.run_comprehensive_test()