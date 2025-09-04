#!/usr/bin/env python3
"""
TUI Expected vs Actual Behavior Test Suite

This test suite creates automated scenarios to identify exactly where 
the interactive TUI interface should exist but is missing or broken.
"""

import subprocess
import sys
import time
import threading
import signal
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import json


@dataclass 
class TUIBehaviorExpectation:
    """Defines what we expect vs what actually happens"""
    scenario: str
    expected_behavior: str
    expected_runtime: str 
    expected_output_patterns: List[str]
    expected_interactive: bool
    
    actual_behavior: str = ""
    actual_runtime: float = 0.0
    actual_output: str = ""
    actual_interactive: bool = False
    
    def is_behavior_correct(self) -> bool:
        """Check if actual matches expected behavior"""
        return (
            self.actual_interactive == self.expected_interactive and
            all(pattern in self.actual_output.lower() for pattern in self.expected_output_patterns)
        )


class TUIBehaviorTester:
    """Test TUI behavior against expectations"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results: List[TUIBehaviorExpectation] = []
    
    def run_tui_test(self, expectation: TUIBehaviorExpectation, timeout: int = 10) -> TUIBehaviorExpectation:
        """Run a single TUI test scenario"""
        print(f"\n🧪 Testing: {expectation.scenario}")
        print(f"Expected: {expectation.expected_behavior}")
        
        start_time = time.time()
        
        try:
            # Run TUI process
            process = subprocess.Popen(
                [sys.executable, "-m", "agentsmcp", "tui"],
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, 
                text=True,
                bufsize=1
            )
            
            # Monitor with timeout
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                expectation.actual_output = stdout + stderr
                expectation.actual_runtime = time.time() - start_time
                
                # Analyze if it was interactive
                expectation.actual_interactive = self._detect_interactive_behavior(expectation.actual_output, expectation.actual_runtime, timeout)
                
            except subprocess.TimeoutExpired:
                # Process timed out - might be interactive!
                process.kill()
                stdout, stderr = process.communicate()
                expectation.actual_output = stdout + stderr
                expectation.actual_runtime = time.time() - start_time
                expectation.actual_interactive = True  # Timed out = was waiting for input
                
        except Exception as e:
            expectation.actual_output = f"Error: {e}"
            expectation.actual_runtime = time.time() - start_time
            expectation.actual_interactive = False
        
        # Set actual behavior description
        if expectation.actual_interactive:
            expectation.actual_behavior = f"Interactive mode - ran for {expectation.actual_runtime:.1f}s (timeout)"
        else:
            expectation.actual_behavior = f"Non-interactive - exited after {expectation.actual_runtime:.1f}s"
        
        # Print results
        self._print_test_result(expectation)
        return expectation
    
    def _detect_interactive_behavior(self, output: str, runtime: float, timeout: int) -> bool:
        """Detect if TUI was actually interactive"""
        
        # If it ran for most of the timeout, it was probably waiting for input
        if runtime >= (timeout - 1):
            return True
        
        # Check for interactive indicators in output
        interactive_patterns = [
            "waiting for input",
            "type your message", 
            "interactive mode",
            "awaiting command",
            "press enter"
        ]
        
        output_lower = output.lower()
        has_interactive_patterns = any(pattern in output_lower for pattern in interactive_patterns)
        
        # Check for demo mode (indicates non-interactive)
        demo_patterns = ["demo mode", "demo countdown", "non-tty"]
        has_demo_patterns = any(pattern in output_lower for pattern in demo_patterns)
        
        # Interactive if has interactive patterns and no demo patterns
        return has_interactive_patterns and not has_demo_patterns
    
    def _print_test_result(self, expectation: TUIBehaviorExpectation):
        """Print test result comparison"""
        print(f"Actual: {expectation.actual_behavior}")
        
        # Check correctness
        is_correct = expectation.is_behavior_correct()
        status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
        print(f"Status: {status}")
        
        if not is_correct:
            print(f"📋 Gap Analysis:")
            print(f"  Expected Interactive: {expectation.expected_interactive}")
            print(f"  Actual Interactive: {expectation.actual_interactive}")
            
            missing_patterns = [p for p in expectation.expected_output_patterns 
                               if p not in expectation.actual_output.lower()]
            if missing_patterns:
                print(f"  Missing Output Patterns: {missing_patterns}")
        
        # Show key output snippets
        output_lines = expectation.actual_output.split('\n')
        key_lines = [line for line in output_lines if any(keyword in line.lower() 
                    for keyword in ['tui', 'interactive', 'demo', 'revolutionary', 'ready'])]
        
        if key_lines:
            print(f"📄 Key Output Lines:")
            for line in key_lines[:5]:  # Show first 5 key lines
                print(f"  {line.strip()}")
    
    def create_test_scenarios(self) -> List[TUIBehaviorExpectation]:
        """Create comprehensive test scenarios"""
        return [
            # Scenario 1: Basic Interactive TUI
            TUIBehaviorExpectation(
                scenario="Basic TUI Startup - Should be Interactive",
                expected_behavior="TUI starts and waits indefinitely for user input",
                expected_runtime="Should timeout (>10s) because it's waiting for input",
                expected_output_patterns=["revolutionary tui", "ready", "interactive"],
                expected_interactive=True
            ),
            
            # Scenario 2: TUI Visual Components  
            TUIBehaviorExpectation(
                scenario="TUI Visual Interface - Rich Components",
                expected_behavior="Rich visual interface with panels and components",
                expected_runtime="Should timeout while displaying interface", 
                expected_output_patterns=["symphony dashboard", "ai command composer", "revolutionary"],
                expected_interactive=True
            ),
            
            # Scenario 3: TUI Input Handling
            TUIBehaviorExpectation(
                scenario="TUI Input System - Ready for Commands",
                expected_behavior="TUI should indicate it's ready to accept user commands",
                expected_runtime="Should timeout while waiting for commands",
                expected_output_patterns=["ready", "command", "input"],
                expected_interactive=True
            ),
            
            # Scenario 4: TUI Lifecycle Management
            TUIBehaviorExpectation(
                scenario="TUI Lifecycle - Persistent Operation", 
                expected_behavior="TUI should stay running until user explicitly exits",
                expected_runtime="Should timeout (not exit automatically)",
                expected_output_patterns=["tui", "interface"],
                expected_interactive=True
            )
        ]
    
    def run_all_tests(self):
        """Run all test scenarios"""
        print("🚀 TUI EXPECTED vs ACTUAL BEHAVIOR TEST SUITE")
        print("=" * 60)
        print("Identifying gaps between what TUI should do vs what it actually does")
        print()
        
        scenarios = self.create_test_scenarios()
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n{'='*20} TEST {i}/{len(scenarios)} {'='*20}")
            result = self.run_tui_test(scenario, timeout=8)
            self.results.append(result)
            
            # Brief pause between tests
            time.sleep(0.5)
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self):
        """Print comprehensive test summary"""
        print(f"\n🏁 TEST SUMMARY")
        print("=" * 60)
        
        correct_count = sum(1 for r in self.results if r.is_behavior_correct())
        total_count = len(self.results)
        
        print(f"Tests Passed: {correct_count}/{total_count}")
        print(f"Tests Failed: {total_count - correct_count}/{total_count}")
        
        if correct_count == total_count:
            print("🎉 ALL TESTS PASSED - TUI behavior matches expectations!")
        else:
            print("⚠️  BEHAVIOR GAPS IDENTIFIED:")
            
            # Group issues by type
            non_interactive = [r for r in self.results if r.expected_interactive and not r.actual_interactive]
            missing_features = [r for r in self.results if not r.is_behavior_correct() and r.actual_interactive == r.expected_interactive]
            
            if non_interactive:
                print(f"\n🔴 CRITICAL ISSUE: TUI Not Interactive ({len(non_interactive)} tests)")
                print("   TUI should stay running and wait for user input, but exits immediately")
                print("   This indicates the core interactive loop is missing or broken")
                
                # Show evidence
                demo_mode_detected = any("demo" in r.actual_output.lower() for r in non_interactive)
                if demo_mode_detected:
                    print("   ROOT CAUSE: TUI detects non-TTY environment and enters demo mode")
                    print("   SOLUTION NEEDED: Proper TTY handling or force interactive mode")
            
            if missing_features:
                print(f"\n🟡 FEATURE GAPS: Expected features missing ({len(missing_features)} tests)")
                for result in missing_features:
                    missing_patterns = [p for p in result.expected_output_patterns 
                                       if p not in result.actual_output.lower()]
                    if missing_patterns:
                        print(f"   {result.scenario}: Missing {missing_patterns}")
        
        print(f"\n📊 DETAILED RESULTS:")
        for i, result in enumerate(self.results, 1):
            status = "✅" if result.is_behavior_correct() else "❌"
            print(f"   {i}. {status} {result.scenario}")
            print(f"      Expected: {result.expected_behavior}")
            print(f"      Actual: {result.actual_behavior}")


def main():
    """Main test execution"""
    project_root = Path(__file__).parent
    tester = TUIBehaviorTester(project_root)
    tester.run_all_tests()


if __name__ == "__main__":
    main()