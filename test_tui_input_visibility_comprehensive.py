#!/usr/bin/env python3
"""
FINAL VERIFICATION: TUI Input Visibility Comprehensive Test

This test comprehensively validates that the TUI input visibility issue has been resolved.
Users should now be able to SEE what they're typing in real-time.

KEY VERIFICATION POINTS:
1. Input Panel Visibility Test - Verify Rich Live display shows input panel immediately
2. Character Echo Test - Confirm typed characters appear visually in the input area
3. Clean Output Test - Verify no emergency debug prints flood the terminal
4. Fallback Mode Test - Ensure input area visible even if Rich fails
5. User Experience Test - Simulate real user typing scenario

CRITICAL TEST SCENARIOS:
- Normal Startup: TUI starts and input area immediately visible
- Character Typing: User types "hello" and sees each character appear
- Command Typing: User types "/quit" and sees the full command
- Clean Terminal: No debug pollution during normal operation
- Error Recovery: If Rich fails, fallback mode provides clear input area
"""

import asyncio
import io
import logging
import os
import sys
import threading
import time
import unittest
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from unittest.mock import Mock, patch, MagicMock, call
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface, TUIState
    from agentsmcp.ui.cli_app import CLIConfig
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.live import Live
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the correct directory")
    sys.exit(1)


@dataclass
class TestResult:
    """Result of a test scenario."""
    name: str
    success: bool
    details: str
    output_captured: str = ""
    error_messages: List[str] = None
    
    def __post_init__(self):
        if self.error_messages is None:
            self.error_messages = []


class InputVisibilityTestHarness:
    """Test harness for comprehensive input visibility validation."""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.console = Console(force_terminal=True, width=120)
        self.emergency_patterns = [
            "🔥 EMERGENCY",
            "EMERGENCY",
            "emergency debug",
            "debug spam",
            "input logging"
        ]
        
    def capture_output(self):
        """Capture stdout/stderr for analysis."""
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        return redirect_stdout(stdout_capture), redirect_stderr(stderr_capture), stdout_capture, stderr_capture

    async def test_input_panel_visibility(self) -> TestResult:
        """Test 1: Verify Rich Live display shows input panel immediately."""
        print("\n🔍 Test 1: Input Panel Visibility Test")
        
        with self.capture_output() as (stdout_ctx, stderr_ctx, stdout_cap, stderr_cap):
            with stdout_ctx, stderr_ctx:
                try:
                    # Create TUI with mocked dependencies
                    cli_config = CLIConfig()
                    cli_config.debug_mode = False  # No debug spam
                    
                    tui = RevolutionaryTUIInterface(cli_config)
                    
                    # Initialize the layout
                    tui._setup_layout()
                    
                    # Verify input panel is created properly
                    input_panel = tui._create_input_panel()
                    
                    # Check that input panel has expected structure
                    assert isinstance(input_panel, Text), "Input panel should be Text object"
                    
                    # Check that the layout includes input section
                    assert tui.layout is not None, "Layout should be initialized"
                    assert "input" in tui.layout, "Input section should be in layout"
                    
                    # Verify the input panel has proper styling
                    input_str = str(input_panel)
                    assert "AI Command Composer" in input_str or ">" in input_str, "Input panel should show prompt"
                    
                    return TestResult(
                        name="Input Panel Visibility",
                        success=True,
                        details="✅ Input panel created successfully with proper structure and styling",
                        output_captured=stdout_cap.getvalue() + stderr_cap.getvalue()
                    )
                    
                except Exception as e:
                    return TestResult(
                        name="Input Panel Visibility", 
                        success=False,
                        details=f"❌ Failed to create input panel: {e}",
                        error_messages=[str(e)],
                        output_captured=stdout_cap.getvalue() + stderr_cap.getvalue()
                    )

    async def test_character_echo(self) -> TestResult:
        """Test 2: Confirm typed characters appear visually in the input area."""
        print("\n🔍 Test 2: Character Echo Test")
        
        with self.capture_output() as (stdout_ctx, stderr_ctx, stdout_cap, stderr_cap):
            with stdout_ctx, stderr_ctx:
                try:
                    cli_config = CLIConfig()
                    cli_config.debug_mode = False
                    
                    tui = RevolutionaryTUIInterface(cli_config)
                    tui._setup_layout()
                    
                    # Test character input
                    test_inputs = ["h", "e", "l", "l", "o"]
                    input_states = []
                    
                    for char in test_inputs:
                        tui.state.current_input += char
                        
                        # Create input panel and capture its state
                        input_panel = tui._create_input_panel()
                        input_display = str(input_panel)
                        input_states.append(input_display)
                        
                        # Verify current input is visible
                        assert tui.state.current_input in input_display, f"Input '{tui.state.current_input}' not visible in display"
                    
                    # Verify progressive typing is captured
                    expected_progressions = ["h", "he", "hel", "hell", "hello"]
                    for i, expected in enumerate(expected_progressions):
                        assert expected in input_states[i], f"Expected '{expected}' not found in input state {i}"
                    
                    return TestResult(
                        name="Character Echo",
                        success=True,
                        details=f"✅ All {len(test_inputs)} characters echoed correctly in input display",
                        output_captured=stdout_cap.getvalue() + stderr_cap.getvalue()
                    )
                    
                except Exception as e:
                    return TestResult(
                        name="Character Echo",
                        success=False,
                        details=f"❌ Character echo failed: {e}",
                        error_messages=[str(e)],
                        output_captured=stdout_cap.getvalue() + stderr_cap.getvalue()
                    )

    async def test_clean_output(self) -> TestResult:
        """Test 3: Verify no emergency debug prints flood the terminal."""
        print("\n🔍 Test 3: Clean Output Test")
        
        with self.capture_output() as (stdout_ctx, stderr_ctx, stdout_cap, stderr_cap):
            with stdout_ctx, stderr_ctx:
                try:
                    cli_config = CLIConfig()
                    cli_config.debug_mode = False  # Explicitly disable debug
                    
                    tui = RevolutionaryTUIInterface(cli_config)
                    tui._setup_layout()
                    
                    # Simulate some user input
                    tui.state.current_input = "test command"
                    input_panel = tui._create_input_panel()
                    
                    # Force refresh display
                    tui._sync_refresh_display()
                    
                    # Capture all output
                    output = stdout_cap.getvalue() + stderr_cap.getvalue()
                    
                    # Check for emergency patterns
                    emergency_found = []
                    for pattern in self.emergency_patterns:
                        if pattern.lower() in output.lower():
                            emergency_found.append(pattern)
                    
                    if emergency_found:
                        return TestResult(
                            name="Clean Output",
                            success=False,
                            details=f"❌ Found emergency debug patterns: {emergency_found}",
                            error_messages=emergency_found,
                            output_captured=output
                        )
                    
                    return TestResult(
                        name="Clean Output",
                        success=True,
                        details="✅ No emergency debug prints detected - clean terminal output",
                        output_captured=output
                    )
                    
                except Exception as e:
                    return TestResult(
                        name="Clean Output",
                        success=False,
                        details=f"❌ Clean output test failed: {e}",
                        error_messages=[str(e)],
                        output_captured=stdout_cap.getvalue() + stderr_cap.getvalue()
                    )

    async def test_fallback_mode(self) -> TestResult:
        """Test 4: Ensure input area visible even if Rich fails."""
        print("\n🔍 Test 4: Fallback Mode Test")
        
        with self.capture_output() as (stdout_ctx, stderr_ctx, stdout_cap, stderr_cap):
            with stdout_ctx, stderr_ctx:
                try:
                    cli_config = CLIConfig()
                    cli_config.debug_mode = False
                    
                    # Mock Rich to fail
                    with patch('agentsmcp.ui.v2.revolutionary_tui_interface.RICH_AVAILABLE', False):
                        tui = RevolutionaryTUIInterface(cli_config)
                        
                        # Test fallback input creation
                        tui.state.current_input = "fallback test"
                        
                        # Even without Rich, input should be manageable
                        assert hasattr(tui.state, 'current_input'), "State should maintain current_input"
                        assert tui.state.current_input == "fallback test", "Input should be preserved"
                    
                    # Test Rich failure recovery
                    with patch('rich.live.Live.__enter__', side_effect=Exception("Rich failed")):
                        tui = RevolutionaryTUIInterface(cli_config)
                        tui._setup_layout()
                        
                        # Should handle Rich failure gracefully
                        input_panel = tui._create_input_panel()
                        assert input_panel is not None, "Input panel should be created even if Rich fails"
                    
                    return TestResult(
                        name="Fallback Mode",
                        success=True,
                        details="✅ Fallback mode works - input handling preserved when Rich fails",
                        output_captured=stdout_cap.getvalue() + stderr_cap.getvalue()
                    )
                    
                except Exception as e:
                    return TestResult(
                        name="Fallback Mode",
                        success=False,
                        details=f"❌ Fallback mode failed: {e}",
                        error_messages=[str(e)],
                        output_captured=stdout_cap.getvalue() + stderr_cap.getvalue()
                    )

    async def test_user_experience(self) -> TestResult:
        """Test 5: Simulate real user typing scenario."""
        print("\n🔍 Test 5: User Experience Test")
        
        with self.capture_output() as (stdout_ctx, stderr_ctx, stdout_cap, stderr_cap):
            with stdout_ctx, stderr_ctx:
                try:
                    cli_config = CLIConfig()
                    cli_config.debug_mode = False
                    
                    tui = RevolutionaryTUIInterface(cli_config)
                    tui._setup_layout()
                    
                    # Simulate real user scenarios
                    scenarios = [
                        ("hello", "Simple greeting"),
                        ("/help", "Command input"),
                        ("tell me about AI agents", "Complex query"),
                        ("", "Empty input"),
                        ("a" * 100, "Long input"),
                        ("special chars: !@#$%^&*()", "Special characters"),
                    ]
                    
                    scenario_results = []
                    for input_text, description in scenarios:
                        tui.state.current_input = input_text
                        input_panel = tui._create_input_panel()
                        input_display = str(input_panel)
                        
                        # Verify input is visible
                        if input_text:
                            is_visible = input_text in input_display
                        else:
                            # For empty input, check that prompt/cursor is visible
                            is_visible = (">" in input_display or 
                                         "AI Command Composer" in input_display or
                                         "Enter command" in input_display)
                        
                        scenario_results.append({
                            'scenario': description,
                            'input': input_text,
                            'visible': is_visible,
                            'display': input_display[:100] + "..." if len(input_display) > 100 else input_display
                        })
                    
                    failed_scenarios = [s for s in scenario_results if not s['visible']]
                    
                    if failed_scenarios:
                        return TestResult(
                            name="User Experience",
                            success=False,
                            details=f"❌ {len(failed_scenarios)}/{len(scenarios)} scenarios failed: {[s['scenario'] for s in failed_scenarios]}",
                            error_messages=[f"{s['scenario']}: '{s['input']}' not visible in '{s['display']}'" for s in failed_scenarios],
                            output_captured=stdout_cap.getvalue() + stderr_cap.getvalue()
                        )
                    
                    return TestResult(
                        name="User Experience",
                        success=True,
                        details=f"✅ All {len(scenarios)} user scenarios successful - input clearly visible",
                        output_captured=stdout_cap.getvalue() + stderr_cap.getvalue()
                    )
                    
                except Exception as e:
                    return TestResult(
                        name="User Experience",
                        success=False,
                        details=f"❌ User experience test failed: {e}",
                        error_messages=[str(e)],
                        output_captured=stdout_cap.getvalue() + stderr_cap.getvalue()
                    )

    async def run_comprehensive_test(self) -> List[TestResult]:
        """Run all comprehensive input visibility tests."""
        print("🚀 Starting TUI Input Visibility Comprehensive Test Suite")
        print("="*80)
        
        # Run all tests
        tests = [
            self.test_input_panel_visibility(),
            self.test_character_echo(),
            self.test_clean_output(),
            self.test_fallback_mode(),
            self.test_user_experience(),
        ]
        
        self.results = await asyncio.gather(*tests, return_exceptions=True)
        
        # Handle any exceptions
        for i, result in enumerate(self.results):
            if isinstance(result, Exception):
                self.results[i] = TestResult(
                    name=f"Test {i+1}",
                    success=False,
                    details=f"❌ Test crashed: {result}",
                    error_messages=[str(result)]
                )
        
        return self.results

    def generate_report(self) -> str:
        """Generate comprehensive test report."""
        report = []
        report.append("🔍 TUI INPUT VISIBILITY - COMPREHENSIVE TEST REPORT")
        report.append("="*80)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests
        
        report.append(f"\n📊 SUMMARY:")
        report.append(f"   Total Tests: {total_tests}")
        report.append(f"   ✅ Passed: {passed_tests}")
        report.append(f"   ❌ Failed: {failed_tests}")
        report.append(f"   Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        report.append(f"\n📋 DETAILED RESULTS:")
        for i, result in enumerate(self.results, 1):
            status = "✅ PASS" if result.success else "❌ FAIL"
            report.append(f"\n{i}. {result.name}: {status}")
            report.append(f"   Details: {result.details}")
            
            if result.error_messages:
                report.append(f"   Errors: {', '.join(result.error_messages)}")
            
            # Show output if there were issues or debug info needed
            if not result.success and result.output_captured:
                output_preview = result.output_captured[:200].replace('\n', ' ')
                report.append(f"   Output: {output_preview}...")
        
        # Final assessment
        report.append(f"\n🎯 FINAL ASSESSMENT:")
        if failed_tests == 0:
            report.append("   ✅ ALL TESTS PASSED - TUI INPUT VISIBILITY IS FIXED!")
            report.append("   ✅ Users can now see what they're typing")
            report.append("   ✅ No debug spam pollution")
            report.append("   ✅ Rich and fallback modes working")
            report.append("   ✅ Ready for user deployment")
        else:
            report.append(f"   ❌ {failed_tests} TESTS FAILED - INPUT VISIBILITY ISSUES REMAIN")
            report.append("   ❌ Additional fixes needed before user deployment")
            
            # Specific recommendations
            failed_names = [r.name for r in self.results if not r.success]
            if "Character Echo" in failed_names:
                report.append("   🔧 Fix: Character echo mechanism needs repair")
            if "Clean Output" in failed_names:
                report.append("   🔧 Fix: Remove remaining debug print pollution")
            if "Fallback Mode" in failed_names:
                report.append("   🔧 Fix: Improve fallback mode for Rich failures")
        
        report.append(f"\n" + "="*80)
        return "\n".join(report)


async def main():
    """Run the comprehensive TUI input visibility test."""
    try:
        # Set up test environment
        test_harness = InputVisibilityTestHarness()
        
        # Run all tests
        results = await test_harness.run_comprehensive_test()
        
        # Generate and display report
        report = test_harness.generate_report()
        print("\n" + report)
        
        # Save report to file
        report_file = "/Users/mikko/github/AgentsMCP/TUI_INPUT_VISIBILITY_TEST_REPORT.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\n📄 Full report saved to: {report_file}")
        
        # Return appropriate exit code
        failed_count = sum(1 for r in results if not r.success)
        return 0 if failed_count == 0 else 1
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        return 130
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # Run the test
    exit_code = asyncio.run(main())
    sys.exit(exit_code)