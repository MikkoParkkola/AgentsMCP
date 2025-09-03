#!/usr/bin/env python3
"""
Automated TUI Acceptance Test Runner

This script runs comprehensive acceptance tests for the Revolutionary TUI Interface
and generates detailed reports. It can be run standalone or integrated into CI/CD.

Usage:
    python run_tui_acceptance_tests.py                    # Run all tests
    python run_tui_acceptance_tests.py --quick            # Run quick tests only
    python run_tui_acceptance_tests.py --verbose          # Verbose output
    python run_tui_acceptance_tests.py --report-only      # Generate report from existing results
"""

import argparse
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Import our comprehensive test suite
from test_tui_acceptance_comprehensive import TUITestRunner, TestResult


class TUIAcceptanceTestSuite:
    """
    Main test suite runner for TUI acceptance tests.
    
    Orchestrates test execution, reporting, and CI/CD integration.
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.start_time = None
        self.results: List[TestResult] = []
        
    def log(self, message: str, level: str = "INFO"):
        """Log message if verbose mode enabled."""
        if self.verbose or level in ["ERROR", "WARNING"]:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [{level}] {message}")
    
    def run_quick_tests(self) -> List[TestResult]:
        """Run quick subset of acceptance tests."""
        self.log("Running quick TUI acceptance tests...")
        
        runner = TUITestRunner(debug=self.verbose)
        
        # Quick test subset - essential functionality only
        quick_tests = [
            ("quit\n", "Quick Launch Test", 10, [r"TUI|Revolutionary"]),
            ("help\nquit\n", "Quick Help Test", 10, [r"help|commands"]),
            ("Hello\nquit\n", "Quick LLM Test", 15, [r"hello|help|assist"]),
        ]
        
        results = []
        for input_seq, name, timeout, patterns in quick_tests:
            result = runner._run_tui_command(
                input_sequence=input_seq,
                test_name=name,
                timeout=timeout,
                expected_patterns=patterns
            )
            results.append(result)
            self.log(f"Test '{name}': {'PASS' if result.success else 'FAIL'}")
        
        return results
    
    def run_full_tests(self) -> List[TestResult]:
        """Run full comprehensive acceptance tests."""
        self.log("Running comprehensive TUI acceptance tests...")
        
        runner = TUITestRunner(debug=self.verbose)
        return runner.run_all_tests()
    
    def run_tests(self, quick: bool = False) -> Dict[str, Any]:
        """
        Run acceptance tests and return results summary.
        
        Args:
            quick: If True, run quick test subset only
            
        Returns:
            Dictionary with test results and metadata
        """
        self.start_time = time.time()
        self.log(f"Starting {'quick' if quick else 'comprehensive'} TUI acceptance tests...")
        
        # Run tests
        if quick:
            self.results = self.run_quick_tests()
        else:
            self.results = self.run_full_tests()
        
        # Calculate summary
        total_time = time.time() - self.start_time
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "test_type": "quick" if quick else "comprehensive",
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": success_rate,
            "total_duration": total_time,
            "results": [
                {
                    "name": r.name,
                    "success": r.success,
                    "duration": r.duration,
                    "failure_reason": r.failure_reason,
                    "exit_code": r.exit_code,
                    "expected_patterns": r.expected_patterns,
                    "found_patterns": r.found_patterns,
                }
                for r in self.results
            ]
        }
        
        self.log(f"Tests completed: {passed_tests}/{total_tests} passed ({success_rate:.1f}%)")
        return summary
    
    def generate_detailed_report(self, summary: Dict[str, Any]) -> str:
        """Generate detailed markdown report."""
        
        report = f"""# Revolutionary TUI Interface Acceptance Test Report

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Test Type**: {summary['test_type'].title()}  
**Duration**: {summary['total_duration']:.2f} seconds

## Executive Summary

{'🎉 **ALL TESTS PASSED**' if summary['failed_tests'] == 0 else '⚠️ **SOME TESTS FAILED**'}

- **Total Tests**: {summary['total_tests']}
- **Passed**: {summary['passed_tests']} ✅
- **Failed**: {summary['failed_tests']} ❌
- **Success Rate**: {summary['success_rate']:.1f}%

## Test Results Detail

"""
        
        # Add individual test results
        for result_data in summary['results']:
            status_icon = "✅" if result_data['success'] else "❌"
            report += f"### {status_icon} {result_data['name']}\n\n"
            report += f"- **Status**: {'PASS' if result_data['success'] else 'FAIL'}\n"
            report += f"- **Duration**: {result_data['duration']:.2f}s\n"
            report += f"- **Exit Code**: {result_data['exit_code']}\n"
            
            if result_data['expected_patterns']:
                report += f"- **Expected Patterns**: {len(result_data['expected_patterns'])}\n"
                report += f"- **Found Patterns**: {len(result_data['found_patterns'])}\n"
                
                # Show pattern details
                for pattern in result_data['expected_patterns']:
                    found = pattern in result_data['found_patterns']
                    report += f"  - `{pattern}`: {'✅' if found else '❌'}\n"
            
            if not result_data['success'] and result_data['failure_reason']:
                report += f"- **Failure Reason**: {result_data['failure_reason']}\n"
            
            report += "\n"
        
        # Add conclusions and recommendations
        report += "## Conclusions and Recommendations\n\n"
        
        if summary['failed_tests'] == 0:
            report += """✅ **Revolutionary TUI Interface is working correctly!**

All acceptance tests passed, indicating that:
- TUI launches successfully with Rich interface
- User input is visible (resolving the original issue)
- LLM integration works properly
- Commands (help, clear, quit) function correctly
- Error handling is graceful
- Interface displays properly without console pollution

**Recommendation**: The Revolutionary TUI Interface is ready for production use.
"""
        else:
            report += f"""⚠️ **{summary['failed_tests']} test(s) failed - Action required**

Failed tests indicate potential issues that need investigation:

"""
            
            # List specific failures
            for result_data in summary['results']:
                if not result_data['success']:
                    report += f"- **{result_data['name']}**: {result_data['failure_reason']}\n"
            
            report += """
**Recommendation**: Investigate and fix failing tests before production deployment.
"""
        
        # Add manual verification steps
        report += """
## Manual Verification Steps

To complement automated testing, manually verify these scenarios:

1. **Launch Test**:
   ```bash
   ./agentsmcp tui
   ```
   - Should show Revolutionary TUI interface with Rich panels
   - Should NOT show basic `> ` prompt
   - Should display colors and proper layout

2. **Input Visibility Test**:
   - Start typing immediately after TUI loads
   - Your typing should be visible as you type (original issue)
   - Should not have delay or invisibility

3. **LLM Interaction Test**:
   ```
   Hello, can you help me with Python?
   ```
   - Should get AI response within reasonable time
   - Response should be properly formatted

4. **Commands Test**:
   ```
   help     # Should show available commands
   clear    # Should clear screen/refresh
   quit     # Should exit gracefully
   ```

5. **Error Handling Test**:
   ```
   invalid_command_xyz
   ```
   - Should handle gracefully without crashing
   - Should show help or error message

## Test Environment

- **Python Version**: {sys.version.split()[0]}
- **Platform**: {sys.platform}
- **Working Directory**: {Path.cwd()}
- **Test Runner**: TUIAcceptanceTestSuite v1.0

---

*This report was automatically generated by the TUI Acceptance Test Suite*
"""
        
        return report
    
    def save_results(self, summary: Dict[str, Any], report: str, base_filename: str = None):
        """Save test results and report to files."""
        if base_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"tui_acceptance_test_{timestamp}"
        
        # Save JSON results
        json_file = f"{base_filename}.json"
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2)
        self.log(f"Saved JSON results to: {json_file}")
        
        # Save markdown report
        report_file = f"{base_filename}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        self.log(f"Saved report to: {report_file}")
        
        return json_file, report_file


def main():
    """Main entry point for TUI acceptance test runner."""
    parser = argparse.ArgumentParser(
        description="Run Revolutionary TUI Interface acceptance tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tui_acceptance_tests.py                    # Run all tests
  python run_tui_acceptance_tests.py --quick            # Quick test subset
  python run_tui_acceptance_tests.py --verbose          # Verbose output
  python run_tui_acceptance_tests.py --output results   # Custom output filename
        """
    )
    
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Run quick test subset only (faster execution)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true', 
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Base filename for output files (without extension)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to files'
    )
    
    args = parser.parse_args()
    
    # Create and run test suite
    suite = TUIAcceptanceTestSuite(verbose=args.verbose)
    
    print("🚀 Revolutionary TUI Interface Acceptance Test Runner")
    print("=" * 60)
    
    try:
        # Run tests
        summary = suite.run_tests(quick=args.quick)
        
        # Generate report
        report = suite.generate_detailed_report(summary)
        
        # Display summary
        print("\n" + "=" * 60)
        print("TEST EXECUTION SUMMARY")
        print("=" * 60)
        print(f"Tests Run: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']} ✅")
        print(f"Failed: {summary['failed_tests']} ❌")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Total Duration: {summary['total_duration']:.2f}s")
        
        # Save results
        if not args.no_save:
            json_file, report_file = suite.save_results(summary, report, args.output)
            print(f"\nResults saved to: {json_file}")
            print(f"Report saved to: {report_file}")
        
        # Print abbreviated report to console
        if args.verbose:
            print("\n" + "=" * 60)
            print("DETAILED REPORT")
            print("=" * 60)
            print(report)
        
        # Return appropriate exit code
        return 0 if summary['failed_tests'] == 0 else 1
        
    except KeyboardInterrupt:
        print("\n\n🛑 Test execution interrupted by user")
        return 130
        
    except Exception as e:
        print(f"\n❌ Test execution failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())