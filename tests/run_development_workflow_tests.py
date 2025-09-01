#!/usr/bin/env python3
"""
Test runner for AgentsMCP software development workflow tests.

This script provides an easy way to run the comprehensive development workflow
test suite with different configurations and reporting options.
"""

import argparse
import sys
import subprocess
from pathlib import Path
from typing import List, Optional


def run_test_suite(
    test_pattern: Optional[str] = None,
    markers: Optional[List[str]] = None,
    verbose: bool = True,
    coverage: bool = False,
    html_report: bool = False,
    parallel: bool = False,
    timeout: int = 300
) -> int:
    """Run the development workflow test suite with specified options."""
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Test files to run
    test_files = [
        "test_software_development_workflows.py",
        "test_development_tool_integration.py", 
        "test_development_performance_scenarios.py",
        "test_comprehensive_software_development.py"
    ]
    
    if test_pattern:
        # Filter tests by pattern
        cmd.extend(["-k", test_pattern])
    else:
        # Add specific test files
        for test_file in test_files:
            test_path = Path(__file__).parent / test_file
            if test_path.exists():
                cmd.append(str(test_path))
    
    # Add markers
    if markers:
        for marker in markers:
            cmd.extend(["-m", marker])
    
    # Verbosity
    if verbose:
        cmd.append("-v")
        cmd.append("--tb=short")
    
    # Coverage reporting
    if coverage:
        cmd.extend([
            "--cov=agentsmcp",
            "--cov-report=term-missing"
        ])
        if html_report:
            cmd.extend(["--cov-report=html:htmlcov"])
    
    # Parallel execution
    if parallel:
        cmd.extend(["-n", "auto"])
    
    # Timeout
    cmd.extend(["--timeout", str(timeout)])
    
    # Additional options for better output
    cmd.extend([
        "--color=yes",
        "--durations=10",  # Show 10 slowest tests
        "--tb=short"
    ])
    
    print(f"Running command: {' '.join(cmd)}")
    print("=" * 80)
    
    # Run the tests
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTest run interrupted by user")
        return 130
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Run AgentsMCP software development workflow tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all development workflow tests
  python run_development_workflow_tests.py

  # Run only integration tests
  python run_development_workflow_tests.py --markers integration

  # Run performance tests with coverage
  python run_development_workflow_tests.py --markers "slow" --coverage

  # Run specific test pattern
  python run_development_workflow_tests.py --pattern "test_multi_agent"

  # Run tests in parallel with HTML coverage report
  python run_development_workflow_tests.py --parallel --coverage --html-report

  # Run quick smoke tests (unit + integration, no slow tests)
  python run_development_workflow_tests.py --markers "not slow"
        """
    )
    
    parser.add_argument(
        "--pattern", "-k",
        help="Run tests matching this pattern"
    )
    
    parser.add_argument(
        "--markers", "-m",
        nargs="+",
        help="Run tests with these pytest markers (e.g., unit integration slow)"
    )
    
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    
    parser.add_argument(
        "--html-report",
        action="store_true", 
        help="Generate HTML coverage report (requires --coverage)"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel (requires pytest-xdist)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout for individual tests in seconds (default: 300)"
    )
    
    # Predefined test suites
    parser.add_argument(
        "--suite",
        choices=["all", "unit", "integration", "performance", "comprehensive"],
        help="Run predefined test suite"
    )
    
    args = parser.parse_args()
    
    # Handle predefined suites
    if args.suite:
        if args.suite == "unit":
            args.markers = ["unit"]
        elif args.suite == "integration":
            args.markers = ["integration", "not slow"]
        elif args.suite == "performance":
            args.markers = ["slow"]
        elif args.suite == "comprehensive":
            args.markers = ["integration"]
        # "all" doesn't set any markers, runs everything
    
    # HTML report requires coverage
    if args.html_report and not args.coverage:
        args.coverage = True
    
    # Run the tests
    exit_code = run_test_suite(
        test_pattern=args.pattern,
        markers=args.markers,
        verbose=not args.quiet,
        coverage=args.coverage,
        html_report=args.html_report,
        parallel=args.parallel,
        timeout=args.timeout
    )
    
    # Summary
    print("=" * 80)
    if exit_code == 0:
        print("✓ All tests passed successfully!")
        if args.coverage:
            print("✓ Coverage report generated")
            if args.html_report:
                print("✓ HTML coverage report available at: htmlcov/index.html")
    else:
        print(f"✗ Tests failed with exit code: {exit_code}")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()