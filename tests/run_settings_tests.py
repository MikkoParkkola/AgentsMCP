#!/usr/bin/env python3
"""
Test runner script for AgentsMCP settings management system.

This script runs the comprehensive test suite for the hierarchical settings
management system, including domain layer, service layer, and API tests.
"""

import sys
import subprocess
import argparse
from pathlib import Path

def run_tests(test_type=None, verbose=False, coverage=False):
    """Run the settings tests with various options."""
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add verbosity
    if verbose:
        cmd.extend(["-v", "-s"])
    
    # Add coverage
    if coverage:
        cmd.extend([
            "--cov=src/agentsmcp/settings",
            "--cov-report=html:tests/coverage_html",
            "--cov-report=term-missing"
        ])
    
    # Select test subset
    if test_type == "domain":
        cmd.append("tests/settings/domain/")
        print("🧪 Running domain layer tests...")
    elif test_type == "services":
        cmd.append("tests/settings/services/")
        print("🔧 Running service layer tests...")
    elif test_type == "api":
        cmd.append("tests/settings/api/")
        print("🌐 Running API layer tests...")
    elif test_type == "unit":
        cmd.extend(["tests/settings/domain/", "tests/settings/services/"])
        print("🧪 Running unit and integration tests...")
    elif test_type == "all" or test_type is None:
        cmd.append("tests/settings/")
        print("🚀 Running complete settings test suite...")
    else:
        print(f"❌ Unknown test type: {test_type}")
        return False
    
    # Add markers for better organization
    cmd.extend(["-m", "not slow"])  # Skip slow tests by default
    
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n❌ Tests interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run AgentsMCP settings management system tests"
    )
    
    parser.add_argument(
        "test_type",
        nargs="?",
        choices=["domain", "services", "api", "unit", "all"],
        default="all",
        help="Type of tests to run (default: all)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "-c", "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Include slow tests"
    )
    
    args = parser.parse_args()
    
    print("🔧 AgentsMCP Settings Management System Test Runner")
    print("=" * 60)
    
    # Check if pytest is available
    try:
        subprocess.run(
            ["python", "-m", "pytest", "--version"], 
            capture_output=True, 
            check=True
        )
    except subprocess.CalledProcessError:
        print("❌ pytest not found. Install with: pip install pytest pytest-asyncio")
        return 1
    
    # Check if required dependencies are available
    try:
        import fastapi
        import pydantic
    except ImportError as e:
        print(f"❌ Missing required dependency: {e}")
        print("Install with: pip install fastapi pydantic")
        return 1
    
    success = run_tests(
        test_type=args.test_type,
        verbose=args.verbose,
        coverage=args.coverage
    )
    
    if success:
        print("\n✅ All tests passed!")
        if args.coverage:
            print("📊 Coverage report generated at: tests/coverage_html/index.html")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())