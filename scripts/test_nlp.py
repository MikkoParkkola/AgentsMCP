#!/usr/bin/env python3
"""Test runner for NLP module with comprehensive coverage reporting."""

import sys
import subprocess
import os
from pathlib import Path


def run_tests():
    """Run NLP tests with various configurations."""
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print("🧠 Running Natural Language Processing Tests")
    print("=" * 60)
    
    # Test configurations
    test_configs = [
        {
            "name": "Unit Tests",
            "command": ["python", "-m", "pytest", "tests/cli/v3/nlp/", "-m", "not integration", "-v"],
            "description": "Fast unit tests for individual components"
        },
        {
            "name": "Integration Tests", 
            "command": ["python", "-m", "pytest", "tests/cli/v3/nlp/test_integration.py", "-v"],
            "description": "End-to-end integration tests"
        },
        {
            "name": "All NLP Tests with Coverage",
            "command": [
                "python", "-m", "pytest", 
                "tests/cli/v3/nlp/",
                "--cov=src/agentsmcp/cli/v3/nlp",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov/nlp",
                "--cov-fail-under=80",
                "-v"
            ],
            "description": "All tests with detailed coverage analysis"
        }
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n📋 {config['name']}")
        print(f"   {config['description']}")
        print("-" * 60)
        
        try:
            result = subprocess.run(
                config["command"],
                capture_output=False,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            results.append({
                "name": config["name"],
                "success": result.returncode == 0,
                "returncode": result.returncode
            })
            
            if result.returncode == 0:
                print(f"✅ {config['name']} PASSED")
            else:
                print(f"❌ {config['name']} FAILED (exit code: {result.returncode})")
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {config['name']} TIMED OUT")
            results.append({
                "name": config["name"], 
                "success": False,
                "returncode": -1
            })
        except Exception as e:
            print(f"💥 {config['name']} ERROR: {e}")
            results.append({
                "name": config["name"],
                "success": False, 
                "returncode": -2
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r["success"])
    
    for result in results:
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"{status} - {result['name']}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} test suites passed")
    
    if passed_tests == total_tests:
        print("🎉 All NLP tests passed!")
        return 0
    else:
        print("🚨 Some tests failed - see details above")
        return 1


def main():
    """Main entry point."""
    try:
        return run_tests()
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
        return 130
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())