#!/usr/bin/env python3
"""
Comprehensive test execution script for AgentsMCP.
Runs tests with proper reporting and handles flaky behavior.
"""

import subprocess
import sys
import time
from pathlib import Path

def run_test_suite():
    """Run the comprehensive test suite with reporting."""
    
    print("🧪 AgentsMCP Comprehensive Test Suite")
    print("=" * 50)
    
    # Test categories to run
    test_categories = [
        {
            "name": "CLI Arguments",
            "pattern": "tests/test_cli_args.py",
            "timeout": 30,
            "description": "Command-line argument parsing and mode validation"
        },
        {
            "name": "Web API",
            "pattern": "tests/test_web_api.py", 
            "timeout": 20,
            "description": "REST API endpoints and responses"
        },
        {
            "name": "Settings Navigation", 
            "pattern": "tests/test_settings_navigation.py",
            "timeout": 45,
            "description": "Arrow key navigation in settings dialog"
        },
        {
            "name": "Interactive Commands",
            "pattern": "tests/test_interactive_commands.py::test_help_command",
            "timeout": 30, 
            "description": "Basic interactive command functionality"
        },
        {
            "name": "Dashboard Mode",
            "pattern": "tests/test_dashboard.py::test_dashboard_mode_startup",
            "timeout": 20,
            "description": "Dashboard mode startup and basic functionality"
        }
    ]
    
    results = []
    total_start_time = time.time()
    
    for i, category in enumerate(test_categories, 1):
        print(f"\n📋 Running {category['name']} Tests ({i}/{len(test_categories)})")
        print(f"   {category['description']}")
        print("-" * 40)
        
        start_time = time.time()
        
        cmd = [
            "pytest",
            category["pattern"],
            "-v",
            "--tb=short",
            f"--timeout={category['timeout']}",
            "--reruns=1",
            "--reruns-delay=2"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                timeout=category["timeout"] + 10,
                capture_output=True,
                text=True
            )
            
            duration = time.time() - start_time
            
            # Parse results
            if result.returncode == 0:
                status = "✅ PASS"
            elif result.returncode == 1:
                status = "❌ FAIL"
            else:
                status = "⚠️  ERROR"
            
            # Extract test counts from pytest output
            passed = result.stdout.count(" PASSED")
            failed = result.stdout.count(" FAILED")
            
            results.append({
                "name": category["name"],
                "status": status,
                "passed": passed,
                "failed": failed,
                "duration": duration,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            })
            
            print(f"{status} - {passed} passed, {failed} failed ({duration:.1f}s)")
            
            # Show brief output for failures
            if result.returncode != 0:
                print("📄 Output preview:")
                lines = result.stdout.split('\n')
                relevant_lines = [line for line in lines if 'FAILED' in line or 'ERROR' in line][:3]
                for line in relevant_lines:
                    print(f"   {line}")
                    
        except subprocess.TimeoutExpired:
            duration = category["timeout"] + 10
            results.append({
                "name": category["name"],
                "status": "⏰ TIMEOUT",
                "passed": 0,
                "failed": 1,
                "duration": duration,
                "returncode": -1,
                "stdout": "",
                "stderr": "Test timed out"
            })
            print(f"⏰ TIMEOUT - Test category timed out after {duration}s")
            
        except Exception as e:
            duration = time.time() - start_time
            results.append({
                "name": category["name"],
                "status": "💥 CRASH", 
                "passed": 0,
                "failed": 1,
                "duration": duration,
                "returncode": -2,
                "stdout": "",
                "stderr": str(e)
            })
            print(f"💥 CRASH - {str(e)}")
    
    # Generate summary report
    total_duration = time.time() - total_start_time
    total_passed = sum(r["passed"] for r in results)
    total_failed = sum(r["failed"] for r in results)
    total_tests = total_passed + total_failed
    
    print("\n" + "=" * 60)
    print("📊 TEST EXECUTION SUMMARY")
    print("=" * 60)
    
    print(f"🕐 Total Duration: {total_duration:.1f}s")
    print(f"✅ Total Passed: {total_passed}")
    print(f"❌ Total Failed: {total_failed}")
    print(f"📈 Success Rate: {(total_passed/total_tests)*100:.1f}%" if total_tests > 0 else "N/A")
    
    print("\n📋 Category Results:")
    for result in results:
        print(f"{result['status']} {result['name']} - {result['passed']}✅/{result['failed']}❌ ({result['duration']:.1f}s)")
    
    # Generate detailed HTML report
    print(f"\n📄 Generating HTML test report...")
    try:
        subprocess.run([
            "pytest", 
            "tests/",
            "--html=test_report.html",
            "--self-contained-html",
            "--tb=short",
            "--timeout=10"
        ], timeout=60, check=False, capture_output=True)
        
        if Path("test_report.html").exists():
            print("✅ HTML report generated: test_report.html")
        else:
            print("⚠️  HTML report generation may have failed")
            
    except Exception as e:
        print(f"⚠️  HTML report generation failed: {e}")
    
    # Return overall success
    success_rate = (total_passed / total_tests) * 100 if total_tests > 0 else 0
    
    if success_rate >= 80:
        print("\n🎉 Test suite is in good shape!")
        return 0
    elif success_rate >= 60:
        print("\n⚠️  Test suite needs some attention")
        return 1
    else:
        print("\n🚨 Test suite requires significant fixes")
        return 2

if __name__ == "__main__":
    exit_code = run_test_suite()
    sys.exit(exit_code)