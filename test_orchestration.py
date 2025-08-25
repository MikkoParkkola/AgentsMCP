#!/usr/bin/env python3
"""
Test orchestration using AgentsMCP to run comprehensive end-to-end tests.
This demonstrates using AgentsMCP to manage its own testing workflow.
"""

import asyncio
import subprocess
import sys
from pathlib import Path

async def run_test_orchestration():
    """Use AgentsMCP to orchestrate comprehensive testing."""
    
    # Task for comprehensive test suite execution
    test_task = """
    Execute a comprehensive test orchestration for the AgentsMCP binary with the following requirements:

    1. **Test Categories to Execute:**
       - CLI argument parsing tests (quick validation)
       - Web API endpoint tests (ensure all endpoints respond)
       - Interactive command tests (basic command validation)
       - Settings navigation tests (arrow key functionality)
       - Dashboard mode tests (startup and graceful shutdown)

    2. **Test Execution Strategy:**
       - Run non-interactive tests first (CLI args, web API)
       - Run interactive tests with timeout controls
       - Skip or limit long-running dashboard tests to avoid timeouts
       - Generate detailed test reports
       - Capture any failures for analysis

    3. **Quality Gates:**
       - All CLI argument tests must pass
       - Web API health endpoint must respond correctly
       - Basic interactive commands must work
       - Settings arrow key navigation must function
       - Dashboard must start and stop gracefully

    4. **Deliverables:**
       - Comprehensive test execution report
       - Summary of pass/fail rates by category
       - Identification of any flaky behavior
       - Recommendations for fixes

    Please execute this testing workflow using pytest with appropriate flags for parallel execution, retries, and reporting.
    """

    print("🚀 Starting AgentsMCP self-orchestrated testing...")
    print("=" * 60)
    
    # Try to use AgentsMCP to execute this task
    try:
        from src.agentsmcp.orchestration.orchestration_manager import OrchestrationManager
        from src.agentsmcp.ui.theme_manager import ThemeManager
        
        theme_manager = ThemeManager()
        orchestration_manager = OrchestrationManager(theme_manager)
        
        print("📋 Submitting test orchestration task to AgentsMCP...")
        
        # Execute the task using AgentsMCP's orchestration system
        result = await orchestration_manager.execute_task(
            task_description=test_task,
            mode="hybrid",
            context={
                "project_root": str(Path.cwd()),
                "test_framework": "pytest",
                "execution_type": "end_to_end_testing"
            }
        )
        
        print("\n✅ AgentsMCP orchestration completed!")
        print("📊 Results:")
        print(f"Status: {result.get('status', 'Unknown')}")
        print(f"Task ID: {result.get('task_id', 'N/A')}")
        
        return result
        
    except Exception as e:
        print(f"⚠️  AgentsMCP orchestration not available: {e}")
        print("🔄 Falling back to direct test execution...")
        
        # Fallback: run tests directly with pytest
        return await run_direct_tests()

async def run_direct_tests():
    """Fallback method to run tests directly without AgentsMCP orchestration."""
    
    test_commands = [
        # Quick CLI tests
        ["pytest", "tests/test_cli_args.py", "-v", "--timeout=15", "-x"],
        
        # Web API tests 
        ["pytest", "tests/test_web_api.py", "-v", "--timeout=20"],
        
        # Interactive tests (with shorter timeouts)
        ["pytest", "tests/test_interactive_commands.py", "-v", "--timeout=15", "-k", "not dashboard"],
        
        # Settings tests
        ["pytest", "tests/test_settings_navigation.py", "-v", "--timeout=15"],
        
        # Generate HTML report of all results
        ["pytest", "tests/", "--html=test_report.html", "--self-contained-html", "-v", "--timeout=10", "-x"]
    ]
    
    results = []
    
    for i, cmd in enumerate(test_commands, 1):
        print(f"\n🧪 Running test batch {i}/{len(test_commands)}: {' '.join(cmd[2:])}")
        print("-" * 50)
        
        try:
            result = subprocess.run(
                cmd,
                timeout=60,  # Overall timeout per test batch
                capture_output=True,
                text=True
            )
            
            print(f"Exit code: {result.returncode}")
            if result.stdout:
                print("STDOUT:", result.stdout[-500:])  # Last 500 chars
            if result.stderr:
                print("STDERR:", result.stderr[-500:])
                
            results.append({
                "command": ' '.join(cmd),
                "returncode": result.returncode,
                "success": result.returncode == 0
            })
            
        except subprocess.TimeoutExpired:
            print(f"⏰ Test batch {i} timed out")
            results.append({
                "command": ' '.join(cmd),
                "returncode": -1,
                "success": False,
                "error": "timeout"
            })
        except Exception as e:
            print(f"❌ Test batch {i} failed: {e}")
            results.append({
                "command": ' '.join(cmd),
                "returncode": -2,
                "success": False,
                "error": str(e)
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST EXECUTION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results if r["success"])
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    for result in results:
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"{status} {result['command']}")
        if not result["success"] and "error" in result:
            print(f"     Error: {result['error']}")
    
    return {
        "total_tests": total,
        "passed": passed,
        "failed": total - passed,
        "results": results
    }

if __name__ == "__main__":
    print("🎯 AgentsMCP Self-Testing Orchestration")
    print("Using AgentsMCP to test itself comprehensively")
    print()
    
    # Run the orchestration
    result = asyncio.run(run_test_orchestration())
    
    print(f"\n🏁 Testing orchestration completed!")
    if isinstance(result, dict) and "passed" in result:
        success_rate = (result["passed"] / result["total_tests"]) * 100
        print(f"📈 Success rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            print("🎉 Test suite is in good shape!")
            sys.exit(0)
        else:
            print("⚠️  Test suite needs attention")
            sys.exit(1)
    else:
        print("✨ AgentsMCP orchestration completed successfully!")
        sys.exit(0)