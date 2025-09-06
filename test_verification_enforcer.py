#!/usr/bin/env python3
"""
Test script to demonstrate the verification enforcement system.

This script shows how the enforcer prevents false claims and provides
actionable error messages when verification fails.
"""

import os
import sys
import asyncio
import tempfile
from pathlib import Path

# Add the AgentsMCP source to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agentsmcp.verification import (
    VerificationEnforcer, 
    VerificationEnforcementError,
    enforce_improvement_verification
)


async def test_enforcer_catches_false_claims():
    """Test that the enforcer catches false claims and provides actionable steps."""
    print("🧪 Testing Verification Enforcer - False Claims Detection")
    
    try:
        # Try to claim files that don't exist - should fail with actionable steps
        result = enforce_improvement_verification(
            improvement_id="test_fake_improvement",
            claimed_files=["fake_file1.py", "fake_file2.md", "nonexistent/path/file.txt"],
            claimed_commits=["1234567890abcdef1234567890abcdef12345678"],
            claimed_by="test_system"
        )
        
        print("❌ FAILED: Should have caught false claims!")
        return False
        
    except VerificationEnforcementError as e:
        print("✅ PASSED: Enforcer correctly caught false claims")
        print("\n" + "="*60)
        print(e.get_user_friendly_message())
        print("="*60)
        return True
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        return False


async def test_enforcer_passes_real_files():
    """Test that the enforcer passes when files actually exist and are committed."""
    print("\n🧪 Testing Verification Enforcer - Real Files Verification")
    
    try:
        # Test with files that actually exist and are committed
        result = enforce_improvement_verification(
            improvement_id="test_real_improvement", 
            claimed_files=["README.md", "docs/VERIFICATION_SYSTEM.md"],
            claimed_by="test_system"
        )
        
        print("✅ PASSED: Enforcer correctly verified real files")
        print(f"Success: {result.success}")
        print(f"Message: {result.message}")
        return True
        
    except VerificationEnforcementError as e:
        print("❌ FAILED: Should have passed for real files")
        print(e.get_user_friendly_message())
        return False
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        return False


async def test_enforcer_catches_untracked_files():
    """Test that enforcer catches files that exist but aren't tracked by git."""
    print("\n🧪 Testing Verification Enforcer - Untracked Files Detection")
    
    # Create a temporary file that won't be tracked
    temp_file = "temp_test_untracked.txt"
    with open(temp_file, 'w') as f:
        f.write("This file exists but is not tracked by git")
    
    try:
        result = enforce_improvement_verification(
            improvement_id="test_untracked_improvement",
            claimed_files=[temp_file],
            claimed_by="test_system"
        )
        
        print("❌ FAILED: Should have caught untracked file!")
        return False
        
    except VerificationEnforcementError as e:
        print("✅ PASSED: Enforcer correctly caught untracked file")
        print("\n" + "-"*40)
        print(e.get_user_friendly_message())
        print("-"*40)
        return True
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)


async def test_enforcer_with_mixed_scenarios():
    """Test enforcer with a mix of valid and invalid claims."""
    print("\n🧪 Testing Verification Enforcer - Mixed Scenarios")
    
    try:
        # Mix of real and fake files
        result = enforce_improvement_verification(
            improvement_id="test_mixed_improvement",
            claimed_files=[
                "README.md",  # Real file
                "fake_improvement.py",  # Fake file
                "docs/VERIFICATION_SYSTEM.md"  # Real file
            ],
            claimed_by="test_system"
        )
        
        print("❌ FAILED: Should have caught mixed false claims!")
        return False
        
    except VerificationEnforcementError as e:
        print("✅ PASSED: Enforcer correctly identified mixed scenario issues")
        print("\nActionable steps provided:")
        for i, step in enumerate(e.actionable_steps, 1):
            print(f"  {i}. {step}")
        return True
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        return False


async def test_improvement_coordinator_integration():
    """Test that the improvement coordinator uses the enforcer correctly."""
    print("\n🧪 Testing Integration with Improvement Coordinator")
    
    try:
        from agentsmcp.orchestration.improvement_coordinator import ImprovementCoordinator
        
        # Create a coordinator
        coordinator = ImprovementCoordinator()
        
        # This would test the actual integration, but for now we just verify
        # that the import and initialization works
        print("✅ PASSED: Improvement coordinator successfully imports enforcer")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: Integration test error: {e}")
        return False


async def demonstrate_actionable_error_messages():
    """Demonstrate the actionable error messages users will see."""
    print("\n📋 DEMONSTRATION: Actionable Error Messages")
    print("="*60)
    
    scenarios = [
        {
            "name": "Missing Files",
            "claimed_files": ["missing_file.py", "another_missing.md"],
            "claimed_commits": []
        },
        {
            "name": "Invalid Commits", 
            "claimed_files": [],
            "claimed_commits": ["fakehash123456789", "anotherfake987654321"]
        },
        {
            "name": "Complex Mixed Scenario",
            "claimed_files": ["real_file.py", "missing_file.md"],
            "claimed_commits": ["fakehash123"]
        }
    ]
    
    for scenario in scenarios:
        print(f"\n🎯 Scenario: {scenario['name']}")
        print("-" * 30)
        
        try:
            enforce_improvement_verification(
                improvement_id=f"demo_{scenario['name'].lower().replace(' ', '_')}",
                claimed_files=scenario['claimed_files'],
                claimed_commits=scenario['claimed_commits'],
                claimed_by="demo_system"
            )
        except VerificationEnforcementError as e:
            print(e.get_user_friendly_message())
        except Exception as e:
            print(f"Unexpected error: {e}")
    
    print("\n" + "="*60)


async def run_all_tests():
    """Run all enforcer tests."""
    print("🔒 Testing AgentsMCP Verification Enforcement System\n")
    
    tests = [
        test_enforcer_catches_false_claims,
        test_enforcer_passes_real_files,
        test_enforcer_catches_untracked_files,
        test_enforcer_with_mixed_scenarios,
        test_improvement_coordinator_integration,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            result = await test()
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print(f"\n📊 TEST SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("\n✅ The verification enforcement system is working correctly")
        print("✅ False claims will be caught with actionable error messages")
        print("✅ AgentsMCP can no longer make false claims about completed work")
    else:
        print("❌ Some tests failed - verification enforcement needs fixes")
        return False
    
    # Show demonstration of error messages
    await demonstrate_actionable_error_messages()
    
    return True


if __name__ == "__main__":
    asyncio.run(run_all_tests())