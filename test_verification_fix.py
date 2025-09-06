#!/usr/bin/env python3
"""
Test script to verify that the AgentsMCP verification fix works correctly.

This script tests the git-aware verification system to ensure AgentsMCP
can no longer make false claims about file operations and commits.
"""

import os
import sys
import asyncio
import tempfile
import shutil
from pathlib import Path

# Add the AgentsMCP source to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agentsmcp.verification import GitAwareVerifier, verify_agentsmcp_claims


async def test_basic_file_verification():
    """Test basic file existence verification."""
    print("=== Testing Basic File Verification ===")
    
    verifier = GitAwareVerifier()
    
    # Test existing file
    result = verifier.verify_file_exists("README.md", "test_system")
    print(f"✓ README.md exists: {result.success}")
    assert result.success, "README.md should exist"
    
    # Test non-existent file  
    result = verifier.verify_file_exists("nonexistent_file.txt", "test_system")
    print(f"✓ Non-existent file correctly detected: {not result.success}")
    assert not result.success, "Non-existent file should be detected"
    assert len(result.false_claims) > 0, "Should have false claims"
    
    print("Basic file verification tests PASSED\n")


async def test_git_tracking_verification():
    """Test git tracking verification."""
    print("=== Testing Git Tracking Verification ===")
    
    verifier = GitAwareVerifier()
    
    # Test tracked file
    result = verifier.verify_file_tracked_by_git("README.md", "test_system")
    print(f"✓ README.md is tracked: {result.success}")
    assert result.success, "README.md should be tracked by git"
    
    # Create a temporary file that won't be tracked
    temp_file = "temp_untracked_test.txt"
    with open(temp_file, 'w') as f:
        f.write("This is a test file")
    
    try:
        result = verifier.verify_file_tracked_by_git(temp_file, "test_system")
        print(f"✓ Untracked file correctly detected: {not result.success}")
        assert not result.success, "Untracked file should be detected"
        assert len(result.false_claims) > 0, "Should have false claims about untracked file"
    finally:
        os.remove(temp_file)
    
    print("Git tracking verification tests PASSED\n")


async def test_commit_verification():
    """Test commit verification."""
    print("=== Testing Commit Verification ===")
    
    verifier = GitAwareVerifier()
    
    # Get the latest commit hash
    git_status = verifier.get_git_status_summary()
    if git_status.get("last_commit"):
        commit_hash = git_status["last_commit"]["hash"]
        
        # Test valid commit
        result = verifier.verify_commit_in_main_branch(commit_hash)
        print(f"✓ Latest commit {commit_hash[:8]} verified: {result.success}")
        assert result.success, f"Latest commit {commit_hash} should be in main branch"
        
        # Test remote push verification
        result = verifier.verify_commit_pushed_to_remote(commit_hash)
        print(f"✓ Remote push check for {commit_hash[:8]}: {result.success}")
        # Note: This might fail in CI or if not pushed yet, which is expected
        
    # Test invalid commit
    fake_commit = "1234567890abcdef1234567890abcdef12345678"
    result = verifier.verify_commit_in_main_branch(fake_commit)
    print(f"✓ Fake commit correctly rejected: {not result.success}")
    assert not result.success, "Fake commit should be rejected"
    
    print("Commit verification tests PASSED\n")


async def test_comprehensive_verification():
    """Test comprehensive documentation verification."""
    print("=== Testing Comprehensive Verification ===")
    
    # Test the P0 and P1 files we just committed
    claimed_files = [
        "README.md",
        "docs/PROVIDERS.md", 
        "docs/CI_MATRIX.md",
        "docs/BENCHMARKS.md",
        "docs/LIVE_DASHBOARD.md"
    ]
    
    result = verify_agentsmcp_claims(claimed_files)
    
    print(f"✓ Comprehensive verification result: {result.success}")
    print(f"✓ False claims detected: {len(result.false_claims)}")
    print(f"✓ Missing operations: {len(result.missing_operations)}")
    
    if result.false_claims:
        print("False claims found:")
        for claim in result.false_claims:
            print(f"  - {claim}")
    
    if result.missing_operations:
        print("Missing operations found:")
        for op in result.missing_operations:
            print(f"  - {op}")
    
    # The result should be successful since we actually committed these files
    assert result.success, f"Verification should pass for committed files. Issues: {result.false_claims + result.missing_operations}"
    
    print("Comprehensive verification tests PASSED\n")


async def test_false_claim_detection():
    """Test detection of false claims (simulate AgentsMCP lying)."""
    print("=== Testing False Claim Detection ===")
    
    # Simulate claiming files that don't exist
    fake_files = [
        "fake_documentation.md",
        "build/staging/fake_readme.md",  # This would be the old bug
        "nonexistent/path/file.txt"
    ]
    
    result = verify_agentsmcp_claims(fake_files)
    
    print(f"✓ False claim detection result: {not result.success}")  # Should fail
    print(f"✓ False claims detected: {len(result.false_claims)}")
    print(f"✓ Missing operations: {len(result.missing_operations)}")
    
    # Should detect these as false claims
    assert not result.success, "Should detect false claims"
    assert len(result.false_claims) > 0 or len(result.missing_operations) > 0, "Should have detected issues"
    
    print("False claim detection tests PASSED\n")


async def test_staging_directory_bug():
    """Test that the staging directory bug is fixed."""
    print("=== Testing Staging Directory Bug Fix ===")
    
    # Create a file in build/staging (simulating the old bug)
    staging_dir = Path("build/staging")
    staging_dir.mkdir(parents=True, exist_ok=True)
    
    fake_staging_file = staging_dir / "fake_improvement.md"
    with open(fake_staging_file, 'w') as f:
        f.write("# Fake Improvement\n\nThis file exists but isn't tracked by git.")
    
    try:
        # Verify that our system correctly identifies this as untracked
        verifier = GitAwareVerifier()
        result = verifier.verify_file_tracked_by_git(str(fake_staging_file), "AgentsMCP")
        
        print(f"✓ Staging file exists: {fake_staging_file.exists()}")
        print(f"✓ Staging file correctly identified as untracked: {not result.success}")
        print(f"✓ False claims detected: {len(result.false_claims)}")
        
        assert not result.success, "Staging file should be identified as untracked"
        assert len(result.false_claims) > 0, "Should generate false claim about untracked file"
        
        # Verify the false claim message is informative
        claim_message = result.false_claims[0] if result.false_claims else ""
        assert "not tracked" in claim_message.lower(), "Error message should mention tracking issue"
        
    finally:
        # Clean up
        if fake_staging_file.exists():
            fake_staging_file.unlink()
    
    print("Staging directory bug fix tests PASSED\n")


async def run_all_tests():
    """Run all verification tests."""
    print("🔍 Running AgentsMCP Verification Fix Tests\n")
    
    try:
        await test_basic_file_verification()
        await test_git_tracking_verification() 
        await test_commit_verification()
        await test_comprehensive_verification()
        await test_false_claim_detection()
        await test_staging_directory_bug()
        
        print("🎉 ALL VERIFICATION TESTS PASSED!")
        print("\n✅ The AgentsMCP verification fix is working correctly!")
        print("✅ False claims about file operations will now be detected")
        print("✅ Staging directory bug has been fixed")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 UNEXPECTED ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_all_tests())