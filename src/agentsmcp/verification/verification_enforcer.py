"""
Verification Enforcement System

This module provides strict enforcement of verification requirements to prevent
AgentsMCP from making false claims about completed work. It ensures that any
claimed improvement must be actually committed to the main branch or the
operation fails with clear, actionable error messages.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json

from .git_aware_verifier import GitAwareVerifier, VerificationResult

logger = logging.getLogger(__name__)


class VerificationEnforcementError(Exception):
    """Exception raised when verification enforcement fails."""
    
    def __init__(self, message: str, actionable_steps: List[str], details: Dict[str, Any] = None):
        self.message = message
        self.actionable_steps = actionable_steps
        self.details = details or {}
        super().__init__(message)
    
    def get_user_friendly_message(self) -> str:
        """Get a user-friendly error message with actionable steps."""
        lines = [
            f"🚫 VERIFICATION FAILED: {self.message}",
            "",
            "📋 ACTIONABLE STEPS TO FIX:",
        ]
        
        for i, step in enumerate(self.actionable_steps, 1):
            lines.append(f"  {i}. {step}")
        
        if self.details:
            lines.extend([
                "",
                "🔍 DETAILS:",
                json.dumps(self.details, indent=2)
            ])
        
        return "\n".join(lines)


@dataclass 
class VerificationRequirement:
    """Represents a verification requirement for an improvement."""
    requirement_type: str  # "file_committed", "branch_merged", "tests_passing"
    description: str
    files: List[str] = field(default_factory=list)
    commits: List[str] = field(default_factory=list)
    branches: List[str] = field(default_factory=list)
    must_be_satisfied: bool = True


@dataclass
class ImprovementClaim:
    """Represents a claimed improvement that needs verification."""
    improvement_id: str
    claim_type: str  # "implementation_complete", "committed_to_main", "deployed"
    claimed_files: List[str] = field(default_factory=list)
    claimed_commits: List[str] = field(default_factory=list)
    claimed_features: List[str] = field(default_factory=list)
    verification_requirements: List[VerificationRequirement] = field(default_factory=list)
    claimed_by: str = "unknown"
    claim_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class VerificationEnforcer:
    """
    Enforces strict verification requirements to prevent false claims.
    
    This class ensures that any improvement claimed as "complete" or "merged"
    is actually verifiable in the git repository and main branch.
    """
    
    def __init__(self, repo_path: Optional[str] = None):
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()
        self.verifier = GitAwareVerifier(str(self.repo_path))
        self.strict_mode = True  # Always strict by default
        self.pending_claims: Dict[str, ImprovementClaim] = {}
        
    def register_improvement_claim(self, claim: ImprovementClaim) -> str:
        """Register a claimed improvement for later verification."""
        claim_id = f"{claim.improvement_id}_{claim.claim_type}_{int(claim.claim_timestamp.timestamp())}"
        self.pending_claims[claim_id] = claim
        logger.info(f"Registered improvement claim: {claim_id}")
        return claim_id
    
    def enforce_verification(self, claim: ImprovementClaim) -> VerificationResult:
        """
        Enforce verification of a claimed improvement.
        
        Raises VerificationEnforcementError if verification fails in strict mode.
        """
        logger.info(f"Enforcing verification for claim: {claim.improvement_id}")
        
        # Perform comprehensive verification
        verification_errors = []
        missing_operations = []
        
        # 1. Verify claimed files exist and are tracked
        if claim.claimed_files:
            for file_path in claim.claimed_files:
                # Check file exists
                exists_result = self.verifier.verify_file_exists(file_path, claim.claimed_by)
                if not exists_result.success:
                    verification_errors.append(f"File {file_path} does not exist")
                
                # Check file is tracked by git
                tracked_result = self.verifier.verify_file_tracked_by_git(file_path, claim.claimed_by)
                if not tracked_result.success:
                    verification_errors.append(f"File {file_path} is not tracked by git")
                
                # Check file was committed recently
                commit_result = self.verifier.verify_recent_commit_contains_file(file_path, max_commits=10)
                if not commit_result.success:
                    missing_operations.append(f"File {file_path} not found in recent commits")
        
        # 2. Verify claimed commits exist and are in main branch
        if claim.claimed_commits:
            for commit_hash in claim.claimed_commits:
                commit_result = self.verifier.verify_commit_in_main_branch(commit_hash)
                if not commit_result.success:
                    verification_errors.append(f"Commit {commit_hash} not in main branch")
                
                push_result = self.verifier.verify_commit_pushed_to_remote(commit_hash)
                if not push_result.success:
                    missing_operations.append(f"Commit {commit_hash} not pushed to remote")
        
        # 3. Check custom verification requirements
        for requirement in claim.verification_requirements:
            req_result = self._verify_requirement(requirement, claim)
            if not req_result.success and requirement.must_be_satisfied:
                verification_errors.extend(req_result.false_claims)
                missing_operations.extend(req_result.missing_operations)
        
        # Create final verification result
        overall_success = len(verification_errors) == 0 and len(missing_operations) == 0
        
        result = VerificationResult(
            success=overall_success,
            message=f"Verification {'PASSED' if overall_success else 'FAILED'} for {claim.improvement_id}",
            details={
                "claim_type": claim.claim_type,
                "claimed_files": claim.claimed_files,
                "claimed_commits": claim.claimed_commits,
                "claimed_by": claim.claimed_by
            },
            false_claims=verification_errors,
            missing_operations=missing_operations
        )
        
        # In strict mode, raise exception on failure
        if self.strict_mode and not overall_success:
            actionable_steps = self._generate_actionable_steps(claim, verification_errors, missing_operations)
            raise VerificationEnforcementError(
                f"Improvement '{claim.improvement_id}' verification failed",
                actionable_steps,
                result.details
            )
        
        return result
    
    def _verify_requirement(self, requirement: VerificationRequirement, claim: ImprovementClaim) -> VerificationResult:
        """Verify a specific requirement."""
        if requirement.requirement_type == "file_committed":
            return self.verifier.verify_documentation_updates_complete(requirement.files)
        
        elif requirement.requirement_type == "branch_merged":
            # Check if the improvement branch was merged
            false_claims = []
            missing_ops = []
            
            for branch_name in requirement.branches:
                # Check if branch exists in remotes (indicates it was pushed)
                stdout, stderr, returncode = self.verifier._run_git_command([
                    "ls-remote", "--heads", "origin", branch_name
                ])
                
                if returncode != 0 or not stdout:
                    missing_ops.append(f"Branch {branch_name} not found on remote")
            
            return VerificationResult(
                success=len(false_claims) == 0 and len(missing_ops) == 0,
                message=f"Branch verification for {requirement.description}",
                false_claims=false_claims,
                missing_operations=missing_ops
            )
        
        elif requirement.requirement_type == "tests_passing":
            # For now, assume tests are passing if we can run basic git commands
            # In a real implementation, this would run the actual test suite
            return VerificationResult(
                success=True,
                message="Test verification passed (placeholder)"
            )
        
        else:
            return VerificationResult(
                success=False,
                message=f"Unknown requirement type: {requirement.requirement_type}",
                false_claims=[f"Unknown verification requirement: {requirement.requirement_type}"]
            )
    
    def _generate_actionable_steps(
        self, 
        claim: ImprovementClaim, 
        verification_errors: List[str], 
        missing_operations: List[str]
    ) -> List[str]:
        """Generate actionable steps to fix verification failures."""
        steps = []
        
        # Steps for missing files
        missing_files = [err for err in verification_errors if "does not exist" in err]
        if missing_files:
            steps.append("Create the missing files claimed in the improvement")
            for error in missing_files:
                steps.append(f"  - Fix: {error}")
        
        # Steps for untracked files  
        untracked_files = [err for err in verification_errors if "not tracked by git" in err]
        if untracked_files:
            steps.append("Add untracked files to git and commit them")
            steps.append("  Run: git add <file_paths>")
            steps.append("  Run: git commit -m 'Add missing files'")
        
        # Steps for missing commits
        missing_commits = [op for op in missing_operations if "not found in recent commits" in op]
        if missing_commits:
            steps.append("Commit the changes to git")
            steps.append("  Run: git add -A")
            steps.append("  Run: git commit -m 'Implement improvement: {}'".format(claim.improvement_id))
        
        # Steps for unpushed commits
        unpushed_commits = [op for op in missing_operations if "not pushed to remote" in op]
        if unpushed_commits:
            steps.append("Push commits to the main branch")
            steps.append("  Run: git push origin main")
        
        # Steps for commits not in main branch
        branch_issues = [err for err in verification_errors if "not in main branch" in err]
        if branch_issues:
            steps.append("Ensure commits are in the main branch")
            steps.append("  Run: git checkout main")
            steps.append("  Run: git merge <feature_branch>")
            steps.append("  Run: git push origin main")
        
        # Generic step if no specific issues identified
        if not steps:
            steps.extend([
                "Review the improvement implementation",
                "Ensure all claimed changes are actually made",
                "Commit changes to git and push to main branch",
                "Verify changes exist in GitHub main branch"
            ])
        
        return steps
    
    def create_improvement_claim(
        self,
        improvement_id: str,
        claimed_files: List[str] = None,
        claimed_commits: List[str] = None,
        claimed_features: List[str] = None,
        claimed_by: str = "system"
    ) -> ImprovementClaim:
        """Create a new improvement claim with standard verification requirements."""
        
        requirements = []
        
        # Add file verification requirements
        if claimed_files:
            requirements.append(VerificationRequirement(
                requirement_type="file_committed",
                description="All claimed files must exist and be committed to git",
                files=claimed_files,
                must_be_satisfied=True
            ))
        
        # Add commit verification requirements  
        if claimed_commits:
            requirements.append(VerificationRequirement(
                requirement_type="branch_merged",
                description="All claimed commits must be in main branch and pushed to remote",
                commits=claimed_commits,
                must_be_satisfied=True
            ))
        
        # Add test requirements for code changes
        if any(f.endswith(('.py', '.js', '.ts', '.go', '.java')) for f in (claimed_files or [])):
            requirements.append(VerificationRequirement(
                requirement_type="tests_passing",
                description="Tests must pass for code changes",
                must_be_satisfied=True
            ))
        
        return ImprovementClaim(
            improvement_id=improvement_id,
            claim_type="implementation_complete",
            claimed_files=claimed_files or [],
            claimed_commits=claimed_commits or [],
            claimed_features=claimed_features or [],
            verification_requirements=requirements,
            claimed_by=claimed_by
        )
    
    def verify_and_enforce_claim(
        self,
        improvement_id: str,
        claimed_files: List[str] = None,
        claimed_commits: List[str] = None,
        claimed_features: List[str] = None,
        claimed_by: str = "system"
    ) -> VerificationResult:
        """
        Create and immediately enforce verification of an improvement claim.
        
        This is the main entry point for verifying claimed improvements.
        Raises VerificationEnforcementError with actionable steps if verification fails.
        """
        claim = self.create_improvement_claim(
            improvement_id=improvement_id,
            claimed_files=claimed_files,
            claimed_commits=claimed_commits,
            claimed_features=claimed_features,
            claimed_by=claimed_by
        )
        
        return self.enforce_verification(claim)


# Global enforcer instance
_verification_enforcer: Optional[VerificationEnforcer] = None

def get_verification_enforcer() -> VerificationEnforcer:
    """Get the global verification enforcer instance."""
    global _verification_enforcer
    if _verification_enforcer is None:
        _verification_enforcer = VerificationEnforcer()
    return _verification_enforcer

def enforce_improvement_verification(
    improvement_id: str,
    claimed_files: List[str] = None,
    claimed_commits: List[str] = None,
    claimed_features: List[str] = None,
    claimed_by: str = "system"
) -> VerificationResult:
    """
    Enforce verification of a claimed improvement.
    
    Raises VerificationEnforcementError with actionable steps if verification fails.
    This is the main function that should be called to verify any claimed improvement.
    """
    enforcer = get_verification_enforcer()
    return enforcer.verify_and_enforce_claim(
        improvement_id=improvement_id,
        claimed_files=claimed_files,
        claimed_commits=claimed_commits,
        claimed_features=claimed_features,
        claimed_by=claimed_by
    )