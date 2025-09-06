"""
Git-Aware Verification System

This module provides verification capabilities that ensure AgentsMCP's claims 
about file operations, commits, and repository changes are actually true.
Addresses the critical issue where AgentsMCP was writing to staging directories
and falsely claiming changes were committed to the main branch.
"""

import os
import subprocess
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class FileOperation:
    """Represents a file operation that should be verified."""
    operation_type: str  # "create", "modify", "delete"
    file_path: str
    expected_content_hash: Optional[str] = None
    claimed_by: str = "unknown"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CommitVerification:
    """Represents a commit verification result."""
    commit_hash: str
    commit_message: str
    files_changed: List[str]
    is_in_main_branch: bool
    is_pushed_to_remote: bool
    verification_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class VerificationResult:
    """Result of a verification operation."""
    success: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    false_claims: List[str] = field(default_factory=list)
    missing_operations: List[str] = field(default_factory=list)


class GitAwareVerifier:
    """
    Verifies that claimed file operations and git commits actually exist
    and are properly tracked in the git repository.
    """
    
    def __init__(self, repo_path: Optional[str] = None):
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()
        self.verified_operations: List[FileOperation] = []
        self.verification_log: List[VerificationResult] = []
        
        # Ensure we're in a git repository
        if not self._is_git_repo():
            raise ValueError(f"Path {self.repo_path} is not a git repository")
    
    def _is_git_repo(self) -> bool:
        """Check if the current path is a git repository."""
        return (self.repo_path / ".git").exists()
    
    def _run_git_command(self, command: List[str]) -> Tuple[str, str, int]:
        """Run a git command and return stdout, stderr, return_code."""
        try:
            result = subprocess.run(
                ["git"] + command,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.stdout.strip(), result.stderr.strip(), result.returncode
        except subprocess.TimeoutExpired:
            return "", "Command timed out", 1
        except Exception as e:
            return "", str(e), 1
    
    def _get_file_hash(self, file_path: str) -> Optional[str]:
        """Get SHA-256 hash of file content."""
        try:
            full_path = self.repo_path / file_path
            if not full_path.exists():
                return None
            
            with open(full_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Error getting hash for {file_path}: {e}")
            return None
    
    def verify_file_exists(self, file_path: str, claimed_by: str = "system") -> VerificationResult:
        """Verify that a claimed file actually exists."""
        full_path = self.repo_path / file_path
        
        if not full_path.exists():
            result = VerificationResult(
                success=False,
                message=f"File {file_path} does not exist",
                false_claims=[f"{claimed_by} claimed to create/modify {file_path} but file missing"]
            )
            self.verification_log.append(result)
            return result
        
        return VerificationResult(
            success=True,
            message=f"File {file_path} exists",
            details={"file_size": full_path.stat().st_size}
        )
    
    def verify_file_tracked_by_git(self, file_path: str, claimed_by: str = "system") -> VerificationResult:
        """Verify that a file is tracked by git (not just in staging/temp)."""
        stdout, stderr, returncode = self._run_git_command(["ls-files", "--error-unmatch", file_path])
        
        if returncode != 0:
            # Check if file exists but is untracked
            if (self.repo_path / file_path).exists():
                result = VerificationResult(
                    success=False,
                    message=f"File {file_path} exists but is not tracked by git",
                    false_claims=[f"{claimed_by} claimed to commit {file_path} but it's not tracked"],
                    details={"git_error": stderr}
                )
            else:
                result = VerificationResult(
                    success=False,
                    message=f"File {file_path} neither exists nor is tracked",
                    false_claims=[f"{claimed_by} claimed {file_path} but file is missing entirely"]
                )
            
            self.verification_log.append(result)
            return result
        
        return VerificationResult(
            success=True,
            message=f"File {file_path} is properly tracked by git"
        )
    
    def verify_recent_commit_contains_file(self, file_path: str, max_commits: int = 10) -> VerificationResult:
        """Verify that a file was included in a recent commit."""
        stdout, stderr, returncode = self._run_git_command([
            "log", f"-{max_commits}", "--name-only", "--pretty=format:%H|%s", file_path
        ])
        
        if returncode != 0 or not stdout:
            return VerificationResult(
                success=False,
                message=f"File {file_path} not found in last {max_commits} commits",
                details={"git_error": stderr}
            )
        
        commits = []
        for line in stdout.split('\n'):
            if '|' in line:
                commit_hash, commit_msg = line.split('|', 1)
                commits.append({"hash": commit_hash, "message": commit_msg})
        
        return VerificationResult(
            success=True,
            message=f"File {file_path} found in recent commits",
            details={"commits": commits}
        )
    
    def verify_commit_in_main_branch(self, commit_hash: str) -> VerificationResult:
        """Verify that a commit exists in the main branch."""
        # First check if commit exists at all
        stdout, stderr, returncode = self._run_git_command(["cat-file", "-e", commit_hash])
        if returncode != 0:
            return VerificationResult(
                success=False,
                message=f"Commit {commit_hash} does not exist",
                details={"git_error": stderr}
            )
        
        # Check if commit is in main branch
        stdout, stderr, returncode = self._run_git_command([
            "branch", "--contains", commit_hash
        ])
        
        is_in_main = "main" in stdout or "master" in stdout
        
        return VerificationResult(
            success=is_in_main,
            message=f"Commit {commit_hash} {'is' if is_in_main else 'is not'} in main branch",
            details={"branches": stdout.split('\n')}
        )
    
    def verify_commit_pushed_to_remote(self, commit_hash: str, remote: str = "origin") -> VerificationResult:
        """Verify that a commit has been pushed to remote repository."""
        stdout, stderr, returncode = self._run_git_command([
            "branch", "-r", "--contains", commit_hash
        ])
        
        if returncode != 0:
            return VerificationResult(
                success=False,
                message=f"Could not check remote branches for commit {commit_hash}",
                details={"git_error": stderr}
            )
        
        remote_branches = stdout.split('\n') if stdout else []
        is_pushed = any(remote in branch for branch in remote_branches)
        
        return VerificationResult(
            success=is_pushed,
            message=f"Commit {commit_hash} {'has been' if is_pushed else 'has not been'} pushed to remote",
            details={"remote_branches": remote_branches}
        )
    
    def verify_documentation_updates_complete(self, claimed_files: List[str]) -> VerificationResult:
        """Comprehensive verification of documentation updates."""
        false_claims = []
        missing_operations = []
        details = {}
        
        for file_path in claimed_files:
            # Check file exists
            exists_result = self.verify_file_exists(file_path)
            if not exists_result.success:
                false_claims.extend(exists_result.false_claims)
            
            # Check file is tracked
            tracked_result = self.verify_file_tracked_by_git(file_path)
            if not tracked_result.success:
                false_claims.extend(tracked_result.false_claims)
            
            # Check file in recent commits
            commit_result = self.verify_recent_commit_contains_file(file_path)
            if not commit_result.success:
                missing_operations.append(f"File {file_path} not committed recently")
            
            details[file_path] = {
                "exists": exists_result.success,
                "tracked": tracked_result.success,
                "committed": commit_result.success,
                "recent_commits": commit_result.details.get("commits", [])
            }
        
        overall_success = len(false_claims) == 0 and len(missing_operations) == 0
        
        return VerificationResult(
            success=overall_success,
            message=f"Documentation verification: {'PASSED' if overall_success else 'FAILED'}",
            details=details,
            false_claims=false_claims,
            missing_operations=missing_operations
        )
    
    def get_git_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive git status for verification."""
        status = {}
        
        # Current branch
        stdout, _, _ = self._run_git_command(["branch", "--show-current"])
        status["current_branch"] = stdout
        
        # Modified files
        stdout, _, _ = self._run_git_command(["diff", "--name-only"])
        status["modified_files"] = stdout.split('\n') if stdout else []
        
        # Staged files
        stdout, _, _ = self._run_git_command(["diff", "--cached", "--name-only"])
        status["staged_files"] = stdout.split('\n') if stdout else []
        
        # Untracked files
        stdout, _, _ = self._run_git_command(["ls-files", "--others", "--exclude-standard"])
        status["untracked_files"] = stdout.split('\n') if stdout else []
        
        # Last commit
        stdout, _, _ = self._run_git_command(["log", "-1", "--pretty=format:%H|%s|%ai"])
        if stdout and '|' in stdout:
            parts = stdout.split('|')
            status["last_commit"] = {
                "hash": parts[0],
                "message": parts[1] if len(parts) > 1 else "",
                "date": parts[2] if len(parts) > 2 else ""
            }
        
        # Remote status
        stdout, _, _ = self._run_git_command(["status", "--porcelain", "-b"])
        status["remote_sync"] = "up to date" in stdout or "up-to-date" in stdout
        
        return status
    
    def create_verification_report(self) -> Dict[str, Any]:
        """Create a comprehensive verification report."""
        git_status = self.get_git_status_summary()
        
        total_verifications = len(self.verification_log)
        successful_verifications = sum(1 for v in self.verification_log if v.success)
        
        all_false_claims = []
        all_missing_operations = []
        
        for verification in self.verification_log:
            all_false_claims.extend(verification.false_claims)
            all_missing_operations.extend(verification.missing_operations)
        
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "repository_path": str(self.repo_path),
            "git_status": git_status,
            "verification_summary": {
                "total_verifications": total_verifications,
                "successful": successful_verifications,
                "failed": total_verifications - successful_verifications,
                "success_rate": successful_verifications / max(1, total_verifications)
            },
            "issues": {
                "false_claims": all_false_claims,
                "missing_operations": all_missing_operations,
                "total_issues": len(all_false_claims) + len(all_missing_operations)
            },
            "detailed_results": [
                {
                    "success": v.success,
                    "message": v.message,
                    "details": v.details,
                    "false_claims": v.false_claims,
                    "missing_operations": v.missing_operations
                }
                for v in self.verification_log
            ]
        }
        
        return report
    
    def save_verification_report(self, output_path: Optional[str] = None) -> str:
        """Save verification report to file."""
        report = self.create_verification_report()
        
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"verification_report_{timestamp}.json"
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Verification report saved to {output_path}")
        return output_path


def verify_agentsmcp_claims(
    claimed_files: List[str], 
    claimed_commits: List[str] = None,
    repo_path: str = None
) -> VerificationResult:
    """
    High-level function to verify AgentsMCP's claims about file operations and commits.
    
    Args:
        claimed_files: List of files claimed to be created/modified
        claimed_commits: List of commit hashes claimed to exist
        repo_path: Path to repository (defaults to current directory)
    
    Returns:
        Comprehensive verification result
    """
    verifier = GitAwareVerifier(repo_path)
    
    # Verify documentation updates
    doc_result = verifier.verify_documentation_updates_complete(claimed_files)
    
    # Verify commits if provided
    if claimed_commits:
        for commit_hash in claimed_commits:
            commit_in_main = verifier.verify_commit_in_main_branch(commit_hash)
            if not commit_in_main.success:
                doc_result.false_claims.append(f"Commit {commit_hash} not in main branch")
            
            commit_pushed = verifier.verify_commit_pushed_to_remote(commit_hash)
            if not commit_pushed.success:
                doc_result.missing_operations.append(f"Commit {commit_hash} not pushed to remote")
    
    return doc_result