"""
Verification Module for AgentsMCP

This module provides verification capabilities to ensure that claimed operations
actually occur and are properly tracked in the repository. Addresses critical
issues where agents claim to perform operations but don't actually complete them.
"""

from .git_aware_verifier import (
    GitAwareVerifier,
    FileOperation,
    CommitVerification,
    VerificationResult,
    verify_agentsmcp_claims
)

__all__ = [
    'GitAwareVerifier',
    'FileOperation', 
    'CommitVerification',
    'VerificationResult',
    'verify_agentsmcp_claims'
]