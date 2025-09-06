"""
Verification Module for AgentsMCP

This module provides verification capabilities to ensure that claimed operations
actually occur and are properly tracked in the repository. Addresses critical
issues where agents claim to perform operations but don't actually complete them.

The verification enforcer provides strict enforcement with actionable error messages
when verification fails, preventing false claims about completed work.
"""

from .git_aware_verifier import (
    GitAwareVerifier,
    FileOperation,
    CommitVerification,
    VerificationResult,
    verify_agentsmcp_claims
)

from .verification_enforcer import (
    VerificationEnforcer,
    VerificationEnforcementError,
    ImprovementClaim,
    VerificationRequirement,
    enforce_improvement_verification,
    get_verification_enforcer
)

__all__ = [
    'GitAwareVerifier',
    'FileOperation', 
    'CommitVerification',
    'VerificationResult',
    'verify_agentsmcp_claims',
    'VerificationEnforcer',
    'VerificationEnforcementError',
    'ImprovementClaim',
    'VerificationRequirement',
    'enforce_improvement_verification',
    'get_verification_enforcer'
]