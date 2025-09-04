"""
Security Manager
~~~~~~~~~~~~~~~~

Provides authentication, authorization, and security policy enforcement for AgentsMCP.
This is currently a stub implementation - real security features need to be implemented.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Set
from enum import Enum

logger = logging.getLogger(__name__)


class SecurityLevel(str, Enum):
    """Security levels for different operations."""
    PUBLIC = "public"
    AUTHENTICATED = "authenticated" 
    AUTHORIZED = "authorized"
    ADMIN = "admin"


class SecurityManager:
    """
    Security manager providing authentication and authorization services.
    
    WARNING: This is currently a stub implementation for development.
    Production deployments MUST implement real security features.
    """
    
    def __init__(self, config: Any | None = None, *, insecure_mode: bool = False):
        """
        Initialize SecurityManager.
        
        Args:
            insecure_mode: If True, allows operation without real security.
                          Should only be used in development.
        """
        self.insecure_mode = insecure_mode
        self._config = config
        self._authenticated_users: Set[str] = set()
        self._user_permissions: Dict[str, Set[str]] = {}
        
        if insecure_mode:
            logger.warning(
                "SecurityManager running in INSECURE mode - "
                "authentication and authorization are disabled!"
            )
        else:
            logger.info("SecurityManager initialized (stub implementation)")
    
    def authenticate(self, token: Optional[str] = None, **kwargs) -> bool:
        """
        Authenticate a user token.
        
        Args:
            token: Authentication token (JWT, API key, etc.)
            **kwargs: Additional authentication parameters
            
        Returns:
            True if authenticated, False otherwise
            
        Raises:
            NotImplementedError: In stub mode when real auth is required
        """
        if self.insecure_mode:
            logger.debug("Authentication bypassed (insecure mode)")
            return True
            
        if not token:
            logger.warning("No authentication token provided")
            return False
            
        # TODO: Implement real authentication
        raise NotImplementedError(
            "Real authentication not implemented yet. "
            "Set insecure_mode=True for development or implement "
            "JWT/OAuth/API key validation here."
        )
    
    def authorize(self, user_id: str, resource: str, action: str) -> bool:
        """
        Check if user is authorized to perform action on resource.
        
        Args:
            user_id: User identifier
            resource: Resource being accessed
            action: Action being performed (read, write, execute, etc.)
            
        Returns:
            True if authorized, False otherwise
            
        Raises:
            NotImplementedError: In stub mode when real authz is required
        """
        if self.insecure_mode:
            logger.debug(f"Authorization bypassed for {user_id} -> {resource}:{action}")
            return True
            
        # TODO: Implement real authorization (RBAC, ABAC, etc.)
        raise NotImplementedError(
            "Real authorization not implemented yet. "
            "Set insecure_mode=True for development or implement "
            "RBAC/ABAC policy checking here."
        )
    
    def get_public_key_pem(self) -> str:
        """
        Get the public key in PEM format for token verification.
        
        Returns:
            Public key as PEM string
            
        Raises:
            NotImplementedError: Always in stub implementation
        """
        if self.insecure_mode:
            return "-----BEGIN PUBLIC KEY-----\n(INSECURE STUB KEY)\n-----END PUBLIC KEY-----"
            
        raise NotImplementedError(
            "Public key management not implemented yet. "
            "Implement cryptographic key management here."
        )
    
    def create_token(self, user_id: str, permissions: Optional[Set[str]] = None) -> str:
        """
        Create an authentication token for a user.
        
        Args:
            user_id: User identifier
            permissions: Set of permissions to embed in token
            
        Returns:
            Signed authentication token
            
        Raises:
            NotImplementedError: Always in stub implementation
        """
        if self.insecure_mode:
            return f"INSECURE_TOKEN_{user_id}"
            
        raise NotImplementedError(
            "Token creation not implemented yet. "
            "Implement JWT signing or other token generation here."
        )
    
    def has_permission(self, user_id: str, permission: str) -> bool:
        """
        Check if user has a specific permission.
        
        Args:
            user_id: User identifier
            permission: Permission to check
            
        Returns:
            True if user has permission, False otherwise
        """
        if self.insecure_mode:
            return True
            
        user_perms = self._user_permissions.get(user_id, set())
        return permission in user_perms
    
    def grant_permission(self, user_id: str, permission: str) -> None:
        """Grant a permission to a user (stub implementation)."""
        if user_id not in self._user_permissions:
            self._user_permissions[user_id] = set()
        self._user_permissions[user_id].add(permission)
        logger.debug(f"Granted permission {permission} to {user_id}")
    
    def revoke_permission(self, user_id: str, permission: str) -> None:
        """Revoke a permission from a user (stub implementation)."""
        if user_id in self._user_permissions:
            self._user_permissions[user_id].discard(permission)
        logger.debug(f"Revoked permission {permission} from {user_id}")


# Default instance for convenience
# SECURITY: Default to secure mode - override only for development
default_security_manager = SecurityManager(insecure_mode=False)


def create_security_manager(insecure_mode: bool = False) -> SecurityManager:
    """
    Create a SecurityManager with explicit security mode configuration.
    
    Args:
        insecure_mode: If True, disable authentication and authorization (DEVELOPMENT ONLY)
        
    Returns:
        Configured SecurityManager instance
        
    Warning:
        Setting insecure_mode=True disables ALL security controls.
        This should NEVER be used in production.
    """
    if insecure_mode:
        logger.critical(
            "⚠️  SECURITY WARNING: Running in INSECURE mode! ⚠️\n"
            "   • Authentication is DISABLED\n" 
            "   • Authorization is DISABLED\n"
            "   • ALL security controls are BYPASSED\n"
            "   • This mode is for DEVELOPMENT ONLY\n"
            "   • NEVER use this in production!"
        )
    
    return SecurityManager(insecure_mode=insecure_mode)


def get_security_manager() -> SecurityManager:
    """Get the default security manager instance."""
    return default_security_manager


__all__ = ["SecurityManager", "SecurityLevel", "get_security_manager", "default_security_manager", "create_security_manager"]
