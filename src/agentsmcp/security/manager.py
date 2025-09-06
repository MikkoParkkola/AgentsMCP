"""
Security Manager
~~~~~~~~~~~~~~~~

Production-ready security manager providing authentication, authorization, 
and security policy enforcement for AgentsMCP with fail-secure defaults.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, Optional, Set, Union
from enum import Enum

# SECURITY: Import production security components
from .config import (
    SecurityConfiguration, SecurityMode, ValidationLevel,
    load_security_config_from_env, validate_security_config, create_default_security_config
)
from .jwt import (
    JWTValidator, JWTClaims, JWTValidationError, 
    TokenExpiredError, InvalidTokenError, set_default_validator
)
from .rbac import (
    RBACSystem, ResourceType, Action, PermissionLevel,
    get_default_rbac_system
)
from cryptography.hazmat.primitives import serialization

logger = logging.getLogger(__name__)


class SecurityLevel(str, Enum):
    """Security levels for different operations."""
    PUBLIC = "public"
    AUTHENTICATED = "authenticated" 
    AUTHORIZED = "authorized"
    ADMIN = "admin"


class SecurityManager:
    """
    Production-ready security manager providing authentication and authorization services.
    
    Features:
    - JWT authentication with RSA-256 signature verification
    - Role-based access control (RBAC) with hierarchical roles
    - Fail-secure configuration validation
    - Comprehensive audit logging with correlation IDs
    - Enterprise-grade security policies
    """
    
    def __init__(self, config: Any | None = None, *, insecure_mode: bool = False):
        """
        Initialize SecurityManager with production security features.
        
        Args:
            config: Security configuration (will be created if None)
            insecure_mode: If True, allows operation without real security (DEVELOPMENT ONLY)
        """
        # SECURITY: Initialize with fail-secure defaults
        self.insecure_mode = insecure_mode
        self._security_config = self._initialize_security_config(config, insecure_mode)
        
        # Initialize security components
        self._jwt_validator: Optional[JWTValidator] = None
        self._rbac_system: Optional[RBACSystem] = None
        self._authenticated_sessions: Dict[str, Dict[str, Any]] = {}
        self._failed_attempts: Dict[str, int] = {}
        
        # Performance metrics
        self._auth_requests = 0
        self._auth_successes = 0
        self._auth_failures = 0
        
        # Initialize components based on security mode
        if self.insecure_mode:
            self._initialize_insecure_mode()
        else:
            self._initialize_secure_mode()
        
        # SECURITY: Log security manager initialization
        logger.info(
            "SecurityManager initialized",
            extra={
                "security_mode": self._security_config.security_mode.value,
                "insecure_mode": self.insecure_mode,
                "jwt_enabled": bool(self._jwt_validator),
                "rbac_enabled": bool(self._rbac_system),
                "correlation_id": str(uuid.uuid4())
            }
        )
    
    def authenticate(self, token: Optional[str] = None, **kwargs) -> Union[bool, JWTClaims]:
        """
        Authenticate a user token with production JWT validation.
        
        Args:
            token: JWT authentication token
            **kwargs: Additional authentication parameters
            
        Returns:
            JWTClaims if authenticated successfully, False otherwise
            
        Raises:
            JWTValidationError: For specific authentication failures
        """
        correlation_id = str(uuid.uuid4())
        self._auth_requests += 1
        start_time = time.time()
        
        try:
            # SECURITY: Always validate in secure mode
            if self.insecure_mode:
                logger.debug(
                    "Authentication bypassed (insecure mode)",
                    extra={"correlation_id": correlation_id}
                )
                self._auth_successes += 1
                return True
            
            # Validate token presence
            if not token:
                self._auth_failures += 1
                logger.warning(
                    "Authentication failed: no token provided",
                    extra={"correlation_id": correlation_id}
                )
                return False
            
            # Extract bearer token if needed
            if self._jwt_validator:
                extracted_token = self._jwt_validator.extract_bearer_token(token)
                if extracted_token:
                    token = extracted_token
            
            # SECURITY: Validate JWT token
            if not self._jwt_validator:
                self._auth_failures += 1
                logger.error(
                    "Authentication failed: no JWT validator configured",
                    extra={"correlation_id": correlation_id}
                )
                return False
            
            # Perform JWT validation
            claims = self._jwt_validator.validate_token(token)
            
            # Create or update session
            if claims.sub:
                session_data = {
                    "user_id": claims.sub,
                    "roles": claims.roles,
                    "permissions": claims.permissions,
                    "authenticated_at": time.time(),
                    "correlation_id": correlation_id,
                    "jwt_id": claims.jti
                }
                self._authenticated_sessions[claims.sub] = session_data
                
                # Initialize user in RBAC if not exists
                if self._rbac_system and claims.roles:
                    for role in claims.roles:
                        self._rbac_system.assign_role(claims.sub, role)
            
            self._auth_successes += 1
            
            # SECURITY: Log successful authentication
            processing_time = (time.time() - start_time) * 1000
            logger.info(
                "Authentication successful",
                extra={
                    "correlation_id": correlation_id,
                    "user_id": claims.sub,
                    "jwt_id": claims.jti,
                    "roles_count": len(claims.roles),
                    "processing_time_ms": processing_time
                }
            )
            
            return claims
            
        except (TokenExpiredError, InvalidTokenError, JWTValidationError) as e:
            self._auth_failures += 1
            processing_time = (time.time() - start_time) * 1000
            logger.warning(
                f"Authentication failed: {e}",
                extra={
                    "correlation_id": correlation_id,
                    "error_type": type(e).__name__,
                    "processing_time_ms": processing_time
                }
            )
            return False
            
        except Exception as e:
            self._auth_failures += 1
            processing_time = (time.time() - start_time) * 1000
            logger.error(
                f"Authentication error: {e}",
                extra={
                    "correlation_id": correlation_id,
                    "error_type": type(e).__name__,
                    "processing_time_ms": processing_time
                }
            )
            # SECURITY: Fail secure on unexpected errors
            return False
    
    def authorize(
        self, 
        user_id: str, 
        resource: str, 
        action: str, 
        resource_id: Optional[str] = None,
        required_level: Optional[str] = None
    ) -> bool:
        """
        Check if user is authorized to perform action on resource using RBAC.
        
        Args:
            user_id: User identifier
            resource: Resource type being accessed
            action: Action being performed
            resource_id: Specific resource ID (optional)
            required_level: Minimum permission level required (optional)
            
        Returns:
            True if authorized, False otherwise
        """
        correlation_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # SECURITY: Always validate in secure mode
            if self.insecure_mode:
                logger.debug(
                    f"Authorization bypassed for {user_id} -> {resource}:{action}",
                    extra={"correlation_id": correlation_id}
                )
                return True
            
            # Check if RBAC system is available
            if not self._rbac_system:
                logger.error(
                    "Authorization failed: no RBAC system configured",
                    extra={"correlation_id": correlation_id}
                )
                return False
            
            # Map string parameters to enums
            try:
                resource_type = ResourceType(resource.lower())
                action_enum = Action(action.lower())
                permission_level = PermissionLevel.READ
                
                if required_level:
                    permission_level = PermissionLevel[required_level.upper()]
                    
            except (ValueError, KeyError) as e:
                logger.warning(
                    f"Authorization failed: invalid parameter {e}",
                    extra={"correlation_id": correlation_id}
                )
                return False
            
            # Check permission using RBAC system
            has_permission = self._rbac_system.check_permission(
                user_id, resource_type, resource_id, action_enum, permission_level
            )
            
            # SECURITY: Log authorization decision
            processing_time = (time.time() - start_time) * 1000
            logger.info(
                f"Authorization {'granted' if has_permission else 'denied'}",
                extra={
                    "correlation_id": correlation_id,
                    "user_id": user_id,
                    "resource": resource,
                    "resource_id": resource_id,
                    "action": action,
                    "required_level": required_level,
                    "result": has_permission,
                    "processing_time_ms": processing_time
                }
            )
            
            return has_permission
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(
                f"Authorization error: {e}",
                extra={
                    "correlation_id": correlation_id,
                    "user_id": user_id,
                    "resource": resource,
                    "action": action,
                    "error_type": type(e).__name__,
                    "processing_time_ms": processing_time
                }
            )
            # SECURITY: Fail secure on errors
            return False
    
    def get_public_key_pem(self) -> str:
        """
        Get the public key in PEM format for token verification.
        
        Returns:
            Public key as PEM string
            
        Raises:
            ValueError: If no public key is configured
        """
        if self.insecure_mode:
            return "-----BEGIN PUBLIC KEY-----\n(INSECURE STUB KEY)\n-----END PUBLIC KEY-----"
        
        if not self._jwt_validator or not self._jwt_validator.public_key:
            raise ValueError("No public key configured for token verification")
        
        # Get public key from JWT validator
        public_key = self._jwt_validator.public_key
        pem_data = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return pem_data.decode('utf-8')
    
    def create_token(self, user_id: str, permissions: Optional[Set[str]] = None) -> str:
        """
        Create an authentication token for a user.
        
        Note: Token creation requires a private key and is typically
        handled by an external identity provider in production.
        
        Args:
            user_id: User identifier
            permissions: Set of permissions to embed in token
            
        Returns:
            Signed authentication token
            
        Raises:
            NotImplementedError: Token signing not implemented (use external IdP)
        """
        if self.insecure_mode:
            return f"INSECURE_TOKEN_{user_id}"
        
        # SECURITY: Token creation should be handled by external identity provider
        # This method is kept for compatibility but should not be used in production
        logger.warning(
            "Token creation requested - should use external identity provider",
            extra={
                "user_id": user_id,
                "permissions_count": len(permissions) if permissions else 0
            }
        )
        
        raise NotImplementedError(
            "Token creation should be handled by external identity provider (Azure AD, Okta, etc.). "
            "This security manager only validates tokens, not creates them."
        )
    
    def has_permission(self, user_id: str, permission: str) -> bool:
        """
        Check if user has a specific permission using RBAC system.
        
        Args:
            user_id: User identifier
            permission: Permission string (format: resource:action:level)
            
        Returns:
            True if user has permission, False otherwise
        """
        if self.insecure_mode:
            return True
        
        if not self._rbac_system:
            logger.warning(f"Permission check failed: no RBAC system configured")
            return False
        
        # Parse permission string (format: resource:action:level)
        try:
            parts = permission.split(':')
            if len(parts) < 2:
                logger.warning(f"Invalid permission format: {permission}")
                return False
            
            resource = parts[0]
            action = parts[1]
            level = parts[2] if len(parts) > 2 else "read"
            
            return self.authorize(user_id, resource, action, required_level=level)
            
        except Exception as e:
            logger.warning(f"Permission check error: {e}")
            return False
    
    def grant_role(self, user_id: str, role_name: str) -> bool:
        """Grant a role to a user using RBAC system."""
        if self.insecure_mode:
            logger.debug(f"Role grant bypassed (insecure mode): {role_name} -> {user_id}")
            return True
        
        if not self._rbac_system:
            logger.error("Cannot grant role: no RBAC system configured")
            return False
        
        try:
            return self._rbac_system.assign_role(user_id, role_name)
        except Exception as e:
            logger.error(f"Failed to grant role {role_name} to {user_id}: {e}")
            return False
    
    def revoke_role(self, user_id: str, role_name: str) -> bool:
        """Revoke a role from a user using RBAC system."""
        if self.insecure_mode:
            logger.debug(f"Role revocation bypassed (insecure mode): {role_name} from {user_id}")
            return True
        
        if not self._rbac_system:
            logger.error("Cannot revoke role: no RBAC system configured")
            return False
        
        try:
            return self._rbac_system.revoke_role(user_id, role_name)
        except Exception as e:
            logger.error(f"Failed to revoke role {role_name} from {user_id}: {e}")
            return False
    
    def _initialize_security_config(self, config: Any, insecure_mode: bool) -> SecurityConfiguration:
        """Initialize security configuration with validation."""
        if config and isinstance(config, SecurityConfiguration):
            security_config = config
        else:
            # Load from environment or create defaults
            try:
                security_config = load_security_config_from_env()
            except Exception as e:
                logger.warning(f"Failed to load security config from environment: {e}")
                security_config = create_default_security_config(insecure_mode)
        
        # Override with insecure mode if requested
        if insecure_mode:
            security_config = create_default_security_config(insecure_mode=True)
        
        # Validate configuration
        try:
            validate_security_config(security_config)
            logger.info("Security configuration validation successful")
        except Exception as e:
            if not insecure_mode:
                logger.error(f"Security configuration validation failed: {e}")
                raise
            logger.warning(f"Security configuration validation bypassed (insecure mode): {e}")
        
        return security_config
    
    def _initialize_insecure_mode(self):
        """Initialize security manager in insecure mode (DEVELOPMENT ONLY)."""
        logger.critical(
            "⚠️  SECURITY WARNING: SecurityManager running in INSECURE mode! ⚠️\n"
            "   • Authentication is DISABLED\n"
            "   • Authorization is DISABLED\n"
            "   • All security controls are BYPASSED\n"
            "   • This mode is for DEVELOPMENT ONLY\n"
            "   • NEVER use this in production!"
        )
        
        # Initialize minimal components for compatibility
        self._rbac_system = get_default_rbac_system()
    
    def _initialize_secure_mode(self):
        """Initialize security manager in secure mode with full protection."""
        logger.info("Initializing production security components")
        
        # Initialize JWT validator
        if self._security_config.jwt_public_key_path:
            try:
                with open(self._security_config.jwt_public_key_path, 'r') as f:
                    public_key_pem = f.read()
                
                self._jwt_validator = JWTValidator(
                    public_key=public_key_pem,
                    algorithms=[self._security_config.jwt_algorithm],
                    audience=self._security_config.jwt_audience,
                    issuer=self._security_config.jwt_issuer,
                    max_token_age_hours=self._security_config.jwt_max_token_age_hours
                )
                
                # Set as default validator
                set_default_validator(self._jwt_validator)
                logger.info("JWT validator initialized")
                
            except Exception as e:
                logger.error(f"Failed to initialize JWT validator: {e}")
                if self._security_config.security_mode == SecurityMode.PRODUCTION:
                    raise ValueError(f"Cannot initialize JWT validator in production mode: {e}")
        else:
            logger.warning("No JWT public key configured - authentication will fail")
        
        # Initialize RBAC system
        self._rbac_system = get_default_rbac_system()
        logger.info("RBAC system initialized")
        
        logger.info("Security manager initialized in secure mode")
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status and metrics."""
        return {
            "security_mode": self._security_config.security_mode.value,
            "insecure_mode": self.insecure_mode,
            "jwt_validator_configured": bool(self._jwt_validator),
            "rbac_system_configured": bool(self._rbac_system),
            "active_sessions": len(self._authenticated_sessions),
            "auth_requests": self._auth_requests,
            "auth_successes": self._auth_successes,
            "auth_failures": self._auth_failures,
            "auth_success_rate": (self._auth_successes / max(1, self._auth_requests)) * 100
        }


# Default instance for convenience
# SECURITY: Default to secure mode - override only for development
default_security_manager: Optional[SecurityManager] = None


def create_security_manager(
    config: Optional[SecurityConfiguration] = None, 
    insecure_mode: bool = False
) -> SecurityManager:
    """
    Create a SecurityManager with explicit security configuration.
    
    Args:
        config: Security configuration (will load from environment if None)
        insecure_mode: If True, disable authentication and authorization (DEVELOPMENT ONLY)
        
    Returns:
        Configured SecurityManager instance
        
    Warning:
        Setting insecure_mode=True disables ALL security controls.
        This should NEVER be used in production.
    """
    return SecurityManager(config=config, insecure_mode=insecure_mode)


def get_security_manager() -> SecurityManager:
    """Get the default security manager instance."""
    global default_security_manager
    if default_security_manager is None:
        # SECURITY: Initialize with secure defaults
        default_security_manager = SecurityManager(insecure_mode=False)
    return default_security_manager


def set_default_security_manager(manager: SecurityManager) -> None:
    """Set the default security manager instance."""
    global default_security_manager
    default_security_manager = manager


__all__ = [
    "SecurityManager", 
    "SecurityLevel", 
    "get_security_manager",
    "set_default_security_manager",
    "create_security_manager"
]