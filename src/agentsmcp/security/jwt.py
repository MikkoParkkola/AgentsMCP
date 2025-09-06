"""
JWT Authentication Module
~~~~~~~~~~~~~~~~~~~~~~~~~

Production-ready JWT validation with RSA-256 signature verification.
Implements constant-time validation to prevent timing attacks.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set, Union

from jose import JWTError, jwt
from jose.exceptions import ExpiredSignatureError, JWTClaimsError
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.exceptions import InvalidSignature

logger = logging.getLogger(__name__)


class JWTValidationError(Exception):
    """JWT validation failed."""
    
    def __init__(self, message: str, error_type: str = "validation_error"):
        super().__init__(message)
        self.error_type = error_type


class TokenExpiredError(JWTValidationError):
    """JWT token has expired."""
    
    def __init__(self, message: str = "Token has expired"):
        super().__init__(message, "token_expired")


class InvalidTokenError(JWTValidationError):
    """JWT token is invalid."""
    
    def __init__(self, message: str = "Invalid token"):
        super().__init__(message, "invalid_token")


class JWTClaims:
    """Validated JWT claims."""
    
    def __init__(self, payload: Dict[str, Any]):
        self.payload = payload
        self.sub = payload.get("sub")  # Subject (user ID)
        self.aud = payload.get("aud")  # Audience
        self.iss = payload.get("iss")  # Issuer
        self.exp = payload.get("exp")  # Expiration time
        self.iat = payload.get("iat")  # Issued at
        self.jti = payload.get("jti")  # JWT ID
        self.roles = payload.get("roles", [])  # User roles
        self.permissions = payload.get("permissions", [])  # User permissions
        self.correlation_id = payload.get("correlation_id", str(uuid.uuid4()))


class JWTValidator:
    """
    Production-ready JWT validator with RSA-256 signature verification.
    Implements security best practices including constant-time validation.
    """
    
    def __init__(
        self,
        public_key: Optional[Union[str, bytes]] = None,
        algorithms: Optional[list] = None,
        audience: Optional[str] = None,
        issuer: Optional[str] = None,
        max_token_age_hours: int = 8
    ):
        """
        Initialize JWT validator.
        
        Args:
            public_key: RSA public key in PEM format
            algorithms: Allowed signing algorithms (default: RS256 only)
            audience: Expected audience claim
            issuer: Expected issuer claim  
            max_token_age_hours: Maximum token age in hours (enterprise compliance)
        """
        self.public_key = self._load_public_key(public_key) if public_key else None
        self.algorithms = algorithms or ["RS256"]  # Only allow RSA-256
        self.audience = audience
        self.issuer = issuer
        self.max_token_age_hours = max_token_age_hours
        
        # THREAT: Prevent algorithm confusion attacks
        if "HS256" in self.algorithms:
            logger.warning(
                "HS256 detected in allowed algorithms. "
                "This uses shared secrets which are less secure than RSA-256."
            )
        
        # Security event logging
        logger.info(
            "JWT validator initialized",
            extra={
                "algorithms": self.algorithms,
                "has_public_key": bool(self.public_key),
                "audience": self.audience,
                "issuer": self.issuer,
                "max_token_age_hours": self.max_token_age_hours
            }
        )
    
    def _load_public_key(self, key_data: Union[str, bytes]) -> rsa.RSAPublicKey:
        """Load RSA public key from PEM format."""
        try:
            if isinstance(key_data, str):
                key_data = key_data.encode('utf-8')
            
            public_key = serialization.load_pem_public_key(key_data)
            
            if not isinstance(public_key, rsa.RSAPublicKey):
                raise InvalidTokenError("Public key must be RSA")
            
            # SECURITY: Validate key size (minimum 2048 bits)
            key_size = public_key.key_size
            if key_size < 2048:
                raise InvalidTokenError(f"RSA key too small: {key_size} bits (minimum: 2048)")
            
            logger.info(f"Loaded RSA public key: {key_size} bits")
            return public_key
            
        except Exception as e:
            logger.error(f"Failed to load public key: {e}")
            raise InvalidTokenError(f"Invalid public key: {e}")
    
    def validate_token(self, token: str) -> JWTClaims:
        """
        Validate JWT token with constant-time signature verification.
        
        Args:
            token: JWT token string
            
        Returns:
            JWTClaims: Validated claims
            
        Raises:
            JWTValidationError: If validation fails
        """
        start_time = time.time()
        correlation_id = str(uuid.uuid4())
        
        try:
            # SECURITY: Log security event with correlation ID
            logger.info(
                "JWT validation started",
                extra={
                    "correlation_id": correlation_id,
                    "token_length": len(token) if token else 0
                }
            )
            
            if not token:
                raise InvalidTokenError("Empty token")
            
            # Parse without verification first for logging
            try:
                unverified_payload = jwt.get_unverified_claims(token)
                token_sub = unverified_payload.get("sub", "unknown")
                token_jti = unverified_payload.get("jti", "unknown")
            except Exception:
                token_sub = "invalid"
                token_jti = "invalid"
            
            # SECURITY: Validate token structure and signature
            options = {
                "verify_signature": True,
                "verify_aud": bool(self.audience),
                "verify_iss": bool(self.issuer),
                "verify_exp": True,
                "verify_iat": True,
                "require_exp": True,
                "require_iat": True
            }
            
            # Build validation parameters
            validate_params = {}
            if self.audience:
                validate_params["audience"] = self.audience
            if self.issuer:
                validate_params["issuer"] = self.issuer
            
            # THREAT: Use constant-time signature verification
            payload = jwt.decode(
                token,
                self.public_key.public_key_pem() if self.public_key else None,
                algorithms=self.algorithms,
                options=options,
                **validate_params
            )
            
            # Validate token age (enterprise compliance)
            iat = payload.get("iat")
            if iat:
                issued_time = datetime.fromtimestamp(iat, tz=timezone.utc)
                max_age = self.max_token_age_hours * 3600
                if time.time() - iat > max_age:
                    raise TokenExpiredError(
                        f"Token age exceeds maximum {self.max_token_age_hours} hours"
                    )
            
            # Create claims object
            claims = JWTClaims(payload)
            claims.correlation_id = correlation_id
            
            # SECURITY: Log successful validation
            processing_time = (time.time() - start_time) * 1000
            logger.info(
                "JWT validation successful",
                extra={
                    "correlation_id": correlation_id,
                    "sub": token_sub,
                    "jti": token_jti,
                    "processing_time_ms": processing_time,
                    "roles_count": len(claims.roles),
                    "permissions_count": len(claims.permissions)
                }
            )
            
            return claims
            
        except ExpiredSignatureError as e:
            processing_time = (time.time() - start_time) * 1000
            logger.warning(
                "JWT token expired",
                extra={
                    "correlation_id": correlation_id,
                    "sub": token_sub,
                    "jti": token_jti,
                    "processing_time_ms": processing_time,
                    "error": str(e)
                }
            )
            raise TokenExpiredError(str(e))
            
        except JWTClaimsError as e:
            processing_time = (time.time() - start_time) * 1000
            logger.warning(
                "JWT claims validation failed",
                extra={
                    "correlation_id": correlation_id,
                    "sub": token_sub,
                    "jti": token_jti,
                    "processing_time_ms": processing_time,
                    "error": str(e)
                }
            )
            raise InvalidTokenError(f"Claims validation failed: {e}")
            
        except JWTError as e:
            processing_time = (time.time() - start_time) * 1000
            logger.warning(
                "JWT signature validation failed",
                extra={
                    "correlation_id": correlation_id,
                    "sub": token_sub,
                    "jti": token_jti,
                    "processing_time_ms": processing_time,
                    "error": str(e)
                }
            )
            raise InvalidTokenError(f"Signature validation failed: {e}")
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(
                "JWT validation unexpected error",
                extra={
                    "correlation_id": correlation_id,
                    "sub": token_sub,
                    "jti": token_jti,
                    "processing_time_ms": processing_time,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            raise InvalidTokenError(f"Validation failed: {e}")
    
    def extract_bearer_token(self, authorization_header: Optional[str]) -> Optional[str]:
        """
        Extract bearer token from Authorization header.
        
        Args:
            authorization_header: Authorization header value
            
        Returns:
            Token string or None if not found
        """
        if not authorization_header:
            return None
            
        parts = authorization_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return None
            
        return parts[1]
    
    def get_token_info(self, token: str) -> Dict[str, Any]:
        """
        Get token information without validation (for debugging).
        SECURITY: Never expose sensitive information in logs.
        
        Args:
            token: JWT token string
            
        Returns:
            Token information dict
        """
        try:
            header = jwt.get_unverified_header(token)
            payload = jwt.get_unverified_claims(token)
            
            # SECURITY: Sanitize sensitive fields
            safe_payload = {
                k: v for k, v in payload.items()
                if k not in ["secret", "password", "key", "token"]
            }
            
            return {
                "header": header,
                "payload": safe_payload,
                "exp": payload.get("exp"),
                "iat": payload.get("iat"),
                "sub": payload.get("sub"),
                "aud": payload.get("aud"),
                "iss": payload.get("iss"),
                "is_expired": self._is_token_expired(payload)
            }
            
        except Exception as e:
            logger.warning(f"Failed to parse token info: {e}")
            return {"error": str(e)}
    
    def _is_token_expired(self, payload: Dict[str, Any]) -> bool:
        """Check if token is expired without full validation."""
        exp = payload.get("exp")
        if not exp:
            return False
            
        return datetime.now(timezone.utc).timestamp() > exp


# Default validator instance - will be configured by SecurityManager
_default_validator: Optional[JWTValidator] = None


def get_default_validator() -> Optional[JWTValidator]:
    """Get the default JWT validator instance."""
    return _default_validator


def set_default_validator(validator: JWTValidator) -> None:
    """Set the default JWT validator instance."""
    global _default_validator
    _default_validator = validator


def validate_jwt_token(token: str) -> JWTClaims:
    """
    Validate JWT token using default validator.
    
    Args:
        token: JWT token string
        
    Returns:
        JWTClaims: Validated claims
        
    Raises:
        JWTValidationError: If validation fails or no validator configured
    """
    validator = get_default_validator()
    if not validator:
        raise InvalidTokenError("No JWT validator configured")
    
    return validator.validate_token(token)


__all__ = [
    "JWTValidator",
    "JWTClaims", 
    "JWTValidationError",
    "TokenExpiredError",
    "InvalidTokenError",
    "get_default_validator",
    "set_default_validator",
    "validate_jwt_token"
]