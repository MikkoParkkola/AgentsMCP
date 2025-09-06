"""
Security Configuration System
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Production-ready security configuration with fail-secure defaults.
Validates all security settings at startup and provides secure key management.
"""

from __future__ import annotations

import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.fernet import Fernet
import base64

logger = logging.getLogger(__name__)


class SecurityMode(Enum):
    """Security operation modes."""
    
    PRODUCTION = "production"
    DEVELOPMENT = "development" 
    TESTING = "testing"
    INSECURE = "insecure"  # DEVELOPMENT ONLY - bypasses all security


class KeyType(Enum):
    """Cryptographic key types."""
    
    JWT_RSA = "jwt_rsa"
    JWT_HMAC = "jwt_hmac"  # Discouraged
    ENCRYPTION = "encryption"
    SIGNING = "signing"


class ValidationLevel(Enum):
    """Configuration validation levels."""
    
    STRICT = auto()      # Production-grade validation
    STANDARD = auto()    # Standard validation
    PERMISSIVE = auto()  # Minimal validation for development
    DISABLED = auto()    # No validation (INSECURE mode only)


@dataclass
class SecurityConfiguration:
    """Security configuration with fail-secure defaults."""
    
    # Core security settings
    security_mode: SecurityMode = SecurityMode.PRODUCTION
    validation_level: ValidationLevel = ValidationLevel.STRICT
    
    # JWT settings
    jwt_algorithm: str = "RS256"
    jwt_issuer: Optional[str] = None
    jwt_audience: Optional[str] = None
    jwt_max_token_age_hours: int = 8  # Enterprise compliance
    jwt_public_key_path: Optional[Path] = None
    jwt_private_key_path: Optional[Path] = None
    
    # Key management
    key_rotation_days: int = 90
    key_strength_bits: int = 4096  # RSA key size
    
    # Security policies
    require_https: bool = True
    require_authentication: bool = True
    require_authorization: bool = True
    log_security_events: bool = True
    
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 60
    
    # Session management
    session_timeout_minutes: int = 480  # 8 hours
    require_session_validation: bool = True
    
    # Audit settings
    audit_enabled: bool = True
    audit_retention_days: int = 365
    
    # Insecure mode warning
    insecure_mode_acknowledged: bool = False
    
    # Runtime state
    _validated: bool = field(default=False, init=False)
    _validation_errors: List[str] = field(default_factory=list, init=False)
    _keys: Dict[str, Any] = field(default_factory=dict, init=False)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.insecure_mode_acknowledged and self.security_mode == SecurityMode.INSECURE:
            raise ValueError(
                "INSECURE mode requires explicit acknowledgment. "
                "Set insecure_mode_acknowledged=True to proceed."
            )


class SecurityConfigValidator:
    """Validates security configuration with fail-secure behavior."""
    
    def __init__(self, config: SecurityConfiguration):
        self.config = config
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate(self) -> bool:
        """
        Validate security configuration.
        
        Returns:
            True if configuration is valid, False otherwise
            
        Note:
            Always fails secure - invalid configurations are rejected
        """
        self.errors.clear()
        self.warnings.clear()
        
        logger.info(f"Validating security configuration (mode: {self.config.security_mode.value})")
        
        # Skip validation in INSECURE mode (with warning)
        if self.config.security_mode == SecurityMode.INSECURE:
            if not self.config.insecure_mode_acknowledged:
                self.errors.append("INSECURE mode requires explicit acknowledgment")
                return False
            
            logger.critical(
                "⚠️  SECURITY WARNING: Running in INSECURE mode! ⚠️\n"
                "   • All authentication is DISABLED\n"
                "   • All authorization is DISABLED\n"
                "   • All cryptographic validation is BYPASSED\n"
                "   • This mode is for DEVELOPMENT ONLY\n"
                "   • NEVER use this in production!"
            )
            
            self.config._validated = True
            return True
        
        # Validate core security settings
        self._validate_security_mode()
        self._validate_jwt_settings()
        self._validate_key_management()
        self._validate_security_policies()
        self._validate_session_settings()
        
        # Check for errors
        if self.errors:
            logger.error(
                f"Security configuration validation failed: {len(self.errors)} errors",
                extra={"errors": self.errors}
            )
            return False
        
        # Log warnings
        if self.warnings:
            logger.warning(
                f"Security configuration has {len(self.warnings)} warnings",
                extra={"warnings": self.warnings}
            )
        
        self.config._validated = True
        logger.info("Security configuration validation successful")
        return True
    
    def _validate_security_mode(self):
        """Validate security mode settings."""
        if self.config.security_mode == SecurityMode.PRODUCTION:
            # Production requires strict validation
            if self.config.validation_level != ValidationLevel.STRICT:
                self.errors.append("Production mode requires STRICT validation level")
            
            # Production must require HTTPS
            if not self.config.require_https:
                self.errors.append("Production mode requires HTTPS")
            
            # Production must require authentication
            if not self.config.require_authentication:
                self.errors.append("Production mode requires authentication")
    
    def _validate_jwt_settings(self):
        """Validate JWT configuration."""
        # Algorithm validation
        if self.config.jwt_algorithm not in ["RS256", "RS384", "RS512"]:
            if self.config.jwt_algorithm in ["HS256", "HS384", "HS512"]:
                self.warnings.append(
                    f"HMAC algorithm {self.config.jwt_algorithm} uses shared secrets. "
                    "RSA algorithms are more secure."
                )
            else:
                self.errors.append(f"Unsupported JWT algorithm: {self.config.jwt_algorithm}")
        
        # Token age validation
        if self.config.jwt_max_token_age_hours > 24:
            self.warnings.append(
                f"JWT max age {self.config.jwt_max_token_age_hours}h exceeds 24h recommendation"
            )
        
        if self.config.jwt_max_token_age_hours > 48:
            self.errors.append(
                f"JWT max age {self.config.jwt_max_token_age_hours}h exceeds maximum 48h"
            )
        
        # Key file validation
        if self.config.jwt_algorithm.startswith("RS"):
            if not self.config.jwt_public_key_path:
                # Only require public key in production mode
                if self.config.security_mode == SecurityMode.PRODUCTION:
                    self.errors.append("RSA algorithm requires public key path in production mode")
                else:
                    self.warnings.append("RSA algorithm recommended to have public key path configured")
            else:
                self._validate_key_file(self.config.jwt_public_key_path, "JWT public key")
    
    def _validate_key_management(self):
        """Validate key management settings."""
        # Key strength validation
        if self.config.key_strength_bits < 2048:
            self.errors.append(f"Key strength {self.config.key_strength_bits} below minimum 2048 bits")
        elif self.config.key_strength_bits < 3072:
            self.warnings.append(
                f"Key strength {self.config.key_strength_bits} below recommended 3072+ bits"
            )
        
        # Rotation validation
        if self.config.key_rotation_days > 365:
            self.warnings.append(
                f"Key rotation period {self.config.key_rotation_days} days exceeds 1 year"
            )
    
    def _validate_security_policies(self):
        """Validate security policy settings."""
        # HTTPS validation
        if self.config.security_mode == SecurityMode.PRODUCTION and not self.config.require_https:
            self.errors.append("Production mode requires HTTPS")
        
        # Authentication validation
        if not self.config.require_authentication and self.config.require_authorization:
            self.errors.append("Authorization requires authentication to be enabled")
        
        # Rate limiting validation
        if self.config.rate_limit_requests_per_minute > 1000:
            self.warnings.append(
                f"Rate limit {self.config.rate_limit_requests_per_minute}/min is very high"
            )
    
    def _validate_session_settings(self):
        """Validate session management settings."""
        # Session timeout validation
        if self.config.session_timeout_minutes > 720:  # 12 hours
            self.warnings.append(
                f"Session timeout {self.config.session_timeout_minutes}min exceeds 12h recommendation"
            )
        
        if self.config.session_timeout_minutes > 1440:  # 24 hours
            self.errors.append(
                f"Session timeout {self.config.session_timeout_minutes}min exceeds maximum 24h"
            )
    
    def _validate_key_file(self, key_path: Path, key_description: str):
        """Validate a key file exists and is readable."""
        if not key_path.exists():
            self.errors.append(f"{key_description} file not found: {key_path}")
            return
        
        if not key_path.is_file():
            self.errors.append(f"{key_description} path is not a file: {key_path}")
            return
        
        # Check file permissions (should not be world-readable)
        try:
            stat = key_path.stat()
            if stat.st_mode & 0o044:  # World or group readable
                self.warnings.append(
                    f"{key_description} file has permissive permissions: {oct(stat.st_mode)}"
                )
        except OSError as e:
            self.warnings.append(f"Could not check {key_description} file permissions: {e}")


class SecurityKeyManager:
    """Manages cryptographic keys with secure generation and rotation."""
    
    def __init__(self, config: SecurityConfiguration):
        self.config = config
        self.keys: Dict[str, Any] = {}
    
    def generate_rsa_keypair(self, key_size: int = None) -> tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]:
        """Generate RSA key pair for JWT signing."""
        key_size = key_size or self.config.key_strength_bits
        
        logger.info(f"Generating RSA key pair: {key_size} bits")
        start_time = time.time()
        
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size
        )
        public_key = private_key.public_key()
        
        generation_time = (time.time() - start_time) * 1000
        logger.info(f"RSA key pair generated in {generation_time:.1f}ms")
        
        return private_key, public_key
    
    def save_key_to_file(self, key: Union[rsa.RSAPrivateKey, rsa.RSAPublicKey], 
                        file_path: Path, password: Optional[bytes] = None):
        """Save key to PEM file with secure permissions."""
        logger.info(f"Saving key to {file_path}")
        
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Serialize key
        if isinstance(key, rsa.RSAPrivateKey):
            if password:
                serialized_key = key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.BestAvailableEncryption(password)
                )
            else:
                serialized_key = key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                )
        else:  # Public key
            serialized_key = key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
        
        # Write with secure permissions
        with open(file_path, 'wb') as f:
            f.write(serialized_key)
        
        # Set restrictive permissions (owner read/write only)
        file_path.chmod(0o600)
        
        logger.info(f"Key saved to {file_path} with secure permissions")
    
    def load_key_from_file(self, file_path: Path, password: Optional[bytes] = None) -> Any:
        """Load key from PEM file."""
        logger.info(f"Loading key from {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"Key file not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            key_data = f.read()
        
        try:
            # Try loading as private key first
            key = serialization.load_pem_private_key(key_data, password=password)
            logger.info("Loaded private key")
            return key
        except ValueError:
            try:
                # Try loading as public key
                key = serialization.load_pem_public_key(key_data)
                logger.info("Loaded public key")
                return key
            except ValueError as e:
                raise ValueError(f"Could not load key from {file_path}: {e}")
    
    def generate_symmetric_key(self) -> bytes:
        """Generate symmetric key for encryption."""
        logger.info("Generating symmetric encryption key")
        return Fernet.generate_key()
    
    def encrypt_data(self, data: bytes, key: bytes) -> bytes:
        """Encrypt data with symmetric key."""
        fernet = Fernet(key)
        return fernet.encrypt(data)
    
    def decrypt_data(self, encrypted_data: bytes, key: bytes) -> bytes:
        """Decrypt data with symmetric key."""
        fernet = Fernet(key)
        return fernet.decrypt(encrypted_data)


def load_security_config_from_env() -> SecurityConfiguration:
    """Load security configuration from environment variables with secure defaults."""
    
    # Determine security mode
    mode_env = os.environ.get("AGENTSMCP_SECURITY_MODE", "production").lower()
    if mode_env == "insecure":
        # Special handling for insecure mode
        insecure_env = os.environ.get("AGENTSMCP_INSECURE", "").lower()
        if insecure_env not in ("true", "1", "yes", "on"):
            raise ValueError(
                "INSECURE mode requires AGENTSMCP_INSECURE=true environment variable"
            )
        security_mode = SecurityMode.INSECURE
        acknowledged = True
    else:
        security_mode = SecurityMode(mode_env)
        acknowledged = False
    
    # JWT settings
    jwt_public_key_path = os.environ.get("AGENTSMCP_JWT_PUBLIC_KEY")
    jwt_private_key_path = os.environ.get("AGENTSMCP_JWT_PRIVATE_KEY")
    
    config = SecurityConfiguration(
        security_mode=security_mode,
        insecure_mode_acknowledged=acknowledged,
        jwt_algorithm=os.environ.get("AGENTSMCP_JWT_ALGORITHM", "RS256"),
        jwt_issuer=os.environ.get("AGENTSMCP_JWT_ISSUER"),
        jwt_audience=os.environ.get("AGENTSMCP_JWT_AUDIENCE"),
        jwt_max_token_age_hours=int(os.environ.get("AGENTSMCP_JWT_MAX_AGE_HOURS", "8")),
        jwt_public_key_path=Path(jwt_public_key_path) if jwt_public_key_path else None,
        jwt_private_key_path=Path(jwt_private_key_path) if jwt_private_key_path else None,
        
        # Security policies from environment
        require_https=os.environ.get("AGENTSMCP_REQUIRE_HTTPS", "true").lower() == "true",
        require_authentication=os.environ.get("AGENTSMCP_REQUIRE_AUTH", "true").lower() == "true",
        require_authorization=os.environ.get("AGENTSMCP_REQUIRE_AUTHZ", "true").lower() == "true",
        
        # Rate limiting
        rate_limit_enabled=os.environ.get("AGENTSMCP_RATE_LIMIT", "true").lower() == "true",
        rate_limit_requests_per_minute=int(os.environ.get("AGENTSMCP_RATE_LIMIT_RPM", "60")),
        
        # Session settings
        session_timeout_minutes=int(os.environ.get("AGENTSMCP_SESSION_TIMEOUT", "480")),
    )
    
    logger.info(
        f"Security configuration loaded from environment (mode: {config.security_mode.value})"
    )
    
    return config


def validate_security_config(config: SecurityConfiguration) -> bool:
    """
    Validate security configuration with fail-secure behavior.
    
    Args:
        config: Security configuration to validate
        
    Returns:
        True if valid, False otherwise
        
    Raises:
        ValueError: For critical security violations
    """
    validator = SecurityConfigValidator(config)
    is_valid = validator.validate()
    
    if not is_valid and config.security_mode != SecurityMode.INSECURE:
        # SECURITY: Fail secure on validation errors
        error_summary = "; ".join(validator.errors)
        raise ValueError(f"Security configuration validation failed: {error_summary}")
    
    return is_valid


def create_default_security_config(insecure_mode: bool = False) -> SecurityConfiguration:
    """
    Create default security configuration.
    
    Args:
        insecure_mode: If True, create insecure configuration for development
        
    Returns:
        SecurityConfiguration with appropriate defaults
    """
    if insecure_mode:
        return SecurityConfiguration(
            security_mode=SecurityMode.INSECURE,
            insecure_mode_acknowledged=True,
            validation_level=ValidationLevel.DISABLED,
            require_https=False,
            require_authentication=False,
            require_authorization=False,
            rate_limit_enabled=False
        )
    
    return SecurityConfiguration(
        security_mode=SecurityMode.PRODUCTION,
        validation_level=ValidationLevel.STRICT
    )


__all__ = [
    "SecurityConfiguration",
    "SecurityConfigValidator", 
    "SecurityKeyManager",
    "SecurityMode",
    "ValidationLevel",
    "KeyType",
    "load_security_config_from_env",
    "validate_security_config",
    "create_default_security_config"
]