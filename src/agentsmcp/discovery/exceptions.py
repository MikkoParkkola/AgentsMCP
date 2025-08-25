"""Custom exception hierarchy for AgentsMCP discovery system.

Provides structured error handling with context preservation and serialization
for monitoring and health-check systems.
"""

from __future__ import annotations

import json
from typing import Any, Optional


class AgentsMcpError(RuntimeError):
    """Base class for all errors raised by the discovery subsystem.
    
    Stores optional context that can be rendered in logs and health reports.
    """
    
    def __init__(
        self,
        message: str,
        *,
        cause: Optional[BaseException] = None,
        payload: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.cause = cause
        self.payload = payload or {}

    def to_dict(self) -> dict[str, Any]:
        """Serializable representation for health-check and monitoring."""
        data = {"error": self.__class__.__name__, "message": str(self)}
        if self.payload:
            data["payload"] = self.payload
        if self.cause:
            data["cause"] = repr(self.cause)
        return data


# Configuration errors
class ConfigError(AgentsMcpError):
    """Raised when essential configuration is missing or malformed."""
    pass


class MissingEnvVarError(ConfigError):
    """Specialized error for missing required environment variables."""
    pass


# Dependency / Import errors
class DependencyError(AgentsMcpError):
    """Raised when a third-party library cannot be imported or is the wrong version."""
    pass


# Network / protocol errors
class NetworkError(AgentsMcpError):
    """Base for transport-level problems (timeouts, connection refused, etc.)."""
    pass


class DiscoveryProtocolError(NetworkError):
    """Errors raised while speaking the AgentsMCP discovery protocol."""
    pass


class ServiceUnavailableError(NetworkError):
    """Returned when the remote side explicitly signals unavailability (e.g. 503)."""
    pass


# Registry and storage errors  
class RegistryError(AgentsMcpError):
    """Base class for registry-related problems (corruption, I/O, etc.)."""
    pass


class RegistryCorruptionError(RegistryError):
    """Raised when registry data is corrupted or invalid."""
    pass


# Security and authentication errors
class SecurityError(AgentsMcpError):
    """Base for security-related problems (invalid signatures, unauthorized, etc.)."""
    pass


class AuthenticationError(SecurityError):
    """Authentication failed (invalid credentials, expired tokens, etc.)."""
    pass


class AuthorizationError(SecurityError):
    """Authorization failed (insufficient permissions, not in allowlist, etc.)."""
    pass


class SignatureError(SecurityError):
    """Digital signature validation failed."""
    pass


# Resource management errors
class ResourceError(AgentsMcpError):
    """Base for resource-related problems (allocation, capacity, health, etc.)."""
    pass


class ResourceUnavailableError(ResourceError):
    """Requested resource is not available or capacity exceeded."""
    pass


class HealthCheckError(ResourceError):
    """Health check failed for a resource or agent."""
    pass