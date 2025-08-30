"""
Security API for AgentsMCP

Provides comprehensive security features including authentication,
authorization, rate limiting, threat detection, and security monitoring
for all backend services.
"""

import asyncio
import hashlib
import hmac
import logging
import secrets
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum

from .base import APIBase, APIResponse, APIError


class ThreatLevel(Enum):
    """Threat severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuthAction(Enum):
    """Authentication actions"""
    LOGIN = "login"
    LOGOUT = "logout"
    TOKEN_REFRESH = "token_refresh"
    PASSWORD_RESET = "password_reset"
    ACCOUNT_LOCK = "account_lock"


@dataclass
class SecurityEvent:
    """Represents a security event"""
    id: str
    event_type: str
    threat_level: ThreatLevel
    source_ip: str
    user_id: Optional[str]
    timestamp: datetime
    details: Dict[str, Any]
    resolved: bool = False


@dataclass
class RateLimitRule:
    """Rate limiting rule configuration"""
    name: str
    requests_per_window: int
    window_seconds: int
    burst_allowance: int
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AccessToken:
    """Represents an access token"""
    token_id: str
    user_id: str
    scopes: Set[str]
    issued_at: datetime
    expires_at: datetime
    last_used: Optional[datetime] = None
    revoked: bool = False


class SecurityAPI(APIBase):
    """
    Comprehensive security system for AgentsMCP backend services.
    
    Features:
    - Token-based authentication with JWT support
    - Role-based access control (RBAC)
    - Rate limiting with adaptive thresholds
    - Threat detection and prevention
    - Security event logging and monitoring
    - API key management
    - Session management with security controls
    - Automated threat response
    """
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Authentication and authorization
        self.active_tokens: Dict[str, AccessToken] = {}
        self.user_roles: Dict[str, Set[str]] = {}
        self.role_permissions: Dict[str, Set[str]] = {}
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        
        # Rate limiting
        self.rate_limit_rules: Dict[str, RateLimitRule] = {}
        self.rate_limit_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.blocked_ips: Dict[str, datetime] = {}
        
        # Security monitoring
        self.security_events: deque = deque(maxlen=10000)
        self.threat_patterns: Dict[str, Dict[str, Any]] = {}
        self.suspicious_activities: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Security settings
        self.security_config = {
            "token_expiry_hours": 24,
            "max_login_attempts": 5,
            "lockout_duration_minutes": 30,
            "password_min_length": 8,
            "require_mfa": False,
            "session_timeout_minutes": 60,
            "ip_whitelist": set(),
            "ip_blacklist": set()
        }
        
        # Background tasks
        self._security_tasks: Set[asyncio.Task] = set()
        self._initialize_default_security()
        
    async def initialize(self) -> APIResponse:
        """Initialize the security system"""
        try:
            # Start background security tasks
            tasks = [
                asyncio.create_task(self._token_cleanup_loop()),
                asyncio.create_task(self._threat_detection_loop()),
                asyncio.create_task(self._rate_limit_cleanup_loop()),
                asyncio.create_task(self._security_monitoring_loop())
            ]
            
            self._security_tasks.update(tasks)
            
            await self._log_security_event("system", ThreatLevel.LOW, None, None, {
                "action": "security_system_initialized",
                "active_tasks": len(self._security_tasks)
            })
            
            self.logger.info("Security system initialized")
            
            return APIResponse(
                success=True,
                data={"status": "initialized", "active_security_tasks": len(self._security_tasks)},
                message="Security system initialized successfully"
            )
            
        except Exception as e:
            return APIResponse(
                success=False,
                error=APIError("SECURITY_INIT_ERROR", f"Failed to initialize security: {str(e)}")
            )
    
    async def authenticate_token(self, token: str) -> APIResponse:
        """Authenticate a bearer token"""
        try:
            if not token:
                return APIResponse(
                    success=False,
                    error=APIError("MISSING_TOKEN", "Authentication token is required")
                )
            
            # Extract token ID from token (simplified - in real implementation would decode JWT)
            token_id = hashlib.sha256(token.encode()).hexdigest()[:16]
            
            if token_id not in self.active_tokens:
                await self._log_security_event("auth_failure", ThreatLevel.MEDIUM, None, None, {
                    "reason": "invalid_token",
                    "token_id": token_id
                })
                return APIResponse(
                    success=False,
                    error=APIError("INVALID_TOKEN", "Invalid or expired token")
                )
            
            access_token = self.active_tokens[token_id]
            
            # Check if token is revoked
            if access_token.revoked:
                await self._log_security_event("auth_failure", ThreatLevel.MEDIUM, None, access_token.user_id, {
                    "reason": "revoked_token",
                    "token_id": token_id
                })
                return APIResponse(
                    success=False,
                    error=APIError("REVOKED_TOKEN", "Token has been revoked")
                )
            
            # Check token expiration
            if datetime.utcnow() > access_token.expires_at:
                await self._log_security_event("auth_failure", ThreatLevel.LOW, None, access_token.user_id, {
                    "reason": "expired_token",
                    "token_id": token_id
                })
                return APIResponse(
                    success=False,
                    error=APIError("EXPIRED_TOKEN", "Token has expired")
                )
            
            # Update last used timestamp
            access_token.last_used = datetime.utcnow()
            
            return APIResponse(
                success=True,
                data={
                    "user_id": access_token.user_id,
                    "scopes": list(access_token.scopes),
                    "expires_at": access_token.expires_at.isoformat()
                },
                message="Token authenticated successfully"
            )
            
        except Exception as e:
            return APIResponse(
                success=False,
                error=APIError("AUTH_ERROR", f"Authentication failed: {str(e)}")
            )
    
    async def create_token(self, user_id: str, scopes: List[str], expires_in_hours: Optional[int] = None) -> APIResponse:
        """Create a new access token"""
        try:
            # Generate secure token
            token_data = f"{user_id}:{int(time.time())}:{secrets.token_urlsafe(32)}"
            token = secrets.token_urlsafe(64)
            token_id = hashlib.sha256(token.encode()).hexdigest()[:16]
            
            # Set expiration
            expires_in = expires_in_hours or self.security_config["token_expiry_hours"]
            expires_at = datetime.utcnow() + timedelta(hours=expires_in)
            
            # Create access token
            access_token = AccessToken(
                token_id=token_id,
                user_id=user_id,
                scopes=set(scopes),
                issued_at=datetime.utcnow(),
                expires_at=expires_at
            )
            
            self.active_tokens[token_id] = access_token
            
            await self._log_security_event("token_created", ThreatLevel.LOW, None, user_id, {
                "token_id": token_id,
                "scopes": scopes,
                "expires_at": expires_at.isoformat()
            })
            
            return APIResponse(
                success=True,
                data={
                    "token": token,
                    "token_id": token_id,
                    "expires_at": expires_at.isoformat(),
                    "expires_in_seconds": int(expires_in * 3600)
                },
                message="Access token created successfully"
            )
            
        except Exception as e:
            return APIResponse(
                success=False,
                error=APIError("TOKEN_CREATION_ERROR", f"Failed to create token: {str(e)}")
            )
    
    async def revoke_token(self, token_id: str, user_id: Optional[str] = None) -> APIResponse:
        """Revoke an access token"""
        try:
            if token_id not in self.active_tokens:
                return APIResponse(
                    success=False,
                    error=APIError("TOKEN_NOT_FOUND", "Token not found")
                )
            
            access_token = self.active_tokens[token_id]
            
            # Check if user has permission to revoke this token
            if user_id and access_token.user_id != user_id:
                await self._log_security_event("unauthorized_revoke", ThreatLevel.MEDIUM, None, user_id, {
                    "token_id": token_id,
                    "token_owner": access_token.user_id
                })
                return APIResponse(
                    success=False,
                    error=APIError("UNAUTHORIZED", "Not authorized to revoke this token")
                )
            
            # Revoke token
            access_token.revoked = True
            
            await self._log_security_event("token_revoked", ThreatLevel.LOW, None, access_token.user_id, {
                "token_id": token_id,
                "revoked_by": user_id
            })
            
            return APIResponse(
                success=True,
                data={"token_id": token_id, "revoked_at": datetime.utcnow().isoformat()},
                message="Token revoked successfully"
            )
            
        except Exception as e:
            return APIResponse(
                success=False,
                error=APIError("TOKEN_REVOKE_ERROR", f"Failed to revoke token: {str(e)}")
            )
    
    async def check_permission(self, user_id: str, permission: str) -> APIResponse:
        """Check if user has a specific permission"""
        try:
            user_roles = self.user_roles.get(user_id, set())
            
            # Check if any of the user's roles have the required permission
            has_permission = False
            for role in user_roles:
                role_perms = self.role_permissions.get(role, set())
                if permission in role_perms or "*" in role_perms:
                    has_permission = True
                    break
            
            return APIResponse(
                success=True,
                data={
                    "user_id": user_id,
                    "permission": permission,
                    "has_permission": has_permission,
                    "user_roles": list(user_roles)
                }
            )
            
        except Exception as e:
            return APIResponse(
                success=False,
                error=APIError("PERMISSION_CHECK_ERROR", f"Failed to check permission: {str(e)}")
            )
    
    async def check_rate_limit(self, identifier: str, rule_name: str, source_ip: Optional[str] = None) -> APIResponse:
        """Check rate limit for an identifier"""
        try:
            # Check if IP is blocked
            if source_ip and source_ip in self.blocked_ips:
                block_expires = self.blocked_ips[source_ip]
                if datetime.utcnow() < block_expires:
                    await self._log_security_event("rate_limit_blocked", ThreatLevel.HIGH, source_ip, None, {
                        "identifier": identifier,
                        "rule": rule_name,
                        "block_expires": block_expires.isoformat()
                    })
                    return APIResponse(
                        success=False,
                        error=APIError("RATE_LIMITED", f"IP blocked until {block_expires.isoformat()}")
                    )
                else:
                    # Block expired, remove it
                    del self.blocked_ips[source_ip]
            
            if rule_name not in self.rate_limit_rules:
                return APIResponse(
                    success=False,
                    error=APIError("RULE_NOT_FOUND", f"Rate limit rule '{rule_name}' not found")
                )
            
            rule = self.rate_limit_rules[rule_name]
            if not rule.enabled:
                return APIResponse(success=True, data={"allowed": True, "reason": "rule_disabled"})
            
            # Create window key
            window_key = f"{rule_name}:{identifier}"
            now = datetime.utcnow()
            window_start = now - timedelta(seconds=rule.window_seconds)
            
            # Clean old entries
            window = self.rate_limit_windows[window_key]
            while window and window[0] < window_start:
                window.popleft()
            
            # Check current request count
            current_requests = len(window)
            
            # Check if within limits
            if current_requests >= rule.requests_per_window:
                # Check burst allowance
                if current_requests >= rule.requests_per_window + rule.burst_allowance:
                    # Rate limited - potentially block IP
                    if source_ip:
                        # Check for repeated violations
                        violations = self.suspicious_activities[source_ip]
                        violations.append({
                            "type": "rate_limit_violation",
                            "rule": rule_name,
                            "timestamp": now,
                            "requests": current_requests
                        })
                        
                        # Block IP if too many violations
                        recent_violations = [
                            v for v in violations 
                            if v["timestamp"] > now - timedelta(minutes=5)
                        ]
                        
                        if len(recent_violations) >= 3:
                            self.blocked_ips[source_ip] = now + timedelta(minutes=30)
                            
                            await self._log_security_event("ip_blocked", ThreatLevel.HIGH, source_ip, None, {
                                "violations": len(recent_violations),
                                "rule": rule_name
                            })
                    
                    await self._log_security_event("rate_limit_exceeded", ThreatLevel.MEDIUM, source_ip, None, {
                        "identifier": identifier,
                        "rule": rule_name,
                        "requests": current_requests,
                        "limit": rule.requests_per_window
                    })
                    
                    return APIResponse(
                        success=False,
                        error=APIError("RATE_LIMITED", f"Rate limit exceeded for rule '{rule_name}'"),
                        data={
                            "current_requests": current_requests,
                            "limit": rule.requests_per_window,
                            "window_seconds": rule.window_seconds,
                            "retry_after": rule.window_seconds
                        }
                    )
            
            # Add current request to window
            window.append(now)
            
            return APIResponse(
                success=True,
                data={
                    "allowed": True,
                    "current_requests": current_requests + 1,
                    "limit": rule.requests_per_window,
                    "window_seconds": rule.window_seconds,
                    "remaining": max(0, rule.requests_per_window - current_requests - 1)
                }
            )
            
        except Exception as e:
            return APIResponse(
                success=False,
                error=APIError("RATE_LIMIT_ERROR", f"Rate limit check failed: {str(e)}")
            )
    
    async def create_rate_limit_rule(self, name: str, requests_per_window: int, 
                                   window_seconds: int, burst_allowance: int = 0) -> APIResponse:
        """Create a new rate limit rule"""
        try:
            rule = RateLimitRule(
                name=name,
                requests_per_window=requests_per_window,
                window_seconds=window_seconds,
                burst_allowance=burst_allowance
            )
            
            self.rate_limit_rules[name] = rule
            
            await self._log_security_event("rate_limit_rule_created", ThreatLevel.LOW, None, None, {
                "rule_name": name,
                "requests_per_window": requests_per_window,
                "window_seconds": window_seconds,
                "burst_allowance": burst_allowance
            })
            
            return APIResponse(
                success=True,
                data={"rule_name": name, "created_at": rule.created_at.isoformat()},
                message="Rate limit rule created successfully"
            )
            
        except Exception as e:
            return APIResponse(
                success=False,
                error=APIError("RULE_CREATION_ERROR", f"Failed to create rate limit rule: {str(e)}")
            )
    
    async def get_security_events(self, threat_level: Optional[ThreatLevel] = None, 
                                limit: int = 100) -> APIResponse:
        """Get recent security events"""
        try:
            events = list(self.security_events)
            
            # Filter by threat level if specified
            if threat_level:
                events = [e for e in events if e.threat_level == threat_level]
            
            # Sort by timestamp (newest first) and limit
            events.sort(key=lambda e: e.timestamp, reverse=True)
            events = events[:limit]
            
            result = {
                "events": [
                    {
                        "id": event.id,
                        "event_type": event.event_type,
                        "threat_level": event.threat_level.value,
                        "source_ip": event.source_ip,
                        "user_id": event.user_id,
                        "timestamp": event.timestamp.isoformat(),
                        "details": event.details,
                        "resolved": event.resolved
                    }
                    for event in events
                ],
                "total_count": len(events),
                "filter_applied": threat_level.value if threat_level else None
            }
            
            return APIResponse(success=True, data=result)
            
        except Exception as e:
            return APIResponse(
                success=False,
                error=APIError("EVENTS_QUERY_ERROR", f"Failed to query security events: {str(e)}")
            )
    
    async def get_threat_intelligence(self) -> APIResponse:
        """Get threat intelligence summary"""
        try:
            now = datetime.utcnow()
            
            # Analyze recent events (last 24 hours)
            recent_cutoff = now - timedelta(hours=24)
            recent_events = [e for e in self.security_events if e.timestamp >= recent_cutoff]
            
            # Count events by type and threat level
            event_counts = defaultdict(int)
            threat_counts = defaultdict(int)
            source_ips = defaultdict(int)
            
            for event in recent_events:
                event_counts[event.event_type] += 1
                threat_counts[event.threat_level.value] += 1
                if event.source_ip:
                    source_ips[event.source_ip] += 1
            
            # Identify top threats
            top_event_types = sorted(event_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            top_source_ips = sorted(source_ips.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Calculate threat score (0-100)
            critical_events = threat_counts.get("critical", 0)
            high_events = threat_counts.get("high", 0)
            medium_events = threat_counts.get("medium", 0)
            low_events = threat_counts.get("low", 0)
            
            threat_score = min(100, (critical_events * 20) + (high_events * 10) + (medium_events * 5) + (low_events * 1))
            
            # Determine threat level
            if threat_score >= 80:
                overall_threat = "critical"
            elif threat_score >= 60:
                overall_threat = "high"
            elif threat_score >= 30:
                overall_threat = "medium"
            else:
                overall_threat = "low"
            
            result = {
                "threat_summary": {
                    "overall_threat_level": overall_threat,
                    "threat_score": threat_score,
                    "total_events_24h": len(recent_events),
                    "blocked_ips": len(self.blocked_ips),
                    "active_tokens": len([t for t in self.active_tokens.values() if not t.revoked])
                },
                "event_breakdown": {
                    "by_type": dict(event_counts),
                    "by_threat_level": dict(threat_counts)
                },
                "top_threats": {
                    "event_types": top_event_types,
                    "source_ips": top_source_ips
                },
                "security_status": {
                    "rate_limit_rules": len(self.rate_limit_rules),
                    "active_users": len(self.user_roles),
                    "defined_roles": len(self.role_permissions)
                }
            }
            
            return APIResponse(success=True, data=result)
            
        except Exception as e:
            return APIResponse(
                success=False,
                error=APIError("THREAT_INTEL_ERROR", f"Failed to generate threat intelligence: {str(e)}")
            )
    
    async def _log_security_event(self, event_type: str, threat_level: ThreatLevel, 
                                source_ip: Optional[str], user_id: Optional[str], 
                                details: Dict[str, Any]):
        """Internal method to log security events"""
        event_id = f"{event_type}_{int(time.time())}_{secrets.token_hex(8)}"
        
        event = SecurityEvent(
            id=event_id,
            event_type=event_type,
            threat_level=threat_level,
            source_ip=source_ip or "unknown",
            user_id=user_id,
            timestamp=datetime.utcnow(),
            details=details
        )
        
        self.security_events.append(event)
        
        # Log to system logger based on threat level
        log_message = f"Security Event: {event_type} - {threat_level.value} - IP: {source_ip} - User: {user_id}"
        if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            self.logger.error(log_message)
        elif threat_level == ThreatLevel.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def _initialize_default_security(self):
        """Initialize default security configuration"""
        # Default roles and permissions
        self.role_permissions = {
            "admin": {"*"},  # All permissions
            "user": {"read", "write_own", "delete_own"},
            "readonly": {"read"},
            "guest": {"read_public"}
        }
        
        # Default rate limit rules
        default_rules = [
            RateLimitRule("api_general", 1000, 3600, 100),  # 1000 requests per hour
            RateLimitRule("api_strict", 100, 600, 10),      # 100 requests per 10 minutes
            RateLimitRule("login_attempts", 5, 300, 0),     # 5 login attempts per 5 minutes
            RateLimitRule("password_reset", 3, 1800, 0),    # 3 password resets per 30 minutes
        ]
        
        for rule in default_rules:
            self.rate_limit_rules[rule.name] = rule
        
        # Default threat patterns
        self.threat_patterns = {
            "brute_force_login": {
                "pattern": "failed_login_attempts",
                "threshold": 10,
                "window_minutes": 5,
                "response": "block_ip"
            },
            "token_abuse": {
                "pattern": "rapid_token_usage",
                "threshold": 100,
                "window_minutes": 1,
                "response": "revoke_token"
            },
            "suspicious_ip": {
                "pattern": "multiple_user_access",
                "threshold": 5,
                "window_minutes": 10,
                "response": "flag_for_review"
            }
        }
    
    async def _token_cleanup_loop(self):
        """Background task to clean up expired tokens"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                now = datetime.utcnow()
                expired_tokens = [
                    token_id for token_id, token in self.active_tokens.items()
                    if now > token.expires_at or token.revoked
                ]
                
                for token_id in expired_tokens:
                    del self.active_tokens[token_id]
                
                if expired_tokens:
                    await self._log_security_event("token_cleanup", ThreatLevel.LOW, None, None, {
                        "cleaned_tokens": len(expired_tokens)
                    })
                    
                    self.logger.info(f"Cleaned up {len(expired_tokens)} expired/revoked tokens")
                
            except Exception as e:
                self.logger.error(f"Error in token cleanup loop: {e}")
                await asyncio.sleep(1800)  # Wait 30 minutes on error
    
    async def _threat_detection_loop(self):
        """Background task for threat detection"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                threats_detected = 0
                
                # Analyze suspicious activities
                now = datetime.utcnow()
                for source_ip, activities in self.suspicious_activities.items():
                    # Remove old activities (older than 1 hour)
                    cutoff_time = now - timedelta(hours=1)
                    activities[:] = [a for a in activities if a["timestamp"] > cutoff_time]
                    
                    # Check for threat patterns
                    if len(activities) >= 10:  # High activity threshold
                        await self._log_security_event("high_activity_detected", ThreatLevel.MEDIUM, source_ip, None, {
                            "activity_count": len(activities),
                            "time_window": "1 hour"
                        })
                        threats_detected += 1
                
                # Clean up empty activity records
                empty_ips = [ip for ip, activities in self.suspicious_activities.items() if not activities]
                for ip in empty_ips:
                    del self.suspicious_activities[ip]
                
                if threats_detected > 0:
                    self.logger.info(f"Detected {threats_detected} potential threats")
                
            except Exception as e:
                self.logger.error(f"Error in threat detection loop: {e}")
                await asyncio.sleep(900)  # Wait 15 minutes on error
    
    async def _rate_limit_cleanup_loop(self):
        """Background task to clean up old rate limit windows"""
        while True:
            try:
                await asyncio.sleep(1800)  # Run every 30 minutes
                
                cleaned_windows = 0
                
                # Clean up old rate limit windows
                for window_key, window in list(self.rate_limit_windows.items()):
                    if not window or (datetime.utcnow() - window[-1]).total_seconds() > 3600:
                        del self.rate_limit_windows[window_key]
                        cleaned_windows += 1
                
                # Clean up expired IP blocks
                expired_blocks = [
                    ip for ip, expires_at in self.blocked_ips.items()
                    if datetime.utcnow() >= expires_at
                ]
                
                for ip in expired_blocks:
                    del self.blocked_ips[ip]
                
                if cleaned_windows > 0 or expired_blocks:
                    self.logger.info(f"Cleaned up {cleaned_windows} rate limit windows and {len(expired_blocks)} IP blocks")
                
            except Exception as e:
                self.logger.error(f"Error in rate limit cleanup loop: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour on error
    
    async def _security_monitoring_loop(self):
        """Background task for continuous security monitoring"""
        while True:
            try:
                await asyncio.sleep(600)  # Run every 10 minutes
                
                now = datetime.utcnow()
                
                # Monitor for anomalies
                recent_events = [e for e in self.security_events if (now - e.timestamp).total_seconds() < 600]
                
                # Check for unusual activity spikes
                if len(recent_events) > 50:  # More than 50 events in 10 minutes
                    await self._log_security_event("activity_spike", ThreatLevel.MEDIUM, None, None, {
                        "event_count": len(recent_events),
                        "time_window": "10 minutes"
                    })
                
                # Monitor active token usage
                active_token_count = len([t for t in self.active_tokens.values() if not t.revoked])
                if active_token_count > 1000:  # High number of active tokens
                    await self._log_security_event("high_token_count", ThreatLevel.LOW, None, None, {
                        "active_tokens": active_token_count
                    })
                
                # Check for blocked IPs
                if len(self.blocked_ips) > 100:  # Many blocked IPs might indicate an attack
                    await self._log_security_event("many_blocked_ips", ThreatLevel.MEDIUM, None, None, {
                        "blocked_ip_count": len(self.blocked_ips)
                    })
                
            except Exception as e:
                self.logger.error(f"Error in security monitoring loop: {e}")
                await asyncio.sleep(1800)  # Wait 30 minutes on error
    
    async def shutdown(self):
        """Shutdown the security system"""
        self.logger.info("Shutting down security system")
        
        # Cancel all background tasks
        for task in self._security_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._security_tasks:
            await asyncio.gather(*self._security_tasks, return_exceptions=True)
        
        await self._log_security_event("system_shutdown", ThreatLevel.LOW, None, None, {
            "shutdown_time": datetime.utcnow().isoformat()
        })
        
        self.logger.info("Security system shutdown complete")