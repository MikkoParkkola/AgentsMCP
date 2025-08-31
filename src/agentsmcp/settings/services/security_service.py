"""
Security service for settings and agent management.

Provides encryption, access control, and security validation
for the settings management system.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import asyncio
import cryptography.fernet
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
import base64

from ..domain.entities import (
    UserProfile,
    SettingsNode,
    AgentDefinition,
    AuditEntry,
)
from ..domain.value_objects import (
    PermissionLevel,
    SettingValue,
    SettingType,
    PermissionGrant,
    PermissionDeniedError,
)
from ..domain.repositories import (
    UserRepository,
    SecretRepository,
    AuditRepository,
)
from ..domain.services import (
    PermissionService,
    AuditService,
)
from ..events.security_events import (
    SecurityViolationEvent,
    EncryptionKeyRotatedEvent,
    PermissionGrantedEvent,
    PermissionRevokedEvent,
)
from ..events.event_publisher import EventPublisher


class SecurityService:
    """
    Main security service providing encryption, access control,
    and security monitoring capabilities.
    """
    
    def __init__(self,
                 user_repository: UserRepository,
                 secret_repository: SecretRepository,
                 audit_repository: AuditRepository,
                 event_publisher: EventPublisher,
                 master_key: str = None):
        self.user_repo = user_repository
        self.secret_repo = secret_repository
        self.audit_repo = audit_repository
        self.event_publisher = event_publisher
        
        # Domain services
        self.permission_service = PermissionService()
        self.audit_service = AuditService()
        
        # Encryption setup
        self.master_key = master_key or self._generate_master_key()
        self.cipher_suite = self._setup_encryption(self.master_key)
        
        # Security policies
        self.security_policies = self._init_security_policies()
        
        # Rate limiting
        self.rate_limits = {}
        self.failed_attempts = {}
    
    async def encrypt_setting(self, value: str, user_id: str,
                            additional_data: str = None) -> Tuple[str, str]:
        """
        Encrypt a setting value and return encrypted data and reference key.
        """
        try:
            # Generate a unique reference key
            reference_key = secrets.token_urlsafe(32)
            
            # Add timestamp and user context
            timestamp = datetime.utcnow().isoformat()
            context = f"{user_id}:{timestamp}:{additional_data or ''}"
            
            # Encrypt the value
            encrypted_data = self.cipher_suite.encrypt(value.encode())
            
            # Store encrypted data with reference key
            await self.secret_repo.store_secret(reference_key, base64.b64encode(encrypted_data).decode(), user_id)
            
            # Create audit entry
            audit_entry = self.audit_service.create_audit_entry(
                user_id=user_id,
                action="encrypt",
                resource_type="secret",
                resource_id=reference_key,
                details={"context": additional_data or "setting_value"}
            )
            await self.audit_repo.save_audit_entry(audit_entry)
            
            return base64.b64encode(encrypted_data).decode(), reference_key
            
        except Exception as e:
            # Log security event
            event = SecurityViolationEvent(
                user_id=user_id,
                violation_type="encryption_failure",
                details={"error": str(e)},
                severity="medium"
            )
            await self.event_publisher.publish(event)
            raise
    
    async def decrypt_setting(self, reference_key: str, user_id: str) -> Optional[str]:
        """
        Decrypt a setting value using its reference key.
        """
        try:
            # Retrieve encrypted data
            encrypted_data_b64 = await self.secret_repo.retrieve_secret(reference_key, user_id)
            if not encrypted_data_b64:
                return None
            
            # Decrypt the value
            encrypted_data = base64.b64decode(encrypted_data_b64.encode())
            decrypted_data = self.cipher_suite.decrypt(encrypted_data)
            
            # Create audit entry
            audit_entry = self.audit_service.create_audit_entry(
                user_id=user_id,
                action="decrypt",
                resource_type="secret",
                resource_id=reference_key,
                details={"access_time": datetime.utcnow().isoformat()}
            )
            await self.audit_repo.save_audit_entry(audit_entry)
            
            return decrypted_data.decode()
            
        except Exception as e:
            # Log security event
            event = SecurityViolationEvent(
                user_id=user_id,
                violation_type="decryption_failure",
                details={"reference_key": reference_key, "error": str(e)},
                severity="high"
            )
            await self.event_publisher.publish(event)
            return None
    
    async def rotate_encryption_key(self, new_master_key: str) -> Dict[str, Any]:
        """
        Rotate the master encryption key and re-encrypt all secrets.
        """
        try:
            old_cipher = self.cipher_suite
            new_cipher = self._setup_encryption(new_master_key)
            
            # This would typically be done in batches for large datasets
            updated_count = await self.secret_repo.rotate_encryption_key(
                self.master_key, new_master_key
            )
            
            # Update current cipher
            self.master_key = new_master_key
            self.cipher_suite = new_cipher
            
            # Create audit entry
            audit_entry = AuditEntry(
                user_id="system",
                action="rotate_encryption_key",
                resource_type="system",
                resource_id="encryption",
                details={"secrets_updated": updated_count}
            )
            await self.audit_repo.save_audit_entry(audit_entry)
            
            # Publish event
            event = EncryptionKeyRotatedEvent(
                updated_secrets_count=updated_count,
                rotation_time=datetime.utcnow()
            )
            await self.event_publisher.publish(event)
            
            return {
                "success": True,
                "updated_secrets": updated_count,
                "rotation_time": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            event = SecurityViolationEvent(
                user_id="system",
                violation_type="key_rotation_failure",
                details={"error": str(e)},
                severity="critical"
            )
            await self.event_publisher.publish(event)
            raise
    
    async def grant_permission(self, granter_id: str, user_id: str,
                             resource_type: str, resource_id: str,
                             permission_level: PermissionLevel,
                             expires_at: Optional[datetime] = None) -> None:
        """
        Grant permission to a user for a resource.
        """
        granter = await self.user_repo.get_user(granter_id)
        user = await self.user_repo.get_user(user_id)
        
        if not granter or not user:
            raise ValueError("Granter or user not found")
        
        # Check if granter has admin/owner rights
        if not self.permission_service.check_permission(
            granter, resource_type, resource_id, PermissionLevel.ADMIN
        ):
            raise PermissionDeniedError(granter_id, f"{resource_type}:{resource_id}", PermissionLevel.ADMIN)
        
        # Create permission grant
        grant = PermissionGrant(
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            permission_level=permission_level,
            granted_by=granter_id,
            expires_at=expires_at
        )
        
        user.add_permission(grant)
        await self.user_repo.save_user(user)
        
        # Create audit entry
        audit_entry = self.audit_service.create_audit_entry(
            user_id=granter_id,
            action="grant_permission",
            resource_type="permission",
            resource_id=f"{resource_type}:{resource_id}",
            details={
                "target_user": user_id,
                "permission_level": permission_level,
                "expires_at": expires_at.isoformat() if expires_at else None
            }
        )
        await self.audit_repo.save_audit_entry(audit_entry)
        
        # Publish event
        event = PermissionGrantedEvent(
            granter_id=granter_id,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            permission_level=permission_level,
            expires_at=expires_at
        )
        await self.event_publisher.publish(event)
    
    async def revoke_permission(self, revoker_id: str, user_id: str,
                              resource_type: str, resource_id: str) -> bool:
        """
        Revoke permission from a user for a resource.
        """
        revoker = await self.user_repo.get_user(revoker_id)
        user = await self.user_repo.get_user(user_id)
        
        if not revoker or not user:
            raise ValueError("Revoker or user not found")
        
        # Check if revoker has admin/owner rights
        if not self.permission_service.check_permission(
            revoker, resource_type, resource_id, PermissionLevel.ADMIN
        ):
            raise PermissionDeniedError(revoker_id, f"{resource_type}:{resource_id}", PermissionLevel.ADMIN)
        
        # Remove permission
        revoked = user.remove_permission(resource_type, resource_id)
        
        if revoked:
            await self.user_repo.save_user(user)
            
            # Create audit entry
            audit_entry = self.audit_service.create_audit_entry(
                user_id=revoker_id,
                action="revoke_permission",
                resource_type="permission",
                resource_id=f"{resource_type}:{resource_id}",
                details={"target_user": user_id}
            )
            await self.audit_repo.save_audit_entry(audit_entry)
            
            # Publish event
            event = PermissionRevokedEvent(
                revoker_id=revoker_id,
                user_id=user_id,
                resource_type=resource_type,
                resource_id=resource_id
            )
            await self.event_publisher.publish(event)
        
        return revoked
    
    async def validate_security_policy(self, user_id: str, resource_type: str,
                                     resource_id: str, operation: str,
                                     context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Validate an operation against security policies.
        """
        context = context or {}
        user = await self.user_repo.get_user(user_id)
        
        if not user:
            return {"allowed": False, "reason": "User not found"}
        
        result = {
            "allowed": True,
            "violations": [],
            "warnings": [],
            "required_actions": []
        }
        
        # Check rate limiting
        rate_limit_result = await self._check_rate_limit(user_id, operation)
        if not rate_limit_result["allowed"]:
            result["allowed"] = False
            result["violations"].append({
                "type": "rate_limit_exceeded",
                "message": rate_limit_result["message"]
            })
        
        # Check security policies
        for policy_name, policy in self.security_policies.items():
            if self._policy_applies(policy, resource_type, operation):
                policy_result = await self._evaluate_policy(policy, user, context)
                
                if not policy_result["allowed"]:
                    result["allowed"] = False
                    result["violations"].append({
                        "type": "policy_violation",
                        "policy": policy_name,
                        "message": policy_result["message"]
                    })
                
                result["warnings"].extend(policy_result.get("warnings", []))
                result["required_actions"].extend(policy_result.get("required_actions", []))
        
        # Log security violations
        if not result["allowed"]:
            event = SecurityViolationEvent(
                user_id=user_id,
                violation_type="policy_violation",
                details={
                    "resource_type": resource_type,
                    "resource_id": resource_id,
                    "operation": operation,
                    "violations": result["violations"]
                },
                severity="medium"
            )
            await self.event_publisher.publish(event)
        
        return result
    
    async def audit_security_events(self, user_id: str, time_range: int = 24) -> Dict[str, Any]:
        """
        Get security audit events for a user in the specified time range (hours).
        """
        since = datetime.utcnow() - timedelta(hours=time_range)
        
        # Get audit entries related to security
        security_actions = [
            "encrypt", "decrypt", "grant_permission", "revoke_permission",
            "login", "logout", "api_access", "failed_login"
        ]
        
        audit_entries = []
        for action in security_actions:
            entries = await self.audit_repo.get_audit_entries_by_action(
                action, since, limit=100
            )
            audit_entries.extend(entries)
        
        # Analyze for patterns
        analysis = {
            "total_events": len(audit_entries),
            "event_types": {},
            "suspicious_patterns": [],
            "recommendations": []
        }
        
        # Count event types
        for entry in audit_entries:
            action = entry.action
            if action not in analysis["event_types"]:
                analysis["event_types"][action] = 0
            analysis["event_types"][action] += 1
        
        # Detect suspicious patterns
        failed_logins = [e for e in audit_entries if e.action == "failed_login"]
        if len(failed_logins) > 5:
            analysis["suspicious_patterns"].append({
                "type": "excessive_failed_logins",
                "count": len(failed_logins),
                "severity": "medium"
            })
        
        # Generate recommendations
        if analysis["event_types"].get("decrypt", 0) > 50:
            analysis["recommendations"].append(
                "High number of decrypt operations detected. Consider reviewing access patterns."
            )
        
        return analysis
    
    def _generate_master_key(self) -> str:
        """Generate a secure master key."""
        return secrets.token_urlsafe(32)
    
    def _setup_encryption(self, master_key: str) -> cryptography.fernet.Fernet:
        """Set up encryption cipher suite."""
        # Derive key from master key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'agentsmcp_salt',  # In production, use a random salt per installation
            iterations=100000,
            backend=default_backend()
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
        return cryptography.fernet.Fernet(key)
    
    def _init_security_policies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize security policies."""
        return {
            "sensitive_data_access": {
                "applies_to": ["secret", "api_key"],
                "operations": ["read", "decrypt"],
                "requirements": {
                    "min_permission_level": PermissionLevel.READ,
                    "require_audit": True,
                    "max_daily_accesses": 100
                }
            },
            "admin_operations": {
                "applies_to": ["user", "agent", "settings_hierarchy"],
                "operations": ["create", "delete", "grant_permission"],
                "requirements": {
                    "min_permission_level": PermissionLevel.ADMIN,
                    "require_audit": True,
                    "require_approval": False
                }
            },
            "bulk_operations": {
                "applies_to": ["*"],
                "operations": ["bulk_update", "bulk_delete"],
                "requirements": {
                    "min_permission_level": PermissionLevel.WRITE,
                    "max_batch_size": 100,
                    "rate_limit": "10/hour"
                }
            }
        }
    
    def _policy_applies(self, policy: Dict[str, Any], resource_type: str, operation: str) -> bool:
        """Check if a policy applies to the given resource type and operation."""
        applies_to = policy.get("applies_to", [])
        operations = policy.get("operations", [])
        
        resource_match = "*" in applies_to or resource_type in applies_to
        operation_match = "*" in operations or operation in operations
        
        return resource_match and operation_match
    
    async def _evaluate_policy(self, policy: Dict[str, Any], user: UserProfile,
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a policy against user and context."""
        result = {
            "allowed": True,
            "message": "",
            "warnings": [],
            "required_actions": []
        }
        
        requirements = policy.get("requirements", {})
        
        # Check minimum permission level
        min_level = requirements.get("min_permission_level")
        if min_level:
            resource_type = context.get("resource_type", "")
            resource_id = context.get("resource_id", "")
            
            if not self.permission_service.check_permission(user, resource_type, resource_id, min_level):
                result["allowed"] = False
                result["message"] = f"Requires {min_level} permission level"
        
        # Check batch size limits
        max_batch_size = requirements.get("max_batch_size")
        if max_batch_size and context.get("batch_size", 0) > max_batch_size:
            result["allowed"] = False
            result["message"] = f"Batch size exceeds limit of {max_batch_size}"
        
        return result
    
    async def _check_rate_limit(self, user_id: str, operation: str) -> Dict[str, Any]:
        """Check if user is within rate limits for an operation."""
        current_time = datetime.utcnow()
        key = f"{user_id}:{operation}"
        
        if key not in self.rate_limits:
            self.rate_limits[key] = {
                "count": 0,
                "window_start": current_time,
                "limit": 100,  # Default limit
                "window_minutes": 60
            }
        
        rate_limit = self.rate_limits[key]
        
        # Check if window has expired
        window_end = rate_limit["window_start"] + timedelta(minutes=rate_limit["window_minutes"])
        if current_time > window_end:
            # Reset window
            rate_limit["count"] = 0
            rate_limit["window_start"] = current_time
        
        # Check limit
        if rate_limit["count"] >= rate_limit["limit"]:
            return {
                "allowed": False,
                "message": f"Rate limit exceeded: {rate_limit['limit']} per {rate_limit['window_minutes']} minutes"
            }
        
        # Increment counter
        rate_limit["count"] += 1
        
        return {"allowed": True}
    
    def hash_sensitive_data(self, data: str, salt: str = None) -> Tuple[str, str]:
        """Hash sensitive data with salt."""
        if not salt:
            salt = secrets.token_hex(16)
        
        hash_obj = hashlib.pbkdf2_hmac('sha256', data.encode(), salt.encode(), 100000)
        return hash_obj.hex(), salt
    
    def verify_hash(self, data: str, hash_value: str, salt: str) -> bool:
        """Verify data against hash."""
        computed_hash, _ = self.hash_sensitive_data(data, salt)
        return hmac.compare_digest(computed_hash, hash_value)