"""
Role-Based Access Control (RBAC) System
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Production-ready RBAC implementation with hierarchical roles and resource permissions.
Supports enterprise-grade authorization patterns.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)


class PermissionLevel(Enum):
    """Permission levels in order of increasing privilege."""
    
    NONE = auto()
    READ = auto()
    WRITE = auto()
    EXECUTE = auto()
    ADMIN = auto()
    
    def __ge__(self, other):
        """Allow comparison for permission hierarchy."""
        if not isinstance(other, PermissionLevel):
            return NotImplemented
        return self.value >= other.value
    
    def __le__(self, other):
        """Allow comparison for permission hierarchy."""
        if not isinstance(other, PermissionLevel):
            return NotImplemented
        return self.value <= other.value


class ResourceType(Enum):
    """Types of resources that can be protected."""
    
    AGENT = "agent"
    ORCHESTRATOR = "orchestrator"
    CONFIG = "config"
    SECURITY = "security"
    METRICS = "metrics"
    LOGS = "logs"
    API = "api"
    SYSTEM = "system"


class Action(Enum):
    """Actions that can be performed on resources."""
    
    # Basic CRUD operations
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    
    # Execution operations
    EXECUTE = "execute"
    INVOKE = "invoke"
    
    # Administrative operations
    CONFIGURE = "configure"
    MONITOR = "monitor"
    AUDIT = "audit"
    
    # Security operations
    AUTHENTICATE = "authenticate"
    AUTHORIZE = "authorize"
    GRANT = "grant"
    REVOKE = "revoke"


@dataclass(frozen=True)
class Permission:
    """A specific permission on a resource."""
    
    resource_type: ResourceType
    resource_id: Optional[str]  # None means all resources of this type
    action: Action
    level: PermissionLevel = PermissionLevel.READ
    conditions: tuple = field(default_factory=tuple)  # Use tuple for hashability
    
    def __str__(self) -> str:
        resource_spec = f"{self.resource_type.value}"
        if self.resource_id:
            resource_spec += f":{self.resource_id}"
        return f"{resource_spec}:{self.action.value}:{self.level.name}"
    
    def matches(self, resource_type: ResourceType, resource_id: Optional[str], action: Action) -> bool:
        """Check if this permission matches the requested access."""
        # Resource type must match
        if self.resource_type != resource_type:
            return False
        
        # Resource ID must match (None = wildcard)
        if self.resource_id is not None and self.resource_id != resource_id:
            return False
        
        # Action must match
        if self.action != action:
            return False
        
        return True


@dataclass
class Role:
    """A role with associated permissions and metadata."""
    
    name: str
    description: str
    permissions: Set[Permission] = field(default_factory=set)
    inherits_from: Set[str] = field(default_factory=set)  # Role inheritance
    is_system: bool = False  # System roles cannot be deleted
    created_at: Optional[float] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
    
    def add_permission(self, permission: Permission) -> None:
        """Add a permission to this role."""
        self.permissions.add(permission)
        logger.debug(f"Added permission {permission} to role {self.name}")
    
    def remove_permission(self, permission: Permission) -> bool:
        """Remove a permission from this role. Returns True if removed."""
        if permission in self.permissions:
            self.permissions.remove(permission)
            logger.debug(f"Removed permission {permission} from role {self.name}")
            return True
        return False
    
    def has_permission(
        self,
        resource_type: ResourceType,
        resource_id: Optional[str],
        action: Action,
        required_level: PermissionLevel = PermissionLevel.READ
    ) -> bool:
        """Check if this role has the required permission."""
        for permission in self.permissions:
            if permission.matches(resource_type, resource_id, action):
                if permission.level >= required_level:
                    return True
        return False


class RBACSystem:
    """
    Production-ready Role-Based Access Control system.
    
    Features:
    - Hierarchical roles with inheritance
    - Fine-grained resource permissions
    - Audit logging with correlation IDs
    - Performance optimization with caching
    """
    
    def __init__(self):
        self.roles: Dict[str, Role] = {}
        self.user_roles: Dict[str, Set[str]] = {}
        self._permission_cache: Dict[str, bool] = {}
        self._cache_ttl = 300  # 5 minutes
        self._cache_timestamps: Dict[str, float] = {}
        
        # Initialize system roles
        self._initialize_system_roles()
        
        logger.info("RBAC system initialized with system roles")
    
    def _initialize_system_roles(self) -> None:
        """Initialize default system roles."""
        
        # Guest role - minimal permissions
        guest_role = Role(
            name="guest",
            description="Minimal read-only access",
            is_system=True
        )
        guest_role.add_permission(Permission(
            ResourceType.API, None, Action.READ, PermissionLevel.READ
        ))
        self.roles["guest"] = guest_role
        
        # User role - basic operational permissions
        user_role = Role(
            name="user", 
            description="Standard user permissions",
            is_system=True,
            inherits_from={"guest"}
        )
        user_role.add_permission(Permission(
            ResourceType.AGENT, None, Action.EXECUTE, PermissionLevel.EXECUTE
        ))
        user_role.add_permission(Permission(
            ResourceType.ORCHESTRATOR, None, Action.INVOKE, PermissionLevel.EXECUTE
        ))
        user_role.add_permission(Permission(
            ResourceType.METRICS, None, Action.READ, PermissionLevel.READ
        ))
        self.roles["user"] = user_role
        
        # Power User role - advanced operations
        power_user_role = Role(
            name="power_user",
            description="Advanced user with configuration access", 
            is_system=True,
            inherits_from={"user"}
        )
        power_user_role.add_permission(Permission(
            ResourceType.CONFIG, None, Action.UPDATE, PermissionLevel.WRITE
        ))
        power_user_role.add_permission(Permission(
            ResourceType.AGENT, None, Action.CONFIGURE, PermissionLevel.WRITE
        ))
        self.roles["power_user"] = power_user_role
        
        # Admin role - full system access
        admin_role = Role(
            name="admin",
            description="Full administrative access",
            is_system=True,
            inherits_from={"power_user"}
        )
        admin_role.add_permission(Permission(
            ResourceType.SECURITY, None, Action.CONFIGURE, PermissionLevel.ADMIN
        ))
        admin_role.add_permission(Permission(
            ResourceType.SYSTEM, None, Action.CONFIGURE, PermissionLevel.ADMIN
        ))
        admin_role.add_permission(Permission(
            ResourceType.LOGS, None, Action.AUDIT, PermissionLevel.ADMIN
        ))
        # Grant/revoke permissions
        admin_role.add_permission(Permission(
            ResourceType.SECURITY, None, Action.GRANT, PermissionLevel.ADMIN
        ))
        admin_role.add_permission(Permission(
            ResourceType.SECURITY, None, Action.REVOKE, PermissionLevel.ADMIN
        ))
        self.roles["admin"] = admin_role
        
        logger.info("System roles initialized: guest, user, power_user, admin")
    
    def create_role(self, name: str, description: str, inherits_from: Optional[Set[str]] = None) -> Role:
        """Create a new custom role."""
        if name in self.roles:
            raise ValueError(f"Role '{name}' already exists")
        
        # Validate inherited roles exist
        inherits_from = inherits_from or set()
        for parent_role in inherits_from:
            if parent_role not in self.roles:
                raise ValueError(f"Parent role '{parent_role}' does not exist")
        
        role = Role(
            name=name,
            description=description,
            inherits_from=inherits_from,
            is_system=False
        )
        
        self.roles[name] = role
        self._invalidate_cache()
        
        logger.info(f"Created custom role: {name}")
        return role
    
    def delete_role(self, name: str) -> bool:
        """Delete a custom role (system roles cannot be deleted)."""
        if name not in self.roles:
            return False
        
        role = self.roles[name]
        if role.is_system:
            raise ValueError(f"Cannot delete system role: {name}")
        
        # Remove role from all users
        users_affected = []
        for user_id, user_roles in self.user_roles.items():
            if name in user_roles:
                user_roles.remove(name)
                users_affected.append(user_id)
        
        del self.roles[name]
        self._invalidate_cache()
        
        logger.info(f"Deleted role {name}, affected {len(users_affected)} users")
        return True
    
    def assign_role(self, user_id: str, role_name: str) -> bool:
        """Assign a role to a user."""
        if role_name not in self.roles:
            raise ValueError(f"Role '{role_name}' does not exist")
        
        if user_id not in self.user_roles:
            self.user_roles[user_id] = set()
        
        if role_name not in self.user_roles[user_id]:
            self.user_roles[user_id].add(role_name)
            self._invalidate_cache_for_user(user_id)
            
            logger.info(f"Assigned role '{role_name}' to user '{user_id}'")
            return True
        
        return False  # Already assigned
    
    def revoke_role(self, user_id: str, role_name: str) -> bool:
        """Revoke a role from a user."""
        if user_id not in self.user_roles:
            return False
        
        if role_name in self.user_roles[user_id]:
            self.user_roles[user_id].remove(role_name)
            self._invalidate_cache_for_user(user_id)
            
            logger.info(f"Revoked role '{role_name}' from user '{user_id}'")
            return True
        
        return False
    
    def get_user_roles(self, user_id: str) -> Set[str]:
        """Get all roles assigned to a user (including inherited)."""
        if user_id not in self.user_roles:
            return set()
        
        # Get direct roles
        direct_roles = self.user_roles[user_id].copy()
        
        # Add inherited roles
        all_roles = set()
        to_process = list(direct_roles)
        
        while to_process:
            role_name = to_process.pop()
            if role_name in all_roles:
                continue  # Avoid cycles
            
            all_roles.add(role_name)
            
            if role_name in self.roles:
                role = self.roles[role_name]
                to_process.extend(role.inherits_from)
        
        return all_roles
    
    def check_permission(
        self,
        user_id: str,
        resource_type: ResourceType,
        resource_id: Optional[str],
        action: Action,
        required_level: PermissionLevel = PermissionLevel.READ
    ) -> bool:
        """
        Check if a user has permission to perform an action.
        
        Args:
            user_id: User identifier
            resource_type: Type of resource being accessed
            resource_id: Specific resource ID (None for type-level access)
            action: Action being performed
            required_level: Minimum permission level required
            
        Returns:
            True if user has permission, False otherwise
        """
        correlation_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Check cache first
        cache_key = f"{user_id}:{resource_type.value}:{resource_id}:{action.value}:{required_level.name}"
        cached_result, is_fresh = self._get_cached_result(cache_key)
        
        if is_fresh and cached_result is not None:
            logger.debug(
                "Permission check (cached)",
                extra={
                    "correlation_id": correlation_id,
                    "user_id": user_id,
                    "resource_type": resource_type.value,
                    "resource_id": resource_id,
                    "action": action.value,
                    "required_level": required_level.name,
                    "result": cached_result,
                    "cache_hit": True
                }
            )
            return cached_result
        
        try:
            # Get all user roles (including inherited)
            user_roles = self.get_user_roles(user_id)
            
            # Check permissions across all roles
            has_permission = False
            for role_name in user_roles:
                if role_name in self.roles:
                    role = self.roles[role_name]
                    if role.has_permission(resource_type, resource_id, action, required_level):
                        has_permission = True
                        break
            
            # Cache the result
            self._cache_result(cache_key, has_permission)
            
            # SECURITY: Log authorization decision
            processing_time = (time.time() - start_time) * 1000
            logger.info(
                "Permission check completed",
                extra={
                    "correlation_id": correlation_id,
                    "user_id": user_id,
                    "resource_type": resource_type.value,
                    "resource_id": resource_id,
                    "action": action.value,
                    "required_level": required_level.name,
                    "result": has_permission,
                    "user_roles": list(user_roles),
                    "processing_time_ms": processing_time,
                    "cache_hit": False
                }
            )
            
            return has_permission
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(
                "Permission check failed",
                extra={
                    "correlation_id": correlation_id,
                    "user_id": user_id,
                    "resource_type": resource_type.value,
                    "resource_id": resource_id,
                    "action": action.value,
                    "required_level": required_level.name,
                    "error": str(e),
                    "processing_time_ms": processing_time
                }
            )
            # SECURITY: Fail secure - deny access on error
            return False
    
    def _get_cached_result(self, cache_key: str) -> tuple[Optional[bool], bool]:
        """Get cached result and whether it's still fresh."""
        if cache_key not in self._permission_cache:
            return None, False
        
        timestamp = self._cache_timestamps.get(cache_key, 0)
        is_fresh = (time.time() - timestamp) < self._cache_ttl
        
        if not is_fresh:
            # Clean up expired entry
            self._permission_cache.pop(cache_key, None)
            self._cache_timestamps.pop(cache_key, None)
            return None, False
        
        return self._permission_cache[cache_key], True
    
    def _cache_result(self, cache_key: str, result: bool) -> None:
        """Cache a permission check result."""
        self._permission_cache[cache_key] = result
        self._cache_timestamps[cache_key] = time.time()
    
    def _invalidate_cache(self) -> None:
        """Invalidate entire permission cache."""
        self._permission_cache.clear()
        self._cache_timestamps.clear()
        logger.debug("Permission cache invalidated")
    
    def _invalidate_cache_for_user(self, user_id: str) -> None:
        """Invalidate cache entries for a specific user."""
        keys_to_remove = [
            key for key in self._permission_cache.keys()
            if key.startswith(f"{user_id}:")
        ]
        
        for key in keys_to_remove:
            self._permission_cache.pop(key, None)
            self._cache_timestamps.pop(key, None)
        
        logger.debug(f"Invalidated {len(keys_to_remove)} cache entries for user {user_id}")
    
    def get_user_permissions_summary(self, user_id: str) -> Dict[str, Any]:
        """Get a summary of all permissions for a user."""
        user_roles = self.get_user_roles(user_id)
        
        permissions_by_resource = {}
        all_permissions = set()
        
        for role_name in user_roles:
            if role_name in self.roles:
                role = self.roles[role_name]
                for permission in role.permissions:
                    all_permissions.add(permission)
                    
                    resource_key = permission.resource_type.value
                    if resource_key not in permissions_by_resource:
                        permissions_by_resource[resource_key] = set()
                    
                    permissions_by_resource[resource_key].add(
                        f"{permission.action.value}:{permission.level.name}"
                    )
        
        return {
            "user_id": user_id,
            "roles": list(user_roles),
            "total_permissions": len(all_permissions),
            "permissions_by_resource": {
                k: list(v) for k, v in permissions_by_resource.items()
            }
        }
    
    def audit_permissions(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate an audit report of permissions."""
        audit_data = {
            "timestamp": time.time(),
            "total_roles": len(self.roles),
            "total_users": len(self.user_roles),
            "system_roles": [name for name, role in self.roles.items() if role.is_system],
            "custom_roles": [name for name, role in self.roles.items() if not role.is_system]
        }
        
        if user_id:
            audit_data["user_summary"] = self.get_user_permissions_summary(user_id)
        else:
            audit_data["users"] = {
                uid: list(roles) for uid, roles in self.user_roles.items()
            }
        
        logger.info("Permission audit completed", extra={"audit_summary": audit_data})
        return audit_data


# Default RBAC system instance
_default_rbac_system: Optional[RBACSystem] = None


def get_default_rbac_system() -> RBACSystem:
    """Get the default RBAC system instance."""
    global _default_rbac_system
    if _default_rbac_system is None:
        _default_rbac_system = RBACSystem()
    return _default_rbac_system


def check_user_permission(
    user_id: str,
    resource_type: ResourceType,
    resource_id: Optional[str],
    action: Action,
    required_level: PermissionLevel = PermissionLevel.READ
) -> bool:
    """Check user permission using default RBAC system."""
    rbac_system = get_default_rbac_system()
    return rbac_system.check_permission(user_id, resource_type, resource_id, action, required_level)


__all__ = [
    "RBACSystem",
    "Role", 
    "Permission",
    "ResourceType",
    "Action",
    "PermissionLevel",
    "get_default_rbac_system",
    "check_user_permission"
]