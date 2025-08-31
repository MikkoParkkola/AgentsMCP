"""
Access Control Interface - Role-based access control (RBAC) management.

This component provides:
- Role-based permission management
- User access levels configuration
- Security settings interface
- Permission auditing and logging
- Group-based access control
- Session management and monitoring
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import hashlib
import secrets

from ..v2.event_system import AsyncEventSystem, Event, EventType
from ..v2.display_renderer import DisplayRenderer
from ..v2.terminal_manager import TerminalManager

logger = logging.getLogger(__name__)


class PermissionType(Enum):
    """Types of permissions in the system."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DELETE = "delete"
    ADMIN = "admin"


class ResourceType(Enum):
    """Types of resources that can be access-controlled."""
    SETTINGS = "settings"
    AGENTS = "agents"
    CONVERSATIONS = "conversations"
    SYSTEM = "system"
    LOGS = "logs"
    CONFIG = "config"


class UserRole(Enum):
    """Standard user roles with predefined permissions."""
    GUEST = "guest"
    USER = "user"
    POWER_USER = "power_user"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


class SessionStatus(Enum):
    """User session status."""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    LOCKED = "locked"


@dataclass
class Permission:
    """A specific permission on a resource."""
    resource_type: ResourceType
    resource_id: Optional[str]  # None for all resources of this type
    permission_type: PermissionType
    granted: bool = True
    conditions: Dict[str, Any] = field(default_factory=dict)  # e.g., time restrictions
    granted_by: Optional[str] = None
    granted_at: datetime = field(default_factory=datetime.now)


@dataclass
class Role:
    """A role containing a set of permissions."""
    name: str
    display_name: str
    description: str
    permissions: List[Permission] = field(default_factory=list)
    inherits_from: List[str] = field(default_factory=list)  # Role inheritance
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)


@dataclass
class User:
    """A user in the system."""
    id: str
    username: str
    display_name: str
    email: Optional[str] = None
    roles: List[str] = field(default_factory=list)
    additional_permissions: List[Permission] = field(default_factory=list)
    groups: List[str] = field(default_factory=list)
    
    # Security settings
    password_hash: Optional[str] = None
    require_2fa: bool = False
    account_locked: bool = False
    failed_login_attempts: int = 0
    last_login: Optional[datetime] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)


@dataclass
class Group:
    """A group of users for easier permission management."""
    id: str
    name: str
    description: str
    members: List[str] = field(default_factory=list)  # User IDs
    permissions: List[Permission] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class UserSession:
    """An active user session."""
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    status: SessionStatus
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    expires_at: Optional[datetime] = None


@dataclass
class SecurityAuditEntry:
    """A security audit log entry."""
    id: str
    timestamp: datetime
    user_id: Optional[str]
    action: str
    resource_type: Optional[ResourceType]
    resource_id: Optional[str]
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None


class AccessControlInterface:
    """
    Comprehensive access control interface for managing users, roles, and permissions.
    
    Features:
    - User management (create, edit, delete, lock/unlock)
    - Role-based access control with inheritance
    - Group-based permissions for easier management
    - Session monitoring and management
    - Security audit logging
    - Permission testing and validation
    - Bulk operations for users and roles
    """
    
    def __init__(self,
                 event_system: AsyncEventSystem,
                 display_renderer: DisplayRenderer,
                 terminal_manager: TerminalManager):
        """Initialize the access control interface."""
        self.event_system = event_system
        self.display_renderer = display_renderer
        self.terminal_manager = terminal_manager
        
        # Access control data
        self.users: Dict[str, User] = {}
        self.roles: Dict[str, Role] = {}
        self.groups: Dict[str, Group] = {}
        self.sessions: Dict[str, UserSession] = {}
        self.audit_log: List[SecurityAuditEntry] = []
        
        # UI state
        self.visible = False
        self.current_view = "users"  # "users", "roles", "groups", "sessions", "audit", "settings"
        self.selected_index = 0
        self.selected_items: Set[str] = set()  # For bulk operations
        self.bulk_mode = False
        
        # Current user context
        self.current_user_id: Optional[str] = None
        self.current_session_id: Optional[str] = None
        
        # Security settings
        self.security_settings = {
            "password_min_length": 8,
            "require_password_complexity": True,
            "session_timeout_minutes": 60,
            "max_failed_login_attempts": 5,
            "account_lockout_duration_minutes": 30,
            "enable_audit_logging": True,
            "require_2fa_for_admin": True
        }
        
        self._initialize_default_roles()
        self._initialize_default_users()
    
    async def initialize(self) -> bool:
        """Initialize the access control interface."""
        try:
            # Register event handlers
            await self._register_event_handlers()
            
            # Load existing data
            await self._load_access_control_data()
            
            # Start session cleanup task
            asyncio.create_task(self._session_cleanup_task())
            
            logger.info("Access control interface initialized successfully")
            return True
            
        except Exception as e:
            logger.exception(f"Failed to initialize access control interface: {e}")
            return False
    
    def _initialize_default_roles(self):
        """Initialize default system roles."""
        
        # Guest role - minimal permissions
        guest_permissions = [
            Permission(ResourceType.CONVERSATIONS, None, PermissionType.READ),
        ]
        self.roles["guest"] = Role(
            name="guest",
            display_name="Guest",
            description="Minimal access for guest users",
            permissions=guest_permissions
        )
        
        # User role - standard permissions
        user_permissions = [
            Permission(ResourceType.CONVERSATIONS, None, PermissionType.READ),
            Permission(ResourceType.CONVERSATIONS, None, PermissionType.WRITE),
            Permission(ResourceType.AGENTS, None, PermissionType.READ),
            Permission(ResourceType.SETTINGS, None, PermissionType.READ),
            Permission(ResourceType.SETTINGS, "user_preferences", PermissionType.WRITE),
        ]
        self.roles["user"] = Role(
            name="user",
            display_name="User",
            description="Standard user permissions",
            permissions=user_permissions,
            inherits_from=["guest"]
        )
        
        # Power User role - extended permissions
        power_user_permissions = [
            Permission(ResourceType.AGENTS, None, PermissionType.WRITE),
            Permission(ResourceType.AGENTS, None, PermissionType.EXECUTE),
            Permission(ResourceType.SETTINGS, None, PermissionType.WRITE),
            Permission(ResourceType.CONFIG, None, PermissionType.READ),
        ]
        self.roles["power_user"] = Role(
            name="power_user",
            display_name="Power User",
            description="Extended permissions for advanced users",
            permissions=power_user_permissions,
            inherits_from=["user"]
        )
        
        # Admin role - administrative permissions
        admin_permissions = [
            Permission(ResourceType.SYSTEM, None, PermissionType.READ),
            Permission(ResourceType.SYSTEM, None, PermissionType.WRITE),
            Permission(ResourceType.CONFIG, None, PermissionType.WRITE),
            Permission(ResourceType.LOGS, None, PermissionType.READ),
            Permission(ResourceType.AGENTS, None, PermissionType.DELETE),
        ]
        self.roles["admin"] = Role(
            name="admin",
            display_name="Administrator",
            description="Administrative access to system resources",
            permissions=admin_permissions,
            inherits_from=["power_user"]
        )
        
        # Super Admin role - full permissions
        super_admin_permissions = [
            Permission(ResourceType.SYSTEM, None, PermissionType.ADMIN),
            Permission(ResourceType.CONFIG, None, PermissionType.DELETE),
            Permission(ResourceType.LOGS, None, PermissionType.WRITE),
            Permission(ResourceType.SETTINGS, None, PermissionType.DELETE),
        ]
        self.roles["super_admin"] = Role(
            name="super_admin",
            display_name="Super Administrator",
            description="Full system access - use with extreme caution",
            permissions=super_admin_permissions,
            inherits_from=["admin"]
        )
    
    def _initialize_default_users(self):
        """Initialize default system users."""
        
        # Default admin user
        admin_user = User(
            id="admin",
            username="admin",
            display_name="System Administrator",
            email="admin@local.system",
            roles=["admin"],
            password_hash=self._hash_password("admin123"),  # Default password, should be changed
            require_2fa=True
        )
        self.users["admin"] = admin_user
        
        # Default guest user
        guest_user = User(
            id="guest",
            username="guest",
            display_name="Guest User",
            roles=["guest"]
        )
        self.users["guest"] = guest_user
    
    async def show(self, view: str = "users"):
        """Show the access control interface."""
        if self.visible:
            return
        
        self.visible = True
        self.current_view = view
        self.selected_index = 0
        
        await self._render_interface()
        
        # Emit interface shown event
        await self.event_system.emit(Event(
            event_type=EventType.UI,
            data={"action": "access_control_shown", "view": view}
        ))
    
    async def hide(self):
        """Hide the access control interface."""
        if not self.visible:
            return
        
        self.visible = False
        await self._clear_interface()
        
        # Emit interface hidden event
        await self.event_system.emit(Event(
            event_type=EventType.UI,
            data={"action": "access_control_hidden"}
        ))
    
    # User Management
    
    async def create_user(self, username: str, display_name: str, email: Optional[str] = None,
                         password: Optional[str] = None, roles: List[str] = None) -> str:
        """Create a new user."""
        # Generate unique user ID
        user_id = self._generate_id("user")
        
        # Validate username uniqueness
        if any(u.username == username for u in self.users.values()):
            raise ValueError(f"Username '{username}' already exists")
        
        # Validate roles
        if roles:
            invalid_roles = [r for r in roles if r not in self.roles]
            if invalid_roles:
                raise ValueError(f"Invalid roles: {invalid_roles}")
        
        # Create user
        user = User(
            id=user_id,
            username=username,
            display_name=display_name,
            email=email,
            roles=roles or ["user"],
            password_hash=self._hash_password(password) if password else None
        )
        
        self.users[user_id] = user
        
        # Log audit entry
        await self._log_audit_entry(
            action="user_created",
            resource_type=ResourceType.SYSTEM,
            resource_id=user_id,
            success=True,
            details={"username": username, "roles": roles}
        )
        
        logger.info(f"Created user: {username} ({user_id})")
        return user_id
    
    async def update_user(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing user."""
        if user_id not in self.users:
            raise ValueError(f"User {user_id} not found")
        
        user = self.users[user_id]
        
        # Apply updates
        for key, value in updates.items():
            if key == "password":
                user.password_hash = self._hash_password(value)
            elif key == "roles":
                # Validate roles
                invalid_roles = [r for r in value if r not in self.roles]
                if invalid_roles:
                    raise ValueError(f"Invalid roles: {invalid_roles}")
                user.roles = value
            elif hasattr(user, key):
                setattr(user, key, value)
        
        user.modified_at = datetime.now()
        
        # Log audit entry
        await self._log_audit_entry(
            action="user_updated",
            resource_type=ResourceType.SYSTEM,
            resource_id=user_id,
            success=True,
            details={"updates": list(updates.keys())}
        )
        
        logger.info(f"Updated user: {user.username} ({user_id})")
        return True
    
    async def delete_user(self, user_id: str) -> bool:
        """Delete a user."""
        if user_id not in self.users:
            raise ValueError(f"User {user_id} not found")
        
        # Prevent deletion of current user
        if user_id == self.current_user_id:
            raise ValueError("Cannot delete currently logged-in user")
        
        user = self.users[user_id]
        
        # Revoke all active sessions for this user
        await self._revoke_user_sessions(user_id)
        
        # Remove from groups
        for group in self.groups.values():
            if user_id in group.members:
                group.members.remove(user_id)
        
        # Remove user
        del self.users[user_id]
        
        # Log audit entry
        await self._log_audit_entry(
            action="user_deleted",
            resource_type=ResourceType.SYSTEM,
            resource_id=user_id,
            success=True,
            details={"username": user.username}
        )
        
        logger.info(f"Deleted user: {user.username} ({user_id})")
        return True
    
    async def lock_user(self, user_id: str, reason: str = "") -> bool:
        """Lock a user account."""
        if user_id not in self.users:
            raise ValueError(f"User {user_id} not found")
        
        user = self.users[user_id]
        user.account_locked = True
        user.modified_at = datetime.now()
        
        # Revoke active sessions
        await self._revoke_user_sessions(user_id)
        
        # Log audit entry
        await self._log_audit_entry(
            action="user_locked",
            resource_type=ResourceType.SYSTEM,
            resource_id=user_id,
            success=True,
            details={"username": user.username, "reason": reason}
        )
        
        logger.warning(f"Locked user account: {user.username} ({user_id})")
        return True
    
    async def unlock_user(self, user_id: str) -> bool:
        """Unlock a user account."""
        if user_id not in self.users:
            raise ValueError(f"User {user_id} not found")
        
        user = self.users[user_id]
        user.account_locked = False
        user.failed_login_attempts = 0
        user.modified_at = datetime.now()
        
        # Log audit entry
        await self._log_audit_entry(
            action="user_unlocked",
            resource_type=ResourceType.SYSTEM,
            resource_id=user_id,
            success=True,
            details={"username": user.username}
        )
        
        logger.info(f"Unlocked user account: {user.username} ({user_id})")
        return True
    
    # Role Management
    
    async def create_role(self, name: str, display_name: str, description: str,
                         permissions: List[Permission] = None) -> bool:
        """Create a new role."""
        if name in self.roles:
            raise ValueError(f"Role '{name}' already exists")
        
        role = Role(
            name=name,
            display_name=display_name,
            description=description,
            permissions=permissions or []
        )
        
        self.roles[name] = role
        
        # Log audit entry
        await self._log_audit_entry(
            action="role_created",
            resource_type=ResourceType.SYSTEM,
            resource_id=name,
            success=True,
            details={"display_name": display_name}
        )
        
        logger.info(f"Created role: {display_name} ({name})")
        return True
    
    async def update_role(self, name: str, updates: Dict[str, Any]) -> bool:
        """Update an existing role."""
        if name not in self.roles:
            raise ValueError(f"Role '{name}' not found")
        
        role = self.roles[name]
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(role, key):
                setattr(role, key, value)
        
        role.modified_at = datetime.now()
        
        # Log audit entry
        await self._log_audit_entry(
            action="role_updated",
            resource_type=ResourceType.SYSTEM,
            resource_id=name,
            success=True,
            details={"updates": list(updates.keys())}
        )
        
        logger.info(f"Updated role: {role.display_name} ({name})")
        return True
    
    async def delete_role(self, name: str) -> bool:
        """Delete a role."""
        if name not in self.roles:
            raise ValueError(f"Role '{name}' not found")
        
        # Prevent deletion of system roles
        if name in ["guest", "user", "admin", "super_admin"]:
            raise ValueError("Cannot delete system roles")
        
        # Remove role from all users
        for user in self.users.values():
            if name in user.roles:
                user.roles.remove(name)
        
        # Remove role inheritance references
        for role in self.roles.values():
            if name in role.inherits_from:
                role.inherits_from.remove(name)
        
        role_display_name = self.roles[name].display_name
        del self.roles[name]
        
        # Log audit entry
        await self._log_audit_entry(
            action="role_deleted",
            resource_type=ResourceType.SYSTEM,
            resource_id=name,
            success=True,
            details={"display_name": role_display_name}
        )
        
        logger.info(f"Deleted role: {role_display_name} ({name})")
        return True
    
    # Permission Checking
    
    async def check_permission(self, user_id: str, resource_type: ResourceType,
                              resource_id: Optional[str], permission_type: PermissionType) -> bool:
        """Check if user has specific permission."""
        if user_id not in self.users:
            return False
        
        user = self.users[user_id]
        
        # Check if account is locked
        if user.account_locked:
            return False
        
        # Collect all permissions for user
        all_permissions = []
        
        # Direct user permissions
        all_permissions.extend(user.additional_permissions)
        
        # Role permissions (with inheritance)
        for role_name in user.roles:
            role_permissions = self._get_role_permissions(role_name)
            all_permissions.extend(role_permissions)
        
        # Group permissions
        for group_id in user.groups:
            if group_id in self.groups:
                all_permissions.extend(self.groups[group_id].permissions)
        
        # Check permissions
        for perm in all_permissions:
            if (perm.resource_type == resource_type and
                perm.permission_type == permission_type and
                (perm.resource_id is None or perm.resource_id == resource_id) and
                perm.granted):
                
                # Check conditions (e.g., time restrictions)
                if self._check_permission_conditions(perm):
                    return True
        
        return False
    
    def _get_role_permissions(self, role_name: str) -> List[Permission]:
        """Get all permissions for a role including inherited permissions."""
        if role_name not in self.roles:
            return []
        
        role = self.roles[role_name]
        all_permissions = role.permissions.copy()
        
        # Add inherited permissions
        for parent_role in role.inherits_from:
            parent_permissions = self._get_role_permissions(parent_role)
            all_permissions.extend(parent_permissions)
        
        return all_permissions
    
    def _check_permission_conditions(self, permission: Permission) -> bool:
        """Check if permission conditions are met."""
        # TODO: Implement time-based and other conditional checks
        return True
    
    # Session Management
    
    async def create_session(self, user_id: str, ip_address: Optional[str] = None,
                           user_agent: Optional[str] = None) -> str:
        """Create a new user session."""
        if user_id not in self.users:
            raise ValueError(f"User {user_id} not found")
        
        user = self.users[user_id]
        
        # Check if user is locked
        if user.account_locked:
            raise ValueError("User account is locked")
        
        # Generate session ID
        session_id = secrets.token_urlsafe(32)
        
        # Calculate expiration
        timeout_minutes = self.security_settings["session_timeout_minutes"]
        expires_at = datetime.now() + timedelta(minutes=timeout_minutes)
        
        # Create session
        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            status=SessionStatus.ACTIVE,
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=expires_at
        )
        
        self.sessions[session_id] = session
        
        # Update user last login
        user.last_login = datetime.now()
        
        # Log audit entry
        await self._log_audit_entry(
            action="session_created",
            resource_type=ResourceType.SYSTEM,
            resource_id=session_id,
            success=True,
            details={"user_id": user_id},
            ip_address=ip_address
        )
        
        logger.info(f"Created session for user {user.username}: {session_id}")
        return session_id
    
    async def validate_session(self, session_id: str) -> Optional[str]:
        """Validate a session and return user ID if valid."""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        # Check if session is active
        if session.status != SessionStatus.ACTIVE:
            return None
        
        # Check if expired
        if session.expires_at and datetime.now() > session.expires_at:
            session.status = SessionStatus.EXPIRED
            return None
        
        # Update last activity
        session.last_activity = datetime.now()
        
        return session.user_id
    
    async def revoke_session(self, session_id: str) -> bool:
        """Revoke a specific session."""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        session.status = SessionStatus.REVOKED
        
        # Log audit entry
        await self._log_audit_entry(
            action="session_revoked",
            resource_type=ResourceType.SYSTEM,
            resource_id=session_id,
            success=True,
            details={"user_id": session.user_id}
        )
        
        logger.info(f"Revoked session: {session_id}")
        return True
    
    async def _revoke_user_sessions(self, user_id: str):
        """Revoke all sessions for a specific user."""
        for session in self.sessions.values():
            if session.user_id == user_id and session.status == SessionStatus.ACTIVE:
                session.status = SessionStatus.REVOKED
        
        logger.info(f"Revoked all sessions for user: {user_id}")
    
    # Navigation and UI
    
    def set_view(self, view: str):
        """Set the current view."""
        if view in ["users", "roles", "groups", "sessions", "audit", "settings"]:
            self.current_view = view
            self.selected_index = 0
            asyncio.create_task(self._render_interface())
    
    def navigate_items(self, direction: int):
        """Navigate through items in current view."""
        if self.current_view == "users":
            max_idx = len(self.users) - 1
        elif self.current_view == "roles":
            max_idx = len(self.roles) - 1
        elif self.current_view == "groups":
            max_idx = len(self.groups) - 1
        elif self.current_view == "sessions":
            active_sessions = [s for s in self.sessions.values() if s.status == SessionStatus.ACTIVE]
            max_idx = len(active_sessions) - 1
        elif self.current_view == "audit":
            max_idx = len(self.audit_log) - 1
        else:
            max_idx = 0
        
        if max_idx >= 0:
            self.selected_index = max(0, min(max_idx, self.selected_index + direction))
            asyncio.create_task(self._render_interface())
    
    def toggle_bulk_mode(self):
        """Toggle bulk operations mode."""
        self.bulk_mode = not self.bulk_mode
        if not self.bulk_mode:
            self.selected_items.clear()
        
        asyncio.create_task(self._render_interface())
    
    def toggle_item_selection(self):
        """Toggle selection of current item for bulk operations."""
        if not self.bulk_mode:
            return
        
        # Get current item ID based on view
        item_id = self._get_current_item_id()
        if item_id:
            if item_id in self.selected_items:
                self.selected_items.remove(item_id)
            else:
                self.selected_items.add(item_id)
            
            asyncio.create_task(self._render_interface())
    
    def _get_current_item_id(self) -> Optional[str]:
        """Get the ID of the currently selected item."""
        if self.current_view == "users":
            user_ids = list(self.users.keys())
            if self.selected_index < len(user_ids):
                return user_ids[self.selected_index]
        elif self.current_view == "roles":
            role_names = list(self.roles.keys())
            if self.selected_index < len(role_names):
                return role_names[self.selected_index]
        elif self.current_view == "groups":
            group_ids = list(self.groups.keys())
            if self.selected_index < len(group_ids):
                return group_ids[self.selected_index]
        elif self.current_view == "sessions":
            session_ids = [s.session_id for s in self.sessions.values() if s.status == SessionStatus.ACTIVE]
            if self.selected_index < len(session_ids):
                return session_ids[self.selected_index]
        
        return None
    
    # Rendering
    
    async def _render_interface(self):
        """Render the access control interface."""
        if not self.visible:
            return
        
        try:
            caps = self.terminal_manager.detect_capabilities()
            width, height = caps.width, caps.height
            
            if self.current_view == "users":
                content = self._render_users_view(width, height)
            elif self.current_view == "roles":
                content = self._render_roles_view(width, height)
            elif self.current_view == "groups":
                content = self._render_groups_view(width, height)
            elif self.current_view == "sessions":
                content = self._render_sessions_view(width, height)
            elif self.current_view == "audit":
                content = self._render_audit_view(width, height)
            elif self.current_view == "settings":
                content = self._render_settings_view(width, height)
            else:
                content = ["Unknown view"]
            
            # Update display
            self.display_renderer.update_region(
                "access_control",
                "\n".join(content),
                force=True
            )
            
        except Exception as e:
            logger.exception(f"Error rendering access control interface: {e}")
    
    def _render_users_view(self, width: int, height: int) -> List[str]:
        """Render the users management view."""
        lines = []
        
        # Header
        bulk_indicator = " [BULK]" if self.bulk_mode else ""
        title = f"╔══ Access Control - Users{bulk_indicator} ══╗".center(width)
        lines.append(title)
        
        # Stats
        total_users = len(self.users)
        locked_users = len([u for u in self.users.values() if u.account_locked])
        active_sessions = len([s for s in self.sessions.values() if s.status == SessionStatus.ACTIVE])
        
        stats = f"Total: {total_users} | Locked: {locked_users} | Active Sessions: {active_sessions}"
        if self.bulk_mode:
            stats += f" | Selected: {len(self.selected_items)}"
        lines.append(stats.center(width))
        lines.append("═" * width)
        
        # User list
        user_ids = list(self.users.keys())
        if not user_ids:
            lines.append("No users found".center(width))
            return lines
        
        for idx, user_id in enumerate(user_ids):
            if idx >= height - 6:  # Reserve space for header/footer
                lines.append("... (more users)")
                break
            
            user = self.users[user_id]
            is_current = idx == self.selected_index
            is_selected = user_id in self.selected_items
            
            # Indicators
            cursor = "►" if is_current else " "
            bulk_marker = "☑" if is_selected else "☐" if self.bulk_mode else " "
            lock_icon = "🔒" if user.account_locked else "🔓"
            admin_icon = "👑" if "admin" in user.roles else " "
            
            # User line
            roles_str = ", ".join(user.roles)
            user_line = f"{cursor}{bulk_marker} {lock_icon} {admin_icon} {user.display_name} ({user.username})"
            if len(user_line) < width - 20:
                user_line += f" - {roles_str}"
            
            lines.append(user_line[:width])
        
        # Footer
        lines.append("─" * width)
        lines.append("Enter: Edit | L: Lock/Unlock | D: Delete | N: New | B: Bulk")
        
        return lines
    
    def _render_roles_view(self, width: int, height: int) -> List[str]:
        """Render the roles management view."""
        lines = []
        
        # Header
        title = "╔══ Access Control - Roles ══╗".center(width)
        lines.append(title)
        lines.append(f"Manage {len(self.roles)} system roles".center(width))
        lines.append("═" * width)
        
        # Role list
        role_names = list(self.roles.keys())
        for idx, role_name in enumerate(role_names):
            if idx >= height - 6:
                lines.append("... (more roles)")
                break
            
            role = self.roles[role_name]
            is_current = idx == self.selected_index
            cursor = "►" if is_current else " "
            
            # System role indicator
            system_icon = "🛡️" if role_name in ["guest", "user", "admin", "super_admin"] else " "
            
            # Permission count
            total_perms = len(role.permissions)
            inherited_perms = sum(len(self._get_role_permissions(parent)) for parent in role.inherits_from)
            
            role_line = f"{cursor} {system_icon} {role.display_name} ({total_perms + inherited_perms} perms)"
            if role.inherits_from:
                role_line += f" ← {', '.join(role.inherits_from)}"
            
            lines.append(role_line[:width])
        
        # Footer
        lines.append("─" * width)
        lines.append("Enter: Edit | D: Delete | N: New | U: Users")
        
        return lines
    
    def _render_sessions_view(self, width: int, height: int) -> List[str]:
        """Render the active sessions view."""
        lines = []
        
        # Header
        title = "╔══ Access Control - Active Sessions ══╗".center(width)
        lines.append(title)
        
        active_sessions = [s for s in self.sessions.values() if s.status == SessionStatus.ACTIVE]
        lines.append(f"Monitoring {len(active_sessions)} active sessions".center(width))
        lines.append("═" * width)
        
        # Session list
        if not active_sessions:
            lines.append("No active sessions".center(width))
            return lines
        
        for idx, session in enumerate(active_sessions):
            if idx >= height - 6:
                lines.append("... (more sessions)")
                break
            
            is_current = idx == self.selected_index
            cursor = "►" if is_current else " "
            
            user = self.users.get(session.user_id)
            username = user.username if user else "Unknown"
            
            # Time calculations
            duration = datetime.now() - session.created_at
            duration_str = f"{int(duration.total_seconds() // 60)}min"
            
            session_line = f"{cursor} {username} | {duration_str} | {session.ip_address or 'Unknown IP'}"
            lines.append(session_line[:width])
        
        # Footer
        lines.append("─" * width)
        lines.append("Enter: Details | R: Revoke | U: Users | A: Audit")
        
        return lines
    
    def _render_audit_view(self, width: int, height: int) -> List[str]:
        """Render the security audit log view."""
        lines = []
        
        # Header
        title = "╔══ Access Control - Security Audit Log ══╗".center(width)
        lines.append(title)
        lines.append(f"Showing {len(self.audit_log)} recent audit entries".center(width))
        lines.append("═" * width)
        
        # Audit log (most recent first)
        recent_entries = sorted(self.audit_log, key=lambda x: x.timestamp, reverse=True)
        
        for idx, entry in enumerate(recent_entries):
            if idx >= height - 6:
                lines.append("... (older entries)")
                break
            
            is_current = idx == self.selected_index
            cursor = "►" if is_current else " "
            
            # Status icon
            status_icon = "✅" if entry.success else "❌"
            
            # Format timestamp
            time_str = entry.timestamp.strftime("%H:%M:%S")
            
            # User info
            user = self.users.get(entry.user_id) if entry.user_id else None
            user_str = user.username if user else "System"
            
            audit_line = f"{cursor} {time_str} {status_icon} {user_str} - {entry.action}"
            lines.append(audit_line[:width])
        
        # Footer
        lines.append("─" * width)
        lines.append("Enter: Details | C: Clear Log | U: Users | S: Sessions")
        
        return lines
    
    def _render_settings_view(self, width: int, height: int) -> List[str]:
        """Render the security settings view."""
        lines = []
        
        # Header
        title = "╔══ Access Control - Security Settings ══╗".center(width)
        lines.append(title)
        lines.append("Configure system security policies".center(width))
        lines.append("═" * width)
        
        # Settings
        settings_items = [
            ("Password Min Length", self.security_settings["password_min_length"]),
            ("Password Complexity", "Yes" if self.security_settings["require_password_complexity"] else "No"),
            ("Session Timeout (min)", self.security_settings["session_timeout_minutes"]),
            ("Max Failed Logins", self.security_settings["max_failed_login_attempts"]),
            ("Account Lockout (min)", self.security_settings["account_lockout_duration_minutes"]),
            ("Audit Logging", "Enabled" if self.security_settings["enable_audit_logging"] else "Disabled"),
            ("Admin 2FA Required", "Yes" if self.security_settings["require_2fa_for_admin"] else "No")
        ]
        
        for idx, (name, value) in enumerate(settings_items):
            is_current = idx == self.selected_index
            cursor = "►" if is_current else " "
            
            setting_line = f"{cursor} {name}: {value}"
            lines.append(setting_line[:width])
        
        # Footer
        lines.append("─" * width)
        lines.append("Enter: Edit | R: Reset Defaults | U: Users")
        
        return lines
    
    # Helper methods
    
    def _generate_id(self, prefix: str = "") -> str:
        """Generate a unique ID."""
        import uuid
        return f"{prefix}_{uuid.uuid4().hex[:8]}" if prefix else uuid.uuid4().hex[:8]
    
    def _hash_password(self, password: str) -> str:
        """Hash a password using secure hashing."""
        # Use bcrypt or similar in production
        import hashlib
        return hashlib.sha256(password.encode()).hexdigest()
    
    async def _log_audit_entry(self, action: str, resource_type: Optional[ResourceType] = None,
                             resource_id: Optional[str] = None, success: bool = True,
                             details: Dict[str, Any] = None, ip_address: Optional[str] = None):
        """Log a security audit entry."""
        if not self.security_settings.get("enable_audit_logging", True):
            return
        
        entry = SecurityAuditEntry(
            id=self._generate_id("audit"),
            timestamp=datetime.now(),
            user_id=self.current_user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            success=success,
            details=details or {},
            ip_address=ip_address
        )
        
        self.audit_log.append(entry)
        
        # Keep only recent entries
        if len(self.audit_log) > 1000:
            self.audit_log = self.audit_log[-1000:]
        
        # Emit audit event
        await self.event_system.emit(Event(
            event_type=EventType.SECURITY,
            data={"action": "audit_entry", "entry": entry}
        ))
    
    async def _session_cleanup_task(self):
        """Background task to clean up expired sessions."""
        while True:
            try:
                current_time = datetime.now()
                expired_sessions = []
                
                for session_id, session in self.sessions.items():
                    if (session.status == SessionStatus.ACTIVE and 
                        session.expires_at and 
                        current_time > session.expires_at):
                        session.status = SessionStatus.EXPIRED
                        expired_sessions.append(session_id)
                
                if expired_sessions:
                    logger.info(f"Expired {len(expired_sessions)} sessions")
                
                # Sleep for 1 minute before next cleanup
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.exception(f"Error in session cleanup task: {e}")
                await asyncio.sleep(60)
    
    async def _register_event_handlers(self):
        """Register event handlers for access control interface."""
        
        async def handle_keyboard_event(event: Event):
            if not self.visible or event.event_type != EventType.KEYBOARD:
                return
            
            key = event.data.get('key', '')
            
            # Global navigation
            if key == 'escape':
                await self.hide()
            elif key == 'up':
                self.navigate_items(-1)
            elif key == 'down':
                self.navigate_items(1)
            elif key == 'u':
                self.set_view("users")
            elif key == 'r' and self.current_view != "roles":
                self.set_view("roles")
            elif key == 's' and self.current_view != "settings":
                self.set_view("settings")
            elif key == 'a' and self.current_view != "audit":
                self.set_view("audit")
            
            # View-specific actions
            elif self.current_view == "users":
                if key == 'l':
                    # Lock/unlock selected user
                    item_id = self._get_current_item_id()
                    if item_id and item_id != self.current_user_id:
                        user = self.users[item_id]
                        if user.account_locked:
                            await self.unlock_user(item_id)
                        else:
                            await self.lock_user(item_id)
                        await self._render_interface()
                elif key == 'b':
                    self.toggle_bulk_mode()
                elif key == 'x' and self.bulk_mode:
                    self.toggle_item_selection()
                elif key == 'd':
                    # Delete user (with confirmation in real implementation)
                    item_id = self._get_current_item_id()
                    if item_id and item_id != self.current_user_id:
                        await self.delete_user(item_id)
                        await self._render_interface()
            
            elif self.current_view == "sessions":
                if key == 'r':
                    # Revoke selected session
                    item_id = self._get_current_item_id()
                    if item_id:
                        await self.revoke_session(item_id)
                        await self._render_interface()
        
        await self.event_system.subscribe(EventType.KEYBOARD, handle_keyboard_event)
    
    async def _load_access_control_data(self):
        """Load access control data from storage."""
        # TODO: Load from persistent storage
        pass
    
    async def _clear_interface(self):
        """Clear the access control interface."""
        if hasattr(self.display_renderer, 'clear_region'):
            self.display_renderer.clear_region("access_control")
    
    async def cleanup(self):
        """Cleanup access control interface resources."""
        if self.visible:
            await self.hide()
        
        # TODO: Save access control data to persistent storage
        
        logger.info("Access control interface cleanup completed")


# Factory function for easy integration
def create_access_control_interface(event_system: AsyncEventSystem,
                                  display_renderer: DisplayRenderer,
                                  terminal_manager: TerminalManager) -> AccessControlInterface:
    """Create and return a configured access control interface instance."""
    return AccessControlInterface(
        event_system=event_system,
        display_renderer=display_renderer,
        terminal_manager=terminal_manager
    )