"""Execution context management for CLI v3 command engine.

This module provides the ExecutionContext class that manages user context,
permissions, resource limits, and session state throughout command execution.
"""

import asyncio
import logging
import time
try:
    import psutil
except ImportError:
    psutil = None
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, AsyncGenerator
from uuid import uuid4

from ..models.command_models import (
    ExecutionMode,
    ResourceLimit,
    ResourceType, 
    UserProfile,
    PermissionDeniedError,
    ResourceExhaustedError,
)


logger = logging.getLogger(__name__)


class SessionState:
    """Manages session-specific state and data."""
    
    def __init__(self, session_id: str = None):
        self.session_id = session_id or str(uuid4())
        self.created_at = datetime.now(timezone.utc)
        self.last_activity = self.created_at
        self.variables: Dict[str, Any] = {}
        self.command_history: List[str] = []
        self.context_stack: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
    
    async def set_variable(self, key: str, value: Any) -> None:
        """Set a session variable."""
        async with self._lock:
            self.variables[key] = value
            self.last_activity = datetime.now(timezone.utc)
    
    async def get_variable(self, key: str, default: Any = None) -> Any:
        """Get a session variable."""
        async with self._lock:
            return self.variables.get(key, default)
    
    async def push_context(self, context: Dict[str, Any]) -> None:
        """Push a new context frame onto the stack."""
        async with self._lock:
            self.context_stack.append(context.copy())
    
    async def pop_context(self) -> Optional[Dict[str, Any]]:
        """Pop the most recent context frame."""
        async with self._lock:
            return self.context_stack.pop() if self.context_stack else None
    
    async def get_current_context(self) -> Dict[str, Any]:
        """Get the current merged context."""
        async with self._lock:
            merged = {}
            for ctx in self.context_stack:
                merged.update(ctx)
            return merged


class ResourceMonitor:
    """Monitors and enforces resource limits during command execution."""
    
    def __init__(self, limits: Dict[ResourceType, ResourceLimit]):
        self.limits = limits
        self.start_time: Optional[float] = None
        self.start_memory: Optional[int] = None
        self.start_cpu: Optional[float] = None
        self._monitoring = False
        self._process = psutil.Process() if psutil else None
    
    def start_monitoring(self) -> None:
        """Start resource monitoring."""
        self.start_time = time.time()
        
        if self._process:
            try:
                memory_info = self._process.memory_info()
                self.start_memory = memory_info.rss // (1024 * 1024)  # MB
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self.start_memory = 0
            
            try:
                self.start_cpu = self._process.cpu_times().user
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self.start_cpu = 0.0
        else:
            self.start_memory = 0
            self.start_cpu = 0.0
        
        self._monitoring = True
        logger.debug(f"Started resource monitoring: mem={self.start_memory}MB, cpu={self.start_cpu}s")
    
    def check_limits(self) -> None:
        """Check current resource usage against limits."""
        if not self._monitoring:
            return
        
        current_time = time.time()
        
        # Check CPU time limit
        if self._process and ResourceType.CPU_TIME in self.limits:
            try:
                current_cpu = self._process.cpu_times().user
                cpu_used = current_cpu - self.start_cpu
                cpu_limit = self.limits[ResourceType.CPU_TIME]
                
                if cpu_used > cpu_limit.max_value:
                    raise ResourceExhaustedError(
                        f"CPU time limit exceeded: {cpu_used:.2f}s > {cpu_limit.max_value}s"
                    )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # Check memory limit
        if self._process and ResourceType.MEMORY in self.limits:
            try:
                current_memory = self._process.memory_info().rss // (1024 * 1024)
                memory_limit = self.limits[ResourceType.MEMORY]
                
                if current_memory > memory_limit.max_value:
                    raise ResourceExhaustedError(
                        f"Memory limit exceeded: {current_memory}MB > {memory_limit.max_value}MB"
                    )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    
    def get_usage(self) -> Dict[ResourceType, float]:
        """Get current resource usage."""
        if not self._monitoring:
            return {}
        
        usage = {}
        current_time = time.time()
        
        # CPU time usage
        if self._process and self.start_cpu is not None:
            try:
                current_cpu = self._process.cpu_times().user
                usage[ResourceType.CPU_TIME] = current_cpu - self.start_cpu
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                usage[ResourceType.CPU_TIME] = 0.0
        
        # Memory usage
        if self._process and self.start_memory is not None:
            try:
                current_memory = self._process.memory_info().rss // (1024 * 1024)
                usage[ResourceType.MEMORY] = current_memory
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                usage[ResourceType.MEMORY] = 0
        
        return usage
    
    def stop_monitoring(self) -> Dict[ResourceType, float]:
        """Stop monitoring and return final usage."""
        final_usage = self.get_usage()
        self._monitoring = False
        return final_usage


class PermissionManager:
    """Manages user permissions and access control."""
    
    def __init__(self, user_profile: UserProfile):
        self.user_profile = user_profile
        self._permissions: Set[str] = set()
        self._denied_permissions: Set[str] = set()
        
        # Set default permissions based on skill level
        self._set_default_permissions()
    
    def _set_default_permissions(self) -> None:
        """Set default permissions based on user skill level."""
        # Basic permissions for all users
        base_perms = {
            "command.help",
            "command.status", 
            "command.history",
            "file.read",
            "config.view"
        }
        
        # Additional permissions by skill level
        if self.user_profile.skill_level.value in ["intermediate", "expert"]:
            base_perms.update({
                "file.write",
                "config.edit",
                "command.batch",
                "system.info"
            })
        
        if self.user_profile.skill_level.value == "expert":
            base_perms.update({
                "system.admin",
                "command.dangerous",
                "file.execute",
                "network.access"
            })
        
        self._permissions = base_perms
    
    def grant_permission(self, permission: str) -> None:
        """Grant a specific permission."""
        self._permissions.add(permission)
        self._denied_permissions.discard(permission)
        logger.debug(f"Granted permission: {permission}")
    
    def deny_permission(self, permission: str) -> None:
        """Explicitly deny a permission."""
        self._denied_permissions.add(permission)
        self._permissions.discard(permission)
        logger.debug(f"Denied permission: {permission}")
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has a specific permission."""
        if permission in self._denied_permissions:
            return False
        
        return permission in self._permissions or self._check_wildcard_permission(permission)
    
    def _check_wildcard_permission(self, permission: str) -> bool:
        """Check for wildcard permissions (e.g., 'command.*' for 'command.help')."""
        parts = permission.split('.')
        for i in range(len(parts)):
            wildcard = '.'.join(parts[:i+1]) + '.*'
            if wildcard in self._permissions:
                return True
        return False
    
    def require_permission(self, permission: str) -> None:
        """Require a permission, raising exception if not granted."""
        if not self.has_permission(permission):
            raise PermissionDeniedError(
                f"Permission denied: {permission} (user skill level: {self.user_profile.skill_level})"
            )


class ExecutionContext:
    """Manages execution context for command processing.
    
    This class provides comprehensive context management including:
    - User authentication and permissions
    - Resource limits and monitoring 
    - Session state persistence
    - Interface mode adaptation
    - Audit trail generation
    """
    
    def __init__(
        self,
        user_profile: UserProfile,
        execution_mode: ExecutionMode = ExecutionMode.CLI,
        resource_limits: Optional[Dict[ResourceType, ResourceLimit]] = None,
        session_id: Optional[str] = None
    ):
        self.user_profile = user_profile
        self.execution_mode = execution_mode
        self.context_id = str(uuid4())
        self.created_at = datetime.now(timezone.utc)
        
        # Initialize core components
        self.session = SessionState(session_id)
        self.permissions = PermissionManager(user_profile)
        self.resource_monitor = ResourceMonitor(resource_limits or {})
        
        # Execution state
        self.current_command: Optional[str] = None
        self.is_active = False
        self._audit_entries: List[Dict[str, Any]] = []
        
        logger.info(f"Created execution context {self.context_id} for user {user_profile.user_id}")
    
    @property
    def capabilities(self) -> List[str]:
        """Get interface capabilities based on execution mode."""
        base_caps = ["text_output", "error_reporting"]
        
        if self.execution_mode == ExecutionMode.TUI:
            base_caps.extend([
                "interactive_input",
                "real_time_updates", 
                "visual_feedback",
                "keyboard_shortcuts"
            ])
        elif self.execution_mode == ExecutionMode.WEB_UI:
            base_caps.extend([
                "rich_formatting",
                "file_upload",
                "drag_drop",
                "visual_charts"
            ])
        elif self.execution_mode == ExecutionMode.API:
            base_caps.extend([
                "json_response",
                "structured_data",
                "bulk_operations"
            ])
        
        return base_caps
    
    def check_permission(self, permission: str) -> bool:
        """Check if current user has permission."""
        return self.permissions.has_permission(permission)
    
    def require_permission(self, permission: str) -> None:
        """Require permission or raise exception."""
        self.permissions.require_permission(permission)
    
    def adapt_for_skill_level(self, content: Any) -> Any:
        """Adapt content complexity for user skill level."""
        if self.user_profile.skill_level == "beginner":
            # Simplify output, add more explanations
            if isinstance(content, dict) and "suggestions" in content:
                # Limit suggestions for beginners
                suggestions = content.get("suggestions", [])[:3]
                content["suggestions"] = [
                    {**s, "explanation": f"This will {s.get('description', 'perform an action')}"} 
                    for s in suggestions
                ]
        elif self.user_profile.skill_level == "expert":
            # Show more technical details, advanced options
            if isinstance(content, dict):
                content["advanced_options"] = True
                content["technical_details"] = True
        
        return content
    
    @asynccontextmanager
    async def command_execution(
        self, 
        command_type: str,
        timeout_ms: Optional[int] = None
    ) -> AsyncGenerator[None, None]:
        """Async context manager for command execution with monitoring."""
        self.current_command = command_type
        self.is_active = True
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        start_time = datetime.now(timezone.utc)
        
        # Start monitoring task
        monitor_task = asyncio.create_task(self._periodic_resource_check())
        
        try:
            # Yield control to the command execution
            yield
            
        except Exception as e:
            # Log execution error
            self._add_audit_entry({
                "event": "command_error",
                "command": command_type,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc)
            })
            raise
        
        finally:
            # Stop monitoring
            self.is_active = False
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
            
            # Stop resource monitoring and get final usage
            final_usage = self.resource_monitor.stop_monitoring()
            
            # Record execution in audit trail
            duration = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self._add_audit_entry({
                "event": "command_completed",
                "command": command_type,
                "duration_ms": int(duration),
                "resource_usage": final_usage,
                "timestamp": datetime.now(timezone.utc)
            })
            
            self.current_command = None
    
    async def _periodic_resource_check(self) -> None:
        """Periodically check resource limits during execution."""
        while self.is_active:
            try:
                self.resource_monitor.check_limits()
                await asyncio.sleep(0.1)  # Check every 100ms
            except ResourceExhaustedError:
                # Resource limit exceeded - cancel execution
                raise
            except asyncio.CancelledError:
                break
    
    def _add_audit_entry(self, entry: Dict[str, Any]) -> None:
        """Add entry to audit trail."""
        entry.update({
            "context_id": self.context_id,
            "user_id": self.user_profile.user_id,
            "execution_mode": self.execution_mode.value
        })
        self._audit_entries.append(entry)
        
        # Keep only last 1000 entries to prevent memory bloat
        if len(self._audit_entries) > 1000:
            self._audit_entries = self._audit_entries[-1000:]
    
    def get_audit_trail(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit trail entries."""
        return self._audit_entries[-limit:] if self._audit_entries else []
    
    def get_context_info(self) -> Dict[str, Any]:
        """Get comprehensive context information for debugging."""
        return {
            "context_id": self.context_id,
            "user_id": self.user_profile.user_id,
            "skill_level": self.user_profile.skill_level.value,
            "execution_mode": self.execution_mode.value,
            "capabilities": self.capabilities,
            "is_active": self.is_active,
            "current_command": self.current_command,
            "session_id": self.session.session_id,
            "created_at": self.created_at.isoformat(),
            "permissions_count": len(self.permissions._permissions),
            "audit_entries_count": len(self._audit_entries)
        }