"""
Repository interfaces (ports) for the settings management system.

These define the contracts for data persistence without specifying
the implementation details, following the ports and adapters pattern.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime

from .entities import (
    SettingsNode,
    SettingsHierarchy,
    UserProfile, 
    AgentDefinition,
    AgentInstance,
    AuditEntry,
)
from .value_objects import (
    SettingsLevel,
    AgentStatus,
    PermissionLevel,
)


class SettingsRepository(ABC):
    """Repository interface for settings management."""
    
    @abstractmethod
    async def save_hierarchy(self, hierarchy: SettingsHierarchy) -> None:
        """Save a settings hierarchy."""
        pass
    
    @abstractmethod
    async def get_hierarchy(self, hierarchy_id: str) -> Optional[SettingsHierarchy]:
        """Get a settings hierarchy by ID."""
        pass
    
    @abstractmethod
    async def get_hierarchy_by_name(self, name: str) -> Optional[SettingsHierarchy]:
        """Get a settings hierarchy by name."""
        pass
    
    @abstractmethod
    async def save_node(self, node: SettingsNode) -> None:
        """Save a settings node."""
        pass
    
    @abstractmethod
    async def get_node(self, node_id: str) -> Optional[SettingsNode]:
        """Get a settings node by ID."""
        pass
    
    @abstractmethod
    async def get_nodes_by_hierarchy(self, hierarchy_id: str) -> List[SettingsNode]:
        """Get all nodes for a hierarchy."""
        pass
    
    @abstractmethod
    async def get_nodes_by_level(self, hierarchy_id: str, level: SettingsLevel) -> List[SettingsNode]:
        """Get nodes by hierarchy and level."""
        pass
    
    @abstractmethod
    async def get_user_settings_node(self, user_id: str) -> Optional[SettingsNode]:
        """Get the settings node for a specific user."""
        pass
    
    @abstractmethod
    async def delete_node(self, node_id: str) -> bool:
        """Delete a settings node."""
        pass
    
    @abstractmethod
    async def delete_hierarchy(self, hierarchy_id: str) -> bool:
        """Delete a settings hierarchy and all its nodes."""
        pass


class UserRepository(ABC):
    """Repository interface for user management."""
    
    @abstractmethod
    async def save_user(self, user: UserProfile) -> None:
        """Save a user profile."""
        pass
    
    @abstractmethod
    async def get_user(self, user_id: str) -> Optional[UserProfile]:
        """Get a user by ID."""
        pass
    
    @abstractmethod
    async def get_user_by_username(self, username: str) -> Optional[UserProfile]:
        """Get a user by username."""
        pass
    
    @abstractmethod
    async def get_user_by_email(self, email: str) -> Optional[UserProfile]:
        """Get a user by email."""
        pass
    
    @abstractmethod
    async def get_users_by_organization(self, organization_id: str) -> List[UserProfile]:
        """Get all users in an organization."""
        pass
    
    @abstractmethod
    async def search_users(self, query: str, limit: int = 50) -> List[UserProfile]:
        """Search users by name or email."""
        pass
    
    @abstractmethod
    async def delete_user(self, user_id: str) -> bool:
        """Delete a user profile."""
        pass
    
    @abstractmethod
    async def get_active_users(self, since: datetime) -> List[UserProfile]:
        """Get users active since a given time."""
        pass


class AgentRepository(ABC):
    """Repository interface for agent management."""
    
    @abstractmethod
    async def save_agent_definition(self, agent: AgentDefinition) -> None:
        """Save an agent definition."""
        pass
    
    @abstractmethod
    async def get_agent_definition(self, agent_id: str) -> Optional[AgentDefinition]:
        """Get an agent definition by ID."""
        pass
    
    @abstractmethod
    async def get_agent_definitions_by_owner(self, owner_id: str) -> List[AgentDefinition]:
        """Get all agent definitions owned by a user."""
        pass
    
    @abstractmethod
    async def get_agent_definitions_by_organization(self, organization_id: str) -> List[AgentDefinition]:
        """Get all agent definitions in an organization."""
        pass
    
    @abstractmethod
    async def get_agent_definitions_by_status(self, status: AgentStatus) -> List[AgentDefinition]:
        """Get agent definitions by status."""
        pass
    
    @abstractmethod
    async def search_agent_definitions(self, query: str, filters: Dict[str, Any], 
                                     limit: int = 50) -> List[AgentDefinition]:
        """Search agent definitions with filters."""
        pass
    
    @abstractmethod
    async def get_popular_agents(self, limit: int = 10) -> List[AgentDefinition]:
        """Get most popular agent definitions by usage."""
        pass
    
    @abstractmethod
    async def delete_agent_definition(self, agent_id: str) -> bool:
        """Delete an agent definition."""
        pass
    
    @abstractmethod
    async def save_agent_instance(self, instance: AgentInstance) -> None:
        """Save an agent instance."""
        pass
    
    @abstractmethod
    async def get_agent_instance(self, instance_id: str) -> Optional[AgentInstance]:
        """Get an agent instance by ID."""
        pass
    
    @abstractmethod
    async def get_agent_instances_by_user(self, user_id: str) -> List[AgentInstance]:
        """Get all agent instances for a user."""
        pass
    
    @abstractmethod
    async def get_agent_instances_by_definition(self, agent_id: str) -> List[AgentInstance]:
        """Get all instances of a specific agent definition."""
        pass
    
    @abstractmethod
    async def get_active_agent_instances(self) -> List[AgentInstance]:
        """Get all currently active agent instances."""
        pass
    
    @abstractmethod
    async def delete_agent_instance(self, instance_id: str) -> bool:
        """Delete an agent instance."""
        pass


class AuditRepository(ABC):
    """Repository interface for audit logging."""
    
    @abstractmethod
    async def save_audit_entry(self, entry: AuditEntry) -> None:
        """Save an audit log entry."""
        pass
    
    @abstractmethod
    async def get_audit_entry(self, entry_id: str) -> Optional[AuditEntry]:
        """Get an audit entry by ID."""
        pass
    
    @abstractmethod
    async def get_audit_entries_by_user(self, user_id: str, 
                                      start_time: Optional[datetime] = None,
                                      end_time: Optional[datetime] = None,
                                      limit: int = 100) -> List[AuditEntry]:
        """Get audit entries for a specific user."""
        pass
    
    @abstractmethod
    async def get_audit_entries_by_resource(self, resource_type: str, resource_id: str,
                                          start_time: Optional[datetime] = None,
                                          end_time: Optional[datetime] = None,
                                          limit: int = 100) -> List[AuditEntry]:
        """Get audit entries for a specific resource."""
        pass
    
    @abstractmethod
    async def get_audit_entries_by_action(self, action: str,
                                        start_time: Optional[datetime] = None,
                                        end_time: Optional[datetime] = None,
                                        limit: int = 100) -> List[AuditEntry]:
        """Get audit entries by action type."""
        pass
    
    @abstractmethod
    async def search_audit_entries(self, filters: Dict[str, Any],
                                 start_time: Optional[datetime] = None,
                                 end_time: Optional[datetime] = None,
                                 limit: int = 100) -> List[AuditEntry]:
        """Search audit entries with complex filters."""
        pass
    
    @abstractmethod
    async def delete_old_entries(self, before: datetime) -> int:
        """Delete audit entries older than specified date."""
        pass
    
    @abstractmethod
    async def get_activity_summary(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get activity summary for a user over specified days."""
        pass


class CacheRepository(ABC):
    """Repository interface for caching resolved settings and computations."""
    
    @abstractmethod
    async def get_resolved_settings(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get cached resolved settings for a node."""
        pass
    
    @abstractmethod
    async def set_resolved_settings(self, node_id: str, settings: Dict[str, Any], 
                                   ttl: int = 300) -> None:
        """Cache resolved settings for a node."""
        pass
    
    @abstractmethod
    async def invalidate_resolved_settings(self, node_id: str) -> None:
        """Invalidate cached settings for a node."""
        pass
    
    @abstractmethod
    async def invalidate_hierarchy_cache(self, hierarchy_id: str) -> None:
        """Invalidate all cached data for a hierarchy."""
        pass
    
    @abstractmethod
    async def get_validation_results(self, node_id: str, rule_version: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached validation results."""
        pass
    
    @abstractmethod
    async def set_validation_results(self, node_id: str, rule_version: str,
                                   results: List[Dict[str, Any]], ttl: int = 300) -> None:
        """Cache validation results."""
        pass
    
    @abstractmethod
    async def clear_all(self) -> None:
        """Clear all cached data."""
        pass


class SecretRepository(ABC):
    """Repository interface for secure storage of encrypted settings."""
    
    @abstractmethod
    async def store_secret(self, key: str, value: str, user_id: str) -> str:
        """Store an encrypted secret and return a reference key."""
        pass
    
    @abstractmethod
    async def retrieve_secret(self, reference_key: str, user_id: str) -> Optional[str]:
        """Retrieve and decrypt a secret by reference key."""
        pass
    
    @abstractmethod
    async def delete_secret(self, reference_key: str, user_id: str) -> bool:
        """Delete a secret by reference key."""
        pass
    
    @abstractmethod
    async def list_user_secrets(self, user_id: str) -> List[str]:
        """List reference keys for all secrets owned by a user."""
        pass
    
    @abstractmethod
    async def rotate_encryption_key(self, old_key: str, new_key: str) -> int:
        """Rotate encryption key and return number of secrets updated."""
        pass