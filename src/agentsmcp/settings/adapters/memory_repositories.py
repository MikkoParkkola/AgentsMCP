"""
In-memory repository implementations for testing and development.

These implementations store data in memory and are suitable
for testing, development, and small-scale deployments.
"""

from __future__ import annotations

import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from ..domain.entities import (
    SettingsNode,
    SettingsHierarchy,
    UserProfile,
    AgentDefinition,
    AgentInstance,
    AuditEntry,
)
from ..domain.value_objects import SettingsLevel, AgentStatus, PermissionLevel
from ..domain.repositories import (
    SettingsRepository,
    UserRepository,
    AgentRepository,
    AuditRepository,
    CacheRepository,
    SecretRepository,
)


class InMemorySettingsRepository(SettingsRepository):
    """In-memory implementation of settings repository."""
    
    def __init__(self):
        self.hierarchies: Dict[str, SettingsHierarchy] = {}
        self.nodes: Dict[str, SettingsNode] = {}
        self.hierarchy_by_name: Dict[str, str] = {}  # name -> id mapping
    
    async def save_hierarchy(self, hierarchy: SettingsHierarchy) -> None:
        self.hierarchies[hierarchy.id] = hierarchy
        self.hierarchy_by_name[hierarchy.name] = hierarchy.id
    
    async def get_hierarchy(self, hierarchy_id: str) -> Optional[SettingsHierarchy]:
        return self.hierarchies.get(hierarchy_id)
    
    async def get_hierarchy_by_name(self, name: str) -> Optional[SettingsHierarchy]:
        hierarchy_id = self.hierarchy_by_name.get(name)
        if hierarchy_id:
            return self.hierarchies.get(hierarchy_id)
        return None
    
    async def save_node(self, node: SettingsNode) -> None:
        self.nodes[node.id] = node
    
    async def get_node(self, node_id: str) -> Optional[SettingsNode]:
        return self.nodes.get(node_id)
    
    async def get_nodes_by_hierarchy(self, hierarchy_id: str) -> List[SettingsNode]:
        hierarchy = self.hierarchies.get(hierarchy_id)
        if not hierarchy:
            return []
        
        return [self.nodes[node_id] for node_id in hierarchy.nodes.keys() 
                if node_id in self.nodes]
    
    async def get_nodes_by_level(self, hierarchy_id: str, level: SettingsLevel) -> List[SettingsNode]:
        nodes = await self.get_nodes_by_hierarchy(hierarchy_id)
        return [node for node in nodes if node.level == level]
    
    async def get_user_settings_node(self, user_id: str) -> Optional[SettingsNode]:
        # Find user's settings node
        for node in self.nodes.values():
            if (node.level == SettingsLevel.USER and 
                node.metadata.get("user_id") == user_id):
                return node
        return None
    
    async def delete_node(self, node_id: str) -> bool:
        if node_id in self.nodes:
            del self.nodes[node_id]
            # Remove from hierarchies
            for hierarchy in self.hierarchies.values():
                if node_id in hierarchy.nodes:
                    del hierarchy.nodes[node_id]
            return True
        return False
    
    async def delete_hierarchy(self, hierarchy_id: str) -> bool:
        hierarchy = self.hierarchies.get(hierarchy_id)
        if not hierarchy:
            return False
        
        # Delete all nodes in hierarchy
        for node_id in list(hierarchy.nodes.keys()):
            await self.delete_node(node_id)
        
        # Delete hierarchy
        del self.hierarchies[hierarchy_id]
        # Remove from name mapping
        for name, hid in list(self.hierarchy_by_name.items()):
            if hid == hierarchy_id:
                del self.hierarchy_by_name[name]
                break
        
        return True


class InMemoryUserRepository(UserRepository):
    """In-memory implementation of user repository."""
    
    def __init__(self):
        self.users: Dict[str, UserProfile] = {}
        self.username_to_id: Dict[str, str] = {}
        self.email_to_id: Dict[str, str] = {}
    
    async def save_user(self, user: UserProfile) -> None:
        self.users[user.id] = user
        self.username_to_id[user.username] = user.id
        self.email_to_id[user.email] = user.id
    
    async def get_user(self, user_id: str) -> Optional[UserProfile]:
        return self.users.get(user_id)
    
    async def get_user_by_username(self, username: str) -> Optional[UserProfile]:
        user_id = self.username_to_id.get(username)
        return self.users.get(user_id) if user_id else None
    
    async def get_user_by_email(self, email: str) -> Optional[UserProfile]:
        user_id = self.email_to_id.get(email)
        return self.users.get(user_id) if user_id else None
    
    async def get_users_by_organization(self, organization_id: str) -> List[UserProfile]:
        return [user for user in self.users.values() 
                if user.organization_id == organization_id]
    
    async def search_users(self, query: str, limit: int = 50) -> List[UserProfile]:
        query = query.lower()
        results = []
        
        for user in self.users.values():
            if (query in user.username.lower() or 
                query in user.email.lower() or 
                query in user.full_name.lower()):
                results.append(user)
                if len(results) >= limit:
                    break
        
        return results
    
    async def delete_user(self, user_id: str) -> bool:
        user = self.users.get(user_id)
        if not user:
            return False
        
        del self.users[user_id]
        del self.username_to_id[user.username]
        del self.email_to_id[user.email]
        return True
    
    async def get_active_users(self, since: datetime) -> List[UserProfile]:
        return [user for user in self.users.values() 
                if user.is_active and user.last_active and user.last_active >= since]


class InMemoryAgentRepository(AgentRepository):
    """In-memory implementation of agent repository."""
    
    def __init__(self):
        self.agent_definitions: Dict[str, AgentDefinition] = {}
        self.agent_instances: Dict[str, AgentInstance] = {}
    
    async def save_agent_definition(self, agent: AgentDefinition) -> None:
        self.agent_definitions[agent.id] = agent
    
    async def get_agent_definition(self, agent_id: str) -> Optional[AgentDefinition]:
        return self.agent_definitions.get(agent_id)
    
    async def get_agent_definitions_by_owner(self, owner_id: str) -> List[AgentDefinition]:
        return [agent for agent in self.agent_definitions.values() 
                if agent.owner_id == owner_id]
    
    async def get_agent_definitions_by_organization(self, organization_id: str) -> List[AgentDefinition]:
        return [agent for agent in self.agent_definitions.values() 
                if agent.organization_id == organization_id]
    
    async def get_agent_definitions_by_status(self, status: AgentStatus) -> List[AgentDefinition]:
        return [agent for agent in self.agent_definitions.values() 
                if agent.status == status]
    
    async def search_agent_definitions(self, query: str, filters: Dict[str, Any], 
                                     limit: int = 50) -> List[AgentDefinition]:
        results = []
        query = query.lower() if query else ""
        
        for agent in self.agent_definitions.values():
            # Text search
            if query and query not in agent.name.lower() and query not in agent.description.lower():
                continue
            
            # Apply filters
            if "status" in filters and agent.status != filters["status"]:
                continue
            if "category" in filters and agent.category != filters["category"]:
                continue
            if "tags" in filters:
                filter_tags = filters["tags"]
                if not any(tag in agent.tags for tag in filter_tags):
                    continue
            
            results.append(agent)
            if len(results) >= limit:
                break
        
        return results
    
    async def get_popular_agents(self, limit: int = 10) -> List[AgentDefinition]:
        # Sort by usage count
        sorted_agents = sorted(self.agent_definitions.values(), 
                             key=lambda a: a.usage_count, reverse=True)
        return sorted_agents[:limit]
    
    async def delete_agent_definition(self, agent_id: str) -> bool:
        if agent_id in self.agent_definitions:
            del self.agent_definitions[agent_id]
            return True
        return False
    
    async def save_agent_instance(self, instance: AgentInstance) -> None:
        self.agent_instances[instance.id] = instance
    
    async def get_agent_instance(self, instance_id: str) -> Optional[AgentInstance]:
        return self.agent_instances.get(instance_id)
    
    async def get_agent_instances_by_user(self, user_id: str) -> List[AgentInstance]:
        return [instance for instance in self.agent_instances.values() 
                if instance.user_id == user_id]
    
    async def get_agent_instances_by_definition(self, agent_id: str) -> List[AgentInstance]:
        return [instance for instance in self.agent_instances.values() 
                if instance.agent_definition_id == agent_id]
    
    async def get_active_agent_instances(self) -> List[AgentInstance]:
        return [instance for instance in self.agent_instances.values() 
                if instance.status == AgentStatus.ACTIVE]
    
    async def delete_agent_instance(self, instance_id: str) -> bool:
        if instance_id in self.agent_instances:
            del self.agent_instances[instance_id]
            return True
        return False


class InMemoryAuditRepository(AuditRepository):
    """In-memory implementation of audit repository."""
    
    def __init__(self):
        self.audit_entries: Dict[str, AuditEntry] = {}
    
    async def save_audit_entry(self, entry: AuditEntry) -> None:
        self.audit_entries[entry.id] = entry
    
    async def get_audit_entry(self, entry_id: str) -> Optional[AuditEntry]:
        return self.audit_entries.get(entry_id)
    
    async def get_audit_entries_by_user(self, user_id: str, 
                                      start_time: Optional[datetime] = None,
                                      end_time: Optional[datetime] = None,
                                      limit: int = 100) -> List[AuditEntry]:
        entries = [entry for entry in self.audit_entries.values() 
                  if entry.user_id == user_id]
        
        if start_time:
            entries = [e for e in entries if e.timestamp >= start_time]
        if end_time:
            entries = [e for e in entries if e.timestamp <= end_time]
        
        # Sort by timestamp descending
        entries.sort(key=lambda e: e.timestamp, reverse=True)
        return entries[:limit]
    
    async def get_audit_entries_by_resource(self, resource_type: str, resource_id: str,
                                          start_time: Optional[datetime] = None,
                                          end_time: Optional[datetime] = None,
                                          limit: int = 100) -> List[AuditEntry]:
        entries = [entry for entry in self.audit_entries.values() 
                  if entry.resource_type == resource_type and entry.resource_id == resource_id]
        
        if start_time:
            entries = [e for e in entries if e.timestamp >= start_time]
        if end_time:
            entries = [e for e in entries if e.timestamp <= end_time]
        
        entries.sort(key=lambda e: e.timestamp, reverse=True)
        return entries[:limit]
    
    async def get_audit_entries_by_action(self, action: str,
                                        start_time: Optional[datetime] = None,
                                        end_time: Optional[datetime] = None,
                                        limit: int = 100) -> List[AuditEntry]:
        entries = [entry for entry in self.audit_entries.values() 
                  if entry.action == action]
        
        if start_time:
            entries = [e for e in entries if e.timestamp >= start_time]
        if end_time:
            entries = [e for e in entries if e.timestamp <= end_time]
        
        entries.sort(key=lambda e: e.timestamp, reverse=True)
        return entries[:limit]
    
    async def search_audit_entries(self, filters: Dict[str, Any],
                                 start_time: Optional[datetime] = None,
                                 end_time: Optional[datetime] = None,
                                 limit: int = 100) -> List[AuditEntry]:
        entries = list(self.audit_entries.values())
        
        # Apply filters
        for key, value in filters.items():
            if hasattr(AuditEntry, key):
                entries = [e for e in entries if getattr(e, key) == value]
        
        if start_time:
            entries = [e for e in entries if e.timestamp >= start_time]
        if end_time:
            entries = [e for e in entries if e.timestamp <= end_time]
        
        entries.sort(key=lambda e: e.timestamp, reverse=True)
        return entries[:limit]
    
    async def delete_old_entries(self, before: datetime) -> int:
        to_delete = [entry_id for entry_id, entry in self.audit_entries.items() 
                    if entry.timestamp < before]
        
        for entry_id in to_delete:
            del self.audit_entries[entry_id]
        
        return len(to_delete)
    
    async def get_activity_summary(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        since = datetime.utcnow() - timedelta(days=days)
        entries = await self.get_audit_entries_by_user(user_id, since, limit=1000)
        
        summary = {
            "total_activities": len(entries),
            "activities_by_day": {},
            "activities_by_type": {},
            "most_active_day": None,
            "last_activity": None
        }
        
        if entries:
            summary["last_activity"] = entries[0].timestamp
            
            # Group by day and action
            for entry in entries:
                day = entry.timestamp.date().isoformat()
                action = entry.action
                
                if day not in summary["activities_by_day"]:
                    summary["activities_by_day"][day] = 0
                summary["activities_by_day"][day] += 1
                
                if action not in summary["activities_by_type"]:
                    summary["activities_by_type"][action] = 0
                summary["activities_by_type"][action] += 1
            
            # Find most active day
            if summary["activities_by_day"]:
                summary["most_active_day"] = max(
                    summary["activities_by_day"].items(), 
                    key=lambda x: x[1]
                )[0]
        
        return summary


class InMemoryCacheRepository(CacheRepository):
    """In-memory implementation of cache repository."""
    
    def __init__(self):
        self.cache: Dict[str, Any] = {}
        self.ttl: Dict[str, datetime] = {}
    
    async def get_resolved_settings(self, node_id: str) -> Optional[Dict[str, Any]]:
        key = f"settings:{node_id}"
        if key in self.cache and self._is_valid(key):
            return self.cache[key]
        return None
    
    async def set_resolved_settings(self, node_id: str, settings: Dict[str, Any], 
                                   ttl: int = 300) -> None:
        key = f"settings:{node_id}"
        self.cache[key] = settings
        self.ttl[key] = datetime.utcnow() + timedelta(seconds=ttl)
    
    async def invalidate_resolved_settings(self, node_id: str) -> None:
        key = f"settings:{node_id}"
        self.cache.pop(key, None)
        self.ttl.pop(key, None)
    
    async def invalidate_hierarchy_cache(self, hierarchy_id: str) -> None:
        # Remove all cache entries for this hierarchy
        keys_to_remove = [k for k in self.cache.keys() 
                         if k.startswith(f"settings:") or k.startswith(f"validation:{hierarchy_id}")]
        
        for key in keys_to_remove:
            self.cache.pop(key, None)
            self.ttl.pop(key, None)
    
    async def get_validation_results(self, node_id: str, rule_version: str) -> Optional[List[Dict[str, Any]]]:
        key = f"validation:{node_id}:{rule_version}"
        if key in self.cache and self._is_valid(key):
            return self.cache[key]
        return None
    
    async def set_validation_results(self, node_id: str, rule_version: str,
                                   results: List[Dict[str, Any]], ttl: int = 300) -> None:
        key = f"validation:{node_id}:{rule_version}"
        self.cache[key] = results
        self.ttl[key] = datetime.utcnow() + timedelta(seconds=ttl)
    
    async def clear_all(self) -> None:
        self.cache.clear()
        self.ttl.clear()
    
    def _is_valid(self, key: str) -> bool:
        """Check if cache entry is still valid."""
        if key not in self.ttl:
            return True  # No TTL set, assume valid
        
        return datetime.utcnow() <= self.ttl[key]


class InMemorySecretRepository(SecretRepository):
    """In-memory implementation of secret repository."""
    
    def __init__(self):
        self.secrets: Dict[str, Dict[str, str]] = {}  # reference_key -> {user_id, encrypted_data}
        self.user_secrets: Dict[str, List[str]] = {}  # user_id -> [reference_keys]
    
    async def store_secret(self, key: str, value: str, user_id: str) -> str:
        reference_key = str(uuid.uuid4())
        
        self.secrets[reference_key] = {
            "user_id": user_id,
            "encrypted_data": value,  # Already encrypted by SecurityService
            "created_at": datetime.utcnow().isoformat()
        }
        
        if user_id not in self.user_secrets:
            self.user_secrets[user_id] = []
        self.user_secrets[user_id].append(reference_key)
        
        return reference_key
    
    async def retrieve_secret(self, reference_key: str, user_id: str) -> Optional[str]:
        secret_data = self.secrets.get(reference_key)
        if not secret_data:
            return None
        
        # Check user ownership
        if secret_data["user_id"] != user_id:
            return None
        
        return secret_data["encrypted_data"]
    
    async def delete_secret(self, reference_key: str, user_id: str) -> bool:
        secret_data = self.secrets.get(reference_key)
        if not secret_data or secret_data["user_id"] != user_id:
            return False
        
        del self.secrets[reference_key]
        
        # Remove from user's list
        if user_id in self.user_secrets:
            self.user_secrets[user_id] = [
                k for k in self.user_secrets[user_id] if k != reference_key
            ]
        
        return True
    
    async def list_user_secrets(self, user_id: str) -> List[str]:
        return self.user_secrets.get(user_id, [])
    
    async def rotate_encryption_key(self, old_key: str, new_key: str) -> int:
        # In a real implementation, this would re-encrypt all secrets
        # For in-memory implementation, we just return the count
        return len(self.secrets)