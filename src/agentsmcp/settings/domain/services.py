"""
Domain services for the settings management system.

Domain services encapsulate business logic that doesn't naturally
fit within a single entity or value object, often coordinating
between multiple domain objects.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
import hashlib
import json

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
    SettingValue,
    SettingType,
    AgentStatus,
    ValidationRule,
    ConflictResolution,
    PermissionLevel,
    SettingsValidationError,
    SettingsConflictError,
    PermissionDeniedError,
    EventMetadata,
)


class SettingsResolutionService:
    """
    Service for resolving settings with inheritance and conflict resolution.
    """
    
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl: Dict[str, datetime] = {}
    
    def resolve_effective_settings(self, hierarchy: SettingsHierarchy, 
                                 node_id: str,
                                 use_cache: bool = True) -> Dict[str, SettingValue]:
        """
        Resolve all effective settings for a node with caching.
        """
        cache_key = f"{hierarchy.id}:{node_id}"
        
        # Check cache first
        if use_cache and self._is_cached(cache_key):
            return self._cache[cache_key]
        
        resolved_settings = hierarchy.resolve_all_settings(node_id)
        
        # Cache the result
        if use_cache:
            self._cache[cache_key] = resolved_settings
            self._cache_ttl[cache_key] = datetime.utcnow() + timedelta(minutes=5)
        
        return resolved_settings
    
    def resolve_setting_with_metadata(self, hierarchy: SettingsHierarchy,
                                    node_id: str, key: str) -> Optional[Dict[str, Any]]:
        """
        Resolve a single setting with metadata about its source and inheritance chain.
        """
        path = hierarchy._get_hierarchy_path(node_id)
        inheritance_chain = []
        final_value = None
        
        for current_node_id in reversed(path):
            node = hierarchy.get_node(current_node_id)
            if node and key in node.settings:
                value = node.settings[key]
                inheritance_chain.append({
                    "node_id": current_node_id,
                    "node_name": node.name,
                    "level": node.level,
                    "value": value.value,
                    "last_modified": value.last_modified,
                    "source_level": value.source_level,
                    "overridden": final_value is not None
                })
                if final_value is None:
                    final_value = value
        
        if final_value is None:
            return None
        
        return {
            "key": key,
            "effective_value": final_value,
            "inheritance_chain": inheritance_chain,
            "resolution_path": path,
            "resolved_at": datetime.utcnow()
        }
    
    def detect_setting_conflicts(self, hierarchy: SettingsHierarchy,
                                node_id: str,
                                resolution_strategy: ConflictResolution = ConflictResolution.OVERRIDE
                                ) -> List[Dict[str, Any]]:
        """
        Detect and analyze setting conflicts with resolution suggestions.
        """
        conflicts = hierarchy.detect_conflicts(node_id)
        
        for conflict in conflicts:
            conflict["resolution_strategy"] = resolution_strategy
            conflict["recommended_action"] = self._get_resolution_recommendation(conflict)
            conflict["impact_assessment"] = self._assess_conflict_impact(conflict)
        
        return conflicts
    
    def resolve_conflicts(self, hierarchy: SettingsHierarchy,
                         conflicts: List[Dict[str, Any]],
                         user_decisions: Dict[str, str] = None) -> Dict[str, SettingValue]:
        """
        Resolve conflicts based on strategy and user decisions.
        """
        user_decisions = user_decisions or {}
        resolved = {}
        
        for conflict in conflicts:
            key = conflict["key"]
            values = conflict["values"]
            
            if key in user_decisions:
                # Use user's explicit decision
                chosen_node_id = user_decisions[key]
                for value_info in values:
                    if value_info["node_id"] == chosen_node_id:
                        resolved[key] = SettingValue(
                            value=value_info["value"],
                            type=self._infer_setting_type(value_info["value"]),
                            source_level=value_info["level"]
                        )
                        break
            else:
                # Use automatic resolution strategy
                strategy = conflict.get("resolution_strategy", ConflictResolution.OVERRIDE)
                resolved[key] = self._auto_resolve_conflict(conflict, strategy)
        
        return resolved
    
    def _is_cached(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self._cache:
            return False
        
        ttl = self._cache_ttl.get(cache_key)
        if ttl and datetime.utcnow() > ttl:
            del self._cache[cache_key]
            del self._cache_ttl[cache_key]
            return False
        
        return True
    
    def _get_resolution_recommendation(self, conflict: Dict[str, Any]) -> str:
        """Generate resolution recommendation for a conflict."""
        values = conflict["values"]
        
        # If there's a user-level setting, recommend using it
        user_values = [v for v in values if v["level"] == SettingsLevel.USER]
        if user_values:
            return f"Use user-level setting: {user_values[0]['value']}"
        
        # Otherwise recommend the most specific (lowest level) setting
        level_priority = {
            SettingsLevel.AGENT: 0,
            SettingsLevel.SESSION: 1,
            SettingsLevel.USER: 2,
            SettingsLevel.ORGANIZATION: 3,
            SettingsLevel.GLOBAL: 4
        }
        
        sorted_values = sorted(values, key=lambda v: level_priority[v["level"]])
        recommended = sorted_values[0]
        return f"Use {recommended['level']} setting: {recommended['value']}"
    
    def _assess_conflict_impact(self, conflict: Dict[str, Any]) -> str:
        """Assess the potential impact of a setting conflict."""
        key = conflict["key"]
        values = conflict["values"]
        
        if key.startswith("api_") or key.startswith("secret_"):
            return "HIGH - Affects API access or security"
        elif key.startswith("model_") or key.startswith("provider_"):
            return "MEDIUM - Affects agent behavior"
        elif len(set(v["value"] for v in values)) > 2:
            return "MEDIUM - Multiple conflicting values"
        else:
            return "LOW - Simple value difference"
    
    def _auto_resolve_conflict(self, conflict: Dict[str, Any], 
                              strategy: ConflictResolution) -> SettingValue:
        """Automatically resolve conflict based on strategy."""
        values = conflict["values"]
        key = conflict["key"]
        
        if strategy == ConflictResolution.OVERRIDE:
            # Use most specific (lowest level) setting
            level_priority = {
                SettingsLevel.AGENT: 0,
                SettingsLevel.SESSION: 1,
                SettingsLevel.USER: 2,
                SettingsLevel.ORGANIZATION: 3,
                SettingsLevel.GLOBAL: 4
            }
            
            sorted_values = sorted(values, key=lambda v: level_priority[v["level"]])
            winner = sorted_values[0]
            
            return SettingValue(
                value=winner["value"],
                type=self._infer_setting_type(winner["value"]),
                source_level=winner["level"]
            )
        
        elif strategy == ConflictResolution.MERGE:
            # Attempt to merge if possible (for object/array types)
            if all(isinstance(v["value"], dict) for v in values):
                merged = {}
                for v in reversed(values):  # Lower levels override higher
                    if isinstance(v["value"], dict):
                        merged.update(v["value"])
                
                return SettingValue(
                    value=merged,
                    type=SettingType.OBJECT,
                    source_level=SettingsLevel.USER  # Mark as computed
                )
        
        # Fallback to override strategy
        return self._auto_resolve_conflict(conflict, ConflictResolution.OVERRIDE)
    
    def _infer_setting_type(self, value: Any) -> SettingType:
        """Infer SettingType from a value."""
        if isinstance(value, bool):
            return SettingType.BOOLEAN
        elif isinstance(value, int):
            return SettingType.INTEGER
        elif isinstance(value, float):
            return SettingType.FLOAT
        elif isinstance(value, (list, tuple)):
            return SettingType.ARRAY
        elif isinstance(value, dict):
            return SettingType.OBJECT
        else:
            return SettingType.STRING
    
    def invalidate_cache(self, hierarchy_id: str = None, node_id: str = None) -> None:
        """Invalidate cached settings."""
        if hierarchy_id and node_id:
            cache_key = f"{hierarchy_id}:{node_id}"
            self._cache.pop(cache_key, None)
            self._cache_ttl.pop(cache_key, None)
        elif hierarchy_id:
            # Invalidate all cache entries for hierarchy
            keys_to_remove = [k for k in self._cache.keys() if k.startswith(f"{hierarchy_id}:")]
            for key in keys_to_remove:
                self._cache.pop(key, None)
                self._cache_ttl.pop(key, None)
        else:
            # Invalidate all cache
            self._cache.clear()
            self._cache_ttl.clear()


class SettingsValidationService:
    """
    Service for validating settings against business rules and constraints.
    """
    
    def __init__(self):
        self.built_in_rules = self._create_built_in_rules()
    
    def validate_settings_node(self, node: SettingsNode,
                             custom_rules: List[ValidationRule] = None) -> List[Dict[str, Any]]:
        """
        Validate all settings in a node against validation rules.
        """
        errors = []
        all_rules = self.built_in_rules.copy()
        
        if custom_rules:
            all_rules.extend(custom_rules)
        
        for key, value in node.settings.items():
            relevant_rules = [r for r in all_rules if r.setting_key == key or r.setting_key == "*"]
            
            for rule in relevant_rules:
                try:
                    is_valid, message = rule.validate(value)
                    if not is_valid:
                        errors.append({
                            "node_id": node.id,
                            "setting_key": key,
                            "rule_id": rule.id,
                            "message": message,
                            "severity": rule.severity,
                            "value": value.value
                        })
                except Exception as e:
                    errors.append({
                        "node_id": node.id,
                        "setting_key": key,
                        "rule_id": rule.id,
                        "message": f"Validation rule failed: {str(e)}",
                        "severity": "error",
                        "value": value.value
                    })
        
        return errors
    
    def validate_agent_settings(self, agent_definition: AgentDefinition,
                               settings: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Validate settings against an agent's schema and requirements.
        """
        errors = []
        
        # Use agent's built-in validation
        agent_errors = agent_definition.validate_settings(settings)
        for error in agent_errors:
            errors.append({
                "agent_id": agent_definition.id,
                "message": error,
                "severity": "error",
                "type": "schema_validation"
            })
        
        # Additional business rule validation
        for key, value in settings.items():
            if key in agent_definition.settings_schema:
                schema = agent_definition.settings_schema[key]
                
                # Check for security-sensitive settings
                if schema.get("sensitive", False) and not isinstance(value, str):
                    errors.append({
                        "agent_id": agent_definition.id,
                        "setting_key": key,
                        "message": "Sensitive setting must be encrypted string",
                        "severity": "error",
                        "type": "security_validation"
                    })
                
                # Check for deprecated settings
                if schema.get("deprecated", False):
                    errors.append({
                        "agent_id": agent_definition.id,
                        "setting_key": key,
                        "message": f"Setting '{key}' is deprecated",
                        "severity": "warning",
                        "type": "deprecation_warning"
                    })
        
        return errors
    
    def validate_hierarchy_integrity(self, hierarchy: SettingsHierarchy) -> List[Dict[str, Any]]:
        """
        Validate the integrity and consistency of a settings hierarchy.
        """
        errors = []
        
        # Check for orphaned nodes
        all_node_ids = set(hierarchy.nodes.keys())
        referenced_ids = {hierarchy.root_node_id}
        
        for node in hierarchy.nodes.values():
            if node.parent_id:
                referenced_ids.add(node.parent_id)
        
        orphaned_nodes = []
        for node in hierarchy.nodes.values():
            if node.id != hierarchy.root_node_id and node.id not in referenced_ids:
                orphaned_nodes.append(node)
        
        if orphaned_nodes:
            errors.append({
                "type": "orphaned_nodes",
                "message": f"Found {len(orphaned_nodes)} orphaned nodes",
                "severity": "warning",
                "details": [{"node_id": n.id, "name": n.name} for n in orphaned_nodes]
            })
        
        # Check for circular references
        circular_refs = self._detect_circular_references(hierarchy)
        if circular_refs:
            errors.append({
                "type": "circular_references",
                "message": f"Found {len(circular_refs)} circular references",
                "severity": "error",
                "details": circular_refs
            })
        
        # Check level consistency
        level_inconsistencies = self._check_level_consistency(hierarchy)
        if level_inconsistencies:
            errors.append({
                "type": "level_inconsistencies",
                "message": f"Found {len(level_inconsistencies)} level inconsistencies",
                "severity": "warning",
                "details": level_inconsistencies
            })
        
        return errors
    
    def _create_built_in_rules(self) -> List[ValidationRule]:
        """Create built-in validation rules."""
        return [
            ValidationRule(
                id="api_key_format",
                setting_key="api_key",
                rule_type="regex",
                parameters={"pattern": r"^[a-zA-Z0-9_-]+$"},
                error_message="API key must contain only alphanumeric characters, underscores, and hyphens"
            ),
            ValidationRule(
                id="required_model_name",
                setting_key="model_name",
                rule_type="required",
                error_message="Model name is required"
            ),
            ValidationRule(
                id="timeout_range",
                setting_key="timeout",
                rule_type="range",
                parameters={"min": 1, "max": 300},
                error_message="Timeout must be between 1 and 300 seconds"
            ),
            ValidationRule(
                id="max_tokens_range", 
                setting_key="max_tokens",
                rule_type="range",
                parameters={"min": 1, "max": 8192},
                error_message="Max tokens must be between 1 and 8192"
            ),
        ]
    
    def _detect_circular_references(self, hierarchy: SettingsHierarchy) -> List[Dict[str, Any]]:
        """Detect circular references in the hierarchy."""
        visited = set()
        path = []
        circular_refs = []
        
        def dfs(node_id: str):
            if node_id in path:
                # Found circular reference
                cycle_start = path.index(node_id)
                cycle = path[cycle_start:] + [node_id]
                circular_refs.append({
                    "cycle": cycle,
                    "nodes": [hierarchy.get_node(nid).name for nid in cycle if hierarchy.get_node(nid)]
                })
                return
            
            if node_id in visited:
                return
            
            visited.add(node_id)
            path.append(node_id)
            
            node = hierarchy.get_node(node_id)
            if node and node.parent_id:
                dfs(node.parent_id)
            
            path.pop()
        
        for node_id in hierarchy.nodes:
            if node_id not in visited:
                dfs(node_id)
        
        return circular_refs
    
    def _check_level_consistency(self, hierarchy: SettingsHierarchy) -> List[Dict[str, Any]]:
        """Check that node levels are consistent with hierarchy structure."""
        inconsistencies = []
        
        level_order = [
            SettingsLevel.GLOBAL,
            SettingsLevel.ORGANIZATION,
            SettingsLevel.USER,
            SettingsLevel.SESSION,
            SettingsLevel.AGENT
        ]
        
        for node in hierarchy.nodes.values():
            if node.parent_id:
                parent = hierarchy.get_node(node.parent_id)
                if parent:
                    parent_level_index = level_order.index(parent.level)
                    node_level_index = level_order.index(node.level)
                    
                    if node_level_index <= parent_level_index:
                        inconsistencies.append({
                            "node_id": node.id,
                            "node_level": node.level,
                            "parent_id": parent.id,
                            "parent_level": parent.level,
                            "message": f"Node level {node.level} should be more specific than parent level {parent.level}"
                        })
        
        return inconsistencies


class PermissionService:
    """
    Service for managing permissions and access control.
    """
    
    def check_permission(self, user: UserProfile, resource_type: str,
                        resource_id: str, required_level: PermissionLevel) -> bool:
        """
        Check if user has required permission for a resource.
        """
        return user.has_permission(resource_type, resource_id, required_level)
    
    def require_permission(self, user: UserProfile, resource_type: str,
                         resource_id: str, required_level: PermissionLevel) -> None:
        """
        Require permission or raise PermissionDeniedError.
        """
        if not self.check_permission(user, resource_type, resource_id, required_level):
            raise PermissionDeniedError(user.id, f"{resource_type}:{resource_id}", required_level)
    
    def get_accessible_resources(self, user: UserProfile, resource_type: str,
                               required_level: PermissionLevel = PermissionLevel.READ
                               ) -> List[str]:
        """
        Get list of resource IDs user can access at the required level.
        """
        accessible = []
        
        for grant in user.permissions:
            if (grant.resource_type == resource_type and
                self._permission_level_sufficient(grant.permission_level, required_level) and
                (grant.expires_at is None or grant.expires_at > datetime.utcnow())):
                accessible.append(grant.resource_id)
        
        return accessible
    
    def _permission_level_sufficient(self, granted: PermissionLevel, 
                                   required: PermissionLevel) -> bool:
        """Check if granted permission level is sufficient for required level."""
        level_hierarchy = {
            PermissionLevel.READ: 1,
            PermissionLevel.WRITE: 2,
            PermissionLevel.ADMIN: 3,
            PermissionLevel.OWNER: 4
        }
        
        return level_hierarchy[granted] >= level_hierarchy[required]


class AuditService:
    """
    Service for creating and managing audit logs.
    """
    
    def create_audit_entry(self, user_id: str, action: str, resource_type: str,
                         resource_id: str, details: Dict[str, Any] = None,
                         old_value: Dict[str, Any] = None,
                         new_value: Dict[str, Any] = None,
                         session_id: str = None,
                         ip_address: str = None) -> AuditEntry:
        """
        Create a new audit entry.
        """
        return AuditEntry(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
            old_value=old_value,
            new_value=new_value,
            session_id=session_id,
            ip_address=ip_address
        )
    
    def create_settings_change_audit(self, user_id: str, node_id: str,
                                   setting_key: str, old_value: Any,
                                   new_value: Any, session_id: str = None) -> AuditEntry:
        """
        Create audit entry for settings changes.
        """
        return self.create_audit_entry(
            user_id=user_id,
            action="update",
            resource_type="settings",
            resource_id=f"{node_id}:{setting_key}",
            details={"setting_key": setting_key, "node_id": node_id},
            old_value={"value": old_value},
            new_value={"value": new_value},
            session_id=session_id
        )
    
    def create_agent_lifecycle_audit(self, user_id: str, agent_id: str,
                                   action: str, details: Dict[str, Any] = None) -> AuditEntry:
        """
        Create audit entry for agent lifecycle events.
        """
        return self.create_audit_entry(
            user_id=user_id,
            action=action,
            resource_type="agent",
            resource_id=agent_id,
            details=details or {}
        )