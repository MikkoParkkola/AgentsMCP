"""
Settings management service.

Provides the main application logic for managing hierarchical settings
with inheritance, validation, and real-time updates.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
import uuid

from ..domain.entities import (
    SettingsNode,
    SettingsHierarchy,
    UserProfile,
    AuditEntry,
)
from ..domain.value_objects import (
    SettingsLevel,
    SettingValue,
    SettingType,
    ValidationRule,
    ConflictResolution,
    PermissionLevel,
    SettingsValidationError,
    SettingsConflictError,
    PermissionDeniedError,
)
from ..domain.repositories import (
    SettingsRepository,
    UserRepository,
    AuditRepository,
    CacheRepository,
    SecretRepository,
)
from ..domain.services import (
    SettingsResolutionService,
    SettingsValidationService,
    PermissionService,
    AuditService,
)
from ..events.settings_events import (
    SettingChangedEvent,
    SettingsNodeCreatedEvent,
    SettingsValidationFailedEvent,
    SettingsConflictDetectedEvent,
)
from ..events.event_publisher import EventPublisher


class SettingsService:
    """
    Main application service for settings management.
    
    Orchestrates domain services and repositories to provide
    high-level settings management operations.
    """
    
    def __init__(self,
                 settings_repository: SettingsRepository,
                 user_repository: UserRepository,
                 audit_repository: AuditRepository,
                 cache_repository: CacheRepository,
                 secret_repository: SecretRepository,
                 event_publisher: EventPublisher):
        self.settings_repo = settings_repository
        self.user_repo = user_repository
        self.audit_repo = audit_repository
        self.cache_repo = cache_repository
        self.secret_repo = secret_repository
        self.event_publisher = event_publisher
        
        # Domain services
        self.resolution_service = SettingsResolutionService()
        self.validation_service = SettingsValidationService()
        self.permission_service = PermissionService()
        self.audit_service = AuditService()
    
    async def create_hierarchy(self, user_id: str, name: str, 
                             description: str = "") -> SettingsHierarchy:
        """
        Create a new settings hierarchy.
        """
        user = await self.user_repo.get_user(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")
        
        # Create hierarchy with root global node
        hierarchy = SettingsHierarchy(name=name)
        
        root_node = SettingsNode(
            level=SettingsLevel.GLOBAL,
            name=f"{name} Global Settings"
        )
        
        hierarchy.add_node(root_node)
        hierarchy.root_node_id = root_node.id
        
        # Save to repository
        await self.settings_repo.save_hierarchy(hierarchy)
        await self.settings_repo.save_node(root_node)
        
        # Create audit entry
        audit_entry = self.audit_service.create_audit_entry(
            user_id=user_id,
            action="create",
            resource_type="settings_hierarchy",
            resource_id=hierarchy.id,
            details={"name": name, "description": description}
        )
        await self.audit_repo.save_audit_entry(audit_entry)
        
        # Publish event
        event = SettingsNodeCreatedEvent(
            hierarchy_id=hierarchy.id,
            node_id=root_node.id,
            user_id=user_id,
            level=SettingsLevel.GLOBAL
        )
        await self.event_publisher.publish(event)
        
        return hierarchy
    
    async def create_settings_node(self, user_id: str, hierarchy_id: str,
                                 level: SettingsLevel, name: str,
                                 parent_id: str = None) -> SettingsNode:
        """
        Create a new settings node in a hierarchy.
        """
        user = await self.user_repo.get_user(user_id)
        hierarchy = await self.settings_repo.get_hierarchy(hierarchy_id)
        
        if not user or not hierarchy:
            raise ValueError("User or hierarchy not found")
        
        # Check permissions
        self.permission_service.require_permission(
            user, "settings_hierarchy", hierarchy_id, PermissionLevel.WRITE
        )
        
        # Validate parent exists if specified
        if parent_id:
            parent = await self.settings_repo.get_node(parent_id)
            if not parent:
                raise ValueError(f"Parent node {parent_id} not found")
        
        # Create node
        node = SettingsNode(
            level=level,
            name=name,
            parent_id=parent_id
        )
        
        # Add to hierarchy and save
        hierarchy.add_node(node)
        await self.settings_repo.save_hierarchy(hierarchy)
        await self.settings_repo.save_node(node)
        
        # Create audit entry
        audit_entry = self.audit_service.create_audit_entry(
            user_id=user_id,
            action="create",
            resource_type="settings_node",
            resource_id=node.id,
            details={"hierarchy_id": hierarchy_id, "level": level, "parent_id": parent_id}
        )
        await self.audit_repo.save_audit_entry(audit_entry)
        
        # Publish event
        event = SettingsNodeCreatedEvent(
            hierarchy_id=hierarchy_id,
            node_id=node.id,
            user_id=user_id,
            level=level,
            parent_id=parent_id
        )
        await self.event_publisher.publish(event)
        
        return node
    
    async def set_setting(self, user_id: str, node_id: str, key: str,
                         value: Any, setting_type: SettingType = None,
                         validate: bool = True) -> None:
        """
        Set a setting value in a node with validation and audit.
        """
        user = await self.user_repo.get_user(user_id)
        node = await self.settings_repo.get_node(node_id)
        
        if not user or not node:
            raise ValueError("User or node not found")
        
        # Check permissions
        self.permission_service.require_permission(
            user, "settings_node", node_id, PermissionLevel.WRITE
        )
        
        # Get old value for audit
        old_setting = node.get_setting(key)
        old_value = old_setting.value if old_setting else None
        
        # Handle secrets
        if setting_type == SettingType.SECRET:
            # Encrypt and store secret
            secret_ref = await self.secret_repo.store_secret(str(value), user_id)
            value = secret_ref
            encrypted = True
        else:
            encrypted = False
        
        # Infer type if not provided
        if not setting_type:
            setting_type = self._infer_setting_type(value)
        
        # Create setting value
        setting_value = SettingValue(
            value=value,
            type=setting_type,
            encrypted=encrypted,
            source_level=node.level
        )
        
        # Validate if requested
        if validate:
            hierarchy = await self._get_hierarchy_for_node(node_id)
            if hierarchy:
                errors = self.validation_service.validate_settings_node(node, hierarchy.validation_rules)
                if errors:
                    error_event = SettingsValidationFailedEvent(
                        node_id=node_id,
                        user_id=user_id,
                        setting_key=key,
                        errors=errors
                    )
                    await self.event_publisher.publish(error_event)
                    raise SettingsValidationError(key, f"Validation failed: {errors}")
        
        # Set the setting
        node.set_setting(key, setting_value)
        await self.settings_repo.save_node(node)
        
        # Invalidate cache
        await self.cache_repo.invalidate_resolved_settings(node_id)
        
        # Create audit entry
        audit_entry = self.audit_service.create_settings_change_audit(
            user_id=user_id,
            node_id=node_id,
            setting_key=key,
            old_value=old_value,
            new_value=value
        )
        await self.audit_repo.save_audit_entry(audit_entry)
        
        # Publish event
        event = SettingChangedEvent(
            node_id=node_id,
            hierarchy_id=(await self._get_hierarchy_for_node(node_id)).id,
            user_id=user_id,
            setting_key=key,
            old_value=old_value,
            new_value=value
        )
        await self.event_publisher.publish(event)
    
    async def get_setting(self, user_id: str, node_id: str, key: str,
                         include_inheritance: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get a setting value with optional inheritance information.
        """
        user = await self.user_repo.get_user(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")
        
        # Check permissions
        self.permission_service.require_permission(
            user, "settings_node", node_id, PermissionLevel.READ
        )
        
        if include_inheritance:
            hierarchy = await self._get_hierarchy_for_node(node_id)
            if not hierarchy:
                return None
            
            return self.resolution_service.resolve_setting_with_metadata(
                hierarchy, node_id, key
            )
        else:
            node = await self.settings_repo.get_node(node_id)
            if not node:
                return None
            
            setting = node.get_setting(key)
            if not setting:
                return None
            
            # Decrypt if secret
            value = setting.value
            if setting.encrypted:
                value = await self.secret_repo.retrieve_secret(setting.value, user_id)
            
            return {
                "key": key,
                "value": value,
                "type": setting.type,
                "encrypted": setting.encrypted,
                "last_modified": setting.last_modified
            }
    
    async def get_effective_settings(self, user_id: str, node_id: str,
                                   use_cache: bool = True) -> Dict[str, Any]:
        """
        Get all effective settings for a node with inheritance resolution.
        """
        user = await self.user_repo.get_user(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")
        
        # Check permissions
        self.permission_service.require_permission(
            user, "settings_node", node_id, PermissionLevel.READ
        )
        
        # Try cache first
        if use_cache:
            cached = await self.cache_repo.get_resolved_settings(node_id)
            if cached:
                return cached
        
        hierarchy = await self._get_hierarchy_for_node(node_id)
        if not hierarchy:
            return {}
        
        resolved_settings = self.resolution_service.resolve_effective_settings(
            hierarchy, node_id, use_cache=False
        )
        
        # Decrypt secrets
        result = {}
        for key, setting_value in resolved_settings.items():
            value = setting_value.value
            if setting_value.encrypted:
                decrypted = await self.secret_repo.retrieve_secret(value, user_id)
                value = decrypted if decrypted else value
            
            result[key] = {
                "value": value,
                "type": setting_value.type,
                "source_level": setting_value.source_level,
                "last_modified": setting_value.last_modified,
                "encrypted": setting_value.encrypted
            }
        
        # Cache result
        if use_cache:
            await self.cache_repo.set_resolved_settings(node_id, result, ttl=300)
        
        return result
    
    async def detect_conflicts(self, user_id: str, node_id: str) -> List[Dict[str, Any]]:
        """
        Detect setting conflicts in the hierarchy for a node.
        """
        user = await self.user_repo.get_user(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")
        
        # Check permissions
        self.permission_service.require_permission(
            user, "settings_node", node_id, PermissionLevel.READ
        )
        
        hierarchy = await self._get_hierarchy_for_node(node_id)
        if not hierarchy:
            return []
        
        conflicts = self.resolution_service.detect_setting_conflicts(
            hierarchy, node_id
        )
        
        # Publish event if conflicts found
        if conflicts:
            event = SettingsConflictDetectedEvent(
                node_id=node_id,
                hierarchy_id=hierarchy.id,
                user_id=user_id,
                conflicts=conflicts
            )
            await self.event_publisher.publish(event)
        
        return conflicts
    
    async def resolve_conflicts(self, user_id: str, node_id: str,
                              resolutions: Dict[str, str]) -> Dict[str, Any]:
        """
        Resolve setting conflicts with user decisions.
        """
        user = await self.user_repo.get_user(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")
        
        # Check permissions
        self.permission_service.require_permission(
            user, "settings_node", node_id, PermissionLevel.WRITE
        )
        
        hierarchy = await self._get_hierarchy_for_node(node_id)
        if not hierarchy:
            raise ValueError("Hierarchy not found")
        
        conflicts = self.resolution_service.detect_setting_conflicts(hierarchy, node_id)
        
        resolved_settings = self.resolution_service.resolve_conflicts(
            hierarchy, conflicts, resolutions
        )
        
        # Apply resolved settings to appropriate nodes
        for key, setting_value in resolved_settings.items():
            # Find the target node based on resolution decision
            target_node_id = resolutions.get(key, node_id)
            await self.set_setting(user_id, target_node_id, key, setting_value.value,
                                 setting_value.type, validate=False)
        
        return {"resolved_count": len(resolved_settings)}
    
    async def validate_settings(self, user_id: str, node_id: str) -> List[Dict[str, Any]]:
        """
        Validate all settings for a node.
        """
        user = await self.user_repo.get_user(user_id)
        node = await self.settings_repo.get_node(node_id)
        
        if not user or not node:
            raise ValueError("User or node not found")
        
        # Check permissions
        self.permission_service.require_permission(
            user, "settings_node", node_id, PermissionLevel.READ
        )
        
        hierarchy = await self._get_hierarchy_for_node(node_id)
        if not hierarchy:
            return []
        
        return self.validation_service.validate_settings_node(node, hierarchy.validation_rules)
    
    async def bulk_update_settings(self, user_id: str, node_id: str,
                                 settings: Dict[str, Any],
                                 validate: bool = True) -> Dict[str, Any]:
        """
        Update multiple settings in a single operation.
        """
        user = await self.user_repo.get_user(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")
        
        # Check permissions
        self.permission_service.require_permission(
            user, "settings_node", node_id, PermissionLevel.WRITE
        )
        
        results = {
            "updated": [],
            "errors": [],
            "total": len(settings)
        }
        
        # Process each setting
        for key, value in settings.items():
            try:
                await self.set_setting(
                    user_id=user_id,
                    node_id=node_id,
                    key=key,
                    value=value,
                    validate=validate
                )
                results["updated"].append(key)
            except Exception as e:
                results["errors"].append({
                    "key": key,
                    "error": str(e)
                })
        
        return results
    
    async def export_settings(self, user_id: str, hierarchy_id: str,
                            include_secrets: bool = False) -> Dict[str, Any]:
        """
        Export settings hierarchy to a portable format.
        """
        user = await self.user_repo.get_user(user_id)
        hierarchy = await self.settings_repo.get_hierarchy(hierarchy_id)
        
        if not user or not hierarchy:
            raise ValueError("User or hierarchy not found")
        
        # Check permissions
        self.permission_service.require_permission(
            user, "settings_hierarchy", hierarchy_id, PermissionLevel.READ
        )
        
        export_data = {
            "hierarchy_id": hierarchy.id,
            "name": hierarchy.name,
            "created_at": hierarchy.created_at.isoformat(),
            "nodes": []
        }
        
        for node in hierarchy.nodes.values():
            node_data = {
                "id": node.id,
                "level": node.level,
                "name": node.name,
                "parent_id": node.parent_id,
                "settings": {}
            }
            
            for key, setting in node.settings.items():
                value = setting.value
                
                # Handle secrets
                if setting.encrypted:
                    if include_secrets:
                        decrypted = await self.secret_repo.retrieve_secret(value, user_id)
                        value = decrypted if decrypted else "<ENCRYPTED>"
                    else:
                        value = "<ENCRYPTED>"
                
                node_data["settings"][key] = {
                    "value": value,
                    "type": setting.type,
                    "encrypted": setting.encrypted,
                    "last_modified": setting.last_modified.isoformat()
                }
            
            export_data["nodes"].append(node_data)
        
        return export_data
    
    async def import_settings(self, user_id: str, import_data: Dict[str, Any],
                            merge_strategy: str = "override") -> Dict[str, Any]:
        """
        Import settings from exported data.
        """
        user = await self.user_repo.get_user(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")
        
        results = {
            "hierarchy_created": False,
            "nodes_created": 0,
            "settings_imported": 0,
            "errors": []
        }
        
        try:
            # Create or update hierarchy
            hierarchy_name = import_data.get("name", f"Imported {datetime.utcnow()}")
            existing_hierarchy = await self.settings_repo.get_hierarchy_by_name(hierarchy_name)
            
            if not existing_hierarchy:
                hierarchy = await self.create_hierarchy(user_id, hierarchy_name)
                results["hierarchy_created"] = True
            else:
                hierarchy = existing_hierarchy
            
            # Import nodes
            for node_data in import_data.get("nodes", []):
                try:
                    # Create node if it doesn't exist
                    existing_node = await self.settings_repo.get_node(node_data["id"])
                    
                    if not existing_node:
                        node = await self.create_settings_node(
                            user_id=user_id,
                            hierarchy_id=hierarchy.id,
                            level=SettingsLevel(node_data["level"]),
                            name=node_data["name"],
                            parent_id=node_data.get("parent_id")
                        )
                        results["nodes_created"] += 1
                    else:
                        node = existing_node
                    
                    # Import settings
                    for key, setting_data in node_data.get("settings", {}).items():
                        if merge_strategy == "skip" and node.get_setting(key):
                            continue
                        
                        await self.set_setting(
                            user_id=user_id,
                            node_id=node.id,
                            key=key,
                            value=setting_data["value"],
                            setting_type=SettingType(setting_data["type"]),
                            validate=False  # Skip validation during import
                        )
                        results["settings_imported"] += 1
                
                except Exception as e:
                    results["errors"].append({
                        "node_id": node_data.get("id"),
                        "error": str(e)
                    })
        
        except Exception as e:
            results["errors"].append({"general": str(e)})
        
        return results
    
    def _infer_setting_type(self, value: Any) -> SettingType:
        """Infer setting type from value."""
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
    
    async def _get_hierarchy_for_node(self, node_id: str) -> Optional[SettingsHierarchy]:
        """Get the hierarchy containing a specific node."""
        # This is a simplified implementation - in practice, you might want
        # to add a node_id -> hierarchy_id mapping in the repository
        node = await self.settings_repo.get_node(node_id)
        if not node:
            return None
        
        # Search through all hierarchies (could be optimized)
        # In a real implementation, you'd have an index
        return None  # Placeholder - implement based on your specific needs