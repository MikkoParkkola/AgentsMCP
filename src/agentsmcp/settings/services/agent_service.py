"""
Agent management service.

Provides application logic for managing agent definitions, instances,
and their lifecycle with settings integration.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import uuid

from ..domain.entities import (
    AgentDefinition,
    AgentInstance,
    UserProfile,
    AuditEntry,
)
from ..domain.value_objects import (
    AgentStatus,
    AgentCapability,
    InstructionTemplate,
    PermissionLevel,
    SettingValue,
    SettingType,
    AgentDefinitionError,
)
from ..domain.repositories import (
    AgentRepository,
    UserRepository,
    AuditRepository,
    SettingsRepository,
)
from ..domain.services import (
    PermissionService,
    AuditService,
)
from ..events.agent_events import (
    AgentDefinitionCreatedEvent,
    AgentDefinitionUpdatedEvent,
    AgentInstanceStartedEvent,
    AgentInstanceStoppedEvent,
    AgentPerformanceUpdatedEvent,
)
from ..events.event_publisher import EventPublisher
from .settings_service import SettingsService


class AgentService:
    """
    Main application service for agent management.
    
    Handles agent definitions, instances, and lifecycle management
    with integration to the settings system.
    """
    
    def __init__(self,
                 agent_repository: AgentRepository,
                 user_repository: UserRepository,
                 audit_repository: AuditRepository,
                 settings_repository: SettingsRepository,
                 settings_service: SettingsService,
                 event_publisher: EventPublisher):
        self.agent_repo = agent_repository
        self.user_repo = user_repository
        self.audit_repo = audit_repository
        self.settings_repo = settings_repository
        self.settings_service = settings_service
        self.event_publisher = event_publisher
        
        # Domain services
        self.permission_service = PermissionService()
        self.audit_service = AuditService()
    
    async def create_agent_definition(self, user_id: str, name: str,
                                    description: str, base_model: str,
                                    instruction_template: Optional[InstructionTemplate] = None,
                                    capabilities: List[AgentCapability] = None,
                                    settings_schema: Dict[str, Any] = None,
                                    default_settings: Dict[str, Any] = None,
                                    tags: List[str] = None,
                                    category: str = "general") -> AgentDefinition:
        """
        Create a new agent definition.
        """
        user = await self.user_repo.get_user(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")
        
        # Convert default settings to SettingValue objects
        converted_defaults = {}
        if default_settings:
            for key, value in default_settings.items():
                setting_type = self._infer_setting_type(value)
                converted_defaults[key] = SettingValue(
                    value=value,
                    type=setting_type
                )
        
        # Create agent definition
        agent = AgentDefinition(
            name=name,
            description=description,
            base_model=base_model,
            owner_id=user_id,
            organization_id=user.organization_id,
            instruction_template=instruction_template,
            capabilities=capabilities or [],
            settings_schema=settings_schema or {},
            default_settings=converted_defaults,
            tags=tags or [],
            category=category
        )
        
        # Save agent definition
        await self.agent_repo.save_agent_definition(agent)
        
        # Create audit entry
        audit_entry = self.audit_service.create_agent_lifecycle_audit(
            user_id=user_id,
            agent_id=agent.id,
            action="create",
            details={
                "name": name,
                "base_model": base_model,
                "category": category,
                "capabilities_count": len(capabilities or [])
            }
        )
        await self.audit_repo.save_audit_entry(audit_entry)
        
        # Publish event
        event = AgentDefinitionCreatedEvent(
            agent_id=agent.id,
            user_id=user_id,
            name=name,
            base_model=base_model,
            category=category
        )
        await self.event_publisher.publish(event)
        
        return agent
    
    async def update_agent_definition(self, user_id: str, agent_id: str,
                                    updates: Dict[str, Any]) -> AgentDefinition:
        """
        Update an existing agent definition.
        """
        user = await self.user_repo.get_user(user_id)
        agent = await self.agent_repo.get_agent_definition(agent_id)
        
        if not user or not agent:
            raise ValueError("User or agent not found")
        
        # Check permissions
        if agent.owner_id != user_id:
            self.permission_service.require_permission(
                user, "agent", agent_id, PermissionLevel.WRITE
            )
        
        # Store old values for audit
        old_values = {
            "name": agent.name,
            "description": agent.description,
            "status": agent.status,
            "version": agent.version
        }
        
        # Apply updates
        if "name" in updates:
            agent.name = updates["name"]
        if "description" in updates:
            agent.description = updates["description"]
        if "instruction_template" in updates:
            agent.instruction_template = updates["instruction_template"]
        if "settings_schema" in updates:
            agent.settings_schema = updates["settings_schema"]
        if "default_settings" in updates:
            # Convert to SettingValue objects
            converted_defaults = {}
            for key, value in updates["default_settings"].items():
                setting_type = self._infer_setting_type(value)
                converted_defaults[key] = SettingValue(
                    value=value,
                    type=setting_type
                )
            agent.default_settings = converted_defaults
        if "tags" in updates:
            agent.tags = updates["tags"]
        if "category" in updates:
            agent.category = updates["category"]
        
        agent.updated_at = datetime.utcnow()
        
        # Save updated agent
        await self.agent_repo.save_agent_definition(agent)
        
        # Create audit entry
        audit_entry = self.audit_service.create_agent_lifecycle_audit(
            user_id=user_id,
            agent_id=agent_id,
            action="update",
            details={"updated_fields": list(updates.keys())}
        )
        audit_entry.old_value = old_values
        audit_entry.new_value = updates
        await self.audit_repo.save_audit_entry(audit_entry)
        
        # Publish event
        event = AgentDefinitionUpdatedEvent(
            agent_id=agent_id,
            user_id=user_id,
            updated_fields=list(updates.keys()),
            old_values=old_values,
            new_values=updates
        )
        await self.event_publisher.publish(event)
        
        return agent
    
    async def publish_agent(self, user_id: str, agent_id: str) -> AgentDefinition:
        """
        Publish an agent definition (make it active).
        """
        user = await self.user_repo.get_user(user_id)
        agent = await self.agent_repo.get_agent_definition(agent_id)
        
        if not user or not agent:
            raise ValueError("User or agent not found")
        
        # Check permissions
        if agent.owner_id != user_id:
            self.permission_service.require_permission(
                user, "agent", agent_id, PermissionLevel.ADMIN
            )
        
        # Validate agent is ready for publishing
        validation_errors = await self._validate_agent_for_publishing(agent)
        if validation_errors:
            raise AgentDefinitionError(f"Agent not ready for publishing: {validation_errors}")
        
        # Update status
        agent.update_status(AgentStatus.ACTIVE)
        await self.agent_repo.save_agent_definition(agent)
        
        # Create audit entry
        audit_entry = self.audit_service.create_agent_lifecycle_audit(
            user_id=user_id,
            agent_id=agent_id,
            action="publish",
            details={"status": AgentStatus.ACTIVE}
        )
        await self.audit_repo.save_audit_entry(audit_entry)
        
        return agent
    
    async def create_agent_instance(self, user_id: str, agent_id: str,
                                  instance_name: str = "",
                                  custom_settings: Dict[str, Any] = None,
                                  session_id: str = None) -> AgentInstance:
        """
        Create a new agent instance from a definition.
        """
        user = await self.user_repo.get_user(user_id)
        agent_def = await self.agent_repo.get_agent_definition(agent_id)
        
        if not user or not agent_def:
            raise ValueError("User or agent definition not found")
        
        # Check permissions
        self.permission_service.require_permission(
            user, "agent", agent_id, PermissionLevel.READ
        )
        
        if agent_def.status != AgentStatus.ACTIVE:
            raise ValueError("Cannot create instance from inactive agent definition")
        
        # Create instance settings by merging defaults with custom settings
        instance_settings = {}
        custom_settings = custom_settings or {}
        
        # Start with defaults
        for key, default_value in agent_def.default_settings.items():
            instance_settings[key] = default_value
        
        # Apply custom settings
        for key, value in custom_settings.items():
            # Validate against schema
            if key in agent_def.settings_schema:
                schema = agent_def.settings_schema[key]
                # Add validation logic here
                pass
            
            setting_type = self._infer_setting_type(value)
            instance_settings[key] = SettingValue(
                value=value,
                type=setting_type
            )
        
        # Create agent instance
        instance = AgentInstance(
            agent_definition_id=agent_id,
            user_id=user_id,
            session_id=session_id,
            name=instance_name or f"{agent_def.name} Instance",
            settings=instance_settings,
            effective_settings=instance_settings.copy()
        )
        
        # Save instance
        await self.agent_repo.save_agent_instance(instance)
        
        # Create audit entry
        audit_entry = self.audit_service.create_agent_lifecycle_audit(
            user_id=user_id,
            agent_id=agent_id,
            action="create_instance",
            details={
                "instance_id": instance.id,
                "instance_name": instance_name,
                "session_id": session_id
            }
        )
        await self.audit_repo.save_audit_entry(audit_entry)
        
        return instance
    
    async def start_agent_instance(self, user_id: str, instance_id: str) -> AgentInstance:
        """
        Start an agent instance.
        """
        user = await self.user_repo.get_user(user_id)
        instance = await self.agent_repo.get_agent_instance(instance_id)
        
        if not user or not instance:
            raise ValueError("User or instance not found")
        
        # Check permissions
        if instance.user_id != user_id:
            self.permission_service.require_permission(
                user, "agent_instance", instance_id, PermissionLevel.WRITE
            )
        
        # Start the instance
        instance.start()
        await self.agent_repo.save_agent_instance(instance)
        
        # Create audit entry
        audit_entry = self.audit_service.create_agent_lifecycle_audit(
            user_id=user_id,
            agent_id=instance.agent_definition_id,
            action="start_instance",
            details={"instance_id": instance_id}
        )
        await self.audit_repo.save_audit_entry(audit_entry)
        
        # Publish event
        event = AgentInstanceStartedEvent(
            instance_id=instance_id,
            agent_id=instance.agent_definition_id,
            user_id=user_id
        )
        await self.event_publisher.publish(event)
        
        return instance
    
    async def stop_agent_instance(self, user_id: str, instance_id: str) -> AgentInstance:
        """
        Stop an agent instance.
        """
        user = await self.user_repo.get_user(user_id)
        instance = await self.agent_repo.get_agent_instance(instance_id)
        
        if not user or not instance:
            raise ValueError("User or instance not found")
        
        # Check permissions
        if instance.user_id != user_id:
            self.permission_service.require_permission(
                user, "agent_instance", instance_id, PermissionLevel.WRITE
            )
        
        # Stop the instance
        instance.stop()
        await self.agent_repo.save_agent_instance(instance)
        
        # Create audit entry
        audit_entry = self.audit_service.create_agent_lifecycle_audit(
            user_id=user_id,
            agent_id=instance.agent_definition_id,
            action="stop_instance",
            details={"instance_id": instance_id}
        )
        await self.audit_repo.save_audit_entry(audit_entry)
        
        # Publish event
        event = AgentInstanceStoppedEvent(
            instance_id=instance_id,
            agent_id=instance.agent_definition_id,
            user_id=user_id
        )
        await self.event_publisher.publish(event)
        
        return instance
    
    async def update_instance_performance(self, instance_id: str,
                                        response_time: float,
                                        success: bool) -> None:
        """
        Update performance metrics for an agent instance.
        """
        instance = await self.agent_repo.get_agent_instance(instance_id)
        if not instance:
            return
        
        old_success_rate = instance.success_rate
        old_avg_response_time = instance.average_response_time
        
        # Record the request
        instance.record_request(response_time, success)
        await self.agent_repo.save_agent_instance(instance)
        
        # Update agent definition statistics
        agent_def = await self.agent_repo.get_agent_definition(instance.agent_definition_id)
        if agent_def:
            agent_def.usage_count += 1
            
            # Update rolling averages
            total_requests = agent_def.usage_count
            if success:
                agent_def.success_rate = ((agent_def.success_rate * (total_requests - 1)) + 100) / total_requests
            else:
                agent_def.success_rate = (agent_def.success_rate * (total_requests - 1)) / total_requests
            
            agent_def.average_response_time = ((agent_def.average_response_time * (total_requests - 1)) + response_time) / total_requests
            
            await self.agent_repo.save_agent_definition(agent_def)
        
        # Publish performance event
        event = AgentPerformanceUpdatedEvent(
            instance_id=instance_id,
            agent_id=instance.agent_definition_id,
            response_time=response_time,
            success=success,
            new_success_rate=instance.success_rate,
            old_success_rate=old_success_rate
        )
        await self.event_publisher.publish(event)
    
    async def get_agent_marketplace(self, user_id: str,
                                  category: str = None,
                                  tags: List[str] = None,
                                  search_query: str = "",
                                  limit: int = 50) -> List[AgentDefinition]:
        """
        Get agents available in the marketplace.
        """
        user = await self.user_repo.get_user(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")
        
        # Build search filters
        filters = {
            "status": AgentStatus.ACTIVE
        }
        
        if category:
            filters["category"] = category
        
        if tags:
            filters["tags"] = tags
        
        # Get agents
        agents = await self.agent_repo.search_agent_definitions(
            query=search_query,
            filters=filters,
            limit=limit
        )
        
        # Filter by permissions (user can see public agents and own agents)
        accessible_agents = []
        for agent in agents:
            if (agent.owner_id == user_id or
                agent.organization_id == user.organization_id or
                self.permission_service.check_permission(
                    user, "agent", agent.id, PermissionLevel.READ
                )):
                accessible_agents.append(agent)
        
        return accessible_agents
    
    async def get_agent_performance_metrics(self, user_id: str, agent_id: str,
                                          days: int = 30) -> Dict[str, Any]:
        """
        Get performance metrics for an agent definition.
        """
        user = await self.user_repo.get_user(user_id)
        agent = await self.agent_repo.get_agent_definition(agent_id)
        
        if not user or not agent:
            raise ValueError("User or agent not found")
        
        # Check permissions
        if agent.owner_id != user_id:
            self.permission_service.require_permission(
                user, "agent", agent_id, PermissionLevel.READ
            )
        
        # Get all instances for this agent
        instances = await self.agent_repo.get_agent_instances_by_definition(agent_id)
        
        # Calculate metrics
        total_requests = sum(inst.total_requests for inst in instances)
        total_successes = sum(inst.successful_requests for inst in instances)
        total_failures = sum(inst.failed_requests for inst in instances)
        
        active_instances = len([inst for inst in instances if inst.status == AgentStatus.ACTIVE])
        
        avg_response_time = 0
        if total_requests > 0:
            total_response_time = sum(inst.total_response_time for inst in instances)
            avg_response_time = total_response_time / total_requests
        
        success_rate = 0
        if total_requests > 0:
            success_rate = (total_successes / total_requests) * 100
        
        return {
            "agent_id": agent_id,
            "agent_name": agent.name,
            "total_instances": len(instances),
            "active_instances": active_instances,
            "total_requests": total_requests,
            "successful_requests": total_successes,
            "failed_requests": total_failures,
            "success_rate": success_rate,
            "average_response_time": avg_response_time,
            "usage_trend": await self._get_usage_trend(agent_id, days)
        }
    
    async def _validate_agent_for_publishing(self, agent: AgentDefinition) -> List[str]:
        """
        Validate that an agent is ready for publishing.
        """
        errors = []
        
        if not agent.name:
            errors.append("Agent name is required")
        
        if not agent.description:
            errors.append("Agent description is required")
        
        if not agent.base_model:
            errors.append("Base model is required")
        
        if not agent.instruction_template:
            errors.append("Instruction template is required")
        
        # Validate settings schema
        for key, schema in agent.settings_schema.items():
            if not isinstance(schema, dict):
                errors.append(f"Invalid schema for setting '{key}'")
        
        # Validate default settings against schema
        schema_errors = agent.validate_settings({
            key: value.value for key, value in agent.default_settings.items()
        })
        errors.extend(schema_errors)
        
        return errors
    
    async def _get_usage_trend(self, agent_id: str, days: int) -> List[Dict[str, Any]]:
        """
        Get usage trend data for an agent over the specified number of days.
        """
        # This would typically query time-series data
        # Placeholder implementation
        return []
    
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