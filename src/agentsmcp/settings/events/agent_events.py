"""
Agent-related domain events.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .base_events import DomainEvent
from ..domain.value_objects import AgentStatus


@dataclass(frozen=True)
class AgentDefinitionCreatedEvent(DomainEvent):
    """Event raised when a new agent definition is created."""
    
    agent_id: str = None
    name: str = None
    base_model: str = None
    category: str = None
    owner_id: str = None
    organization_id: Optional[str] = None
    
    @property
    def event_type(self) -> str:
        return "agent_definition_created"
    
    @property
    def aggregate_id(self) -> str:
        return self.agent_id


@dataclass(frozen=True)
class AgentDefinitionUpdatedEvent(DomainEvent):
    """Event raised when an agent definition is updated."""
    
    agent_id: str = None
    updated_fields: List[str] = None
    old_values: Dict[str, Any] = None
    new_values: Dict[str, Any] = None
    
    @property
    def event_type(self) -> str:
        return "agent_definition_updated"
    
    @property
    def aggregate_id(self) -> str:
        return self.agent_id


@dataclass(frozen=True)
class AgentDefinitionStatusChangedEvent(DomainEvent):
    """Event raised when an agent definition's status changes."""
    
    agent_id: str = None
    old_status: AgentStatus = None
    new_status: AgentStatus = None
    reason: str = ""
    
    @property
    def event_type(self) -> str:
        return "agent_definition_status_changed"
    
    @property
    def aggregate_id(self) -> str:
        return self.agent_id


@dataclass(frozen=True)
class AgentDefinitionPublishedEvent(DomainEvent):
    """Event raised when an agent definition is published."""
    
    agent_id: str = None
    name: str = None
    version: str = None
    publisher_id: str = None
    
    @property
    def event_type(self) -> str:
        return "agent_definition_published"
    
    @property
    def aggregate_id(self) -> str:
        return self.agent_id


@dataclass(frozen=True)
class AgentDefinitionDeletedEvent(DomainEvent):
    """Event raised when an agent definition is deleted."""
    
    agent_id: str = None
    name: str = None
    owner_id: str = None
    active_instances_count: int = None
    
    @property
    def event_type(self) -> str:
        return "agent_definition_deleted"
    
    @property
    def aggregate_id(self) -> str:
        return self.agent_id


@dataclass(frozen=True)
class AgentInstanceCreatedEvent(DomainEvent):
    """Event raised when a new agent instance is created."""
    
    instance_id: str = None
    agent_id: str = None
    name: str = None
    session_id: Optional[str] = None
    
    @property
    def event_type(self) -> str:
        return "agent_instance_created"
    
    @property
    def aggregate_id(self) -> str:
        return self.instance_id


@dataclass(frozen=True)
class AgentInstanceStartedEvent(DomainEvent):
    """Event raised when an agent instance is started."""
    
    instance_id: str = None
    agent_id: str = None
    started_at: datetime = None
    
    @property
    def event_type(self) -> str:
        return "agent_instance_started"
    
    @property
    def aggregate_id(self) -> str:
        return self.instance_id


@dataclass(frozen=True)
class AgentInstanceStoppedEvent(DomainEvent):
    """Event raised when an agent instance is stopped."""
    
    instance_id: str = None
    agent_id: str = None
    stopped_at: datetime = None
    reason: str = "manual"
    
    @property
    def event_type(self) -> str:
        return "agent_instance_stopped"
    
    @property
    def aggregate_id(self) -> str:
        return self.instance_id


@dataclass(frozen=True)
class AgentInstanceDeletedEvent(DomainEvent):
    """Event raised when an agent instance is deleted."""
    
    instance_id: str = None
    agent_id: str = None
    total_requests: int = None
    success_rate: float = None
    
    @property
    def event_type(self) -> str:
        return "agent_instance_deleted"
    
    @property
    def aggregate_id(self) -> str:
        return self.instance_id


@dataclass(frozen=True)
class AgentPerformanceUpdatedEvent(DomainEvent):
    """Event raised when agent performance metrics are updated."""
    
    instance_id: str = None
    agent_id: str = None
    response_time: float = None
    success: bool = None
    new_success_rate: float = None
    old_success_rate: float = None
    
    @property
    def event_type(self) -> str:
        return "agent_performance_updated"
    
    @property
    def aggregate_id(self) -> str:
        return self.instance_id


@dataclass(frozen=True)
class AgentCapabilityAddedEvent(DomainEvent):
    """Event raised when a capability is added to an agent."""
    
    agent_id: str = None
    capability_name: str = None
    capability_version: str = None
    
    @property
    def event_type(self) -> str:
        return "agent_capability_added"
    
    @property
    def aggregate_id(self) -> str:
        return self.agent_id


@dataclass(frozen=True)
class AgentCapabilityRemovedEvent(DomainEvent):
    """Event raised when a capability is removed from an agent."""
    
    agent_id: str = None
    capability_name: str = None
    
    @property
    def event_type(self) -> str:
        return "agent_capability_removed"
    
    @property
    def aggregate_id(self) -> str:
        return self.agent_id


@dataclass(frozen=True)
class AgentInstructionTemplateUpdatedEvent(DomainEvent):
    """Event raised when an agent's instruction template is updated."""
    
    agent_id: str = None
    template_id: str = None
    template_version: str = None
    
    @property
    def event_type(self) -> str:
        return "agent_instruction_template_updated"
    
    @property
    def aggregate_id(self) -> str:
        return self.agent_id


@dataclass(frozen=True)
class AgentSettingsSchemaUpdatedEvent(DomainEvent):
    """Event raised when an agent's settings schema is updated."""
    
    agent_id: str = None
    added_settings: List[str] = None
    removed_settings: List[str] = None
    modified_settings: List[str] = None
    
    @property
    def event_type(self) -> str:
        return "agent_settings_schema_updated"
    
    @property
    def aggregate_id(self) -> str:
        return self.agent_id