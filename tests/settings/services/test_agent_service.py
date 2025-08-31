"""
Integration tests for AgentService.
"""

import pytest
import uuid
from datetime import datetime, timezone
from typing import Dict, Any

from agentsmcp.settings.services.agent_service import AgentService
from agentsmcp.settings.adapters.memory_repositories import (
    InMemoryAgentRepository,
    InMemoryUserRepository,
    InMemoryAuditRepository,
)
from agentsmcp.settings.domain.entities import UserProfile, AgentDefinition, AgentInstance
from agentsmcp.settings.domain.value_objects import (
    AgentStatus,
    PermissionLevel,
    ValidationRule,
)
from agentsmcp.settings.events.event_publisher import EventPublisher


@pytest.fixture
def repositories():
    """Create repository fixtures."""
    return {
        "agent": InMemoryAgentRepository(),
        "user": InMemoryUserRepository(),
        "audit": InMemoryAuditRepository(),
    }


@pytest.fixture
def event_publisher():
    """Create event publisher fixture."""
    return EventPublisher()


@pytest.fixture
def agent_service(repositories, event_publisher):
    """Create agent service fixture."""
    return AgentService(
        agent_repository=repositories["agent"],
        user_repository=repositories["user"],
        audit_repository=repositories["audit"],
        event_publisher=event_publisher
    )


@pytest.fixture
def sample_user():
    """Create sample user fixture."""
    return UserProfile(
        user_id="test_user_123",
        username="testuser",
        email="test@example.com",
        permissions=[PermissionLevel.READ, PermissionLevel.WRITE]
    )


class TestAgentService:
    """Test AgentService integration."""
    
    @pytest.mark.asyncio
    async def test_create_agent_definition(self, agent_service, repositories, sample_user):
        """Test creating a new agent definition."""
        await repositories["user"].save_user(sample_user)
        
        config = {
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        agent_id = await agent_service.create_agent_definition(
            user_id=sample_user.user_id,
            name="Test Agent",
            description="A test agent for automated tasks",
            configuration=config
        )
        
        assert isinstance(agent_id, str)
        
        # Verify agent was saved
        agent_def = await repositories["agent"].get_agent_definition(agent_id)
        assert agent_def is not None
        assert agent_def.name == "Test Agent"
        assert agent_def.description == "A test agent for automated tasks"
        assert agent_def.configuration == config
        assert agent_def.user_id == sample_user.user_id
        assert agent_def.status == AgentStatus.DRAFT
        
        # Verify audit entry was created
        audit_entries = await repositories["audit"].get_user_entries(sample_user.user_id)
        create_entry = next((e for e in audit_entries if e.action == "agent.create"), None)
        assert create_entry is not None
        assert create_entry.resource_id == agent_id
    
    @pytest.mark.asyncio
    async def test_update_agent_definition(self, agent_service, repositories, sample_user):
        """Test updating an agent definition."""
        await repositories["user"].save_user(sample_user)
        
        # Create agent first
        agent_id = await agent_service.create_agent_definition(
            user_id=sample_user.user_id,
            name="Original Agent",
            description="Original description",
            configuration={"model": "gpt-3.5"}
        )
        
        # Update agent
        new_config = {"model": "gpt-4", "temperature": 0.8}
        updated = await agent_service.update_agent_definition(
            user_id=sample_user.user_id,
            agent_id=agent_id,
            name="Updated Agent",
            description="Updated description",
            configuration=new_config
        )
        
        assert updated is True
        
        # Verify updates
        agent_def = await repositories["agent"].get_agent_definition(agent_id)
        assert agent_def.name == "Updated Agent"
        assert agent_def.description == "Updated description"
        assert agent_def.configuration == new_config
        assert agent_def.updated_at > agent_def.created_at
    
    @pytest.mark.asyncio
    async def test_add_validation_rule(self, agent_service, repositories, sample_user):
        """Test adding validation rule to agent."""
        await repositories["user"].save_user(sample_user)
        
        # Create agent
        agent_id = await agent_service.create_agent_definition(
            user_id=sample_user.user_id,
            name="Validation Test Agent",
            description="Agent for testing validation"
        )
        
        # Add validation rule
        rule = ValidationRule(
            rule_type="pattern",
            rule_value=r"^\w+@\w+\.\w+$",
            error_message="Must be a valid email format"
        )
        
        success = await agent_service.add_validation_rule(
            user_id=sample_user.user_id,
            agent_id=agent_id,
            rule=rule
        )
        
        assert success is True
        
        # Verify rule was added
        agent_def = await repositories["agent"].get_agent_definition(agent_id)
        assert rule in agent_def.validation_rules
    
    @pytest.mark.asyncio
    async def test_change_agent_status(self, agent_service, repositories, sample_user):
        """Test changing agent status."""
        await repositories["user"].save_user(sample_user)
        
        # Create agent
        agent_id = await agent_service.create_agent_definition(
            user_id=sample_user.user_id,
            name="Status Test Agent",
            description="Agent for testing status changes"
        )
        
        # Change status to testing
        success = await agent_service.change_agent_status(
            user_id=sample_user.user_id,
            agent_id=agent_id,
            new_status=AgentStatus.TESTING
        )
        
        assert success is True
        
        # Verify status changed
        agent_def = await repositories["agent"].get_agent_definition(agent_id)
        assert agent_def.status == AgentStatus.TESTING
        
        # Verify audit entry
        audit_entries = await repositories["audit"].get_user_entries(sample_user.user_id)
        status_entry = next((e for e in audit_entries if e.action == "agent.status_change"), None)
        assert status_entry is not None
        assert status_entry.details["new_status"] == AgentStatus.TESTING.value
    
    @pytest.mark.asyncio
    async def test_delete_agent_definition(self, agent_service, repositories, sample_user):
        """Test deleting an agent definition."""
        await repositories["user"].save_user(sample_user)
        
        # Create agent
        agent_id = await agent_service.create_agent_definition(
            user_id=sample_user.user_id,
            name="Delete Test Agent",
            description="Agent for testing deletion"
        )
        
        # Verify agent exists
        agent_def = await repositories["agent"].get_agent_definition(agent_id)
        assert agent_def is not None
        
        # Delete agent
        success = await agent_service.delete_agent_definition(
            user_id=sample_user.user_id,
            agent_id=agent_id
        )
        
        assert success is True
        
        # Verify agent was deleted
        agent_def = await repositories["agent"].get_agent_definition(agent_id)
        assert agent_def is None
        
        # Verify audit entry
        audit_entries = await repositories["audit"].get_user_entries(sample_user.user_id)
        delete_entry = next((e for e in audit_entries if e.action == "agent.delete"), None)
        assert delete_entry is not None
        assert delete_entry.resource_id == agent_id
    
    @pytest.mark.asyncio
    async def test_create_agent_instance(self, agent_service, repositories, sample_user):
        """Test creating an agent instance."""
        await repositories["user"].save_user(sample_user)
        
        # Create agent definition first
        agent_def_id = await agent_service.create_agent_definition(
            user_id=sample_user.user_id,
            name="Instance Test Agent",
            description="Agent for testing instances"
        )
        
        # Create agent instance
        session_id = str(uuid.uuid4())
        runtime_config = {"current_task": "analysis", "context": "document_review"}
        
        instance_id = await agent_service.create_agent_instance(
            user_id=sample_user.user_id,
            agent_definition_id=agent_def_id,
            session_id=session_id,
            runtime_config=runtime_config
        )
        
        assert isinstance(instance_id, str)
        
        # Verify instance was created
        instance = await repositories["agent"].get_agent_instance(instance_id)
        assert instance is not None
        assert instance.agent_definition_id == agent_def_id
        assert instance.user_id == sample_user.user_id
        assert instance.session_id == session_id
        assert instance.runtime_config == runtime_config
        assert instance.status == AgentStatus.ACTIVE
    
    @pytest.mark.asyncio
    async def test_update_instance_config(self, agent_service, repositories, sample_user):
        """Test updating agent instance runtime configuration."""
        await repositories["user"].save_user(sample_user)
        
        # Create agent definition and instance
        agent_def_id = await agent_service.create_agent_definition(
            user_id=sample_user.user_id,
            name="Config Test Agent",
            description="Agent for testing config updates"
        )
        
        instance_id = await agent_service.create_agent_instance(
            user_id=sample_user.user_id,
            agent_definition_id=agent_def_id,
            session_id=str(uuid.uuid4())
        )
        
        # Update instance configuration
        new_config = {
            "current_task": "code_review",
            "context": "python_file",
            "settings": {"strict_mode": True}
        }
        
        success = await agent_service.update_instance_config(
            user_id=sample_user.user_id,
            instance_id=instance_id,
            runtime_config=new_config
        )
        
        assert success is True
        
        # Verify configuration was updated
        instance = await repositories["agent"].get_agent_instance(instance_id)
        assert instance.runtime_config == new_config
    
    @pytest.mark.asyncio
    async def test_update_performance_metrics(self, agent_service, repositories, sample_user):
        """Test updating agent instance performance metrics."""
        await repositories["user"].save_user(sample_user)
        
        # Create agent definition and instance
        agent_def_id = await agent_service.create_agent_definition(
            user_id=sample_user.user_id,
            name="Metrics Test Agent",
            description="Agent for testing metrics"
        )
        
        instance_id = await agent_service.create_agent_instance(
            user_id=sample_user.user_id,
            agent_definition_id=agent_def_id,
            session_id=str(uuid.uuid4())
        )
        
        # Update performance metrics
        metrics = {
            "response_time_ms": 1500,
            "accuracy_score": 0.95,
            "tasks_completed": 42,
            "errors_count": 2
        }
        
        success = await agent_service.update_performance_metrics(
            user_id=sample_user.user_id,
            instance_id=instance_id,
            metrics=metrics
        )
        
        assert success is True
        
        # Verify metrics were updated
        instance = await repositories["agent"].get_agent_instance(instance_id)
        assert instance.performance_metrics == metrics
    
    @pytest.mark.asyncio
    async def test_get_agent_instances_by_definition(self, agent_service, repositories, sample_user):
        """Test getting all instances for an agent definition."""
        await repositories["user"].save_user(sample_user)
        
        # Create agent definition
        agent_def_id = await agent_service.create_agent_definition(
            user_id=sample_user.user_id,
            name="Multi-Instance Agent",
            description="Agent with multiple instances"
        )
        
        # Create multiple instances
        instance_ids = []
        for i in range(3):
            instance_id = await agent_service.create_agent_instance(
                user_id=sample_user.user_id,
                agent_definition_id=agent_def_id,
                session_id=f"session_{i}"
            )
            instance_ids.append(instance_id)
        
        # Get instances by definition
        instances = await agent_service.get_agent_instances_by_definition(
            user_id=sample_user.user_id,
            agent_definition_id=agent_def_id
        )
        
        assert len(instances) == 3
        retrieved_ids = [instance.id for instance in instances]
        assert all(instance_id in retrieved_ids for instance_id in instance_ids)
    
    @pytest.mark.asyncio
    async def test_get_user_agents(self, agent_service, repositories, sample_user):
        """Test getting all agents for a user."""
        await repositories["user"].save_user(sample_user)
        
        # Create multiple agent definitions
        agent_ids = []
        for i in range(2):
            agent_id = await agent_service.create_agent_definition(
                user_id=sample_user.user_id,
                name=f"User Agent {i}",
                description=f"Agent {i} for user testing"
            )
            agent_ids.append(agent_id)
        
        # Get user agents
        agents = await agent_service.get_user_agents(sample_user.user_id)
        
        assert len(agents) == 2
        retrieved_ids = [agent.id for agent in agents]
        assert all(agent_id in retrieved_ids for agent_id in agent_ids)
    
    @pytest.mark.asyncio
    async def test_unauthorized_agent_access(self, agent_service, repositories, sample_user):
        """Test that unauthorized users cannot access other users' agents."""
        # Create authorized user and agent
        await repositories["user"].save_user(sample_user)
        
        agent_id = await agent_service.create_agent_definition(
            user_id=sample_user.user_id,
            name="Private Agent",
            description="Should not be accessible to others"
        )
        
        # Create unauthorized user
        unauthorized_user = UserProfile(
            user_id="unauthorized_user",
            username="hacker",
            permissions=[PermissionLevel.READ]
        )
        await repositories["user"].save_user(unauthorized_user)
        
        # Try to update the agent as unauthorized user - should fail
        with pytest.raises(PermissionError):
            await agent_service.update_agent_definition(
                user_id=unauthorized_user.user_id,
                agent_id=agent_id,
                name="Hacked Agent"
            )
    
    @pytest.mark.asyncio
    async def test_agent_lifecycle_workflow(self, agent_service, repositories, sample_user):
        """Test complete agent lifecycle workflow."""
        await repositories["user"].save_user(sample_user)
        
        # 1. Create agent definition
        agent_id = await agent_service.create_agent_definition(
            user_id=sample_user.user_id,
            name="Lifecycle Agent",
            description="Agent for testing lifecycle",
            configuration={"model": "gpt-4"}
        )
        
        # 2. Add validation rules
        rule = ValidationRule("required", True, "Name is required")
        await agent_service.add_validation_rule(
            user_id=sample_user.user_id,
            agent_id=agent_id,
            rule=rule
        )
        
        # 3. Change status to testing
        await agent_service.change_agent_status(
            user_id=sample_user.user_id,
            agent_id=agent_id,
            new_status=AgentStatus.TESTING
        )
        
        # 4. Create instance for testing
        instance_id = await agent_service.create_agent_instance(
            user_id=sample_user.user_id,
            agent_definition_id=agent_id,
            session_id="test_session"
        )
        
        # 5. Update performance metrics
        await agent_service.update_performance_metrics(
            user_id=sample_user.user_id,
            instance_id=instance_id,
            metrics={"test_score": 0.9}
        )
        
        # 6. Change status to active after successful testing
        await agent_service.change_agent_status(
            user_id=sample_user.user_id,
            agent_id=agent_id,
            new_status=AgentStatus.ACTIVE
        )
        
        # Verify final state
        agent_def = await repositories["agent"].get_agent_definition(agent_id)
        assert agent_def.status == AgentStatus.ACTIVE
        assert len(agent_def.validation_rules) == 1
        
        instance = await repositories["agent"].get_agent_instance(instance_id)
        assert instance.performance_metrics["test_score"] == 0.9
        
        # Check audit trail
        audit_entries = await repositories["audit"].get_user_entries(sample_user.user_id)
        actions = [entry.action for entry in audit_entries]
        expected_actions = [
            "agent.create",
            "agent.validation_rule_add",
            "agent.status_change",
            "agent_instance.create",
            "agent_instance.performance_update",
            "agent.status_change"
        ]
        
        for action in expected_actions:
            assert action in actions
    
    @pytest.mark.asyncio
    async def test_instance_cleanup_on_agent_deletion(self, agent_service, repositories, sample_user):
        """Test that instances are cleaned up when agent definition is deleted."""
        await repositories["user"].save_user(sample_user)
        
        # Create agent definition
        agent_id = await agent_service.create_agent_definition(
            user_id=sample_user.user_id,
            name="Cleanup Test Agent",
            description="Agent for testing cleanup"
        )
        
        # Create instances
        instance_ids = []
        for i in range(2):
            instance_id = await agent_service.create_agent_instance(
                user_id=sample_user.user_id,
                agent_definition_id=agent_id,
                session_id=f"cleanup_session_{i}"
            )
            instance_ids.append(instance_id)
        
        # Verify instances exist
        for instance_id in instance_ids:
            instance = await repositories["agent"].get_agent_instance(instance_id)
            assert instance is not None
        
        # Delete agent definition
        await agent_service.delete_agent_definition(
            user_id=sample_user.user_id,
            agent_id=agent_id
        )
        
        # Verify instances were also deleted
        for instance_id in instance_ids:
            instance = await repositories["agent"].get_agent_instance(instance_id)
            assert instance is None


if __name__ == "__main__":
    pytest.main([__file__])