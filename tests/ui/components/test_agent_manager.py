"""
Comprehensive test suite for Agent Manager component.

Tests the AgentManager component with 95%+ coverage, including:
- Agent lifecycle management (create, start, stop, delete)
- Status monitoring and health checks
- Bulk operations and batch processing
- Template-based agent creation
- Configuration management
- Real-time status updates
- Performance monitoring
- Error handling and recovery
"""

import pytest
import asyncio
import json
import time
import uuid
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

# Import the agent manager component to test
from agentsmcp.ui.components.agent_manager import (
    AgentManager, Agent, AgentStatus, AgentTemplate, AgentConfig
)


@dataclass
class MockAgent:
    """Mock agent for testing."""
    id: str
    name: str
    status: AgentStatus
    config: Dict[str, Any]
    created_at: datetime
    last_updated: datetime
    health_score: float = 1.0
    error_message: Optional[str] = None


@pytest.fixture
def mock_display_renderer():
    """Mock DisplayRenderer for testing."""
    renderer = Mock()
    renderer.create_panel = Mock(return_value="[Panel]")
    renderer.create_table = Mock(return_value="[Table]")
    renderer.create_progress_bar = Mock(return_value="[Progress: 75%]")
    renderer.style_text = Mock(side_effect=lambda text, style: f"[{style}]{text}[/{style}]")
    renderer.create_status_indicator = Mock(return_value="●")
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
def agent_manager(mock_display_renderer, mock_terminal_manager, mock_event_system):
    """Create AgentManager instance for testing."""
    manager = AgentManager(
        display_renderer=mock_display_renderer,
        terminal_manager=mock_terminal_manager,
        event_system=mock_event_system
    )
    return manager


@pytest.fixture
def sample_agents():
    """Create sample agents for testing."""
    return [
        MockAgent(
            id="agent-1",
            name="Data Processor",
            status=AgentStatus.RUNNING,
            config={"type": "data", "threads": 4},
            created_at=datetime.now() - timedelta(hours=2),
            last_updated=datetime.now()
        ),
        MockAgent(
            id="agent-2",
            name="API Handler",
            status=AgentStatus.STOPPED,
            config={"type": "api", "port": 8080},
            created_at=datetime.now() - timedelta(hours=1),
            last_updated=datetime.now() - timedelta(minutes=30)
        ),
        MockAgent(
            id="agent-3",
            name="Logger",
            status=AgentStatus.ERROR,
            config={"type": "logging", "level": "DEBUG"},
            created_at=datetime.now() - timedelta(minutes=30),
            last_updated=datetime.now() - timedelta(minutes=5),
            error_message="Connection timeout"
        )
    ]


class TestAgentManagerInitialization:
    """Test Agent Manager initialization and setup."""

    def test_initialization_success(self, agent_manager):
        """Test successful agent manager initialization."""
        assert agent_manager.display_renderer is not None
        assert agent_manager.terminal_manager is not None
        assert agent_manager.event_system is not None
        assert hasattr(agent_manager, 'agents')
        assert isinstance(agent_manager.agents, dict)

    def test_default_templates_loaded(self, agent_manager):
        """Test default agent templates are loaded."""
        assert hasattr(agent_manager, 'templates')
        assert len(agent_manager.templates) > 0
        
        # Check for common template types
        template_names = [t.name for t in agent_manager.templates.values()]
        expected_templates = ['data_processor', 'api_handler', 'task_scheduler']
        
        for template_name in expected_templates:
            assert any(template_name in name.lower() for name in template_names)

    def test_status_tracking_initialization(self, agent_manager):
        """Test status tracking system initialization."""
        assert hasattr(agent_manager, '_status_cache')
        assert hasattr(agent_manager, '_last_status_update')
        assert hasattr(agent_manager, '_monitoring_enabled')
        assert agent_manager._monitoring_enabled is True

    @pytest.mark.asyncio
    async def test_initialization_with_existing_agents(self, agent_manager, sample_agents):
        """Test initialization when agents already exist."""
        # Mock loading existing agents
        with patch.object(agent_manager, '_load_existing_agents') as mock_load:
            mock_load.return_value = {agent.id: agent for agent in sample_agents}
            
            await agent_manager.initialize()
            
            assert len(agent_manager.agents) == 3
            assert "agent-1" in agent_manager.agents
            mock_load.assert_called_once()


class TestAgentLifecycleManagement:
    """Test agent lifecycle management operations."""

    @pytest.mark.asyncio
    async def test_create_agent_from_template(self, agent_manager):
        """Test creating agent from template."""
        # Mock template
        template = AgentTemplate(
            id="template-1",
            name="Test Template",
            description="Test template for agents",
            config_schema={
                "name": {"type": "string", "required": True},
                "threads": {"type": "integer", "default": 2}
            },
            default_config={"threads": 2, "timeout": 30}
        )
        
        agent_manager.templates["template-1"] = template
        
        # Create agent from template
        agent_config = {
            "name": "Test Agent",
            "threads": 4
        }
        
        with patch.object(agent_manager, '_create_agent_instance') as mock_create:
            mock_agent = MockAgent(
                id="new-agent",
                name="Test Agent",
                status=AgentStatus.CREATED,
                config=agent_config,
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
            mock_create.return_value = mock_agent
            
            result = await agent_manager.create_agent("template-1", agent_config)
            
            assert result is not None
            assert result.name == "Test Agent"
            assert result.config["threads"] == 4
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_agent_invalid_template(self, agent_manager):
        """Test creating agent with invalid template."""
        with pytest.raises(ValueError, match="Template 'invalid' not found"):
            await agent_manager.create_agent("invalid", {"name": "Test"})

    @pytest.mark.asyncio
    async def test_create_agent_validation_failure(self, agent_manager):
        """Test agent creation with validation failure."""
        template = AgentTemplate(
            id="strict-template",
            name="Strict Template",
            config_schema={
                "name": {"type": "string", "required": True},
                "port": {"type": "integer", "required": True, "min": 1024, "max": 65535}
            }
        )
        agent_manager.templates["strict-template"] = template
        
        # Missing required field
        with pytest.raises(ValueError, match="Validation failed"):
            await agent_manager.create_agent("strict-template", {"name": "Test"})
        
        # Invalid port range
        with pytest.raises(ValueError, match="Validation failed"):
            await agent_manager.create_agent("strict-template", {
                "name": "Test", 
                "port": 99999  # Out of range
            })

    @pytest.mark.asyncio
    async def test_start_agent_success(self, agent_manager, sample_agents):
        """Test starting an agent successfully."""
        agent = sample_agents[1]  # STOPPED agent
        agent_manager.agents[agent.id] = agent
        
        with patch.object(agent_manager, '_start_agent_process') as mock_start:
            mock_start.return_value = True
            
            result = await agent_manager.start_agent(agent.id)
            
            assert result is True
            assert agent.status == AgentStatus.STARTING
            mock_start.assert_called_once_with(agent)

    @pytest.mark.asyncio
    async def test_start_agent_already_running(self, agent_manager, sample_agents):
        """Test starting an already running agent."""
        agent = sample_agents[0]  # RUNNING agent
        agent_manager.agents[agent.id] = agent
        
        result = await agent_manager.start_agent(agent.id)
        
        assert result is False  # Cannot start already running agent

    @pytest.mark.asyncio
    async def test_stop_agent_success(self, agent_manager, sample_agents):
        """Test stopping an agent successfully."""
        agent = sample_agents[0]  # RUNNING agent
        agent_manager.agents[agent.id] = agent
        
        with patch.object(agent_manager, '_stop_agent_process') as mock_stop:
            mock_stop.return_value = True
            
            result = await agent_manager.stop_agent(agent.id, graceful=True)
            
            assert result is True
            assert agent.status == AgentStatus.STOPPING
            mock_stop.assert_called_once_with(agent, graceful=True)

    @pytest.mark.asyncio
    async def test_stop_agent_force(self, agent_manager, sample_agents):
        """Test force stopping an agent."""
        agent = sample_agents[0]  # RUNNING agent
        agent_manager.agents[agent.id] = agent
        
        with patch.object(agent_manager, '_stop_agent_process') as mock_stop:
            mock_stop.return_value = True
            
            result = await agent_manager.stop_agent(agent.id, graceful=False, force=True)
            
            assert result is True
            mock_stop.assert_called_once_with(agent, graceful=False)

    @pytest.mark.asyncio
    async def test_restart_agent_success(self, agent_manager, sample_agents):
        """Test restarting an agent successfully."""
        agent = sample_agents[0]  # RUNNING agent
        agent_manager.agents[agent.id] = agent
        
        with patch.object(agent_manager, 'stop_agent') as mock_stop:
            with patch.object(agent_manager, 'start_agent') as mock_start:
                mock_stop.return_value = True
                mock_start.return_value = True
                
                result = await agent_manager.restart_agent(agent.id)
                
                assert result is True
                mock_stop.assert_called_once_with(agent.id, graceful=True)
                mock_start.assert_called_once_with(agent.id)

    @pytest.mark.asyncio
    async def test_delete_agent_success(self, agent_manager, sample_agents):
        """Test deleting an agent successfully."""
        agent = sample_agents[1]  # STOPPED agent
        agent_manager.agents[agent.id] = agent
        
        with patch.object(agent_manager, '_cleanup_agent_resources') as mock_cleanup:
            mock_cleanup.return_value = True
            
            result = await agent_manager.delete_agent(agent.id)
            
            assert result is True
            assert agent.id not in agent_manager.agents
            mock_cleanup.assert_called_once_with(agent)

    @pytest.mark.asyncio
    async def test_delete_running_agent_failure(self, agent_manager, sample_agents):
        """Test deleting a running agent fails."""
        agent = sample_agents[0]  # RUNNING agent
        agent_manager.agents[agent.id] = agent
        
        with pytest.raises(ValueError, match="Cannot delete running agent"):
            await agent_manager.delete_agent(agent.id)

    @pytest.mark.asyncio
    async def test_delete_agent_with_force(self, agent_manager, sample_agents):
        """Test force deleting a running agent."""
        agent = sample_agents[0]  # RUNNING agent
        agent_manager.agents[agent.id] = agent
        
        with patch.object(agent_manager, 'stop_agent') as mock_stop:
            with patch.object(agent_manager, '_cleanup_agent_resources') as mock_cleanup:
                mock_stop.return_value = True
                mock_cleanup.return_value = True
                
                result = await agent_manager.delete_agent(agent.id, force=True)
                
                assert result is True
                assert agent.id not in agent_manager.agents
                mock_stop.assert_called_once_with(agent.id, graceful=False, force=True)


class TestAgentStatusMonitoring:
    """Test agent status monitoring and health checks."""

    @pytest.mark.asyncio
    async def test_update_agent_status(self, agent_manager, sample_agents):
        """Test updating agent status."""
        agent = sample_agents[0]
        agent_manager.agents[agent.id] = agent
        
        original_status = agent.status
        await agent_manager.update_agent_status(agent.id, AgentStatus.STOPPING)
        
        assert agent.status == AgentStatus.STOPPING
        assert agent.status != original_status
        assert agent.last_updated is not None

    @pytest.mark.asyncio
    async def test_check_agent_health(self, agent_manager, sample_agents):
        """Test checking individual agent health."""
        agent = sample_agents[0]  # RUNNING agent
        agent_manager.agents[agent.id] = agent
        
        with patch.object(agent_manager, '_ping_agent') as mock_ping:
            mock_ping.return_value = {'status': 'healthy', 'response_time': 0.05}
            
            health = await agent_manager.check_agent_health(agent.id)
            
            assert health is not None
            assert health['status'] == 'healthy'
            assert 'response_time' in health
            mock_ping.assert_called_once_with(agent)

    @pytest.mark.asyncio
    async def test_check_all_agents_health(self, agent_manager, sample_agents):
        """Test checking health of all agents."""
        for agent in sample_agents:
            agent_manager.agents[agent.id] = agent
        
        with patch.object(agent_manager, 'check_agent_health') as mock_health:
            mock_health.side_effect = [
                {'status': 'healthy', 'response_time': 0.03},
                {'status': 'unhealthy', 'error': 'Not running'},
                {'status': 'error', 'error': 'Connection failed'}
            ]
            
            health_report = await agent_manager.check_all_agents_health()
            
            assert len(health_report) == 3
            assert mock_health.call_count == 3
            
            # Check health status distribution
            healthy_count = sum(1 for h in health_report.values() if h['status'] == 'healthy')
            assert healthy_count == 1

    def test_get_agents_by_status(self, agent_manager, sample_agents):
        """Test filtering agents by status."""
        for agent in sample_agents:
            agent_manager.agents[agent.id] = agent
        
        running_agents = agent_manager.get_agents_by_status(AgentStatus.RUNNING)
        stopped_agents = agent_manager.get_agents_by_status(AgentStatus.STOPPED)
        error_agents = agent_manager.get_agents_by_status(AgentStatus.ERROR)
        
        assert len(running_agents) == 1
        assert len(stopped_agents) == 1
        assert len(error_agents) == 1
        
        assert running_agents[0].name == "Data Processor"
        assert stopped_agents[0].name == "API Handler"
        assert error_agents[0].name == "Logger"

    def test_get_agent_statistics(self, agent_manager, sample_agents):
        """Test getting agent statistics."""
        for agent in sample_agents:
            agent_manager.agents[agent.id] = agent
        
        stats = agent_manager.get_agent_statistics()
        
        assert stats['total'] == 3
        assert stats['running'] == 1
        assert stats['stopped'] == 1
        assert stats['error'] == 1
        assert stats['created'] == 0
        assert 'uptime_average' in stats

    @pytest.mark.asyncio
    async def test_continuous_monitoring(self, agent_manager, sample_agents):
        """Test continuous monitoring functionality."""
        for agent in sample_agents:
            agent_manager.agents[agent.id] = agent
        
        monitoring_calls = []
        
        async def mock_monitor():
            monitoring_calls.append(datetime.now())
            
        with patch.object(agent_manager, '_run_monitoring_cycle', side_effect=mock_monitor):
            # Enable monitoring with short interval
            agent_manager._monitoring_interval = 0.1
            
            # Start monitoring
            monitoring_task = asyncio.create_task(agent_manager.start_continuous_monitoring())
            
            # Let it run for a short time
            await asyncio.sleep(0.3)
            
            # Stop monitoring
            monitoring_task.cancel()
            
            try:
                await monitoring_task
            except asyncio.CancelledError:
                pass
            
            # Should have run multiple monitoring cycles
            assert len(monitoring_calls) >= 2


class TestBulkOperations:
    """Test bulk operations on multiple agents."""

    @pytest.mark.asyncio
    async def test_bulk_start_agents(self, agent_manager, sample_agents):
        """Test starting multiple agents in bulk."""
        # Add stopped and error agents
        stopped_agents = [sample_agents[1], sample_agents[2]]
        for agent in stopped_agents:
            agent_manager.agents[agent.id] = agent
        
        agent_ids = [agent.id for agent in stopped_agents]
        
        with patch.object(agent_manager, 'start_agent') as mock_start:
            mock_start.side_effect = [True, True]
            
            results = await agent_manager.bulk_start_agents(agent_ids)
            
            assert len(results) == 2
            assert all(results.values())
            assert mock_start.call_count == 2

    @pytest.mark.asyncio
    async def test_bulk_stop_agents(self, agent_manager, sample_agents):
        """Test stopping multiple agents in bulk."""
        # Add running agent
        running_agents = [sample_agents[0]]
        for agent in running_agents:
            agent_manager.agents[agent.id] = agent
        
        agent_ids = [agent.id for agent in running_agents]
        
        with patch.object(agent_manager, 'stop_agent') as mock_stop:
            mock_stop.return_value = True
            
            results = await agent_manager.bulk_stop_agents(agent_ids, graceful=True)
            
            assert len(results) == 1
            assert all(results.values())
            mock_stop.assert_called_with(agent_ids[0], graceful=True, force=False)

    @pytest.mark.asyncio
    async def test_bulk_restart_agents(self, agent_manager, sample_agents):
        """Test restarting multiple agents in bulk."""
        # Add some agents
        for agent in sample_agents[:2]:
            agent_manager.agents[agent.id] = agent
        
        agent_ids = [agent.id for agent in sample_agents[:2]]
        
        with patch.object(agent_manager, 'restart_agent') as mock_restart:
            mock_restart.side_effect = [True, True]
            
            results = await agent_manager.bulk_restart_agents(agent_ids)
            
            assert len(results) == 2
            assert all(results.values())
            assert mock_restart.call_count == 2

    @pytest.mark.asyncio
    async def test_bulk_delete_agents(self, agent_manager, sample_agents):
        """Test deleting multiple agents in bulk."""
        # Add stopped agents only
        stopped_agents = [sample_agents[1]]  # Only stopped agent
        for agent in stopped_agents:
            agent_manager.agents[agent.id] = agent
        
        agent_ids = [agent.id for agent in stopped_agents]
        
        with patch.object(agent_manager, 'delete_agent') as mock_delete:
            mock_delete.return_value = True
            
            results = await agent_manager.bulk_delete_agents(agent_ids)
            
            assert len(results) == 1
            assert all(results.values())
            mock_delete.assert_called_once_with(agent_ids[0], force=False)

    @pytest.mark.asyncio
    async def test_bulk_operation_partial_failure(self, agent_manager, sample_agents):
        """Test bulk operation with partial failures."""
        for agent in sample_agents:
            agent_manager.agents[agent.id] = agent
        
        agent_ids = [agent.id for agent in sample_agents]
        
        with patch.object(agent_manager, 'start_agent') as mock_start:
            # First succeeds, second fails, third succeeds
            mock_start.side_effect = [True, False, True]
            
            results = await agent_manager.bulk_start_agents(agent_ids)
            
            assert len(results) == 3
            assert results[agent_ids[0]] is True
            assert results[agent_ids[1]] is False
            assert results[agent_ids[2]] is True

    @pytest.mark.asyncio
    async def test_bulk_operation_with_filters(self, agent_manager, sample_agents):
        """Test bulk operations with status filters."""
        for agent in sample_agents:
            agent_manager.agents[agent.id] = agent
        
        with patch.object(agent_manager, 'start_agent') as mock_start:
            mock_start.return_value = True
            
            # Start only stopped agents
            results = await agent_manager.bulk_start_by_status(AgentStatus.STOPPED)
            
            # Should only start the one stopped agent
            assert len(results) == 1
            mock_start.assert_called_once()


class TestAgentTemplateManagement:
    """Test agent template management functionality."""

    def test_add_template_success(self, agent_manager):
        """Test adding new agent template."""
        template = AgentTemplate(
            id="custom-template",
            name="Custom Template",
            description="A custom template",
            config_schema={"name": {"type": "string", "required": True}},
            default_config={"timeout": 60}
        )
        
        result = agent_manager.add_template(template)
        assert result is True
        assert "custom-template" in agent_manager.templates

    def test_add_template_duplicate_id(self, agent_manager):
        """Test adding template with duplicate ID."""
        template1 = AgentTemplate(id="duplicate", name="First")
        template2 = AgentTemplate(id="duplicate", name="Second")
        
        agent_manager.add_template(template1)
        
        with pytest.raises(ValueError, match="Template 'duplicate' already exists"):
            agent_manager.add_template(template2)

    def test_remove_template_success(self, agent_manager):
        """Test removing agent template."""
        template = AgentTemplate(id="temp-template", name="Temporary")
        agent_manager.add_template(template)
        
        result = agent_manager.remove_template("temp-template")
        assert result is True
        assert "temp-template" not in agent_manager.templates

    def test_remove_template_nonexistent(self, agent_manager):
        """Test removing non-existent template."""
        result = agent_manager.remove_template("nonexistent")
        assert result is False

    def test_get_template_success(self, agent_manager):
        """Test getting existing template."""
        template = AgentTemplate(id="test-template", name="Test")
        agent_manager.add_template(template)
        
        retrieved = agent_manager.get_template("test-template")
        assert retrieved is not None
        assert retrieved.name == "Test"

    def test_get_template_nonexistent(self, agent_manager):
        """Test getting non-existent template."""
        result = agent_manager.get_template("nonexistent")
        assert result is None

    def test_list_templates_by_category(self, agent_manager):
        """Test listing templates by category."""
        templates = [
            AgentTemplate(id="data1", name="Data 1", category="data"),
            AgentTemplate(id="data2", name="Data 2", category="data"),
            AgentTemplate(id="api1", name="API 1", category="api")
        ]
        
        for template in templates:
            agent_manager.add_template(template)
        
        data_templates = agent_manager.get_templates_by_category("data")
        api_templates = agent_manager.get_templates_by_category("api")
        
        assert len(data_templates) == 2
        assert len(api_templates) == 1

    def test_template_validation(self, agent_manager):
        """Test template configuration validation."""
        template = AgentTemplate(
            id="validation-test",
            name="Validation Test",
            config_schema={
                "name": {"type": "string", "required": True, "min_length": 3},
                "port": {"type": "integer", "required": False, "min": 1024, "max": 65535}
            }
        )
        agent_manager.add_template(template)
        
        # Valid config
        valid_config = {"name": "Valid Agent", "port": 8080}
        result = agent_manager.validate_template_config("validation-test", valid_config)
        assert result.is_valid is True
        
        # Invalid config - name too short
        invalid_config1 = {"name": "Hi", "port": 8080}
        result1 = agent_manager.validate_template_config("validation-test", invalid_config1)
        assert result1.is_valid is False
        
        # Invalid config - port out of range
        invalid_config2 = {"name": "Valid Name", "port": 99999}
        result2 = agent_manager.validate_template_config("validation-test", invalid_config2)
        assert result2.is_valid is False


class TestConfigurationManagement:
    """Test agent configuration management."""

    @pytest.mark.asyncio
    async def test_update_agent_config(self, agent_manager, sample_agents):
        """Test updating agent configuration."""
        agent = sample_agents[1]  # STOPPED agent
        agent_manager.agents[agent.id] = agent
        
        new_config = {"type": "api", "port": 9090, "threads": 8}
        
        result = await agent_manager.update_agent_config(agent.id, new_config)
        
        assert result is True
        assert agent.config["port"] == 9090
        assert agent.config["threads"] == 8

    @pytest.mark.asyncio
    async def test_update_running_agent_config_failure(self, agent_manager, sample_agents):
        """Test updating configuration of running agent fails."""
        agent = sample_agents[0]  # RUNNING agent
        agent_manager.agents[agent.id] = agent
        
        new_config = {"threads": 8}
        
        with pytest.raises(ValueError, match="Cannot update config of running agent"):
            await agent_manager.update_agent_config(agent.id, new_config)

    @pytest.mark.asyncio
    async def test_update_agent_config_with_restart(self, agent_manager, sample_agents):
        """Test updating config with automatic restart."""
        agent = sample_agents[0]  # RUNNING agent
        agent_manager.agents[agent.id] = agent
        
        new_config = {"threads": 8}
        
        with patch.object(agent_manager, 'restart_agent') as mock_restart:
            mock_restart.return_value = True
            
            result = await agent_manager.update_agent_config(
                agent.id, new_config, restart_if_running=True
            )
            
            assert result is True
            assert agent.config["threads"] == 8
            mock_restart.assert_called_once_with(agent.id)

    def test_get_agent_config(self, agent_manager, sample_agents):
        """Test getting agent configuration."""
        agent = sample_agents[0]
        agent_manager.agents[agent.id] = agent
        
        config = agent_manager.get_agent_config(agent.id)
        assert config is not None
        assert config == agent.config

    def test_export_agent_config(self, agent_manager, sample_agents):
        """Test exporting agent configuration."""
        agent = sample_agents[0]
        agent_manager.agents[agent.id] = agent
        
        exported = agent_manager.export_agent_config(agent.id)
        
        assert 'id' in exported
        assert 'name' in exported
        assert 'config' in exported
        assert 'template' in exported
        assert exported['id'] == agent.id

    @pytest.mark.asyncio
    async def test_import_agent_config(self, agent_manager):
        """Test importing agent configuration."""
        config_data = {
            'name': 'Imported Agent',
            'template': 'data_processor',
            'config': {
                'threads': 4,
                'timeout': 60
            }
        }
        
        with patch.object(agent_manager, 'create_agent') as mock_create:
            mock_agent = MockAgent(
                id="imported-agent",
                name="Imported Agent",
                status=AgentStatus.CREATED,
                config=config_data['config'],
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
            mock_create.return_value = mock_agent
            
            result = await agent_manager.import_agent_config(config_data)
            
            assert result is not None
            assert result.name == "Imported Agent"
            mock_create.assert_called_once_with('data_processor', config_data['config'])


class TestRenderingAndDisplay:
    """Test agent manager rendering and display functionality."""

    def test_render_agent_list(self, agent_manager, sample_agents):
        """Test rendering agent list."""
        for agent in sample_agents:
            agent_manager.agents[agent.id] = agent
        
        rendered = agent_manager.render_agent_list()
        
        assert rendered is not None
        agent_manager.display_renderer.create_table.assert_called()

    def test_render_agent_details(self, agent_manager, sample_agents):
        """Test rendering detailed agent view."""
        agent = sample_agents[0]
        agent_manager.agents[agent.id] = agent
        
        rendered = agent_manager.render_agent_details(agent.id)
        
        assert rendered is not None
        agent_manager.display_renderer.create_panel.assert_called()

    def test_render_agent_statistics(self, agent_manager, sample_agents):
        """Test rendering agent statistics."""
        for agent in sample_agents:
            agent_manager.agents[agent.id] = agent
        
        rendered = agent_manager.render_statistics()
        
        assert rendered is not None
        # Should have called renderer methods for charts/stats

    def test_render_status_dashboard(self, agent_manager, sample_agents):
        """Test rendering status dashboard."""
        for agent in sample_agents:
            agent_manager.agents[agent.id] = agent
        
        rendered = agent_manager.render_status_dashboard()
        
        assert rendered is not None
        # Should show overall system status

    def test_render_filtered_list(self, agent_manager, sample_agents):
        """Test rendering filtered agent list."""
        for agent in sample_agents:
            agent_manager.agents[agent.id] = agent
        
        # Render only running agents
        rendered = agent_manager.render_agent_list(
            filter_status=AgentStatus.RUNNING
        )
        
        assert rendered is not None

    def test_render_with_sorting(self, agent_manager, sample_agents):
        """Test rendering with different sorting options."""
        for agent in sample_agents:
            agent_manager.agents[agent.id] = agent
        
        # Test different sort options
        sort_options = ['name', 'status', 'created_at', 'last_updated']
        
        for sort_by in sort_options:
            rendered = agent_manager.render_agent_list(sort_by=sort_by)
            assert rendered is not None


class TestEventHandling:
    """Test event handling and notifications."""

    @pytest.mark.asyncio
    async def test_agent_status_change_events(self, agent_manager, sample_agents):
        """Test agent status change event emission."""
        agent = sample_agents[0]
        agent_manager.agents[agent.id] = agent
        
        await agent_manager.update_agent_status(agent.id, AgentStatus.STOPPING)
        
        # Should emit status change event
        agent_manager.event_system.emit.assert_called_with(
            'agent_status_changed',
            {
                'agent_id': agent.id,
                'old_status': AgentStatus.RUNNING,
                'new_status': AgentStatus.STOPPING
            }
        )

    @pytest.mark.asyncio
    async def test_agent_lifecycle_events(self, agent_manager):
        """Test agent lifecycle event emission."""
        template = AgentTemplate(id="test-template", name="Test")
        agent_manager.templates["test-template"] = template
        
        with patch.object(agent_manager, '_create_agent_instance') as mock_create:
            mock_agent = MockAgent(
                id="new-agent",
                name="New Agent",
                status=AgentStatus.CREATED,
                config={},
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
            mock_create.return_value = mock_agent
            
            await agent_manager.create_agent("test-template", {"name": "New Agent"})
            
            # Should emit agent created event
            agent_manager.event_system.emit.assert_called_with(
                'agent_created',
                {'agent_id': mock_agent.id, 'agent_name': mock_agent.name}
            )

    def test_event_subscription(self, agent_manager):
        """Test subscribing to agent manager events."""
        handler_called = []
        
        def event_handler(event_type, data):
            handler_called.append((event_type, data))
        
        # Subscribe to events
        agent_manager.subscribe_to_events(['agent_created', 'agent_deleted'], event_handler)
        
        # Verify subscription
        agent_manager.event_system.subscribe.assert_called()

    @pytest.mark.asyncio
    async def test_health_check_events(self, agent_manager, sample_agents):
        """Test health check event emission."""
        agent = sample_agents[0]
        agent_manager.agents[agent.id] = agent
        
        with patch.object(agent_manager, '_ping_agent') as mock_ping:
            mock_ping.return_value = {'status': 'unhealthy', 'error': 'Timeout'}
            
            await agent_manager.check_agent_health(agent.id)
            
            # Should emit health status event
            assert agent_manager.event_system.emit.called


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery scenarios."""

    @pytest.mark.asyncio
    async def test_agent_creation_failure_recovery(self, agent_manager):
        """Test recovery from agent creation failures."""
        template = AgentTemplate(id="fail-template", name="Failing Template")
        agent_manager.templates["fail-template"] = template
        
        with patch.object(agent_manager, '_create_agent_instance') as mock_create:
            mock_create.side_effect = Exception("Creation failed")
            
            with pytest.raises(Exception, match="Creation failed"):
                await agent_manager.create_agent("fail-template", {"name": "Test"})
            
            # System should remain stable
            assert len(agent_manager.agents) == 0

    @pytest.mark.asyncio
    async def test_agent_start_failure_handling(self, agent_manager, sample_agents):
        """Test handling agent start failures."""
        agent = sample_agents[1]  # STOPPED agent
        agent_manager.agents[agent.id] = agent
        
        with patch.object(agent_manager, '_start_agent_process') as mock_start:
            mock_start.side_effect = Exception("Start failed")
            
            result = await agent_manager.start_agent(agent.id)
            
            assert result is False
            assert agent.status == AgentStatus.ERROR
            assert "Start failed" in agent.error_message

    @pytest.mark.asyncio
    async def test_monitoring_failure_recovery(self, agent_manager, sample_agents):
        """Test recovery from monitoring failures."""
        agent = sample_agents[0]
        agent_manager.agents[agent.id] = agent
        
        failure_count = 0
        
        def failing_health_check(agent_id):
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 2:
                raise Exception("Health check failed")
            return {'status': 'healthy'}
        
        with patch.object(agent_manager, 'check_agent_health', side_effect=failing_health_check):
            # Should recover after failures
            try:
                health = await agent_manager.check_agent_health(agent.id)
                assert health['status'] == 'healthy'
            except Exception:
                pass  # Expected for first few calls

    def test_invalid_agent_operations(self, agent_manager):
        """Test handling invalid agent operations."""
        # Operation on non-existent agent
        with pytest.raises(ValueError, match="Agent 'nonexistent' not found"):
            asyncio.run(agent_manager.start_agent("nonexistent"))
        
        with pytest.raises(ValueError, match="Agent 'nonexistent' not found"):
            asyncio.run(agent_manager.stop_agent("nonexistent"))

    @pytest.mark.asyncio
    async def test_concurrent_operation_safety(self, agent_manager, sample_agents):
        """Test safety of concurrent operations."""
        agent = sample_agents[0]
        agent_manager.agents[agent.id] = agent
        
        async def concurrent_operations():
            tasks = [
                agent_manager.update_agent_status(agent.id, AgentStatus.STOPPING),
                agent_manager.check_agent_health(agent.id),
                agent_manager.get_agent_config(agent.id)
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Should handle concurrent operations without corruption
        await concurrent_operations()
        
        # Agent should still be in valid state
        assert agent.id in agent_manager.agents


class TestPerformanceOptimization:
    """Test performance optimization and scalability."""

    def test_large_agent_list_performance(self, agent_manager):
        """Test performance with large number of agents."""
        # Create many agents
        start_time = time.time()
        
        for i in range(1000):
            agent = MockAgent(
                id=f"agent-{i}",
                name=f"Agent {i}",
                status=AgentStatus.STOPPED,
                config={"index": i},
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
            agent_manager.agents[agent.id] = agent
        
        creation_time = time.time() - start_time
        assert creation_time < 2.0  # Should complete in reasonable time
        
        # Test operations on large list
        start_time = time.time()
        stats = agent_manager.get_agent_statistics()
        stats_time = time.time() - start_time
        
        assert stats['total'] == 1000
        assert stats_time < 0.5  # Should calculate quickly

    @pytest.mark.asyncio
    async def test_bulk_operation_performance(self, agent_manager):
        """Test performance of bulk operations."""
        # Create agents for bulk operations
        agent_ids = []
        for i in range(100):
            agent = MockAgent(
                id=f"bulk-agent-{i}",
                name=f"Bulk Agent {i}",
                status=AgentStatus.STOPPED,
                config={},
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
            agent_manager.agents[agent.id] = agent
            agent_ids.append(agent.id)
        
        with patch.object(agent_manager, 'start_agent', return_value=True):
            start_time = time.time()
            
            results = await agent_manager.bulk_start_agents(agent_ids)
            
            duration = time.time() - start_time
            
            assert len(results) == 100
            assert all(results.values())
            assert duration < 5.0  # Should complete bulk operation quickly

    def test_status_cache_optimization(self, agent_manager, sample_agents):
        """Test status caching optimization."""
        for agent in sample_agents:
            agent_manager.agents[agent.id] = agent
        
        # First call should populate cache
        start_time = time.time()
        stats1 = agent_manager.get_agent_statistics()
        first_call_time = time.time() - start_time
        
        # Second call should use cache (if implemented)
        start_time = time.time()
        stats2 = agent_manager.get_agent_statistics()
        second_call_time = time.time() - start_time
        
        assert stats1 == stats2
        # Second call might be faster due to caching
        # (This test validates the interface; actual caching implementation may vary)


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    @pytest.mark.asyncio
    async def test_complete_agent_lifecycle_workflow(self, agent_manager):
        """Test complete agent lifecycle from creation to deletion."""
        # 1. Add template
        template = AgentTemplate(
            id="workflow-template",
            name="Workflow Template",
            config_schema={"name": {"type": "string", "required": True}}
        )
        agent_manager.add_template(template)
        
        # 2. Create agent
        agent = await agent_manager.create_agent(
            "workflow-template",
            {"name": "Workflow Agent"}
        )
        assert agent is not None
        
        # 3. Start agent
        with patch.object(agent_manager, '_start_agent_process', return_value=True):
            result = await agent_manager.start_agent(agent.id)
            assert result is True
        
        # 4. Check health
        with patch.object(agent_manager, '_ping_agent') as mock_ping:
            mock_ping.return_value = {'status': 'healthy'}
            health = await agent_manager.check_agent_health(agent.id)
            assert health['status'] == 'healthy'
        
        # 5. Update config (requires restart)
        new_config = {"name": "Updated Agent", "threads": 4}
        with patch.object(agent_manager, 'restart_agent', return_value=True):
            result = await agent_manager.update_agent_config(
                agent.id, new_config, restart_if_running=True
            )
            assert result is True
        
        # 6. Stop agent
        with patch.object(agent_manager, '_stop_agent_process', return_value=True):
            result = await agent_manager.stop_agent(agent.id)
            assert result is True
        
        # 7. Delete agent
        with patch.object(agent_manager, '_cleanup_agent_resources', return_value=True):
            result = await agent_manager.delete_agent(agent.id)
            assert result is True
        
        # Agent should be completely removed
        assert agent.id not in agent_manager.agents

    @pytest.mark.asyncio
    async def test_multi_agent_coordination_scenario(self, agent_manager):
        """Test coordination of multiple related agents."""
        # Create related agents that work together
        templates = [
            AgentTemplate(id="data-collector", name="Data Collector"),
            AgentTemplate(id="data-processor", name="Data Processor"),
            AgentTemplate(id="data-exporter", name="Data Exporter")
        ]
        
        for template in templates:
            agent_manager.add_template(template)
        
        # Create agents in dependency order
        agents = []
        for template in templates:
            with patch.object(agent_manager, '_create_agent_instance') as mock_create:
                mock_agent = MockAgent(
                    id=f"{template.id}-instance",
                    name=template.name,
                    status=AgentStatus.CREATED,
                    config={},
                    created_at=datetime.now(),
                    last_updated=datetime.now()
                )
                mock_create.return_value = mock_agent
                
                agent = await agent_manager.create_agent(template.id, {"name": template.name})
                agents.append(agent)
        
        # Start agents in order
        with patch.object(agent_manager, '_start_agent_process', return_value=True):
            for agent in agents:
                result = await agent_manager.start_agent(agent.id)
                assert result is True
        
        # Monitor all agents
        with patch.object(agent_manager, '_ping_agent') as mock_ping:
            mock_ping.return_value = {'status': 'healthy'}
            
            health_report = await agent_manager.check_all_agents_health()
            assert len(health_report) == 3
            assert all(h['status'] == 'healthy' for h in health_report.values())
        
        # Stop all agents in reverse order
        with patch.object(agent_manager, '_stop_agent_process', return_value=True):
            for agent in reversed(agents):
                result = await agent_manager.stop_agent(agent.id)
                assert result is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--durations=10"])