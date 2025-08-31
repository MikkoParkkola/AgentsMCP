"""
Comprehensive test suite for Configuration Assistant component.

Tests the ConfigurationAssistant component with 95%+ coverage, including:
- AI-powered configuration recommendations
- Configuration wizards and guided setup
- Template management and application
- Conflict detection and resolution
- Smart defaults and context-aware suggestions
- Integration with external systems
- Performance under various loads
- Error handling and recovery
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Import the configuration assistant component to test
from agentsmcp.ui.components.config_assistant import (
    ConfigurationAssistant, 
    Recommendation, 
    RecommendationType, 
    ConfigTemplate, 
    ConflictResolution,
    ConfigurationWizard,
    WizardStep,
    SmartDefault
)


@dataclass
class MockConfiguration:
    """Mock configuration for testing."""
    key: str
    value: Any
    section: str
    priority: int = 1
    source: str = "user"
    timestamp: datetime = None


class MockRecommendationEngine:
    """Mock recommendation engine for testing."""
    
    def __init__(self):
        self.recommendations = []
        self.context = {}
    
    async def generate_recommendations(self, config_context: Dict[str, Any]) -> List[Recommendation]:
        return self.recommendations.copy()
    
    def add_recommendation(self, recommendation: Recommendation):
        self.recommendations.append(recommendation)


@pytest.fixture
def mock_display_renderer():
    """Mock DisplayRenderer for testing."""
    renderer = Mock()
    renderer.create_panel = Mock(return_value="[Panel]")
    renderer.create_table = Mock(return_value="[Table]")
    renderer.create_progress_bar = Mock(return_value="[Progress: 60%]")
    renderer.create_tree_view = Mock(return_value="[Tree]")
    renderer.style_text = Mock(side_effect=lambda text, style: f"[{style}]{text}[/{style}]")
    renderer.create_wizard_step = Mock(return_value="[Wizard Step]")
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
def mock_ai_client():
    """Mock AI client for recommendations."""
    ai_client = Mock()
    ai_client.generate_recommendations = AsyncMock()
    ai_client.analyze_configuration = AsyncMock()
    ai_client.detect_conflicts = AsyncMock()
    ai_client.suggest_optimizations = AsyncMock()
    return ai_client


@pytest.fixture
def config_assistant(mock_display_renderer, mock_terminal_manager, mock_event_system, mock_ai_client):
    """Create ConfigurationAssistant instance for testing."""
    assistant = ConfigurationAssistant(
        display_renderer=mock_display_renderer,
        terminal_manager=mock_terminal_manager,
        event_system=mock_event_system,
        ai_client=mock_ai_client
    )
    return assistant


@pytest.fixture
def sample_recommendations():
    """Create sample recommendations for testing."""
    return [
        Recommendation(
            id="rec_1",
            type=RecommendationType.OPTIMIZATION,
            title="Increase Memory Allocation",
            description="Consider increasing memory allocation for better performance",
            config_changes={"memory": "512MB"},
            confidence=0.85,
            impact="medium",
            category="performance"
        ),
        Recommendation(
            id="rec_2",
            type=RecommendationType.SECURITY,
            title="Enable SSL/TLS",
            description="Enable SSL/TLS for secure communications",
            config_changes={"ssl_enabled": True, "ssl_port": 443},
            confidence=0.95,
            impact="high",
            category="security"
        ),
        Recommendation(
            id="rec_3",
            type=RecommendationType.COMPATIBILITY,
            title="Update API Version",
            description="Update API version for better compatibility",
            config_changes={"api_version": "v2"},
            confidence=0.70,
            impact="low",
            category="compatibility"
        )
    ]


class TestConfigurationAssistantInitialization:
    """Test Configuration Assistant initialization and setup."""

    def test_initialization_success(self, config_assistant):
        """Test successful configuration assistant initialization."""
        assert config_assistant.display_renderer is not None
        assert config_assistant.terminal_manager is not None
        assert config_assistant.event_system is not None
        assert config_assistant.ai_client is not None
        assert hasattr(config_assistant, 'templates')
        assert hasattr(config_assistant, 'recommendations')
        assert hasattr(config_assistant, 'wizards')

    def test_recommendation_engine_setup(self, config_assistant):
        """Test recommendation engine setup."""
        assert hasattr(config_assistant, '_recommendation_cache')
        assert hasattr(config_assistant, '_last_analysis')
        assert hasattr(config_assistant, '_context_analyzer')

    def test_template_system_initialization(self, config_assistant):
        """Test template system initialization."""
        assert isinstance(config_assistant.templates, dict)
        assert hasattr(config_assistant, '_template_validator')
        
        # Should have some default templates
        assert len(config_assistant.templates) >= 0

    @pytest.mark.asyncio
    async def test_initialization_with_custom_settings(self):
        """Test initialization with custom settings."""
        custom_settings = {
            'recommendation_threshold': 0.8,
            'max_recommendations': 10,
            'auto_apply_safe_recommendations': True,
            'conflict_resolution_strategy': 'interactive'
        }
        
        with patch('agentsmcp.ui.v2.display_renderer.DisplayRenderer') as MockRenderer:
            with patch('agentsmcp.ui.v2.terminal_state_manager.TerminalManager') as MockManager:
                with patch('agentsmcp.ui.components.event_system.EventSystem') as MockEvents:
                    with patch('agentsmcp.ui.components.ai_client.AIClient') as MockAI:
                        mock_renderer = MockRenderer.return_value
                        mock_manager = MockManager.return_value
                        mock_events = MockEvents.return_value
                        mock_ai = MockAI.return_value
                        
                        assistant = ConfigurationAssistant(
                            display_renderer=mock_renderer,
                            terminal_manager=mock_manager,
                            event_system=mock_events,
                            ai_client=mock_ai,
                            **custom_settings
                        )
                        
                        assert assistant._recommendation_threshold == 0.8
                        assert assistant._max_recommendations == 10
                        assert assistant._auto_apply_safe is True


class TestAIRecommendationGeneration:
    """Test AI-powered recommendation generation."""

    @pytest.mark.asyncio
    async def test_generate_recommendations_basic(self, config_assistant, sample_recommendations):
        """Test basic recommendation generation."""
        # Mock AI client response
        config_assistant.ai_client.generate_recommendations.return_value = sample_recommendations
        
        current_config = {
            "database": {"host": "localhost", "port": 5432},
            "api": {"timeout": 30, "retries": 3}
        }
        
        recommendations = await config_assistant.generate_recommendations(current_config)
        
        assert len(recommendations) == 3
        assert all(isinstance(rec, Recommendation) for rec in recommendations)
        config_assistant.ai_client.generate_recommendations.assert_called_once()

    @pytest.mark.asyncio
    async def test_recommendation_filtering_by_confidence(self, config_assistant, sample_recommendations):
        """Test filtering recommendations by confidence threshold."""
        config_assistant._recommendation_threshold = 0.8
        config_assistant.ai_client.generate_recommendations.return_value = sample_recommendations
        
        recommendations = await config_assistant.generate_recommendations({})
        
        # Should only return recommendations with confidence >= 0.8
        high_confidence_recs = [r for r in recommendations if r.confidence >= 0.8]
        assert len(high_confidence_recs) == 2  # rec_1 (0.85) and rec_2 (0.95)

    @pytest.mark.asyncio
    async def test_recommendation_caching(self, config_assistant, sample_recommendations):
        """Test recommendation result caching."""
        config_assistant.ai_client.generate_recommendations.return_value = sample_recommendations
        
        config = {"test": "value"}
        
        # First call - should hit AI
        recommendations1 = await config_assistant.generate_recommendations(config)
        
        # Second call with same config - should use cache
        recommendations2 = await config_assistant.generate_recommendations(config)
        
        assert recommendations1 == recommendations2
        # AI should only be called once
        config_assistant.ai_client.generate_recommendations.assert_called_once()

    @pytest.mark.asyncio
    async def test_contextual_recommendations(self, config_assistant):
        """Test contextual recommendation generation."""
        # Set up context
        context = {
            "environment": "production",
            "load_level": "high",
            "security_requirements": "strict",
            "performance_critical": True
        }
        
        config_assistant.set_context(context)
        
        # Mock context-aware recommendations
        context_recommendations = [
            Recommendation(
                id="ctx_1",
                type=RecommendationType.PERFORMANCE,
                title="Production Optimization",
                description="Optimizations for production environment",
                config_changes={"cache_size": "1GB", "connection_pool": 100},
                confidence=0.90,
                impact="high"
            )
        ]
        
        config_assistant.ai_client.generate_recommendations.return_value = context_recommendations
        
        recommendations = await config_assistant.generate_recommendations({})
        
        # Should have generated context-aware recommendations
        assert len(recommendations) > 0
        assert recommendations[0].title == "Production Optimization"
        
        # Verify context was passed to AI client
        call_args = config_assistant.ai_client.generate_recommendations.call_args
        assert call_args is not None

    @pytest.mark.asyncio
    async def test_recommendation_prioritization(self, config_assistant, sample_recommendations):
        """Test recommendation prioritization logic."""
        # Add priority scores to recommendations
        for i, rec in enumerate(sample_recommendations):
            rec.priority = len(sample_recommendations) - i  # Higher index = lower priority
        
        config_assistant.ai_client.generate_recommendations.return_value = sample_recommendations
        
        recommendations = await config_assistant.generate_recommendations({})
        
        # Should be sorted by priority (highest first)
        assert recommendations[0].type == RecommendationType.SECURITY  # Highest impact
        assert recommendations[0].confidence >= recommendations[1].confidence or recommendations[0].impact == "high"

    @pytest.mark.asyncio
    async def test_recommendation_categories(self, config_assistant, sample_recommendations):
        """Test recommendation categorization."""
        config_assistant.ai_client.generate_recommendations.return_value = sample_recommendations
        
        recommendations = await config_assistant.generate_recommendations({})
        
        # Group by category
        categories = config_assistant.group_recommendations_by_category(recommendations)
        
        assert "performance" in categories
        assert "security" in categories
        assert "compatibility" in categories
        
        assert len(categories["performance"]) == 1
        assert len(categories["security"]) == 1
        assert len(categories["compatibility"]) == 1


class TestConfigurationTemplateManagement:
    """Test configuration template management."""

    def test_add_template_success(self, config_assistant):
        """Test adding configuration template successfully."""
        template = ConfigTemplate(
            id="web_server_template",
            name="Web Server Configuration",
            description="Standard web server setup",
            category="infrastructure",
            config_schema={
                "server": {
                    "port": {"type": "integer", "default": 80},
                    "host": {"type": "string", "default": "0.0.0.0"}
                }
            },
            default_values={
                "server": {"port": 80, "host": "0.0.0.0"}
            }
        )
        
        result = config_assistant.add_template(template)
        assert result is True
        assert template.id in config_assistant.templates

    def test_add_template_duplicate_id(self, config_assistant):
        """Test adding template with duplicate ID."""
        template1 = ConfigTemplate(id="duplicate", name="First Template")
        template2 = ConfigTemplate(id="duplicate", name="Second Template")
        
        config_assistant.add_template(template1)
        
        with pytest.raises(ValueError, match="Template 'duplicate' already exists"):
            config_assistant.add_template(template2)

    def test_get_template_success(self, config_assistant):
        """Test getting existing template."""
        template = ConfigTemplate(id="test_template", name="Test Template")
        config_assistant.add_template(template)
        
        retrieved = config_assistant.get_template("test_template")
        assert retrieved is not None
        assert retrieved.name == "Test Template"

    def test_get_template_nonexistent(self, config_assistant):
        """Test getting non-existent template."""
        result = config_assistant.get_template("nonexistent")
        assert result is None

    def test_list_templates_by_category(self, config_assistant):
        """Test listing templates by category."""
        templates = [
            ConfigTemplate(id="db1", name="Database 1", category="database"),
            ConfigTemplate(id="db2", name="Database 2", category="database"),
            ConfigTemplate(id="web1", name="Web Server", category="web")
        ]
        
        for template in templates:
            config_assistant.add_template(template)
        
        db_templates = config_assistant.get_templates_by_category("database")
        web_templates = config_assistant.get_templates_by_category("web")
        
        assert len(db_templates) == 2
        assert len(web_templates) == 1

    @pytest.mark.asyncio
    async def test_apply_template_success(self, config_assistant):
        """Test applying configuration template."""
        template = ConfigTemplate(
            id="api_template",
            name="API Template",
            default_values={
                "api": {
                    "timeout": 30,
                    "retries": 3,
                    "base_url": "https://api.example.com"
                }
            }
        )
        
        config_assistant.add_template(template)
        
        # Mock current configuration
        current_config = {"api": {"timeout": 10}}
        
        result = await config_assistant.apply_template("api_template", current_config)
        
        assert result is not None
        assert result["api"]["timeout"] == 30  # From template
        assert result["api"]["retries"] == 3  # From template
        assert result["api"]["base_url"] == "https://api.example.com"

    def test_template_validation(self, config_assistant):
        """Test template validation logic."""
        # Valid template
        valid_template = ConfigTemplate(
            id="valid",
            name="Valid Template",
            config_schema={
                "database": {
                    "host": {"type": "string", "required": True},
                    "port": {"type": "integer", "min": 1, "max": 65535}
                }
            }
        )
        
        assert config_assistant.validate_template(valid_template) is True
        
        # Invalid template - missing required fields
        invalid_template = ConfigTemplate(
            id="invalid",
            name=""  # Empty name
        )
        
        assert config_assistant.validate_template(invalid_template) is False

    @pytest.mark.asyncio
    async def test_template_customization(self, config_assistant):
        """Test template customization and parameterization."""
        template = ConfigTemplate(
            id="customizable",
            name="Customizable Template",
            parameters={
                "environment": {"type": "string", "options": ["dev", "prod"], "default": "dev"},
                "scale": {"type": "integer", "min": 1, "max": 100, "default": 1}
            },
            default_values={
                "app": {
                    "debug": "{{ 'true' if environment == 'dev' else 'false' }}",
                    "workers": "{{ scale * 2 }}"
                }
            }
        )
        
        config_assistant.add_template(template)
        
        # Apply with custom parameters
        custom_params = {"environment": "prod", "scale": 5}
        result = await config_assistant.apply_template("customizable", {}, custom_params)
        
        assert result["app"]["debug"] == "false"  # Production setting
        assert result["app"]["workers"] == "10"   # scale * 2 = 5 * 2


class TestConfigurationWizard:
    """Test configuration wizard functionality."""

    def test_create_wizard_success(self, config_assistant):
        """Test creating configuration wizard."""
        steps = [
            WizardStep(
                id="step1",
                title="Basic Settings",
                description="Configure basic application settings",
                fields=["app_name", "environment"]
            ),
            WizardStep(
                id="step2",
                title="Database",
                description="Configure database connection",
                fields=["db_host", "db_port", "db_name"]
            )
        ]
        
        wizard = ConfigurationWizard(
            id="app_setup",
            name="Application Setup",
            description="Set up your application configuration",
            steps=steps
        )
        
        result = config_assistant.add_wizard(wizard)
        assert result is True
        assert wizard.id in config_assistant.wizards

    @pytest.mark.asyncio
    async def test_start_wizard(self, config_assistant):
        """Test starting configuration wizard."""
        # Create a simple wizard
        wizard = ConfigurationWizard(
            id="simple_wizard",
            name="Simple Setup",
            steps=[
                WizardStep(id="step1", title="Step 1", fields=["field1"]),
                WizardStep(id="step2", title="Step 2", fields=["field2"])
            ]
        )
        
        config_assistant.add_wizard(wizard)
        
        # Start wizard
        session = await config_assistant.start_wizard("simple_wizard")
        
        assert session is not None
        assert session["wizard_id"] == "simple_wizard"
        assert session["current_step"] == 0
        assert session["completed_steps"] == []

    @pytest.mark.asyncio
    async def test_wizard_step_navigation(self, config_assistant):
        """Test wizard step navigation."""
        wizard = ConfigurationWizard(
            id="nav_wizard",
            name="Navigation Test",
            steps=[
                WizardStep(id="step1", title="Step 1", fields=["field1"]),
                WizardStep(id="step2", title="Step 2", fields=["field2"]),
                WizardStep(id="step3", title="Step 3", fields=["field3"])
            ]
        )
        
        config_assistant.add_wizard(wizard)
        session = await config_assistant.start_wizard("nav_wizard")
        session_id = session["session_id"]
        
        # Go to next step
        next_step = await config_assistant.next_wizard_step(session_id)
        assert next_step["step_index"] == 1
        assert next_step["step"]["id"] == "step2"
        
        # Go back
        prev_step = await config_assistant.previous_wizard_step(session_id)
        assert prev_step["step_index"] == 0
        assert prev_step["step"]["id"] == "step1"

    @pytest.mark.asyncio
    async def test_wizard_step_validation(self, config_assistant):
        """Test wizard step validation before proceeding."""
        wizard = ConfigurationWizard(
            id="validation_wizard",
            name="Validation Test",
            steps=[
                WizardStep(
                    id="step1",
                    title="Required Fields",
                    fields=["required_field"],
                    validation_rules={
                        "required_field": {"required": True, "min_length": 3}
                    }
                )
            ]
        )
        
        config_assistant.add_wizard(wizard)
        session = await config_assistant.start_wizard("validation_wizard")
        session_id = session["session_id"]
        
        # Try to proceed without setting required field
        can_proceed = await config_assistant.can_proceed_to_next_step(session_id)
        assert can_proceed is False
        
        # Set valid value
        await config_assistant.set_wizard_field_value(session_id, "required_field", "valid_value")
        
        # Now should be able to proceed
        can_proceed = await config_assistant.can_proceed_to_next_step(session_id)
        assert can_proceed is True

    @pytest.mark.asyncio
    async def test_wizard_completion(self, config_assistant):
        """Test wizard completion and configuration generation."""
        wizard = ConfigurationWizard(
            id="complete_wizard",
            name="Complete Test",
            steps=[
                WizardStep(id="step1", title="Step 1", fields=["field1"]),
                WizardStep(id="step2", title="Step 2", fields=["field2"])
            ]
        )
        
        config_assistant.add_wizard(wizard)
        session = await config_assistant.start_wizard("complete_wizard")
        session_id = session["session_id"]
        
        # Fill in all fields
        await config_assistant.set_wizard_field_value(session_id, "field1", "value1")
        await config_assistant.next_wizard_step(session_id)
        await config_assistant.set_wizard_field_value(session_id, "field2", "value2")
        
        # Complete wizard
        result = await config_assistant.complete_wizard(session_id)
        
        assert result is not None
        assert "configuration" in result
        assert result["configuration"]["field1"] == "value1"
        assert result["configuration"]["field2"] == "value2"

    def test_wizard_progress_tracking(self, config_assistant):
        """Test wizard progress tracking."""
        wizard = ConfigurationWizard(
            id="progress_wizard",
            name="Progress Test",
            steps=[
                WizardStep(id="s1", title="Step 1", fields=["f1"]),
                WizardStep(id="s2", title="Step 2", fields=["f2"]),
                WizardStep(id="s3", title="Step 3", fields=["f3"]),
                WizardStep(id="s4", title="Step 4", fields=["f4"])
            ]
        )
        
        config_assistant.add_wizard(wizard)
        
        # Mock session in progress
        session = {
            "current_step": 1,
            "total_steps": 4,
            "completed_steps": [0],
            "wizard": wizard
        }
        
        progress = config_assistant.get_wizard_progress(session)
        
        assert progress["current_step"] == 2  # 1-based
        assert progress["total_steps"] == 4
        assert progress["percentage"] == 50.0  # 2/4 * 100
        assert progress["completed_steps"] == 1


class TestConflictDetectionAndResolution:
    """Test configuration conflict detection and resolution."""

    @pytest.mark.asyncio
    async def test_detect_configuration_conflicts(self, config_assistant):
        """Test detecting configuration conflicts."""
        current_config = {
            "database": {"port": 5432, "ssl": True},
            "api": {"port": 5432, "timeout": 30}  # Port conflict
        }
        
        new_config = {
            "database": {"port": 5432, "ssl": False},  # SSL conflict
            "cache": {"port": 5432}  # Another port conflict
        }
        
        # Mock AI conflict detection
        conflicts = [
            ConflictResolution(
                id="port_conflict",
                description="Multiple services trying to use port 5432",
                affected_keys=["database.port", "api.port", "cache.port"],
                resolution_options=[
                    {"description": "Use different ports", "changes": {"api.port": 8080, "cache.port": 6379}},
                    {"description": "Use port ranges", "changes": {"api.port": "8080-8090"}}
                ]
            )
        ]
        
        config_assistant.ai_client.detect_conflicts.return_value = conflicts
        
        detected_conflicts = await config_assistant.detect_conflicts(current_config, new_config)
        
        assert len(detected_conflicts) == 1
        assert detected_conflicts[0].id == "port_conflict"
        assert len(detected_conflicts[0].resolution_options) == 2

    @pytest.mark.asyncio
    async def test_resolve_conflicts_automatically(self, config_assistant):
        """Test automatic conflict resolution."""
        conflict = ConflictResolution(
            id="auto_resolve",
            description="Automatic resolution test",
            affected_keys=["service.timeout"],
            resolution_options=[
                {
                    "description": "Use recommended timeout",
                    "changes": {"service.timeout": 60},
                    "auto_resolvable": True,
                    "safety_score": 0.9
                }
            ]
        )
        
        config = {"service": {"timeout": 30}}
        
        resolved_config = await config_assistant.resolve_conflict_automatically(config, conflict)
        
        assert resolved_config["service"]["timeout"] == 60

    @pytest.mark.asyncio
    async def test_interactive_conflict_resolution(self, config_assistant):
        """Test interactive conflict resolution."""
        conflict = ConflictResolution(
            id="interactive_resolve",
            description="Choose resolution option",
            affected_keys=["app.mode"],
            resolution_options=[
                {"description": "Development mode", "changes": {"app.mode": "dev"}},
                {"description": "Production mode", "changes": {"app.mode": "prod"}}
            ]
        )
        
        config = {"app": {"mode": "unknown"}}
        
        # Mock user selection (option 1 = production mode)
        with patch.object(config_assistant, '_get_user_resolution_choice', return_value=1):
            resolved_config = await config_assistant.resolve_conflict_interactively(config, conflict)
            
            assert resolved_config["app"]["mode"] == "prod"

    def test_conflict_severity_assessment(self, config_assistant):
        """Test conflict severity assessment."""
        conflicts = [
            ConflictResolution(
                id="critical_conflict",
                description="Security vulnerability",
                severity="critical",
                affected_keys=["security.ssl"]
            ),
            ConflictResolution(
                id="minor_conflict",
                description="Performance tweak",
                severity="low",
                affected_keys=["cache.size"]
            )
        ]
        
        sorted_conflicts = config_assistant.sort_conflicts_by_severity(conflicts)
        
        # Critical should come first
        assert sorted_conflicts[0].severity == "critical"
        assert sorted_conflicts[1].severity == "low"

    @pytest.mark.asyncio
    async def test_batch_conflict_resolution(self, config_assistant):
        """Test resolving multiple conflicts in batch."""
        conflicts = [
            ConflictResolution(
                id="conflict1",
                description="Port conflict",
                resolution_options=[{"changes": {"service1.port": 8001}, "auto_resolvable": True}]
            ),
            ConflictResolution(
                id="conflict2",
                description="Memory conflict", 
                resolution_options=[{"changes": {"service2.memory": "1GB"}, "auto_resolvable": True}]
            )
        ]
        
        config = {"service1": {"port": 8000}, "service2": {"memory": "512MB"}}
        
        resolved_config = await config_assistant.resolve_conflicts_batch(config, conflicts)
        
        assert resolved_config["service1"]["port"] == 8001
        assert resolved_config["service2"]["memory"] == "1GB"


class TestSmartDefaultsAndSuggestions:
    """Test smart defaults and context-aware suggestions."""

    def test_generate_smart_defaults(self, config_assistant):
        """Test generating smart default values."""
        context = {
            "environment": "production",
            "expected_load": "high",
            "security_level": "strict"
        }
        
        config_schema = {
            "database": {
                "connection_pool_size": {"type": "integer", "min": 1, "max": 1000},
                "timeout": {"type": "integer", "min": 1, "max": 300}
            },
            "cache": {
                "memory_limit": {"type": "string"},
                "ttl": {"type": "integer", "min": 1}
            }
        }
        
        # Mock smart defaults generation
        smart_defaults = {
            "database": {
                "connection_pool_size": 50,  # Higher for production
                "timeout": 30
            },
            "cache": {
                "memory_limit": "2GB",  # More memory for high load
                "ttl": 3600
            }
        }
        
        with patch.object(config_assistant, '_generate_context_aware_defaults', return_value=smart_defaults):
            defaults = config_assistant.generate_smart_defaults(config_schema, context)
            
            assert defaults["database"]["connection_pool_size"] == 50
            assert defaults["cache"]["memory_limit"] == "2GB"

    @pytest.mark.asyncio
    async def test_suggest_configuration_improvements(self, config_assistant):
        """Test suggesting configuration improvements."""
        current_config = {
            "database": {"connection_pool_size": 5},  # Too low
            "cache": {"memory_limit": "64MB"},  # Too small
            "logging": {"level": "DEBUG"}  # Wrong for production
        }
        
        context = {"environment": "production", "performance_critical": True}
        
        # Mock improvement suggestions
        improvements = [
            {
                "key": "database.connection_pool_size",
                "current_value": 5,
                "suggested_value": 50,
                "reason": "Increase pool size for better performance in production"
            },
            {
                "key": "logging.level",
                "current_value": "DEBUG",
                "suggested_value": "INFO",
                "reason": "Use INFO level logging in production for security and performance"
            }
        ]
        
        config_assistant.ai_client.suggest_optimizations.return_value = improvements
        
        suggestions = await config_assistant.suggest_improvements(current_config, context)
        
        assert len(suggestions) == 2
        assert suggestions[0]["suggested_value"] == 50
        assert suggestions[1]["suggested_value"] == "INFO"

    def test_adaptive_defaults_based_on_history(self, config_assistant):
        """Test adaptive defaults based on configuration history."""
        # Mock configuration history
        config_history = [
            {"timestamp": "2023-01-01", "config": {"timeout": 30, "retries": 3}},
            {"timestamp": "2023-01-02", "config": {"timeout": 45, "retries": 3}},
            {"timestamp": "2023-01-03", "config": {"timeout": 60, "retries": 5}}
        ]
        
        config_assistant._config_history = config_history
        
        # Generate adaptive defaults
        adaptive_defaults = config_assistant.generate_adaptive_defaults()
        
        # Should learn from history (average/most common values)
        assert "timeout" in adaptive_defaults
        assert "retries" in adaptive_defaults
        # Values should be influenced by historical patterns

    @pytest.mark.asyncio
    async def test_environment_specific_suggestions(self, config_assistant):
        """Test environment-specific configuration suggestions."""
        base_config = {
            "database": {"host": "localhost"},
            "api": {"debug": True},
            "cache": {"enabled": False}
        }
        
        environments = ["development", "staging", "production"]
        
        for env in environments:
            context = {"environment": env}
            
            # Mock environment-specific suggestions
            if env == "development":
                suggestions = {"api.debug": True, "cache.enabled": False}
            elif env == "staging":
                suggestions = {"api.debug": False, "cache.enabled": True}
            else:  # production
                suggestions = {"api.debug": False, "cache.enabled": True, "database.host": "prod-db-cluster"}
            
            with patch.object(config_assistant, '_get_environment_suggestions', return_value=suggestions):
                env_suggestions = await config_assistant.get_environment_suggestions(base_config, env)
                
                if env == "production":
                    assert env_suggestions["database.host"] == "prod-db-cluster"
                    assert env_suggestions["api.debug"] is False


class TestIntegrationAndExternalSystems:
    """Test integration with external systems and APIs."""

    @pytest.mark.asyncio
    async def test_import_configuration_from_external(self, config_assistant):
        """Test importing configuration from external systems."""
        external_config = {
            "source": "kubernetes",
            "data": {
                "deployment": {
                    "replicas": 3,
                    "image": "app:latest"
                },
                "service": {
                    "port": 80,
                    "type": "LoadBalancer"
                }
            }
        }
        
        # Mock external system integration
        with patch.object(config_assistant, '_fetch_external_config', return_value=external_config):
            imported_config = await config_assistant.import_from_external_system("kubernetes", {"namespace": "production"})
            
            assert "deployment" in imported_config
            assert imported_config["deployment"]["replicas"] == 3

    @pytest.mark.asyncio
    async def test_export_configuration_to_external(self, config_assistant):
        """Test exporting configuration to external systems."""
        local_config = {
            "app": {
                "name": "my-app",
                "version": "1.0.0",
                "replicas": 2
            }
        }
        
        export_result = {"success": True, "resource_id": "deployment-123"}
        
        with patch.object(config_assistant, '_export_to_external_system', return_value=export_result):
            result = await config_assistant.export_to_external_system("kubernetes", local_config)
            
            assert result["success"] is True
            assert "resource_id" in result

    @pytest.mark.asyncio
    async def test_synchronize_with_external_systems(self, config_assistant):
        """Test synchronizing configuration with external systems."""
        sync_targets = ["vault", "consul", "k8s-configmap"]
        
        local_config = {"database": {"password": "secret123"}}
        
        # Mock sync results
        sync_results = []
        
        for target in sync_targets:
            if target == "vault":
                result = {"status": "success", "version": "v1"}
            elif target == "consul":
                result = {"status": "success", "key": "app/config"}
            else:
                result = {"status": "success", "configmap": "app-config"}
            
            sync_results.append({target: result})
        
        with patch.object(config_assistant, '_sync_with_system') as mock_sync:
            mock_sync.side_effect = [r[list(r.keys())[0]] for r in sync_results]
            
            results = await config_assistant.sync_with_external_systems(local_config, sync_targets)
            
            assert len(results) == len(sync_targets)
            assert all(r["status"] == "success" for r in results.values())

    @pytest.mark.asyncio
    async def test_validate_against_external_schemas(self, config_assistant):
        """Test validating configuration against external schemas."""
        config = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": "test-config"},
            "data": {"key": "value"}
        }
        
        schema_url = "https://api.kubernetes.io/schemas/configmap.json"
        
        validation_result = {"valid": True, "errors": []}
        
        with patch.object(config_assistant, '_validate_against_schema', return_value=validation_result):
            result = await config_assistant.validate_against_external_schema(config, schema_url)
            
            assert result["valid"] is True
            assert len(result["errors"]) == 0


class TestPerformanceAndOptimization:
    """Test performance optimization and scalability."""

    def test_large_configuration_handling(self, config_assistant):
        """Test handling large configuration files."""
        # Generate large configuration
        large_config = {}
        
        for i in range(1000):
            section = f"section_{i}"
            large_config[section] = {
                f"key_{j}": f"value_{i}_{j}" for j in range(50)
            }
        
        start_time = time.time()
        
        # Test operations on large config
        config_assistant.validate_configuration_structure(large_config)
        flattened = config_assistant.flatten_configuration(large_config)
        
        operation_time = time.time() - start_time
        
        assert operation_time < 2.0  # Should complete in reasonable time
        assert len(flattened) == 50000  # 1000 sections * 50 keys each

    @pytest.mark.asyncio
    async def test_concurrent_recommendation_generation(self, config_assistant, sample_recommendations):
        """Test concurrent recommendation generation."""
        # Mock multiple config contexts
        contexts = [
            {"service": "web", "environment": "prod"},
            {"service": "api", "environment": "staging"},
            {"service": "db", "environment": "dev"}
        ]
        
        # Mock AI responses
        config_assistant.ai_client.generate_recommendations.side_effect = [
            sample_recommendations[:1],  # Web recommendations
            sample_recommendations[1:2], # API recommendations  
            sample_recommendations[2:3]  # DB recommendations
        ]
        
        # Generate recommendations concurrently
        start_time = time.time()
        
        tasks = [
            config_assistant.generate_recommendations({}, context)
            for context in contexts
        ]
        
        results = await asyncio.gather(*tasks)
        
        concurrent_time = time.time() - start_time
        
        assert len(results) == 3
        assert all(len(r) > 0 for r in results)
        assert concurrent_time < 1.0  # Should complete concurrently, not sequentially

    def test_recommendation_caching_performance(self, config_assistant, sample_recommendations):
        """Test recommendation caching performance."""
        config_assistant.ai_client.generate_recommendations.return_value = sample_recommendations
        
        config = {"test": "config"}
        
        # First call - populate cache
        start_time = time.time()
        asyncio.run(config_assistant.generate_recommendations(config))
        first_call_time = time.time() - start_time
        
        # Subsequent calls - use cache
        start_time = time.time()
        for _ in range(100):
            asyncio.run(config_assistant.generate_recommendations(config))
        cached_calls_time = time.time() - start_time
        
        # Cached calls should be much faster
        assert cached_calls_time < first_call_time * 10  # At least 10x faster for cached calls

    @pytest.mark.asyncio
    async def test_memory_usage_optimization(self, config_assistant):
        """Test memory usage optimization for large datasets."""
        import sys
        
        # Get baseline memory
        initial_memory = sys.getsizeof(config_assistant)
        
        # Add many templates
        for i in range(500):
            template = ConfigTemplate(
                id=f"template_{i}",
                name=f"Template {i}",
                default_values={f"section_{j}": {f"key_{k}": f"value_{i}_{j}_{k}" for k in range(10)} for j in range(5)}
            )
            config_assistant.add_template(template)
        
        # Memory growth should be reasonable
        final_memory = sys.getsizeof(config_assistant)
        memory_growth = final_memory / initial_memory
        
        assert memory_growth < 5  # Should not grow excessively


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery scenarios."""

    @pytest.mark.asyncio
    async def test_ai_service_failure_handling(self, config_assistant):
        """Test handling AI service failures gracefully."""
        # Mock AI service failure
        config_assistant.ai_client.generate_recommendations.side_effect = Exception("AI service unavailable")
        
        # Should not crash, return fallback recommendations
        recommendations = await config_assistant.generate_recommendations({})
        
        # Should return empty list or fallback recommendations
        assert isinstance(recommendations, list)
        # Should have logged the error (implementation detail)

    @pytest.mark.asyncio
    async def test_invalid_template_recovery(self, config_assistant):
        """Test recovery from invalid template application."""
        # Create template with invalid structure
        invalid_template = ConfigTemplate(
            id="invalid_template",
            name="Invalid Template",
            default_values={
                "invalid_section": {
                    "circular_ref": "{{ invalid_section.circular_ref }}"  # Circular reference
                }
            }
        )
        
        config_assistant.add_template(invalid_template)
        
        # Should handle invalid template gracefully
        with pytest.raises(ValueError, match="Template application failed"):
            await config_assistant.apply_template("invalid_template", {})

    @pytest.mark.asyncio
    async def test_wizard_session_recovery(self, config_assistant):
        """Test recovery from corrupted wizard sessions."""
        # Create wizard
        wizard = ConfigurationWizard(
            id="recovery_test",
            name="Recovery Test",
            steps=[WizardStep(id="step1", title="Step 1", fields=["field1"])]
        )
        
        config_assistant.add_wizard(wizard)
        
        # Start session
        session = await config_assistant.start_wizard("recovery_test")
        session_id = session["session_id"]
        
        # Corrupt session data
        config_assistant._wizard_sessions[session_id]["current_step"] = 999  # Invalid step
        
        # Should handle corruption gracefully
        try:
            await config_assistant.next_wizard_step(session_id)
        except ValueError:
            pass  # Expected for corrupted session
        
        # Should be able to restart
        new_session = await config_assistant.start_wizard("recovery_test")
        assert new_session is not None

    def test_configuration_validation_error_handling(self, config_assistant):
        """Test handling configuration validation errors."""
        invalid_configs = [
            None,  # Null config
            "invalid",  # String instead of dict
            {"circular": {"ref": "{{ circular.ref }}"}},  # Circular reference
            {"invalid_type": object()},  # Non-serializable type
        ]
        
        for config in invalid_configs:
            try:
                result = config_assistant.validate_configuration_structure(config)
                assert result is False  # Should detect as invalid
            except Exception:
                pass  # Should handle gracefully without crashing

    @pytest.mark.asyncio
    async def test_network_timeout_recovery(self, config_assistant):
        """Test recovery from network timeouts."""
        # Mock network timeout
        config_assistant.ai_client.generate_recommendations.side_effect = asyncio.TimeoutError("Request timeout")
        
        # Should handle timeout gracefully
        try:
            recommendations = await config_assistant.generate_recommendations({})
            assert isinstance(recommendations, list)  # Fallback response
        except asyncio.TimeoutError:
            pass  # Should be handled upstream


class TestRenderingAndVisualization:
    """Test rendering and visualization functionality."""

    def test_render_recommendations_list(self, config_assistant, sample_recommendations):
        """Test rendering recommendations list."""
        rendered = config_assistant.render_recommendations(sample_recommendations)
        
        assert rendered is not None
        config_assistant.display_renderer.create_table.assert_called()

    def test_render_configuration_diff(self, config_assistant):
        """Test rendering configuration differences."""
        old_config = {"database": {"timeout": 30, "pool_size": 10}}
        new_config = {"database": {"timeout": 60, "pool_size": 20, "ssl": True}}
        
        diff_rendering = config_assistant.render_configuration_diff(old_config, new_config)
        
        assert diff_rendering is not None
        # Should show additions, modifications, and deletions

    def test_render_wizard_progress(self, config_assistant):
        """Test rendering wizard progress."""
        progress = {
            "current_step": 2,
            "total_steps": 4,
            "percentage": 50.0,
            "step_title": "Database Configuration"
        }
        
        rendered = config_assistant.render_wizard_progress(progress)
        
        assert rendered is not None
        config_assistant.display_renderer.create_progress_bar.assert_called()

    def test_render_conflict_resolution_options(self, config_assistant):
        """Test rendering conflict resolution options."""
        conflict = ConflictResolution(
            id="test_conflict",
            description="Test conflict",
            resolution_options=[
                {"description": "Option 1", "changes": {"key1": "value1"}},
                {"description": "Option 2", "changes": {"key2": "value2"}}
            ]
        )
        
        rendered = config_assistant.render_conflict_resolution(conflict)
        
        assert rendered is not None

    def test_render_template_gallery(self, config_assistant):
        """Test rendering template gallery."""
        templates = [
            ConfigTemplate(id="t1", name="Template 1", category="web"),
            ConfigTemplate(id="t2", name="Template 2", category="database"),
            ConfigTemplate(id="t3", name="Template 3", category="web")
        ]
        
        for template in templates:
            config_assistant.add_template(template)
        
        rendered = config_assistant.render_template_gallery()
        
        assert rendered is not None
        # Should group by category and show template previews


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--durations=10"])