#!/usr/bin/env python3
"""
Tests for retrospective TUI mock data generation.

Validates that mock data generators produce consistent, realistic,
and comprehensive test data for all retrospective components.
"""

import pytest
import sys
import os
from datetime import datetime, timedelta
from collections import Counter

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from agentsmcp.ui.v3.retrospective_tui_interface import (
    RetrospectiveTUIInterface, ImprovementSuggestion
)
from agentsmcp.ui.v3.approval_interaction_handler import ApprovalDecision
from agentsmcp.ui.v3.progress_monitoring_view import (
    ProgressMonitoringView, ImplementationStatus, AgentInfo
)
from agentsmcp.ui.v3.safety_status_display import (
    SafetyStatusDisplay, SafetyStatus, HealthCheck
)
from agentsmcp.ui.v3.terminal_capabilities import TerminalCapabilities


class TestImprovementSuggestionGeneration:
    """Test improvement suggestion mock data generation."""
    
    def setup_method(self):
        """Setup for each test method."""
        capabilities = Mock(spec=TerminalCapabilities)
        capabilities.console = Mock()
        capabilities.width = 120
        capabilities.height = 40
        capabilities.supports_color = True
        capabilities.supports_live = True
        self.interface = RetrospectiveTUIInterface(capabilities=capabilities)
    
    def test_improvement_generation_consistency(self):
        """Test that improvement generation is consistent and complete."""
        improvements = self.interface._generate_mock_improvements()
        
        # Should generate multiple improvements
        assert len(improvements) >= 5
        
        # Each improvement should be properly formed
        for improvement in improvements:
            assert isinstance(improvement, ImprovementSuggestion)
            assert improvement.id
            assert improvement.title
            assert improvement.description
            assert improvement.category
            assert improvement.impact
            assert improvement.effort
            assert improvement.implementation_notes
    
    def test_improvement_category_distribution(self):
        """Test that improvements cover expected categories."""
        improvements = self.interface._generate_mock_improvements()
        
        expected_categories = [
            "Performance", "Code Quality", "Architecture", 
            "Testing", "Documentation", "Security"
        ]
        
        categories = [imp.category for imp in improvements]
        category_counts = Counter(categories)
        
        # Should have variety in categories
        assert len(category_counts) >= 3
        
        # All categories should be from expected set
        for category in categories:
            assert category in expected_categories
    
    def test_improvement_impact_levels(self):
        """Test that improvements have realistic impact levels."""
        improvements = self.interface._generate_mock_improvements()
        
        impacts = [imp.impact for imp in improvements]
        valid_impacts = ["High", "Medium", "Low"]
        
        # All impacts should be valid
        for impact in impacts:
            assert impact in valid_impacts
        
        # Should have variety in impact levels
        impact_counts = Counter(impacts)
        assert len(impact_counts) >= 2
    
    def test_improvement_effort_levels(self):
        """Test that improvements have realistic effort estimates."""
        improvements = self.interface._generate_mock_improvements()
        
        efforts = [imp.effort for imp in improvements]
        valid_efforts = ["Small", "Medium", "Large"]
        
        # All efforts should be valid
        for effort in efforts:
            assert effort in valid_efforts
        
        # Should have variety in effort levels
        effort_counts = Counter(efforts)
        assert len(effort_counts) >= 2
    
    def test_improvement_markdown_content(self):
        """Test that improvement descriptions contain markdown content."""
        improvements = self.interface._generate_mock_improvements()
        
        # At least some should have markdown formatting
        markdown_indicators = ["##", "###", "**", "*", "-", "`", "```"]
        markdown_found = False
        
        for improvement in improvements:
            for indicator in markdown_indicators:
                if indicator in improvement.description:
                    markdown_found = True
                    break
            if markdown_found:
                break
        
        assert markdown_found, "No markdown formatting found in improvement descriptions"
    
    def test_improvement_uniqueness(self):
        """Test that generated improvements are unique."""
        improvements = self.interface._generate_mock_improvements()
        
        # IDs should be unique
        ids = [imp.id for imp in improvements]
        assert len(ids) == len(set(ids)), "Improvement IDs should be unique"
        
        # Titles should be unique
        titles = [imp.title for imp in improvements]
        assert len(titles) == len(set(titles)), "Improvement titles should be unique"


class TestAgentInfoGeneration:
    """Test agent info mock data generation."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.monitoring = ProgressMonitoringView()
    
    def test_agent_generation_consistency(self):
        """Test that agent generation is consistent."""
        agents = self.monitoring._create_mock_agents()
        
        # Should generate multiple agents
        assert len(agents) >= 3
        
        # Each agent should be properly formed
        for agent in agents:
            assert isinstance(agent, AgentInfo)
            assert agent.id
            assert agent.name
            assert agent.role
            assert agent.status
            assert isinstance(agent.progress, int)
            assert 0 <= agent.progress <= 100
            assert agent.current_task
            assert isinstance(agent.last_update, datetime)
    
    def test_agent_role_distribution(self):
        """Test that agents have diverse roles."""
        agents = self.monitoring._create_mock_agents()
        
        expected_roles = [
            "Senior Developer", "Code Reviewer", "Test Engineer",
            "DevOps Engineer", "Security Analyst", "Documentation Writer"
        ]
        
        roles = [agent.role for agent in agents]
        
        # Should have variety in roles
        role_counts = Counter(roles)
        assert len(role_counts) >= 3
        
        # All roles should be from expected set
        for role in roles:
            assert role in expected_roles
    
    def test_agent_status_validity(self):
        """Test that agent statuses are valid."""
        agents = self.monitoring._create_mock_agents()
        
        valid_statuses = [status.value for status in ImplementationStatus]
        
        for agent in agents:
            assert agent.status in valid_statuses
    
    def test_agent_progress_realism(self):
        """Test that agent progress values are realistic."""
        agents = self.monitoring._create_mock_agents()
        
        # Progress should be distributed reasonably
        progress_values = [agent.progress for agent in agents]
        
        # Should not all be at 0% or 100%
        assert not all(p == 0 for p in progress_values)
        assert not all(p == 100 for p in progress_values)
        
        # Should have some variety
        assert len(set(progress_values)) >= 2
    
    def test_agent_timestamp_validity(self):
        """Test that agent timestamps are reasonable."""
        agents = self.monitoring._create_mock_agents()
        
        now = datetime.now()
        for agent in agents:
            # Last update should be recent (within last hour)
            time_diff = now - agent.last_update
            assert time_diff <= timedelta(hours=1)
            assert time_diff >= timedelta(0)


class TestHealthCheckGeneration:
    """Test health check mock data generation."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.safety = SafetyStatusDisplay()
    
    def test_health_check_generation_completeness(self):
        """Test that health check generation is complete."""
        health_checks = self.safety._generate_health_checks()
        
        # Should generate expected number of checks
        assert len(health_checks) == 7
        
        expected_categories = [
            "System Performance", "Data Integrity", "Security Validation",
            "Dependency Health", "Configuration Validation", 
            "Service Availability", "Resource Usage"
        ]
        
        # Each category should be represented
        categories = [check.category for check in health_checks]
        for expected_category in expected_categories:
            assert expected_category in categories
    
    def test_health_check_structure(self):
        """Test that health checks have proper structure."""
        health_checks = self.safety._generate_health_checks()
        
        for check in health_checks:
            assert isinstance(check, HealthCheck)
            assert check.category
            assert check.name
            assert check.status
            assert check.message
            assert isinstance(check.last_check, datetime)
    
    def test_health_check_status_distribution(self):
        """Test that health checks have realistic status distribution."""
        health_checks = self.safety._generate_health_checks()
        
        valid_statuses = [status.value for status in SafetyStatus]
        
        statuses = [check.status for check in health_checks]
        
        # All statuses should be valid
        for status in statuses:
            assert status in valid_statuses
        
        # Should have mostly healthy statuses with some warnings/issues
        status_counts = Counter(statuses)
        healthy_count = status_counts.get(SafetyStatus.HEALTHY.value, 0)
        
        # Most checks should be healthy in a normal system
        assert healthy_count >= len(health_checks) // 2
    
    def test_health_check_message_quality(self):
        """Test that health check messages are informative."""
        health_checks = self.safety._generate_health_checks()
        
        for check in health_checks:
            # Messages should be non-empty and informative
            assert len(check.message) > 10
            assert check.message != check.name
    
    def test_health_check_timestamp_validity(self):
        """Test that health check timestamps are valid."""
        health_checks = self.safety._generate_health_checks()
        
        now = datetime.now()
        for check in health_checks:
            # Last check should be recent
            time_diff = now - check.last_check
            assert time_diff <= timedelta(minutes=10)
            assert time_diff >= timedelta(0)


class TestRollbackOptionsGeneration:
    """Test rollback options mock data generation."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.safety = SafetyStatusDisplay()
    
    def test_rollback_options_generation(self):
        """Test that rollback options are properly generated."""
        options = self.safety._generate_rollback_options()
        
        # Should generate multiple options
        assert len(options) >= 3
        
        # Each option should have required fields
        for option in options:
            assert 'id' in option
            assert 'name' in option
            assert 'description' in option
            assert 'risk_level' in option
            assert 'estimated_time' in option
    
    def test_rollback_risk_levels(self):
        """Test that rollback options have appropriate risk levels."""
        options = self.safety._generate_rollback_options()
        
        valid_risk_levels = ["Low", "Medium", "High"]
        risk_levels = [option['risk_level'] for option in options]
        
        # All risk levels should be valid
        for risk_level in risk_levels:
            assert risk_level in valid_risk_levels
        
        # Should have variety in risk levels
        risk_counts = Counter(risk_levels)
        assert len(risk_counts) >= 2
    
    def test_rollback_option_uniqueness(self):
        """Test that rollback options are unique."""
        options = self.safety._generate_rollback_options()
        
        # IDs should be unique
        ids = [option['id'] for option in options]
        assert len(ids) == len(set(ids))
        
        # Names should be unique
        names = [option['name'] for option in options]
        assert len(names) == len(set(names))
    
    def test_rollback_time_estimates(self):
        """Test that rollback time estimates are realistic."""
        options = self.safety._generate_rollback_options()
        
        for option in options:
            time_estimate = option['estimated_time']
            # Should be a reasonable time estimate string
            assert isinstance(time_estimate, str)
            assert len(time_estimate) > 0
            # Should contain time units
            time_indicators = ['minute', 'hour', 'second', 'min', 'hr', 'sec']
            assert any(indicator in time_estimate.lower() for indicator in time_indicators)


class TestMockDataIntegration:
    """Test integration between different mock data generators."""
    
    def test_data_consistency_across_components(self):
        """Test that mock data is consistent across components."""
        capabilities = Mock(spec=TerminalCapabilities)
        capabilities.console = Mock()
        capabilities.width = 120
        capabilities.height = 40
        capabilities.supports_color = True
        capabilities.supports_live = True
        interface = RetrospectiveTUIInterface(capabilities=capabilities)
        monitoring = ProgressMonitoringView()
        safety = SafetyStatusDisplay()
        
        # Generate data from different components
        improvements = interface._generate_mock_improvements()
        agents = monitoring._create_mock_agents()
        health_checks = safety._generate_health_checks()
        
        # Data should be generated successfully
        assert len(improvements) > 0
        assert len(agents) > 0
        assert len(health_checks) > 0
        
        # Data should have consistent quality standards
        # All should have proper IDs/identifiers
        improvement_ids = [imp.id for imp in improvements]
        agent_ids = [agent.id for agent in agents]
        
        assert all(len(id_) > 0 for id_ in improvement_ids)
        assert all(len(id_) > 0 for id_ in agent_ids)
    
    def test_mock_data_reproducibility(self):
        """Test that mock data generation is somewhat reproducible."""
        capabilities = Mock(spec=TerminalCapabilities)
        capabilities.console = Mock()
        capabilities.width = 120
        capabilities.height = 40
        capabilities.supports_color = True
        capabilities.supports_live = True
        interface = RetrospectiveTUIInterface(capabilities=capabilities)
        
        # Generate data multiple times
        improvements1 = interface._generate_mock_improvements()
        improvements2 = interface._generate_mock_improvements()
        
        # Should generate same number of items (if using fixed seed or logic)
        assert len(improvements1) == len(improvements2)
        
        # Categories should be from same pool
        categories1 = set(imp.category for imp in improvements1)
        categories2 = set(imp.category for imp in improvements2)
        
        # Should use same category set (categories should overlap significantly)
        overlap = len(categories1.intersection(categories2))
        assert overlap >= len(categories1) // 2


if __name__ == "__main__":
    """Run tests directly."""
    print("🧪 Running Retrospective Mock Data Tests")
    print("=" * 60)
    
    # Run pytest programmatically
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--no-header",
        "--disable-warnings"
    ])
    
    sys.exit(exit_code)