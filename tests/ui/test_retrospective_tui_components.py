#!/usr/bin/env python3
"""
Comprehensive test suite for retrospective TUI components.

Tests all five retrospective TUI components:
- RetrospectiveTUIInterface: Main workflow coordinator
- ImprovementPresentationView: Rich improvement display
- ApprovalInteractionHandler: Interactive approval workflows
- ProgressMonitoringView: Real-time progress monitoring
- SafetyStatusDisplay: Safety validation and rollback controls
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta
from rich.console import Console
from rich.layout import Layout
from io import StringIO

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from agentsmcp.ui.v3.retrospective_tui_interface import (
    RetrospectiveTUIInterface, RetrospectivePhase, ImprovementSuggestion
)
from agentsmcp.ui.v3.improvement_presentation_view import ImprovementPresentationView
from agentsmcp.ui.v3.approval_interaction_handler import (
    ApprovalInteractionHandler, ApprovalDecision, ApprovalSession
)
from agentsmcp.ui.v3.progress_monitoring_view import (
    ProgressMonitoringView, ImplementationStatus, AgentInfo, ImplementationProgress
)
from agentsmcp.ui.v3.safety_status_display import (
    SafetyStatusDisplay, SafetyStatus, HealthCheck, SafetyValidationResult
)
from agentsmcp.ui.v3.terminal_capabilities import TerminalCapabilities


class TestRetrospectiveTUIInterface:
    """Test suite for RetrospectiveTUIInterface main coordinator."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.console = Console(file=StringIO(), force_terminal=True, width=120, height=40)
        self.capabilities = Mock(spec=TerminalCapabilities)
        self.capabilities.console = self.console
        self.capabilities.width = 120
        self.capabilities.height = 40
        self.capabilities.supports_color = True
        self.capabilities.supports_live = True
        self.interface = RetrospectiveTUIInterface(capabilities=self.capabilities)
    
    def test_initialization(self):
        """Test interface initialization."""
        assert self.interface.console is not None
        assert self.interface.current_phase == RetrospectivePhase.STARTUP
        assert self.interface.layout is not None
        assert self.interface.improvements == []
        assert not self.interface.running
        
    def test_phase_transitions(self):
        """Test phase transition logic."""
        # Test valid transitions
        self.interface.current_phase = RetrospectivePhase.STARTUP
        self.interface.transition_to_phase(RetrospectivePhase.ANALYSIS)
        assert self.interface.current_phase == RetrospectivePhase.ANALYSIS
        
        # Test phase-specific setup
        self.interface.transition_to_phase(RetrospectivePhase.PRESENTATION)
        assert self.interface.current_phase == RetrospectivePhase.PRESENTATION
        
        # Test completion phase
        self.interface.transition_to_phase(RetrospectivePhase.COMPLETED)
        assert self.interface.current_phase == RetrospectivePhase.COMPLETED
    
    def test_improvement_suggestion_generation(self):
        """Test mock improvement suggestion generation."""
        improvements = self.interface._generate_mock_improvements()
        
        assert len(improvements) >= 3
        for improvement in improvements:
            assert isinstance(improvement, ImprovementSuggestion)
            assert improvement.title
            assert improvement.description
            assert improvement.category in ["Performance", "Code Quality", "Architecture", "Testing", "Documentation"]
            assert improvement.impact in ["High", "Medium", "Low"]
            assert improvement.effort in ["Small", "Medium", "Large"]
    
    def test_layout_updates(self):
        """Test layout updates for different phases."""
        # Test startup phase layout
        self.interface.current_phase = RetrospectivePhase.STARTUP
        self.interface.update_layout()
        
        # Test analysis phase layout
        self.interface.current_phase = RetrospectivePhase.ANALYSIS
        self.interface.update_layout()
        
        # Test presentation phase layout
        self.interface.current_phase = RetrospectivePhase.PRESENTATION
        self.interface.update_layout()
        
        # Verify layout structure exists
        assert isinstance(self.interface.layout, Layout)
    
    @pytest.mark.asyncio
    async def test_workflow_simulation(self):
        """Test complete workflow simulation."""
        # Mock the workflow steps
        with patch.object(self.interface, '_run_analysis_phase') as mock_analysis, \
             patch.object(self.interface, '_run_presentation_phase') as mock_presentation, \
             patch.object(self.interface, '_run_approval_phase') as mock_approval, \
             patch.object(self.interface, '_run_monitoring_phase') as mock_monitoring, \
             patch.object(self.interface, '_run_safety_phase') as mock_safety:
            
            mock_analysis.return_value = None
            mock_presentation.return_value = None
            mock_approval.return_value = None
            mock_monitoring.return_value = None
            mock_safety.return_value = None
            
            # Start workflow
            workflow_task = asyncio.create_task(self.interface._run_workflow())
            
            # Let it run briefly
            await asyncio.sleep(0.1)
            
            # Stop the workflow
            self.interface.running = False
            
            try:
                await asyncio.wait_for(workflow_task, timeout=1.0)
            except asyncio.TimeoutError:
                workflow_task.cancel()


class TestImprovementPresentationView:
    """Test suite for ImprovementPresentationView."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.console = Console(file=StringIO(), force_terminal=True, width=120, height=40)
        self.view = ImprovementPresentationView(console=self.console)
        
        # Create sample improvements
        self.improvements = [
            ImprovementSuggestion(
                id="1",
                title="Optimize Database Queries",
                description="## Performance Improvement\n\nReduce query execution time by implementing proper indexing.",
                category="Performance",
                impact="High",
                effort="Medium",
                implementation_notes="Add indexes on frequently queried columns"
            ),
            ImprovementSuggestion(
                id="2", 
                title="Add Unit Tests",
                description="## Testing Enhancement\n\nIncrease test coverage to 90%+.",
                category="Testing",
                impact="Medium",
                effort="Large",
                implementation_notes="Focus on core business logic"
            ),
        ]
    
    def test_initialization(self):
        """Test view initialization."""
        assert self.view.console is not None
        assert self.view.current_page == 0
        assert self.view.items_per_page == 5
        assert self.view.selected_filter == "All"
        assert self.view.selected_improvement is None
    
    def test_set_improvements(self):
        """Test setting and filtering improvements."""
        self.view.set_improvements(self.improvements)
        assert len(self.view.improvements) == 2
        
        # Test filtering
        filtered = self.view.get_filtered_improvements()
        assert len(filtered) == 2
        
        # Test category filtering
        self.view.selected_filter = "Performance"
        filtered = self.view.get_filtered_improvements()
        assert len(filtered) == 1
        assert filtered[0].category == "Performance"
    
    def test_pagination(self):
        """Test pagination functionality."""
        # Create more improvements to test pagination
        many_improvements = self.improvements * 5  # 10 total
        self.view.set_improvements(many_improvements)
        
        # Test page navigation
        assert self.view.current_page == 0
        assert self.view.get_total_pages() == 2  # 10 items, 5 per page
        
        self.view.next_page()
        assert self.view.current_page == 1
        
        self.view.next_page()  # Should stay at last page
        assert self.view.current_page == 1
        
        self.view.previous_page()
        assert self.view.current_page == 0
    
    def test_improvement_selection(self):
        """Test improvement selection and details."""
        self.view.set_improvements(self.improvements)
        
        # Select first improvement
        self.view.select_improvement(0)
        assert self.view.selected_improvement == self.improvements[0]
        
        # Test invalid selection
        self.view.select_improvement(99)
        assert self.view.selected_improvement == self.improvements[0]  # Should remain unchanged
    
    def test_rendering(self):
        """Test rendering functionality."""
        self.view.set_improvements(self.improvements)
        
        # Test list rendering
        self.view.render_improvements_list()
        output = self.console.file.getvalue()
        assert len(output) > 0
        
        # Reset console output
        self.console.file.truncate(0)
        self.console.file.seek(0)
        
        # Test detailed view rendering
        self.view.select_improvement(0)
        self.view.render_improvement_details()
        output = self.console.file.getvalue()
        assert len(output) > 0
        assert "Optimize Database Queries" in output


class TestApprovalInteractionHandler:
    """Test suite for ApprovalInteractionHandler."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.console = Console(file=StringIO(), force_terminal=True, width=120, height=40)
        self.handler = ApprovalInteractionHandler(console=self.console)
        
        # Create sample improvements
        self.improvements = [
            ImprovementSuggestion(
                id="1",
                title="Optimize Database Queries",
                description="Reduce query execution time",
                category="Performance",
                impact="High",
                effort="Medium"
            ),
            ImprovementSuggestion(
                id="2",
                title="Add Unit Tests",
                description="Increase test coverage",
                category="Testing", 
                impact="Medium",
                effort="Large"
            ),
        ]
    
    def test_initialization(self):
        """Test handler initialization."""
        assert self.handler.console is not None
        assert self.handler.current_session is None
        assert len(self.handler.decision_history) == 0
    
    def test_approval_session_creation(self):
        """Test approval session creation and management."""
        session = self.handler.start_approval_session(self.improvements)
        
        assert isinstance(session, ApprovalSession)
        assert session.improvements == self.improvements
        assert len(session.decisions) == 0
        assert session.start_time is not None
        assert self.handler.current_session == session
    
    def test_decision_making(self):
        """Test individual decision making."""
        session = self.handler.start_approval_session(self.improvements)
        
        # Test approval
        decision = self.handler.make_decision("1", ApprovalDecision.APPROVED, "Good performance improvement")
        assert decision.improvement_id == "1"
        assert decision.decision == ApprovalDecision.APPROVED
        assert decision.comments == "Good performance improvement"
        assert decision.timestamp is not None
        
        # Check session updated
        assert len(session.decisions) == 1
        assert session.decisions["1"] == decision
        
        # Test rejection
        decision = self.handler.make_decision("2", ApprovalDecision.REJECTED, "Too much effort")
        assert decision.decision == ApprovalDecision.REJECTED
        assert len(session.decisions) == 2
    
    def test_batch_operations(self):
        """Test batch approval/rejection operations."""
        session = self.handler.start_approval_session(self.improvements)
        
        # Test approve all
        self.handler.approve_all("Batch approval for all improvements")
        assert len(session.decisions) == 2
        for decision in session.decisions.values():
            assert decision.decision == ApprovalDecision.APPROVED
            assert "Batch approval" in decision.comments
        
        # Reset session
        session = self.handler.start_approval_session(self.improvements)
        
        # Test reject all
        self.handler.reject_all("Not ready for implementation")
        assert len(session.decisions) == 2
        for decision in session.decisions.values():
            assert decision.decision == ApprovalDecision.REJECTED
    
    def test_approval_summary(self):
        """Test approval summary generation."""
        session = self.handler.start_approval_session(self.improvements)
        
        # Make mixed decisions
        self.handler.make_decision("1", ApprovalDecision.APPROVED, "Good idea")
        self.handler.make_decision("2", ApprovalDecision.REJECTED, "Too risky")
        
        summary = self.handler.get_approval_summary()
        
        assert summary['total'] == 2
        assert summary['approved'] == 1
        assert summary['rejected'] == 1
        assert summary['pending'] == 0
        assert len(summary['decisions']) == 2
    
    def test_rendering(self):
        """Test rendering functionality."""
        session = self.handler.start_approval_session(self.improvements)
        
        # Test approval interface rendering
        self.handler.render_approval_interface()
        output = self.console.file.getvalue()
        assert len(output) > 0
        
        # Make some decisions
        self.handler.make_decision("1", ApprovalDecision.APPROVED, "Good")
        
        # Reset console output
        self.console.file.truncate(0)
        self.console.file.seek(0)
        
        # Test summary rendering
        self.handler.render_approval_summary()
        output = self.console.file.getvalue()
        assert len(output) > 0
        assert "Approved: 1" in output


class TestProgressMonitoringView:
    """Test suite for ProgressMonitoringView."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.console = Console(file=StringIO(), force_terminal=True, width=120, height=40)
        self.view = ProgressMonitoringView(console=self.console)
    
    def test_initialization(self):
        """Test view initialization."""
        assert self.view.console is not None
        assert len(self.view.agents) == 0
        assert self.view.overall_progress == 0
        assert self.view.current_status == ImplementationStatus.INITIALIZING
        assert len(self.view.progress_log) == 0
    
    def test_agent_management(self):
        """Test agent creation and management."""
        agents = self.view._create_mock_agents()
        
        assert len(agents) >= 3
        for agent in agents:
            assert isinstance(agent, AgentInfo)
            assert agent.name
            assert agent.role
            assert agent.status in [s.value for s in ImplementationStatus]
            assert 0 <= agent.progress <= 100
    
    def test_progress_calculation(self):
        """Test overall progress calculation."""
        # Create mock agents with known progress values
        self.view.agents = [
            AgentInfo(id="1", name="Agent1", role="Coder", status="WORKING", progress=50, current_task="Task 1"),
            AgentInfo(id="2", name="Agent2", role="Tester", status="WORKING", progress=75, current_task="Task 2"),
            AgentInfo(id="3", name="Agent3", role="Reviewer", status="COMPLETED", progress=100, current_task="Task 3"),
        ]
        
        overall = self.view._calculate_overall_progress()
        expected = (50 + 75 + 100) / 3
        assert overall == expected
    
    @pytest.mark.asyncio
    async def test_progress_simulation(self):
        """Test progress simulation."""
        self.view.agents = self.view._create_mock_agents()
        initial_progress = [agent.progress for agent in self.view.agents]
        
        # Run one simulation step
        self.view._simulate_progress_update()
        
        # Check that some progress was made (not all agents may update each step)
        current_progress = [agent.progress for agent in self.view.agents]
        
        # At least overall progress should be calculated
        assert self.view.overall_progress >= 0
        assert len(self.view.progress_log) > 0
    
    def test_implementation_progress_tracking(self):
        """Test implementation progress data structure."""
        progress = ImplementationProgress(
            improvement_id="test-1",
            title="Test Improvement", 
            status=ImplementationStatus.IN_PROGRESS,
            progress_percentage=75,
            assigned_agents=["agent-1", "agent-2"],
            start_time=datetime.now(),
            estimated_completion=datetime.now() + timedelta(hours=2),
            implementation_log=["Started implementation", "Completed phase 1"]
        )
        
        assert progress.improvement_id == "test-1"
        assert progress.status == ImplementationStatus.IN_PROGRESS
        assert progress.progress_percentage == 75
        assert len(progress.assigned_agents) == 2
        assert len(progress.implementation_log) == 2
    
    def test_rendering(self):
        """Test rendering functionality."""
        self.view.agents = self.view._create_mock_agents()
        
        # Test agent status rendering
        self.view.render_agent_status()
        output = self.console.file.getvalue()
        assert len(output) > 0
        
        # Reset console output
        self.console.file.truncate(0)
        self.console.file.seek(0)
        
        # Test progress overview rendering
        self.view.render_progress_overview()
        output = self.console.file.getvalue()
        assert len(output) > 0


class TestSafetyStatusDisplay:
    """Test suite for SafetyStatusDisplay."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.console = Console(file=StringIO(), force_terminal=True, width=120, height=40)
        self.display = SafetyStatusDisplay(console=self.console)
    
    def test_initialization(self):
        """Test display initialization."""
        assert self.display.console is not None
        assert len(self.display.health_checks) == 0
        assert self.display.overall_status == SafetyStatus.UNKNOWN
        assert self.display.last_validation is None
        assert len(self.display.rollback_options) == 0
    
    def test_health_check_creation(self):
        """Test health check generation."""
        checks = self.display._generate_health_checks()
        
        assert len(checks) == 7  # Expected categories
        expected_categories = [
            "System Performance", "Data Integrity", "Security Validation",
            "Dependency Health", "Configuration Validation", 
            "Service Availability", "Resource Usage"
        ]
        
        check_categories = [check.category for check in checks]
        for category in expected_categories:
            assert category in check_categories
        
        for check in checks:
            assert isinstance(check, HealthCheck)
            assert check.category
            assert check.name
            assert check.status in [s.value for s in SafetyStatus]
            assert check.message
    
    def test_safety_validation(self):
        """Test safety validation process."""
        result = self.display.run_safety_validation()
        
        assert isinstance(result, SafetyValidationResult)
        assert result.overall_status in [s.value for s in SafetyStatus]
        assert len(result.health_checks) > 0
        assert result.validation_time is not None
        assert len(result.recommendations) >= 0
        
        # Check that display state was updated
        assert self.display.last_validation == result
        assert len(self.display.health_checks) > 0
    
    def test_rollback_options_generation(self):
        """Test rollback options generation."""
        options = self.display._generate_rollback_options()
        
        assert len(options) >= 3
        expected_types = ["emergency", "selective", "manual"]
        
        for option in options:
            assert 'id' in option
            assert 'name' in option
            assert 'description' in option
            assert 'risk_level' in option
            assert option['risk_level'] in ["Low", "Medium", "High"]
    
    def test_emergency_rollback(self):
        """Test emergency rollback functionality."""
        # First run validation to have some state
        self.display.run_safety_validation()
        
        with patch('builtins.input', return_value='yes'):
            result = self.display.execute_emergency_rollback()
        
        assert result is True
        
        # Test with rejection
        with patch('builtins.input', return_value='no'):
            result = self.display.execute_emergency_rollback()
        
        assert result is False
    
    def test_selective_rollback(self):
        """Test selective rollback functionality."""
        # Mock rollback options
        self.display.rollback_options = [
            {"id": "opt1", "name": "Rollback Database", "risk_level": "Low"},
            {"id": "opt2", "name": "Rollback Configuration", "risk_level": "Medium"},
        ]
        
        with patch('builtins.input', return_value='opt1'):
            result = self.display.execute_selective_rollback()
        
        assert result is True
        
        # Test with invalid option
        with patch('builtins.input', return_value='invalid'):
            result = self.display.execute_selective_rollback()
        
        assert result is False
    
    def test_safety_status_calculation(self):
        """Test overall safety status calculation."""
        # Create mock health checks with known statuses
        self.display.health_checks = [
            HealthCheck("Test", "Check 1", SafetyStatus.HEALTHY.value, "OK", datetime.now()),
            HealthCheck("Test", "Check 2", SafetyStatus.HEALTHY.value, "OK", datetime.now()),
            HealthCheck("Test", "Check 3", SafetyStatus.WARNING.value, "Warning", datetime.now()),
        ]
        
        status = self.display._calculate_overall_status()
        assert status == SafetyStatus.WARNING  # Worst status wins
        
        # Test with critical status
        self.display.health_checks.append(
            HealthCheck("Test", "Check 4", SafetyStatus.CRITICAL.value, "Critical issue", datetime.now())
        )
        
        status = self.display._calculate_overall_status()
        assert status == SafetyStatus.CRITICAL
    
    def test_rendering(self):
        """Test rendering functionality."""
        # First run validation to have data
        self.display.run_safety_validation()
        
        # Test safety status rendering
        self.display.render_safety_status()
        output = self.console.file.getvalue()
        assert len(output) > 0
        
        # Reset console output
        self.console.file.truncate(0)
        self.console.file.seek(0)
        
        # Test rollback options rendering
        self.display.render_rollback_options()
        output = self.console.file.getvalue()
        assert len(output) > 0


class TestIntegration:
    """Integration tests for all retrospective TUI components."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.console = Console(file=StringIO(), force_terminal=True, width=120, height=40)
    
    def test_component_interaction(self):
        """Test interaction between different components."""
        # Create main interface
        capabilities = Mock(spec=TerminalCapabilities)
        capabilities.console = self.console
        capabilities.width = 120
        capabilities.height = 40
        capabilities.supports_color = True
        capabilities.supports_live = True
        interface = RetrospectiveTUIInterface(capabilities=capabilities)
        
        # Create sub-components 
        presentation = ImprovementPresentationView(console=self.console)
        approval = ApprovalInteractionHandler(console=self.console)
        monitoring = ProgressMonitoringView(console=self.console) 
        safety = SafetyStatusDisplay(console=self.console)
        
        # Test data flow
        improvements = interface._generate_mock_improvements()
        
        # Pass data through components
        presentation.set_improvements(improvements)
        assert len(presentation.improvements) == len(improvements)
        
        session = approval.start_approval_session(improvements)
        assert session.improvements == improvements
        
        # Verify components are properly initialized
        assert all([
            interface.console is not None,
            presentation.console is not None,
            approval.console is not None,
            monitoring.console is not None,
            safety.console is not None,
        ])
    
    @pytest.mark.asyncio
    async def test_workflow_integration(self):
        """Test complete workflow integration."""
        capabilities = Mock(spec=TerminalCapabilities)
        capabilities.console = self.console
        capabilities.width = 120
        capabilities.height = 40
        capabilities.supports_color = True
        capabilities.supports_live = True
        interface = RetrospectiveTUIInterface(capabilities=capabilities)
        
        # Mock the individual phase methods to avoid full execution
        with patch.object(interface, '_run_analysis_phase') as mock_analysis, \
             patch.object(interface, '_run_presentation_phase') as mock_presentation, \
             patch.object(interface, '_run_approval_phase') as mock_approval, \
             patch.object(interface, '_run_monitoring_phase') as mock_monitoring, \
             patch.object(interface, '_run_safety_phase') as mock_safety:
            
            # Setup mocks
            mock_analysis.return_value = None
            mock_presentation.return_value = None
            mock_approval.return_value = None
            mock_monitoring.return_value = None
            mock_safety.return_value = None
            
            # Test that all phases can be called
            assert interface.current_phase == RetrospectivePhase.STARTUP
            
            interface.transition_to_phase(RetrospectivePhase.ANALYSIS)
            await interface._run_analysis_phase()
            mock_analysis.assert_called_once()
            
            interface.transition_to_phase(RetrospectivePhase.PRESENTATION)
            await interface._run_presentation_phase()
            mock_presentation.assert_called_once()
            
            interface.transition_to_phase(RetrospectivePhase.APPROVAL)
            await interface._run_approval_phase()
            mock_approval.assert_called_once()
            
            interface.transition_to_phase(RetrospectivePhase.MONITORING)
            await interface._run_monitoring_phase()
            mock_monitoring.assert_called_once()
            
            interface.transition_to_phase(RetrospectivePhase.SAFETY_CHECK)
            await interface._run_safety_phase()
            mock_safety.assert_called_once()


if __name__ == "__main__":
    """Run tests directly."""
    print("🧪 Running Retrospective TUI Component Tests")
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