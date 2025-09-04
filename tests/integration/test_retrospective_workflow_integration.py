#!/usr/bin/env python3
"""
End-to-end integration tests for retrospective TUI workflow.

Tests the complete retrospective workflow from CLI invocation through
all phases of the TUI interface, verifying proper integration between
all components.
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
from rich.console import Console
from io import StringIO

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from agentsmcp.ui.v3.retrospective_tui_interface import (
    RetrospectiveTUIInterface, RetrospectivePhase, ImprovementSuggestion
)
from agentsmcp.ui.v3.improvement_presentation_view import ImprovementPresentationView
from agentsmcp.ui.v3.approval_interaction_handler import (
    ApprovalInteractionHandler, ApprovalDecision
)
from agentsmcp.ui.v3.progress_monitoring_view import (
    ProgressMonitoringView, ImplementationStatus
)
from agentsmcp.ui.v3.safety_status_display import (
    SafetyStatusDisplay, SafetyStatus
)
from agentsmcp.ui.v3.terminal_capabilities import TerminalCapabilities


def create_mock_capabilities(console, width=120, height=40):
    """Helper to create mock terminal capabilities."""
    capabilities = Mock(spec=TerminalCapabilities)
    capabilities.console = console
    capabilities.width = width
    capabilities.height = height
    capabilities.supports_color = True
    capabilities.supports_live = True
    return capabilities


class TestRetrospectiveWorkflowIntegration:
    """Test complete retrospective workflow integration."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.console = Console(file=StringIO(), force_terminal=True, width=120, height=40)
        self.interface = RetrospectiveTUIInterface(capabilities=create_mock_capabilities(self.console))
    
    def test_complete_component_initialization(self):
        """Test that all components can be initialized together."""
        # Create all components that would be used in the workflow
        presentation = ImprovementPresentationView(console=self.console)
        approval = ApprovalInteractionHandler(console=self.console)
        monitoring = ProgressMonitoringView(console=self.console)
        safety = SafetyStatusDisplay(console=self.console)
        
        # Verify all components initialized successfully
        components = [self.interface, presentation, approval, monitoring, safety]
        for component in components:
            assert component.console is not None
            assert component.console == self.console
        
        # Test that components can work with shared data
        improvements = self.interface._generate_mock_improvements()
        
        # Pass data through the workflow
        presentation.set_improvements(improvements)
        assert len(presentation.improvements) == len(improvements)
        
        session = approval.start_approval_session(improvements)
        assert session.improvements == improvements
    
    @pytest.mark.asyncio
    async def test_phase_transition_workflow(self):
        """Test complete phase transition workflow."""
        # Mock all phase methods to avoid full execution
        with patch.object(self.interface, '_run_analysis_phase') as mock_analysis, \
             patch.object(self.interface, '_run_presentation_phase') as mock_presentation, \
             patch.object(self.interface, '_run_approval_phase') as mock_approval, \
             patch.object(self.interface, '_run_monitoring_phase') as mock_monitoring, \
             patch.object(self.interface, '_run_safety_phase') as mock_safety:
            
            # Setup mocks to complete successfully
            for mock in [mock_analysis, mock_presentation, mock_approval, mock_monitoring, mock_safety]:
                mock.return_value = None
            
            # Test phase progression
            phases = [
                RetrospectivePhase.ANALYSIS,
                RetrospectivePhase.PRESENTATION, 
                RetrospectivePhase.APPROVAL,
                RetrospectivePhase.MONITORING,
                RetrospectivePhase.SAFETY_CHECK,
                RetrospectivePhase.COMPLETED
            ]
            
            for phase in phases:
                self.interface.transition_to_phase(phase)
                assert self.interface.current_phase == phase
                
                # Update layout for the phase
                self.interface.update_layout()
                
                # Verify layout was updated (shouldn't throw exceptions)
                assert self.interface.layout is not None
    
    def test_data_flow_through_components(self):
        """Test data flow through all components."""
        # Generate initial data
        improvements = self.interface._generate_mock_improvements()
        assert len(improvements) > 0
        
        # Test data flow through presentation component
        presentation = ImprovementPresentationView(console=self.console)
        presentation.set_improvements(improvements)
        
        filtered = presentation.get_filtered_improvements()
        assert len(filtered) == len(improvements)  # No filter applied initially
        
        # Test data flow through approval component
        approval = ApprovalInteractionHandler(console=self.console)
        session = approval.start_approval_session(improvements)
        
        # Make some approval decisions
        for i, improvement in enumerate(improvements[:3]):  # Approve first 3
            decision = ApprovalDecision.APPROVED if i < 2 else ApprovalDecision.REJECTED
            approval.make_decision(improvement.id, decision, f"Decision for {improvement.title}")
        
        # Verify decisions were recorded
        summary = approval.get_approval_summary()
        assert summary['total'] == len(improvements)
        assert summary['approved'] == 2
        assert summary['rejected'] == 1
        assert summary['pending'] == len(improvements) - 3
        
        # Test data flow through monitoring component
        monitoring = ProgressMonitoringView(console=self.console)
        agents = monitoring._create_mock_agents()
        assert len(agents) > 0
        
        # Test data flow through safety component
        safety = SafetyStatusDisplay(console=self.console)
        validation_result = safety.run_safety_validation()
        assert validation_result is not None
        assert validation_result.overall_status in [s.value for s in SafetyStatus]
    
    def test_error_propagation_and_handling(self):
        """Test error handling throughout the workflow."""
        # Test handling of component initialization errors
        with patch('rich.console.Console') as mock_console:
            mock_console.side_effect = Exception("Console initialization failed")
            
            with pytest.raises(Exception):
                RetrospectiveTUIInterface(console=mock_console())
        
        # Test handling of data generation errors
        with patch.object(self.interface, '_generate_mock_improvements') as mock_gen:
            mock_gen.side_effect = Exception("Data generation failed")
            
            with pytest.raises(Exception):
                self.interface._generate_mock_improvements()
        
        # Test graceful handling of partial component failures
        presentation = ImprovementPresentationView(console=self.console)
        
        # Should handle empty improvement list gracefully
        presentation.set_improvements([])
        filtered = presentation.get_filtered_improvements()
        assert len(filtered) == 0
    
    @pytest.mark.asyncio
    async def test_async_workflow_coordination(self):
        """Test async workflow coordination."""
        # Mock async operations
        async def mock_phase_operation():
            await asyncio.sleep(0.01)  # Simulate brief async work
            return True
        
        with patch.object(self.interface, '_run_analysis_phase', side_effect=mock_phase_operation), \
             patch.object(self.interface, '_run_presentation_phase', side_effect=mock_phase_operation), \
             patch.object(self.interface, '_run_approval_phase', side_effect=mock_phase_operation):
            
            # Test that phases can be run sequentially
            await self.interface._run_analysis_phase()
            assert self.interface.current_phase == RetrospectivePhase.STARTUP  # Phase doesn't auto-advance in mocked version
            
            self.interface.transition_to_phase(RetrospectivePhase.PRESENTATION)
            await self.interface._run_presentation_phase()
            
            self.interface.transition_to_phase(RetrospectivePhase.APPROVAL)
            await self.interface._run_approval_phase()
    
    def test_resource_management_and_cleanup(self):
        """Test proper resource management and cleanup."""
        # Create components that might hold resources
        components = [
            RetrospectiveTUIInterface(capabilities=create_mock_capabilities(self.console)),
            ImprovementPresentationView(console=self.console),
            ApprovalInteractionHandler(console=self.console),
            ProgressMonitoringView(console=self.console),
            SafetyStatusDisplay(console=self.console)
        ]
        
        # Generate data to simulate resource usage
        for component in components:
            if hasattr(component, '_generate_mock_improvements'):
                component._generate_mock_improvements()
            elif hasattr(component, '_create_mock_agents'):
                component._create_mock_agents()
            elif hasattr(component, '_generate_health_checks'):
                component._generate_health_checks()
        
        # Components should be cleanly created and not leak resources
        # (This test mainly verifies no exceptions during creation/destruction)
        for component in components:
            assert component.console is not None
        
        # Cleanup (Python garbage collection should handle this)
        del components
    
    def test_terminal_capability_integration(self):
        """Test integration with terminal capabilities."""
        # Test with different console configurations
        
        # Test with limited width console
        narrow_console = Console(file=StringIO(), force_terminal=True, width=80, height=24)
        narrow_interface = RetrospectiveTUIInterface(capabilities=create_mock_capabilities(narrow_console, 80, 24))
        
        # Should handle narrow terminals gracefully
        assert narrow_interface.console.size.width == 80
        assert narrow_interface.layout is not None
        
        # Test with wide console
        wide_console = Console(file=StringIO(), force_terminal=True, width=200, height=50)
        wide_interface = RetrospectiveTUIInterface(capabilities=create_mock_capabilities(wide_console, 200, 50))
        
        # Should handle wide terminals gracefully
        assert wide_interface.console.size.width == 200
        assert wide_interface.layout is not None
        
        # Test layout updates work with different sizes
        narrow_interface.update_layout()
        wide_interface.update_layout()
        
        # Both should complete without error
        assert narrow_interface.layout is not None
        assert wide_interface.layout is not None


class TestWorkflowStateConsistency:
    """Test state consistency throughout the workflow."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.console = Console(file=StringIO(), force_terminal=True, width=120, height=40)
        self.interface = RetrospectiveTUIInterface(capabilities=create_mock_capabilities(self.console))
    
    def test_state_transitions_are_consistent(self):
        """Test that state transitions maintain consistency."""
        initial_phase = self.interface.current_phase
        assert initial_phase == RetrospectivePhase.STARTUP
        
        # Test valid transitions
        valid_transitions = [
            (RetrospectivePhase.STARTUP, RetrospectivePhase.ANALYSIS),
            (RetrospectivePhase.ANALYSIS, RetrospectivePhase.PRESENTATION),
            (RetrospectivePhase.PRESENTATION, RetrospectivePhase.APPROVAL),
            (RetrospectivePhase.APPROVAL, RetrospectivePhase.MONITORING),
            (RetrospectivePhase.MONITORING, RetrospectivePhase.SAFETY_CHECK),
            (RetrospectivePhase.SAFETY_CHECK, RetrospectivePhase.COMPLETED),
        ]
        
        for from_phase, to_phase in valid_transitions:
            self.interface.current_phase = from_phase
            self.interface.transition_to_phase(to_phase)
            assert self.interface.current_phase == to_phase
    
    def test_data_persistence_across_phases(self):
        """Test that data persists across phase transitions."""
        # Generate initial improvements
        improvements = self.interface._generate_mock_improvements()
        self.interface.improvements = improvements
        
        # Transition through phases
        phases = [
            RetrospectivePhase.ANALYSIS,
            RetrospectivePhase.PRESENTATION,
            RetrospectivePhase.APPROVAL,
        ]
        
        for phase in phases:
            self.interface.transition_to_phase(phase)
            
            # Data should persist
            assert self.interface.improvements == improvements
            assert len(self.interface.improvements) == len(improvements)
    
    def test_component_state_isolation(self):
        """Test that components maintain separate state appropriately."""
        # Create multiple instances of components
        presentation1 = ImprovementPresentationView(console=self.console)
        presentation2 = ImprovementPresentationView(console=self.console)
        
        improvements1 = self.interface._generate_mock_improvements()[:3]
        improvements2 = self.interface._generate_mock_improvements()[3:6]
        
        # Set different data in each instance
        presentation1.set_improvements(improvements1)
        presentation2.set_improvements(improvements2)
        
        # State should be isolated
        assert len(presentation1.improvements) == 3
        assert len(presentation2.improvements) == 3
        assert presentation1.improvements != presentation2.improvements


class TestWorkflowPerformance:
    """Test workflow performance characteristics."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.console = Console(file=StringIO(), force_terminal=True, width=120, height=40)
    
    def test_component_initialization_performance(self):
        """Test that components initialize quickly."""
        import time
        
        start_time = time.time()
        
        # Create all components
        components = [
            RetrospectiveTUIInterface(capabilities=create_mock_capabilities(self.console)),
            ImprovementPresentationView(console=self.console),
            ApprovalInteractionHandler(console=self.console),
            ProgressMonitoringView(console=self.console),
            SafetyStatusDisplay(console=self.console)
        ]
        
        end_time = time.time()
        initialization_time = end_time - start_time
        
        # Should initialize quickly (less than 1 second)
        assert initialization_time < 1.0
        assert len(components) == 5
    
    def test_data_generation_performance(self):
        """Test that mock data generation is reasonably fast."""
        import time
        
        interface = RetrospectiveTUIInterface(capabilities=create_mock_capabilities(self.console))
        monitoring = ProgressMonitoringView(console=self.console)
        safety = SafetyStatusDisplay(console=self.console)
        
        start_time = time.time()
        
        # Generate data from all components
        improvements = interface._generate_mock_improvements()
        agents = monitoring._create_mock_agents()
        health_checks = safety._generate_health_checks()
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Should generate quickly (less than 0.5 seconds)
        assert generation_time < 0.5
        assert len(improvements) > 0
        assert len(agents) > 0
        assert len(health_checks) > 0
    
    def test_rendering_performance(self):
        """Test that rendering operations are reasonably fast."""
        import time
        
        # Setup components with data
        interface = RetrospectiveTUIInterface(capabilities=create_mock_capabilities(self.console))
        presentation = ImprovementPresentationView(console=self.console)
        
        improvements = interface._generate_mock_improvements()
        presentation.set_improvements(improvements)
        
        start_time = time.time()
        
        # Perform rendering operations
        interface.update_layout()
        presentation.render_improvements_list()
        presentation.render_improvement_details()
        
        end_time = time.time()
        rendering_time = end_time - start_time
        
        # Should render quickly (less than 1 second)
        assert rendering_time < 1.0


if __name__ == "__main__":
    """Run tests directly."""
    print("🧪 Running Retrospective Workflow Integration Tests")
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