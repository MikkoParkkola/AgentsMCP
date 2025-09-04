#!/usr/bin/env python3
"""
Focused End-to-End Retrospective System Integration Test

This test validates the complete workflow using the actual implemented components:
1. Process coach orchestration (mandatory leadership)
2. Retrospective analysis with pattern detection
3. Improvement generation and prioritization
4. Safety validation with rollback protection
5. User approval workflow (all modes)
6. TUI interface integration
7. Complete system integration validation

This focuses on testing the components that are actually implemented.
"""

import pytest
import asyncio
import sys
import os
import tempfile
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path
from io import StringIO

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agentsmcp.orchestration.process_coach import ProcessCoach
from agentsmcp.retrospective.safety.safety_orchestrator import SafetyOrchestrator
from agentsmcp.retrospective.approval.approval_workflow import ApprovalWorkflow
from agentsmcp.retrospective.generation.improvement_engine import ImprovementEngine
from agentsmcp.retrospective.storage.log_store import LogStore
from agentsmcp.retrospective.analysis.retrospective_analyzer import RetrospectiveAnalyzer
from agentsmcp.ui.v3.retrospective_tui_interface import RetrospectiveTUIInterface
from agentsmcp.ui.v3.terminal_capabilities import TerminalCapabilities
from rich.console import Console


class TestRetrospectiveSystemIntegration:
    """Integration test for the complete retrospective system."""

    def setup_method(self):
        """Setup for each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.console = Console(file=StringIO(), force_terminal=True, width=120, height=40)
        
        # Initialize system components
        self.components = {
            'process_coach': ProcessCoach(),
            'safety_orchestrator': SafetyOrchestrator(),
            'approval_workflow': ApprovalWorkflow(),
            'improvement_engine': ImprovementEngine(),
            'log_store': LogStore(str(self.temp_path / "logs.db")),
            'retrospective_analyzer': RetrospectiveAnalyzer(),
            'tui_interface': self._create_tui_interface()
        }

    def teardown_method(self):
        """Clean up after each test."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _create_tui_interface(self):
        """Create TUI interface with proper capabilities."""
        capabilities = Mock(spec=TerminalCapabilities)
        capabilities.console = self.console
        capabilities.width = 120
        capabilities.height = 40
        capabilities.supports_color = True
        capabilities.supports_live = True
        return RetrospectiveTUIInterface(capabilities=capabilities)

    def _create_mock_execution_data(self, task_id="test_task"):
        """Create mock task execution data."""
        return {
            'task_id': task_id,
            'agent_name': 'test_agent',
            'start_time': datetime.now() - timedelta(minutes=5),
            'end_time': datetime.now(),
            'status': 'completed',
            'metrics': {
                'execution_time': 300.0,
                'tokens_used': 1500,
                'api_calls': 12,
                'success_rate': 0.95
            },
            'outputs': ['Feature X implemented successfully', 'All tests passing'],
            'context': {
                'task_type': 'feature_implementation',
                'complexity': 'medium',
                'domain': 'backend'
            },
            'errors': [],
            'warnings': ['Deprecated API used in one location']
        }

    @pytest.mark.asyncio
    async def test_complete_retrospective_workflow(self):
        """Test the complete end-to-end retrospective workflow."""
        print("\n🔄 RETROSPECTIVE SYSTEM INTEGRATION TEST")
        print("=" * 70)
        print("Testing complete workflow from execution to improvement implementation")
        
        # Phase 1: Setup execution data
        print("\nPhase 1: Setting up execution data...")
        execution_data = self._create_mock_execution_data()
        
        # Store execution log
        log_entry = await self.components['log_store'].store_execution_log(execution_data)
        assert log_entry is not None
        print(f"✅ Execution log stored with ID: {log_entry.get('id', 'N/A')}")

        # Phase 2: Process Coach Orchestration
        print("\nPhase 2: Process coach orchestration...")
        process_coach = self.components['process_coach']
        
        # Verify process coach is active (mandatory leadership)
        assert process_coach.is_active is True
        print(f"✅ Process coach active and leading: {process_coach.is_active}")
        
        # Initiate retrospective cycle
        cycle_id = await process_coach.initiate_retrospective_cycle(
            trigger_data={
                'completed_tasks': [execution_data],
                'trigger_type': 'post_task_completion'
            }
        )
        
        assert cycle_id is not None
        print(f"✅ Retrospective cycle initiated: {cycle_id}")

        # Phase 3: Retrospective Analysis
        print("\nPhase 3: Running retrospective analysis...")
        analyzer = self.components['retrospective_analyzer']
        
        # Analyze execution patterns
        analysis_result = await analyzer.analyze_execution_logs(
            log_entries=[execution_data],
            analysis_focus=['performance', 'reliability', 'user_experience', 'maintainability']
        )
        
        assert analysis_result is not None
        assert 'patterns' in analysis_result
        assert 'opportunities' in analysis_result
        print(f"✅ Analysis completed:")
        print(f"   - Patterns detected: {len(analysis_result.get('patterns', []))}")
        print(f"   - Opportunities found: {len(analysis_result.get('opportunities', []))}")

        # Phase 4: Improvement Generation & Prioritization
        print("\nPhase 4: Generating and prioritizing improvements...")
        improvement_engine = self.components['improvement_engine']
        
        # Generate context-aware improvements
        improvements = await improvement_engine.generate_improvements(
            analysis_result=analysis_result,
            context={'agent_name': execution_data['agent_name'], 'domain': 'backend'},
            max_suggestions=8
        )
        
        assert len(improvements) > 0
        print(f"✅ Generated {len(improvements)} improvements:")
        for i, improvement in enumerate(improvements[:3], 1):  # Show first 3
            print(f"   {i}. {improvement.title} (Impact: {improvement.impact_estimate})")

        # Test different prioritization strategies
        priority_strategies = ['expected_value', 'risk_adjusted', 'quick_wins']
        for strategy in priority_strategies:
            prioritized = improvement_engine.prioritize_improvements(
                improvements, strategy=strategy
            )
            assert len(prioritized) == len(improvements)
            print(f"✅ Prioritization strategy '{strategy}' applied successfully")

        # Phase 5: Safety Validation & Rollback Preparation
        print("\nPhase 5: Safety validation and rollback preparation...")
        safety_orchestrator = self.components['safety_orchestrator']
        
        # Create safety context
        safety_context = {
            'improvements': improvements,
            'current_system_state': {
                'version': '1.2.0', 
                'health_score': 0.92,
                'performance_baseline': execution_data['metrics']
            },
            'risk_tolerance': 'medium',
            'rollback_window_hours': 24
        }
        
        # Run comprehensive safety validation
        safety_result = await safety_orchestrator.validate_improvements(
            improvements=improvements,
            context=safety_context
        )
        
        assert safety_result is not None
        assert safety_result.overall_safety_level in ['safe', 'caution', 'risky']
        print(f"✅ Safety validation completed:")
        print(f"   - Overall safety level: {safety_result.overall_safety_level}")
        print(f"   - Approved improvements: {len(safety_result.validated_improvements)}")
        
        # Create rollback point for safe recovery
        rollback_point = await safety_orchestrator.create_rollback_point(
            description=f"Pre-retrospective-cycle-{cycle_id}",
            context=safety_context
        )
        
        assert rollback_point is not None
        print(f"✅ Rollback point created: {rollback_point}")

        # Phase 6: User Approval Workflow Testing
        print("\nPhase 6: User approval workflow testing...")
        approval_workflow = self.components['approval_workflow']
        
        # Test different approval modes
        approval_modes = ['batch_approve', 'selective', 'auto_approve']
        
        for mode in approval_modes:
            print(f"   Testing approval mode: {mode}")
            
            approval_config = {
                'mode': mode,
                'timeout_seconds': 30,
                'require_justification': mode == 'selective',
                'auto_approve_threshold': 0.8 if mode == 'auto_approve' else None
            }
            
            session = await approval_workflow.start_approval_session(
                improvements=safety_result.validated_improvements,
                config=approval_config
            )
            
            assert session is not None
            
            # Simulate approval based on mode
            if mode == 'batch_approve':
                result = await approval_workflow.process_batch_approval(
                    session_id=session.session_id,
                    decision='approve_all',
                    justification='Integration test approval'
                )
                assert result.approved_count > 0
                
            elif mode == 'auto_approve':
                result = await approval_workflow.process_auto_approval(
                    session_id=session.session_id
                )
                assert result is not None
            
            print(f"   ✅ {mode} workflow completed successfully")

        # Phase 7: TUI Interface Integration
        print("\nPhase 7: TUI interface integration testing...")
        tui_interface = self.components['tui_interface']
        
        # Set comprehensive retrospective data
        retrospective_data = {
            'cycle_id': cycle_id,
            'analysis_result': analysis_result,
            'improvements': improvements,
            'safety_result': safety_result,
            'rollback_point': rollback_point,
            'approval_sessions': []  # Would contain session results
        }
        
        tui_interface.set_retrospective_data(retrospective_data)
        
        # Test all phase transitions
        phases = [
            'STARTUP', 'ANALYSIS', 'PRESENTATION', 
            'APPROVAL', 'MONITORING', 'SAFETY_CHECK', 'COMPLETED'
        ]
        
        for phase in phases:
            tui_interface.transition_to_phase(phase)
            tui_interface.update_layout()
            assert tui_interface.current_phase == phase
        
        print("✅ TUI phase transitions working correctly")

        # Phase 8: Agent Progress Monitoring Integration
        print("\nPhase 8: Agent progress monitoring...")
        
        # Simulate realistic agent progress during retrospective cycle
        progress_timeline = [
            {'agent': 'retrospective_analyzer', 'task': 'Analyzing execution patterns', 'progress': 20},
            {'agent': 'improvement_generator', 'task': 'Generating improvement suggestions', 'progress': 40},
            {'agent': 'safety_validator', 'task': 'Running safety validation checks', 'progress': 60},
            {'agent': 'approval_coordinator', 'task': 'Processing user approvals', 'progress': 80},
            {'agent': 'implementation_monitor', 'task': 'Monitoring improvement deployment', 'progress': 100}
        ]
        
        for update in progress_timeline:
            await process_coach.update_agent_progress(
                agent_name=update['agent'],
                current_task=update['task'], 
                progress_percentage=update['progress']
            )
            
            # Verify progress was recorded
            status = process_coach.get_agent_status(update['agent'])
            assert status is not None
            assert status['progress'] == update['progress']
        
        print("✅ Agent progress monitoring integrated successfully")

        # Phase 9: Complete System Validation
        print("\nPhase 9: Complete system validation...")
        
        # Verify cycle completion
        cycle_status = await process_coach.get_cycle_status(cycle_id)
        assert cycle_status is not None
        print(f"✅ Cycle status: {cycle_status.get('phase', 'unknown')}")
        
        # Verify learning outcomes
        learning_outcomes = await process_coach.get_learning_outcomes(cycle_id)
        assert learning_outcomes is not None
        assert 'improvements_identified' in learning_outcomes
        assert 'safety_measures_applied' in learning_outcomes
        
        print("✅ Learning outcomes recorded successfully")
        
        # Final Integration Validation
        print("\n🎉 RETROSPECTIVE SYSTEM INTEGRATION SUCCESS")
        print("=" * 70)
        print(f"✅ Cycle ID: {cycle_id}")
        print(f"✅ Improvements Generated: {len(improvements)}")
        print(f"✅ Safety Validated: {len(safety_result.validated_improvements)}")
        print(f"✅ Rollback Protection: Active")
        print(f"✅ TUI Integration: Complete")
        print(f"✅ Agent Monitoring: Operational")
        print(f"✅ Process Coach Leadership: Confirmed")
        print(f"✅ Multi-Mode Approvals: Tested")
        print("\n🚀 Self-improving retrospective system fully validated and operational!")

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self):
        """Test system error handling and rollback mechanisms."""
        print("\n⚠️  TESTING ERROR HANDLING AND RECOVERY")
        print("=" * 60)
        
        safety_orchestrator = self.components['safety_orchestrator']
        
        # Create rollback point
        context = {'test_mode': True, 'version': '1.0.0'}
        rollback_id = await safety_orchestrator.create_rollback_point(
            description="Pre-error-simulation",
            context=context
        )
        
        print(f"✅ Rollback point created: {rollback_id}")
        
        # Simulate critical system failure
        failure_details = {
            'type': 'critical_system_failure',
            'message': 'Simulated integration test failure',
            'timestamp': datetime.now().isoformat(),
            'severity': 'high'
        }
        
        # Test rollback mechanism
        recovery_result = await safety_orchestrator.handle_system_failure(
            error_details=failure_details,
            rollback_to=rollback_id
        )
        
        assert recovery_result is not None
        print("✅ Error handling and rollback mechanism validated")

    @pytest.mark.asyncio
    async def test_concurrent_retrospective_cycles(self):
        """Test handling of concurrent retrospective cycles."""
        print("\n🔄 TESTING CONCURRENT RETROSPECTIVE CYCLES")
        print("=" * 60)
        
        process_coach = self.components['process_coach']
        
        # Create multiple concurrent execution scenarios
        concurrent_tasks = []
        for i in range(3):
            execution_data = self._create_mock_execution_data(f'concurrent_task_{i}')
            task = asyncio.create_task(
                process_coach.initiate_retrospective_cycle(
                    trigger_data={'completed_tasks': [execution_data]}
                )
            )
            concurrent_tasks.append(task)
        
        # Wait for all cycles to complete
        cycle_ids = await asyncio.gather(*concurrent_tasks)
        
        # Verify all cycles were created successfully
        assert len(cycle_ids) == 3
        assert all(cycle_id is not None for cycle_id in cycle_ids)
        assert len(set(cycle_ids)) == 3  # All unique
        
        print(f"✅ Successfully handled {len(cycle_ids)} concurrent retrospective cycles")

    def test_system_configuration_and_customization(self):
        """Test system configuration capabilities."""
        print("\n⚙️  TESTING SYSTEM CONFIGURATION")
        print("=" * 50)
        
        process_coach = self.components['process_coach']
        
        # Test configuration updates
        config_updates = {
            'retrospective_frequency': 'after_each_task',
            'safety_validation_level': 'strict',
            'auto_approval_threshold': 0.85,
            'rollback_retention_days': 30,
            'concurrent_cycle_limit': 5,
            'improvement_batch_size': 10
        }
        
        # Apply configuration
        process_coach.update_configuration(config_updates)
        
        # Verify configuration was applied
        current_config = process_coach.get_configuration()
        for key, expected_value in config_updates.items():
            actual_value = current_config.get(key)
            assert actual_value == expected_value, f"Config {key}: expected {expected_value}, got {actual_value}"
        
        print("✅ System configuration successfully updated and validated")


async def run_integration_test():
    """Run the comprehensive retrospective system integration test."""
    print("🧪 RETROSPECTIVE SYSTEM INTEGRATION TEST")
    print("=" * 80)
    print("Validating complete self-improving retrospective workflow")
    print("Components: Process Coach | Safety | Approval | TUI | Analysis | Generation")
    print("=" * 80)
    
    try:
        # Create and run test instance
        test_instance = TestRetrospectiveSystemIntegration()
        test_instance.setup_method()
        
        try:
            # Run comprehensive workflow test
            await test_instance.test_complete_retrospective_workflow()
            
            # Run error handling test
            await test_instance.test_error_handling_and_recovery()
            
            # Run concurrent cycles test
            await test_instance.test_concurrent_retrospective_cycles()
            
            # Run configuration test
            test_instance.test_system_configuration_and_customization()
            
            print("\n✅ ALL INTEGRATION TESTS PASSED")
            print("🚀 Retrospective system fully validated and production-ready!")
            return True
            
        finally:
            test_instance.teardown_method()
            
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    """Run the integration test directly."""
    success = asyncio.run(run_integration_test())
    sys.exit(0 if success else 1)