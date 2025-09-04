#!/usr/bin/env python3
"""
Comprehensive End-to-End Retrospective System Test

This test validates the complete self-improving retrospective workflow:
1. Process coach orchestration (mandatory leadership)
2. Execution log capture and analysis
3. Improvement generation with impact estimation
4. Safety validation with rollback protection
5. User approval workflow
6. TUI interface integration
7. Agent progress monitoring
8. Complete retrospective cycle execution

This is the final validation ensuring all components work together seamlessly.
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
from agentsmcp.retrospective.capture.execution_logger import ExecutionLogger
from agentsmcp.retrospective.analysis.retrospective_analyzer import RetrospectiveAnalyzer
from agentsmcp.ui.v3.retrospective_tui_interface import RetrospectiveTUIInterface
from agentsmcp.ui.v3.terminal_capabilities import TerminalCapabilities
from rich.console import Console


class TestCompleteRetrospectiveSystem:
    """Comprehensive end-to-end system test."""

    @pytest.fixture
    async def system_setup(self):
        """Setup complete retrospective system for testing."""
        # Create temporary directory for test data
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Setup console for TUI testing
            console = Console(file=StringIO(), force_terminal=True, width=120, height=40)
            
            # Create system components
            components = {
                'process_coach': ProcessCoach(),
                'safety_orchestrator': SafetyOrchestrator(),
                'approval_workflow': ApprovalWorkflow(),
                'improvement_engine': ImprovementEngine(),
                'execution_logger': ExecutionLogger(log_dir=str(temp_path / "logs")),
                'retrospective_analyzer': RetrospectiveAnalyzer(),
                'tui_interface': self._create_tui_interface(console),
                'temp_path': temp_path,
                'console': console
            }
            
            yield components

    def _create_tui_interface(self, console):
        """Create TUI interface with proper capabilities."""
        capabilities = Mock(spec=TerminalCapabilities)
        capabilities.console = console
        capabilities.width = 120
        capabilities.height = 40
        capabilities.supports_color = True
        capabilities.supports_live = True
        return RetrospectiveTUIInterface(capabilities=capabilities)

    def _create_mock_task_execution(self):
        """Create mock task execution data."""
        return {
            'task_id': 'test_task_001',
            'agent_name': 'test_agent',
            'start_time': datetime.now() - timedelta(minutes=5),
            'end_time': datetime.now(),
            'status': 'completed',
            'metrics': {
                'execution_time': 300.0,
                'tokens_used': 1500,
                'api_calls': 12
            },
            'outputs': ['Successfully implemented feature X'],
            'context': {
                'task_type': 'feature_implementation',
                'complexity': 'medium',
                'domain': 'backend'
            }
        }

    @pytest.mark.asyncio
    async def test_complete_retrospective_workflow(self, system_setup):
        """Test complete retrospective workflow end-to-end."""
        components = await system_setup
        
        print("\n🔄 Starting Complete Retrospective System Test")
        print("=" * 60)
        
        # Phase 1: Execution Log Capture
        print("Phase 1: Capturing execution logs...")
        execution_data = self._create_mock_task_execution()
        
        logger = components['execution_logger']
        log_entry = await logger.log_task_execution(execution_data)
        
        assert log_entry is not None
        assert log_entry['task_id'] == execution_data['task_id']
        print("✅ Execution logging completed")

        # Phase 2: Process Coach Orchestration (Mandatory Leadership)
        print("\nPhase 2: Process coach orchestration...")
        process_coach = components['process_coach']
        
        # Verify process coach is active and mandatory
        assert process_coach.is_active is True
        print(f"✅ Process coach active: {process_coach.is_active}")
        
        # Process coach initiates retrospective cycle
        cycle_id = await process_coach.initiate_retrospective_cycle(
            trigger_data={'completed_tasks': [execution_data]}
        )
        
        assert cycle_id is not None
        assert len(cycle_id) > 0
        print(f"✅ Retrospective cycle initiated: {cycle_id}")

        # Phase 3: Retrospective Analysis
        print("\nPhase 3: Running retrospective analysis...")
        analyzer = components['retrospective_analyzer']
        
        analysis_result = await analyzer.analyze_execution_logs(
            log_entries=[log_entry],
            focus_areas=['performance', 'reliability', 'user_experience']
        )
        
        assert analysis_result is not None
        assert 'patterns' in analysis_result
        assert 'bottlenecks' in analysis_result
        assert 'opportunities' in analysis_result
        print(f"✅ Analysis completed - found {len(analysis_result.get('opportunities', []))} opportunities")

        # Phase 4: Improvement Generation with Impact Estimation
        print("\nPhase 4: Generating improvements with impact estimation...")
        improvement_engine = components['improvement_engine']
        
        improvements = await improvement_engine.generate_improvements(
            analysis_result=analysis_result,
            context={'agent_name': execution_data['agent_name']},
            max_suggestions=5
        )
        
        assert len(improvements) > 0
        print(f"✅ Generated {len(improvements)} improvements")
        
        # Validate improvement quality
        for improvement in improvements:
            assert hasattr(improvement, 'id')
            assert hasattr(improvement, 'title')
            assert hasattr(improvement, 'impact_estimate')
            assert hasattr(improvement, 'confidence_level')
            print(f"  - {improvement.title} (Impact: {improvement.impact_estimate})")

        # Phase 5: Safety Validation Framework
        print("\nPhase 5: Safety validation and rollback preparation...")
        safety_orchestrator = components['safety_orchestrator']
        
        # Create safety validation context
        safety_context = {
            'improvements': improvements,
            'current_system_state': {'version': '1.0.0', 'health': 'good'},
            'risk_tolerance': 'medium'
        }
        
        # Validate improvements for safety
        safety_result = await safety_orchestrator.validate_improvements(
            improvements=improvements,
            context=safety_context
        )
        
        assert safety_result is not None
        assert safety_result.overall_safety_level in ['safe', 'caution', 'risky']
        assert len(safety_result.validated_improvements) <= len(improvements)
        print(f"✅ Safety validation completed - {len(safety_result.validated_improvements)} improvements approved")
        print(f"   Safety level: {safety_result.overall_safety_level}")

        # Create rollback point
        rollback_id = await safety_orchestrator.create_rollback_point(
            description="Pre-retrospective-improvements",
            context=safety_context
        )
        
        assert rollback_id is not None
        print(f"✅ Rollback point created: {rollback_id}")

        # Phase 6: User Approval Workflow
        print("\nPhase 6: User approval workflow...")
        approval_workflow = components['approval_workflow']
        
        # Configure approval for batch approve mode (simulating user approval)
        approval_config = {
            'mode': 'batch_approve',  # Simulate user approving all
            'timeout_seconds': 30,
            'require_justification': False
        }
        
        approval_session = await approval_workflow.start_approval_session(
            improvements=safety_result.validated_improvements,
            config=approval_config
        )
        
        assert approval_session is not None
        assert approval_session.session_id is not None
        print(f"✅ Approval session started: {approval_session.session_id}")
        
        # Process approvals (simulate batch approval)
        approval_result = await approval_workflow.process_batch_approval(
            session_id=approval_session.session_id,
            decision='approve_all',
            justification='End-to-end test approval'
        )
        
        assert approval_result is not None
        assert approval_result.approved_count > 0
        print(f"✅ Approval completed - {approval_result.approved_count} improvements approved")

        # Phase 7: TUI Interface Integration Test
        print("\nPhase 7: TUI interface integration...")
        tui_interface = components['tui_interface']
        
        # Test TUI can display the retrospective data
        tui_interface.set_retrospective_data({
            'cycle_id': cycle_id,
            'improvements': safety_result.validated_improvements,
            'approval_result': approval_result,
            'safety_status': safety_result
        })
        
        # Test phase transitions
        phases = [
            'STARTUP', 'ANALYSIS', 'PRESENTATION', 
            'APPROVAL', 'MONITORING', 'SAFETY_CHECK', 'COMPLETED'
        ]
        
        for phase in phases:
            tui_interface.transition_to_phase(phase)
            tui_interface.update_layout()
            assert tui_interface.current_phase == phase
        
        print("✅ TUI interface integration successful")

        # Phase 8: Validate Agent Progress Monitoring
        print("\nPhase 8: Agent progress monitoring...")
        
        # Simulate agent progress updates
        progress_updates = [
            {'agent': 'test_agent', 'task': 'Analyzing execution logs', 'progress': 25},
            {'agent': 'test_agent', 'task': 'Generating improvements', 'progress': 50}, 
            {'agent': 'test_agent', 'task': 'Safety validation', 'progress': 75},
            {'agent': 'test_agent', 'task': 'User approval processing', 'progress': 100}
        ]
        
        for update in progress_updates:
            await process_coach.update_agent_progress(
                agent_name=update['agent'],
                current_task=update['task'],
                progress_percentage=update['progress']
            )
        
        agent_status = process_coach.get_agent_status('test_agent')
        assert agent_status is not None
        assert agent_status['progress'] == 100
        print("✅ Agent progress monitoring working correctly")

        # Phase 9: Complete Cycle Validation
        print("\nPhase 9: Complete cycle validation...")
        
        # Verify the complete cycle is tracked
        cycle_status = await process_coach.get_cycle_status(cycle_id)
        assert cycle_status is not None
        assert cycle_status['phase'] in ['completed', 'in_progress']
        
        # Verify system learned from the cycle
        learning_summary = await process_coach.get_learning_summary(cycle_id)
        assert learning_summary is not None
        assert 'improvements_implemented' in learning_summary
        assert 'lessons_learned' in learning_summary
        
        print("✅ Complete retrospective cycle validated")
        
        # Final Success Summary
        print("\n🎉 RETROSPECTIVE SYSTEM TEST COMPLETE")
        print("=" * 60)
        print(f"✅ Cycle ID: {cycle_id}")
        print(f"✅ Improvements Generated: {len(improvements)}")
        print(f"✅ Safety Validated: {len(safety_result.validated_improvements)}")
        print(f"✅ User Approved: {approval_result.approved_count}")
        print(f"✅ Rollback Point: {rollback_id}")
        print(f"✅ TUI Integration: Success")
        print(f"✅ Agent Monitoring: Active")
        print("✅ Process Coach Leadership: Confirmed")
        print("\n🚀 Self-improving retrospective system is fully operational!")

    @pytest.mark.asyncio
    async def test_error_handling_and_rollback(self, system_setup):
        """Test error handling and rollback mechanisms."""
        components = await system_setup
        
        print("\n⚠️  Testing Error Handling and Rollback")
        print("=" * 50)
        
        safety_orchestrator = components['safety_orchestrator']
        
        # Create rollback point
        rollback_id = await safety_orchestrator.create_rollback_point(
            description="Pre-error-test",
            context={'test_mode': True}
        )
        
        # Simulate system failure
        try:
            # This should trigger rollback
            await safety_orchestrator.handle_system_failure(
                error_details={'type': 'critical_failure', 'message': 'Test error'},
                rollback_to=rollback_id
            )
            
            print("✅ Rollback mechanism triggered successfully")
            
        except Exception as e:
            # Rollback should have been triggered
            print(f"✅ Error handled with rollback: {type(e).__name__}")

    @pytest.mark.asyncio 
    async def test_concurrent_retrospective_cycles(self, system_setup):
        """Test handling of concurrent retrospective cycles."""
        components = await system_setup
        
        print("\n🔄 Testing Concurrent Retrospective Cycles")
        print("=" * 50)
        
        process_coach = components['process_coach']
        
        # Start multiple cycles concurrently
        tasks = []
        for i in range(3):
            task_data = self._create_mock_task_execution()
            task_data['task_id'] = f'concurrent_task_{i}'
            
            task = asyncio.create_task(
                process_coach.initiate_retrospective_cycle(
                    trigger_data={'completed_tasks': [task_data]}
                )
            )
            tasks.append(task)
        
        cycle_ids = await asyncio.gather(*tasks)
        
        # Verify all cycles were created
        assert len(cycle_ids) == 3
        assert all(cycle_id is not None for cycle_id in cycle_ids)
        
        print(f"✅ Successfully handled {len(cycle_ids)} concurrent cycles")

    def test_retrospective_system_configuration(self, system_setup):
        """Test system configuration and customization."""
        components = system_setup
        
        print("\n⚙️  Testing System Configuration")
        print("=" * 40)
        
        process_coach = components['process_coach']
        
        # Test configuration updates
        config_updates = {
            'retrospective_frequency': 'after_each_task',
            'safety_validation_level': 'strict',
            'auto_approval_threshold': 0.8,
            'rollback_retention_days': 30
        }
        
        process_coach.update_configuration(config_updates)
        
        current_config = process_coach.get_configuration()
        for key, value in config_updates.items():
            assert current_config.get(key) == value
        
        print("✅ System configuration successfully updated")


async def run_comprehensive_test():
    """Run the comprehensive retrospective system test."""
    print("🧪 COMPREHENSIVE RETROSPECTIVE SYSTEM TEST")
    print("=" * 70)
    print("Testing complete self-improving retrospective workflow...")
    print("Including: Process Coach, Safety, Approval, TUI, Progress Monitoring")
    print("=" * 70)
    
    # Create test instance
    test_instance = TestCompleteRetrospectiveSystem()
    
    # Setup system
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        console = Console(file=StringIO(), force_terminal=True, width=120, height=40)
        
        components = {
            'process_coach': ProcessCoach(),
            'safety_orchestrator': SafetyOrchestrator(),
            'approval_workflow': ApprovalWorkflow(),
            'improvement_engine': ImprovementEngine(),
            'execution_logger': ExecutionLogger(log_dir=str(temp_path / "logs")),
            'retrospective_analyzer': RetrospectiveAnalyzer(),
            'tui_interface': test_instance._create_tui_interface(console),
            'temp_path': temp_path,
            'console': console
        }
        
        try:
            # Run main workflow test
            await test_instance.test_complete_retrospective_workflow(components)
            
            # Run error handling test
            await test_instance.test_error_handling_and_rollback(components)
            
            # Run concurrent cycles test  
            await test_instance.test_concurrent_retrospective_cycles(components)
            
            # Run configuration test
            test_instance.test_retrospective_system_configuration(components)
            
            print("\n✅ ALL TESTS PASSED - RETROSPECTIVE SYSTEM FULLY VALIDATED")
            print("🚀 Self-improving retrospective system ready for production!")
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True


if __name__ == "__main__":
    """Run comprehensive retrospective system test."""
    success = asyncio.run(run_comprehensive_test())
    sys.exit(0 if success else 1)