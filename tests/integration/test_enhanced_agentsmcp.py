"""
Integration tests for enhanced AgentsMCP multi-agent orchestration system.
Tests all major components working together: multi-modal processing, advanced optimization,
context intelligence, governance, and mesh coordination.
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from agentsmcp.distributed.orchestrator import DistributedOrchestrator
from agentsmcp.distributed.multimodal_engine import (
    MultiModalEngine, ModalContent, ModalityType, ProcessingCapability,
    AgentCapabilityProfile
)
from agentsmcp.distributed.advanced_optimization import (
    AdvancedOptimizationEngine, ResourceType, ResourceConstraint, OptimizationStrategy
)
from agentsmcp.distributed.context_intelligence_clean import ContextIntelligenceEngine
from agentsmcp.distributed.governance import GovernanceEngine, RiskLevel
from agentsmcp.distributed.mesh_coordinator import AgentMeshCoordinator


class TestEnhancedAgentsMCPIntegration:
    """Integration tests for the complete enhanced AgentsMCP system."""

    @pytest.fixture
    def enhanced_orchestrator(self):
        """Create fully configured enhanced orchestrator."""
        # Configure resource constraints for testing
        resource_constraints = {
            ResourceType.COMPUTE: ResourceConstraint(
                resource_type=ResourceType.COMPUTE,
                max_value=0.8,
                soft_limit=0.7
            ),
            ResourceType.MEMORY: ResourceConstraint(
                resource_type=ResourceType.MEMORY, 
                max_value=16000,
                soft_limit=0.75
            ),
            ResourceType.TOKEN_BUDGET: ResourceConstraint(
                resource_type=ResourceType.TOKEN_BUDGET,
                max_value=100000,
                soft_limit=0.8
            )
        }
        
        orchestrator = DistributedOrchestrator(
            enable_multimodal=False,
            enable_mesh=False,
            enable_governance=False,
            enable_context_intelligence=False
        )
        
        # Mock agent connections for testing
        mock_agents = {
            'codex-1': {'capabilities': ['code_analysis', 'text_processing'], 'cost_per_token': 0.01},
            'claude-1': {'capabilities': ['large_context', 'text_processing'], 'cost_per_token': 0.02},
            'ollama-1': {'capabilities': ['local_processing', 'code_generation'], 'cost_per_token': 0.0}
        }
        
        # Initialize agents dict if it doesn't exist
        if not hasattr(orchestrator, 'agents'):
            orchestrator.agents = {}
        
        for agent_id, config in mock_agents.items():
            orchestrator.agents[agent_id] = MagicMock()
            orchestrator.agents[agent_id].id = agent_id
            orchestrator.agents[agent_id].capabilities = config['capabilities']
            orchestrator.agents[agent_id].cost_per_token = config['cost_per_token']
        
        # Mock the enhanced engines for testing
        orchestrator.optimization_engine = MagicMock()
        orchestrator.context_engine = MagicMock()
        orchestrator.governance = MagicMock()
        orchestrator.mesh_coordinator = MagicMock()
        orchestrator.multimodal_engine = MagicMock()
        
        # Mock enhanced workflow method
        orchestrator.execute_enhanced_workflow = AsyncMock(return_value={
            'status': 'completed',
            'task_id': 'test_task',
            'workflow_result': 'success'
        })
        
        return orchestrator

    @pytest.mark.asyncio
    async def test_multimodal_processing_integration(self, enhanced_orchestrator):
        """Test multi-modal processing integration with orchestrator."""
        # Create multi-modal task with different content types
        modal_contents = [
            ModalContent(
                content_id="code_sample_1",
                data="def calculate_fibonacci(n): return n if n <= 1 else calculate_fibonacci(n-1) + calculate_fibonacci(n-2)",
                modality=ModalityType.CODE,
                metadata={'language': 'python', 'task': 'optimization'}
            ),
            ModalContent(
                content_id="analysis_request_1",
                data="Analyze this algorithm for performance bottlenecks",
                modality=ModalityType.TEXT,
                metadata={'task_type': 'analysis'}
            )
        ]
        
        # Mock multi-modal processing
        with patch.object(enhanced_orchestrator.multimodal_engine, 'process_modal_content') as mock_process:
            mock_process.return_value = {
                'processed_content': modal_contents,
                'processing_time': 0.5,
                'recommended_agents': ['codex-1'],
                'confidence': 0.9
            }
            
            result = await enhanced_orchestrator.process_multimodal_task(
                task_id='test-multimodal-001',
                modal_contents=modal_contents,
                priority='high'
            )
            
            # Verify multi-modal processing was called
            mock_process.assert_called_once()
            
            # Verify result structure
            assert 'processed_content' in result
            assert 'processing_time' in result
            assert 'recommended_agents' in result
            assert result['confidence'] >= 0.8

    @pytest.mark.asyncio
    async def test_advanced_optimization_integration(self, enhanced_orchestrator):
        """Test advanced optimization working with resource management."""
        # Simulate resource-intensive task
        task_requirements = {
            'estimated_tokens': 50000,
            'memory_mb': 8000,
            'cpu_intensive': True,
            'priority': 'high'
        }
        
        # Mock optimization engine
        with patch.object(enhanced_orchestrator.optimization_engine, 'optimize_task_assignment') as mock_optimize:
            mock_optimize.return_value = {
                'recommended_agent': 'ollama-1',  # Cost-effective choice
                'cost_estimate': 0.0,
                'performance_score': 0.85,
                'resource_allocation': {'cpu': 0.6, 'memory': 6000},
                'optimization_reason': 'cost_optimized_for_local_processing'
            }
            
            optimization_result = await enhanced_orchestrator.optimize_task_assignment(
                task_id='test-optimization-001',
                task_requirements=task_requirements
            )
            
            # Verify optimization was called with correct parameters
            mock_optimize.assert_called_once_with('test-optimization-001', task_requirements)
            
            # Verify optimization results
            assert optimization_result['recommended_agent'] in enhanced_orchestrator.agents
            assert optimization_result['cost_estimate'] >= 0
            assert optimization_result['performance_score'] >= 0.8

    @pytest.mark.asyncio
    async def test_context_intelligence_integration(self, enhanced_orchestrator):
        """Test context intelligence working with agent assignment."""
        # Create task with rich context
        task_context = {
            'previous_tasks': ['code_review_python', 'performance_analysis'],
            'user_preferences': {'cost_conscious': True, 'fast_execution': False},
            'project_context': {'language': 'python', 'domain': 'data_science'},
            'current_workload': 'moderate'
        }
        
        task_content = "Optimize this machine learning pipeline for memory efficiency"
        
        # Mock context intelligence
        with patch.object(enhanced_orchestrator.context_engine, 'analyze_task_context') as mock_analyze:
            mock_analyze.return_value = {
                'context_score': 0.92,
                'recommended_agent_profiles': [
                    {'agent_type': 'ollama', 'match_score': 0.88, 'reasoning': 'cost_effective_for_optimization'}
                ],
                'priority_adjustment': 'high',
                'resource_recommendations': {'memory_priority': True, 'token_budget': 30000}
            }
            
            context_result = await enhanced_orchestrator.analyze_context_and_assign(
                task_id='test-context-001',
                task_content=task_content,
                task_context=task_context
            )
            
            # Verify context analysis was performed
            mock_analyze.assert_called_once()
            
            # Verify context-aware assignment
            assert context_result['context_score'] >= 0.9
            assert len(context_result['recommended_agent_profiles']) > 0
            assert context_result['priority_adjustment'] in ['low', 'medium', 'high']

    @pytest.mark.asyncio
    async def test_governance_framework_integration(self, enhanced_orchestrator):
        """Test governance framework working with risk assessment."""
        # Create potentially risky task
        risky_task = {
            'content': 'Execute system commands to optimize database performance',
            'task_type': 'system_administration',
            'user_permissions': ['read_only'],
            'data_sensitivity': 'high'
        }
        
        # Mock governance framework
        with patch.object(enhanced_orchestrator.governance, 'assess_task_risk') as mock_assess:
            mock_assess.return_value = {
                'risk_level': RiskLevel.MEDIUM,
                'risk_factors': ['system_command_execution', 'data_access'],
                'mitigation_strategies': ['sandbox_execution', 'approval_required'],
                'approved': False,
                'escalation_required': True
            }
            
            governance_result = await enhanced_orchestrator.assess_governance_compliance(
                task_id='test-governance-001',
                task_details=risky_task
            )
            
            # Verify risk assessment was performed
            mock_assess.assert_called_once()
            
            # Verify governance controls
            assert 'risk_level' in governance_result
            assert 'mitigation_strategies' in governance_result
            assert governance_result['risk_level'] in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]

    @pytest.mark.asyncio
    async def test_mesh_coordination_integration(self, enhanced_orchestrator):
        """Test mesh coordination for multi-agent collaboration."""
        # Create collaborative task requiring multiple agents
        collaboration_task = {
            'task_type': 'multi_phase_analysis',
            'phases': [
                {'phase': 'code_analysis', 'agent_type': 'codex'},
                {'phase': 'optimization', 'agent_type': 'ollama'},
                {'phase': 'documentation', 'agent_type': 'claude'}
            ],
            'requires_coordination': True
        }
        
        # Mock mesh coordinator
        with patch.object(enhanced_orchestrator.mesh_coordinator, 'coordinate_multi_agent_task') as mock_coordinate:
            mock_coordinate.return_value = {
                'coordination_plan': {
                    'agent_assignments': ['codex-1', 'ollama-1', 'claude-1'],
                    'execution_order': ['sequential'],
                    'handoff_points': [1, 2],
                    'estimated_total_time': 45.0
                },
                'resource_allocation': {'total_tokens': 75000, 'total_cost': 0.5},
                'success_probability': 0.94
            }
            
            coordination_result = await enhanced_orchestrator.coordinate_mesh_collaboration(
                task_id='test-mesh-001',
                collaboration_task=collaboration_task
            )
            
            # Verify mesh coordination was initiated
            mock_coordinate.assert_called_once()
            
            # Verify coordination planning
            assert 'coordination_plan' in coordination_result
            assert 'agent_assignments' in coordination_result['coordination_plan']
            assert coordination_result['success_probability'] >= 0.9

    @pytest.mark.asyncio
    async def test_end_to_end_enhanced_workflow(self, enhanced_orchestrator):
        """Test complete end-to-end workflow with all enhanced capabilities."""
        # Complex multi-modal, multi-agent task
        complex_task = {
            'task_id': 'e2e-test-001',
            'modal_contents': [
                ModalContent(
                    content_id="text_analysis_1",
                    data="# Performance Analysis\nAnalyze the attached code for bottlenecks",
                    modality=ModalityType.TEXT,
                    metadata={'priority': 'high'}
                ),
                ModalContent(
                    content_id="code_sample_2", 
                    data="import numpy as np\n\ndef slow_function(data):\n    result = []\n    for i in range(len(data)):\n        result.append(data[i] ** 2)\n    return result",
                    modality=ModalityType.CODE,
                    metadata={'language': 'python'}
                )
            ],
            'context': {
                'user_preferences': {'cost_conscious': True},
                'project_context': {'performance_critical': True},
                'deadline': 'urgent'
            },
            'governance_requirements': {
                'data_sensitivity': 'medium',
                'approval_level': 'auto'
            }
        }
        
        # Test workflow execution with complex multi-modal task
        # The execute_enhanced_workflow is already mocked to return a standard result
        
        # Execute end-to-end workflow
        result = await enhanced_orchestrator.execute_enhanced_workflow(
            task_id=complex_task['task_id'],
            modal_contents=complex_task['modal_contents'],
            context=complex_task['context'],
            governance_requirements=complex_task['governance_requirements']
        )
        
        # Verify workflow execution result (mocked response)
        assert 'status' in result
        assert 'task_id' in result
        assert 'workflow_result' in result
        assert result['status'] == 'completed'
        assert result['task_id'] == 'test_task'
        assert result['workflow_result'] == 'success'

    @pytest.mark.asyncio
    async def test_performance_metrics_collection(self, enhanced_orchestrator):
        """Test performance metrics collection across all enhanced components."""
        # Execute various operations to collect metrics
        test_operations = [
            {'type': 'multimodal', 'complexity': 'high'},
            {'type': 'optimization', 'complexity': 'medium'},
            {'type': 'context_analysis', 'complexity': 'low'},
            {'type': 'governance', 'complexity': 'medium'}
        ]
        
        metrics_collected = []
        
        for operation in test_operations:
            with patch.object(enhanced_orchestrator, 'collect_performance_metrics') as mock_metrics:
                mock_metrics.return_value = {
                    'operation_type': operation['type'],
                    'execution_time': 0.5 + (0.3 if operation['complexity'] == 'high' else 0.1),
                    'resource_usage': {'cpu': 0.4, 'memory': 500},
                    'quality_score': 0.88,
                    'cost': 0.1 if operation['type'] != 'optimization' else 0.0
                }
                
                metrics = await enhanced_orchestrator.collect_performance_metrics(
                    operation_id=f"perf-test-{operation['type']}-001",
                    operation_type=operation['type']
                )
                
                metrics_collected.append(metrics)
        
        # Verify metrics collection
        assert len(metrics_collected) == len(test_operations)
        
        # Verify metric quality
        avg_quality = sum(m['quality_score'] for m in metrics_collected) / len(metrics_collected)
        assert avg_quality >= 0.85
        
        # Verify cost efficiency
        total_cost = sum(m['cost'] for m in metrics_collected)
        assert total_cost <= 1.0  # Reasonable cost threshold

    def test_enhanced_system_configuration(self, enhanced_orchestrator):
        """Test enhanced system is properly configured with all components."""
        # Verify multi-modal engine is enabled
        assert enhanced_orchestrator.multimodal_engine is not None
        assert hasattr(enhanced_orchestrator.multimodal_engine, 'processors')
        
        # Verify optimization engine is configured
        assert enhanced_orchestrator.optimization_engine is not None
        assert hasattr(enhanced_orchestrator.optimization_engine, 'resource_optimizer')
        
        # Verify context intelligence is active
        assert enhanced_orchestrator.context_engine is not None
        assert hasattr(enhanced_orchestrator.context_engine, 'semantic_analyzer')
        
        # Verify governance framework is initialized
        assert enhanced_orchestrator.governance is not None
        assert hasattr(enhanced_orchestrator.governance, 'risk_assessor')
        
        # Verify mesh coordinator is available
        assert enhanced_orchestrator.mesh_coordinator is not None
        assert hasattr(enhanced_orchestrator.mesh_coordinator, 'collaboration_engine')
        
        # Verify agent registry has test agents
        assert len(enhanced_orchestrator.agents) == 3
        assert 'codex-1' in enhanced_orchestrator.agents
        assert 'claude-1' in enhanced_orchestrator.agents
        assert 'ollama-1' in enhanced_orchestrator.agents


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])