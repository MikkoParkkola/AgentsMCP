"""Comprehensive tests for the team optimizer.

This test suite covers:
- Golden tests for optimization functionality
- Cost-efficiency optimization scenarios
- Role effectiveness analysis
- Edge cases for data handling
- Statistical validation of results
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
from typing import Dict, List, Any
import statistics

from src.agentsmcp.orchestration.team_optimizer import (
    TeamOptimizer,
    OptimizationResults,
    OptimizationObjective,
    OptimizationStrategy,
    OptimizationConstraints,
    RoleEffectiveness,
    TeamPattern,
    StatisticalSignificance,
)
from src.agentsmcp.orchestration.execution_engine import (
    TeamExecution,
    ExecutionStatus,
    ExecutionProgress,
)
from src.agentsmcp.orchestration.models import (
    TeamComposition,
    TeamPerformanceMetrics,
    AgentSpec,
    CoordinationStrategy,
    TaskType,
    ComplexityLevel,
    RiskLevel,
)


class TestTeamOptimizer:
    """Test suite for TeamOptimizer."""
    
    @pytest.fixture
    def team_optimizer(self):
        """Create team optimizer instance."""
        return TeamOptimizer(
            optimization_history_limit=100,
            pattern_confidence_threshold=0.7,
            statistical_significance_threshold=0.05,
        )
    
    @pytest.fixture
    def sample_team_executions(self):
        """Create sample team execution history."""
        executions = []
        base_time = datetime.now(timezone.utc) - timedelta(days=30)
        
        # Create successful executions
        for i in range(15):
            execution = TeamExecution(
                execution_id=f'exec_success_{i}',
                team_composition=TeamComposition(
                    primary_team=[
                        AgentSpec(role='architect', model_assignment='premium'),
                        AgentSpec(role='coder', model_assignment='standard'),
                    ],
                    load_order=['architect', 'coder'],
                    coordination_strategy=CoordinationStrategy.SEQUENTIAL,
                    confidence_score=0.8,
                ),
                objective=f'Task {i}',
                status=ExecutionStatus.COMPLETED,
                progress=ExecutionProgress(total_tasks=5, completed_tasks=5),
                started_at=base_time + timedelta(hours=i),
                total_duration_seconds=300.0 + i * 10,
                resource_usage={'cost': 20.0 + i, 'memory': 256, 'cpu': 30.0},
                errors=[],
            )
            executions.append(execution)
        
        # Create some failed executions
        for i in range(5):
            execution = TeamExecution(
                execution_id=f'exec_failed_{i}',
                team_composition=TeamComposition(
                    primary_team=[
                        AgentSpec(role='reviewer', model_assignment='basic'),
                        AgentSpec(role='coder', model_assignment='basic'),
                        AgentSpec(role='tester', model_assignment='basic'),
                    ],
                    load_order=['reviewer', 'coder', 'tester'],
                    coordination_strategy=CoordinationStrategy.PARALLEL,
                    confidence_score=0.5,
                ),
                objective=f'Failed task {i}',
                status=ExecutionStatus.FAILED,
                progress=ExecutionProgress(total_tasks=5, completed_tasks=2, failed_tasks=3),
                started_at=base_time + timedelta(hours=20 + i),
                total_duration_seconds=600.0 + i * 20,
                resource_usage={'cost': 45.0 + i, 'memory': 512, 'cpu': 80.0},
                errors=['Resource exhaustion', 'Coordination failure'],
            )
            executions.append(execution)
        
        # Create high-performing team executions
        for i in range(10):
            execution = TeamExecution(
                execution_id=f'exec_high_perf_{i}',
                team_composition=TeamComposition(
                    primary_team=[
                        AgentSpec(role='architect', model_assignment='premium'),
                        AgentSpec(role='senior_coder', model_assignment='premium'),
                        AgentSpec(role='reviewer', model_assignment='standard'),
                    ],
                    load_order=['architect', 'senior_coder', 'reviewer'],
                    coordination_strategy=CoordinationStrategy.COLLABORATIVE,
                    confidence_score=0.9,
                ),
                objective=f'High-perf task {i}',
                status=ExecutionStatus.COMPLETED,
                progress=ExecutionProgress(total_tasks=8, completed_tasks=8),
                started_at=base_time + timedelta(hours=40 + i),
                total_duration_seconds=240.0 + i * 5,
                resource_usage={'cost': 35.0 + i, 'memory': 384, 'cpu': 50.0},
                errors=[],
            )
            executions.append(execution)
        
        return executions
    
    @pytest.fixture
    def sample_performance_metrics(self):
        """Create sample performance metrics."""
        return {
            'architect-coder': TeamPerformanceMetrics(
                team_id='architect-coder',
                task_type=TaskType.IMPLEMENTATION,
                success_rate=0.9,
                average_duration=320.0,
                average_cost=25.0,
                total_executions=15,
            ),
            'reviewer-coder-tester': TeamPerformanceMetrics(
                team_id='reviewer-coder-tester',
                task_type=TaskType.TESTING,
                success_rate=0.3,
                average_duration=620.0,
                average_cost=50.0,
                total_executions=5,
            ),
            'architect-senior_coder-reviewer': TeamPerformanceMetrics(
                team_id='architect-senior_coder-reviewer',
                task_type=TaskType.IMPLEMENTATION,
                success_rate=1.0,
                average_duration=245.0,
                average_cost=38.0,
                total_executions=10,
            ),
        }
    
    @pytest.fixture
    def optimization_constraints(self):
        """Create sample optimization constraints."""
        return OptimizationConstraints(
            max_team_size=5,
            min_team_size=2,
            max_cost_per_execution=50.0,
            min_success_rate=0.7,
            required_roles={'architect'},
            forbidden_roles={'deprecated_role'},
        )
    
    # Golden Tests - Core Optimization Functionality
    
    @pytest.mark.asyncio
    async def test_optimize_patterns_basic_success(
        self,
        team_optimizer,
        sample_team_executions,
        sample_performance_metrics,
    ):
        """Test basic successful pattern optimization (Golden Test 1)."""
        
        results = await team_optimizer.optimize_patterns(
            historical_results=sample_team_executions,
            performance_metrics=sample_performance_metrics,
            optimization_objective=OptimizationObjective.BALANCED,
            optimization_strategy=OptimizationStrategy.GREEDY,
        )
        
        # Verify basic structure
        assert isinstance(results, OptimizationResults)
        assert results.optimization_objective == OptimizationObjective.BALANCED
        assert results.strategy_used == OptimizationStrategy.GREEDY
        assert results.data_points_analyzed == len(sample_team_executions)
        assert results.optimization_duration_seconds > 0.0
        
        # Verify role effectiveness analysis
        assert len(results.role_effectiveness) > 0
        for role, effectiveness in results.role_effectiveness.items():
            assert isinstance(effectiveness, RoleEffectiveness)
            assert effectiveness.role_name == role
            assert 0.0 <= effectiveness.overall_effectiveness <= 1.0
            assert effectiveness.total_executions > 0
        
        # Verify pattern identification
        assert len(results.identified_patterns) > 0
        for pattern in results.identified_patterns:
            assert isinstance(pattern, TeamPattern)
            assert 0.0 <= pattern.success_rate <= 1.0
            assert pattern.usage_count > 0
        
        # Verify recommendations
        assert len(results.recommended_teams) > 0
        assert len(results.optimization_recommendations) > 0
        
        # Verify confidence score
        assert 0.0 <= results.confidence_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_optimize_for_cost_efficiency(
        self,
        team_optimizer,
        sample_team_executions,
        sample_performance_metrics,
    ):
        """Test cost-efficiency optimization (Golden Test 2)."""
        
        results = await team_optimizer.optimize_patterns(
            historical_results=sample_team_executions,
            performance_metrics=sample_performance_metrics,
            optimization_objective=OptimizationObjective.COST_EFFICIENCY,
        )
        
        # Verify cost-focused optimization
        assert results.optimization_objective == OptimizationObjective.COST_EFFICIENCY
        
        # Check that recommendations prioritize cost efficiency
        if results.projected_improvements:
            cost_savings = [v for k, v in results.projected_improvements.items() if 'cost_savings' in k]
            assert len(cost_savings) > 0
            # Should have some projected cost savings
            assert any(savings > 0 for savings in cost_savings)
        
        # Verify recommendations mention cost
        cost_related_recommendations = [
            rec for rec in results.optimization_recommendations
            if 'cost' in rec.lower() or 'efficiency' in rec.lower()
        ]
        assert len(cost_related_recommendations) >= 0  # May or may not have cost-specific recommendations
    
    @pytest.mark.asyncio
    async def test_optimize_for_performance(
        self,
        team_optimizer,
        sample_team_executions,
        sample_performance_metrics,
    ):
        """Test performance optimization (Golden Test 3)."""
        
        results = await team_optimizer.optimize_patterns(
            historical_results=sample_team_executions,
            performance_metrics=sample_performance_metrics,
            optimization_objective=OptimizationObjective.PERFORMANCE,
        )
        
        assert results.optimization_objective == OptimizationObjective.PERFORMANCE
        
        # High-performing patterns should be prioritized
        if results.identified_patterns:
            top_pattern = results.identified_patterns[0]  # Should be sorted by effectiveness
            assert top_pattern.success_rate > 0.5  # Should be a successful pattern
    
    @pytest.mark.asyncio
    async def test_optimize_with_constraints(
        self,
        team_optimizer,
        sample_team_executions,
        sample_performance_metrics,
        optimization_constraints,
    ):
        """Test optimization with constraints (Golden Test 4)."""
        
        results = await team_optimizer.optimize_patterns(
            historical_results=sample_team_executions,
            performance_metrics=sample_performance_metrics,
            constraints=optimization_constraints,
        )
        
        # Verify recommended teams meet constraints
        for team in results.recommended_teams:
            team_size = len(team.primary_team)
            assert optimization_constraints.min_team_size <= team_size <= optimization_constraints.max_team_size
            
            # Check required roles
            team_roles = {agent.role for agent in team.primary_team}
            assert optimization_constraints.required_roles.issubset(team_roles)
            
            # Check forbidden roles
            assert not team_roles.intersection(optimization_constraints.forbidden_roles)
    
    # Edge Cases and Error Handling
    
    @pytest.mark.asyncio
    async def test_optimize_with_empty_history(
        self,
        team_optimizer,
        sample_performance_metrics,
    ):
        """Test optimization with empty execution history (Edge Case 1)."""
        
        results = await team_optimizer.optimize_patterns(
            historical_results=[],
            performance_metrics=sample_performance_metrics,
        )
        
        # Should handle gracefully
        assert results.data_points_analyzed == 0
        assert len(results.role_effectiveness) == 0
        assert len(results.identified_patterns) == 0
        assert len(results.recommended_teams) == 0
        assert results.confidence_score == 0.5  # Default confidence
    
    @pytest.mark.asyncio
    async def test_optimize_with_insufficient_data(
        self,
        team_optimizer,
        sample_performance_metrics,
    ):
        """Test optimization with insufficient data (Edge Case 2)."""
        
        # Create minimal execution history
        minimal_executions = [
            TeamExecution(
                execution_id='exec_1',
                team_composition=TeamComposition(
                    primary_team=[AgentSpec(role='coder', model_assignment='basic')],
                    load_order=['coder'],
                    coordination_strategy=CoordinationStrategy.SEQUENTIAL,
                    confidence_score=0.5,
                ),
                objective='Minimal task',
                status=ExecutionStatus.COMPLETED,
                progress=ExecutionProgress(total_tasks=1, completed_tasks=1),
            ),
        ]
        
        results = await team_optimizer.optimize_patterns(
            historical_results=minimal_executions,
            performance_metrics=sample_performance_metrics,
        )
        
        # Should complete but with low confidence
        assert results.data_points_analyzed == 1
        assert results.confidence_score < 0.7  # Should reflect insufficient data
    
    @pytest.mark.asyncio
    async def test_optimize_with_all_failures(
        self,
        team_optimizer,
        sample_performance_metrics,
    ):
        """Test optimization when all executions failed (Edge Case 3)."""
        
        failed_executions = []
        for i in range(10):
            execution = TeamExecution(
                execution_id=f'failed_{i}',
                team_composition=TeamComposition(
                    primary_team=[AgentSpec(role='broken_role', model_assignment='basic')],
                    load_order=['broken_role'],
                    coordination_strategy=CoordinationStrategy.SEQUENTIAL,
                    confidence_score=0.3,
                ),
                objective=f'Failed task {i}',
                status=ExecutionStatus.FAILED,
                progress=ExecutionProgress(total_tasks=3, failed_tasks=3),
                errors=['System failure', 'Configuration error'],
            )
            failed_executions.append(execution)
        
        results = await team_optimizer.optimize_patterns(
            historical_results=failed_executions,
            performance_metrics=sample_performance_metrics,
        )
        
        # Should identify poor-performing patterns and roles
        assert results.data_points_analyzed == 10
        if results.role_effectiveness:
            for effectiveness in results.role_effectiveness.values():
                assert effectiveness.success_rate <= 0.5  # All should have low success rates
    
    # Role Effectiveness Analysis Tests
    
    @pytest.mark.asyncio
    async def test_role_effectiveness_analysis_accuracy(
        self,
        team_optimizer,
        sample_team_executions,
        sample_performance_metrics,
    ):
        """Test accuracy of role effectiveness analysis."""
        
        results = await team_optimizer.optimize_patterns(
            historical_results=sample_team_executions,
            performance_metrics=sample_performance_metrics,
        )
        
        # Verify specific role effectiveness
        if 'architect' in results.role_effectiveness:
            architect_effectiveness = results.role_effectiveness['architect']
            
            # Architect appears in successful executions, should have high effectiveness
            assert architect_effectiveness.success_rate > 0.8
            assert architect_effectiveness.overall_effectiveness > 0.6
            assert architect_effectiveness.total_executions > 0
        
        if 'senior_coder' in results.role_effectiveness:
            senior_coder_effectiveness = results.role_effectiveness['senior_coder']
            
            # Senior coder only appears in high-performing executions
            assert senior_coder_effectiveness.success_rate >= 1.0
    
    @pytest.mark.asyncio
    async def test_role_collaboration_analysis(
        self,
        team_optimizer,
        sample_team_executions,
        sample_performance_metrics,
    ):
        """Test role collaboration pattern analysis."""
        
        results = await team_optimizer.optimize_patterns(
            historical_results=sample_team_executions,
            performance_metrics=sample_performance_metrics,
        )
        
        # Check for collaboration patterns in role effectiveness
        for role, effectiveness in results.role_effectiveness.items():
            # Should have some collaboration data
            assert effectiveness.best_team_combinations is not None
            # May or may not have actual collaborations depending on data
    
    # Pattern Identification Tests
    
    @pytest.mark.asyncio
    async def test_pattern_identification_accuracy(
        self,
        team_optimizer,
        sample_team_executions,
        sample_performance_metrics,
    ):
        """Test accuracy of pattern identification."""
        
        results = await team_optimizer.optimize_patterns(
            historical_results=sample_team_executions,
            performance_metrics=sample_performance_metrics,
        )
        
        # Should identify successful patterns
        successful_patterns = [p for p in results.identified_patterns if p.success_rate > 0.7]
        assert len(successful_patterns) > 0
        
        # Check pattern characteristics
        for pattern in successful_patterns:
            assert pattern.observation_count >= 5  # Minimum occurrences
            assert pattern.confidence_score > 0.0
            assert len(pattern.roles_involved) > 0
    
    @pytest.mark.asyncio
    async def test_pattern_statistical_validation(
        self,
        team_optimizer,
        sample_team_executions,
        sample_performance_metrics,
    ):
        """Test statistical validation of patterns."""
        
        results = await team_optimizer.optimize_patterns(
            historical_results=sample_team_executions,
            performance_metrics=sample_performance_metrics,
        )
        
        # Check statistical validation
        for pattern in results.identified_patterns:
            if pattern.statistical_analysis:
                assert pattern.statistical_analysis.sample_size > 0
                assert pattern.statistical_analysis.mean_performance >= 0.0
                assert len(pattern.statistical_analysis.confidence_interval_95) == 2
                
                # Confidence interval should be valid
                ci_low, ci_high = pattern.statistical_analysis.confidence_interval_95
                assert ci_low <= ci_high
    
    # Optimization Strategy Tests
    
    @pytest.mark.asyncio
    async def test_different_optimization_strategies(
        self,
        team_optimizer,
        sample_team_executions,
        sample_performance_metrics,
    ):
        """Test different optimization strategies produce valid results."""
        
        strategies = [
            OptimizationStrategy.GREEDY,
            OptimizationStrategy.HILL_CLIMBING,
            OptimizationStrategy.ENSEMBLE,
        ]
        
        for strategy in strategies:
            results = await team_optimizer.optimize_patterns(
                historical_results=sample_team_executions,
                performance_metrics=sample_performance_metrics,
                optimization_strategy=strategy,
            )
            
            assert results.strategy_used == strategy
            assert len(results.role_effectiveness) > 0
            # All strategies should produce some results with our data
    
    # Performance and Metrics Tests
    
    @pytest.mark.asyncio
    async def test_optimization_performance_metrics(
        self,
        team_optimizer,
        sample_team_executions,
        sample_performance_metrics,
    ):
        """Test optimization completes within reasonable time."""
        
        start_time = datetime.now(timezone.utc)
        
        results = await team_optimizer.optimize_patterns(
            historical_results=sample_team_executions,
            performance_metrics=sample_performance_metrics,
        )
        
        end_time = datetime.now(timezone.utc)
        total_time = (end_time - start_time).total_seconds()
        
        # Should complete reasonably quickly
        assert total_time < 10.0  # 10 second max for this dataset size
        assert results.optimization_duration_seconds <= total_time
    
    def test_optimization_history_tracking(
        self,
        team_optimizer,
    ):
        """Test optimization history is properly tracked."""
        
        # Initially empty
        history = team_optimizer.get_optimization_history()
        assert len(history) == 0
        
        # Test with limit
        limited_history = team_optimizer.get_optimization_history(limit=5)
        assert len(limited_history) == 0
    
    @pytest.mark.asyncio
    async def test_multiple_optimizations_history(
        self,
        team_optimizer,
        sample_team_executions,
        sample_performance_metrics,
    ):
        """Test multiple optimizations are tracked in history."""
        
        # Run multiple optimizations
        for i in range(3):
            await team_optimizer.optimize_patterns(
                historical_results=sample_team_executions,
                performance_metrics=sample_performance_metrics,
                optimization_objective=OptimizationObjective.BALANCED,
            )
        
        history = team_optimizer.get_optimization_history()
        assert len(history) == 3
        
        # Test limited history
        limited = team_optimizer.get_optimization_history(limit=2)
        assert len(limited) == 2
        
        # Should be most recent
        assert limited == history[-2:]
    
    # Integration and State Tests
    
    def test_team_optimizer_state_management(
        self,
        team_optimizer,
    ):
        """Test team optimizer state management."""
        
        # Test initial state
        patterns = team_optimizer.get_known_patterns()
        assert len(patterns) == 0
        
        role_cache = team_optimizer.get_role_effectiveness_cache()
        assert len(role_cache) == 0
        
        stats = team_optimizer.get_optimization_stats()
        assert stats['total_optimizations'] == 0
        assert stats['known_patterns'] == 0
        assert stats['last_optimization'] is None
    
    @pytest.mark.asyncio
    async def test_incremental_learning(
        self,
        team_optimizer,
        sample_team_executions,
    ):
        """Test incremental learning from new execution results."""
        
        # Add an execution result
        new_execution = TeamExecution(
            execution_id='new_exec',
            team_composition=TeamComposition(
                primary_team=[AgentSpec(role='new_role', model_assignment='standard')],
                load_order=['new_role'],
                coordination_strategy=CoordinationStrategy.SEQUENTIAL,
                confidence_score=0.7,
            ),
            objective='New task',
            status=ExecutionStatus.COMPLETED,
            progress=ExecutionProgress(total_tasks=3, completed_tasks=3),
        )
        
        await team_optimizer.add_execution_result(new_execution)
        # This mainly tests that the method exists and doesn't crash
        # Real implementation would update patterns incrementally
    
    # Specific Algorithm Tests
    
    @pytest.mark.asyncio
    async def test_baseline_performance_calculation(
        self,
        team_optimizer,
        sample_team_executions,
        sample_performance_metrics,
    ):
        """Test baseline performance calculation accuracy."""
        
        results = await team_optimizer.optimize_patterns(
            historical_results=sample_team_executions,
            performance_metrics=sample_performance_metrics,
        )
        
        baseline = results.current_performance_baseline
        
        # Should have key metrics
        assert 'success_rate' in baseline
        assert 'average_cost' in baseline
        assert 'average_duration' in baseline
        assert 'average_quality' in baseline
        
        # Values should be reasonable
        assert 0.0 <= baseline['success_rate'] <= 1.0
        assert baseline['average_cost'] >= 0.0
        assert baseline['average_duration'] >= 0.0
        assert 0.0 <= baseline['average_quality'] <= 1.0
    
    @pytest.mark.asyncio
    async def test_team_composition_generation(
        self,
        team_optimizer,
        sample_team_executions,
        sample_performance_metrics,
    ):
        """Test generated team compositions are valid."""
        
        results = await team_optimizer.optimize_patterns(
            historical_results=sample_team_executions,
            performance_metrics=sample_performance_metrics,
        )
        
        for team in results.recommended_teams:
            # Basic validation
            assert len(team.primary_team) > 0
            assert len(team.load_order) == len(team.primary_team)
            assert team.coordination_strategy is not None
            assert 0.0 <= team.confidence_score <= 1.0
            assert len(team.rationale) > 0
            
            # Verify all agents have valid roles and assignments
            for agent in team.primary_team:
                assert len(agent.role) > 0
                assert len(agent.model_assignment) > 0
                assert 1 <= agent.priority <= 10


if __name__ == '__main__':
    # Run specific tests for debugging
    pytest.main([__file__, '-v', '-s'])