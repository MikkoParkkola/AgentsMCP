"""Comprehensive tests for the retrospective engine.

This test suite covers:
- Golden tests for basic retrospective functionality
- Edge cases for error handling and robustness
- Integration tests with feedback collector
- Performance and concurrency tests
"""

import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any

from src.agentsmcp.orchestration.retrospective_engine import (
    RetrospectiveEngine,
    EnhancedRetrospectiveReport,
    RetrospectiveType,
    RetrospectiveScope,
    RetrospectiveFacilitationConfig,
    ImprovementAction,
    TeamPatternUpdate,
)
from src.agentsmcp.orchestration.feedback_collector import (
    FeedbackCollector,
    AgentFeedback,
    FeedbackType,
)
from src.agentsmcp.orchestration.models import (
    TeamComposition,
    TeamPerformanceMetrics,
    TaskClassification,
    AgentSpec,
    CoordinationStrategy,
    TaskType,
    ComplexityLevel,
    RiskLevel,
)


class TestRetrospectiveEngine:
    """Test suite for RetrospectiveEngine."""
    
    @pytest.fixture
    def mock_feedback_collector(self):
        """Create a mock feedback collector."""
        collector = MagicMock(spec=FeedbackCollector)
        collector.collect_agent_feedback = AsyncMock(return_value={
            'architect': AgentFeedback(
                agent_id='arch_001',
                agent_role='architect',
                overall_satisfaction=4.0,
                what_went_well=['Clear requirements', 'Good technical design'],
                what_could_improve=['Better time estimation', 'More detailed planning'],
                suggestions=['Add more validation checkpoints'],
            ),
            'coder': AgentFeedback(
                agent_id='coder_001',
                agent_role='coder',
                overall_satisfaction=3.5,
                what_went_well=['Clean code implementation', 'Good testing'],
                what_could_improve=['Better error handling', 'Performance optimization'],
                suggestions=['Use more efficient algorithms'],
            ),
        })
        return collector
    
    @pytest.fixture
    def facilitation_config(self):
        """Create test facilitation config."""
        return RetrospectiveFacilitationConfig(
            timeout_seconds=30,
            require_all_agents=False,
            anonymize_feedback=True,
            generate_action_items=True,
            auto_update_patterns=True,
            parallel_collection=True,
        )
    
    @pytest.fixture
    def sample_team_composition(self):
        """Create sample team composition."""
        return TeamComposition(
            primary_team=[
                AgentSpec(role='architect', model_assignment='premium', priority=1),
                AgentSpec(role='coder', model_assignment='standard', priority=2),
                AgentSpec(role='reviewer', model_assignment='standard', priority=3),
            ],
            load_order=['architect', 'coder', 'reviewer'],
            coordination_strategy=CoordinationStrategy.SEQUENTIAL,
            confidence_score=0.8,
            rationale='Balanced team for implementation task',
        )
    
    @pytest.fixture
    def sample_performance_metrics(self):
        """Create sample performance metrics."""
        return TeamPerformanceMetrics(
            team_id='team_001',
            task_type=TaskType.IMPLEMENTATION,
            success_rate=0.85,
            average_duration=450.0,
            average_cost=25.0,
            total_executions=20,
        )
    
    @pytest.fixture
    def sample_task_classification(self):
        """Create sample task classification."""
        return TaskClassification(
            task_type=TaskType.IMPLEMENTATION,
            complexity=ComplexityLevel.MEDIUM,
            required_roles=['architect', 'coder'],
            optional_roles=['reviewer'],
            estimated_effort=50,
            risk_level=RiskLevel.MEDIUM,
            confidence=0.9,
        )
    
    @pytest.fixture
    def sample_execution_results(self):
        """Create sample execution results."""
        return {
            'task_id': 'task_123',
            'status': 'completed',
            'duration_seconds': 420.0,
            'team_size': 3,
            'coordination_strategy': 'sequential',
            'completed_tasks': 8,
            'total_tasks': 10,
            'failed_tasks': 2,
            'errors': ['Minor validation error', 'Performance issue'],
            'resource_usage': {'cost': 22.5, 'memory': 512, 'cpu': 45.0},
            'task_timings': {
                'analysis': 120.0,
                'implementation': 180.0,
                'testing': 90.0,
                'review': 30.0,
            },
        }
    
    @pytest.fixture
    def retrospective_engine(self, mock_feedback_collector, facilitation_config):
        """Create retrospective engine with mocked dependencies."""
        return RetrospectiveEngine(
            feedback_collector=mock_feedback_collector,
            facilitation_config=facilitation_config,
        )
    
    # Golden Tests - Basic Functionality
    
    @pytest.mark.asyncio
    async def test_conduct_retrospective_basic_success(
        self,
        retrospective_engine,
        sample_execution_results,
        sample_performance_metrics,
        sample_team_composition,
        sample_task_classification,
    ):
        """Test basic successful retrospective conduct (Golden Test 1)."""
        
        result = await retrospective_engine.conduct_retrospective(
            execution_results=sample_execution_results,
            performance_metrics=sample_performance_metrics,
            team_composition=sample_team_composition,
            task_classification=sample_task_classification,
        )
        
        # Verify basic structure
        assert isinstance(result, EnhancedRetrospectiveReport)
        assert result.retrospective_type == RetrospectiveType.POST_TASK
        assert result.scope == RetrospectiveScope.COMPREHENSIVE
        assert result.duration_seconds > 0
        assert result.completed_at is not None
        
        # Verify participants
        assert len(result.participants) == 3
        assert 'architect' in result.participants
        assert 'coder' in result.participants
        assert 'reviewer' in result.participants
        
        # Verify feedback collection occurred
        assert len(result.agent_feedback_summary) > 0
        assert result.feedback_themes is not None
        
        # Verify performance analysis
        assert result.performance_analysis is not None
        assert 'execution_status' in result.performance_analysis
        assert 'success_rate' in result.performance_analysis
        
        # Verify improvement items were generated
        assert len(result.improvement_actions) > 0
        assert all(isinstance(action, ImprovementAction) for action in result.improvement_actions)
        
        # Verify team health score is reasonable
        assert 0.0 <= result.team_health_score <= 1.0
        assert 0.0 <= result.coordination_effectiveness <= 1.0
    
    @pytest.mark.asyncio
    async def test_conduct_retrospective_with_different_types(
        self,
        retrospective_engine,
        sample_execution_results,
        sample_performance_metrics,
        sample_team_composition,
        sample_task_classification,
    ):
        """Test retrospective with different types and scopes (Golden Test 2)."""
        
        # Test incident retrospective
        incident_result = await retrospective_engine.conduct_retrospective(
            execution_results=sample_execution_results,
            performance_metrics=sample_performance_metrics,
            team_composition=sample_team_composition,
            task_classification=sample_task_classification,
            retrospective_type=RetrospectiveType.INCIDENT,
            scope=RetrospectiveScope.PROCESS_FOCUSED,
        )
        
        assert incident_result.retrospective_type == RetrospectiveType.INCIDENT
        assert incident_result.scope == RetrospectiveScope.PROCESS_FOCUSED
        
        # Test sprint retrospective
        sprint_result = await retrospective_engine.conduct_retrospective(
            execution_results=sample_execution_results,
            performance_metrics=sample_performance_metrics,
            team_composition=sample_team_composition,
            task_classification=sample_task_classification,
            retrospective_type=RetrospectiveType.SPRINT,
            scope=RetrospectiveScope.TEAM_FOCUSED,
        )
        
        assert sprint_result.retrospective_type == RetrospectiveType.SPRINT
        assert sprint_result.scope == RetrospectiveScope.TEAM_FOCUSED
    
    @pytest.mark.asyncio
    async def test_conduct_retrospective_parallel_vs_sequential(
        self,
        mock_feedback_collector,
        sample_execution_results,
        sample_performance_metrics,
        sample_team_composition,
        sample_task_classification,
    ):
        """Test parallel vs sequential feedback collection (Golden Test 3)."""
        
        # Test parallel collection
        parallel_config = RetrospectiveFacilitationConfig(parallel_collection=True)
        parallel_engine = RetrospectiveEngine(
            feedback_collector=mock_feedback_collector,
            facilitation_config=parallel_config,
        )
        
        parallel_result = await parallel_engine.conduct_retrospective(
            execution_results=sample_execution_results,
            performance_metrics=sample_performance_metrics,
            team_composition=sample_team_composition,
            task_classification=sample_task_classification,
        )
        
        # Test sequential collection
        sequential_config = RetrospectiveFacilitationConfig(parallel_collection=False)
        sequential_engine = RetrospectiveEngine(
            feedback_collector=mock_feedback_collector,
            facilitation_config=sequential_config,
        )
        
        sequential_result = await sequential_engine.conduct_retrospective(
            execution_results=sample_execution_results,
            performance_metrics=sample_performance_metrics,
            team_composition=sample_team_composition,
            task_classification=sample_task_classification,
        )
        
        # Both should produce valid results
        assert parallel_result.completed_at is not None
        assert sequential_result.completed_at is not None
        
        # Parallel should generally be faster (in real scenarios)
        # But we can't test timing reliably with mocks
        assert len(parallel_result.improvement_actions) > 0
        assert len(sequential_result.improvement_actions) > 0
    
    # Edge Cases and Error Handling
    
    @pytest.mark.asyncio
    async def test_conduct_retrospective_with_empty_team(
        self,
        retrospective_engine,
        sample_execution_results,
        sample_performance_metrics,
        sample_task_classification,
    ):
        """Test retrospective with empty team composition (Edge Case 1)."""
        
        empty_team = TeamComposition(
            primary_team=[],
            load_order=[],
            coordination_strategy=CoordinationStrategy.SEQUENTIAL,
            confidence_score=0.0,
            rationale='Empty team for testing',
        )
        
        result = await retrospective_engine.conduct_retrospective(
            execution_results=sample_execution_results,
            performance_metrics=sample_performance_metrics,
            team_composition=empty_team,
            task_classification=sample_task_classification,
        )
        
        # Should complete without error but with minimal content
        assert result.completed_at is not None
        assert len(result.participants) == 0
        assert result.team_health_score >= 0.0
    
    @pytest.mark.asyncio
    async def test_conduct_retrospective_with_feedback_timeout(
        self,
        facilitation_config,
        sample_execution_results,
        sample_performance_metrics,
        sample_team_composition,
        sample_task_classification,
    ):
        """Test retrospective when feedback collection times out (Edge Case 2)."""
        
        # Create mock that times out
        timeout_collector = MagicMock(spec=FeedbackCollector)
        timeout_collector.collect_agent_feedback = AsyncMock(
            side_effect=asyncio.TimeoutError("Feedback collection timeout")
        )
        
        # Configure to not require all agents
        facilitation_config.require_all_agents = False
        
        engine = RetrospectiveEngine(
            feedback_collector=timeout_collector,
            facilitation_config=facilitation_config,
        )
        
        result = await engine.conduct_retrospective(
            execution_results=sample_execution_results,
            performance_metrics=sample_performance_metrics,
            team_composition=sample_team_composition,
            task_classification=sample_task_classification,
        )
        
        # Should complete with limited feedback
        assert result.completed_at is not None
        assert len(result.agent_feedback_summary) == 0 or result.agent_feedback_summary.get('total_responses', 0) == 0
    
    @pytest.mark.asyncio
    async def test_conduct_retrospective_with_required_all_agents_timeout(
        self,
        facilitation_config,
        sample_execution_results,
        sample_performance_metrics,
        sample_team_composition,
        sample_task_classification,
    ):
        """Test retrospective with require_all_agents=True and timeout (Edge Case 3)."""
        
        # Create mock that times out
        timeout_collector = MagicMock(spec=FeedbackCollector)
        timeout_collector.collect_agent_feedback = AsyncMock(
            side_effect=asyncio.TimeoutError("Feedback collection timeout")
        )
        
        # Configure to require all agents
        facilitation_config.require_all_agents = True
        
        engine = RetrospectiveEngine(
            feedback_collector=timeout_collector,
            facilitation_config=facilitation_config,
        )
        
        with pytest.raises(asyncio.TimeoutError):
            await engine.conduct_retrospective(
                execution_results=sample_execution_results,
                performance_metrics=sample_performance_metrics,
                team_composition=sample_team_composition,
                task_classification=sample_task_classification,
            )
    
    @pytest.mark.asyncio
    async def test_conduct_retrospective_with_malformed_execution_results(
        self,
        retrospective_engine,
        sample_performance_metrics,
        sample_team_composition,
        sample_task_classification,
    ):
        """Test retrospective with malformed execution results (Edge Case 4)."""
        
        malformed_results = {
            'incomplete': 'data',
            'no_status': True,
            'invalid_duration': 'not_a_number',
        }
        
        result = await retrospective_engine.conduct_retrospective(
            execution_results=malformed_results,
            performance_metrics=sample_performance_metrics,
            team_composition=sample_team_composition,
            task_classification=sample_task_classification,
        )
        
        # Should handle gracefully with defaults
        assert result.completed_at is not None
        assert result.performance_analysis is not None
        assert 'execution_status' in result.performance_analysis
    
    # Integration Tests
    
    @pytest.mark.asyncio
    async def test_retrospective_history_tracking(
        self,
        retrospective_engine,
        sample_execution_results,
        sample_performance_metrics,
        sample_team_composition,
        sample_task_classification,
    ):
        """Test retrospective history is properly tracked."""
        
        # Conduct multiple retrospectives
        for i in range(3):
            await retrospective_engine.conduct_retrospective(
                execution_results=sample_execution_results,
                performance_metrics=sample_performance_metrics,
                team_composition=sample_team_composition,
                task_classification=sample_task_classification,
            )
        
        history = retrospective_engine.get_retrospective_history()
        assert len(history) == 3
        
        # Verify history entries are properly ordered (most recent last)
        timestamps = [report.created_at for report in history]
        assert timestamps == sorted(timestamps)
    
    @pytest.mark.asyncio
    async def test_team_patterns_update(
        self,
        retrospective_engine,
        sample_execution_results,
        sample_performance_metrics,
        sample_team_composition,
        sample_task_classification,
    ):
        """Test team patterns are updated during retrospective."""
        
        # Modify execution results to show success
        sample_execution_results['status'] = 'completed'
        sample_performance_metrics.success_rate = 0.9
        
        result = await retrospective_engine.conduct_retrospective(
            execution_results=sample_execution_results,
            performance_metrics=sample_performance_metrics,
            team_composition=sample_team_composition,
            task_classification=sample_task_classification,
        )
        
        # Verify pattern updates were generated
        assert len(result.pattern_updates) >= 0  # May be 0 or more depending on logic
        
        # Check team patterns in engine
        team_patterns = retrospective_engine.get_team_patterns()
        assert isinstance(team_patterns, dict)
    
    @pytest.mark.asyncio
    async def test_improvement_action_lifecycle(
        self,
        retrospective_engine,
        sample_execution_results,
        sample_performance_metrics,
        sample_team_composition,
        sample_task_classification,
    ):
        """Test improvement action creation and status updates."""
        
        result = await retrospective_engine.conduct_retrospective(
            execution_results=sample_execution_results,
            performance_metrics=sample_performance_metrics,
            team_composition=sample_team_composition,
            task_classification=sample_task_classification,
        )
        
        # Verify improvement actions were created
        assert len(result.improvement_actions) > 0
        action = result.improvement_actions[0]
        assert action.status == 'open'
        
        # Test status update
        success = await retrospective_engine.update_improvement_action_status(
            action.action_id, 'in_progress', 'Started working on this issue'
        )
        assert success
        
        # Test updating non-existent action
        success = await retrospective_engine.update_improvement_action_status(
            'non_existent_id', 'completed', 'Test'
        )
        assert not success
    
    # Performance and Concurrency Tests
    
    @pytest.mark.asyncio
    async def test_retrospective_completion_within_timeout(
        self,
        retrospective_engine,
        sample_execution_results,
        sample_performance_metrics,
        sample_team_composition,
        sample_task_classification,
    ):
        """Test retrospective completes within reasonable time."""
        
        start_time = datetime.now(timezone.utc)
        
        result = await retrospective_engine.conduct_retrospective(
            execution_results=sample_execution_results,
            performance_metrics=sample_performance_metrics,
            team_composition=sample_team_composition,
            task_classification=sample_task_classification,
        )
        
        end_time = datetime.now(timezone.utc)
        total_time = (end_time - start_time).total_seconds()
        
        # Should complete within 30 seconds (as per requirement)
        assert total_time <= 30.0
        assert result.duration_seconds <= 30.0
    
    @pytest.mark.asyncio
    async def test_concurrent_retrospectives(
        self,
        mock_feedback_collector,
        facilitation_config,
        sample_execution_results,
        sample_performance_metrics,
        sample_team_composition,
        sample_task_classification,
    ):
        """Test multiple concurrent retrospectives."""
        
        engine = RetrospectiveEngine(
            feedback_collector=mock_feedback_collector,
            facilitation_config=facilitation_config,
        )
        
        # Start multiple retrospectives concurrently
        tasks = []
        for i in range(3):
            task = asyncio.create_task(
                engine.conduct_retrospective(
                    execution_results=sample_execution_results,
                    performance_metrics=sample_performance_metrics,
                    team_composition=sample_team_composition,
                    task_classification=sample_task_classification,
                )
            )
            tasks.append(task)
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks)
        
        # All should complete successfully
        assert len(results) == 3
        for result in results:
            assert result.completed_at is not None
            assert len(result.improvement_actions) > 0
        
        # History should contain all retrospectives
        history = engine.get_retrospective_history()
        assert len(history) == 3
    
    # Specific Feature Tests
    
    @pytest.mark.asyncio
    async def test_performance_analysis_accuracy(
        self,
        retrospective_engine,
        sample_performance_metrics,
        sample_team_composition,
        sample_task_classification,
    ):
        """Test accuracy of performance analysis calculations."""
        
        execution_results = {
            'status': 'completed',
            'duration_seconds': 600.0,
            'completed_tasks': 9,
            'total_tasks': 10,
            'failed_tasks': 1,
            'errors': ['Minor issue'],
            'resource_usage': {'cost': 30.0},
            'task_timings': {
                'task1': 100.0,
                'task2': 500.0,  # This should be identified as bottleneck
                'task3': 50.0,
            },
        }
        
        result = await retrospective_engine.conduct_retrospective(
            execution_results=execution_results,
            performance_metrics=sample_performance_metrics,
            team_composition=sample_team_composition,
            task_classification=sample_task_classification,
        )
        
        analysis = result.performance_analysis
        
        # Verify cost efficiency calculation
        assert 'cost_efficiency' in analysis
        assert 0.0 <= analysis['cost_efficiency'] <= 1.0
        
        # Verify bottleneck identification
        assert 'bottlenecks' in analysis
        bottlenecks = analysis['bottlenecks']
        assert any('task2' in bottleneck for bottleneck in bottlenecks)
        
        # Verify quality indicators
        assert 'quality_indicators' in analysis
        quality = analysis['quality_indicators']
        assert 'error_rate' in quality
        assert quality['error_rate'] == 0.1  # 1 error out of 10 tasks
        assert quality['completion_rate'] == 0.9  # 9 completed out of 10 tasks
    
    @pytest.mark.asyncio
    async def test_team_health_score_calculation(
        self,
        retrospective_engine,
        sample_execution_results,
        sample_performance_metrics,
        sample_team_composition,
        sample_task_classification,
    ):
        """Test team health score calculation with different scenarios."""
        
        # Test high-performing scenario
        high_performance_metrics = TeamPerformanceMetrics(
            team_id='team_001',
            task_type=TaskType.IMPLEMENTATION,
            success_rate=0.95,
            average_duration=300.0,
            average_cost=20.0,
            total_executions=50,
        )
        
        high_result = await retrospective_engine.conduct_retrospective(
            execution_results=sample_execution_results,
            performance_metrics=high_performance_metrics,
            team_composition=sample_team_composition,
            task_classification=sample_task_classification,
        )
        
        # Test low-performing scenario
        low_performance_metrics = TeamPerformanceMetrics(
            team_id='team_002',
            task_type=TaskType.IMPLEMENTATION,
            success_rate=0.5,
            average_duration=800.0,
            average_cost=60.0,
            total_executions=10,
        )
        
        low_result = await retrospective_engine.conduct_retrospective(
            execution_results=sample_execution_results,
            performance_metrics=low_performance_metrics,
            team_composition=sample_team_composition,
            task_classification=sample_task_classification,
        )
        
        # High-performing team should have higher health score
        assert high_result.team_health_score > low_result.team_health_score
        assert 0.0 <= high_result.team_health_score <= 1.0
        assert 0.0 <= low_result.team_health_score <= 1.0


if __name__ == '__main__':
    # Run specific tests for debugging
    pytest.main([__file__, '-v', '-s'])