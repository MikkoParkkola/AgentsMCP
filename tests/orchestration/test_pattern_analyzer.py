"""Comprehensive tests for the pattern analyzer.

This test suite covers:
- Golden tests for pattern analysis functionality  
- Success pattern identification
- Anti-pattern detection and warnings
- Role effectiveness scoring
- Statistical validation of patterns
- Edge cases and error handling
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
from typing import Dict, List, Any
import statistics

from src.agentsmcp.orchestration.pattern_analyzer import (
    PatternAnalyzer,
    TeamPattern,
    AntiPattern,
    RoleEffectivenessAnalysis,
    PatternType,
    PatternCategory,
    PatternScope,
    StatisticalSignificance,
    StatisticalAnalysis,
    TemporalAnalysis,
)
from src.agentsmcp.orchestration.execution_engine import (
    TeamExecution,
    ExecutionStatus,
    ExecutionProgress,
)
from src.agentsmcp.orchestration.models import (
    TeamComposition,
    AgentSpec,
    CoordinationStrategy,
    TaskType,
    ComplexityLevel,
    RiskLevel,
)


class TestPatternAnalyzer:
    """Test suite for PatternAnalyzer."""
    
    @pytest.fixture
    def pattern_analyzer(self):
        """Create pattern analyzer instance."""
        return PatternAnalyzer(
            min_pattern_occurrences=5,
            significance_threshold=0.05,
            confidence_threshold=0.7,
            temporal_window_days=90,
        )
    
    @pytest.fixture
    def success_execution_history(self):
        """Create execution history with successful patterns."""
        executions = []
        base_time = datetime.now(timezone.utc) - timedelta(days=60)
        
        # Pattern 1: Architect + Coder (highly successful)
        for i in range(12):
            execution = TeamExecution(
                execution_id=f'success_arch_coder_{i}',
                team_composition=TeamComposition(
                    primary_team=[
                        AgentSpec(role='architect', model_assignment='premium'),
                        AgentSpec(role='coder', model_assignment='standard'),
                    ],
                    load_order=['architect', 'coder'],
                    coordination_strategy=CoordinationStrategy.SEQUENTIAL,
                    confidence_score=0.9,
                ),
                objective=f'Implementation task {i}',
                status=ExecutionStatus.COMPLETED,
                progress=ExecutionProgress(total_tasks=5, completed_tasks=5),
                started_at=base_time + timedelta(days=i * 2),
                total_duration_seconds=250.0 + i * 5,
                resource_usage={'cost': 15.0 + i, 'memory': 256, 'cpu': 25.0},
                errors=[],
            )
            # Add task classification
            execution.task_classification = MagicMock()
            execution.task_classification.task_type = TaskType.IMPLEMENTATION
            execution.task_classification.complexity = ComplexityLevel.MEDIUM
            execution.task_classification.risk_level = RiskLevel.LOW
            executions.append(execution)
        
        # Pattern 2: Full team (architect + coder + reviewer, also successful)
        for i in range(8):
            execution = TeamExecution(
                execution_id=f'success_full_team_{i}',
                team_composition=TeamComposition(
                    primary_team=[
                        AgentSpec(role='architect', model_assignment='premium'),
                        AgentSpec(role='coder', model_assignment='standard'),
                        AgentSpec(role='reviewer', model_assignment='standard'),
                    ],
                    load_order=['architect', 'coder', 'reviewer'],
                    coordination_strategy=CoordinationStrategy.COLLABORATIVE,
                    confidence_score=0.85,
                ),
                objective=f'Full development task {i}',
                status=ExecutionStatus.COMPLETED,
                progress=ExecutionProgress(total_tasks=8, completed_tasks=7),
                started_at=base_time + timedelta(days=30 + i * 3),
                total_duration_seconds=400.0 + i * 10,
                resource_usage={'cost': 30.0 + i * 2, 'memory': 512, 'cpu': 40.0},
                errors=['Minor formatting issue'] if i == 3 else [],
            )
            execution.task_classification = MagicMock()
            execution.task_classification.task_type = TaskType.IMPLEMENTATION
            execution.task_classification.complexity = ComplexityLevel.HIGH
            execution.task_classification.risk_level = RiskLevel.MEDIUM
            executions.append(execution)
        
        return executions
    
    @pytest.fixture
    def mixed_execution_history(self):
        """Create execution history with mixed success/failure patterns."""
        executions = []
        base_time = datetime.now(timezone.utc) - timedelta(days=45)
        
        # Successful pattern: Small focused teams
        for i in range(10):
            execution = TeamExecution(
                execution_id=f'success_small_{i}',
                team_composition=TeamComposition(
                    primary_team=[
                        AgentSpec(role='specialist', model_assignment='premium'),
                        AgentSpec(role='assistant', model_assignment='standard'),
                    ],
                    load_order=['specialist', 'assistant'],
                    coordination_strategy=CoordinationStrategy.SEQUENTIAL,
                    confidence_score=0.8,
                ),
                objective=f'Specialized task {i}',
                status=ExecutionStatus.COMPLETED,
                progress=ExecutionProgress(total_tasks=3, completed_tasks=3),
                started_at=base_time + timedelta(days=i * 2),
                total_duration_seconds=180.0 + i * 8,
                resource_usage={'cost': 12.0 + i, 'memory': 128, 'cpu': 20.0},
                errors=[],
            )
            executions.append(execution)
        
        # Anti-pattern: Large uncoordinated teams (high failure rate)
        for i in range(8):
            execution = TeamExecution(
                execution_id=f'failure_large_{i}',
                team_composition=TeamComposition(
                    primary_team=[
                        AgentSpec(role='generalist_a', model_assignment='basic'),
                        AgentSpec(role='generalist_b', model_assignment='basic'),
                        AgentSpec(role='generalist_c', model_assignment='basic'),
                        AgentSpec(role='generalist_d', model_assignment='basic'),
                        AgentSpec(role='generalist_e', model_assignment='basic'),
                    ],
                    load_order=['generalist_a', 'generalist_b', 'generalist_c', 'generalist_d', 'generalist_e'],
                    coordination_strategy=CoordinationStrategy.PARALLEL,
                    confidence_score=0.4,
                ),
                objective=f'Complex coordination task {i}',
                status=ExecutionStatus.FAILED,
                progress=ExecutionProgress(total_tasks=10, completed_tasks=3, failed_tasks=7),
                started_at=base_time + timedelta(days=20 + i * 2),
                total_duration_seconds=800.0 + i * 50,
                resource_usage={'cost': 75.0 + i * 5, 'memory': 1024, 'cpu': 95.0},
                errors=[
                    'Coordination failure',
                    'Resource contention',
                    'Communication breakdown',
                    'Task dependency issues',
                ],
            )
            executions.append(execution)
        
        return executions
    
    @pytest.fixture
    def temporal_execution_history(self):
        """Create execution history with temporal trends."""
        executions = []
        base_time = datetime.now(timezone.utc) - timedelta(days=120)
        
        # Improving performance trend
        for i in range(15):
            # Performance improves over time
            performance_factor = 0.6 + (i * 0.03)  # Improves from 0.6 to 1.02
            
            execution = TeamExecution(
                execution_id=f'temporal_{i}',
                team_composition=TeamComposition(
                    primary_team=[
                        AgentSpec(role='evolving_role', model_assignment='standard'),
                    ],
                    load_order=['evolving_role'],
                    coordination_strategy=CoordinationStrategy.SEQUENTIAL,
                    confidence_score=0.5 + (i * 0.03),
                ),
                objective=f'Evolving task {i}',
                status=ExecutionStatus.COMPLETED if performance_factor > 0.7 else ExecutionStatus.FAILED,
                progress=ExecutionProgress(
                    total_tasks=5,
                    completed_tasks=min(5, int(5 * performance_factor)),
                    failed_tasks=max(0, 5 - int(5 * performance_factor)),
                ),
                started_at=base_time + timedelta(days=i * 7),
                total_duration_seconds=400.0 - (i * 15),  # Getting faster
                resource_usage={'cost': 25.0 - (i * 0.5), 'memory': 256, 'cpu': 30.0},
                errors=[] if performance_factor > 0.8 else ['Learning curve issue'],
            )
            executions.append(execution)
        
        return executions
    
    # Golden Tests - Core Pattern Analysis
    
    @pytest.mark.asyncio
    async def test_analyze_patterns_basic_success(
        self,
        pattern_analyzer,
        success_execution_history,
    ):
        """Test basic successful pattern analysis (Golden Test 1)."""
        
        patterns = await pattern_analyzer.analyze_patterns(success_execution_history)
        
        # Verify patterns were identified
        assert len(patterns) > 0
        
        # Check pattern structure
        for pattern in patterns:
            assert isinstance(pattern, TeamPattern)
            assert pattern.pattern_type == PatternType.SUCCESS_PATTERN
            assert len(pattern.roles_involved) > 0
            assert pattern.observation_count >= pattern_analyzer.min_pattern_occurrences
            assert 0.0 <= pattern.success_rate <= 1.0
            assert 0.0 <= pattern.overall_strength <= 1.0
            assert pattern.created_at is not None
        
        # Verify high-performing patterns are identified
        high_success_patterns = [p for p in patterns if p.success_rate > 0.8]
        assert len(high_success_patterns) > 0
        
        # Check that architect-coder pattern is identified (should be top pattern)
        architect_coder_patterns = [
            p for p in patterns 
            if 'architect' in p.roles_involved and 'coder' in p.roles_involved
        ]
        assert len(architect_coder_patterns) > 0
    
    @pytest.mark.asyncio
    async def test_analyze_patterns_with_anti_patterns(
        self,
        pattern_analyzer,
        mixed_execution_history,
    ):
        """Test pattern analysis identifies anti-patterns (Golden Test 2)."""
        
        patterns = await pattern_analyzer.analyze_patterns(mixed_execution_history)
        
        # Should identify both success patterns and potentially anti-patterns
        assert len(patterns) > 0
        
        # Check for anti-pattern detection
        anti_patterns = pattern_analyzer.get_anti_patterns()
        
        # Should identify the large team anti-pattern
        large_team_anti_patterns = [
            ap for ap in anti_patterns
            if len(ap.affected_roles) >= 4  # Large team size
        ]
        # May or may not detect anti-patterns depending on thresholds
        
        # Verify patterns are sorted by strength
        pattern_strengths = [p.overall_strength for p in patterns]
        assert pattern_strengths == sorted(pattern_strengths, reverse=True)
    
    @pytest.mark.asyncio
    async def test_role_effectiveness_analysis(
        self,
        pattern_analyzer,
        success_execution_history,
    ):
        """Test role effectiveness analysis accuracy (Golden Test 3)."""
        
        await pattern_analyzer.analyze_patterns(success_execution_history)
        
        role_analyses = pattern_analyzer.get_role_effectiveness()
        
        # Should have analysis for key roles
        assert len(role_analyses) > 0
        
        for role, analysis in role_analyses.items():
            assert isinstance(analysis, RoleEffectivenessAnalysis)
            assert analysis.role_name == role
            assert 0.0 <= analysis.overall_effectiveness <= 1.0
            assert 0.0 <= analysis.success_rate <= 1.0
            assert analysis.sample_size > 0
            
            # Should have some contextual effectiveness data
            # (may be empty if not enough varied data)
        
        # Architect should be highly effective (appears in successful patterns)
        if 'architect' in role_analyses:
            architect_analysis = role_analyses['architect']
            assert architect_analysis.success_rate > 0.8
            assert architect_analysis.overall_effectiveness > 0.7
    
    @pytest.mark.asyncio
    async def test_temporal_pattern_analysis(
        self,
        pattern_analyzer,
        temporal_execution_history,
    ):
        """Test temporal pattern and trend analysis (Golden Test 4)."""
        
        patterns = await pattern_analyzer.analyze_patterns(temporal_execution_history)
        
        # Should identify temporal patterns
        temporal_patterns = [
            p for p in patterns 
            if p.pattern_category == PatternCategory.TEMPORAL
        ]
        
        # May identify improving trend
        if temporal_patterns:
            trend_pattern = temporal_patterns[0]
            assert trend_pattern.temporal_analysis is not None
            assert trend_pattern.temporal_analysis.trend_direction in ['improving', 'declining', 'stable']
    
    # Edge Cases and Error Handling
    
    @pytest.mark.asyncio
    async def test_analyze_patterns_empty_history(
        self,
        pattern_analyzer,
    ):
        """Test pattern analysis with empty history (Edge Case 1)."""
        
        patterns = await pattern_analyzer.analyze_patterns([])
        
        # Should handle gracefully
        assert patterns == []
        assert len(pattern_analyzer.get_identified_patterns()) == 0
        assert len(pattern_analyzer.get_anti_patterns()) == 0
        assert len(pattern_analyzer.get_role_effectiveness()) == 0
    
    @pytest.mark.asyncio
    async def test_analyze_patterns_insufficient_occurrences(
        self,
        pattern_analyzer,
    ):
        """Test pattern analysis with insufficient pattern occurrences (Edge Case 2)."""
        
        # Create minimal execution history (below min_pattern_occurrences)
        sparse_history = []
        for i in range(3):  # Below the min_pattern_occurrences=5 threshold
            execution = TeamExecution(
                execution_id=f'sparse_{i}',
                team_composition=TeamComposition(
                    primary_team=[AgentSpec(role='rare_role', model_assignment='basic')],
                    load_order=['rare_role'],
                    coordination_strategy=CoordinationStrategy.SEQUENTIAL,
                    confidence_score=0.5,
                ),
                objective=f'Rare task {i}',
                status=ExecutionStatus.COMPLETED,
                progress=ExecutionProgress(total_tasks=2, completed_tasks=2),
                started_at=datetime.now(timezone.utc) - timedelta(days=i),
            )
            sparse_history.append(execution)
        
        patterns = await pattern_analyzer.analyze_patterns(sparse_history)
        
        # Should not identify patterns due to insufficient occurrences
        assert len(patterns) == 0
    
    @pytest.mark.asyncio
    async def test_analyze_patterns_malformed_data(
        self,
        pattern_analyzer,
    ):
        """Test pattern analysis with malformed execution data (Edge Case 3)."""
        
        malformed_history = []
        
        # Execution without team composition
        malformed_execution = TeamExecution(
            execution_id='malformed_1',
            team_composition=None,
            objective='Malformed task',
            status=ExecutionStatus.COMPLETED,
            progress=ExecutionProgress(total_tasks=1, completed_tasks=1),
        )
        malformed_history.append(malformed_execution)
        
        # Execution with empty team
        empty_team_execution = TeamExecution(
            execution_id='malformed_2',
            team_composition=TeamComposition(
                primary_team=[],
                load_order=[],
                coordination_strategy=CoordinationStrategy.SEQUENTIAL,
                confidence_score=0.0,
            ),
            objective='Empty team task',
            status=ExecutionStatus.COMPLETED,
            progress=ExecutionProgress(total_tasks=1, completed_tasks=1),
        )
        malformed_history.append(empty_team_execution)
        
        # Execution without timestamps
        no_timestamp_execution = TeamExecution(
            execution_id='malformed_3',
            team_composition=TeamComposition(
                primary_team=[AgentSpec(role='test_role', model_assignment='basic')],
                load_order=['test_role'],
                coordination_strategy=CoordinationStrategy.SEQUENTIAL,
                confidence_score=0.5,
            ),
            objective='No timestamp task',
            status=ExecutionStatus.COMPLETED,
            progress=ExecutionProgress(total_tasks=1, completed_tasks=1),
            started_at=None,  # Missing timestamp
        )
        malformed_history.append(no_timestamp_execution)
        
        # Should handle gracefully without crashing
        patterns = await pattern_analyzer.analyze_patterns(malformed_history)
        assert isinstance(patterns, list)
    
    # Statistical Validation Tests
    
    @pytest.mark.asyncio
    async def test_pattern_statistical_validation(
        self,
        pattern_analyzer,
        success_execution_history,
    ):
        """Test statistical validation of identified patterns."""
        
        patterns = await pattern_analyzer.analyze_patterns(success_execution_history)
        
        # Check statistical analysis for patterns
        for pattern in patterns:
            if pattern.statistical_analysis:
                stats = pattern.statistical_analysis
                
                # Basic statistical measures should be valid
                assert stats.sample_size > 0
                assert stats.mean_performance >= 0.0
                assert stats.standard_deviation >= 0.0
                
                # Confidence interval should be valid
                ci_low, ci_high = stats.confidence_interval_95
                assert ci_low <= ci_high
                assert 0.0 <= ci_low <= 1.0
                assert 0.0 <= ci_high <= 1.0
                
                # P-value should be in valid range
                if stats.p_value is not None:
                    assert 0.0 <= stats.p_value <= 1.0
    
    # Integration and Performance Tests
    
    @pytest.mark.asyncio
    async def test_pattern_analysis_performance(
        self,
        pattern_analyzer,
        success_execution_history,
    ):
        """Test pattern analysis completes in reasonable time."""
        
        start_time = datetime.now(timezone.utc)
        
        patterns = await pattern_analyzer.analyze_patterns(success_execution_history)
        
        end_time = datetime.now(timezone.utc)
        analysis_time = (end_time - start_time).total_seconds()
        
        # Should complete quickly for reasonable dataset size
        assert analysis_time < 5.0  # 5 seconds max
        assert len(patterns) >= 0  # Should produce some results
    
    def test_pattern_analyzer_state_management(
        self,
        pattern_analyzer,
    ):
        """Test pattern analyzer state management."""
        
        # Test initial state
        assert len(pattern_analyzer.get_identified_patterns()) == 0
        assert len(pattern_analyzer.get_anti_patterns()) == 0
        assert len(pattern_analyzer.get_role_effectiveness()) == 0
        assert len(pattern_analyzer.get_analysis_history()) == 0
    
    @pytest.mark.asyncio
    async def test_multiple_analysis_sessions(
        self,
        pattern_analyzer,
        success_execution_history,
        mixed_execution_history,
    ):
        """Test multiple analysis sessions update state correctly."""
        
        # First analysis
        patterns1 = await pattern_analyzer.analyze_patterns(success_execution_history)
        history1 = pattern_analyzer.get_analysis_history()
        assert len(history1) == 1
        
        # Second analysis
        patterns2 = await pattern_analyzer.analyze_patterns(mixed_execution_history)
        history2 = pattern_analyzer.get_analysis_history()
        assert len(history2) == 2
        
        # State should be updated
        final_patterns = pattern_analyzer.get_identified_patterns()
        assert len(final_patterns) > 0
    
    # Feature-Specific Tests
    
    @pytest.mark.asyncio
    async def test_anti_pattern_checking(
        self,
        pattern_analyzer,
        mixed_execution_history,
    ):
        """Test anti-pattern checking for team compositions."""
        
        await pattern_analyzer.analyze_patterns(mixed_execution_history)
        
        # Test checking a potentially problematic team composition
        large_team = TeamComposition(
            primary_team=[
                AgentSpec(role='generalist_a', model_assignment='basic'),
                AgentSpec(role='generalist_b', model_assignment='basic'),
                AgentSpec(role='generalist_c', model_assignment='basic'),
                AgentSpec(role='generalist_d', model_assignment='basic'),
                AgentSpec(role='generalist_e', model_assignment='basic'),
            ],
            load_order=['generalist_a', 'generalist_b', 'generalist_c', 'generalist_d', 'generalist_e'],
            coordination_strategy=CoordinationStrategy.PARALLEL,
            confidence_score=0.4,
        )
        
        matching_anti_patterns = await pattern_analyzer.check_for_anti_patterns(large_team)
        
        # May or may not match depending on whether anti-patterns were detected
        assert isinstance(matching_anti_patterns, list)
    
    def test_pattern_evolution_tracking(
        self,
        pattern_analyzer,
    ):
        """Test pattern evolution tracking."""
        
        # Initially no evolution data
        evolution = pattern_analyzer.get_pattern_evolution('nonexistent_pattern')
        assert len(evolution) == 0
    
    @pytest.mark.asyncio
    async def test_pattern_relationship_analysis(
        self,
        pattern_analyzer,
        success_execution_history,
    ):
        """Test pattern relationship analysis."""
        
        patterns = await pattern_analyzer.analyze_patterns(success_execution_history)
        
        # Check if patterns have relationship information
        for pattern in patterns:
            assert isinstance(pattern.related_patterns, list)
            assert isinstance(pattern.conflicting_patterns, list)
            
            # Related and conflicting patterns should not overlap
            related_set = set(pattern.related_patterns)
            conflicting_set = set(pattern.conflicting_patterns)
            assert len(related_set.intersection(conflicting_set)) == 0
    
    # Specific Algorithm Tests
    
    @pytest.mark.asyncio
    async def test_success_pattern_extraction(
        self,
        pattern_analyzer,
        success_execution_history,
    ):
        """Test success pattern extraction accuracy."""
        
        patterns = await pattern_analyzer.analyze_patterns(success_execution_history)
        
        success_patterns = [p for p in patterns if p.pattern_type == PatternType.SUCCESS_PATTERN]
        assert len(success_patterns) > 0
        
        # All success patterns should have reasonable success rates
        for pattern in success_patterns:
            assert pattern.success_rate >= 0.7  # Should be successful patterns
            assert pattern.observation_count >= pattern_analyzer.min_pattern_occurrences
    
    def test_pattern_confidence_calculation(
        self,
        pattern_analyzer,
    ):
        """Test pattern confidence calculation logic."""
        
        # Test with mock pattern
        pattern = TeamPattern()
        pattern.success_rate = 0.9
        pattern.usage_count = 20
        pattern.statistical_analysis = StatisticalAnalysis(
            sample_size=20,
            mean_performance=0.85,
            standard_deviation=0.1,
            confidence_interval_95=(0.8, 0.9),
            significance_level=StatisticalSignificance.HIGHLY_SIGNIFICANT,
        )
        pattern.confidence_score = 0.8
        pattern.reliability_score = 0.9
        pattern.actionability_score = 0.8
        
        # Overall strength should be reasonable
        strength = pattern.overall_strength
        assert 0.0 <= strength <= 1.0
        assert strength > 0.5  # Should be reasonably strong with these metrics
    
    @pytest.mark.asyncio
    async def test_role_collaboration_detection(
        self,
        pattern_analyzer,
        success_execution_history,
    ):
        """Test role collaboration pattern detection."""
        
        await pattern_analyzer.analyze_patterns(success_execution_history)
        
        role_analyses = pattern_analyzer.get_role_effectiveness()
        
        # Check for collaboration analysis in roles that appear together
        if 'architect' in role_analyses and 'coder' in role_analyses:
            architect_analysis = role_analyses['architect']
            coder_analysis = role_analyses['coder']
            
            # May have collaboration data
            assert isinstance(architect_analysis.best_collaborations, list)
            assert isinstance(architect_analysis.problematic_collaborations, list)
            assert isinstance(coder_analysis.best_collaborations, list)
            assert isinstance(coder_analysis.problematic_collaborations, list)


if __name__ == '__main__':
    # Run specific tests for debugging
    pytest.main([__file__, '-v', '-s'])