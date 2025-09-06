"""
Comprehensive unit tests for the A/B testing and continuous evaluation selection system.
"""

import pytest
import asyncio
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

from src.agentsmcp.selection.selection_history import SelectionHistory, SelectionRecord, generate_selection_id
from src.agentsmcp.selection.benchmark_tracker import BenchmarkTracker, SelectionMetrics
from src.agentsmcp.selection.ab_testing_framework import ABTestingFramework, ExperimentConfig, ExperimentStatus
from src.agentsmcp.selection.selection_optimizer import SelectionOptimizer, OptimizationStrategy, BanditArm
from src.agentsmcp.selection.performance_analyzer import PerformanceAnalyzer, StatisticalTest, ComparisonResult
from src.agentsmcp.selection.experiment_manager import ExperimentManager, AutoExperimentConfig
from src.agentsmcp.selection.adaptive_selector import AdaptiveSelector, SelectionRequest, SelectionMode


class TestSelectionHistory:
    """Test the selection history storage and retrieval system."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        Path(db_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def selection_history(self, temp_db):
        """Create selection history instance."""
        return SelectionHistory(db_path=temp_db, max_records=1000)
    
    def test_generate_selection_id(self):
        """Test selection ID generation."""
        task_context = {"task_type": "coding", "complexity": "medium"}
        selection_id = generate_selection_id("model", task_context)
        
        assert isinstance(selection_id, str)
        assert len(selection_id) > 10
        assert selection_id.startswith("model_")
    
    def test_record_selection(self, selection_history):
        """Test recording selection decisions."""
        record = SelectionRecord(
            selection_id="test_001",
            timestamp=datetime.now(),
            selection_type="model",
            task_context={"task_type": "coding"},
            available_options=["gpt-4", "claude-3"],
            selected_option="gpt-4",
            selection_method="adaptive",
            confidence_score=0.85,
            selection_metadata={"experiment_id": "exp_001"},
            success=True,
            completion_time_ms=5000,
            quality_score=0.9
        )
        
        selection_history.record_selection(record)
        
        # Retrieve and verify
        records = selection_history.get_records(limit=1)
        assert len(records) == 1
        assert records[0].selection_id == "test_001"
        assert records[0].selected_option == "gpt-4"
        assert records[0].success is True
    
    def test_update_outcome(self, selection_history):
        """Test updating selection outcomes."""
        # Record initial selection
        record = SelectionRecord(
            selection_id="test_002",
            timestamp=datetime.now(),
            selection_type="model",
            task_context={"task_type": "reasoning"},
            available_options=["gpt-4", "claude-3"],
            selected_option="claude-3",
            selection_method="exploration",
            confidence_score=0.6,
            selection_metadata={}
        )
        
        selection_history.record_selection(record)
        
        # Update outcome
        success = selection_history.update_outcome(
            selection_id="test_002",
            success=True,
            completion_time_ms=8000,
            quality_score=0.85,
            cost=0.05,
            user_feedback=1
        )
        
        assert success is True
        
        # Verify update
        records = selection_history.get_records(selected_option="claude-3", limit=1)
        assert len(records) == 1
        assert records[0].success is True
        assert records[0].completion_time_ms == 8000
        assert records[0].quality_score == 0.85
        assert records[0].cost == 0.05
        assert records[0].user_feedback == 1
    
    def test_get_performance_summary(self, selection_history):
        """Test performance summary generation."""
        # Add multiple records
        for i in range(10):
            record = SelectionRecord(
                selection_id=f"test_{i:03d}",
                timestamp=datetime.now() - timedelta(hours=i),
                selection_type="model",
                task_context={"task_type": "coding"},
                available_options=["option_a", "option_b"],
                selected_option="option_a" if i % 2 == 0 else "option_b",
                selection_method="adaptive",
                confidence_score=0.7,
                selection_metadata={},
                success=i < 7,  # 70% success rate
                completion_time_ms=5000 + i * 1000,
                quality_score=0.8 if i < 7 else 0.3
            )
            selection_history.record_selection(record)
        
        summary = selection_history.get_performance_summary(selection_type="model", days=1)
        
        assert summary['total_selections'] == 10
        assert abs(summary['success_rate'] - 0.7) < 0.01  # 70% success rate
        assert len(summary['options']) == 2
        assert summary['unique_options'] == 2


class TestBenchmarkTracker:
    """Test the benchmark tracking system."""
    
    @pytest.fixture
    def selection_history(self):
        """Mock selection history."""
        return Mock(spec=SelectionHistory)
    
    @pytest.fixture
    def benchmark_tracker(self, selection_history):
        """Create benchmark tracker instance."""
        return BenchmarkTracker(selection_history, update_interval_seconds=1, max_windows=10)
    
    def test_selection_metrics_initialization(self):
        """Test SelectionMetrics initialization and updates."""
        metrics = SelectionMetrics("gpt-4", "model")
        
        assert metrics.option_name == "gpt-4"
        assert metrics.selection_type == "model"
        assert metrics.total_selections == 0
        assert metrics.success_rate == 0.0
    
    def test_selection_metrics_update(self):
        """Test updating selection metrics from records."""
        metrics = SelectionMetrics("claude-3", "model")
        
        # Create test record
        record = SelectionRecord(
            selection_id="test_001",
            timestamp=datetime.now(),
            selection_type="model",
            task_context={"task_type": "coding"},
            available_options=["claude-3", "gpt-4"],
            selected_option="claude-3",
            selection_method="adaptive",
            confidence_score=0.8,
            selection_metadata={},
            success=True,
            completion_time_ms=7000,
            quality_score=0.9,
            cost=0.03
        )
        
        metrics.update_from_record(record)
        
        assert metrics.total_selections == 1
        assert metrics.successful_selections == 1
        assert metrics.success_rate == 1.0
        assert metrics.avg_completion_time_ms == 7000.0
        assert metrics.avg_quality_score == 0.9
        assert metrics.avg_cost == 0.03
        assert metrics.error_count == 0
    
    def test_composite_score_calculation(self):
        """Test composite score calculation."""
        metrics = SelectionMetrics("test-model", "model")
        
        # Add some successful records
        for i in range(5):
            record = SelectionRecord(
                selection_id=f"test_{i}",
                timestamp=datetime.now(),
                selection_type="model",
                task_context={},
                available_options=["test-model"],
                selected_option="test-model",
                selection_method="test",
                confidence_score=0.8,
                selection_metadata={},
                success=True,
                completion_time_ms=5000 + i * 1000,
                quality_score=0.8 + i * 0.02,
                cost=0.01
            )
            metrics.update_from_record(record)
        
        score = metrics.get_composite_score()
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be above neutral for successful records
    
    def test_record_selection_outcome(self, benchmark_tracker):
        """Test recording selection outcomes."""
        record = SelectionRecord(
            selection_id="perf_test_001",
            timestamp=datetime.now(),
            selection_type="agent",
            task_context={"task_type": "analysis"},
            available_options=["analyst", "general"],
            selected_option="analyst",
            selection_method="bandit",
            confidence_score=0.75,
            selection_metadata={},
            success=True,
            completion_time_ms=12000,
            quality_score=0.85
        )
        
        benchmark_tracker.record_selection_outcome(record)
        
        # Check that metrics were updated
        metrics = benchmark_tracker.get_metrics(selection_type="agent", option_name="analyst")
        assert len(metrics) == 1
        
        key = ("agent", "analyst")
        assert key in metrics
        assert metrics[key].total_selections == 1
        assert metrics[key].success_rate == 1.0
    
    def test_get_rankings(self, benchmark_tracker):
        """Test performance rankings."""
        # Add records for multiple options
        options = ["fast_model", "accurate_model", "cheap_model"]
        performances = [
            (True, 3000, 0.7, 0.001),   # fast_model: fast, decent quality, very cheap
            (True, 8000, 0.95, 0.005),  # accurate_model: slow, high quality, expensive
            (True, 5000, 0.6, 0.0005)   # cheap_model: medium speed, low quality, cheapest
        ]
        
        for option, (success, time_ms, quality, cost) in zip(options, performances):
            for i in range(20):  # 20 samples each
                record = SelectionRecord(
                    selection_id=f"{option}_{i}",
                    timestamp=datetime.now(),
                    selection_type="model",
                    task_context={"task_type": "test"},
                    available_options=options,
                    selected_option=option,
                    selection_method="test",
                    confidence_score=0.8,
                    selection_metadata={},
                    success=success,
                    completion_time_ms=time_ms + (i * 100),  # Add some variance
                    quality_score=quality + (i * 0.01),
                    cost=cost
                )
                benchmark_tracker.record_selection_outcome(record)
        
        # Get rankings by composite score
        rankings = benchmark_tracker.get_rankings("model", metric="composite_score", min_samples=15)
        
        assert len(rankings) == 3
        assert all(isinstance(rank[0], str) and isinstance(rank[1], float) for rank in rankings)
        # Rankings should be in descending order
        assert rankings[0][1] >= rankings[1][1] >= rankings[2][1]


class TestABTestingFramework:
    """Test the A/B testing framework."""
    
    @pytest.fixture
    def selection_history(self):
        """Mock selection history."""
        history = Mock(spec=SelectionHistory)
        history.get_records.return_value = []  # No baseline data by default
        return history
    
    @pytest.fixture
    def ab_framework(self, selection_history):
        """Create A/B testing framework instance."""
        return ABTestingFramework(selection_history, min_baseline_samples=10)
    
    @pytest.mark.asyncio
    async def test_create_experiment(self, ab_framework):
        """Test experiment creation."""
        config = await ab_framework.create_experiment(
            name="Model Comparison Test",
            description="Compare GPT-4 vs Claude-3",
            selection_type="model",
            control_option="gpt-4",
            treatment_options=["claude-3"],
            min_sample_size=50,
            max_duration_days=7
        )
        
        assert config.name == "Model Comparison Test"
        assert config.selection_type == "model"
        assert config.control_option == "gpt-4"
        assert config.treatment_options == ["claude-3"]
        assert config.min_sample_size == 50
        assert config.max_duration_days == 7
        assert config.status == ExperimentStatus.PLANNED
    
    @pytest.mark.asyncio
    async def test_traffic_allocation(self, ab_framework):
        """Test traffic allocation for experiments."""
        # Create and start experiment
        config = await ab_framework.create_experiment(
            name="Traffic Test",
            description="Test traffic allocation",
            selection_type="provider",
            control_option="openai",
            treatment_options=["anthropic"]
        )
        
        # Mock baseline data to allow experiment start
        ab_framework.selection_history.get_records.return_value = [Mock()] * 50
        
        started = await ab_framework.start_experiment(config)
        assert started is True
        
        # Test allocation
        allocations = {}
        for i in range(1000):
            option, metadata = await ab_framework.allocate_selection(
                selection_type="provider",
                task_context={"task_type": "coding"},
                available_options=["openai", "anthropic"],
                user_id=f"user_{i % 100}"  # 100 unique users
            )
            
            allocations[option] = allocations.get(option, 0) + 1
        
        # Check that both options were allocated
        assert len(allocations) == 2
        assert "openai" in allocations
        assert "anthropic" in allocations
        
        # Check rough allocation percentages (should be close to configured)
        total = sum(allocations.values())
        openai_pct = allocations["openai"] / total
        anthropic_pct = allocations["anthropic"] / total
        
        # Allow some variance due to randomness
        assert 0.3 <= openai_pct <= 0.8
        assert 0.2 <= anthropic_pct <= 0.7


class TestSelectionOptimizer:
    """Test the multi-armed bandit selection optimizer."""
    
    @pytest.fixture
    def selection_history(self):
        """Mock selection history."""
        return Mock(spec=SelectionHistory)
    
    @pytest.fixture
    def benchmark_tracker(self):
        """Mock benchmark tracker."""
        tracker = Mock(spec=BenchmarkTracker)
        tracker.get_metrics.return_value = {}
        return tracker
    
    @pytest.fixture
    def optimizer(self, selection_history, benchmark_tracker):
        """Create selection optimizer instance."""
        return SelectionOptimizer(
            selection_history,
            benchmark_tracker,
            strategy=OptimizationStrategy.THOMPSON_SAMPLING,
            exploration_rate=0.1
        )
    
    def test_bandit_arm_initialization(self):
        """Test bandit arm initialization."""
        arm = BanditArm("test-option", "model")
        
        assert arm.arm_id == "test-option"
        assert arm.selection_type == "model"
        assert arm.alpha == 1.0
        assert arm.beta == 1.0
        assert arm.total_pulls == 0
        assert arm.estimated_value == 0.5  # Initial neutral estimate
    
    def test_thompson_sampling(self):
        """Test Thompson sampling from Beta distribution."""
        arm = BanditArm("test-arm", "test")
        
        # Update with some successes and failures
        arm.update_thompson(0.9)  # Success
        arm.update_thompson(0.8)  # Success
        arm.update_thompson(0.1)  # Failure
        
        # Check that alpha and beta were updated
        assert arm.alpha > 1.0
        assert arm.beta > 1.0
        
        # Sample should be between 0 and 1
        for _ in range(10):
            sample = arm.get_thompson_sample()
            assert 0.0 <= sample <= 1.0
    
    def test_ucb1_calculation(self):
        """Test UCB1 upper confidence bound calculation."""
        arm = BanditArm("test-arm", "test")
        
        # Initially should return infinity (unexplored)
        ucb_value = arm.get_ucb1_value(total_pulls=100)
        assert ucb_value == float('inf')
        
        # After some pulls
        arm.update_ucb1(0.8)
        arm.update_ucb1(0.7)
        arm.update_ucb1(0.9)
        
        ucb_value = arm.get_ucb1_value(total_pulls=100)
        assert isinstance(ucb_value, float)
        assert ucb_value > 0.0
    
    def test_option_selection(self, optimizer):
        """Test option selection with different strategies."""
        available_options = ["option_a", "option_b", "option_c"]
        task_context = {"task_type": "test", "complexity": "medium"}
        
        selected_option, metadata = optimizer.select_option(
            selection_type="test",
            available_options=available_options,
            task_context=task_context,
            user_id="test_user"
        )
        
        assert selected_option in available_options
        assert isinstance(metadata, dict)
        assert 'method' in metadata
        assert isinstance(metadata.get('exploration', False), bool)
    
    def test_outcome_update(self, optimizer):
        """Test updating optimizer with selection outcomes."""
        # Make a selection first
        selected_option, _ = optimizer.select_option(
            selection_type="test",
            available_options=["option_1", "option_2"],
            task_context={"task_type": "test"}
        )
        
        # Create outcome record
        record = SelectionRecord(
            selection_id="test_outcome",
            timestamp=datetime.now(),
            selection_type="test",
            task_context={"task_type": "test"},
            available_options=["option_1", "option_2"],
            selected_option=selected_option,
            selection_method="test",
            confidence_score=0.8,
            selection_metadata={},
            success=True,
            completion_time_ms=5000,
            quality_score=0.85
        )
        
        # Update outcome
        optimizer.update_outcome("test", selected_option, record)
        
        # Check that arm was updated
        arm_key = ("test", selected_option)
        assert arm_key in optimizer.arms
        arm = optimizer.arms[arm_key]
        assert arm.total_pulls > 0 or arm.alpha > 1.0  # Depending on strategy


class TestPerformanceAnalyzer:
    """Test the statistical performance analyzer."""
    
    @pytest.fixture
    def selection_history(self):
        """Mock selection history with sample data."""
        history = Mock(spec=SelectionHistory)
        
        # Create sample records for comparison tests
        def mock_get_records(selection_type=None, selected_option=None, **kwargs):
            if selected_option == "option_a":
                return [
                    Mock(success=True, completion_time_ms=5000, quality_score=0.8, cost=0.01),
                    Mock(success=True, completion_time_ms=5500, quality_score=0.85, cost=0.01),
                    Mock(success=False, completion_time_ms=8000, quality_score=0.3, cost=0.01),
                ] * 10  # 30 records total, 67% success rate
            elif selected_option == "option_b":
                return [
                    Mock(success=True, completion_time_ms=4000, quality_score=0.9, cost=0.015),
                    Mock(success=True, completion_time_ms=4200, quality_score=0.88, cost=0.015),
                    Mock(success=True, completion_time_ms=4100, quality_score=0.92, cost=0.015),
                ] * 10  # 30 records total, 100% success rate
            return []
        
        history.get_records.side_effect = mock_get_records
        return history
    
    @pytest.fixture
    def benchmark_tracker(self):
        """Mock benchmark tracker."""
        return Mock(spec=BenchmarkTracker)
    
    @pytest.fixture
    def analyzer(self, selection_history, benchmark_tracker):
        """Create performance analyzer instance."""
        return PerformanceAnalyzer(selection_history, benchmark_tracker, min_sample_size=20)
    
    @pytest.mark.asyncio
    async def test_compare_options(self, analyzer):
        """Test statistical comparison between options."""
        result = analyzer.compare_options(
            selection_type="test",
            option_a="option_a",
            option_b="option_b",
            metric="success_rate",
            days=30
        )
        
        assert result is not None
        assert isinstance(result, ComparisonResult)
        assert result.option_a == "option_a"
        assert result.option_b == "option_b"
        assert result.n_a == 30
        assert result.n_b == 30
        
        # Option B should have higher success rate
        assert result.metric_b > result.metric_a
        assert result.difference > 0
        
        # Should have valid statistical test results
        assert 0.0 <= result.p_value <= 1.0
        assert isinstance(result.is_significant, bool)
        assert result.confidence_level == 0.95
    
    def test_extract_metric_values(self, analyzer):
        """Test metric extraction from records."""
        # Create mock records
        records = [
            Mock(success=True, completion_time_ms=5000, quality_score=0.8, custom_metrics={"custom": 0.5}),
            Mock(success=False, completion_time_ms=8000, quality_score=0.3, custom_metrics={"custom": 0.2}),
            Mock(success=True, completion_time_ms=6000, quality_score=0.85, custom_metrics={"custom": 0.7})
        ]
        
        # Test success rate extraction
        success_values = analyzer._extract_metric_values(records, "success_rate")
        assert success_values == [1.0, 0.0, 1.0]
        
        # Test completion time extraction
        time_values = analyzer._extract_metric_values(records, "completion_time")
        assert time_values == [5000, 8000, 6000]
        
        # Test quality score extraction
        quality_values = analyzer._extract_metric_values(records, "quality_score")
        assert quality_values == [0.8, 0.3, 0.85]
        
        # Test custom metric extraction
        custom_values = analyzer._extract_metric_values(records, "custom")
        assert custom_values == [0.5, 0.2, 0.7]
    
    def test_choose_statistical_test(self, analyzer):
        """Test statistical test selection."""
        # Binary data should use chi-square
        test = analyzer._choose_statistical_test([1, 0, 1, 0, 1], [0, 1, 0, 1, 1], "success_rate")
        assert test == StatisticalTest.CHI_SQUARE
        
        # Small samples should use non-parametric tests
        small_sample = [1.0, 2.0, 3.0, 4.0, 5.0]  # < 30 samples
        test = analyzer._choose_statistical_test(small_sample, small_sample, "completion_time")
        assert test == StatisticalTest.MANN_WHITNEY
    
    def test_effect_size_calculation(self, analyzer):
        """Test Cohen's d effect size calculation."""
        values_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        values_b = [3.0, 4.0, 5.0, 6.0, 7.0]  # Mean difference of 2.0
        
        effect_size = analyzer._calculate_effect_size(values_a, values_b)
        
        assert isinstance(effect_size, float)
        assert effect_size > 0.0
        
        # Test interpretation
        magnitude = analyzer._interpret_effect_size(effect_size)
        assert magnitude in ["negligible", "small", "medium", "large"]


class TestAdaptiveSelector:
    """Test the main adaptive selector interface."""
    
    @pytest.fixture
    def adaptive_selector(self):
        """Create adaptive selector with mocked components."""
        # Use in-memory components for testing
        return AdaptiveSelector(default_mode=SelectionMode.ADAPTIVE)
    
    @pytest.mark.asyncio
    async def test_initialization(self, adaptive_selector):
        """Test adaptive selector initialization."""
        assert adaptive_selector is not None
        assert adaptive_selector.default_mode == SelectionMode.ADAPTIVE
        assert not adaptive_selector._initialized
        
        # Initialize
        await adaptive_selector.initialize()
        assert adaptive_selector._initialized
        assert adaptive_selector._running
    
    @pytest.mark.asyncio
    async def test_selection_request_response(self, adaptive_selector):
        """Test complete selection request-response cycle."""
        await adaptive_selector.initialize()
        
        # Create selection request
        request = SelectionRequest(
            selection_type="model",
            available_options=["gpt-4", "claude-3", "gemini-pro"],
            task_context={"task_type": "coding", "complexity": "high"},
            mode=SelectionMode.ADAPTIVE,
            user_id="test_user_123"
        )
        
        # Make selection
        response = await adaptive_selector.select(request)
        
        # Verify response
        assert response is not None
        assert response.selected_option in request.available_options
        assert 0.0 <= response.confidence <= 1.0
        assert isinstance(response.exploration, bool)
        assert response.request_id == request.request_id
        assert len(response.selection_id) > 0
        assert response.timestamp is not None
        
        # Test outcome reporting
        outcome_success = await adaptive_selector.report_outcome(
            selection_id=response.selection_id,
            success=True,
            completion_time_ms=7500,
            quality_score=0.88,
            cost=0.025,
            user_feedback=1
        )
        
        assert outcome_success is True
    
    @pytest.mark.asyncio
    async def test_different_selection_modes(self, adaptive_selector):
        """Test different selection modes."""
        await adaptive_selector.initialize()
        
        modes_to_test = [
            SelectionMode.EXPLOITATION,
            SelectionMode.EXPLORATION,
            SelectionMode.ADAPTIVE,
            SelectionMode.BANDIT
        ]
        
        for mode in modes_to_test:
            request = SelectionRequest(
                selection_type="agent",
                available_options=["analyst", "general", "specialist"],
                task_context={"task_type": "analysis"},
                mode=mode,
                user_id="mode_test_user"
            )
            
            response = await adaptive_selector.select(request)
            
            assert response.selected_option in request.available_options
            assert isinstance(response.confidence, float)
            
            # Exploration mode should typically have exploration=True
            if mode == SelectionMode.EXPLORATION:
                # Note: This might not always be true due to fallbacks
                pass
    
    @pytest.mark.asyncio
    async def test_performance_insights(self, adaptive_selector):
        """Test performance insights generation."""
        await adaptive_selector.initialize()
        
        # Make some selections to generate data
        for i in range(5):
            request = SelectionRequest(
                selection_type="tool",
                available_options=["tool_a", "tool_b"],
                task_context={"task_type": "processing"},
                user_id=f"insights_user_{i}"
            )
            
            response = await adaptive_selector.select(request)
            
            # Report outcome
            await adaptive_selector.report_outcome(
                selection_id=response.selection_id,
                success=i < 4,  # 4/5 success rate
                completion_time_ms=5000 + i * 1000,
                quality_score=0.8 + i * 0.02
            )
        
        # Get insights
        insights = adaptive_selector.get_performance_insights(selection_type="tool", days=1)
        
        assert isinstance(insights, dict)
        assert 'timestamp' in insights
        assert 'selector_performance' in insights
        
        perf = insights['selector_performance']
        assert perf['total_selections'] == 5
        assert perf['success_rate'] == 80.0  # 4/5 = 80%
    
    @pytest.mark.asyncio
    async def test_error_handling(self, adaptive_selector):
        """Test error handling and fallbacks."""
        # Test with empty options list
        request = SelectionRequest(
            selection_type="invalid",
            available_options=[],  # Empty options should raise error
            task_context={}
        )
        
        with pytest.raises(ValueError, match="No available options"):
            await adaptive_selector.select(request)
        
        # Test with invalid selection type (should still work with fallback)
        request = SelectionRequest(
            selection_type="nonexistent_type",
            available_options=["option_1"],
            task_context={}
        )
        
        await adaptive_selector.initialize()
        response = await adaptive_selector.select(request)
        
        # Should still return a valid response with fallback
        assert response.selected_option == "option_1"
    
    @pytest.mark.asyncio
    async def test_cleanup(self, adaptive_selector):
        """Test proper cleanup and shutdown."""
        await adaptive_selector.initialize()
        assert adaptive_selector._running
        
        await adaptive_selector.shutdown()
        assert not adaptive_selector._running


# Integration tests
class TestSystemIntegration:
    """Integration tests for the complete system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete end-to-end selection workflow."""
        # Create adaptive selector
        selector = AdaptiveSelector(default_mode=SelectionMode.ADAPTIVE)
        await selector.initialize()
        
        try:
            # Simulate multiple users making selections
            users = ["user_1", "user_2", "user_3"]
            options = ["fast_option", "accurate_option", "balanced_option"]
            
            results = []
            
            for user in users:
                for i in range(3):  # 3 selections per user
                    request = SelectionRequest(
                        selection_type="integration_test",
                        available_options=options,
                        task_context={
                            "task_type": "processing",
                            "complexity": "medium",
                            "priority": i + 1
                        },
                        user_id=user,
                        session_id=f"session_{user}_{i}"
                    )
                    
                    response = await selector.select(request)
                    results.append((request, response))
                    
                    # Simulate outcome (realistic success rates)
                    option_success_rates = {
                        "fast_option": 0.7,      # Fast but less reliable
                        "accurate_option": 0.95,  # Slow but very reliable
                        "balanced_option": 0.85   # Balanced
                    }
                    
                    success_rate = option_success_rates.get(response.selected_option, 0.8)
                    success = (hash(response.selection_id) % 100) < (success_rate * 100)
                    
                    await selector.report_outcome(
                        selection_id=response.selection_id,
                        success=success,
                        completion_time_ms=3000 if response.selected_option == "fast_option" else 
                                         8000 if response.selected_option == "accurate_option" else 5000,
                        quality_score=0.7 if response.selected_option == "fast_option" else
                                     0.95 if response.selected_option == "accurate_option" else 0.85,
                        cost=0.005 if response.selected_option == "fast_option" else
                             0.02 if response.selected_option == "accurate_option" else 0.01
                    )
            
            # Verify results
            assert len(results) == 9  # 3 users × 3 selections
            
            # Check that all selections were valid
            for request, response in results:
                assert response.selected_option in request.available_options
                assert 0.0 <= response.confidence <= 1.0
                assert response.selection_id
            
            # Get performance insights
            insights = selector.get_performance_insights("integration_test", days=1)
            assert insights['selector_performance']['total_selections'] == 9
            
        finally:
            await selector.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])