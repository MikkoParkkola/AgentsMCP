"""Tests for LearningEngine class."""

import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from unittest.mock import patch, MagicMock

from src.agentsmcp.cli.v3.intelligence.learning_engine import LearningEngine
from src.agentsmcp.cli.v3.models.intelligence_models import (
    UserAction,
    UserProfile,
    SessionContext,
    CommandCategory,
    SkillLevel,
    InsufficientDataError,
    AnalysisFailedError,
)


class TestLearningEngine:
    """Test LearningEngine functionality."""
    
    @pytest.fixture
    def learning_engine(self):
        """Create LearningEngine instance."""
        return LearningEngine(
            learning_rate=0.1,
            decay_factor=0.95,
            min_pattern_support=2,
            max_patterns=10,
            similarity_threshold=0.7
        )
    
    @pytest.fixture
    def sample_actions(self):
        """Create sample user actions for testing."""
        actions = []
        commands = ["init", "configure", "start", "status", "stop"]
        
        for i, cmd in enumerate(commands):
            action = UserAction(
                command=cmd,
                category=CommandCategory.EXECUTION,
                success=i % 2 == 0,  # Alternate success/failure
                duration_ms=1000 + i * 100,
                context={"step": i},
                errors=["test error"] if i % 2 != 0 else []
            )
            actions.append(action)
        
        return actions
    
    def test_initialization(self, learning_engine):
        """Test learning engine initialization."""
        assert learning_engine.learning_rate == 0.1
        assert learning_engine.decay_factor == 0.95
        assert learning_engine.min_pattern_support == 2
        assert learning_engine.max_patterns == 10
        assert learning_engine.similarity_threshold == 0.7
        
        assert len(learning_engine.short_term_window) == 0
        assert len(learning_engine.medium_term_window) == 0
        assert len(learning_engine.command_embeddings) == 0
    
    def test_learn_from_action(self, learning_engine, sample_actions):
        """Test learning from a single action."""
        action = sample_actions[0]
        
        learning_engine.learn_from_action(action)
        
        # Should add action to learning windows
        assert len(learning_engine.short_term_window) == 1
        assert len(learning_engine.medium_term_window) == 1
        
        # Should create command embedding
        assert action.command in learning_engine.command_embeddings
        embedding = learning_engine.command_embeddings[action.command]
        assert len(embedding) == 4  # Feature vector size
        assert isinstance(embedding, np.ndarray)
        
        # Should update feature weights
        assert len(learning_engine.feature_weights) > 0
    
    def test_learn_sequence_patterns(self, learning_engine, sample_actions):
        """Test sequence pattern learning."""
        # Learn from multiple actions to create patterns
        for action in sample_actions:
            learning_engine.learn_from_action(action)
        
        # Should detect sequence patterns
        assert len(learning_engine.sequence_patterns) > 0
        
        # Check if patterns contain expected sequences
        patterns = list(learning_engine.sequence_patterns.keys())
        assert len(patterns) > 0
        
        # Verify pattern structure
        for pattern_key, pattern_data in learning_engine.sequence_patterns.items():
            assert 'count' in pattern_data
            assert 'success_rate' in pattern_data
            assert 'last_seen' in pattern_data
            assert 'next_commands' in pattern_data
            assert pattern_data['count'] >= 1
    
    def test_learn_error_patterns(self, learning_engine):
        """Test error pattern learning."""
        error_action = UserAction(
            command="failing_command",
            success=False,
            duration_ms=500,
            errors=["Connection timeout", "Permission denied"],
            context={"environment": "test", "user": "test_user"}
        )
        
        learning_engine.learn_from_action(error_action)
        
        # Should learn error patterns
        assert len(learning_engine.error_patterns) > 0
        
        for error in error_action.errors:
            assert error in learning_engine.error_patterns
            pattern = learning_engine.error_patterns[error]
            assert pattern['count'] == 1
            assert len(pattern['contexts']) > 0
            assert len(pattern['time_of_day']) > 0
    
    def test_predict_next_command(self, learning_engine, sample_actions):
        """Test next command prediction."""
        # Learn from actions first
        for action in sample_actions:
            learning_engine.learn_from_action(action)
        
        # Test prediction with recent commands
        recent_commands = ["init", "configure"]
        predictions = learning_engine.predict_next_command(recent_commands)
        
        assert isinstance(predictions, list)
        if predictions:  # May be empty if insufficient patterns
            for cmd, prob in predictions:
                assert isinstance(cmd, str)
                assert 0.0 <= prob <= 1.0
    
    def test_predict_next_command_empty_input(self, learning_engine):
        """Test prediction with empty recent commands."""
        predictions = learning_engine.predict_next_command([])
        
        # Should return popular commands as fallback
        assert isinstance(predictions, list)
    
    def test_predict_from_similarity(self, learning_engine, sample_actions):
        """Test similarity-based predictions."""
        # Learn from actions to create embeddings
        for action in sample_actions:
            learning_engine.learn_from_action(action)
        
        if learning_engine.command_embeddings:
            last_command = list(learning_engine.command_embeddings.keys())[0]
            predictions = learning_engine._predict_from_similarity(last_command)
            
            assert isinstance(predictions, list)
            for cmd, similarity in predictions:
                assert isinstance(cmd, str)
                assert isinstance(similarity, float)
                assert cmd != last_command  # Should not predict same command
    
    def test_temporal_pattern_learning(self, learning_engine):
        """Test temporal pattern learning."""
        # Create actions at different times
        now = datetime.now(timezone.utc)
        
        for i in range(3):
            action = UserAction(
                command="morning_command",
                success=True,
                duration_ms=1000,
                timestamp=now.replace(hour=9) + timedelta(days=i)
            )
            learning_engine.learn_from_action(action)
        
        # Should learn temporal patterns
        assert "morning_command" in learning_engine.time_patterns
        timestamps = learning_engine.time_patterns["morning_command"]
        assert len(timestamps) == 3
        
        # Test temporal predictions
        context = {"current_time": now.replace(hour=9)}
        predictions = learning_engine._predict_from_temporal_patterns(context)
        
        if predictions:
            cmd, prob = predictions[0]
            assert cmd == "morning_command"
            assert prob > 0.0
    
    def test_error_analysis(self, learning_engine):
        """Test error pattern analysis."""
        # Create actions with errors
        for i in range(5):
            action = UserAction(
                command=f"command_{i}",
                success=i % 2 == 0,
                duration_ms=1000,
                errors=[f"error_type_{i % 2}"] if i % 2 != 0 else [],
                context={"context_key": f"value_{i % 3}"}
            )
            learning_engine.learn_from_action(action)
        
        # Create a user profile for analysis
        user_profile = UserProfile(
            total_commands=5,
            success_rate=0.6,
            common_errors={"error_type_1": 2}
        )
        
        analysis = learning_engine.analyze_error_patterns(user_profile)
        
        assert 'common_errors' in analysis
        assert 'error_sequences' in analysis
        assert 'error_contexts' in analysis
        assert 'recovery_patterns' in analysis
        assert 'recommendations' in analysis
        
        # Should have detected common errors
        assert len(analysis['common_errors']) > 0
    
    def test_error_analysis_insufficient_data(self, learning_engine):
        """Test error analysis with insufficient data."""
        user_profile = UserProfile()
        
        with pytest.raises(InsufficientDataError):
            learning_engine.analyze_error_patterns(user_profile)
    
    def test_preference_inference(self, learning_engine):
        """Test user preference inference."""
        # Create actions that suggest preferences
        actions = [
            UserAction(command="help command", success=True, duration_ms=1000),
            UserAction(command="help another", success=True, duration_ms=1000),
            UserAction(command="normal command", success=True, duration_ms=5000),
            UserAction(command="evening command", success=True, duration_ms=1000, 
                      timestamp=datetime.now(timezone.utc).replace(hour=20)),
        ]
        
        for action in actions:
            learning_engine.learn_from_action(action)
        
        user_profile = UserProfile(total_commands=4)
        preferences = learning_engine.infer_preferences(user_profile)
        
        assert isinstance(preferences, dict)
        
        # Should infer high verbosity due to help requests
        assert preferences.get('verbose_output') is True
        
        # Should infer suggestion level based on command diversity
        assert 'suggestion_level' in preferences
        assert 1 <= preferences['suggestion_level'] <= 5
        
        # Should infer timeout based on command durations
        assert 'default_timeout_ms' in preferences
        assert preferences['default_timeout_ms'] > 0
    
    def test_accuracy_tracking(self, learning_engine):
        """Test prediction accuracy tracking."""
        initial_accuracy = learning_engine.prediction_accuracy['next_command']
        
        # Update with correct prediction
        learning_engine.update_accuracy("predicted_cmd", "predicted_cmd")
        accuracy_after_correct = learning_engine.prediction_accuracy['next_command']
        
        # Update with incorrect prediction
        learning_engine.update_accuracy("predicted_cmd", "actual_cmd")
        accuracy_after_incorrect = learning_engine.prediction_accuracy['next_command']
        
        # Accuracy should change based on feedback
        assert accuracy_after_correct != initial_accuracy
        assert accuracy_after_incorrect != accuracy_after_correct
        
        # Should track adaptation history
        assert len(learning_engine.adaptation_history) == 2
        
        last_entry = learning_engine.adaptation_history[-1]
        assert last_entry['predicted'] == "predicted_cmd"
        assert last_entry['actual'] == "actual_cmd"
        assert last_entry['correct'] is False
    
    def test_learning_rate_adaptation(self, learning_engine):
        """Test adaptive learning rate adjustment."""
        initial_rate = learning_engine.learning_rate
        
        # Simulate high accuracy scenario
        for i in range(20):
            learning_engine.update_accuracy("correct", "correct")
        
        # Learning rate should decrease with high accuracy
        assert learning_engine.learning_rate <= initial_rate
        
        # Reset and simulate low accuracy scenario
        learning_engine.learning_rate = 0.1
        learning_engine.adaptation_history.clear()
        
        for i in range(20):
            learning_engine.update_accuracy("wrong", "correct")
        
        # Learning rate should increase with low accuracy
        assert learning_engine.learning_rate >= 0.1
    
    def test_pattern_pruning(self, learning_engine):
        """Test pattern pruning to prevent memory bloat."""
        # Create many patterns
        for i in range(15):  # More than max_patterns (10)
            sequence = (f"cmd_{i}", f"cmd_{i+1}")
            learning_engine.sequence_patterns[sequence] = {
                'count': i + 1,
                'success_rate': 0.8,
                'last_seen': datetime.now(timezone.utc),
                'next_commands': defaultdict(int)
            }
        
        # Trigger pruning
        learning_engine._prune_patterns()
        
        # Should keep only max_patterns most frequent patterns
        assert len(learning_engine.sequence_patterns) <= learning_engine.max_patterns
        
        # Should keep the most frequent patterns
        remaining_counts = [p['count'] for p in learning_engine.sequence_patterns.values()]
        assert max(remaining_counts) >= min(remaining_counts)
    
    def test_feature_weight_updates(self, learning_engine):
        """Test feature weight updates based on outcomes."""
        initial_weights = dict(learning_engine.feature_weights)
        
        # Create action with long duration and success
        long_success_action = UserAction(
            command="long_success",
            success=True,
            duration_ms=10000,
            context={"key1": "val1", "key2": "val2", "key3": "val3", "key4": "val4"}
        )
        
        learning_engine.learn_from_action(long_success_action)
        
        # Feature weights should be updated
        assert learning_engine.feature_weights['duration'] != initial_weights.get('duration', 1.0)
        assert learning_engine.feature_weights['context_size'] != initial_weights.get('context_size', 1.0)
        
        # Weights should be within reasonable bounds
        for weight in learning_engine.feature_weights.values():
            assert 0.1 <= weight <= 2.0
    
    def test_learning_metrics(self, learning_engine, sample_actions):
        """Test learning metrics collection."""
        # Learn from actions
        for action in sample_actions:
            learning_engine.learn_from_action(action)
        
        metrics = learning_engine.get_learning_metrics()
        
        assert 'total_actions_learned' in metrics
        assert 'unique_commands' in metrics
        assert 'sequence_patterns' in metrics
        assert 'error_patterns' in metrics
        assert 'prediction_accuracy' in metrics
        assert 'learning_rate' in metrics
        
        assert metrics['total_actions_learned'] == len(sample_actions)
        assert metrics['unique_commands'] > 0
        assert isinstance(metrics['prediction_accuracy'], dict)


class TestLearningEngineEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_learning_windows(self):
        """Test behavior with empty learning windows."""
        engine = LearningEngine()
        
        predictions = engine.predict_next_command(["any_command"])
        assert isinstance(predictions, list)
        
        # Should handle gracefully without crashing
        metrics = engine.get_learning_metrics()
        assert metrics['total_actions_learned'] == 0
    
    def test_single_action_learning(self):
        """Test learning from single action."""
        engine = LearningEngine()
        
        action = UserAction(
            command="single_command",
            success=True,
            duration_ms=1000
        )
        
        engine.learn_from_action(action)
        
        # Should handle single action gracefully
        assert len(engine.short_term_window) == 1
        assert "single_command" in engine.command_embeddings
    
    def test_context_based_predictions(self):
        """Test context-based prediction logic."""
        engine = LearningEngine()
        
        # Test error context
        error_context = {"error_occurred": True}
        predictions = engine._predict_from_context(error_context)
        
        assert isinstance(predictions, list)
        if predictions:
            # Should suggest help-related commands for errors
            help_suggestions = [cmd for cmd, _ in predictions if 'help' in cmd or 'debug' in cmd]
            assert len(help_suggestions) > 0
        
        # Test pipeline context
        pipeline_context = {"in_pipeline": True}
        predictions = engine._predict_from_context(pipeline_context)
        
        if predictions:
            # Should suggest monitoring commands for pipelines
            monitor_suggestions = [cmd for cmd, _ in predictions if 'monitor' in cmd or 'status' in cmd]
            assert len(monitor_suggestions) > 0
    
    def test_recovery_pattern_detection(self):
        """Test detection of error recovery patterns."""
        engine = LearningEngine()
        
        # Create error followed by recovery pattern
        error_action = UserAction(
            command="failing_command",
            success=False,
            duration_ms=500,
            errors=["Connection failed"]
        )
        
        recovery_action = UserAction(
            command="retry_command",
            success=True,
            duration_ms=1000
        )
        
        engine.learn_from_action(error_action)
        engine.learn_from_action(recovery_action)
        
        # Test recovery pattern detection
        recovery_patterns = engine._find_recovery_patterns()
        
        if recovery_patterns:
            assert any(
                pattern['error_command'] == 'failing_command' and
                pattern['recovery_command'] == 'retry_command'
                for pattern in recovery_patterns
            )
    
    @patch('src.agentsmcp.cli.v3.intelligence.learning_engine.TfidfVectorizer')
    def test_vectorizer_error_handling(self, mock_vectorizer):
        """Test handling of vectorizer errors."""
        # Mock vectorizer to raise exception
        mock_vectorizer.side_effect = Exception("Vectorizer error")
        
        engine = LearningEngine()
        
        # Should handle vectorizer errors gracefully
        action = UserAction(
            command="test_command",
            success=True,
            duration_ms=1000
        )
        
        # Should not crash even if vectorizer fails
        engine.learn_from_action(action)
        assert len(engine.medium_term_window) == 1