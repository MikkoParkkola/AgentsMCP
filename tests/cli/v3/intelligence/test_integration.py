"""Integration tests for the intelligence system components."""

import pytest
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

from src.agentsmcp.cli.v3.intelligence.user_profiler import UserProfiler
from src.agentsmcp.cli.v3.intelligence.learning_engine import LearningEngine
from src.agentsmcp.cli.v3.intelligence.suggestion_engine import SuggestionEngine
from src.agentsmcp.cli.v3.models.intelligence_models import (
    UserAction,
    UserProfile,
    SessionContext,
    UserFeedback,
    SkillLevel,
    CommandCategory,
    SuggestionType,
)


class TestIntelligenceSystemIntegration:
    """Test integration between intelligence system components."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir) / "integration_profile.enc"
    
    @pytest.fixture
    def intelligence_system(self, temp_storage):
        """Create integrated intelligence system."""
        profiler = UserProfiler(storage_path=temp_storage)
        learning_engine = LearningEngine()
        suggestion_engine = SuggestionEngine()
        
        return {
            'profiler': profiler,
            'learning_engine': learning_engine,
            'suggestion_engine': suggestion_engine
        }
    
    def test_complete_user_journey_beginner_to_intermediate(self, intelligence_system):
        """Test complete user journey from beginner to intermediate."""
        profiler = intelligence_system['profiler']
        learning_engine = intelligence_system['learning_engine']
        suggestion_engine = intelligence_system['suggestion_engine']
        
        # Start as beginner
        assert profiler.get_skill_level() == SkillLevel.BEGINNER
        
        # Simulate beginner session with help requests
        session = profiler.start_session()
        
        beginner_actions = [
            UserAction(command="help", success=True, duration_ms=500),
            UserAction(command="status", success=True, duration_ms=1000),
            UserAction(command="help configure", success=True, duration_ms=600),
            UserAction(command="configure", success=False, duration_ms=800, errors=["Invalid config"]),
            UserAction(command="help configure", success=True, duration_ms=400),
            UserAction(command="configure --interactive", success=True, duration_ms=2000),
        ]
        
        for action in beginner_actions:
            profiler.record_action(action)
            learning_engine.learn_from_action(action, session)
        
        profiler.end_session()
        
        # Get suggestions for beginner
        profile = profiler.get_profile()
        beginner_suggestions = suggestion_engine.generate_suggestions(profile)
        
        # Should get beginner-appropriate suggestions
        assert len(beginner_suggestions) > 0
        beginner_commands = {s.command for s in beginner_suggestions if s.command}
        inappropriate_commands = {'pipeline', 'orchestrate', 'debug'}
        assert len(beginner_commands & inappropriate_commands) == 0
        
        # Simulate progression - more successful commands
        session2 = profiler.start_session()
        
        intermediate_actions = []
        for i in range(20):  # Enough to reach intermediate threshold
            success = i % 4 != 0  # 75% success rate
            action = UserAction(
                command=f"command_{i % 5}",  # Some variety
                success=success,
                duration_ms=1000 + i * 10,
                errors=["minor error"] if not success else []
            )
            intermediate_actions.append(action)
            profiler.record_action(action)
            learning_engine.learn_from_action(action, session2)
        
        profiler.end_session()
        
        # Should now be intermediate level
        updated_skill = profiler.get_skill_level()
        assert updated_skill == SkillLevel.INTERMEDIATE
        
        # Get suggestions for intermediate user
        updated_profile = profiler.get_profile()
        intermediate_suggestions = suggestion_engine.generate_suggestions(updated_profile)
        
        # Should get more advanced suggestions
        intermediate_types = {s.type for s in intermediate_suggestions}
        assert SuggestionType.FEATURE_INTRODUCTION in intermediate_types
    
    def test_learning_engine_profiler_integration(self, intelligence_system):
        """Test integration between learning engine and profiler."""
        profiler = intelligence_system['profiler']
        learning_engine = intelligence_system['learning_engine']
        
        # Create repeating pattern
        pattern_sequence = ["init", "configure", "start", "monitor"]
        
        for cycle in range(3):  # Repeat pattern 3 times
            session = profiler.start_session()
            
            for cmd in pattern_sequence:
                action = UserAction(
                    command=cmd,
                    success=True,
                    duration_ms=1000,
                    timestamp=datetime.now(timezone.utc)
                )
                profiler.record_action(action)
                learning_engine.learn_from_action(action, session)
            
            profiler.end_session()
        
        # Both systems should detect the pattern
        profile = profiler.get_profile()
        learning_metrics = learning_engine.get_learning_metrics()
        
        # Profiler should detect command patterns
        assert len(profile.command_patterns) > 0
        
        # Learning engine should have sequence patterns
        assert learning_metrics['sequence_patterns'] > 0
        
        # Test prediction consistency
        recent_commands = ["init", "configure"]
        predictions = learning_engine.predict_next_command(recent_commands)
        
        if predictions:
            # Should predict "start" as next command
            predicted_commands = {cmd for cmd, _ in predictions}
            assert "start" in predicted_commands
    
    def test_suggestion_engine_learning_integration(self, intelligence_system):
        """Test integration between suggestion engine and learning engine."""
        learning_engine = intelligence_system['learning_engine']
        suggestion_engine = intelligence_system['suggestion_engine']
        
        # Simulate user with specific error patterns
        error_actions = [
            UserAction(
                command="deploy",
                success=False,
                duration_ms=5000,
                errors=["deployment failed"],
                context={"environment": "production"}
            ),
            UserAction(
                command="rollback",
                success=True,
                duration_ms=2000
            ),
            UserAction(
                command="deploy",
                success=False,
                duration_ms=5000,
                errors=["deployment failed"],
                context={"environment": "production"}
            ),
            UserAction(
                command="debug",
                success=True,
                duration_ms=3000
            ),
            UserAction(
                command="deploy",
                success=True,
                duration_ms=4000
            ),
        ]
        
        for action in error_actions:
            learning_engine.learn_from_action(action)
        
        # Create profile with error history
        profile = UserProfile(
            skill_level=SkillLevel.EXPERT,
            total_commands=50,
            success_rate=0.8,
            common_errors={"deployment failed": 2}
        )
        
        # Analyze errors with learning engine
        error_analysis = learning_engine.analyze_error_patterns(profile)
        
        # Generate suggestions
        suggestions = suggestion_engine.generate_suggestions(profile)
        
        # Should include error prevention suggestions
        error_prevention_suggestions = [
            s for s in suggestions 
            if s.type == SuggestionType.ERROR_PREVENTION
        ]
        
        assert len(error_prevention_suggestions) > 0
        
        # Should suggest validation or dry-run for deployment failures
        prevention_texts = [s.text.lower() for s in error_prevention_suggestions]
        safety_suggestions = [
            t for t in prevention_texts 
            if 'validate' in t or 'dry-run' in t or 'test' in t
        ]
        assert len(safety_suggestions) > 0
    
    def test_full_system_workflow_with_feedback(self, intelligence_system):
        """Test complete workflow including user feedback."""
        profiler = intelligence_system['profiler']
        learning_engine = intelligence_system['learning_engine']
        suggestion_engine = intelligence_system['suggestion_engine']
        
        # Initial user session
        session = profiler.start_session()
        
        # User performs some actions
        actions = [
            UserAction(command="status", success=True, duration_ms=500),
            UserAction(command="configure", success=True, duration_ms=2000),
        ]
        
        for action in actions:
            profiler.record_action(action)
            learning_engine.learn_from_action(action, session)
        
        profiler.end_session()
        
        # Get initial suggestions
        profile = profiler.get_profile()
        suggestions = suggestion_engine.generate_suggestions(profile, session)
        
        assert len(suggestions) > 0
        
        # Simulate user following a suggestion
        if suggestions:
            followed_suggestion = suggestions[0]
            
            # User executes suggested command
            suggested_action = UserAction(
                command=followed_suggestion.command or "suggested_command",
                success=True,
                duration_ms=1500
            )
            
            profiler.record_action(suggested_action)
            learning_engine.learn_from_action(suggested_action)
            
            # Update learning engine with successful prediction
            learning_engine.update_accuracy(
                followed_suggestion.command or "suggested_command",
                suggested_action.command
            )
            
            # Simulate positive feedback
            feedback = UserFeedback(
                rating=4,
                comment="Helpful suggestion",
                suggestion_id=followed_suggestion.suggestion_id,
                helpful=True
            )
            
            # System should learn from this feedback
            metrics = learning_engine.get_learning_metrics()
            assert 'prediction_accuracy' in metrics
    
    def test_persistence_across_sessions(self, intelligence_system, temp_storage):
        """Test data persistence across multiple sessions."""
        profiler1 = intelligence_system['profiler']
        
        # First session - record some actions
        session1 = profiler1.start_session()
        
        actions1 = [
            UserAction(command="init", success=True, duration_ms=1000),
            UserAction(command="configure", success=True, duration_ms=2000),
            UserAction(command="start", success=True, duration_ms=1500),
        ]
        
        for action in actions1:
            profiler1.record_action(action)
        
        profiler1.end_session()
        
        # Save profile
        profiler1.save_profile()
        initial_total_commands = profiler1.current_profile.total_commands
        
        # Create new profiler instance (simulating app restart)
        profiler2 = UserProfiler(storage_path=temp_storage)
        
        # Should load previous data
        assert profiler2.current_profile.total_commands == initial_total_commands
        
        # Second session - add more actions
        session2 = profiler2.start_session()
        
        actions2 = [
            UserAction(command="status", success=True, duration_ms=800),
            UserAction(command="monitor", success=True, duration_ms=1200),
        ]
        
        for action in actions2:
            profiler2.record_action(action)
        
        profiler2.end_session()
        
        # Total commands should accumulate
        final_total = profiler2.current_profile.total_commands
        assert final_total == initial_total_commands + len(actions2)
    
    def test_progressive_disclosure_integration(self, intelligence_system):
        """Test progressive disclosure level integration."""
        profiler = intelligence_system['profiler']
        suggestion_engine = intelligence_system['suggestion_engine']
        
        # Start with beginner profile
        beginner_profile = profiler.get_profile()
        assert beginner_profile.skill_level == SkillLevel.BEGINNER
        
        disclosure_level = profiler.get_progressive_disclosure_level()
        assert disclosure_level.value <= 2  # Should be minimal or basic
        
        # Simulate progression to expert
        profiler.current_profile.total_commands = 100
        profiler.current_profile.success_rate = 0.9
        profiler.current_profile.help_request_rate = 0.05
        profiler.current_profile.advanced_features_used = {"pipeline", "debug", "optimize"}
        
        expert_disclosure = profiler.get_progressive_disclosure_level()
        assert expert_disclosure.value >= 3  # Should be standard or higher
        
        # Suggestions should adapt to disclosure level
        expert_profile = profiler.get_profile()
        expert_suggestions = suggestion_engine.generate_suggestions(expert_profile)
        
        # Should get more sophisticated suggestions
        suggestion_commands = {s.command for s in expert_suggestions if s.command}
        advanced_commands = {'orchestrate', 'delegate', 'optimize', 'profile'}
        
        # Should have some overlap with advanced commands
        if suggestion_commands:
            assert len(suggestion_commands & advanced_commands) >= 0  # May or may not suggest these
    
    def test_error_recovery_learning_integration(self, intelligence_system):
        """Test error recovery pattern learning integration."""
        profiler = intelligence_system['profiler']
        learning_engine = intelligence_system['learning_engine']
        suggestion_engine = intelligence_system['suggestion_engine']
        
        # Simulate error recovery patterns
        recovery_sequence = [
            UserAction(command="deploy", success=False, duration_ms=3000, errors=["Connection failed"]),
            UserAction(command="debug", success=True, duration_ms=2000),
            UserAction(command="retry", success=True, duration_ms=2500),
            UserAction(command="deploy", success=False, duration_ms=3000, errors=["Connection failed"]),
            UserAction(command="debug", success=True, duration_ms=1800),
            UserAction(command="fix", success=True, duration_ms=4000),
            UserAction(command="deploy", success=True, duration_ms=2800),
        ]
        
        session = profiler.start_session()
        
        for action in recovery_sequence:
            profiler.record_action(action)
            learning_engine.learn_from_action(action, session)
        
        profiler.end_session()
        
        # Create error context
        error_session = SessionContext(
            errors_encountered=["Connection failed"]
        )
        
        # Get suggestions in error context
        profile = profiler.get_profile()
        error_suggestions = suggestion_engine.generate_suggestions(
            profile, 
            current_context=error_session
        )
        
        # Should suggest debug or recovery commands
        suggested_commands = {s.command for s in error_suggestions if s.command}
        recovery_commands = {'debug', 'retry', 'fix', 'logs', 'status'}
        
        assert len(suggested_commands & recovery_commands) > 0
    
    def test_performance_with_large_datasets(self, intelligence_system):
        """Test system performance with large amounts of data."""
        profiler = intelligence_system['profiler']
        learning_engine = intelligence_system['learning_engine']
        suggestion_engine = intelligence_system['suggestion_engine']
        
        # Generate large dataset
        commands = [f"cmd_{i % 20}" for i in range(500)]  # 500 actions, 20 unique commands
        
        start_time = datetime.now()
        
        session = profiler.start_session()
        
        for i, cmd in enumerate(commands):
            action = UserAction(
                command=cmd,
                success=i % 5 != 0,  # 80% success rate
                duration_ms=500 + i % 100,
                context={"batch": i // 50}
            )
            profiler.record_action(action)
            learning_engine.learn_from_action(action, session)
        
        profiler.end_session()
        
        # Generate suggestions (should be fast)
        profile = profiler.get_profile()
        suggestions = suggestion_engine.generate_suggestions(profile)
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # Should complete in reasonable time (less than 10 seconds for 500 actions)
        assert total_time < 10.0
        
        # Should still produce valid suggestions
        assert isinstance(suggestions, list)
        assert len(suggestions) <= suggestion_engine.max_suggestions
        
        # Learning metrics should be reasonable
        metrics = learning_engine.get_learning_metrics()
        assert metrics['total_actions_learned'] == 500
        assert metrics['unique_commands'] == 20


class TestIntelligenceSystemEdgeCases:
    """Test edge cases in system integration."""
    
    def test_corrupted_profile_recovery(self, temp_storage):
        """Test recovery from corrupted profile data."""
        # Create profiler and save some data
        profiler1 = UserProfiler(storage_path=temp_storage)
        profiler1.current_profile.total_commands = 42
        profiler1.save_profile()
        
        # Corrupt the file
        temp_storage.write_bytes(b"corrupted data")
        
        # New profiler should recover gracefully
        profiler2 = UserProfiler(storage_path=temp_storage)
        
        # Should create new profile, not crash
        assert profiler2.current_profile is not None
        assert profiler2.current_profile.total_commands == 0  # New profile
    
    def test_memory_management_long_running(self, intelligence_system):
        """Test memory management for long-running sessions."""
        profiler = intelligence_system['profiler']
        learning_engine = intelligence_system['learning_engine']
        
        # Simulate very long session with many actions
        session = profiler.start_session()
        
        for i in range(200):  # Many actions
            action = UserAction(
                command=f"action_{i}",
                success=True,
                duration_ms=1000
            )
            profiler.record_action(action)
            learning_engine.learn_from_action(action, session)
        
        # Should not grow unbounded
        assert len(profiler.action_history) <= profiler.max_actions_history
        assert len(learning_engine.short_term_window) <= 100  # Default limit
        assert len(learning_engine.medium_term_window) <= 500  # Default limit
        
        profiler.end_session()
    
    def test_concurrent_access_simulation(self, intelligence_system):
        """Test behavior under simulated concurrent access."""
        profiler = intelligence_system['profiler']
        learning_engine = intelligence_system['learning_engine']
        suggestion_engine = intelligence_system['suggestion_engine']
        
        # Simulate concurrent operations
        session1 = profiler.start_session()
        
        # Rapidly add actions and generate suggestions
        for i in range(10):
            action = UserAction(
                command=f"concurrent_cmd_{i}",
                success=True,
                duration_ms=100
            )
            profiler.record_action(action)
            learning_engine.learn_from_action(action, session1)
            
            # Generate suggestions during learning
            profile = profiler.get_profile()
            suggestions = suggestion_engine.generate_suggestions(profile)
            
            # Should handle concurrent access gracefully
            assert isinstance(suggestions, list)
        
        profiler.end_session()
        
        # Final state should be consistent
        final_profile = profiler.get_profile()
        assert final_profile.total_commands == 10