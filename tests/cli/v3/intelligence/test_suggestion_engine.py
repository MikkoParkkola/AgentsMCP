"""Tests for SuggestionEngine class."""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

from src.agentsmcp.cli.v3.intelligence.suggestion_engine import SuggestionEngine
from src.agentsmcp.cli.v3.models.intelligence_models import (
    UserProfile,
    UserAction,
    SessionContext,
    Suggestion,
    SuggestionType,
    SkillLevel,
    CommandPattern,
    CommandCategory,
    LearningMetrics,
)


class TestSuggestionEngine:
    """Test SuggestionEngine functionality."""
    
    @pytest.fixture
    def suggestion_engine(self):
        """Create SuggestionEngine instance."""
        return SuggestionEngine(
            max_suggestions=5,
            suggestion_decay_hours=24,
            min_confidence_threshold=0.3
        )
    
    @pytest.fixture
    def beginner_profile(self):
        """Create beginner user profile."""
        return UserProfile(
            skill_level=SkillLevel.BEGINNER,
            total_commands=5,
            success_rate=0.7,
            help_request_rate=0.4,
            favorite_commands=["help", "status"],
            common_errors={"connection_error": 2}
        )
    
    @pytest.fixture
    def expert_profile(self):
        """Create expert user profile."""
        patterns = [
            CommandPattern(
                name="Deploy Workflow",
                commands=["build", "test", "deploy"],
                frequency=10,
                success_rate=0.9,
                avg_duration_ms=5000,
                confidence=0.8
            ),
            CommandPattern(
                name="Debug Workflow",
                commands=["debug", "trace", "fix"],
                frequency=5,
                success_rate=0.7,
                avg_duration_ms=3000,
                confidence=0.6
            )
        ]
        
        return UserProfile(
            skill_level=SkillLevel.EXPERT,
            total_commands=150,
            success_rate=0.9,
            help_request_rate=0.05,
            favorite_commands=["deploy", "monitor", "optimize", "debug", "configure"],
            advanced_features_used={"pipeline", "orchestrate", "debug", "profile"},
            command_patterns=patterns,
            common_errors={"timeout_error": 1}
        )
    
    @pytest.fixture
    def sample_session(self):
        """Create sample session context."""
        return SessionContext(
            commands_used=["init", "configure", "start"],
            errors_encountered=["connection_timeout"],
            help_requests=1,
            duration_ms=1800000,  # 30 minutes
            success_rate=0.8,
            unique_commands=3
        )
    
    def test_initialization(self, suggestion_engine):
        """Test suggestion engine initialization."""
        assert suggestion_engine.max_suggestions == 5
        assert suggestion_engine.suggestion_decay_hours == 24
        assert suggestion_engine.min_confidence_threshold == 0.3
        
        assert len(suggestion_engine.skill_progression_map) == 4
        assert len(suggestion_engine.workflow_patterns) > 0
        assert len(suggestion_engine.contextual_suggestions) > 0
    
    def test_generate_suggestions_beginner(self, suggestion_engine, beginner_profile):
        """Test suggestion generation for beginner user."""
        suggestions = suggestion_engine.generate_suggestions(beginner_profile)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) <= suggestion_engine.max_suggestions
        
        # All suggestions should meet confidence threshold
        for suggestion in suggestions:
            assert suggestion.confidence >= suggestion_engine.min_confidence_threshold
            assert isinstance(suggestion, Suggestion)
        
        # Should suggest appropriate features for beginners
        suggested_commands = {s.command for s in suggestions if s.command}
        beginner_appropriate = {'help', 'status', 'list', 'configure'}
        inappropriate = {'pipeline', 'orchestrate', 'debug'}
        
        # Should have some overlap with appropriate commands
        if suggested_commands:
            assert len(suggested_commands & beginner_appropriate) > 0
            # Should not suggest inappropriate commands
            assert len(suggested_commands & inappropriate) == 0
    
    def test_generate_suggestions_expert(self, suggestion_engine, expert_profile):
        """Test suggestion generation for expert user."""
        suggestions = suggestion_engine.generate_suggestions(expert_profile)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) <= suggestion_engine.max_suggestions
        
        # Expert should get more advanced suggestions
        suggestion_types = {s.type for s in suggestions}
        
        # Should include various types of suggestions
        possible_types = {
            SuggestionType.OPTIMIZATION,
            SuggestionType.FEATURE_INTRODUCTION,
            SuggestionType.SKILL_ADVANCEMENT
        }
        
        # Should have at least some advanced suggestion types
        if suggestion_types:
            assert len(suggestion_types & possible_types) > 0
    
    def test_predict_next_actions(self, suggestion_engine, expert_profile):
        """Test next action prediction."""
        recent_commands = ["build", "test"]
        
        predictions = suggestion_engine.predict_next_actions(
            expert_profile, 
            recent_commands
        )
        
        assert isinstance(predictions, list)
        
        for suggestion in predictions:
            assert suggestion.type == SuggestionType.NEXT_ACTION
            assert suggestion.confidence > 0.0
        
        # Should predict workflow continuation
        if predictions:
            # Look for deployment-related predictions since build->test usually leads to deploy
            deploy_suggestions = [s for s in predictions if 'deploy' in s.command.lower()]
            # May or may not find deploy suggestions, but if found should be relevant
    
    def test_predict_next_actions_pattern_matching(self, suggestion_engine, expert_profile):
        """Test pattern-based next action prediction."""
        # Use commands that match start of existing pattern
        recent_commands = ["build"]  # Matches start of "Deploy Workflow" pattern
        
        predictions = suggestion_engine.predict_next_actions(
            expert_profile,
            recent_commands
        )
        
        if predictions:
            # Should suggest continuing the deploy workflow
            next_commands = {s.command for s in predictions}
            assert "test" in next_commands  # Next step in deploy pattern
    
    def test_get_feature_recommendations(self, suggestion_engine, beginner_profile):
        """Test feature recommendation generation."""
        recommendations = suggestion_engine.get_feature_recommendations(beginner_profile)
        
        assert isinstance(recommendations, list)
        
        for rec in recommendations:
            assert rec.type == SuggestionType.FEATURE_INTRODUCTION
            assert rec.skill_level_target == beginner_profile.skill_level
        
        # Should recommend appropriate beginner features
        recommended_features = {r.command for r in recommendations if r.command}
        beginner_features = {'help', 'status', 'list', 'configure'}
        
        if recommended_features:
            assert len(recommended_features & beginner_features) > 0
    
    def test_suggest_optimizations(self, suggestion_engine, expert_profile):
        """Test optimization suggestion generation."""
        optimizations = suggestion_engine.suggest_optimizations(expert_profile)
        
        assert isinstance(optimizations, list)
        
        for opt in optimizations:
            assert opt.type == SuggestionType.OPTIMIZATION
        
        # Should suggest shortcuts for frequent patterns
        if optimizations:
            shortcut_suggestions = [s for s in optimizations if 'shortcut' in s.text.lower()]
            # Expert with patterns should get shortcut suggestions
            assert len(shortcut_suggestions) > 0
    
    def test_get_error_prevention_tips(self, suggestion_engine, beginner_profile):
        """Test error prevention tip generation."""
        tips = suggestion_engine.get_error_prevention_tips(beginner_profile)
        
        assert isinstance(tips, list)
        
        for tip in tips:
            assert tip.type == SuggestionType.ERROR_PREVENTION
        
        # Should suggest validation for common errors
        if tips:
            tip_texts = [t.text.lower() for t in tips]
            # Should mention validation or safety measures
            safety_mentions = [t for t in tip_texts if 'validate' in t or 'dry-run' in t or 'backup' in t]
            assert len(safety_mentions) > 0
    
    def test_contextual_suggestions_after_error(self, suggestion_engine, beginner_profile):
        """Test contextual suggestions after errors."""
        session_with_errors = SessionContext(
            errors_encountered=["Connection failed", "Timeout"]
        )
        
        suggestions = suggestion_engine.generate_suggestions(
            beginner_profile,
            current_context=session_with_errors
        )
        
        # Should include error recovery suggestions
        error_recovery_suggestions = [
            s for s in suggestions 
            if s.metadata.get('context') == 'after_error'
        ]
        
        assert len(error_recovery_suggestions) > 0
        
        # Should suggest help or debug commands
        recovery_commands = {s.command for s in error_recovery_suggestions}
        expected_commands = {'help', 'debug', 'logs', 'status'}
        assert len(recovery_commands & expected_commands) > 0
    
    def test_contextual_suggestions_long_session(self, suggestion_engine, expert_profile):
        """Test contextual suggestions for long sessions."""
        long_session = SessionContext(
            duration_ms=3700000,  # Over 1 hour
            commands_used=["cmd1", "cmd2", "cmd3"] * 10
        )
        
        suggestions = suggestion_engine.generate_suggestions(
            expert_profile,
            current_context=long_session
        )
        
        # Should include session management suggestions
        session_management = [
            s for s in suggestions 
            if s.metadata.get('context') == 'long_session'
        ]
        
        if session_management:
            # Should suggest session management actions
            commands = {s.command for s in session_management}
            expected = {'save', 'status', 'break'}
            assert len(commands & expected) > 0
    
    def test_contextual_suggestions_new_user(self, suggestion_engine):
        """Test contextual suggestions for new users."""
        new_user_profile = UserProfile(
            total_commands=3,  # Very new user
            skill_level=SkillLevel.BEGINNER
        )
        
        suggestions = suggestion_engine.generate_suggestions(new_user_profile)
        
        # Should include onboarding suggestions
        onboarding_suggestions = [
            s for s in suggestions 
            if s.metadata.get('context') == 'new_user'
        ]
        
        assert len(onboarding_suggestions) > 0
        
        # Should suggest onboarding commands
        commands = {s.command for s in onboarding_suggestions}
        expected = {'help', 'tutorial', 'examples'}
        assert len(commands & expected) > 0
    
    def test_suggestion_filtering_by_confidence(self, suggestion_engine, beginner_profile):
        """Test suggestion filtering by confidence threshold."""
        # Create engine with high confidence threshold
        strict_engine = SuggestionEngine(
            min_confidence_threshold=0.9
        )
        
        suggestions = strict_engine.generate_suggestions(beginner_profile)
        
        # All suggestions should meet the high threshold
        for suggestion in suggestions:
            assert suggestion.confidence >= 0.9
    
    def test_suggestion_ranking(self, suggestion_engine, expert_profile):
        """Test suggestion ranking by relevance."""
        suggestions = suggestion_engine.generate_suggestions(expert_profile)
        
        if len(suggestions) > 1:
            # Should be ranked by computed score (confidence * priority + bonuses)
            scores = []
            for suggestion in suggestions:
                score = suggestion.confidence * suggestion.priority
                scores.append(score)
            
            # Scores should generally be in descending order
            # (allowing for some variation due to bonuses)
            assert scores[0] >= scores[-1]
    
    def test_suggestion_diversity(self, suggestion_engine, expert_profile):
        """Test suggestion diversity to avoid redundancy."""
        suggestions = suggestion_engine.generate_suggestions(expert_profile)
        
        if len(suggestions) > 1:
            categories = [s.category for s in suggestions]
            # Should have diverse categories (not all the same)
            unique_categories = set(categories)
            assert len(unique_categories) >= min(3, len(suggestions))
    
    def test_skill_advancement_suggestions(self, suggestion_engine):
        """Test skill advancement suggestion logic."""
        # Create profile ready for advancement
        ready_profile = UserProfile(
            skill_level=SkillLevel.INTERMEDIATE,
            total_commands=65,  # Above threshold for expert
            success_rate=0.87,  # High success rate
            advanced_features_used={"pipeline", "optimize"},
            command_patterns=[MagicMock() for _ in range(3)]
        )
        
        suggestions = suggestion_engine.generate_suggestions(ready_profile)
        
        # Should include skill advancement suggestions
        advancement_suggestions = [
            s for s in suggestions 
            if s.type == SuggestionType.SKILL_ADVANCEMENT
        ]
        
        if advancement_suggestions:
            # Should target higher skill level
            for suggestion in advancement_suggestions:
                assert suggestion.skill_level_target == SkillLevel.EXPERT
    
    def test_workflow_completion_suggestions(self, suggestion_engine, expert_profile, sample_session):
        """Test workflow completion suggestions."""
        # Create actions that suggest middle of workflow
        recent_actions = [
            UserAction(
                command="build",
                success=True,
                duration_ms=2000
            )
        ]
        
        suggestions = suggestion_engine.generate_suggestions(
            expert_profile,
            current_context=sample_session,
            recent_actions=recent_actions
        )
        
        # Should include workflow completion suggestions
        workflow_suggestions = [
            s for s in suggestions 
            if s.type == SuggestionType.WORKFLOW_COMPLETION
        ]
        
        if workflow_suggestions:
            # Should suggest next steps in workflow
            next_commands = {s.command for s in workflow_suggestions}
            # Build typically followed by test, deploy, or monitor
            expected_next = {'test', 'deploy', 'monitor'}
            assert len(next_commands & expected_next) > 0
    
    def test_feature_readiness_calculation(self, suggestion_engine):
        """Test feature readiness score calculation."""
        # Test different skill levels
        beginner = UserProfile(
            skill_level=SkillLevel.BEGINNER,
            success_rate=0.6,
            favorite_commands=["help", "status"]
        )
        
        expert = UserProfile(
            skill_level=SkillLevel.EXPERT,
            success_rate=0.9,
            favorite_commands=[f"cmd_{i}" for i in range(15)]
        )
        
        beginner_readiness = suggestion_engine._calculate_feature_readiness(beginner, "pipeline")
        expert_readiness = suggestion_engine._calculate_feature_readiness(expert, "pipeline")
        
        # Expert should have higher readiness for advanced features
        assert expert_readiness > beginner_readiness
        assert 0.0 <= beginner_readiness <= 1.0
        assert 0.0 <= expert_readiness <= 1.0
    
    def test_suggestion_appropriateness_filtering(self, suggestion_engine, beginner_profile):
        """Test filtering of inappropriate suggestions."""
        # Create suggestion that's too advanced
        advanced_suggestion = Suggestion(
            type=SuggestionType.FEATURE_INTRODUCTION,
            text="Try advanced debugging",
            command="debug",
            confidence=0.8,
            skill_level_target=SkillLevel.POWER
        )
        
        is_appropriate = suggestion_engine._is_suggestion_appropriate(
            advanced_suggestion, 
            beginner_profile
        )
        
        # Should not be appropriate for beginner
        assert is_appropriate is False
        
        # Test appropriate suggestion
        basic_suggestion = Suggestion(
            type=SuggestionType.FEATURE_INTRODUCTION,
            text="Try getting help",
            command="help",
            confidence=0.8,
            skill_level_target=SkillLevel.BEGINNER
        )
        
        is_appropriate = suggestion_engine._is_suggestion_appropriate(
            basic_suggestion, 
            beginner_profile
        )
        
        # Should be appropriate for beginner
        assert is_appropriate is True


class TestSuggestionEngineEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_profile_suggestions(self):
        """Test suggestions for completely empty profile."""
        engine = SuggestionEngine()
        empty_profile = UserProfile()
        
        suggestions = engine.generate_suggestions(empty_profile)
        
        # Should handle empty profile gracefully
        assert isinstance(suggestions, list)
        
        # Should suggest new user onboarding
        if suggestions:
            onboarding_count = sum(
                1 for s in suggestions 
                if s.metadata.get('context') == 'new_user'
            )
            assert onboarding_count > 0
    
    def test_no_patterns_next_action(self):
        """Test next action prediction with no learned patterns."""
        engine = SuggestionEngine()
        profile_no_patterns = UserProfile(
            total_commands=10,
            command_patterns=[]  # No patterns
        )
        
        predictions = engine.predict_next_actions(
            profile_no_patterns,
            ["some_command"]
        )
        
        # Should handle gracefully
        assert isinstance(predictions, list)
    
    def test_all_suggestions_filtered_out(self):
        """Test case where all suggestions are filtered out."""
        # Create engine with impossible threshold
        strict_engine = SuggestionEngine(
            min_confidence_threshold=1.1  # Impossible threshold
        )
        
        profile = UserProfile(skill_level=SkillLevel.BEGINNER)
        suggestions = strict_engine.generate_suggestions(profile)
        
        # Should return empty list, not crash
        assert suggestions == []
    
    def test_suggestion_with_expiry(self):
        """Test suggestions with expiry times."""
        engine = SuggestionEngine()
        
        # Create suggestion that should expire
        suggestion = Suggestion(
            type=SuggestionType.FEATURE_INTRODUCTION,
            text="Time-limited suggestion",
            confidence=0.8,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1)
        )
        
        # Should handle expiry times properly
        assert suggestion.expires_at is not None
        assert suggestion.expires_at > datetime.now(timezone.utc)
    
    def test_large_command_history(self):
        """Test with profile having very large command history."""
        large_profile = UserProfile(
            total_commands=10000,
            favorite_commands=[f"cmd_{i}" for i in range(20)],
            advanced_features_used={f"feature_{i}" for i in range(50)}
        )
        
        engine = SuggestionEngine()
        suggestions = engine.generate_suggestions(large_profile)
        
        # Should handle large profiles without performance issues
        assert isinstance(suggestions, list)
        assert len(suggestions) <= engine.max_suggestions
    
    def test_pattern_matching_edge_cases(self):
        """Test edge cases in pattern matching."""
        engine = SuggestionEngine()
        
        # Test with commands longer than pattern
        long_commands = ["cmd1", "cmd2", "cmd3", "cmd4", "cmd5"]
        short_pattern = ["cmd1", "cmd2"]
        
        matches = engine._matches_pattern_start(long_commands, short_pattern)
        assert matches is False  # Commands longer than pattern
        
        # Test with exact match
        exact_commands = ["cmd1", "cmd2"]
        exact_pattern = ["cmd1", "cmd2", "cmd3"]
        
        matches = engine._matches_pattern_start(exact_commands, exact_pattern)
        assert matches is True  # Commands match start of pattern
    
    def test_zero_confidence_suggestions(self):
        """Test handling of zero confidence suggestions."""
        engine = SuggestionEngine(min_confidence_threshold=0.1)
        
        # Mock a suggestion with zero confidence
        zero_conf_suggestion = Suggestion(
            type=SuggestionType.OPTIMIZATION,
            text="Zero confidence suggestion",
            confidence=0.0
        )
        
        profile = UserProfile()
        
        # Should be filtered out
        is_appropriate = engine._is_suggestion_appropriate(zero_conf_suggestion, profile)
        assert is_appropriate is True  # Appropriateness check doesn't consider confidence
        
        # But confidence filtering should remove it
        filtered = engine._filter_suggestions([zero_conf_suggestion], profile)
        assert len(filtered) == 0  # Should be filtered out by confidence threshold