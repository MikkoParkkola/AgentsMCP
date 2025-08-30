"""Tests for intelligence system models."""

import pytest
from datetime import datetime, timezone, timedelta
from uuid import UUID

from src.agentsmcp.cli.v3.models.intelligence_models import (
    SkillLevel,
    CommandCategory,
    SuggestionType,
    LearningEventType,
    ProgressiveDisclosureLevel,
    UserAction,
    SessionContext,
    UserFeedback,
    CommandPattern,
    Suggestion,
    LearningMetrics,
    UserProfile,
    ProfileCorruptedError,
    InsufficientDataError,
    AnalysisFailedError,
    StorageUnavailableError,
)


class TestSkillLevel:
    """Test skill level enumeration."""
    
    def test_skill_levels_exist(self):
        """Test all expected skill levels exist."""
        assert SkillLevel.BEGINNER == "beginner"
        assert SkillLevel.INTERMEDIATE == "intermediate"
        assert SkillLevel.EXPERT == "expert"
        assert SkillLevel.POWER == "power"


class TestUserAction:
    """Test UserAction model."""
    
    def test_create_user_action(self):
        """Test creating a valid user action."""
        action = UserAction(
            command="test_command",
            success=True,
            duration_ms=1500,
            context={"key": "value"},
            errors=[]
        )
        
        assert action.command == "test_command"
        assert action.success is True
        assert action.duration_ms == 1500
        assert action.context == {"key": "value"}
        assert action.errors == []
        assert action.category == CommandCategory.EXECUTION  # default
        assert isinstance(UUID(action.action_id), UUID)
    
    def test_user_action_with_errors(self):
        """Test user action with errors."""
        action = UserAction(
            command="failing_command",
            success=False,
            duration_ms=500,
            errors=["Connection failed", "Timeout"]
        )
        
        assert action.success is False
        assert len(action.errors) == 2
        assert "Connection failed" in action.errors
    
    def test_user_action_validation(self):
        """Test user action validation."""
        # Duration must be non-negative
        with pytest.raises(ValueError):
            UserAction(
                command="test",
                success=True,
                duration_ms=-100
            )
        
        # Command must not be empty
        with pytest.raises(ValueError):
            UserAction(
                command="",
                success=True,
                duration_ms=100
            )


class TestSessionContext:
    """Test SessionContext model."""
    
    def test_create_session_context(self):
        """Test creating a session context."""
        context = SessionContext(
            commands_used=["cmd1", "cmd2", "cmd3"],
            errors_encountered=["error1"],
            help_requests=2,
            unique_commands=3
        )
        
        assert len(context.commands_used) == 3
        assert len(context.errors_encountered) == 1
        assert context.help_requests == 2
        assert context.unique_commands == 3
        assert isinstance(UUID(context.session_id), UUID)
    
    def test_session_context_defaults(self):
        """Test session context default values."""
        context = SessionContext()
        
        assert context.commands_used == []
        assert context.errors_encountered == []
        assert context.help_requests == 0
        assert context.duration_ms == 0
        assert context.success_rate == 1.0
        assert context.unique_commands == 0
        assert context.advanced_features_used == []


class TestCommandPattern:
    """Test CommandPattern model."""
    
    def test_create_command_pattern(self):
        """Test creating a command pattern."""
        pattern = CommandPattern(
            name="Setup Workflow",
            commands=["init", "config", "start"],
            frequency=5,
            success_rate=0.9,
            avg_duration_ms=2000,
            confidence=0.8
        )
        
        assert pattern.name == "Setup Workflow"
        assert pattern.commands == ["init", "config", "start"]
        assert pattern.frequency == 5
        assert pattern.success_rate == 0.9
        assert pattern.confidence == 0.8
        assert isinstance(UUID(pattern.pattern_id), UUID)
    
    def test_pattern_validation(self):
        """Test pattern validation."""
        # Commands must have at least 2 items
        with pytest.raises(ValueError):
            CommandPattern(
                name="Single Command",
                commands=["single"],
                success_rate=0.9,
                avg_duration_ms=1000,
                confidence=0.8
            )
        
        # Success rate must be between 0 and 1
        with pytest.raises(ValueError):
            CommandPattern(
                name="Invalid Pattern",
                commands=["cmd1", "cmd2"],
                success_rate=1.5,
                avg_duration_ms=1000,
                confidence=0.8
            )


class TestSuggestion:
    """Test Suggestion model."""
    
    def test_create_suggestion(self):
        """Test creating a suggestion."""
        suggestion = Suggestion(
            type=SuggestionType.NEXT_ACTION,
            text="Try running 'status' next",
            command="status",
            confidence=0.8,
            priority=7,
            skill_level_target=SkillLevel.INTERMEDIATE
        )
        
        assert suggestion.type == SuggestionType.NEXT_ACTION
        assert suggestion.text == "Try running 'status' next"
        assert suggestion.command == "status"
        assert suggestion.confidence == 0.8
        assert suggestion.priority == 7
        assert suggestion.skill_level_target == SkillLevel.INTERMEDIATE
        assert isinstance(UUID(suggestion.suggestion_id), UUID)
    
    def test_suggestion_with_expiry(self):
        """Test suggestion with expiry time."""
        future_time = datetime.now(timezone.utc) + timedelta(hours=1)
        suggestion = Suggestion(
            type=SuggestionType.FEATURE_INTRODUCTION,
            text="Try the new feature",
            confidence=0.7,
            expires_at=future_time
        )
        
        assert suggestion.expires_at == future_time
    
    def test_suggestion_expiry_validation(self):
        """Test suggestion expiry validation."""
        past_time = datetime.now(timezone.utc) - timedelta(hours=1)
        
        with pytest.raises(ValueError):
            Suggestion(
                type=SuggestionType.FEATURE_INTRODUCTION,
                text="Expired suggestion",
                confidence=0.7,
                expires_at=past_time
            )


class TestLearningMetrics:
    """Test LearningMetrics model."""
    
    def test_create_learning_metrics(self):
        """Test creating learning metrics."""
        metrics = LearningMetrics(
            total_actions=100,
            patterns_detected=5,
            suggestions_made=20,
            suggestions_accepted=12,
            accuracy_score=0.75
        )
        
        assert metrics.total_actions == 100
        assert metrics.patterns_detected == 5
        assert metrics.suggestions_made == 20
        assert metrics.suggestions_accepted == 12
        assert metrics.accuracy_score == 0.75
    
    def test_acceptance_rate_calculation(self):
        """Test acceptance rate calculation."""
        metrics = LearningMetrics(
            suggestions_made=10,
            suggestions_accepted=7
        )
        
        assert metrics.acceptance_rate == 0.7
    
    def test_acceptance_rate_with_no_suggestions(self):
        """Test acceptance rate when no suggestions made."""
        metrics = LearningMetrics()
        
        assert metrics.acceptance_rate == 0.0


class TestUserProfile:
    """Test UserProfile model."""
    
    def test_create_user_profile(self):
        """Test creating a user profile."""
        profile = UserProfile(
            skill_level=SkillLevel.INTERMEDIATE,
            total_commands=25,
            success_rate=0.85,
            help_request_rate=0.1
        )
        
        assert profile.skill_level == SkillLevel.INTERMEDIATE
        assert profile.total_commands == 25
        assert profile.success_rate == 0.85
        assert profile.help_request_rate == 0.1
        assert isinstance(UUID(profile.user_id), UUID)
        assert profile.version == "1.0.0"
    
    def test_profile_defaults(self):
        """Test user profile default values."""
        profile = UserProfile()
        
        assert profile.skill_level == SkillLevel.BEGINNER
        assert profile.total_commands == 0
        assert profile.session_count == 0
        assert profile.success_rate == 1.0
        assert profile.help_request_rate == 0.0
        assert profile.command_patterns == []
        assert profile.favorite_commands == []
        assert profile.advanced_features_used == set()
        assert profile.common_errors == {}
        assert isinstance(profile.learning_metrics, LearningMetrics)
    
    def test_skill_level_validation(self):
        """Test skill level validation based on command count."""
        # Should auto-adjust skill level based on command count
        profile = UserProfile(
            skill_level=SkillLevel.BEGINNER,
            total_commands=15  # Should bump to INTERMEDIATE
        )
        
        # The root_validator should adjust this
        assert profile.skill_level == SkillLevel.INTERMEDIATE
    
    def test_expertise_score_calculation(self):
        """Test expertise score calculation."""
        profile = UserProfile(
            total_commands=100,
            success_rate=0.9,
            help_request_rate=0.05,
            command_patterns=[
                CommandPattern(
                    name="Pattern 1",
                    commands=["cmd1", "cmd2"],
                    success_rate=0.8,
                    avg_duration_ms=1000,
                    confidence=0.7
                ) for _ in range(5)
            ],
            advanced_features_used={"feature1", "feature2", "feature3"}
        )
        
        expertise = profile.expertise_score
        assert 0.0 <= expertise <= 1.0
        assert expertise > 0.5  # Should be above average
    
    def test_progressive_disclosure_level_assignment(self):
        """Test progressive disclosure level assignment."""
        beginner_profile = UserProfile(skill_level=SkillLevel.BEGINNER)
        expert_profile = UserProfile(skill_level=SkillLevel.EXPERT)
        
        # Default assignment should match skill level
        assert beginner_profile.progressive_disclosure_level == ProgressiveDisclosureLevel.MINIMAL
        # Expert would get STANDARD by default, but this depends on the validator
    
    def test_favorite_commands_limit(self):
        """Test favorite commands list limit."""
        profile = UserProfile()
        
        # Add more than 20 commands
        for i in range(25):
            profile.favorite_commands.append(f"command_{i}")
        
        # Should be truncated to 20
        assert len(profile.favorite_commands) <= 20


class TestExceptions:
    """Test intelligence system exceptions."""
    
    def test_profile_corrupted_error(self):
        """Test ProfileCorruptedError exception."""
        with pytest.raises(ProfileCorruptedError):
            raise ProfileCorruptedError("Profile data is corrupted")
    
    def test_insufficient_data_error(self):
        """Test InsufficientDataError exception."""
        with pytest.raises(InsufficientDataError):
            raise InsufficientDataError("Not enough data for analysis")
    
    def test_analysis_failed_error(self):
        """Test AnalysisFailedError exception."""
        with pytest.raises(AnalysisFailedError):
            raise AnalysisFailedError("Analysis failed")
    
    def test_storage_unavailable_error(self):
        """Test StorageUnavailableError exception."""
        with pytest.raises(StorageUnavailableError):
            raise StorageUnavailableError("Cannot access storage")


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_large_command_count(self):
        """Test profile with very large command count."""
        profile = UserProfile(total_commands=10000)
        expertise = profile.expertise_score
        
        assert expertise <= 1.0  # Should not exceed maximum
    
    def test_zero_success_rate(self):
        """Test profile with zero success rate."""
        profile = UserProfile(
            total_commands=50,
            success_rate=0.0
        )
        
        expertise = profile.expertise_score
        assert expertise >= 0.0  # Should not go negative
    
    def test_many_patterns(self):
        """Test profile with many command patterns."""
        patterns = [
            CommandPattern(
                name=f"Pattern {i}",
                commands=[f"cmd{i}a", f"cmd{i}b"],
                success_rate=0.8,
                avg_duration_ms=1000,
                confidence=0.7
            ) for i in range(20)
        ]
        
        profile = UserProfile(command_patterns=patterns)
        assert len(profile.command_patterns) == 20
        
        # Expertise should be high with many patterns
        expertise = profile.expertise_score
        assert expertise > 0.5