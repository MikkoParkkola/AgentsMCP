"""Tests for UserProfiler class."""

import json
import pytest
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from cryptography.fernet import Fernet

from src.agentsmcp.cli.v3.intelligence.user_profiler import UserProfiler
from src.agentsmcp.cli.v3.models.intelligence_models import (
    UserAction,
    UserProfile,
    SessionContext,
    SkillLevel,
    CommandCategory,
    ProgressiveDisclosureLevel,
    InsufficientDataError,
    StorageUnavailableError,
)


class TestUserProfiler:
    """Test UserProfiler functionality."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir) / "test_profile.enc"
    
    @pytest.fixture
    def profiler(self, temp_storage):
        """Create UserProfiler instance."""
        return UserProfiler(
            storage_path=temp_storage,
            max_actions_history=100,
            pattern_detection_threshold=2
        )
    
    def test_initialization(self, profiler):
        """Test profiler initialization."""
        assert profiler is not None
        assert profiler.max_actions_history == 100
        assert profiler.pattern_threshold == 2
        assert profiler.current_profile is not None
        assert isinstance(profiler.current_profile, UserProfile)
    
    def test_start_session(self, profiler):
        """Test starting a new session."""
        session = profiler.start_session()
        
        assert isinstance(session, SessionContext)
        assert profiler.current_session == session
        assert session.start_time <= datetime.now(timezone.utc)
        assert profiler.session_stats['command_count'] == 0
        assert profiler.session_stats['error_count'] == 0
    
    def test_record_action_success(self, profiler):
        """Test recording a successful action."""
        profiler.start_session()
        
        action = UserAction(
            command="test_command",
            success=True,
            duration_ms=1000,
            context={"test": "context"}
        )
        
        profiler.record_action(action)
        
        assert len(profiler.action_history) == 1
        assert profiler.session_stats['command_count'] == 1
        assert profiler.session_stats['error_count'] == 0
        assert "test_command" in profiler.session_stats['unique_commands']
        assert profiler.current_session.commands_used == ["test_command"]
    
    def test_record_action_failure(self, profiler):
        """Test recording a failed action."""
        profiler.start_session()
        
        action = UserAction(
            command="failing_command",
            success=False,
            duration_ms=500,
            errors=["Test error"]
        )
        
        profiler.record_action(action)
        
        assert len(profiler.action_history) == 1
        assert profiler.session_stats['command_count'] == 1
        assert profiler.session_stats['error_count'] == 1
        assert profiler.current_session.errors_encountered == ["Test error"]
    
    def test_record_help_action(self, profiler):
        """Test recording help requests."""
        profiler.start_session()
        
        action = UserAction(
            command="help something",
            success=True,
            duration_ms=200
        )
        
        profiler.record_action(action)
        
        assert profiler.session_stats['help_requests'] == 1
        assert profiler.current_session.help_requests == 1
    
    def test_end_session(self, profiler):
        """Test ending a session."""
        session = profiler.start_session()
        
        # Record some actions
        for i in range(3):
            action = UserAction(
                command=f"command_{i}",
                success=i != 1,  # Make second command fail
                duration_ms=1000
            )
            profiler.record_action(action)
        
        completed_session = profiler.end_session()
        
        assert completed_session == session
        assert completed_session.duration_ms > 0
        assert completed_session.unique_commands == 3
        assert completed_session.success_rate == 2/3  # 2 out of 3 succeeded
        assert profiler.current_session is None
        assert profiler.current_profile.session_count == 1
    
    def test_skill_level_detection_beginner(self, profiler):
        """Test skill level detection for beginner."""
        # Simulate beginner behavior
        profiler.current_profile.total_commands = 5
        profiler.current_profile.success_rate = 0.6
        profiler.current_profile.help_request_rate = 0.3
        
        skill_level = profiler.get_skill_level()
        assert skill_level == SkillLevel.BEGINNER
    
    def test_skill_level_detection_intermediate(self, profiler):
        """Test skill level detection for intermediate."""
        # Simulate intermediate behavior
        profiler.current_profile.total_commands = 25
        profiler.current_profile.success_rate = 0.8
        profiler.current_profile.help_request_rate = 0.1
        profiler.current_profile.favorite_commands = ["cmd1", "cmd2", "cmd3", "cmd4"]
        
        skill_level = profiler.get_skill_level()
        assert skill_level == SkillLevel.INTERMEDIATE
    
    def test_skill_level_detection_expert(self, profiler):
        """Test skill level detection for expert."""
        # Simulate expert behavior
        profiler.current_profile.total_commands = 100
        profiler.current_profile.success_rate = 0.9
        profiler.current_profile.help_request_rate = 0.05
        profiler.current_profile.advanced_features_used = {"pipeline", "orchestrate", "debug"}
        profiler.current_profile.command_patterns = [
            MagicMock() for _ in range(5)
        ]
        
        skill_level = profiler.get_skill_level()
        assert skill_level in [SkillLevel.EXPERT, SkillLevel.POWER]
    
    def test_progressive_disclosure_level(self, profiler):
        """Test progressive disclosure level calculation."""
        # Test beginner level
        profiler.current_profile.skill_level = SkillLevel.BEGINNER
        profiler.current_profile.success_rate = 0.9
        
        level = profiler.get_progressive_disclosure_level()
        assert level == ProgressiveDisclosureLevel.MINIMAL
        
        # Test expert with advanced features
        profiler.current_profile.skill_level = SkillLevel.EXPERT
        profiler.current_profile.advanced_features_used = {f"feature_{i}" for i in range(12)}
        
        level = profiler.get_progressive_disclosure_level()
        assert level >= ProgressiveDisclosureLevel.STANDARD
    
    def test_command_frequency_tracking(self, profiler):
        """Test command frequency tracking."""
        # Record multiple actions with some repeats
        commands = ["status", "help", "status", "configure", "status", "help"]
        
        for cmd in commands:
            action = UserAction(
                command=cmd,
                success=True,
                duration_ms=1000
            )
            profiler.record_action(action)
        
        frequencies = profiler.get_command_frequency()
        
        assert frequencies["status"] == 3
        assert frequencies["help"] == 2
        assert frequencies["configure"] == 1
    
    def test_pattern_detection(self, profiler):
        """Test command pattern detection."""
        # Create a repeating sequence
        sequence = ["init", "configure", "start"]
        
        # Repeat the sequence multiple times to trigger pattern detection
        for _ in range(3):
            for cmd in sequence:
                action = UserAction(
                    command=cmd,
                    success=True,
                    duration_ms=1000
                )
                profiler.record_action(action)
        
        # Trigger pattern detection
        session = profiler.start_session()
        profiler.end_session()
        
        # Should detect the repeating pattern
        patterns = profiler.current_profile.command_patterns
        pattern_commands = [p.commands for p in patterns]
        
        # Should find subsequences of the repeated pattern
        assert len(patterns) > 0
        # Check if any detected pattern contains our sequence elements
        found_pattern = any(
            any(cmd in pattern for cmd in sequence)
            for pattern in pattern_commands
        )
        assert found_pattern
    
    def test_preferences_update(self, profiler):
        """Test preference updates."""
        new_prefs = {
            "theme": "dark",
            "verbose": True,
            "timeout": 30000
        }
        
        profiler.update_preferences(new_prefs)
        
        for key, value in new_prefs.items():
            assert profiler.current_profile.preferences[key] == value
    
    def test_profile_persistence(self, temp_storage):
        """Test profile save and load."""
        profiler1 = UserProfiler(storage_path=temp_storage)
        
        # Modify profile
        profiler1.current_profile.total_commands = 42
        profiler1.current_profile.skill_level = SkillLevel.INTERMEDIATE
        original_user_id = profiler1.current_profile.user_id
        
        # Save profile
        profiler1.save_profile()
        
        # Create new profiler with same storage
        profiler2 = UserProfiler(storage_path=temp_storage)
        
        # Should load the saved profile
        assert profiler2.current_profile.total_commands == 42
        assert profiler2.current_profile.skill_level == SkillLevel.INTERMEDIATE
        assert profiler2.current_profile.user_id == original_user_id
    
    def test_get_profile_without_data(self):
        """Test getting profile when no data available."""
        profiler = UserProfiler()
        # Clear the profile
        profiler.current_profile = None
        
        with pytest.raises(InsufficientDataError):
            profiler.get_profile()
    
    def test_advanced_command_detection(self, profiler):
        """Test detection of advanced command usage."""
        advanced_action = UserAction(
            command="pipeline create complex-workflow",
            success=True,
            duration_ms=2000
        )
        
        profiler.record_action(advanced_action)
        
        assert "pipeline create complex-workflow" in profiler.current_profile.advanced_features_used
    
    def test_error_tracking(self, profiler):
        """Test error pattern tracking."""
        # Record actions with specific errors
        for i in range(3):
            action = UserAction(
                command=f"command_{i}",
                success=False,
                duration_ms=500,
                errors=["Connection timeout"]
            )
            profiler.record_action(action)
        
        # Add a different error
        action = UserAction(
            command="other_command",
            success=False,
            duration_ms=300,
            errors=["Permission denied"]
        )
        profiler.record_action(action)
        
        # Check error tracking
        assert profiler.current_profile.common_errors["Connection timeout"] == 3
        assert profiler.current_profile.common_errors["Permission denied"] == 1
    
    def test_session_statistics_update(self, profiler):
        """Test session statistics updates."""
        session = profiler.start_session()
        
        # Record various actions
        actions = [
            UserAction(command="cmd1", success=True, duration_ms=1000),
            UserAction(command="cmd2", success=False, duration_ms=500, errors=["Error"]),
            UserAction(command="help", success=True, duration_ms=200),
        ]
        
        for action in actions:
            profiler.record_action(action)
        
        completed_session = profiler.end_session()
        
        assert completed_session.unique_commands == 3
        assert len(completed_session.commands_used) == 3
        assert len(completed_session.errors_encountered) == 1
        assert completed_session.help_requests == 1
        assert completed_session.success_rate == 2/3
    
    def test_storage_error_handling(self, temp_storage):
        """Test storage error handling."""
        profiler = UserProfiler(storage_path=temp_storage)
        
        # Make storage path read-only to trigger error
        temp_storage.parent.chmod(0o444)
        
        try:
            with pytest.raises(StorageUnavailableError):
                profiler.save_profile()
        finally:
            # Restore permissions for cleanup
            temp_storage.parent.chmod(0o755)
    
    def test_encryption_functionality(self, temp_storage):
        """Test profile encryption/decryption."""
        profiler = UserProfiler(storage_path=temp_storage)
        
        # Save profile
        profiler.save_profile()
        
        # Read raw file - should be encrypted
        raw_data = temp_storage.read_bytes()
        
        # Should not contain plain text profile data
        assert b"user_id" not in raw_data
        assert b"skill_level" not in raw_data
        
        # But should be decryptable by profiler
        new_profiler = UserProfiler(storage_path=temp_storage)
        assert new_profiler.current_profile is not None


class TestUserProfilerEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_action_history(self):
        """Test behavior with empty action history."""
        profiler = UserProfiler()
        
        # Should handle empty history gracefully
        frequencies = profiler.get_command_frequency()
        assert frequencies == {}
        
        skill_level = profiler.get_skill_level()
        assert skill_level == SkillLevel.BEGINNER
    
    def test_pattern_detection_insufficient_data(self):
        """Test pattern detection with insufficient data."""
        profiler = UserProfiler(pattern_detection_threshold=5)
        
        # Record only a few actions
        for i in range(2):
            action = UserAction(
                command=f"cmd_{i}",
                success=True,
                duration_ms=1000
            )
            profiler.record_action(action)
        
        session = profiler.start_session()
        profiler.end_session()
        
        # Should not detect patterns with insufficient data
        assert len(profiler.current_profile.command_patterns) == 0
    
    def test_large_action_history(self):
        """Test handling of large action history."""
        profiler = UserProfiler(max_actions_history=10)
        
        # Record more actions than the limit
        for i in range(15):
            action = UserAction(
                command=f"cmd_{i}",
                success=True,
                duration_ms=1000
            )
            profiler.record_action(action)
        
        # Should keep only the most recent actions
        assert len(profiler.action_history) <= 10
    
    def test_session_without_actions(self):
        """Test ending session without recording any actions."""
        profiler = UserProfiler()
        
        session = profiler.start_session()
        completed_session = profiler.end_session()
        
        assert completed_session.unique_commands == 0
        assert completed_session.success_rate == 1.0  # No failures
        assert len(completed_session.commands_used) == 0
    
    def test_corrupted_storage_recovery(self, temp_storage):
        """Test recovery from corrupted storage."""
        # Write invalid data to storage file
        temp_storage.parent.mkdir(parents=True, exist_ok=True)
        temp_storage.write_bytes(b"corrupted data")
        
        # Should create new profile when loading fails
        profiler = UserProfiler(storage_path=temp_storage)
        assert profiler.current_profile is not None
        assert profiler.current_profile.total_commands == 0