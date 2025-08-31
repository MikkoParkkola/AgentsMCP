"""User Profiler for AgentsMCP CLI v3 Intelligence System.

This module handles user skill level detection, preference learning, command
pattern recognition, and session context management for personalized experiences.
"""

import json
import logging
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from cryptography.fernet import Fernet

from ..models.intelligence_models import (
    UserProfile,
    SkillLevel,
    UserAction,
    SessionContext,
    CommandPattern,
    CommandCategory,
    ProgressiveDisclosureLevel,
    LearningMetrics,
    ProfileCorruptedError,
    InsufficientDataError,
    StorageUnavailableError,
)


logger = logging.getLogger(__name__)


class UserProfiler:
    """Manages user profile creation, updates, and skill level detection."""
    
    def __init__(
        self,
        storage_path: Optional[Path] = None,
        encryption_key: Optional[bytes] = None,
        max_actions_history: int = 1000,
        pattern_detection_threshold: int = 3,
    ):
        """Initialize the user profiler.
        
        Args:
            storage_path: Path for encrypted profile storage
            encryption_key: Key for profile encryption (generates if None)
            max_actions_history: Maximum actions to keep in memory
            pattern_detection_threshold: Minimum occurrences to detect pattern
        """
        self.storage_path = storage_path or Path.home() / ".agentsmcp" / "profile.enc"
        self.max_actions_history = max_actions_history
        self.pattern_threshold = pattern_detection_threshold
        
        # Initialize encryption
        if encryption_key:
            self.cipher = Fernet(encryption_key)
        else:
            key = Fernet.generate_key()
            self.cipher = Fernet(key)
            self._save_key(key)
        
        # Runtime state
        self.current_profile: Optional[UserProfile] = None
        self.current_session: Optional[SessionContext] = None
        self.action_history: deque = deque(maxlen=max_actions_history)
        self.command_sequences: List[List[str]] = []
        self.session_stats = {
            'command_count': 0,
            'error_count': 0,
            'help_requests': 0,
            'start_time': None,
            'unique_commands': set(),
        }
        
        # Skill level thresholds
        self.skill_thresholds = {
            SkillLevel.BEGINNER: 0,
            SkillLevel.INTERMEDIATE: 11,
            SkillLevel.EXPERT: 51,
            SkillLevel.POWER: 201,
        }
        
        # Advanced command indicators
        self.advanced_commands = {
            'pipeline', 'orchestrate', 'delegate', 'chain', 'template',
            'customize', 'profile', 'benchmark', 'debug', 'trace',
            'optimize', 'scale', 'monitor', 'analyze', 'configure'
        }
        
        self._load_profile()
    
    def _save_key(self, key: bytes) -> None:
        """Save encryption key securely."""
        key_path = self.storage_path.parent / ".key"
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.write_bytes(key)
        key_path.chmod(0o600)  # Owner read/write only
    
    def start_session(self) -> SessionContext:
        """Start a new user session for tracking."""
        self.current_session = SessionContext(
            start_time=datetime.now(timezone.utc)
        )
        self.session_stats = {
            'command_count': 0,
            'error_count': 0,
            'help_requests': 0,
            'start_time': datetime.now(timezone.utc),
            'unique_commands': set(),
        }
        logger.info(f"Started new session: {self.current_session.session_id}")
        return self.current_session
    
    def record_action(self, action: UserAction) -> None:
        """Record a user action for learning and pattern detection.
        
        Args:
            action: The user action to record
        """
        self.action_history.append(action)
        
        # Update session stats
        self.session_stats['command_count'] += 1
        if not action.success:
            self.session_stats['error_count'] += 1
        if 'help' in action.command.lower():
            self.session_stats['help_requests'] += 1
        self.session_stats['unique_commands'].add(action.command)
        
        # Update current session if active
        if self.current_session:
            self.current_session.commands_used.append(action.command)
            if not action.success:
                self.current_session.errors_encountered.extend(action.errors)
            if 'help' in action.command.lower():
                self.current_session.help_requests += 1
        
        # Check for advanced command usage
        if any(cmd in action.command.lower() for cmd in self.advanced_commands):
            if self.current_profile:
                self.current_profile.advanced_features_used.add(action.command)
        
        # Update command sequences for pattern detection
        if len(self.command_sequences) == 0 or len(self.command_sequences[-1]) >= 10:
            self.command_sequences.append([])
        self.command_sequences[-1].append(action.command)
        
        # Update profile with new action
        self._update_profile_from_action(action)
        
        logger.debug(f"Recorded action: {action.command} (success: {action.success})")
    
    def end_session(self) -> SessionContext:
        """End the current session and update profile.
        
        Returns:
            The completed session context
        """
        if not self.current_session:
            raise ValueError("No active session to end")
        
        # Calculate final session metrics
        now = datetime.now(timezone.utc)
        duration = now - self.current_session.start_time
        self.current_session.duration_ms = int(duration.total_seconds() * 1000)
        self.current_session.unique_commands = len(self.session_stats['unique_commands'])
        
        if self.session_stats['command_count'] > 0:
            self.current_session.success_rate = 1.0 - (
                self.session_stats['error_count'] / self.session_stats['command_count']
            )
        
        # Update profile session statistics
        if self.current_profile:
            self.current_profile.session_count += 1
            total_duration = (
                self.current_profile.avg_session_duration_ms * (self.current_profile.session_count - 1) +
                self.current_session.duration_ms
            )
            self.current_profile.avg_session_duration_ms = int(
                total_duration / self.current_profile.session_count
            )
            self.current_profile.last_active = now
        
        # Detect new patterns from this session
        self._detect_patterns()
        
        # Save updated profile
        self.save_profile()
        
        session = self.current_session
        self.current_session = None
        
        logger.info(f"Ended session: {session.session_id} (duration: {session.duration_ms}ms)")
        return session
    
    def get_skill_level(self) -> SkillLevel:
        """Detect current user skill level based on usage patterns.
        
        Returns:
            Detected skill level
        """
        if not self.current_profile:
            return SkillLevel.BEGINNER
        
        profile = self.current_profile
        
        # Base skill on command count
        base_skill = SkillLevel.BEGINNER
        for level, threshold in self.skill_thresholds.items():
            if profile.total_commands >= threshold:
                base_skill = level
        
        # Adjust based on other factors
        skill_score = 0
        
        # Command diversity bonus
        if profile.total_commands > 0:
            unique_ratio = len(profile.favorite_commands) / min(profile.total_commands, 20)
            skill_score += unique_ratio * 10
        
        # Success rate bonus
        skill_score += profile.success_rate * 20
        
        # Advanced features bonus
        skill_score += len(profile.advanced_features_used) * 2
        
        # Help request penalty
        skill_score -= profile.help_request_rate * 15
        
        # Pattern recognition bonus
        skill_score += len(profile.command_patterns) * 3
        
        # Adjust skill level based on score
        if skill_score >= 40:
            if base_skill == SkillLevel.EXPERT:
                return SkillLevel.POWER
            elif base_skill == SkillLevel.INTERMEDIATE:
                return SkillLevel.EXPERT
            elif base_skill == SkillLevel.BEGINNER:
                return SkillLevel.INTERMEDIATE
        elif skill_score >= 25:
            if base_skill == SkillLevel.INTERMEDIATE:
                return SkillLevel.EXPERT
            elif base_skill == SkillLevel.BEGINNER:
                return SkillLevel.INTERMEDIATE
        elif skill_score >= 10:
            if base_skill == SkillLevel.BEGINNER:
                return SkillLevel.INTERMEDIATE
        
        return base_skill
    
    def update_preferences(self, preferences: Dict[str, any]) -> None:
        """Update user preferences from observed behavior.
        
        Args:
            preferences: Dictionary of preference updates
        """
        if not self.current_profile:
            self._create_profile()
        
        self.current_profile.preferences.update(preferences)
        logger.debug(f"Updated preferences: {preferences}")
    
    def get_progressive_disclosure_level(self) -> ProgressiveDisclosureLevel:
        """Determine appropriate UI complexity level.
        
        Returns:
            Progressive disclosure level (1-5)
        """
        if not self.current_profile:
            return ProgressiveDisclosureLevel.MINIMAL
        
        skill_level = self.get_skill_level()
        
        # Map skill levels to disclosure levels
        level_mapping = {
            SkillLevel.BEGINNER: ProgressiveDisclosureLevel.MINIMAL,
            SkillLevel.INTERMEDIATE: ProgressiveDisclosureLevel.BASIC,
            SkillLevel.EXPERT: ProgressiveDisclosureLevel.STANDARD,
            SkillLevel.POWER: ProgressiveDisclosureLevel.ADVANCED,
        }
        
        base_level = level_mapping[skill_level]
        
        # Adjust based on advanced feature usage
        if len(self.current_profile.advanced_features_used) >= 10:
            base_level = min(base_level + 1, ProgressiveDisclosureLevel.DEVELOPER)
        
        # Adjust based on error rate (lower complexity for high error rate)
        if self.current_profile.success_rate < 0.7:
            base_level = max(base_level - 1, ProgressiveDisclosureLevel.MINIMAL)
        
        return ProgressiveDisclosureLevel(base_level)
    
    def get_command_frequency(self) -> Dict[str, int]:
        """Get command usage frequency statistics.
        
        Returns:
            Dictionary mapping commands to usage counts
        """
        if not self.current_profile:
            return {}
        
        # Count commands from action history
        command_counts = Counter()
        for action in self.action_history:
            command_counts[action.command] += 1
        
        return dict(command_counts.most_common())
    
    def get_profile(self) -> UserProfile:
        """Get current user profile.
        
        Returns:
            Current user profile
            
        Raises:
            InsufficientDataError: If no profile data available
        """
        if not self.current_profile:
            raise InsufficientDataError("No user profile available")
        
        # Update skill level before returning
        self.current_profile.skill_level = self.get_skill_level()
        self.current_profile.progressive_disclosure_level = self.get_progressive_disclosure_level()
        
        return self.current_profile
    
    def save_profile(self) -> None:
        """Save user profile to encrypted storage."""
        if not self.current_profile:
            return
        
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert profile to JSON
            profile_data = self.current_profile.dict()
            profile_json = json.dumps(profile_data, default=str)
            
            # Encrypt and save
            encrypted_data = self.cipher.encrypt(profile_json.encode())
            self.storage_path.write_bytes(encrypted_data)
            
            logger.debug(f"Saved profile to {self.storage_path}")
            
        except Exception as e:
            logger.error(f"Failed to save profile: {e}")
            raise StorageUnavailableError(f"Cannot save profile: {e}")
    
    def _load_profile(self) -> None:
        """Load user profile from encrypted storage."""
        if not self.storage_path.exists():
            self._create_profile()
            return
        
        try:
            # Load and decrypt
            encrypted_data = self.storage_path.read_bytes()
            profile_json = self.cipher.decrypt(encrypted_data).decode()
            profile_data = json.loads(profile_json)
            
            # Convert back to UserProfile
            self.current_profile = UserProfile(**profile_data)
            logger.info(f"Loaded profile for user {self.current_profile.user_id}")
            
        except Exception as e:
            logger.error(f"Failed to load profile: {e}")
            # Create new profile if loading fails
            self._create_profile()
    
    def _create_profile(self) -> None:
        """Create a new user profile."""
        self.current_profile = UserProfile()
        logger.info(f"Created new profile: {self.current_profile.user_id}")
    
    def _update_profile_from_action(self, action: UserAction) -> None:
        """Update profile statistics from user action.
        
        Args:
            action: User action to process
        """
        if not self.current_profile:
            self._create_profile()
        
        profile = self.current_profile
        
        # Update basic statistics
        profile.total_commands += 1
        
        # Update success rate (running average)
        if profile.total_commands == 1:
            profile.success_rate = 1.0 if action.success else 0.0
        else:
            profile.success_rate = (
                (profile.success_rate * (profile.total_commands - 1) + (1.0 if action.success else 0.0))
                / profile.total_commands
            )
        
        # Update help request rate
        is_help = 'help' in action.command.lower()
        if profile.total_commands == 1:
            profile.help_request_rate = 1.0 if is_help else 0.0
        else:
            profile.help_request_rate = (
                (profile.help_request_rate * (profile.total_commands - 1) + (1.0 if is_help else 0.0))
                / profile.total_commands
            )
        
        # Update favorite commands
        if action.command not in profile.favorite_commands:
            profile.favorite_commands.append(action.command)
            if len(profile.favorite_commands) > 20:
                profile.favorite_commands.pop(0)
        
        # Track common errors
        if not action.success:
            for error in action.errors:
                profile.common_errors[error] = profile.common_errors.get(error, 0) + 1
        
        # Update learning metrics
        profile.learning_metrics.total_actions += 1
        profile.learning_metrics.last_updated = datetime.now(timezone.utc)
    
    def _detect_patterns(self) -> List[CommandPattern]:
        """Detect command usage patterns from recent actions.
        
        Returns:
            List of newly detected patterns
        """
        if not self.current_profile or len(self.action_history) < self.pattern_threshold:
            return []
        
        new_patterns = []
        
        # Analyze command sequences
        for sequence in self.command_sequences:
            if len(sequence) < 2:
                continue
            
            # Find repeated subsequences
            for length in range(2, min(6, len(sequence) + 1)):
                for start in range(len(sequence) - length + 1):
                    subseq = sequence[start:start + length]
                    
                    # Count occurrences of this subsequence
                    occurrences = 0
                    for seq in self.command_sequences:
                        for i in range(len(seq) - length + 1):
                            if seq[i:i + length] == subseq:
                                occurrences += 1
                    
                    if occurrences >= self.pattern_threshold:
                        # Check if pattern already exists
                        pattern_exists = False
                        for existing in self.current_profile.command_patterns:
                            if existing.commands == subseq:
                                existing.frequency = occurrences
                                existing.last_seen = datetime.now(timezone.utc)
                                pattern_exists = True
                                break
                        
                        if not pattern_exists:
                            # Calculate success rate for this pattern
                            successes = sum(
                                1 for action in self.action_history
                                if action.command in subseq and action.success
                            )
                            total_pattern_actions = sum(
                                1 for action in self.action_history
                                if action.command in subseq
                            )
                            success_rate = successes / max(total_pattern_actions, 1)
                            
                            # Create new pattern
                            pattern = CommandPattern(
                                name=f"Workflow: {' → '.join(subseq[:3])}{'...' if len(subseq) > 3 else ''}",
                                commands=subseq,
                                frequency=occurrences,
                                success_rate=success_rate,
                                avg_duration_ms=2000,  # Default estimate
                                confidence=min(occurrences / 10.0, 1.0),
                            )
                            
                            self.current_profile.command_patterns.append(pattern)
                            new_patterns.append(pattern)
                            
                            logger.info(f"Detected new pattern: {pattern.name}")
        
        # Update pattern detection metrics
        if new_patterns:
            self.current_profile.learning_metrics.patterns_detected += len(new_patterns)
        
        return new_patterns