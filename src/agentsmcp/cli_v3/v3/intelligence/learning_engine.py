"""Learning Engine for AgentsMCP CLI v3 Intelligence System.

This module implements adaptive learning algorithms that improve over time,
including pattern recognition, error analysis, and preference inference.
"""

import json
import logging
import math
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..models.intelligence_models import (
    UserAction,
    UserProfile,
    SessionContext,
    CommandPattern,
    LearningEventType,
    AnalysisFailedError,
    InsufficientDataError,
)


logger = logging.getLogger(__name__)


class LearningEngine:
    """Adaptive learning engine for user behavior analysis and prediction."""
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        decay_factor: float = 0.95,
        min_pattern_support: int = 3,
        max_patterns: int = 50,
        similarity_threshold: float = 0.7,
    ):
        """Initialize the learning engine.
        
        Args:
            learning_rate: How quickly the system adapts to new patterns
            decay_factor: How quickly old patterns fade (0.0-1.0)
            min_pattern_support: Minimum occurrences to consider a pattern
            max_patterns: Maximum number of patterns to maintain
            similarity_threshold: Threshold for pattern similarity detection
        """
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.min_pattern_support = min_pattern_support
        self.max_patterns = max_patterns
        self.similarity_threshold = similarity_threshold
        
        # Learning state
        self.command_embeddings: Dict[str, np.ndarray] = {}
        self.sequence_patterns: Dict[str, Dict] = {}
        self.error_patterns: Dict[str, Dict] = {}
        self.time_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self.feature_weights: Dict[str, float] = defaultdict(lambda: 1.0)
        
        # Temporal learning windows
        self.short_term_window = deque(maxlen=100)  # Recent actions for immediate adaptation
        self.medium_term_window = deque(maxlen=500)  # Medium-term patterns
        self.long_term_memory: List[Dict] = []  # Persistent patterns
        
        # Performance tracking
        self.prediction_accuracy: Dict[str, float] = defaultdict(lambda: 0.5)
        self.adaptation_history: List[Dict] = []
        
        # Text vectorizer for command similarity
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=None,
            ngram_range=(1, 2)
        )
        self._is_vectorizer_fitted = False
    
    def learn_from_action(self, action: UserAction, context: Optional[SessionContext] = None) -> None:
        """Learn from a single user action.
        
        Args:
            action: User action to learn from
            context: Optional session context for additional learning signals
        """
        # Add to learning windows
        action_data = {
            'command': action.command,
            'category': action.category,
            'success': action.success,
            'duration_ms': action.duration_ms,
            'timestamp': action.timestamp,
            'errors': action.errors,
            'context': action.context,
        }
        
        self.short_term_window.append(action_data)
        self.medium_term_window.append(action_data)
        
        # Update command embeddings
        self._update_command_embedding(action.command, action_data)
        
        # Learn sequence patterns
        self._learn_sequence_patterns()
        
        # Learn error patterns
        if not action.success:
            self._learn_error_patterns(action)
        
        # Learn temporal patterns
        self._learn_temporal_patterns(action)
        
        # Update feature weights based on success
        self._update_feature_weights(action)
        
        logger.debug(f"Learned from action: {action.command} (success: {action.success})")
    
    def predict_next_command(
        self, 
        recent_commands: List[str], 
        context: Optional[Dict] = None
    ) -> List[Tuple[str, float]]:
        """Predict likely next commands based on sequence patterns.
        
        Args:
            recent_commands: List of recently executed commands
            context: Optional context for prediction refinement
            
        Returns:
            List of (command, probability) tuples sorted by probability
        """
        if len(recent_commands) == 0:
            return self._get_popular_commands()
        
        predictions = defaultdict(float)
        
        # Sequence-based predictions
        sequence_predictions = self._predict_from_sequences(recent_commands)
        for cmd, prob in sequence_predictions:
            predictions[cmd] += prob * 0.4
        
        # Similarity-based predictions
        similarity_predictions = self._predict_from_similarity(recent_commands[-1])
        for cmd, prob in similarity_predictions:
            predictions[cmd] += prob * 0.3
        
        # Temporal pattern predictions
        temporal_predictions = self._predict_from_temporal_patterns(context)
        for cmd, prob in temporal_predictions:
            predictions[cmd] += prob * 0.2
        
        # Context-based predictions
        if context:
            context_predictions = self._predict_from_context(context)
            for cmd, prob in context_predictions:
                predictions[cmd] += prob * 0.1
        
        # Sort by probability and return top predictions
        sorted_predictions = sorted(
            predictions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_predictions[:10]
    
    def analyze_error_patterns(self, user_profile: UserProfile) -> Dict[str, Dict]:
        """Analyze user error patterns for targeted assistance.
        
        Args:
            user_profile: User profile to analyze
            
        Returns:
            Dictionary with error analysis results
        """
        if len(self.medium_term_window) < 10:
            raise InsufficientDataError("Not enough data for error analysis")
        
        error_analysis = {
            'common_errors': {},
            'error_sequences': [],
            'error_contexts': {},
            'recovery_patterns': [],
            'recommendations': []
        }
        
        # Analyze common error types
        error_counts = defaultdict(int)
        for action in self.medium_term_window:
            if not action['success']:
                for error in action['errors']:
                    error_counts[error] += 1
        
        error_analysis['common_errors'] = dict(
            sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        )
        
        # Analyze error sequences
        error_sequences = self._find_error_sequences()
        error_analysis['error_sequences'] = error_sequences
        
        # Analyze error contexts
        error_contexts = self._analyze_error_contexts()
        error_analysis['error_contexts'] = error_contexts
        
        # Find recovery patterns
        recovery_patterns = self._find_recovery_patterns()
        error_analysis['recovery_patterns'] = recovery_patterns
        
        # Generate recommendations
        recommendations = self._generate_error_recommendations(error_analysis)
        error_analysis['recommendations'] = recommendations
        
        return error_analysis
    
    def infer_preferences(self, user_profile: UserProfile) -> Dict[str, any]:
        """Infer user preferences from behavior patterns.
        
        Args:
            user_profile: User profile to analyze
            
        Returns:
            Dictionary of inferred preferences
        """
        preferences = {}
        
        if len(self.medium_term_window) < 5:
            return preferences
        
        # Infer verbosity preference from help requests
        help_actions = [a for a in self.medium_term_window if 'help' in a['command'].lower()]
        preferences['verbose_output'] = len(help_actions) > len(self.medium_term_window) * 0.2
        
        # Infer auto-confirmation preference from success rate
        success_rate = sum(1 for a in self.medium_term_window if a['success']) / len(self.medium_term_window)
        preferences['auto_confirm'] = success_rate > 0.9
        
        # Infer suggestion level from command diversity
        unique_commands = len(set(a['command'] for a in self.medium_term_window))
        total_commands = len(self.medium_term_window)
        diversity_ratio = unique_commands / max(total_commands, 1)
        
        if diversity_ratio > 0.7:
            preferences['suggestion_level'] = 2  # Low - user explores independently
        elif diversity_ratio > 0.4:
            preferences['suggestion_level'] = 3  # Medium - balanced suggestions
        else:
            preferences['suggestion_level'] = 4  # High - user benefits from guidance
        
        # Infer timeout preference from command durations
        avg_duration = sum(a['duration_ms'] for a in self.medium_term_window) / len(self.medium_term_window)
        if avg_duration > 10000:  # Long-running commands
            preferences['default_timeout_ms'] = 60000
        elif avg_duration > 5000:
            preferences['default_timeout_ms'] = 30000
        else:
            preferences['default_timeout_ms'] = 15000
        
        # Infer theme preference from session timing
        evening_commands = sum(
            1 for a in self.medium_term_window
            if a['timestamp'].hour >= 18 or a['timestamp'].hour <= 6
        )
        if evening_commands > len(self.medium_term_window) * 0.6:
            preferences['theme'] = 'dark'
        else:
            preferences['theme'] = 'light'
        
        return preferences
    
    def update_accuracy(self, prediction: str, actual: str) -> None:
        """Update prediction accuracy based on actual outcomes.
        
        Args:
            prediction: What was predicted
            actual: What actually happened
        """
        prediction_type = 'next_command'
        
        was_correct = prediction == actual
        current_accuracy = self.prediction_accuracy[prediction_type]
        
        # Update accuracy using exponential moving average
        self.prediction_accuracy[prediction_type] = (
            current_accuracy * (1 - self.learning_rate) +
            (1.0 if was_correct else 0.0) * self.learning_rate
        )
        
        # Track adaptation
        self.adaptation_history.append({
            'timestamp': datetime.now(timezone.utc),
            'prediction_type': prediction_type,
            'predicted': prediction,
            'actual': actual,
            'correct': was_correct,
            'accuracy': self.prediction_accuracy[prediction_type]
        })
        
        # Adjust learning rate based on recent accuracy
        if len(self.adaptation_history) >= 20:
            recent_accuracy = sum(
                1 for entry in self.adaptation_history[-20:]
                if entry['correct']
            ) / 20
            
            if recent_accuracy > 0.8:
                self.learning_rate *= 0.9  # Decrease learning rate when accurate
            elif recent_accuracy < 0.4:
                self.learning_rate *= 1.1  # Increase learning rate when inaccurate
            
            self.learning_rate = max(0.01, min(0.3, self.learning_rate))
    
    def get_learning_metrics(self) -> Dict[str, Union[int, float]]:
        """Get current learning performance metrics.
        
        Returns:
            Dictionary of learning metrics
        """
        return {
            'total_actions_learned': len(self.medium_term_window),
            'unique_commands': len(self.command_embeddings),
            'sequence_patterns': len(self.sequence_patterns),
            'error_patterns': len(self.error_patterns),
            'prediction_accuracy': dict(self.prediction_accuracy),
            'learning_rate': self.learning_rate,
            'feature_weights': dict(self.feature_weights),
            'temporal_patterns': len(self.time_patterns),
        }
    
    def _update_command_embedding(self, command: str, action_data: Dict) -> None:
        """Update command embedding with new action data."""
        # Create feature vector from action
        features = np.array([
            action_data['duration_ms'] / 10000.0,  # Normalized duration
            1.0 if action_data['success'] else 0.0,  # Success indicator
            len(action_data['errors']),  # Error count
            len(action_data['context']),  # Context richness
        ])
        
        if command in self.command_embeddings:
            # Update existing embedding with exponential moving average
            self.command_embeddings[command] = (
                self.command_embeddings[command] * (1 - self.learning_rate) +
                features * self.learning_rate
            )
        else:
            # Create new embedding
            self.command_embeddings[command] = features
    
    def _learn_sequence_patterns(self) -> None:
        """Learn command sequence patterns from recent actions."""
        if len(self.short_term_window) < 3:
            return
        
        # Extract recent command sequences
        commands = [action['command'] for action in list(self.short_term_window)[-10:]]
        
        # Find patterns of different lengths
        for length in range(2, min(5, len(commands))):
            for i in range(len(commands) - length + 1):
                sequence = tuple(commands[i:i + length])
                
                if sequence not in self.sequence_patterns:
                    self.sequence_patterns[sequence] = {
                        'count': 0,
                        'success_rate': 0.0,
                        'last_seen': datetime.now(timezone.utc),
                        'next_commands': defaultdict(int)
                    }
                
                pattern = self.sequence_patterns[sequence]
                pattern['count'] += 1
                pattern['last_seen'] = datetime.now(timezone.utc)
                
                # Track next command if available
                if i + length < len(commands):
                    next_cmd = commands[i + length]
                    pattern['next_commands'][next_cmd] += 1
        
        # Prune old patterns
        self._prune_patterns()
    
    def _learn_error_patterns(self, action: UserAction) -> None:
        """Learn patterns from error occurrences."""
        for error in action.errors:
            if error not in self.error_patterns:
                self.error_patterns[error] = {
                    'count': 0,
                    'contexts': defaultdict(int),
                    'preceding_commands': defaultdict(int),
                    'time_of_day': defaultdict(int)
                }
            
            pattern = self.error_patterns[error]
            pattern['count'] += 1
            
            # Learn error contexts
            for key, value in action.context.items():
                pattern['contexts'][f"{key}:{value}"] += 1
            
            # Learn preceding commands
            if len(self.short_term_window) > 1:
                prev_command = list(self.short_term_window)[-2]['command']
                pattern['preceding_commands'][prev_command] += 1
            
            # Learn temporal patterns
            hour = action.timestamp.hour
            pattern['time_of_day'][hour] += 1
    
    def _learn_temporal_patterns(self, action: UserAction) -> None:
        """Learn time-based usage patterns."""
        command = action.command
        self.time_patterns[command].append(action.timestamp)
        
        # Keep only recent timestamps
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        self.time_patterns[command] = [
            ts for ts in self.time_patterns[command]
            if ts > cutoff
        ]
    
    def _update_feature_weights(self, action: UserAction) -> None:
        """Update feature importance weights based on action outcomes."""
        # Features that contributed to success/failure
        features = ['duration', 'context_size', 'category']
        
        success_weight = 1.0 if action.success else -0.5
        
        # Duration feature
        if action.duration_ms > 5000:  # Long command
            self.feature_weights['duration'] += success_weight * 0.1
        
        # Context richness feature
        if len(action.context) > 3:  # Rich context
            self.feature_weights['context_size'] += success_weight * 0.1
        
        # Category feature
        category_key = f"category_{action.category}"
        self.feature_weights[category_key] += success_weight * 0.05
        
        # Normalize weights to prevent drift
        for key in self.feature_weights:
            self.feature_weights[key] = max(0.1, min(2.0, self.feature_weights[key]))
    
    def _predict_from_sequences(self, recent_commands: List[str]) -> List[Tuple[str, float]]:
        """Predict next commands based on sequence patterns."""
        predictions = defaultdict(float)
        
        for length in range(min(4, len(recent_commands)), 0, -1):
            sequence = tuple(recent_commands[-length:])
            
            if sequence in self.sequence_patterns:
                pattern = self.sequence_patterns[sequence]
                total_next = sum(pattern['next_commands'].values())
                
                if total_next > 0:
                    weight = min(1.0, pattern['count'] / 10.0)  # Weight by frequency
                    
                    for cmd, count in pattern['next_commands'].items():
                        prob = (count / total_next) * weight
                        predictions[cmd] += prob
        
        return list(predictions.items())
    
    def _predict_from_similarity(self, last_command: str) -> List[Tuple[str, float]]:
        """Predict commands based on similarity to the last command."""
        if last_command not in self.command_embeddings:
            return []
        
        last_embedding = self.command_embeddings[last_command]
        similarities = []
        
        for cmd, embedding in self.command_embeddings.items():
            if cmd != last_command:
                similarity = cosine_similarity(
                    last_embedding.reshape(1, -1),
                    embedding.reshape(1, -1)
                )[0][0]
                similarities.append((cmd, float(similarity)))
        
        # Sort by similarity and return top matches
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:5]
    
    def _predict_from_temporal_patterns(self, context: Optional[Dict]) -> List[Tuple[str, float]]:
        """Predict commands based on temporal usage patterns."""
        if not context or 'current_time' not in context:
            return []
        
        current_time = context['current_time']
        if isinstance(current_time, str):
            current_time = datetime.fromisoformat(current_time)
        
        predictions = defaultdict(float)
        current_hour = current_time.hour
        
        for command, timestamps in self.time_patterns.items():
            if not timestamps:
                continue
            
            # Count occurrences in similar time windows
            similar_hour_count = sum(
                1 for ts in timestamps
                if abs(ts.hour - current_hour) <= 1
            )
            
            if similar_hour_count > 0:
                prob = min(1.0, similar_hour_count / len(timestamps))
                predictions[command] = prob
        
        return list(predictions.items())
    
    def _predict_from_context(self, context: Dict) -> List[Tuple[str, float]]:
        """Predict commands based on current context."""
        predictions = defaultdict(float)
        
        # Simple context-based predictions
        if context.get('error_occurred'):
            predictions['help'] = 0.8
            predictions['debug'] = 0.6
            predictions['status'] = 0.4
        
        if context.get('in_pipeline'):
            predictions['monitor'] = 0.7
            predictions['status'] = 0.5
        
        return list(predictions.items())
    
    def _get_popular_commands(self) -> List[Tuple[str, float]]:
        """Get popular commands as fallback predictions."""
        command_counts = defaultdict(int)
        
        for action in self.medium_term_window:
            command_counts[action['command']] += 1
        
        total = sum(command_counts.values())
        if total == 0:
            return []
        
        popular = [
            (cmd, count / total)
            for cmd, count in command_counts.items()
        ]
        
        return sorted(popular, key=lambda x: x[1], reverse=True)[:5]
    
    def _find_error_sequences(self) -> List[Dict]:
        """Find sequences that commonly lead to errors."""
        error_sequences = []
        
        actions = list(self.medium_term_window)
        for i in range(len(actions)):
            if not actions[i]['success']:
                # Look for preceding sequence
                start = max(0, i - 3)
                sequence = [actions[j]['command'] for j in range(start, i)]
                
                if len(sequence) >= 2:
                    error_sequences.append({
                        'sequence': sequence,
                        'error': actions[i]['errors'],
                        'timestamp': actions[i]['timestamp']
                    })
        
        return error_sequences
    
    def _analyze_error_contexts(self) -> Dict[str, int]:
        """Analyze contexts where errors commonly occur."""
        error_contexts = defaultdict(int)
        
        for action in self.medium_term_window:
            if not action['success']:
                for key, value in action['context'].items():
                    error_contexts[f"{key}={value}"] += 1
        
        return dict(error_contexts)
    
    def _find_recovery_patterns(self) -> List[Dict]:
        """Find patterns of recovery after errors."""
        recovery_patterns = []
        
        actions = list(self.medium_term_window)
        for i in range(len(actions) - 1):
            if not actions[i]['success'] and actions[i + 1]['success']:
                recovery_patterns.append({
                    'error_command': actions[i]['command'],
                    'recovery_command': actions[i + 1]['command'],
                    'errors': actions[i]['errors']
                })
        
        return recovery_patterns
    
    def _generate_error_recommendations(self, error_analysis: Dict) -> List[str]:
        """Generate recommendations based on error analysis."""
        recommendations = []
        
        # Recommendations based on common errors
        for error, count in error_analysis['common_errors'].items():
            if count > 3:
                recommendations.append(f"Consider using help or documentation for '{error}' (occurs frequently)")
        
        # Recommendations based on recovery patterns
        recovery_commands = defaultdict(int)
        for pattern in error_analysis['recovery_patterns']:
            recovery_commands[pattern['recovery_command']] += 1
        
        for cmd, count in recovery_commands.items():
            if count > 2:
                recommendations.append(f"Try '{cmd}' when encountering errors (successful recovery pattern)")
        
        return recommendations
    
    def _prune_patterns(self) -> None:
        """Remove old or infrequent patterns to prevent memory bloat."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=7)
        
        # Prune sequence patterns
        patterns_to_remove = []
        for sequence, pattern in self.sequence_patterns.items():
            if (pattern['last_seen'] < cutoff_time or 
                pattern['count'] < self.min_pattern_support):
                patterns_to_remove.append(sequence)
        
        for sequence in patterns_to_remove:
            del self.sequence_patterns[sequence]
        
        # Limit total patterns
        if len(self.sequence_patterns) > self.max_patterns:
            sorted_patterns = sorted(
                self.sequence_patterns.items(),
                key=lambda x: x[1]['count'],
                reverse=True
            )
            self.sequence_patterns = dict(sorted_patterns[:self.max_patterns])