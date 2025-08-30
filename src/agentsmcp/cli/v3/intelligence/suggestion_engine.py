"""Suggestion Engine for AgentsMCP CLI v3 Intelligence System.

This module provides context-aware suggestion generation, personalized
recommendations, and proactive assistance based on user patterns and skill level.
"""

import logging
from collections import defaultdict, Counter
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Tuple

from ..models.intelligence_models import (
    UserProfile,
    UserAction,
    SessionContext,
    Suggestion,
    SuggestionType,
    SkillLevel,
    CommandPattern,
    CommandCategory,
    InsufficientDataError,
)


logger = logging.getLogger(__name__)


class SuggestionEngine:
    """Generates personalized suggestions and recommendations for users."""
    
    def __init__(
        self,
        max_suggestions: int = 5,
        suggestion_decay_hours: int = 24,
        min_confidence_threshold: float = 0.3,
    ):
        """Initialize the suggestion engine.
        
        Args:
            max_suggestions: Maximum number of suggestions to provide
            suggestion_decay_hours: Hours after which suggestions expire
            min_confidence_threshold: Minimum confidence score for suggestions
        """
        self.max_suggestions = max_suggestions
        self.suggestion_decay_hours = suggestion_decay_hours
        self.min_confidence_threshold = min_confidence_threshold
        
        # Suggestion templates and rules
        self.skill_progression_map = {
            SkillLevel.BEGINNER: {
                'introduce': ['help', 'status', 'list', 'configure'],
                'avoid': ['pipeline', 'orchestrate', 'delegate', 'debug'],
                'complexity_limit': 2,
            },
            SkillLevel.INTERMEDIATE: {
                'introduce': ['pipeline', 'template', 'monitor', 'analyze'],
                'avoid': ['debug', 'trace', 'customize'],
                'complexity_limit': 4,
            },
            SkillLevel.EXPERT: {
                'introduce': ['orchestrate', 'delegate', 'optimize', 'profile'],
                'avoid': ['basic_help'],
                'complexity_limit': 6,
            },
            SkillLevel.POWER: {
                'introduce': ['customize', 'debug', 'trace', 'benchmark'],
                'avoid': [],
                'complexity_limit': 10,
            },
        }
        
        # Command workflows and next-action patterns
        self.workflow_patterns = {
            'setup': {
                'sequence': ['configure', 'setup', 'initialize'],
                'next_actions': ['test', 'run', 'status'],
                'category': 'setup'
            },
            'development': {
                'sequence': ['create', 'edit', 'test'],
                'next_actions': ['run', 'debug', 'commit'],
                'category': 'development'
            },
            'deployment': {
                'sequence': ['build', 'test', 'deploy'],
                'next_actions': ['monitor', 'status', 'logs'],
                'category': 'deployment'
            },
            'troubleshooting': {
                'sequence': ['error', 'debug', 'trace'],
                'next_actions': ['fix', 'test', 'validate'],
                'category': 'debugging'
            },
        }
        
        # Context-specific suggestions
        self.contextual_suggestions = {
            'after_error': [
                ('help', 'Get help with the error', 0.9),
                ('debug', 'Debug the issue', 0.7),
                ('logs', 'Check logs for details', 0.6),
                ('status', 'Check system status', 0.5),
            ],
            'after_success': [
                ('monitor', 'Monitor the results', 0.6),
                ('optimize', 'Optimize performance', 0.4),
                ('save', 'Save configuration', 0.5),
            ],
            'long_session': [
                ('save', 'Save your progress', 0.8),
                ('status', 'Check system status', 0.6),
                ('break', 'Take a break', 0.4),
            ],
            'new_user': [
                ('help', 'Get started with help', 0.9),
                ('tutorial', 'Try the tutorial', 0.8),
                ('examples', 'View examples', 0.7),
            ],
        }
    
    def generate_suggestions(
        self,
        user_profile: UserProfile,
        current_context: Optional[SessionContext] = None,
        recent_actions: Optional[List[UserAction]] = None,
    ) -> List[Suggestion]:
        """Generate personalized suggestions for the user.
        
        Args:
            user_profile: Current user profile with preferences and patterns
            current_context: Optional current session context
            recent_actions: Optional list of recent user actions
            
        Returns:
            List of personalized suggestions
        """
        suggestions = []
        
        # Generate different types of suggestions
        suggestions.extend(self._generate_next_action_suggestions(user_profile, recent_actions))
        suggestions.extend(self._generate_feature_introduction_suggestions(user_profile))
        suggestions.extend(self._generate_optimization_suggestions(user_profile))
        suggestions.extend(self._generate_error_prevention_suggestions(user_profile))
        suggestions.extend(self._generate_workflow_suggestions(user_profile, recent_actions))
        suggestions.extend(self._generate_skill_advancement_suggestions(user_profile))
        
        # Add contextual suggestions if context available
        if current_context:
            suggestions.extend(self._generate_contextual_suggestions(user_profile, current_context))
        
        # Filter and rank suggestions
        filtered_suggestions = self._filter_suggestions(suggestions, user_profile)
        ranked_suggestions = self._rank_suggestions(filtered_suggestions, user_profile)
        
        # Limit to max suggestions and ensure diversity
        final_suggestions = self._select_diverse_suggestions(ranked_suggestions)
        
        logger.info(f"Generated {len(final_suggestions)} suggestions for user {user_profile.user_id}")
        return final_suggestions
    
    def predict_next_actions(
        self,
        user_profile: UserProfile,
        recent_commands: List[str],
        context: Optional[Dict] = None,
    ) -> List[Suggestion]:
        """Predict likely next actions based on command patterns.
        
        Args:
            user_profile: User profile with command patterns
            recent_commands: Recently executed commands
            context: Optional context information
            
        Returns:
            List of next action suggestions
        """
        if not recent_commands:
            return []
        
        predictions = []
        
        # Pattern-based predictions
        for pattern in user_profile.command_patterns:
            if self._matches_pattern_start(recent_commands, pattern.commands):
                remaining_commands = pattern.commands[len(recent_commands):]
                if remaining_commands:
                    next_cmd = remaining_commands[0]
                    confidence = pattern.confidence * pattern.success_rate
                    
                    suggestion = Suggestion(
                        type=SuggestionType.NEXT_ACTION,
                        text=f"Continue with '{next_cmd}' (part of '{pattern.name}' workflow)",
                        command=next_cmd,
                        category="workflow_continuation",
                        confidence=confidence,
                        priority=8,
                        skill_level_target=user_profile.skill_level,
                        metadata={'pattern_name': pattern.name, 'workflow': True}
                    )
                    predictions.append(suggestion)
        
        # Workflow-based predictions
        last_command = recent_commands[-1]
        for workflow_name, workflow in self.workflow_patterns.items():
            if last_command in workflow['sequence']:
                idx = workflow['sequence'].index(last_command)
                if idx < len(workflow['sequence']) - 1:
                    next_cmd = workflow['sequence'][idx + 1]
                    
                    suggestion = Suggestion(
                        type=SuggestionType.NEXT_ACTION,
                        text=f"Next step: '{next_cmd}' in {workflow_name} workflow",
                        command=next_cmd,
                        category=workflow['category'],
                        confidence=0.7,
                        priority=7,
                        skill_level_target=user_profile.skill_level,
                        metadata={'workflow': workflow_name}
                    )
                    predictions.append(suggestion)
        
        return predictions
    
    def get_feature_recommendations(
        self,
        user_profile: UserProfile,
        exclude_known: bool = True,
    ) -> List[Suggestion]:
        """Get recommendations for new features to introduce.
        
        Args:
            user_profile: User profile to analyze
            exclude_known: Whether to exclude features user already knows
            
        Returns:
            List of feature introduction suggestions
        """
        skill_config = self.skill_progression_map.get(user_profile.skill_level, {})
        features_to_introduce = skill_config.get('introduce', [])
        features_to_avoid = skill_config.get('avoid', [])
        
        recommendations = []
        
        for feature in features_to_introduce:
            # Skip if user already uses this feature frequently
            if exclude_known and feature in user_profile.favorite_commands[:5]:
                continue
            
            # Skip if feature should be avoided for this skill level
            if feature in features_to_avoid:
                continue
            
            confidence = self._calculate_feature_readiness(user_profile, feature)
            
            if confidence >= self.min_confidence_threshold:
                suggestion = Suggestion(
                    type=SuggestionType.FEATURE_INTRODUCTION,
                    text=self._get_feature_introduction_text(feature, user_profile.skill_level),
                    command=feature,
                    category="feature_discovery",
                    confidence=confidence,
                    priority=6,
                    skill_level_target=user_profile.skill_level,
                    metadata={'feature': feature, 'introduction': True}
                )
                recommendations.append(suggestion)
        
        return recommendations
    
    def suggest_optimizations(
        self,
        user_profile: UserProfile,
        recent_actions: Optional[List[UserAction]] = None,
    ) -> List[Suggestion]:
        """Suggest optimizations based on user patterns.
        
        Args:
            user_profile: User profile to analyze
            recent_actions: Optional recent actions for analysis
            
        Returns:
            List of optimization suggestions
        """
        optimizations = []
        
        # Suggest shortcuts for frequently used command sequences
        for pattern in user_profile.command_patterns:
            if pattern.frequency > 5 and len(pattern.commands) > 2:
                suggestion = Suggestion(
                    type=SuggestionType.OPTIMIZATION,
                    text=f"Create a shortcut for '{pattern.name}' (used {pattern.frequency} times)",
                    command=f"alias {pattern.name.lower().replace(' ', '_')}",
                    category="efficiency",
                    confidence=min(0.9, pattern.frequency / 10.0),
                    priority=5,
                    skill_level_target=user_profile.skill_level,
                    metadata={'pattern': pattern.name, 'frequency': pattern.frequency}
                )
                optimizations.append(suggestion)
        
        # Suggest configuration optimizations based on error patterns
        if user_profile.success_rate < 0.8:
            suggestion = Suggestion(
                type=SuggestionType.OPTIMIZATION,
                text="Consider adjusting configuration to reduce errors",
                command="configure",
                category="reliability",
                confidence=0.6,
                priority=7,
                skill_level_target=user_profile.skill_level,
                metadata={'reason': 'high_error_rate'}
            )
            optimizations.append(suggestion)
        
        # Suggest timeout adjustments for slow operations
        if recent_actions:
            avg_duration = sum(a.duration_ms for a in recent_actions) / len(recent_actions)
            if avg_duration > 10000:  # >10 seconds
                suggestion = Suggestion(
                    type=SuggestionType.OPTIMIZATION,
                    text="Consider increasing timeout for long-running operations",
                    command="configure --timeout 60",
                    category="performance",
                    confidence=0.5,
                    priority=4,
                    skill_level_target=user_profile.skill_level,
                    metadata={'avg_duration_ms': avg_duration}
                )
                optimizations.append(suggestion)
        
        return optimizations
    
    def get_error_prevention_tips(
        self,
        user_profile: UserProfile,
        error_context: Optional[Dict] = None,
    ) -> List[Suggestion]:
        """Generate tips to prevent common errors.
        
        Args:
            user_profile: User profile with error history
            error_context: Optional context about recent errors
            
        Returns:
            List of error prevention suggestions
        """
        prevention_tips = []
        
        # Analyze common errors
        if user_profile.common_errors:
            most_common_error = max(user_profile.common_errors.items(), key=lambda x: x[1])
            error_type, count = most_common_error
            
            if count > 3:
                suggestion = Suggestion(
                    type=SuggestionType.ERROR_PREVENTION,
                    text=f"Tip: Use 'validate' before operations to prevent '{error_type}'",
                    command="validate",
                    category="error_prevention",
                    confidence=0.8,
                    priority=8,
                    skill_level_target=user_profile.skill_level,
                    metadata={'error_type': error_type, 'frequency': count}
                )
                prevention_tips.append(suggestion)
        
        # Suggest dry-run for destructive operations
        if user_profile.skill_level in [SkillLevel.BEGINNER, SkillLevel.INTERMEDIATE]:
            suggestion = Suggestion(
                type=SuggestionType.ERROR_PREVENTION,
                text="Use '--dry-run' to preview changes before applying them",
                command="--dry-run",
                category="safety",
                confidence=0.7,
                priority=6,
                skill_level_target=user_profile.skill_level,
                metadata={'safety_tip': True}
            )
            prevention_tips.append(suggestion)
        
        # Suggest backup before major changes
        if user_profile.skill_level != SkillLevel.BEGINNER:
            suggestion = Suggestion(
                type=SuggestionType.ERROR_PREVENTION,
                text="Create backup before major configuration changes",
                command="backup",
                category="safety",
                confidence=0.6,
                priority=5,
                skill_level_target=user_profile.skill_level,
                metadata={'backup_tip': True}
            )
            prevention_tips.append(suggestion)
        
        return prevention_tips
    
    def _generate_next_action_suggestions(
        self,
        user_profile: UserProfile,
        recent_actions: Optional[List[UserAction]],
    ) -> List[Suggestion]:
        """Generate next action suggestions."""
        if not recent_actions:
            return []
        
        recent_commands = [action.command for action in recent_actions[-5:]]
        return self.predict_next_actions(user_profile, recent_commands)
    
    def _generate_feature_introduction_suggestions(
        self,
        user_profile: UserProfile,
    ) -> List[Suggestion]:
        """Generate feature introduction suggestions."""
        return self.get_feature_recommendations(user_profile)
    
    def _generate_optimization_suggestions(
        self,
        user_profile: UserProfile,
    ) -> List[Suggestion]:
        """Generate optimization suggestions."""
        return self.suggest_optimizations(user_profile)
    
    def _generate_error_prevention_suggestions(
        self,
        user_profile: UserProfile,
    ) -> List[Suggestion]:
        """Generate error prevention suggestions."""
        return self.get_error_prevention_tips(user_profile)
    
    def _generate_workflow_suggestions(
        self,
        user_profile: UserProfile,
        recent_actions: Optional[List[UserAction]],
    ) -> List[Suggestion]:
        """Generate workflow completion suggestions."""
        suggestions = []
        
        if not recent_actions:
            return suggestions
        
        last_action = recent_actions[-1]
        
        # Check if user is in middle of a known workflow
        for workflow_name, workflow in self.workflow_patterns.items():
            if last_action.command in workflow['sequence']:
                next_actions = workflow.get('next_actions', [])
                
                for next_cmd in next_actions:
                    suggestion = Suggestion(
                        type=SuggestionType.WORKFLOW_COMPLETION,
                        text=f"Complete {workflow_name} workflow with '{next_cmd}'",
                        command=next_cmd,
                        category=workflow['category'],
                        confidence=0.6,
                        priority=6,
                        skill_level_target=user_profile.skill_level,
                        metadata={'workflow': workflow_name}
                    )
                    suggestions.append(suggestion)
                break
        
        return suggestions
    
    def _generate_skill_advancement_suggestions(
        self,
        user_profile: UserProfile,
    ) -> List[Suggestion]:
        """Generate suggestions for skill advancement."""
        suggestions = []
        
        # Suggest advancement if user is ready
        if self._should_advance_skill_level(user_profile):
            next_level = self._get_next_skill_level(user_profile.skill_level)
            if next_level:
                next_features = self.skill_progression_map.get(next_level, {}).get('introduce', [])
                
                if next_features:
                    feature = next_features[0]  # Introduce first advanced feature
                    suggestion = Suggestion(
                        type=SuggestionType.SKILL_ADVANCEMENT,
                        text=f"Ready for advanced features? Try '{feature}'",
                        command=feature,
                        category="skill_progression",
                        confidence=0.7,
                        priority=4,
                        skill_level_target=next_level,
                        metadata={'advancement': True, 'target_level': next_level}
                    )
                    suggestions.append(suggestion)
        
        return suggestions
    
    def _generate_contextual_suggestions(
        self,
        user_profile: UserProfile,
        context: SessionContext,
    ) -> List[Suggestion]:
        """Generate context-specific suggestions."""
        suggestions = []
        
        # After error suggestions
        if context.errors_encountered:
            for cmd, desc, confidence in self.contextual_suggestions['after_error']:
                suggestion = Suggestion(
                    type=SuggestionType.ERROR_PREVENTION,
                    text=desc,
                    command=cmd,
                    category="error_recovery",
                    confidence=confidence,
                    priority=9,
                    skill_level_target=user_profile.skill_level,
                    metadata={'context': 'after_error'}
                )
                suggestions.append(suggestion)
        
        # Long session suggestions
        if context.duration_ms > 3600000:  # > 1 hour
            for cmd, desc, confidence in self.contextual_suggestions['long_session']:
                suggestion = Suggestion(
                    type=SuggestionType.OPTIMIZATION,
                    text=desc,
                    command=cmd,
                    category="session_management",
                    confidence=confidence,
                    priority=3,
                    skill_level_target=user_profile.skill_level,
                    metadata={'context': 'long_session'}
                )
                suggestions.append(suggestion)
        
        # New user suggestions
        if user_profile.total_commands < 5:
            for cmd, desc, confidence in self.contextual_suggestions['new_user']:
                suggestion = Suggestion(
                    type=SuggestionType.FEATURE_INTRODUCTION,
                    text=desc,
                    command=cmd,
                    category="onboarding",
                    confidence=confidence,
                    priority=10,
                    skill_level_target=SkillLevel.BEGINNER,
                    metadata={'context': 'new_user'}
                )
                suggestions.append(suggestion)
        
        return suggestions
    
    def _filter_suggestions(
        self,
        suggestions: List[Suggestion],
        user_profile: UserProfile,
    ) -> List[Suggestion]:
        """Filter suggestions based on user profile and preferences."""
        filtered = []
        
        for suggestion in suggestions:
            # Filter by confidence threshold
            if suggestion.confidence < self.min_confidence_threshold:
                continue
            
            # Filter by skill level appropriateness
            if not self._is_suggestion_appropriate(suggestion, user_profile):
                continue
            
            # Filter out recently suggested items (avoid repetition)
            if self._was_recently_suggested(suggestion, user_profile):
                continue
            
            filtered.append(suggestion)
        
        return filtered
    
    def _rank_suggestions(
        self,
        suggestions: List[Suggestion],
        user_profile: UserProfile,
    ) -> List[Suggestion]:
        """Rank suggestions by relevance and importance."""
        def score_suggestion(suggestion: Suggestion) -> float:
            score = suggestion.confidence * suggestion.priority
            
            # Boost score based on user preferences
            suggestion_level = user_profile.preferences.get('suggestion_level', 3)
            if suggestion_level > 3:
                score *= 1.2
            elif suggestion_level < 3:
                score *= 0.8
            
            # Boost contextual suggestions
            if suggestion.metadata.get('context'):
                score *= 1.3
            
            # Boost workflow suggestions
            if suggestion.metadata.get('workflow'):
                score *= 1.2
            
            return score
        
        ranked = sorted(suggestions, key=score_suggestion, reverse=True)
        return ranked
    
    def _select_diverse_suggestions(
        self,
        suggestions: List[Suggestion],
    ) -> List[Suggestion]:
        """Select diverse suggestions to avoid redundancy."""
        if len(suggestions) <= self.max_suggestions:
            return suggestions
        
        selected = []
        categories_used = set()
        
        for suggestion in suggestions:
            if len(selected) >= self.max_suggestions:
                break
            
            # Ensure diversity by category
            if suggestion.category not in categories_used or len(categories_used) >= 3:
                selected.append(suggestion)
                categories_used.add(suggestion.category)
        
        return selected
    
    def _matches_pattern_start(self, commands: List[str], pattern: List[str]) -> bool:
        """Check if recent commands match the start of a pattern."""
        if len(commands) >= len(pattern):
            return False
        
        for i, cmd in enumerate(commands):
            if i >= len(pattern) or cmd != pattern[i]:
                return False
        
        return True
    
    def _calculate_feature_readiness(self, user_profile: UserProfile, feature: str) -> float:
        """Calculate readiness score for introducing a new feature."""
        base_confidence = 0.5
        
        # Increase confidence based on skill level
        skill_bonus = {
            SkillLevel.BEGINNER: 0.0,
            SkillLevel.INTERMEDIATE: 0.2,
            SkillLevel.EXPERT: 0.3,
            SkillLevel.POWER: 0.4,
        }.get(user_profile.skill_level, 0.0)
        
        # Increase confidence based on success rate
        success_bonus = (user_profile.success_rate - 0.5) * 0.4
        
        # Increase confidence based on command diversity
        diversity_bonus = min(len(user_profile.favorite_commands) / 20.0, 0.2)
        
        return min(1.0, base_confidence + skill_bonus + success_bonus + diversity_bonus)
    
    def _get_feature_introduction_text(self, feature: str, skill_level: SkillLevel) -> str:
        """Get appropriate introduction text for a feature."""
        feature_descriptions = {
            'help': "Get help and documentation for commands",
            'status': "Check system and service status",
            'configure': "Configure system settings and preferences",
            'pipeline': "Create and manage command pipelines",
            'template': "Use and manage command templates",
            'monitor': "Monitor system performance and activities",
            'orchestrate': "Orchestrate complex multi-step workflows",
            'delegate': "Delegate tasks to specialized agents",
            'optimize': "Optimize system performance and efficiency",
            'debug': "Debug issues and trace execution",
            'profile': "Profile system performance and usage",
            'customize': "Customize system behavior and appearance",
        }
        
        description = feature_descriptions.get(feature, f"Try the '{feature}' command")
        
        if skill_level == SkillLevel.BEGINNER:
            return f"New feature: {description}"
        else:
            return f"Advanced feature: {description}"
    
    def _should_advance_skill_level(self, user_profile: UserProfile) -> bool:
        """Determine if user is ready to advance to next skill level."""
        thresholds = {
            SkillLevel.BEGINNER: user_profile.total_commands >= 15 and user_profile.success_rate > 0.8,
            SkillLevel.INTERMEDIATE: user_profile.total_commands >= 60 and user_profile.success_rate > 0.85,
            SkillLevel.EXPERT: user_profile.total_commands >= 250 and len(user_profile.advanced_features_used) >= 5,
        }
        
        return thresholds.get(user_profile.skill_level, False)
    
    def _get_next_skill_level(self, current_level: SkillLevel) -> Optional[SkillLevel]:
        """Get the next skill level for advancement."""
        progression = {
            SkillLevel.BEGINNER: SkillLevel.INTERMEDIATE,
            SkillLevel.INTERMEDIATE: SkillLevel.EXPERT,
            SkillLevel.EXPERT: SkillLevel.POWER,
        }
        
        return progression.get(current_level)
    
    def _is_suggestion_appropriate(self, suggestion: Suggestion, user_profile: UserProfile) -> bool:
        """Check if suggestion is appropriate for user's skill level."""
        skill_config = self.skill_progression_map.get(user_profile.skill_level, {})
        avoided_features = skill_config.get('avoid', [])
        
        # Don't suggest avoided features
        if suggestion.command in avoided_features:
            return False
        
        # Don't suggest features too advanced for skill level
        target_level = suggestion.skill_level_target
        if target_level == SkillLevel.POWER and user_profile.skill_level in [SkillLevel.BEGINNER, SkillLevel.INTERMEDIATE]:
            return False
        
        return True
    
    def _was_recently_suggested(self, suggestion: Suggestion, user_profile: UserProfile) -> bool:
        """Check if suggestion was made recently to avoid repetition."""
        # This would typically check against a history of suggestions
        # For now, we'll implement basic deduplication
        return False  # Placeholder implementation