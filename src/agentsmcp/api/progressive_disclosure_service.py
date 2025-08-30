"""
Progressive Disclosure Service with User Profiling

Adaptive complexity management system that automatically adjusts interface
complexity based on user skill level, behavior patterns, and learning progression.
"""

import json
import asyncio
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
import math

from .base import APIBase, APIResponse, APIError


class SkillLevel(str, Enum):
    """User skill levels for progressive disclosure."""
    NOVICE = "novice"          # First-time users, need maximum guidance
    BEGINNER = "beginner"      # Basic understanding, need explanations
    INTERMEDIATE = "intermediate"  # Comfortable with basics, want efficiency
    ADVANCED = "advanced"      # Experienced, want advanced features
    EXPERT = "expert"         # Power users, want full control


class LearningStage(str, Enum):
    """Learning stages for skill progression tracking."""
    AWARENESS = "awareness"    # Just learned about feature
    TRIAL = "trial"           # Trying feature for first time
    ADOPTION = "adoption"     # Using feature regularly
    MASTERY = "mastery"       # Expert level usage


class InterfaceComplexity(str, Enum):
    """Interface complexity levels."""
    MINIMAL = "minimal"       # Only essential features
    BASIC = "basic"          # Core features with guidance
    STANDARD = "standard"    # Full features with some guidance
    ADVANCED = "advanced"    # All features, minimal guidance
    EXPERT = "expert"        # All features, no hand-holding


@dataclass
class UserProfile:
    """Comprehensive user profile for progressive disclosure."""
    user_id: str
    skill_level: SkillLevel
    created_at: datetime
    last_active: datetime
    
    # Usage statistics
    total_sessions: int = 0
    total_commands: int = 0
    successful_commands: int = 0
    error_rate: float = 0.0
    
    # Feature usage
    features_discovered: Set[str] = field(default_factory=set)
    features_mastered: Set[str] = field(default_factory=set)
    feature_usage_count: Dict[str, int] = field(default_factory=dict)
    
    # Learning progression
    learning_stages: Dict[str, LearningStage] = field(default_factory=dict)
    skill_progression_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Preferences
    preferred_complexity: Optional[InterfaceComplexity] = None
    help_frequency_preference: str = "auto"  # auto, frequent, minimal, none
    explanation_preference: str = "contextual"  # full, contextual, minimal
    
    # Behavioral patterns
    session_patterns: Dict[str, Any] = field(default_factory=dict)
    common_workflows: List[List[str]] = field(default_factory=list)
    pain_points: List[str] = field(default_factory=list)


@dataclass
class ContextualHint:
    """Contextual hint for progressive disclosure."""
    id: str
    content: str
    trigger_conditions: Dict[str, Any]
    complexity_level: InterfaceComplexity
    importance: float  # 0.0 to 1.0
    shown_count: int = 0
    last_shown: Optional[datetime] = None
    user_dismissed: bool = False


@dataclass
class AdaptiveInterface:
    """Adaptive interface configuration."""
    user_id: str
    current_complexity: InterfaceComplexity
    visible_features: List[str]
    hidden_features: List[str]
    contextual_hints: List[ContextualHint]
    simplified_commands: Dict[str, str]
    guided_workflows: List[Dict[str, Any]]


class ProgressiveDisclosureService(APIBase):
    """Advanced progressive disclosure service with ML-powered user profiling."""
    
    def __init__(self):
        super().__init__("progressive_disclosure_service")
        self.user_profiles: Dict[str, UserProfile] = {}
        self.adaptive_interfaces: Dict[str, AdaptiveInterface] = {}
        self.feature_definitions: Dict[str, Dict[str, Any]] = {}
        self.contextual_hints: Dict[str, ContextualHint] = {}
        self.complexity_thresholds: Dict[str, float] = {}
        
        # Initialize system
        asyncio.create_task(self._initialize_system())
        asyncio.create_task(self._periodic_profile_updates())
    
    async def _initialize_system(self):
        """Initialize the progressive disclosure system."""
        # Define feature complexity levels and requirements
        self.feature_definitions = {
            "basic_chat": {
                "complexity": InterfaceComplexity.MINIMAL,
                "required_skills": [],
                "prerequisites": [],
                "description": "Basic chat functionality"
            },
            "agent_selection": {
                "complexity": InterfaceComplexity.BASIC,
                "required_skills": ["basic_chat"],
                "prerequisites": ["basic_chat"],
                "description": "Choose specific agents for conversation"
            },
            "pipeline_execution": {
                "complexity": InterfaceComplexity.STANDARD,
                "required_skills": ["basic_chat", "agent_selection"],
                "prerequisites": ["agent_selection"],
                "description": "Execute automated workflows"
            },
            "symphony_mode": {
                "complexity": InterfaceComplexity.ADVANCED,
                "required_skills": ["pipeline_execution", "agent_management"],
                "prerequisites": ["pipeline_execution", "agent_management"],
                "description": "Multi-agent coordination"
            },
            "custom_orchestration": {
                "complexity": InterfaceComplexity.EXPERT,
                "required_skills": ["symphony_mode", "advanced_configuration"],
                "prerequisites": ["symphony_mode"],
                "description": "Custom multi-agent workflows"
            },
            "agent_management": {
                "complexity": InterfaceComplexity.STANDARD,
                "required_skills": ["agent_selection"],
                "prerequisites": ["agent_selection"],
                "description": "Create and manage agents"
            },
            "advanced_configuration": {
                "complexity": InterfaceComplexity.ADVANCED,
                "required_skills": ["agent_management", "pipeline_execution"],
                "prerequisites": ["agent_management"],
                "description": "Advanced system configuration"
            },
            "api_integration": {
                "complexity": InterfaceComplexity.EXPERT,
                "required_skills": ["advanced_configuration"],
                "prerequisites": ["advanced_configuration"],
                "description": "Custom API integrations"
            }
        }
        
        # Define skill progression thresholds
        self.complexity_thresholds = {
            "beginner_to_intermediate": 20,  # 20 successful commands
            "intermediate_to_advanced": 100,  # 100 successful commands
            "advanced_to_expert": 500,       # 500 successful commands
            "feature_mastery": 10,           # 10 successful uses of a feature
            "error_rate_threshold": 0.2      # 20% error rate indicates struggle
        }
        
        # Initialize contextual hints
        await self._initialize_contextual_hints()
    
    async def _initialize_contextual_hints(self):
        """Initialize contextual hints for different scenarios."""
        hints = [
            ContextualHint(
                id="first_time_user",
                content="Welcome! Try starting with a simple chat command like 'chat with an expert'.",
                trigger_conditions={"total_commands": 0, "skill_level": "novice"},
                complexity_level=InterfaceComplexity.MINIMAL,
                importance=0.9
            ),
            ContextualHint(
                id="agent_selection_intro",
                content="You can specify which agent to chat with using --agent [name] or by saying 'chat with [agent name]'.",
                trigger_conditions={"feature": "agent_selection", "stage": "awareness"},
                complexity_level=InterfaceComplexity.BASIC,
                importance=0.8
            ),
            ContextualHint(
                id="pipeline_introduction",
                content="Pipelines let you automate complex workflows. Try 'run data analysis pipeline' to get started.",
                trigger_conditions={"successful_commands": 10, "feature": "pipeline_execution", "stage": "awareness"},
                complexity_level=InterfaceComplexity.STANDARD,
                importance=0.7
            ),
            ContextualHint(
                id="symphony_mode_teaser",
                content="Ready for advanced coordination? Symphony mode lets multiple agents work together on complex tasks.",
                trigger_conditions={"successful_commands": 50, "features_mastered": ["pipeline_execution"]},
                complexity_level=InterfaceComplexity.ADVANCED,
                importance=0.6
            ),
            ContextualHint(
                id="expert_features",
                content="You've mastered the basics! Explore custom orchestration and API integrations for maximum flexibility.",
                trigger_conditions={"skill_level": "advanced", "features_mastered_count": 5},
                complexity_level=InterfaceComplexity.EXPERT,
                importance=0.5
            )
        ]
        
        for hint in hints:
            self.contextual_hints[hint.id] = hint
    
    async def create_or_update_profile(
        self, 
        user_id: str,
        initial_skill_level: Optional[SkillLevel] = None
    ) -> APIResponse:
        """Create or update a user profile for progressive disclosure."""
        return await self._execute_with_metrics(
            "create_or_update_profile",
            self._create_or_update_profile_internal,
            user_id,
            initial_skill_level
        )
    
    async def _create_or_update_profile_internal(
        self,
        user_id: str,
        initial_skill_level: Optional[SkillLevel]
    ) -> UserProfile:
        """Internal logic for profile creation/update."""
        current_time = datetime.utcnow()
        
        if user_id in self.user_profiles:
            # Update existing profile
            profile = self.user_profiles[user_id]
            profile.last_active = current_time
        else:
            # Create new profile
            profile = UserProfile(
                user_id=user_id,
                skill_level=initial_skill_level or SkillLevel.NOVICE,
                created_at=current_time,
                last_active=current_time
            )
            self.user_profiles[user_id] = profile
            
            # Create adaptive interface
            await self._create_adaptive_interface(profile)
        
        return profile
    
    async def _create_adaptive_interface(self, profile: UserProfile):
        """Create adaptive interface configuration for user."""
        complexity = self._determine_interface_complexity(profile)
        
        # Determine visible features based on complexity and skill level
        visible_features = self._get_visible_features_for_complexity(complexity, profile.skill_level)
        hidden_features = [
            feature for feature in self.feature_definitions.keys()
            if feature not in visible_features
        ]
        
        # Get relevant contextual hints
        relevant_hints = self._get_relevant_hints(profile)
        
        # Generate simplified commands for beginners
        simplified_commands = self._generate_simplified_commands(profile.skill_level)
        
        # Create guided workflows
        guided_workflows = self._create_guided_workflows(profile.skill_level)
        
        adaptive_interface = AdaptiveInterface(
            user_id=profile.user_id,
            current_complexity=complexity,
            visible_features=visible_features,
            hidden_features=hidden_features,
            contextual_hints=relevant_hints,
            simplified_commands=simplified_commands,
            guided_workflows=guided_workflows
        )
        
        self.adaptive_interfaces[profile.user_id] = adaptive_interface
    
    def _determine_interface_complexity(self, profile: UserProfile) -> InterfaceComplexity:
        """Determine appropriate interface complexity for user."""
        # Use explicit preference if set
        if profile.preferred_complexity:
            return profile.preferred_complexity
        
        # Calculate complexity based on skill level and usage patterns
        skill_factor = {
            SkillLevel.NOVICE: 0.0,
            SkillLevel.BEGINNER: 0.2,
            SkillLevel.INTERMEDIATE: 0.5,
            SkillLevel.ADVANCED: 0.8,
            SkillLevel.EXPERT: 1.0
        }[profile.skill_level]
        
        # Adjust based on success rate
        success_rate = (profile.successful_commands / profile.total_commands) if profile.total_commands > 0 else 1.0
        success_factor = min(success_rate * 1.2, 1.0)  # Cap at 1.0
        
        # Adjust based on feature mastery
        mastery_factor = min(len(profile.features_mastered) / 10.0, 1.0)  # Cap at 1.0
        
        # Combined complexity score
        complexity_score = (skill_factor * 0.5 + success_factor * 0.3 + mastery_factor * 0.2)
        
        # Map to complexity levels
        if complexity_score < 0.2:
            return InterfaceComplexity.MINIMAL
        elif complexity_score < 0.4:
            return InterfaceComplexity.BASIC
        elif complexity_score < 0.7:
            return InterfaceComplexity.STANDARD
        elif complexity_score < 0.9:
            return InterfaceComplexity.ADVANCED
        else:
            return InterfaceComplexity.EXPERT
    
    def _get_visible_features_for_complexity(
        self, 
        complexity: InterfaceComplexity,
        skill_level: SkillLevel
    ) -> List[str]:
        """Get features that should be visible for given complexity level."""
        visible_features = []
        
        complexity_order = [
            InterfaceComplexity.MINIMAL,
            InterfaceComplexity.BASIC,
            InterfaceComplexity.STANDARD,
            InterfaceComplexity.ADVANCED,
            InterfaceComplexity.EXPERT
        ]
        
        current_complexity_index = complexity_order.index(complexity)
        
        for feature_name, feature_def in self.feature_definitions.items():
            feature_complexity_index = complexity_order.index(feature_def["complexity"])
            
            # Include feature if it matches or is below current complexity
            if feature_complexity_index <= current_complexity_index:
                visible_features.append(feature_name)
        
        return visible_features
    
    def _get_relevant_hints(self, profile: UserProfile) -> List[ContextualHint]:
        """Get contextually relevant hints for the user."""
        relevant_hints = []
        
        for hint in self.contextual_hints.values():
            if hint.user_dismissed:
                continue
            
            # Check if hint conditions are met
            if self._hint_conditions_met(hint, profile):
                # Check if hint hasn't been shown too frequently
                if self._should_show_hint(hint):
                    relevant_hints.append(hint)
        
        # Sort by importance
        relevant_hints.sort(key=lambda h: h.importance, reverse=True)
        
        return relevant_hints[:5]  # Max 5 hints at a time
    
    def _hint_conditions_met(self, hint: ContextualHint, profile: UserProfile) -> bool:
        """Check if hint trigger conditions are met for the user."""
        conditions = hint.trigger_conditions
        
        # Check skill level condition
        if "skill_level" in conditions:
            if profile.skill_level.value != conditions["skill_level"]:
                return False
        
        # Check command count conditions
        if "total_commands" in conditions:
            if profile.total_commands != conditions["total_commands"]:
                return False
        
        if "successful_commands" in conditions:
            if profile.successful_commands < conditions["successful_commands"]:
                return False
        
        # Check feature conditions
        if "feature" in conditions:
            feature = conditions["feature"]
            if "stage" in conditions:
                required_stage = LearningStage(conditions["stage"])
                current_stage = profile.learning_stages.get(feature, LearningStage.AWARENESS)
                if current_stage != required_stage:
                    return False
        
        # Check mastery conditions
        if "features_mastered" in conditions:
            required_features = conditions["features_mastered"]
            if not all(feature in profile.features_mastered for feature in required_features):
                return False
        
        if "features_mastered_count" in conditions:
            if len(profile.features_mastered) < conditions["features_mastered_count"]:
                return False
        
        return True
    
    def _should_show_hint(self, hint: ContextualHint) -> bool:
        """Determine if hint should be shown based on frequency rules."""
        # Don't show hint too frequently
        if hint.last_shown:
            time_since_last = datetime.utcnow() - hint.last_shown
            if time_since_last < timedelta(hours=24):  # Wait 24 hours between showings
                return False
        
        # Don't show hint too many times
        if hint.shown_count >= 3:  # Max 3 times
            return False
        
        return True
    
    def _generate_simplified_commands(self, skill_level: SkillLevel) -> Dict[str, str]:
        """Generate simplified command mappings for beginners."""
        if skill_level in [SkillLevel.NOVICE, SkillLevel.BEGINNER]:
            return {
                "chat": "agentsmcp chat --interactive",
                "help": "agentsmcp --help",
                "list agents": "agentsmcp discovery list --type agent",
                "run workflow": "agentsmcp pipeline run basic",
                "show config": "agentsmcp config show"
            }
        else:
            return {}  # No simplification for intermediate+ users
    
    def _create_guided_workflows(self, skill_level: SkillLevel) -> List[Dict[str, Any]]:
        """Create guided workflows based on skill level."""
        workflows = []
        
        if skill_level == SkillLevel.NOVICE:
            workflows.append({
                "name": "Getting Started",
                "description": "Your first conversation with an AI agent",
                "steps": [
                    {"action": "chat", "description": "Start a conversation"},
                    {"action": "ask_question", "description": "Ask a simple question"},
                    {"action": "explore_help", "description": "Learn about available commands"}
                ],
                "estimated_duration": "5 minutes"
            })
        
        elif skill_level == SkillLevel.BEGINNER:
            workflows.extend([
                {
                    "name": "Choose Your Agent",
                    "description": "Learn to select different agents",
                    "steps": [
                        {"action": "list_agents", "description": "See available agents"},
                        {"action": "select_agent", "description": "Choose an agent"},
                        {"action": "compare_agents", "description": "Try different agents"}
                    ],
                    "estimated_duration": "10 minutes"
                }
            ])
        
        elif skill_level in [SkillLevel.INTERMEDIATE, SkillLevel.ADVANCED]:
            workflows.append({
                "name": "Automate with Pipelines",
                "description": "Set up automated workflows",
                "steps": [
                    {"action": "explore_templates", "description": "Browse pipeline templates"},
                    {"action": "create_pipeline", "description": "Create your first pipeline"},
                    {"action": "run_pipeline", "description": "Execute the pipeline"}
                ],
                "estimated_duration": "20 minutes"
            })
        
        return workflows
    
    async def track_user_interaction(
        self,
        user_id: str,
        command: str,
        success: bool,
        features_used: List[str] = None,
        context: Dict[str, Any] = None
    ) -> APIResponse:
        """Track user interaction for learning and adaptation."""
        return await self._execute_with_metrics(
            "track_user_interaction",
            self._track_user_interaction_internal,
            user_id,
            command,
            success,
            features_used or [],
            context or {}
        )
    
    async def _track_user_interaction_internal(
        self,
        user_id: str,
        command: str,
        success: bool,
        features_used: List[str],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Internal logic for tracking user interactions."""
        # Ensure user profile exists
        if user_id not in self.user_profiles:
            await self._create_or_update_profile_internal(user_id, None)
        
        profile = self.user_profiles[user_id]
        
        # Update basic statistics
        profile.total_commands += 1
        profile.last_active = datetime.utcnow()
        
        if success:
            profile.successful_commands += 1
        
        # Update error rate
        profile.error_rate = 1.0 - (profile.successful_commands / profile.total_commands)
        
        # Track feature usage
        for feature in features_used:
            # Update usage count
            profile.feature_usage_count[feature] = profile.feature_usage_count.get(feature, 0) + 1
            
            # Add to discovered features
            profile.features_discovered.add(feature)
            
            # Update learning stage
            usage_count = profile.feature_usage_count[feature]
            if usage_count == 1:
                profile.learning_stages[feature] = LearningStage.TRIAL
            elif usage_count >= 3:
                profile.learning_stages[feature] = LearningStage.ADOPTION
            elif usage_count >= self.complexity_thresholds["feature_mastery"]:
                profile.learning_stages[feature] = LearningStage.MASTERY
                profile.features_mastered.add(feature)
        
        # Check for skill level progression
        await self._check_skill_progression(profile)
        
        # Update adaptive interface
        await self._update_adaptive_interface(profile)
        
        return {
            "interaction_tracked": True,
            "current_skill_level": profile.skill_level.value,
            "features_discovered": len(profile.features_discovered),
            "features_mastered": len(profile.features_mastered),
            "success_rate": profile.successful_commands / profile.total_commands
        }
    
    async def _check_skill_progression(self, profile: UserProfile):
        """Check if user should progress to next skill level."""
        current_level = profile.skill_level
        success_rate = profile.successful_commands / profile.total_commands if profile.total_commands > 0 else 0.0
        
        # Only progress if error rate is low
        if profile.error_rate > self.complexity_thresholds["error_rate_threshold"]:
            return
        
        # Check progression conditions
        if (current_level == SkillLevel.NOVICE and 
            profile.successful_commands >= 5 and 
            len(profile.features_discovered) >= 2):
            
            await self._promote_skill_level(profile, SkillLevel.BEGINNER)
            
        elif (current_level == SkillLevel.BEGINNER and 
              profile.successful_commands >= self.complexity_thresholds["beginner_to_intermediate"] and
              len(profile.features_mastered) >= 2):
            
            await self._promote_skill_level(profile, SkillLevel.INTERMEDIATE)
            
        elif (current_level == SkillLevel.INTERMEDIATE and 
              profile.successful_commands >= self.complexity_thresholds["intermediate_to_advanced"] and
              len(profile.features_mastered) >= 5):
            
            await self._promote_skill_level(profile, SkillLevel.ADVANCED)
            
        elif (current_level == SkillLevel.ADVANCED and 
              profile.successful_commands >= self.complexity_thresholds["advanced_to_expert"] and
              len(profile.features_mastered) >= 8):
            
            await self._promote_skill_level(profile, SkillLevel.EXPERT)
    
    async def _promote_skill_level(self, profile: UserProfile, new_level: SkillLevel):
        """Promote user to new skill level."""
        old_level = profile.skill_level
        profile.skill_level = new_level
        
        # Record progression in history
        progression_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "from_level": old_level.value,
            "to_level": new_level.value,
            "trigger": "automatic",
            "successful_commands": profile.successful_commands,
            "features_mastered": len(profile.features_mastered)
        }
        
        profile.skill_progression_history.append(progression_entry)
        
        self.logger.info(f"User {profile.user_id} promoted from {old_level.value} to {new_level.value}")
    
    async def _update_adaptive_interface(self, profile: UserProfile):
        """Update adaptive interface based on current profile."""
        if profile.user_id in self.adaptive_interfaces:
            # Recreate adaptive interface with updated profile
            await self._create_adaptive_interface(profile)
    
    async def get_adaptive_interface(self, user_id: str) -> APIResponse:
        """Get adaptive interface configuration for user."""
        return await self._execute_with_metrics(
            "get_adaptive_interface",
            self._get_adaptive_interface_internal,
            user_id
        )
    
    async def _get_adaptive_interface_internal(self, user_id: str) -> Dict[str, Any]:
        """Internal logic for getting adaptive interface."""
        if user_id not in self.user_profiles:
            await self._create_or_update_profile_internal(user_id, None)
        
        if user_id not in self.adaptive_interfaces:
            await self._create_adaptive_interface(self.user_profiles[user_id])
        
        interface = self.adaptive_interfaces[user_id]
        
        return {
            "user_id": interface.user_id,
            "current_complexity": interface.current_complexity.value,
            "visible_features": interface.visible_features,
            "contextual_hints": [asdict(hint) for hint in interface.contextual_hints],
            "simplified_commands": interface.simplified_commands,
            "guided_workflows": interface.guided_workflows,
            "recommendation": self._generate_next_step_recommendation(user_id)
        }
    
    def _generate_next_step_recommendation(self, user_id: str) -> Dict[str, Any]:
        """Generate next step recommendation for user."""
        profile = self.user_profiles[user_id]
        interface = self.adaptive_interfaces[user_id]
        
        # Find features user hasn't tried yet
        untried_features = [
            feature for feature in interface.visible_features
            if feature not in profile.features_discovered
        ]
        
        # Find features user could master next
        features_to_master = [
            feature for feature in profile.features_discovered
            if feature not in profile.features_mastered
            and profile.feature_usage_count.get(feature, 0) > 0
        ]
        
        if untried_features:
            recommended_feature = untried_features[0]
            feature_def = self.feature_definitions[recommended_feature]
            return {
                "type": "discover_feature",
                "feature": recommended_feature,
                "description": feature_def["description"],
                "reason": "Try this new feature to expand your capabilities"
            }
        
        elif features_to_master:
            recommended_feature = features_to_master[0]
            usage_count = profile.feature_usage_count[recommended_feature]
            needed = self.complexity_thresholds["feature_mastery"] - usage_count
            return {
                "type": "master_feature",
                "feature": recommended_feature,
                "current_usage": usage_count,
                "needed_for_mastery": needed,
                "reason": f"Use this feature {needed} more times to master it"
            }
        
        else:
            return {
                "type": "skill_progression",
                "current_level": profile.skill_level.value,
                "reason": "Keep practicing to unlock more advanced features"
            }
    
    async def _periodic_profile_updates(self):
        """Periodic task to update user profiles and detect patterns."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                for profile in self.user_profiles.values():
                    # Detect usage patterns
                    await self._analyze_usage_patterns(profile)
                    
                    # Update pain points
                    await self._detect_pain_points(profile)
                    
                    # Clean up old data
                    await self._cleanup_old_data(profile)
                
            except Exception as e:
                self.logger.error(f"Periodic profile update error: {e}")
    
    async def _analyze_usage_patterns(self, profile: UserProfile):
        """Analyze user behavior patterns."""
        # This would implement advanced pattern analysis
        # For now, basic session pattern tracking
        current_time = datetime.utcnow()
        
        # Track session frequency
        if profile.last_active:
            time_since_last = (current_time - profile.last_active).total_seconds()
            profile.session_patterns["avg_session_gap"] = (
                profile.session_patterns.get("avg_session_gap", 0) * 0.8 + 
                time_since_last * 0.2
            )
    
    async def _detect_pain_points(self, profile: UserProfile):
        """Detect user pain points from interaction patterns."""
        # High error rate indicates difficulty
        if profile.error_rate > 0.3:
            if "high_error_rate" not in profile.pain_points:
                profile.pain_points.append("high_error_rate")
        
        # Repeated failed attempts at same features
        for feature, count in profile.feature_usage_count.items():
            if feature not in profile.features_mastered and count > 5:
                pain_point = f"struggling_with_{feature}"
                if pain_point not in profile.pain_points:
                    profile.pain_points.append(pain_point)
    
    async def _cleanup_old_data(self, profile: UserProfile):
        """Clean up old data to prevent bloat."""
        # Keep only recent progression history
        if len(profile.skill_progression_history) > 10:
            profile.skill_progression_history = profile.skill_progression_history[-10:]
        
        # Keep only recent pain points
        if len(profile.pain_points) > 5:
            profile.pain_points = profile.pain_points[-5:]
    
    async def get_user_profile(self, user_id: str) -> APIResponse:
        """Get comprehensive user profile information."""
        return await self._execute_with_metrics(
            "get_user_profile",
            self._get_user_profile_internal,
            user_id
        )
    
    async def _get_user_profile_internal(self, user_id: str) -> Dict[str, Any]:
        """Internal logic for getting user profile."""
        if user_id not in self.user_profiles:
            raise APIError(f"User profile {user_id} not found", "PROFILE_NOT_FOUND", 404)
        
        profile = self.user_profiles[user_id]
        
        return {
            "user_id": profile.user_id,
            "skill_level": profile.skill_level.value,
            "created_at": profile.created_at.isoformat(),
            "last_active": profile.last_active.isoformat(),
            "statistics": {
                "total_sessions": profile.total_sessions,
                "total_commands": profile.total_commands,
                "successful_commands": profile.successful_commands,
                "success_rate": profile.successful_commands / profile.total_commands if profile.total_commands > 0 else 0.0,
                "error_rate": profile.error_rate
            },
            "features": {
                "discovered": list(profile.features_discovered),
                "mastered": list(profile.features_mastered),
                "usage_counts": profile.feature_usage_count
            },
            "learning_stages": {k: v.value for k, v in profile.learning_stages.items()},
            "progression_history": profile.skill_progression_history,
            "preferences": {
                "complexity": profile.preferred_complexity.value if profile.preferred_complexity else "auto",
                "help_frequency": profile.help_frequency_preference,
                "explanations": profile.explanation_preference
            },
            "pain_points": profile.pain_points
        }
    
    async def get_analytics(self) -> APIResponse:
        """Get analytics about progressive disclosure usage."""
        return await self._execute_with_metrics(
            "get_analytics",
            self._get_analytics_internal
        )
    
    async def _get_analytics_internal(self) -> Dict[str, Any]:
        """Internal logic for getting analytics."""
        total_users = len(self.user_profiles)
        
        if total_users == 0:
            return {
                "total_users": 0,
                "skill_distribution": {},
                "feature_adoption": {},
                "average_progression_time": 0
            }
        
        # Skill level distribution
        skill_distribution = {}
        for profile in self.user_profiles.values():
            level = profile.skill_level.value
            skill_distribution[level] = skill_distribution.get(level, 0) + 1
        
        # Feature adoption rates
        feature_adoption = {}
        for feature_name in self.feature_definitions.keys():
            adopters = sum(1 for p in self.user_profiles.values() if feature_name in p.features_discovered)
            masters = sum(1 for p in self.user_profiles.values() if feature_name in p.features_mastered)
            feature_adoption[feature_name] = {
                "adoption_rate": adopters / total_users,
                "mastery_rate": masters / total_users
            }
        
        return {
            "total_users": total_users,
            "skill_distribution": skill_distribution,
            "feature_adoption": feature_adoption,
            "interfaces_active": len(self.adaptive_interfaces),
            "hints_available": len(self.contextual_hints)
        }