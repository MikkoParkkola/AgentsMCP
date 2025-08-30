"""
Progressive Disclosure Manager - Adaptive UI that scales with user expertise.

This component implements intelligent UI adaptation that:
- Detects user skill level through interaction patterns
- Provides contextual help and guidance for beginners
- Offers advanced features and shortcuts for experts
- Smoothly transitions between complexity levels
- Maintains user preferences and learning progress
"""

from __future__ import annotations

import asyncio
import logging
import json
from typing import Optional, Dict, List, Any, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path

from ..v2.event_system import AsyncEventSystem, Event, EventType

logger = logging.getLogger(__name__)


class SkillLevel(Enum):
    """User skill levels for progressive disclosure."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"
    ADAPTIVE = "adaptive"  # Automatically adjusts based on behavior


class UIComplexity(Enum):
    """UI complexity levels."""
    MINIMAL = "minimal"      # Essential features only
    STANDARD = "standard"    # Common features visible
    ADVANCED = "advanced"    # All features available
    CUSTOM = "custom"        # User-customized layout


class InteractionPattern(Enum):
    """Types of user interaction patterns tracked."""
    COMMAND_USAGE = "command_usage"
    SHORTCUT_USAGE = "shortcut_usage"
    ERROR_FREQUENCY = "error_frequency"
    HELP_REQUESTS = "help_requests"
    TASK_COMPLETION_SPEED = "task_completion_speed"
    FEATURE_DISCOVERY = "feature_discovery"


@dataclass
class SkillMetric:
    """A metric used to assess user skill level."""
    name: str
    value: float
    weight: float
    trend: str  # "improving", "stable", "declining"
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class UIElement:
    """A UI element that can be shown/hidden based on skill level."""
    id: str
    name: str
    description: str
    category: str
    min_skill_level: SkillLevel
    complexity_level: UIComplexity
    help_text: str = ""
    tutorial_available: bool = False
    shortcuts: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # Other element IDs required
    usage_count: int = 0
    last_used: Optional[datetime] = None


@dataclass
class ContextualHelp:
    """Contextual help information for UI elements."""
    element_id: str
    title: str
    content: str
    tips: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    related_features: List[str] = field(default_factory=list)
    difficulty: SkillLevel = SkillLevel.BEGINNER


@dataclass
class UserProfile:
    """Complete user profile for progressive disclosure."""
    skill_level: SkillLevel = SkillLevel.BEGINNER
    ui_complexity: UIComplexity = UIComplexity.MINIMAL
    session_count: int = 0
    total_usage_time: timedelta = field(default_factory=lambda: timedelta())
    skill_metrics: Dict[str, SkillMetric] = field(default_factory=dict)
    preferred_features: Set[str] = field(default_factory=set)
    hidden_features: Set[str] = field(default_factory=set)
    onboarding_completed: bool = False
    last_skill_assessment: Optional[datetime] = None
    preferences: Dict[str, Any] = field(default_factory=dict)


class ProgressiveDisclosureManager:
    """
    Revolutionary progressive disclosure system that adapts UI complexity to user skill.
    
    Features:
    - Intelligent skill level detection through behavior analysis
    - Contextual help system with just-in-time guidance
    - Smooth transitions between complexity levels
    - Personalized feature recommendations
    - Adaptive onboarding and tutorials
    - Accessibility-focused design patterns
    """
    
    def __init__(self, event_system: AsyncEventSystem, config_path: Optional[Path] = None):
        """Initialize the progressive disclosure manager."""
        self.event_system = event_system
        self.config_path = config_path or Path.home() / ".agentsmcp" / "progressive_disclosure.json"
        
        # User profile and state
        self.user_profile = UserProfile()
        self.session_start_time = datetime.now()
        
        # UI element registry
        self.ui_elements: Dict[str, UIElement] = {}
        self.contextual_help: Dict[str, ContextualHelp] = {}
        
        # Skill assessment system
        self.interaction_history: List[Dict[str, Any]] = []
        self.assessment_triggers = {
            InteractionPattern.COMMAND_USAGE: 10,      # Assess after 10 commands
            InteractionPattern.SHORTCUT_USAGE: 5,     # Assess after 5 shortcuts
            InteractionPattern.ERROR_FREQUENCY: 3,    # Assess after 3 errors
            InteractionPattern.HELP_REQUESTS: 2,      # Assess after 2 help requests
        }
        
        # Active UI state
        self.visible_elements: Set[str] = set()
        self.active_help_requests: Dict[str, datetime] = {}
        
        # Performance tracking
        self.adaptation_times: List[float] = []
        self.user_satisfaction_scores: List[float] = []
        
        # Callbacks for UI integration
        self._callbacks: Dict[str, Callable] = {}
        
        # Initialize standard UI elements
        self._initialize_ui_elements()
        self._initialize_contextual_help()
    
    def _initialize_ui_elements(self):
        """Initialize the standard UI elements registry."""
        elements = [
            # Basic chat elements
            UIElement(
                id="chat_input",
                name="Chat Input",
                description="Basic text input for chatting with AI",
                category="chat",
                min_skill_level=SkillLevel.BEGINNER,
                complexity_level=UIComplexity.MINIMAL,
                help_text="Type your message and press Enter to send"
            ),
            UIElement(
                id="chat_history",
                name="Chat History",
                description="Conversation history display",
                category="chat",
                min_skill_level=SkillLevel.BEGINNER,
                complexity_level=UIComplexity.MINIMAL,
                help_text="Shows your conversation with the AI agent"
            ),
            UIElement(
                id="agent_selector",
                name="Agent Selector",
                description="Choose which AI agent to chat with",
                category="agents",
                min_skill_level=SkillLevel.INTERMEDIATE,
                complexity_level=UIComplexity.STANDARD,
                help_text="Switch between different AI agents like Claude, GPT, or Ollama",
                shortcuts=["/agent", "/a"]
            ),
            
            # Command system elements
            UIElement(
                id="command_palette",
                name="Command Palette",
                description="Quick access to all available commands",
                category="commands",
                min_skill_level=SkillLevel.INTERMEDIATE,
                complexity_level=UIComplexity.STANDARD,
                help_text="Press Ctrl+Shift+P to open command palette",
                shortcuts=["c-s-p"],
                tutorial_available=True
            ),
            UIElement(
                id="natural_language_commands",
                name="Natural Language Commands",
                description="Use natural language instead of specific commands",
                category="commands",
                min_skill_level=SkillLevel.BEGINNER,
                complexity_level=UIComplexity.MINIMAL,
                help_text="Just type naturally, like 'help me with Python' or 'show my files'"
            ),
            UIElement(
                id="advanced_shortcuts",
                name="Advanced Shortcuts",
                description="Power user keyboard shortcuts",
                category="shortcuts",
                min_skill_level=SkillLevel.EXPERT,
                complexity_level=UIComplexity.ADVANCED,
                help_text="Comprehensive keyboard shortcuts for maximum efficiency",
                dependencies=["command_palette"]
            ),
            
            # Status and monitoring elements
            UIElement(
                id="status_bar",
                name="Status Bar",
                description="Shows system status and current activity",
                category="status",
                min_skill_level=SkillLevel.BEGINNER,
                complexity_level=UIComplexity.MINIMAL,
                help_text="Displays current agent, connection status, and quick actions"
            ),
            UIElement(
                id="performance_metrics",
                name="Performance Metrics",
                description="Real-time system performance indicators",
                category="status",
                min_skill_level=SkillLevel.EXPERT,
                complexity_level=UIComplexity.ADVANCED,
                help_text="Shows response times, token usage, and system health"
            ),
            UIElement(
                id="debug_panel",
                name="Debug Panel",
                description="Advanced debugging and inspection tools",
                category="debug",
                min_skill_level=SkillLevel.EXPERT,
                complexity_level=UIComplexity.ADVANCED,
                help_text="Inspect requests, responses, and system internals",
                shortcuts=["f12", "c-s-d"]
            ),
            
            # Configuration elements
            UIElement(
                id="quick_settings",
                name="Quick Settings",
                description="Essential settings toggle",
                category="settings",
                min_skill_level=SkillLevel.INTERMEDIATE,
                complexity_level=UIComplexity.STANDARD,
                help_text="Toggle dark mode, notifications, and other common settings"
            ),
            UIElement(
                id="advanced_config",
                name="Advanced Configuration",
                description="Comprehensive system configuration",
                category="settings",
                min_skill_level=SkillLevel.EXPERT,
                complexity_level=UIComplexity.ADVANCED,
                help_text="Fine-tune all system parameters and behaviors",
                dependencies=["quick_settings"]
            ),
            
            # Help and learning elements
            UIElement(
                id="contextual_tips",
                name="Contextual Tips",
                description="Just-in-time help and tips",
                category="help",
                min_skill_level=SkillLevel.BEGINNER,
                complexity_level=UIComplexity.MINIMAL,
                help_text="Helpful tips appear when you need them"
            ),
            UIElement(
                id="tutorial_system",
                name="Interactive Tutorials",
                description="Step-by-step guided tutorials",
                category="help",
                min_skill_level=SkillLevel.BEGINNER,
                complexity_level=UIComplexity.MINIMAL,
                help_text="Learn features through interactive walkthroughs",
                tutorial_available=True
            ),
            UIElement(
                id="advanced_help",
                name="Advanced Documentation",
                description="Comprehensive documentation and examples",
                category="help",
                min_skill_level=SkillLevel.EXPERT,
                complexity_level=UIComplexity.ADVANCED,
                help_text="In-depth documentation and advanced usage examples"
            )
        ]
        
        self.ui_elements = {element.id: element for element in elements}
    
    def _initialize_contextual_help(self):
        """Initialize contextual help content."""
        help_items = [
            ContextualHelp(
                element_id="chat_input",
                title="Getting Started with Chat",
                content="This is where you type messages to chat with AI agents. Just type naturally!",
                tips=[
                    "Press Enter to send your message",
                    "Use Shift+Enter for multi-line messages",
                    "Start with '/' for commands (like /help)"
                ],
                examples=[
                    "Hello, how can you help me today?",
                    "Can you explain quantum computing?",
                    "/help - show available commands"
                ]
            ),
            ContextualHelp(
                element_id="agent_selector",
                title="Choosing AI Agents",
                content="Different AI agents have different strengths. Choose based on your task.",
                tips=[
                    "Claude: Great for detailed analysis and writing",
                    "GPT: Excellent for general conversation and coding",
                    "Ollama: Privacy-focused local AI models"
                ],
                examples=[
                    "/agent claude - switch to Claude",
                    "/agent gpt-4 - use GPT-4 model",
                    "/agent list - see available agents"
                ],
                related_features=["command_palette", "natural_language_commands"]
            ),
            ContextualHelp(
                element_id="command_palette",
                title="Command Palette Power",
                content="The command palette gives you quick access to any feature in the system.",
                tips=[
                    "Press Ctrl+Shift+P to open",
                    "Start typing to search commands",
                    "Use arrow keys to navigate",
                    "Press Enter to execute"
                ],
                examples=[
                    "Type 'theme' to change appearance",
                    "Type 'agent' to switch AI models",
                    "Type 'export' to save conversations"
                ],
                difficulty=SkillLevel.INTERMEDIATE
            ),
            ContextualHelp(
                element_id="natural_language_commands",
                title="Natural Language is Powerful",
                content="You don't need to memorize commands. Just describe what you want to do!",
                tips=[
                    "Describe your goal in plain English",
                    "The system will suggest appropriate actions",
                    "You can always review before executing"
                ],
                examples=[
                    "Change the theme to dark mode",
                    "Show me my conversation history",
                    "Help me write a Python function",
                    "Switch to Claude and ask about machine learning"
                ]
            )
        ]
        
        self.contextual_help = {help_item.element_id: help_item for help_item in help_items}
    
    async def initialize(self) -> bool:
        """Initialize the progressive disclosure manager."""
        try:
            # Load user profile from disk
            await self._load_user_profile()
            
            # Update UI based on current profile
            await self._update_ui_visibility()
            
            # Set up event listeners
            await self._setup_event_listeners()
            
            logger.info(f"Progressive disclosure initialized for {self.user_profile.skill_level.value} user")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize progressive disclosure: {e}")
            return False
    
    async def _load_user_profile(self):
        """Load user profile from persistent storage."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                
                # Reconstruct user profile
                profile_data = data.get('user_profile', {})
                self.user_profile.skill_level = SkillLevel(profile_data.get('skill_level', 'beginner'))
                self.user_profile.ui_complexity = UIComplexity(profile_data.get('ui_complexity', 'minimal'))
                self.user_profile.session_count = profile_data.get('session_count', 0)
                self.user_profile.total_usage_time = timedelta(seconds=profile_data.get('total_usage_time_seconds', 0))
                self.user_profile.preferred_features = set(profile_data.get('preferred_features', []))
                self.user_profile.hidden_features = set(profile_data.get('hidden_features', []))
                self.user_profile.onboarding_completed = profile_data.get('onboarding_completed', False)
                self.user_profile.preferences = profile_data.get('preferences', {})
                
                # Reconstruct skill metrics
                metrics_data = profile_data.get('skill_metrics', {})
                for metric_name, metric_data in metrics_data.items():
                    self.user_profile.skill_metrics[metric_name] = SkillMetric(
                        name=metric_data['name'],
                        value=metric_data['value'],
                        weight=metric_data['weight'],
                        trend=metric_data['trend'],
                        last_updated=datetime.fromisoformat(metric_data['last_updated'])
                    )
                
                logger.info("User profile loaded from disk")
            else:
                # First time user - start onboarding
                logger.info("New user detected, starting with beginner profile")
                
        except Exception as e:
            logger.error(f"Error loading user profile: {e}")
            # Continue with default profile
    
    async def _save_user_profile(self):
        """Save user profile to persistent storage."""
        try:
            # Ensure config directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Serialize user profile
            profile_data = {
                'skill_level': self.user_profile.skill_level.value,
                'ui_complexity': self.user_profile.ui_complexity.value,
                'session_count': self.user_profile.session_count,
                'total_usage_time_seconds': self.user_profile.total_usage_time.total_seconds(),
                'preferred_features': list(self.user_profile.preferred_features),
                'hidden_features': list(self.user_profile.hidden_features),
                'onboarding_completed': self.user_profile.onboarding_completed,
                'preferences': self.user_profile.preferences,
                'skill_metrics': {}
            }
            
            # Serialize skill metrics
            for metric_name, metric in self.user_profile.skill_metrics.items():
                profile_data['skill_metrics'][metric_name] = {
                    'name': metric.name,
                    'value': metric.value,
                    'weight': metric.weight,
                    'trend': metric.trend,
                    'last_updated': metric.last_updated.isoformat()
                }
            
            data = {
                'user_profile': profile_data,
                'last_saved': datetime.now().isoformat()
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving user profile: {e}")
    
    async def _setup_event_listeners(self):
        """Set up event listeners for interaction tracking."""
        await self.event_system.subscribe(EventType.KEYBOARD, self._handle_interaction_event)
        await self.event_system.subscribe(EventType.CUSTOM, self._handle_custom_event)
    
    async def _handle_interaction_event(self, event: Event):
        """Handle interaction events for skill assessment."""
        try:
            event_data = event.data
            interaction_type = event_data.get('type')
            
            # Track different types of interactions
            interaction_record = {
                'timestamp': datetime.now().isoformat(),
                'type': interaction_type,
                'data': event_data
            }
            
            self.interaction_history.append(interaction_record)
            
            # Trigger skill assessment if threshold reached
            if interaction_type in [InteractionPattern.COMMAND_USAGE.value, 
                                  InteractionPattern.SHORTCUT_USAGE.value,
                                  InteractionPattern.ERROR_FREQUENCY.value]:
                await self._maybe_assess_skill_level()
                
            # Update element usage
            element_id = event_data.get('element_id')
            if element_id and element_id in self.ui_elements:
                element = self.ui_elements[element_id]
                element.usage_count += 1
                element.last_used = datetime.now()
                
        except Exception as e:
            logger.error(f"Error handling interaction event: {e}")
    
    async def _handle_custom_event(self, event: Event):
        """Handle custom events from other components."""
        try:
            event_data = event.data
            component = event_data.get('component')
            action = event_data.get('action')
            
            if component == "enhanced_command_interface" and action == "interpretation":
                # Track command interpretation success
                result_data = event_data.get('result', {})
                confidence = result_data.get('confidence')
                processing_time = result_data.get('processing_time_ms', 0)
                
                # Record skill metrics
                await self._update_skill_metric(
                    'command_interpretation_confidence',
                    1.0 if confidence in ['high', 'medium'] else 0.0,
                    0.3
                )
                
                await self._update_skill_metric(
                    'task_completion_speed',
                    1.0 if processing_time < 100 else 0.5 if processing_time < 500 else 0.0,
                    0.2
                )
            
        except Exception as e:
            logger.error(f"Error handling custom event: {e}")
    
    async def _maybe_assess_skill_level(self):
        """Assess skill level if conditions are met."""
        # Count recent interactions by type
        recent_interactions = [
            i for i in self.interaction_history[-50:]  # Last 50 interactions
            if datetime.now() - datetime.fromisoformat(i['timestamp']) < timedelta(minutes=30)
        ]
        
        interaction_counts = {}
        for interaction in recent_interactions:
            itype = interaction['type']
            interaction_counts[itype] = interaction_counts.get(itype, 0) + 1
        
        # Check if we should assess
        should_assess = False
        for pattern, threshold in self.assessment_triggers.items():
            if interaction_counts.get(pattern.value, 0) >= threshold:
                should_assess = True
                break
        
        if should_assess:
            await self._assess_skill_level()
    
    async def _assess_skill_level(self):
        """Perform comprehensive skill level assessment."""
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Calculate skill scores
            command_proficiency = self._calculate_command_proficiency()
            error_rate = self._calculate_error_rate()
            feature_usage_breadth = self._calculate_feature_usage_breadth()
            efficiency_score = self._calculate_efficiency_score()
            
            # Update skill metrics
            await self._update_skill_metric('command_proficiency', command_proficiency, 0.4)
            await self._update_skill_metric('error_rate', 1.0 - error_rate, 0.3)  # Invert error rate
            await self._update_skill_metric('feature_usage_breadth', feature_usage_breadth, 0.2)
            await self._update_skill_metric('efficiency_score', efficiency_score, 0.1)
            
            # Calculate overall skill score
            overall_score = sum(
                metric.value * metric.weight
                for metric in self.user_profile.skill_metrics.values()
            )
            
            # Determine skill level
            previous_skill = self.user_profile.skill_level
            if overall_score >= 0.8:
                new_skill_level = SkillLevel.EXPERT
            elif overall_score >= 0.6:
                new_skill_level = SkillLevel.INTERMEDIATE
            else:
                new_skill_level = SkillLevel.BEGINNER
            
            # Update if changed
            if new_skill_level != previous_skill:
                self.user_profile.skill_level = new_skill_level
                await self._handle_skill_level_change(previous_skill, new_skill_level)
            
            # Update assessment timestamp
            self.user_profile.last_skill_assessment = datetime.now()
            
            # Track performance
            assessment_time = asyncio.get_event_loop().time() - start_time
            self.adaptation_times.append(assessment_time)
            
            # Save updated profile
            await self._save_user_profile()
            
            logger.info(f"Skill assessment completed: {new_skill_level.value} (score: {overall_score:.2f})")
            
        except Exception as e:
            logger.error(f"Error during skill assessment: {e}")
    
    def _calculate_command_proficiency(self) -> float:
        """Calculate user's command proficiency score."""
        if not self.interaction_history:
            return 0.0
        
        # Recent command usage patterns
        recent_commands = [
            i for i in self.interaction_history[-100:]
            if i['type'] == InteractionPattern.COMMAND_USAGE.value
        ]
        
        if not recent_commands:
            return 0.0
        
        # Count unique commands used
        unique_commands = set()
        shortcut_usage = 0
        
        for cmd in recent_commands:
            command = cmd['data'].get('command', '')
            if command:
                unique_commands.add(command)
                if cmd['data'].get('used_shortcut', False):
                    shortcut_usage += 1
        
        # Score based on command diversity and shortcut usage
        diversity_score = min(len(unique_commands) / 10.0, 1.0)  # Up to 10 unique commands = 1.0
        shortcut_score = min(shortcut_usage / len(recent_commands), 1.0)
        
        return (diversity_score * 0.7) + (shortcut_score * 0.3)
    
    def _calculate_error_rate(self) -> float:
        """Calculate user's error rate."""
        if not self.interaction_history:
            return 0.0
        
        recent_interactions = self.interaction_history[-50:]
        error_interactions = [
            i for i in recent_interactions
            if i['type'] == InteractionPattern.ERROR_FREQUENCY.value
        ]
        
        return len(error_interactions) / len(recent_interactions) if recent_interactions else 0.0
    
    def _calculate_feature_usage_breadth(self) -> float:
        """Calculate breadth of feature usage."""
        used_elements = set()
        for element in self.ui_elements.values():
            if element.usage_count > 0:
                used_elements.add(element.id)
        
        total_available = len([e for e in self.ui_elements.values() 
                             if e.min_skill_level in [SkillLevel.BEGINNER, SkillLevel.INTERMEDIATE]])
        
        return len(used_elements) / total_available if total_available > 0 else 0.0
    
    def _calculate_efficiency_score(self) -> float:
        """Calculate user's efficiency score based on task completion times."""
        # This would be based on timing data from completed tasks
        # For now, use a placeholder based on recent activity
        recent_activity = len([
            i for i in self.interaction_history[-20:]
            if datetime.now() - datetime.fromisoformat(i['timestamp']) < timedelta(minutes=5)
        ])
        
        # Higher recent activity suggests efficiency
        return min(recent_activity / 10.0, 1.0)
    
    async def _update_skill_metric(self, metric_name: str, value: float, weight: float):
        """Update a skill metric with trend analysis."""
        if metric_name in self.user_profile.skill_metrics:
            metric = self.user_profile.skill_metrics[metric_name]
            old_value = metric.value
            
            # Simple trend detection
            if value > old_value * 1.1:
                trend = "improving"
            elif value < old_value * 0.9:
                trend = "declining"
            else:
                trend = "stable"
            
            metric.value = (metric.value * 0.7) + (value * 0.3)  # Weighted average
            metric.trend = trend
            metric.last_updated = datetime.now()
        else:
            # Create new metric
            self.user_profile.skill_metrics[metric_name] = SkillMetric(
                name=metric_name,
                value=value,
                weight=weight,
                trend="stable"
            )
    
    async def _handle_skill_level_change(self, old_level: SkillLevel, new_level: SkillLevel):
        """Handle when user's skill level changes."""
        logger.info(f"User skill level changed: {old_level.value} -> {new_level.value}")
        
        # Update UI complexity if appropriate
        if new_level == SkillLevel.EXPERT and self.user_profile.ui_complexity != UIComplexity.ADVANCED:
            await self.set_ui_complexity(UIComplexity.ADVANCED)
        elif new_level == SkillLevel.INTERMEDIATE and self.user_profile.ui_complexity == UIComplexity.MINIMAL:
            await self.set_ui_complexity(UIComplexity.STANDARD)
        
        # Emit skill level change event
        event = Event(
            event_type=EventType.CUSTOM,
            data={
                "component": "progressive_disclosure",
                "action": "skill_level_changed",
                "old_level": old_level.value,
                "new_level": new_level.value,
                "auto_adapted": True
            }
        )
        await self.event_system.emit_event(event)
        
        # Notify UI components
        if "skill_changed" in self._callbacks:
            await self._callbacks["skill_changed"](old_level, new_level)
    
    async def _update_ui_visibility(self):
        """Update UI element visibility based on current settings."""
        self.visible_elements.clear()
        
        # Determine which elements should be visible
        for element_id, element in self.ui_elements.items():
            should_show = self._should_show_element(element)
            
            if should_show:
                self.visible_elements.add(element_id)
        
        # Emit UI update event
        event = Event(
            event_type=EventType.CUSTOM,
            data={
                "component": "progressive_disclosure",
                "action": "ui_visibility_updated",
                "visible_elements": list(self.visible_elements),
                "skill_level": self.user_profile.skill_level.value,
                "ui_complexity": self.user_profile.ui_complexity.value
            }
        )
        await self.event_system.emit_event(event)
        
        # Notify callbacks
        if "visibility_changed" in self._callbacks:
            await self._callbacks["visibility_changed"](self.visible_elements)
    
    def _should_show_element(self, element: UIElement) -> bool:
        """Determine if a UI element should be shown."""
        # Check if explicitly hidden
        if element.id in self.user_profile.hidden_features:
            return False
        
        # Check if explicitly preferred (always show)
        if element.id in self.user_profile.preferred_features:
            return True
        
        # Check skill level requirement
        skill_levels = [SkillLevel.BEGINNER, SkillLevel.INTERMEDIATE, SkillLevel.EXPERT]
        user_skill_index = skill_levels.index(self.user_profile.skill_level)
        required_skill_index = skill_levels.index(element.min_skill_level)
        
        if user_skill_index < required_skill_index:
            return False
        
        # Check UI complexity requirement
        complexity_levels = [UIComplexity.MINIMAL, UIComplexity.STANDARD, UIComplexity.ADVANCED]
        user_complexity_index = complexity_levels.index(self.user_profile.ui_complexity)
        required_complexity_index = complexity_levels.index(element.complexity_level)
        
        if user_complexity_index < required_complexity_index:
            return False
        
        # Check dependencies
        for dep_id in element.dependencies:
            if dep_id not in self.visible_elements and dep_id not in self.user_profile.preferred_features:
                return False
        
        return True
    
    async def set_skill_level(self, skill_level: SkillLevel):
        """Manually set user's skill level."""
        old_level = self.user_profile.skill_level
        self.user_profile.skill_level = skill_level
        
        await self._update_ui_visibility()
        await self._save_user_profile()
        
        # Emit event
        event = Event(
            event_type=EventType.CUSTOM,
            data={
                "component": "progressive_disclosure",
                "action": "skill_level_changed",
                "old_level": old_level.value,
                "new_level": skill_level.value,
                "auto_adapted": False
            }
        )
        await self.event_system.emit_event(event)
    
    async def set_ui_complexity(self, complexity: UIComplexity):
        """Set UI complexity level."""
        old_complexity = self.user_profile.ui_complexity
        self.user_profile.ui_complexity = complexity
        
        await self._update_ui_visibility()
        await self._save_user_profile()
        
        # Emit event
        event = Event(
            event_type=EventType.CUSTOM,
            data={
                "component": "progressive_disclosure",
                "action": "ui_complexity_changed",
                "old_complexity": old_complexity.value,
                "new_complexity": complexity.value
            }
        )
        await self.event_system.emit_event(event)
    
    async def toggle_feature(self, element_id: str, force_state: Optional[bool] = None):
        """Toggle visibility of a specific feature."""
        if element_id not in self.ui_elements:
            logger.warning(f"Unknown UI element: {element_id}")
            return
        
        if force_state is None:
            # Toggle current state
            if element_id in self.user_profile.hidden_features:
                self.user_profile.hidden_features.remove(element_id)
                self.user_profile.preferred_features.add(element_id)
            else:
                self.user_profile.preferred_features.discard(element_id)
                self.user_profile.hidden_features.add(element_id)
        else:
            # Set specific state
            if force_state:
                self.user_profile.hidden_features.discard(element_id)
                self.user_profile.preferred_features.add(element_id)
            else:
                self.user_profile.preferred_features.discard(element_id)
                self.user_profile.hidden_features.add(element_id)
        
        await self._update_ui_visibility()
        await self._save_user_profile()
    
    async def get_contextual_help(self, element_id: str) -> Optional[ContextualHelp]:
        """Get contextual help for a UI element."""
        if element_id in self.contextual_help:
            help_content = self.contextual_help[element_id]
            
            # Track help request
            self.active_help_requests[element_id] = datetime.now()
            
            # Record interaction
            interaction_record = {
                'timestamp': datetime.now().isoformat(),
                'type': InteractionPattern.HELP_REQUESTS.value,
                'data': {'element_id': element_id}
            }
            self.interaction_history.append(interaction_record)
            
            return help_content
        
        return None
    
    async def start_onboarding(self) -> List[str]:
        """Start the onboarding process for new users."""
        if self.user_profile.onboarding_completed:
            return []  # Already completed
        
        # Return ordered list of elements to introduce
        onboarding_flow = [
            "chat_input",
            "chat_history",
            "status_bar",
            "natural_language_commands",
            "contextual_tips"
        ]
        
        # Filter based on current visibility
        available_flow = [
            element_id for element_id in onboarding_flow
            if element_id in self.visible_elements
        ]
        
        return available_flow
    
    async def complete_onboarding(self):
        """Mark onboarding as completed."""
        self.user_profile.onboarding_completed = True
        await self._save_user_profile()
        
        # Emit completion event
        event = Event(
            event_type=EventType.CUSTOM,
            data={
                "component": "progressive_disclosure",
                "action": "onboarding_completed"
            }
        )
        await self.event_system.emit_event(event)
    
    def get_visible_elements(self) -> Set[str]:
        """Get currently visible UI elements."""
        return self.visible_elements.copy()
    
    def get_user_profile(self) -> UserProfile:
        """Get current user profile."""
        return self.user_profile
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            "skill_level": self.user_profile.skill_level.value,
            "ui_complexity": self.user_profile.ui_complexity.value,
            "session_count": self.user_profile.session_count,
            "total_usage_time": str(self.user_profile.total_usage_time),
            "visible_elements_count": len(self.visible_elements),
            "skill_metrics": {
                name: {
                    "value": metric.value,
                    "trend": metric.trend,
                    "last_updated": metric.last_updated.isoformat()
                }
                for name, metric in self.user_profile.skill_metrics.items()
            },
            "adaptation_performance": {
                "average_adaptation_time_ms": sum(self.adaptation_times) * 1000 / len(self.adaptation_times) if self.adaptation_times else 0,
                "total_adaptations": len(self.adaptation_times)
            },
            "onboarding_completed": self.user_profile.onboarding_completed
        }
    
    def add_callback(self, event_type: str, callback: Callable):
        """Add callback for specific event types."""
        self._callbacks[event_type] = callback
    
    def remove_callback(self, event_type: str):
        """Remove callback for specific event types."""
        if event_type in self._callbacks:
            del self._callbacks[event_type]
    
    async def cleanup(self):
        """Cleanup the progressive disclosure manager."""
        # Update session statistics
        session_duration = datetime.now() - self.session_start_time
        self.user_profile.session_count += 1
        self.user_profile.total_usage_time += session_duration
        
        # Save final state
        await self._save_user_profile()
        
        # Clear callbacks and caches
        self._callbacks.clear()
        self.interaction_history.clear()
        self.active_help_requests.clear()
        
        logger.info("Progressive disclosure manager cleaned up")


# Utility function for easy instantiation
def create_progressive_disclosure_manager(event_system: AsyncEventSystem, 
                                       config_path: Optional[Path] = None) -> ProgressiveDisclosureManager:
    """Create and return a new ProgressiveDisclosureManager instance."""
    return ProgressiveDisclosureManager(event_system, config_path)