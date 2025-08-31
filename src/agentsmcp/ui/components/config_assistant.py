"""
Configuration Assistant - AI-powered configuration recommendations and guidance.

This component provides:
- AI-powered configuration recommendations
- Context-aware help and guidance
- Configuration templates and presets
- Smart defaults based on usage patterns
- Interactive configuration wizard
- Conflict detection and resolution suggestions
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

from ..v2.event_system import AsyncEventSystem, Event, EventType
from ..v2.display_renderer import DisplayRenderer
from ..v2.terminal_manager import TerminalManager

logger = logging.getLogger(__name__)


class RecommendationType(Enum):
    """Type of configuration recommendation."""
    OPTIMIZATION = "optimization"
    SECURITY = "security"
    COMPATIBILITY = "compatibility"
    BEST_PRACTICE = "best_practice"
    PERFORMANCE = "performance"
    TROUBLESHOOTING = "troubleshooting"


class RecommendationSeverity(Enum):
    """Severity level of recommendation."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class WizardStep(Enum):
    """Configuration wizard steps."""
    WELCOME = "welcome"
    USE_CASE = "use_case"
    ENVIRONMENT = "environment"
    PROVIDERS = "providers"
    PERFORMANCE = "performance"
    SECURITY = "security"
    REVIEW = "review"
    COMPLETE = "complete"


@dataclass
class ConfigRecommendation:
    """A configuration recommendation from the AI assistant."""
    id: str
    title: str
    description: str
    recommendation_type: RecommendationType
    severity: RecommendationSeverity
    
    # Configuration changes
    suggested_changes: Dict[str, Any]
    current_values: Dict[str, Any]
    
    # Rationale and impact
    rationale: str
    expected_impact: str
    potential_risks: List[str] = field(default_factory=list)
    
    # Implementation
    auto_applicable: bool = False
    requires_restart: bool = False
    confidence_score: float = 0.0  # 0.0-1.0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    applied: bool = False
    dismissed: bool = False


@dataclass
class UseCaseProfile:
    """User's use case profile for tailored recommendations."""
    name: str
    description: str
    typical_configs: Dict[str, Any]
    recommended_providers: List[str]
    performance_profile: Dict[str, Any]
    security_requirements: List[str]


@dataclass
class ConfigTemplate:
    """Pre-defined configuration template."""
    id: str
    name: str
    description: str
    use_case: str
    configuration: Dict[str, Any]
    pros: List[str]
    cons: List[str]
    requirements: List[str] = field(default_factory=list)


class ConfigurationAssistant:
    """
    AI-powered configuration assistant with intelligent recommendations.
    
    Features:
    - Analyzes current configuration and suggests improvements
    - Provides context-aware help and explanations
    - Offers configuration templates for common use cases
    - Detects conflicts and compatibility issues
    - Interactive wizard for new users
    - Learning from user preferences and patterns
    """
    
    def __init__(self,
                 event_system: AsyncEventSystem,
                 display_renderer: DisplayRenderer,
                 terminal_manager: TerminalManager):
        """Initialize the configuration assistant."""
        self.event_system = event_system
        self.display_renderer = display_renderer
        self.terminal_manager = terminal_manager
        
        # Assistant state
        self.visible = False
        self.current_view = "recommendations"  # "recommendations", "wizard", "templates", "help"
        self.selected_index = 0
        
        # Recommendations
        self.recommendations: List[ConfigRecommendation] = []
        self.dismissed_recommendations: List[str] = []
        
        # Configuration wizard
        self.wizard_active = False
        self.current_wizard_step = WizardStep.WELCOME
        self.wizard_data: Dict[str, Any] = {}
        
        # Templates and profiles
        self.templates: List[ConfigTemplate] = []
        self.use_case_profiles: List[UseCaseProfile] = []
        
        # AI integration (placeholder for future integration)
        self.ai_enabled = False
        
        # Current configuration context
        self.current_config: Dict[str, Any] = {}
        self.config_history: List[Dict[str, Any]] = []
        
        self._initialize_templates()
        self._initialize_use_case_profiles()
    
    async def initialize(self) -> bool:
        """Initialize the configuration assistant."""
        try:
            # Register event handlers
            await self._register_event_handlers()
            
            # Load user preferences and history
            await self._load_user_preferences()
            
            # Generate initial recommendations
            await self._generate_recommendations()
            
            logger.info("Configuration assistant initialized successfully")
            return True
            
        except Exception as e:
            logger.exception(f"Failed to initialize configuration assistant: {e}")
            return False
    
    def _initialize_templates(self):
        """Initialize configuration templates for common use cases."""
        self.templates = [
            ConfigTemplate(
                id="beginner_friendly",
                name="Beginner Friendly Setup",
                description="Easy-to-use configuration for new users",
                use_case="General purpose with simple setup",
                configuration={
                    "default_provider": "ollama-turbo",
                    "max_concurrent_agents": 2,
                    "log_level": "INFO",
                    "auto_save": True,
                    "enable_suggestions": True,
                    "theme": "auto"
                },
                pros=[
                    "Simple and straightforward",
                    "Fast setup with good defaults",
                    "Helpful guidance enabled"
                ],
                cons=[
                    "Limited customization",
                    "May not be optimal for advanced use cases"
                ]
            ),
            ConfigTemplate(
                id="developer_power_user",
                name="Developer Power User",
                description="Advanced configuration for experienced developers",
                use_case="Intensive coding and development work",
                configuration={
                    "default_provider": "codex",
                    "max_concurrent_agents": 5,
                    "log_level": "DEBUG",
                    "enable_performance_monitoring": True,
                    "context_window": 16384,
                    "temperature": 0.3,
                    "auto_save_interval": 120
                },
                pros=[
                    "Optimized for coding tasks",
                    "High performance configuration",
                    "Advanced debugging enabled"
                ],
                cons=[
                    "More complex setup",
                    "Higher resource usage",
                    "Requires experience to tune"
                ],
                requirements=[
                    "Codex API access",
                    "Minimum 4GB RAM",
                    "Fast internet connection"
                ]
            ),
            ConfigTemplate(
                id="cost_optimized",
                name="Cost-Optimized Setup",
                description="Minimal resource usage with local providers",
                use_case="Budget-conscious or resource-constrained environments",
                configuration={
                    "default_provider": "ollama",
                    "max_concurrent_agents": 1,
                    "log_level": "WARN",
                    "enable_telemetry": False,
                    "max_memory_mb": 512,
                    "context_window": 4096
                },
                pros=[
                    "Low resource usage",
                    "No external API costs",
                    "Works offline"
                ],
                cons=[
                    "Limited capabilities",
                    "Slower response times",
                    "May not handle complex tasks well"
                ]
            ),
            ConfigTemplate(
                id="high_security",
                name="High Security Configuration",
                description="Security-focused setup with privacy protection",
                use_case="Sensitive work requiring maximum privacy",
                configuration={
                    "default_provider": "ollama",  # Local only
                    "encrypt_local_data": True,
                    "require_auth": True,
                    "session_timeout_minutes": 15,
                    "disable_telemetry": True,
                    "log_level": "ERROR",
                    "enable_audit_log": True
                },
                pros=[
                    "Maximum privacy protection",
                    "No data sent to external services",
                    "Comprehensive security measures"
                ],
                cons=[
                    "Complex setup and maintenance",
                    "Limited to local AI capabilities",
                    "More security overhead"
                ],
                requirements=[
                    "Local Ollama installation",
                    "Encrypted storage support",
                    "Regular security updates"
                ]
            )
        ]
    
    def _initialize_use_case_profiles(self):
        """Initialize use case profiles for personalized recommendations."""
        self.use_case_profiles = [
            UseCaseProfile(
                name="Software Development",
                description="Code generation, review, and debugging assistance",
                typical_configs={
                    "default_provider": "codex",
                    "temperature": 0.3,
                    "max_tokens": 8192,
                    "context_window": 16384
                },
                recommended_providers=["codex", "claude", "ollama-turbo"],
                performance_profile={
                    "priority": "accuracy",
                    "response_time": "medium",
                    "resource_usage": "high"
                },
                security_requirements=["code_privacy", "api_key_security"]
            ),
            UseCaseProfile(
                name="Research & Analysis",
                description="Large document analysis and research assistance",
                typical_configs={
                    "default_provider": "claude",
                    "temperature": 0.4,
                    "context_window": 100000,
                    "max_tokens": 4096
                },
                recommended_providers=["claude", "codex"],
                performance_profile={
                    "priority": "context_size",
                    "response_time": "slow",
                    "resource_usage": "high"
                },
                security_requirements=["data_privacy", "secure_storage"]
            ),
            UseCaseProfile(
                name="Quick Assistance",
                description="Fast responses for general questions and tasks",
                typical_configs={
                    "default_provider": "ollama-turbo",
                    "temperature": 0.7,
                    "max_tokens": 2048,
                    "streaming": True
                },
                recommended_providers=["ollama-turbo", "ollama"],
                performance_profile={
                    "priority": "speed",
                    "response_time": "fast",
                    "resource_usage": "low"
                },
                security_requirements=["basic_privacy"]
            )
        ]
    
    async def show(self, view: str = "recommendations"):
        """Show the configuration assistant."""
        if self.visible:
            return
        
        self.visible = True
        self.current_view = view
        self.selected_index = 0
        
        # Generate fresh recommendations
        await self._generate_recommendations()
        
        # Render the interface
        await self._render_interface()
        
        # Emit assistant shown event
        await self.event_system.emit(Event(
            event_type=EventType.UI,
            data={"action": "config_assistant_shown", "view": view}
        ))
    
    async def hide(self):
        """Hide the configuration assistant."""
        if not self.visible:
            return
        
        self.visible = False
        await self._clear_interface()
        
        # Emit assistant hidden event
        await self.event_system.emit(Event(
            event_type=EventType.UI,
            data={"action": "config_assistant_hidden"}
        ))
    
    async def set_current_config(self, config: Dict[str, Any]):
        """Set the current configuration context for analysis."""
        self.current_config = config
        
        # Add to history
        config_snapshot = {
            "config": config.copy(),
            "timestamp": datetime.now()
        }
        self.config_history.append(config_snapshot)
        
        # Keep only recent history
        if len(self.config_history) > 50:
            self.config_history = self.config_history[-50:]
        
        # Regenerate recommendations with new context
        await self._generate_recommendations()
    
    # Recommendation Management
    
    async def _generate_recommendations(self):
        """Generate configuration recommendations based on current context."""
        recommendations = []
        
        # Clear old recommendations
        self.recommendations = []
        
        try:
            # Analyze current configuration
            if self.current_config:
                # Performance optimizations
                perf_recs = await self._generate_performance_recommendations()
                recommendations.extend(perf_recs)
                
                # Security recommendations  
                security_recs = await self._generate_security_recommendations()
                recommendations.extend(security_recs)
                
                # Compatibility checks
                compat_recs = await self._generate_compatibility_recommendations()
                recommendations.extend(compat_recs)
                
                # Best practice suggestions
                best_practice_recs = await self._generate_best_practice_recommendations()
                recommendations.extend(best_practice_recs)
            
            # Sort by severity and confidence
            recommendations.sort(key=lambda r: (r.severity.value, -r.confidence_score))
            
            # Limit number of recommendations to avoid overwhelming user
            self.recommendations = recommendations[:10]
            
            logger.info(f"Generated {len(self.recommendations)} recommendations")
            
        except Exception as e:
            logger.exception(f"Error generating recommendations: {e}")
    
    async def _generate_performance_recommendations(self) -> List[ConfigRecommendation]:
        """Generate performance-related recommendations."""
        recommendations = []
        
        # Check memory configuration
        max_memory = self.current_config.get("max_memory_mb", 1024)
        if max_memory < 512:
            recommendations.append(ConfigRecommendation(
                id="low_memory_warning",
                title="Increase Memory Allocation",
                description="Current memory allocation may be too low for optimal performance",
                recommendation_type=RecommendationType.PERFORMANCE,
                severity=RecommendationSeverity.MEDIUM,
                suggested_changes={"max_memory_mb": 1024},
                current_values={"max_memory_mb": max_memory},
                rationale="Low memory allocation can cause performance issues and failures with large tasks",
                expected_impact="Improved stability and performance for memory-intensive operations",
                auto_applicable=True,
                confidence_score=0.8
            ))
        
        # Check concurrent agents
        max_agents = self.current_config.get("max_concurrent_agents", 1)
        if max_agents == 1 and max_memory >= 2048:
            recommendations.append(ConfigRecommendation(
                id="increase_concurrent_agents",
                title="Enable Multiple Concurrent Agents",
                description="Your system can handle more than one agent simultaneously",
                recommendation_type=RecommendationType.PERFORMANCE,
                severity=RecommendationSeverity.LOW,
                suggested_changes={"max_concurrent_agents": 3},
                current_values={"max_concurrent_agents": max_agents},
                rationale="With sufficient memory, multiple agents can improve productivity",
                expected_impact="Faster completion of parallel tasks",
                auto_applicable=True,
                confidence_score=0.7
            ))
        
        return recommendations
    
    async def _generate_security_recommendations(self) -> List[ConfigRecommendation]:
        """Generate security-related recommendations."""
        recommendations = []
        
        # Check authentication
        require_auth = self.current_config.get("require_auth", False)
        if not require_auth:
            recommendations.append(ConfigRecommendation(
                id="enable_authentication",
                title="Enable Authentication",
                description="Consider enabling authentication for enhanced security",
                recommendation_type=RecommendationType.SECURITY,
                severity=RecommendationSeverity.MEDIUM,
                suggested_changes={"require_auth": True, "session_timeout_minutes": 60},
                current_values={"require_auth": require_auth},
                rationale="Authentication prevents unauthorized access to your AI agents",
                expected_impact="Improved security, but requires login credentials",
                potential_risks=["User inconvenience", "Locked out if credentials are lost"],
                auto_applicable=False,
                confidence_score=0.6
            ))
        
        # Check data encryption
        encrypt_data = self.current_config.get("encrypt_local_data", False)
        if not encrypt_data:
            recommendations.append(ConfigRecommendation(
                id="enable_encryption",
                title="Enable Local Data Encryption",
                description="Encrypt locally stored data for better privacy protection",
                recommendation_type=RecommendationType.SECURITY,
                severity=RecommendationSeverity.HIGH,
                suggested_changes={"encrypt_local_data": True},
                current_values={"encrypt_local_data": encrypt_data},
                rationale="Encryption protects your data if device is compromised",
                expected_impact="Enhanced data privacy and security",
                requires_restart=True,
                auto_applicable=True,
                confidence_score=0.9
            ))
        
        return recommendations
    
    async def _generate_compatibility_recommendations(self) -> List[ConfigRecommendation]:
        """Generate compatibility-related recommendations."""
        recommendations = []
        
        # Check provider availability
        default_provider = self.current_config.get("default_provider", "")
        if default_provider == "codex":
            # Check if Codex is actually available
            recommendations.append(ConfigRecommendation(
                id="verify_codex_access",
                title="Verify Codex Access",
                description="Ensure you have proper Codex API access configured",
                recommendation_type=RecommendationType.COMPATIBILITY,
                severity=RecommendationSeverity.HIGH,
                suggested_changes={},
                current_values={"default_provider": default_provider},
                rationale="Codex requires valid API credentials and may have usage limits",
                expected_impact="Prevent runtime errors and failed requests",
                auto_applicable=False,
                confidence_score=0.7
            ))
        
        return recommendations
    
    async def _generate_best_practice_recommendations(self) -> List[ConfigRecommendation]:
        """Generate best practice recommendations."""
        recommendations = []
        
        # Check logging level
        log_level = self.current_config.get("log_level", "INFO")
        if log_level == "DEBUG":
            recommendations.append(ConfigRecommendation(
                id="adjust_log_level",
                title="Consider Adjusting Log Level",
                description="DEBUG logging may impact performance and fill up disk space",
                recommendation_type=RecommendationType.BEST_PRACTICE,
                severity=RecommendationSeverity.LOW,
                suggested_changes={"log_level": "INFO"},
                current_values={"log_level": log_level},
                rationale="DEBUG logging creates verbose output that's usually not needed in normal operation",
                expected_impact="Better performance and cleaner logs",
                auto_applicable=True,
                confidence_score=0.5
            ))
        
        return recommendations
    
    async def apply_recommendation(self, recommendation_id: str) -> bool:
        """Apply a specific recommendation."""
        recommendation = next((r for r in self.recommendations if r.id == recommendation_id), None)
        
        if not recommendation:
            logger.warning(f"Recommendation {recommendation_id} not found")
            return False
        
        try:
            # Apply the suggested changes
            for key, value in recommendation.suggested_changes.items():
                self.current_config[key] = value
            
            # Mark as applied
            recommendation.applied = True
            
            # Emit configuration change event
            await self.event_system.emit(Event(
                event_type=EventType.CONFIG,
                data={
                    "action": "recommendation_applied",
                    "recommendation_id": recommendation_id,
                    "changes": recommendation.suggested_changes
                }
            ))
            
            # Regenerate recommendations after changes
            await self._generate_recommendations()
            
            logger.info(f"Applied recommendation: {recommendation.title}")
            return True
            
        except Exception as e:
            logger.exception(f"Error applying recommendation {recommendation_id}: {e}")
            return False
    
    async def dismiss_recommendation(self, recommendation_id: str):
        """Dismiss a recommendation."""
        recommendation = next((r for r in self.recommendations if r.id == recommendation_id), None)
        
        if recommendation:
            recommendation.dismissed = True
            self.dismissed_recommendations.append(recommendation_id)
            
            # Remove from active recommendations
            self.recommendations = [r for r in self.recommendations if r.id != recommendation_id]
            
            await self._render_interface()
            
            logger.info(f"Dismissed recommendation: {recommendation.title}")
    
    # Configuration Wizard
    
    async def start_wizard(self):
        """Start the configuration wizard."""
        self.wizard_active = True
        self.current_wizard_step = WizardStep.WELCOME
        self.wizard_data = {}
        self.current_view = "wizard"
        
        await self._render_interface()
        
        logger.info("Started configuration wizard")
    
    async def wizard_next_step(self):
        """Move to next wizard step."""
        if not self.wizard_active:
            return
        
        # Process current step data
        await self._process_wizard_step()
        
        # Advance to next step
        steps = list(WizardStep)
        current_index = steps.index(self.current_wizard_step)
        
        if current_index < len(steps) - 1:
            self.current_wizard_step = steps[current_index + 1]
            await self._render_interface()
        else:
            # Wizard complete
            await self._complete_wizard()
    
    async def wizard_previous_step(self):
        """Move to previous wizard step."""
        if not self.wizard_active or self.current_wizard_step == WizardStep.WELCOME:
            return
        
        steps = list(WizardStep)
        current_index = steps.index(self.current_wizard_step)
        
        if current_index > 0:
            self.current_wizard_step = steps[current_index - 1]
            await self._render_interface()
    
    async def _process_wizard_step(self):
        """Process current wizard step data."""
        # TODO: Process user input for current step
        # For now, this is a placeholder
        pass
    
    async def _complete_wizard(self):
        """Complete the configuration wizard."""
        # Generate final configuration based on wizard data
        final_config = await self._generate_wizard_config()
        
        # Apply the configuration
        self.current_config.update(final_config)
        
        # End wizard
        self.wizard_active = False
        self.current_view = "recommendations"
        
        # Generate new recommendations
        await self._generate_recommendations()
        
        await self._render_interface()
        
        # Emit wizard completion event
        await self.event_system.emit(Event(
            event_type=EventType.CONFIG,
            data={
                "action": "wizard_completed",
                "final_config": final_config
            }
        ))
        
        logger.info("Configuration wizard completed")
    
    async def _generate_wizard_config(self) -> Dict[str, Any]:
        """Generate configuration based on wizard responses."""
        config = {}
        
        # Extract use case from wizard data
        use_case = self.wizard_data.get("use_case", "general")
        
        # Apply appropriate template
        if use_case == "development":
            template = next((t for t in self.templates if t.id == "developer_power_user"), None)
        elif use_case == "research":
            # Use high-context configuration
            config.update({
                "default_provider": "claude",
                "context_window": 100000,
                "max_tokens": 4096
            })
        elif use_case == "budget":
            template = next((t for t in self.templates if t.id == "cost_optimized"), None)
        else:
            template = next((t for t in self.templates if t.id == "beginner_friendly"), None)
        
        if template:
            config.update(template.configuration)
        
        return config
    
    # Navigation and UI
    
    def set_view(self, view: str):
        """Set the current view."""
        if view in ["recommendations", "wizard", "templates", "help"]:
            self.current_view = view
            self.selected_index = 0
            asyncio.create_task(self._render_interface())
    
    def navigate_items(self, direction: int):
        """Navigate through items in current view."""
        if self.current_view == "recommendations":
            max_idx = len(self.recommendations) - 1
        elif self.current_view == "templates":
            max_idx = len(self.templates) - 1
        else:
            max_idx = 0
        
        if max_idx >= 0:
            self.selected_index = max(0, min(max_idx, self.selected_index + direction))
            asyncio.create_task(self._render_interface())
    
    # Rendering
    
    async def _render_interface(self):
        """Render the configuration assistant interface."""
        if not self.visible:
            return
        
        try:
            caps = self.terminal_manager.detect_capabilities()
            width, height = caps.width, caps.height
            
            if self.current_view == "recommendations":
                content = self._render_recommendations_view(width, height)
            elif self.current_view == "wizard":
                content = self._render_wizard_view(width, height)
            elif self.current_view == "templates":
                content = self._render_templates_view(width, height)
            elif self.current_view == "help":
                content = self._render_help_view(width, height)
            else:
                content = ["Unknown view"]
            
            # Update display
            self.display_renderer.update_region(
                "config_assistant",
                "\n".join(content),
                force=True
            )
            
        except Exception as e:
            logger.exception(f"Error rendering configuration assistant: {e}")
    
    def _render_recommendations_view(self, width: int, height: int) -> List[str]:
        """Render the recommendations view."""
        lines = []
        
        # Header
        title = "╔══ Configuration Assistant - Recommendations ══╗".center(width)
        lines.append(title)
        
        # Stats
        total_recs = len(self.recommendations)
        high_priority = len([r for r in self.recommendations if r.severity == RecommendationSeverity.HIGH])
        
        stats = f"Total: {total_recs} recommendations | High priority: {high_priority}"
        lines.append(stats.center(width))
        lines.append("═" * width)
        
        # Recommendations
        if not self.recommendations:
            lines.append("✅ No recommendations at this time".center(width))
            lines.append("Your configuration looks good!".center(width))
        else:
            for idx, rec in enumerate(self.recommendations[:height-6]):  # Reserve space for header/footer
                is_selected = idx == self.selected_index
                rec_line = self._render_recommendation_line(rec, width, is_selected)
                lines.append(rec_line)
        
        # Footer
        lines.append("─" * width)
        lines.append("Enter: View Details | A: Apply | D: Dismiss | T: Templates | W: Wizard")
        
        return lines
    
    def _render_recommendation_line(self, rec: ConfigRecommendation, width: int, selected: bool) -> str:
        """Render a single recommendation line."""
        # Severity icon
        severity_icons = {
            RecommendationSeverity.LOW: "💡",
            RecommendationSeverity.MEDIUM: "⚠️",
            RecommendationSeverity.HIGH: "🔴",
            RecommendationSeverity.CRITICAL: "🚨"
        }
        severity_icon = severity_icons[rec.severity]
        
        # Selection marker
        marker = "►" if selected else " "
        
        # Confidence indicator
        confidence_indicator = "⭐" * int(rec.confidence_score * 5)
        
        # Build line
        line = f"{marker} {severity_icon} {rec.title}"
        if rec.auto_applicable:
            line += " [Auto]"
        
        return line[:width]
    
    def _render_wizard_view(self, width: int, height: int) -> List[str]:
        """Render the configuration wizard view."""
        lines = []
        
        # Header
        step_name = self.current_wizard_step.value.replace("_", " ").title()
        title = f"╔══ Configuration Wizard - {step_name} ══╗".center(width)
        lines.append(title)
        
        # Progress indicator
        steps = list(WizardStep)
        current_index = steps.index(self.current_wizard_step)
        progress = f"Step {current_index + 1} of {len(steps)}"
        lines.append(progress.center(width))
        lines.append("═" * width)
        
        # Step content based on current step
        if self.current_wizard_step == WizardStep.WELCOME:
            lines.extend([
                "Welcome to the Configuration Assistant!",
                "",
                "This wizard will help you set up AgentsMCP",
                "with optimal settings for your specific needs.",
                "",
                "The process takes about 2-3 minutes and will",
                "configure providers, performance settings,",
                "and security options based on your preferences."
            ])
        
        elif self.current_wizard_step == WizardStep.USE_CASE:
            lines.extend([
                "What will you primarily use AgentsMCP for?",
                "",
                "1. Software Development & Coding",
                "2. Research & Document Analysis", 
                "3. General Questions & Assistance",
                "4. Budget-Conscious Usage",
                "",
                "Select the option that best matches your needs."
            ])
        
        # Add more wizard steps as needed
        
        # Footer
        lines.append("─" * width)
        if self.current_wizard_step == WizardStep.WELCOME:
            lines.append("Enter: Start Wizard | Esc: Cancel")
        else:
            lines.append("Enter: Next | B: Back | Esc: Cancel")
        
        return lines
    
    def _render_templates_view(self, width: int, height: int) -> List[str]:
        """Render the configuration templates view."""
        lines = []
        
        # Header
        title = "╔══ Configuration Templates ══╗".center(width)
        lines.append(title)
        lines.append(f"Choose from {len(self.templates)} pre-configured setups".center(width))
        lines.append("═" * width)
        
        # Templates
        for idx, template in enumerate(self.templates):
            if idx >= height - 6:  # Reserve space for header/footer
                lines.append("... (more templates)")
                break
            
            is_selected = idx == self.selected_index
            marker = "►" if is_selected else " "
            
            template_line = f"{marker} {template.name} - {template.description}"
            lines.append(template_line[:width])
            
            if is_selected:
                # Show template details
                lines.append(f"     Use case: {template.use_case}")
                if template.pros:
                    lines.append(f"     Pros: {', '.join(template.pros[:2])}")
        
        # Footer
        lines.append("─" * width)
        lines.append("Enter: Apply Template | R: Recommendations | W: Wizard")
        
        return lines
    
    def _render_help_view(self, width: int, height: int) -> List[str]:
        """Render the help view."""
        lines = []
        
        # Header
        title = "╔══ Configuration Assistant Help ══╗".center(width)
        lines.append(title)
        lines.append("═" * width)
        
        # Help content
        help_content = [
            "Configuration Assistant Features:",
            "",
            "🔍 RECOMMENDATIONS",
            "  • AI-powered configuration analysis",
            "  • Performance optimization suggestions", 
            "  • Security and compatibility checks",
            "  • Auto-applicable improvements",
            "",
            "🧙 SETUP WIZARD",
            "  • Step-by-step configuration guide",
            "  • Personalized based on your use case",
            "  • Optimal settings for your needs",
            "",
            "📋 TEMPLATES",
            "  • Pre-configured setups for common use cases",
            "  • Beginner-friendly to power-user options",
            "  • One-click application",
            "",
            "Keyboard Shortcuts:",
            "  • ↑/↓: Navigate items",
            "  • Enter: Select/Apply",
            "  • A: Apply recommendation",
            "  • D: Dismiss recommendation",
            "  • W: Start wizard",
            "  • T: View templates",
            "  • H: Show help (this screen)",
            "  • Esc: Close assistant"
        ]
        
        for line in help_content[:height-4]:
            lines.append(line)
        
        # Footer
        lines.append("─" * width)
        lines.append("R: Recommendations | W: Wizard | T: Templates | Esc: Close")
        
        return lines
    
    # Event handling and utilities
    
    async def _register_event_handlers(self):
        """Register event handlers for assistant interaction."""
        
        async def handle_keyboard_event(event: Event):
            if not self.visible or event.event_type != EventType.KEYBOARD:
                return
            
            key = event.data.get('key', '')
            
            # Global navigation
            if key == 'escape':
                if self.wizard_active:
                    self.wizard_active = False
                    self.current_view = "recommendations"
                    await self._render_interface()
                else:
                    await self.hide()
            elif key == 'up':
                self.navigate_items(-1)
            elif key == 'down':
                self.navigate_items(1)
            
            # View-specific actions
            elif self.current_view == "recommendations":
                if key == 'enter':
                    # TODO: Show recommendation details
                    pass
                elif key == 'a':
                    if self.recommendations and self.selected_index < len(self.recommendations):
                        rec = self.recommendations[self.selected_index]
                        await self.apply_recommendation(rec.id)
                elif key == 'd':
                    if self.recommendations and self.selected_index < len(self.recommendations):
                        rec = self.recommendations[self.selected_index]
                        await self.dismiss_recommendation(rec.id)
                elif key == 't':
                    self.set_view("templates")
                elif key == 'w':
                    await self.start_wizard()
                elif key == 'h':
                    self.set_view("help")
            
            elif self.current_view == "wizard":
                if key == 'enter':
                    await self.wizard_next_step()
                elif key == 'b':
                    await self.wizard_previous_step()
            
            elif self.current_view == "templates":
                if key == 'enter':
                    # Apply selected template
                    if self.selected_index < len(self.templates):
                        template = self.templates[self.selected_index]
                        self.current_config.update(template.configuration)
                        await self._generate_recommendations()
                        await self._render_interface()
                elif key == 'r':
                    self.set_view("recommendations")
                elif key == 'w':
                    await self.start_wizard()
            
            elif self.current_view == "help":
                if key == 'r':
                    self.set_view("recommendations")
                elif key == 'w':
                    await self.start_wizard()
                elif key == 't':
                    self.set_view("templates")
        
        await self.event_system.subscribe(EventType.KEYBOARD, handle_keyboard_event)
    
    async def _load_user_preferences(self):
        """Load user preferences and configuration history."""
        # TODO: Load from persistent storage
        pass
    
    async def _clear_interface(self):
        """Clear the assistant interface."""
        if hasattr(self.display_renderer, 'clear_region'):
            self.display_renderer.clear_region("config_assistant")
    
    async def cleanup(self):
        """Cleanup assistant resources."""
        if self.visible:
            await self.hide()
        
        # TODO: Save user preferences and dismissed recommendations
        
        logger.info("Configuration assistant cleanup completed")


# Factory function for easy integration
def create_configuration_assistant(event_system: AsyncEventSystem,
                                 display_renderer: DisplayRenderer,
                                 terminal_manager: TerminalManager) -> ConfigurationAssistant:
    """Create and return a configured configuration assistant instance."""
    return ConfigurationAssistant(
        event_system=event_system,
        display_renderer=display_renderer,
        terminal_manager=terminal_manager
    )