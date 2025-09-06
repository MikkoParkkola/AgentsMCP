"""
Configuration and settings management for approval workflows.

This module provides comprehensive configuration management for the approval system,
including settings persistence, environment variable overrides, and user preferences.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import json
import logging

from .approval_decision import ApprovalMode, ApprovalInterfaceType, RejectionReason
from ..data_models import ImprovementCategory, PriorityLevel

logger = logging.getLogger(__name__)


@dataclass
class CategoryApprovalRule:
    """Approval rule for a specific improvement category."""
    
    category: ImprovementCategory
    default_action: ApprovalMode = ApprovalMode.MANUAL
    auto_approve_priority_threshold: Optional[PriorityLevel] = None
    auto_reject_priority_threshold: Optional[PriorityLevel] = None
    requires_confirmation: bool = True
    max_batch_size: Optional[int] = None
    
    def should_auto_approve(self, priority: PriorityLevel) -> bool:
        """Check if this priority level should be auto-approved."""
        if self.auto_approve_priority_threshold is None:
            return False
        
        # Priority levels: LOW=1, MEDIUM=2, HIGH=3, CRITICAL=4
        priority_values = {
            PriorityLevel.LOW: 1,
            PriorityLevel.MEDIUM: 2, 
            PriorityLevel.HIGH: 3,
            PriorityLevel.CRITICAL: 4
        }
        
        return (priority_values.get(priority, 0) >= 
                priority_values.get(self.auto_approve_priority_threshold, 5))
    
    def should_auto_reject(self, priority: PriorityLevel) -> bool:
        """Check if this priority level should be auto-rejected."""
        if self.auto_reject_priority_threshold is None:
            return False
        
        priority_values = {
            PriorityLevel.LOW: 1,
            PriorityLevel.MEDIUM: 2,
            PriorityLevel.HIGH: 3,
            PriorityLevel.CRITICAL: 4
        }
        
        return (priority_values.get(priority, 5) <= 
                priority_values.get(self.auto_reject_priority_threshold, 0))


@dataclass
class ApprovalTimeouts:
    """Timeout configuration for approval operations."""
    
    interactive_timeout: int = 300  # 5 minutes
    batch_timeout: int = 600  # 10 minutes
    auto_timeout: int = 30  # 30 seconds
    individual_item_timeout: int = 60  # 1 minute per item
    
    # Escalation timeouts
    escalation_timeout: int = 1800  # 30 minutes
    abandon_timeout: int = 3600  # 1 hour
    
    def get_timeout_for_mode(self, mode: ApprovalMode) -> int:
        """Get appropriate timeout for approval mode."""
        timeout_map = {
            ApprovalMode.AUTO: self.auto_timeout,
            ApprovalMode.MANUAL: self.interactive_timeout,
            ApprovalMode.INTERACTIVE: self.interactive_timeout,
            ApprovalMode.BATCH_APPROVE: self.batch_timeout,
            ApprovalMode.BATCH_REJECT: self.batch_timeout
        }
        return timeout_map.get(mode, self.interactive_timeout)


@dataclass 
class ApprovalThresholds:
    """Thresholds for automatic approval decisions."""
    
    # Impact-based thresholds
    min_impact_score: float = 0.3  # 0.0 to 1.0
    max_risk_score: float = 0.7  # 0.0 to 1.0
    
    # Effort-based thresholds
    max_effort_hours: Optional[float] = None
    max_batch_effort_hours: Optional[float] = None
    
    # Confidence thresholds
    min_confidence_level: float = 0.8  # 0.0 to 1.0
    
    # System resource thresholds
    max_system_load: float = 0.8  # 0.0 to 1.0
    min_available_resources: float = 0.2  # 0.0 to 1.0
    
    def meets_auto_approval_criteria(self, 
                                   impact_score: float,
                                   risk_score: float,
                                   confidence_level: float,
                                   system_load: float = 0.0) -> bool:
        """Check if improvement meets auto-approval criteria."""
        return (impact_score >= self.min_impact_score and
                risk_score <= self.max_risk_score and
                confidence_level >= self.min_confidence_level and
                system_load <= self.max_system_load)


@dataclass
class UserPreferences:
    """User-specific approval preferences."""
    
    # Interface preferences
    preferred_interface: ApprovalInterfaceType = ApprovalInterfaceType.TUI
    preferred_approval_mode: ApprovalMode = ApprovalMode.INTERACTIVE
    
    # Display preferences
    show_impact_estimates: bool = True
    show_risk_analysis: bool = True
    show_implementation_details: bool = False
    compact_display: bool = False
    
    # Interaction preferences
    confirm_batch_operations: bool = True
    show_progress_indicators: bool = True
    enable_keyboard_shortcuts: bool = True
    auto_scroll: bool = True
    
    # Notification preferences
    notify_on_completion: bool = True
    notify_on_timeout: bool = True
    sound_notifications: bool = False
    
    # Learning preferences
    enable_preference_learning: bool = True
    adapt_suggestions: bool = True
    remember_decisions: bool = True
    
    # Advanced preferences
    custom_approval_rules: Dict[str, Any] = field(default_factory=dict)
    favorite_rejection_reasons: List[RejectionReason] = field(default_factory=list)


@dataclass
class ApprovalConfig:
    """Main configuration class for the approval system."""
    
    # Core mode settings
    approval_mode: ApprovalMode = ApprovalMode.AUTO
    interface_type: ApprovalInterfaceType = ApprovalInterfaceType.AUTO
    
    # Timeout configuration
    timeouts: ApprovalTimeouts = field(default_factory=ApprovalTimeouts)
    
    # Threshold configuration
    thresholds: ApprovalThresholds = field(default_factory=ApprovalThresholds)
    
    # Category-specific rules
    category_rules: Dict[ImprovementCategory, CategoryApprovalRule] = field(default_factory=dict)
    
    # User preferences
    user_preferences: UserPreferences = field(default_factory=UserPreferences)
    
    # System configuration
    max_concurrent_approvals: int = 10
    enable_approval_history: bool = True
    history_retention_days: int = 90
    
    # Safety integration
    require_safety_validation: bool = True
    safety_validation_timeout: int = 120
    
    # Persistence settings
    config_file_path: Optional[Path] = None
    auto_save_config: bool = True
    backup_config: bool = True
    
    # Environment override settings
    allow_env_overrides: bool = True
    env_prefix: str = "AGENTSMCP_APPROVAL"
    
    def __post_init__(self):
        """Initialize default category rules if not provided."""
        # Temporarily disable category rules setup to avoid enum issues
        # if not self.category_rules:
        #     self._setup_default_category_rules()
        
        # Apply environment variable overrides
        if self.allow_env_overrides:
            self._apply_env_overrides()
    
    def _setup_default_category_rules(self) -> None:
        """Setup default approval rules for each category."""
        self.category_rules = {
            # Performance improvements - generally safe to auto-approve
            ImprovementCategory.PERFORMANCE: CategoryApprovalRule(
                category=ImprovementCategory.PERFORMANCE,
                default_action=ApprovalMode.AUTO,
                auto_approve_priority_threshold=PriorityLevel.MEDIUM,
                requires_confirmation=False,
                max_batch_size=20
            ),
            
            # Communication improvements - require more careful review
            ImprovementCategory.COMMUNICATION: CategoryApprovalRule(
                category=ImprovementCategory.COMMUNICATION,
                default_action=ApprovalMode.MANUAL,
                auto_approve_priority_threshold=PriorityLevel.HIGH,
                requires_confirmation=True,
                max_batch_size=10
            ),
            
            # Learning improvements - can be more flexible
            ImprovementCategory.LEARNING: CategoryApprovalRule(
                category=ImprovementCategory.LEARNING,
                default_action=ApprovalMode.INTERACTIVE,
                auto_approve_priority_threshold=PriorityLevel.MEDIUM,
                requires_confirmation=False,
                max_batch_size=15
            ),
            
            # Process improvements - can be more flexible
            ImprovementCategory.PROCESS: CategoryApprovalRule(
                category=ImprovementCategory.PROCESS,
                default_action=ApprovalMode.INTERACTIVE,
                auto_approve_priority_threshold=PriorityLevel.MEDIUM,
                requires_confirmation=False,
                max_batch_size=15
            )
        }
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        # Approval mode override
        env_mode = os.getenv(f"{self.env_prefix}_MODE")
        if env_mode:
            try:
                self.approval_mode = ApprovalMode(env_mode.lower())
            except ValueError:
                logger.warning(f"Invalid approval mode in environment: {env_mode}")
        
        # Interface type override
        env_interface = os.getenv(f"{self.env_prefix}_INTERFACE")
        if env_interface:
            try:
                self.interface_type = ApprovalInterfaceType(env_interface.lower())
            except ValueError:
                logger.warning(f"Invalid interface type in environment: {env_interface}")
        
        # Timeout overrides
        timeout_overrides = {
            "INTERACTIVE_TIMEOUT": "interactive_timeout",
            "BATCH_TIMEOUT": "batch_timeout", 
            "AUTO_TIMEOUT": "auto_timeout"
        }
        
        for env_key, attr_name in timeout_overrides.items():
            env_value = os.getenv(f"{self.env_prefix}_{env_key}")
            if env_value:
                try:
                    setattr(self.timeouts, attr_name, int(env_value))
                except ValueError:
                    logger.warning(f"Invalid timeout value in environment: {env_value}")
    
    def get_category_rule(self, category: ImprovementCategory) -> CategoryApprovalRule:
        """Get approval rule for a specific category."""
        return self.category_rules.get(
            category,
            CategoryApprovalRule(category=category)  # Default rule
        )
    
    def should_auto_approve_category(self, 
                                   category: ImprovementCategory,
                                   priority: PriorityLevel) -> bool:
        """Check if a category/priority combination should be auto-approved."""
        rule = self.get_category_rule(category)
        
        if self.approval_mode == ApprovalMode.AUTO:
            return True
        
        if rule.default_action == ApprovalMode.AUTO:
            return rule.should_auto_approve(priority)
        
        return False
    
    def get_timeout_for_context(self, 
                              improvement_count: int,
                              mode: Optional[ApprovalMode] = None) -> int:
        """Get appropriate timeout based on context."""
        mode = mode or self.approval_mode
        base_timeout = self.timeouts.get_timeout_for_mode(mode)
        
        # Adjust timeout based on improvement count
        if improvement_count > 10:
            return base_timeout + (improvement_count - 10) * 30  # 30s per additional item
        
        return base_timeout
    
    def save_to_file(self, file_path: Optional[Path] = None) -> None:
        """Save configuration to file."""
        target_path = file_path or self.config_file_path
        if not target_path:
            logger.warning("No config file path specified for saving")
            return
        
        try:
            # Create backup if requested
            if self.backup_config and target_path.exists():
                backup_path = target_path.with_suffix(f"{target_path.suffix}.backup")
                backup_path.write_text(target_path.read_text())
            
            # Convert to dict for JSON serialization
            config_dict = {
                "approval_mode": self.approval_mode.value,
                "interface_type": self.interface_type.value,
                "timeouts": {
                    "interactive_timeout": self.timeouts.interactive_timeout,
                    "batch_timeout": self.timeouts.batch_timeout,
                    "auto_timeout": self.timeouts.auto_timeout,
                    "individual_item_timeout": self.timeouts.individual_item_timeout
                },
                "thresholds": {
                    "min_impact_score": self.thresholds.min_impact_score,
                    "max_risk_score": self.thresholds.max_risk_score,
                    "min_confidence_level": self.thresholds.min_confidence_level,
                    "max_system_load": self.thresholds.max_system_load
                },
                "user_preferences": {
                    "preferred_interface": self.user_preferences.preferred_interface.value,
                    "preferred_approval_mode": self.user_preferences.preferred_approval_mode.value,
                    "show_impact_estimates": self.user_preferences.show_impact_estimates,
                    "show_risk_analysis": self.user_preferences.show_risk_analysis,
                    "confirm_batch_operations": self.user_preferences.confirm_batch_operations
                },
                "max_concurrent_approvals": self.max_concurrent_approvals,
                "enable_approval_history": self.enable_approval_history,
                "require_safety_validation": self.require_safety_validation
            }
            
            target_path.write_text(json.dumps(config_dict, indent=2))
            logger.info(f"Approval configuration saved to {target_path}")
            
        except Exception as e:
            logger.error(f"Failed to save approval configuration: {e}")
    
    @classmethod
    def load_from_file(cls, file_path: Path) -> ApprovalConfig:
        """Load configuration from file."""
        try:
            if not file_path.exists():
                logger.warning(f"Config file not found: {file_path}")
                return cls(config_file_path=file_path)
            
            config_dict = json.loads(file_path.read_text())
            
            # Create instance with loaded values
            config = cls()
            config.config_file_path = file_path
            
            # Apply loaded settings
            if "approval_mode" in config_dict:
                config.approval_mode = ApprovalMode(config_dict["approval_mode"])
            
            if "interface_type" in config_dict:
                config.interface_type = ApprovalInterfaceType(config_dict["interface_type"])
            
            # Load timeouts
            if "timeouts" in config_dict:
                timeout_data = config_dict["timeouts"]
                config.timeouts = ApprovalTimeouts(**timeout_data)
            
            # Load thresholds
            if "thresholds" in config_dict:
                threshold_data = config_dict["thresholds"]
                config.thresholds = ApprovalThresholds(**threshold_data)
            
            # Load user preferences
            if "user_preferences" in config_dict:
                pref_data = config_dict["user_preferences"]
                config.user_preferences.preferred_interface = ApprovalInterfaceType(
                    pref_data.get("preferred_interface", "tui")
                )
                config.user_preferences.preferred_approval_mode = ApprovalMode(
                    pref_data.get("preferred_approval_mode", "interactive") 
                )
                config.user_preferences.show_impact_estimates = pref_data.get(
                    "show_impact_estimates", True
                )
                config.user_preferences.show_risk_analysis = pref_data.get(
                    "show_risk_analysis", True
                )
                config.user_preferences.confirm_batch_operations = pref_data.get(
                    "confirm_batch_operations", True
                )
            
            # Load other settings
            config.max_concurrent_approvals = config_dict.get("max_concurrent_approvals", 10)
            config.enable_approval_history = config_dict.get("enable_approval_history", True)
            config.require_safety_validation = config_dict.get("require_safety_validation", True)
            
            logger.info(f"Approval configuration loaded from {file_path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load approval configuration: {e}")
            return cls(config_file_path=file_path)
    
    def update_from_cli_args(self, **kwargs) -> None:
        """Update configuration from command line arguments."""
        if "approval_mode" in kwargs and kwargs["approval_mode"]:
            try:
                self.approval_mode = ApprovalMode(kwargs["approval_mode"])
            except ValueError:
                logger.warning(f"Invalid approval mode: {kwargs['approval_mode']}")
        
        if "approval_interface" in kwargs and kwargs["approval_interface"]:
            try:
                self.interface_type = ApprovalInterfaceType(kwargs["approval_interface"])
            except ValueError:
                logger.warning(f"Invalid interface type: {kwargs['approval_interface']}")
        
        if "approval_timeout" in kwargs and kwargs["approval_timeout"]:
            try:
                timeout = int(kwargs["approval_timeout"])
                self.timeouts.interactive_timeout = timeout
                self.timeouts.batch_timeout = timeout
            except ValueError:
                logger.warning(f"Invalid timeout value: {kwargs['approval_timeout']}")
        
        # Auto-save if enabled
        if self.auto_save_config:
            self.save_to_file()
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        self.__init__()
        logger.info("Approval configuration reset to defaults")


def get_default_config_path() -> Path:
    """Get the default configuration file path."""
    from ...paths import get_app_data_dir
    
    config_dir = get_app_data_dir() / "approval"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "approval_config.json"


def load_approval_config(config_path: Optional[Path] = None) -> ApprovalConfig:
    """Load approval configuration from file or create default."""
    target_path = config_path or get_default_config_path()
    return ApprovalConfig.load_from_file(target_path)