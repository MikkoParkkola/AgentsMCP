"""Configuration and thresholds for safety validation framework."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum


class SafetyLevel(str, Enum):
    """Safety validation levels."""
    MINIMAL = "minimal"  # Basic validation only
    STANDARD = "standard"  # Standard production safety checks
    STRICT = "strict"  # Maximum safety with comprehensive checks
    EMERGENCY = "emergency"  # Emergency mode with immediate rollback


class RollbackTrigger(str, Enum):
    """Conditions that trigger automatic rollback."""
    ERROR_RATE_SPIKE = "error_rate_spike"
    RESPONSE_TIME_DEGRADATION = "response_time_degradation"
    MEMORY_LEAK = "memory_leak"
    CRITICAL_ROLE_FAILURE = "critical_role_failure"
    MANUAL_TRIGGER = "manual_trigger"
    HEALTH_CHECK_FAILURE = "health_check_failure"


@dataclass
class SafetyThresholds:
    """Configurable safety thresholds for monitoring and rollback."""
    
    # Response time thresholds
    max_response_time_increase_percent: float = 50.0
    response_time_violation_window_seconds: int = 60
    
    # Error rate thresholds
    max_error_rate_increase_percent: float = 20.0
    error_rate_violation_window_seconds: int = 30
    
    # Memory usage thresholds
    max_memory_increase_percent: float = 30.0
    memory_leak_detection_window_seconds: int = 300
    
    # Health check thresholds
    health_check_failure_threshold: int = 3  # consecutive failures
    health_check_timeout_seconds: int = 10
    
    # Rollback timing
    rollback_decision_timeout_seconds: int = 30
    max_rollback_attempts: int = 3
    
    # Critical system protection
    critical_roles: Set[str] = field(default_factory=lambda: {
        "orchestrator", "process-coach", "architect"
    })


@dataclass 
class SafetyConfig:
    """Comprehensive configuration for safety validation framework."""
    
    # Safety level and mode
    safety_level: SafetyLevel = SafetyLevel.STANDARD
    enable_auto_rollback: bool = True
    rollback_timeout_seconds: int = 120
    
    # Monitoring configuration
    health_monitoring_enabled: bool = True
    health_check_interval_seconds: int = 30
    baseline_collection_duration_seconds: int = 300  # 5 minutes
    post_change_monitoring_duration_seconds: int = 600  # 10 minutes
    
    # Validation configuration
    enable_critical_path_protection: bool = True
    enable_configuration_backup: bool = True
    enable_state_isolation: bool = True
    require_manual_approval_for_critical_changes: bool = True
    
    # Rollback configuration
    max_rollback_history: int = 10
    rollback_state_persistence_enabled: bool = True
    rollback_state_storage_path: str = field(
        default_factory=lambda: os.path.expanduser("~/.agentsmcp/safety/rollback_states")
    )
    
    # File system safety
    backup_directory: str = field(
        default_factory=lambda: os.path.expanduser("~/.agentsmcp/safety/backups")
    )
    max_backup_retention_days: int = 30
    
    # Thresholds
    thresholds: SafetyThresholds = field(default_factory=SafetyThresholds)
    
    # Validation rules
    validation_rules_enabled: Dict[str, bool] = field(default_factory=lambda: {
        "no_critical_role_changes": True,
        "preserve_core_functionality": True,
        "validate_configuration_syntax": True,
        "check_dependency_compatibility": True,
        "verify_resource_availability": True,
        "validate_security_constraints": True,
    })
    
    # Integration settings
    integrate_with_improvement_implementer: bool = True
    notify_on_safety_violations: bool = True
    log_all_safety_events: bool = True
    
    # Testing and development
    enable_dry_run_mode: bool = False
    skip_safety_in_development: bool = False
    development_mode_markers: List[str] = field(default_factory=lambda: [
        "localhost", "127.0.0.1", "dev", "test", "staging"
    ])
    
    @classmethod
    def create_development_config(cls) -> SafetyConfig:
        """Create a development-friendly configuration."""
        return cls(
            safety_level=SafetyLevel.MINIMAL,
            enable_auto_rollback=False,
            health_check_interval_seconds=60,
            post_change_monitoring_duration_seconds=120,
            require_manual_approval_for_critical_changes=False,
            enable_dry_run_mode=True,
            skip_safety_in_development=True,
        )
    
    @classmethod
    def create_production_config(cls) -> SafetyConfig:
        """Create a production-ready configuration."""
        return cls(
            safety_level=SafetyLevel.STRICT,
            enable_auto_rollback=True,
            health_check_interval_seconds=15,
            post_change_monitoring_duration_seconds=900,  # 15 minutes
            require_manual_approval_for_critical_changes=True,
            enable_dry_run_mode=False,
            skip_safety_in_development=False,
        )
    
    @classmethod
    def create_emergency_config(cls) -> SafetyConfig:
        """Create emergency mode configuration with immediate rollback."""
        return cls(
            safety_level=SafetyLevel.EMERGENCY,
            enable_auto_rollback=True,
            rollback_timeout_seconds=30,
            health_check_interval_seconds=5,
            post_change_monitoring_duration_seconds=60,
            thresholds=SafetyThresholds(
                max_response_time_increase_percent=10.0,
                max_error_rate_increase_percent=5.0,
                rollback_decision_timeout_seconds=10,
            )
        )
    
    def is_development_environment(self) -> bool:
        """Check if we're running in a development environment."""
        if self.skip_safety_in_development:
            hostname = os.environ.get("HOSTNAME", "")
            if any(marker in hostname.lower() for marker in self.development_mode_markers):
                return True
        return False
    
    def get_effective_safety_level(self) -> SafetyLevel:
        """Get the effective safety level considering environment."""
        if self.is_development_environment():
            return SafetyLevel.MINIMAL
        return self.safety_level
    
    def should_skip_safety_checks(self) -> bool:
        """Determine if safety checks should be skipped."""
        return (
            self.skip_safety_in_development and 
            self.is_development_environment()
        )
    
    def validate(self) -> List[str]:
        """Validate configuration and return any errors."""
        errors = []
        
        if self.rollback_timeout_seconds <= 0:
            errors.append("rollback_timeout_seconds must be positive")
        
        if self.health_check_interval_seconds <= 0:
            errors.append("health_check_interval_seconds must be positive")
        
        if self.thresholds.max_response_time_increase_percent <= 0:
            errors.append("max_response_time_increase_percent must be positive")
        
        if self.thresholds.max_error_rate_increase_percent <= 0:
            errors.append("max_error_rate_increase_percent must be positive")
        
        if not os.path.exists(os.path.dirname(self.rollback_state_storage_path)):
            try:
                os.makedirs(os.path.dirname(self.rollback_state_storage_path), exist_ok=True)
            except OSError as e:
                errors.append(f"Cannot create rollback state directory: {e}")
        
        if not os.path.exists(os.path.dirname(self.backup_directory)):
            try:
                os.makedirs(os.path.dirname(self.backup_directory), exist_ok=True)
            except OSError as e:
                errors.append(f"Cannot create backup directory: {e}")
        
        return errors