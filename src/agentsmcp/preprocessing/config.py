"""
Configuration management for the preprocessing system.

Provides centralized configuration with environment variable support,
validation, and dynamic updates for the preprocessing pipeline.
"""

import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

from .prompt_optimizer import OptimizationLevel

logger = logging.getLogger(__name__)


class EnvironmentType(Enum):
    """Environment types for configuration."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


@dataclass
class PreprocessingSettings:
    """Complete preprocessing configuration."""
    
    # Core processing settings
    confidence_threshold: float = 0.9
    processing_timeout_ms: int = 5000
    max_clarification_iterations: int = 3
    
    # Feature toggles
    enable_clarification: bool = True
    enable_optimization: bool = True
    enable_context_learning: bool = True
    track_conversation_history: bool = True
    
    # Optimization settings
    optimization_level: OptimizationLevel = OptimizationLevel.STANDARD
    preserve_user_style: bool = True
    add_examples: bool = True
    
    # Context and memory settings
    max_context_entries: int = 1000
    max_session_turns: int = 100
    context_cleanup_interval_minutes: int = 60
    session_timeout_hours: int = 24
    
    # Performance and resource limits
    max_concurrent_sessions: int = 100
    max_questions_per_clarification: int = 5
    intent_analysis_cache_size: int = 500
    optimization_cache_size: int = 200
    
    # Telemetry and monitoring
    enable_telemetry: bool = True
    enable_performance_monitoring: bool = True
    log_level: str = "INFO"
    enable_debug_mode: bool = False
    
    # Quality and validation
    min_confidence_for_delegation: float = 0.7
    max_ambiguous_terms_threshold: int = 5
    require_success_criteria_for_tasks: bool = False
    
    # Language and localization
    default_language: str = "en"
    supported_languages: list = None
    
    # Integration settings
    integration_timeout_ms: int = 30000
    retry_failed_processing: bool = True
    max_retry_attempts: int = 2
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.supported_languages is None:
            self.supported_languages = ["en"]
        
        # Validation
        if not 0.5 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.5 and 1.0")
        
        if self.processing_timeout_ms < 1000:
            raise ValueError("processing_timeout_ms must be at least 1000ms")
        
        if self.max_clarification_iterations < 1:
            raise ValueError("max_clarification_iterations must be at least 1")


class ConfigurationManager:
    """
    Manages preprocessing configuration with environment support.
    
    Handles loading configuration from environment variables,
    validation, and runtime updates.
    """
    
    def __init__(self, environment: EnvironmentType = EnvironmentType.DEVELOPMENT):
        """Initialize configuration manager."""
        self.environment = environment
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Load base configuration
        self._settings = self._load_base_config()
        
        # Apply environment-specific overrides
        self._apply_environment_overrides()
        
        # Apply environment variable overrides
        self._apply_env_var_overrides()
        
        self.logger.info(f"Configuration loaded for {environment.value} environment")
    
    def _load_base_config(self) -> PreprocessingSettings:
        """Load base configuration."""
        return PreprocessingSettings()
    
    def _apply_environment_overrides(self):
        """Apply environment-specific configuration overrides."""
        env_configs = {
            EnvironmentType.DEVELOPMENT: {
                "enable_debug_mode": True,
                "log_level": "DEBUG",
                "processing_timeout_ms": 10000,
                "enable_telemetry": True,
                "max_context_entries": 500,
            },
            
            EnvironmentType.TESTING: {
                "enable_debug_mode": False,
                "log_level": "WARNING",
                "processing_timeout_ms": 2000,
                "enable_telemetry": False,
                "track_conversation_history": False,
                "max_context_entries": 100,
                "max_session_turns": 20,
            },
            
            EnvironmentType.PRODUCTION: {
                "enable_debug_mode": False,
                "log_level": "INFO",
                "processing_timeout_ms": 3000,
                "enable_telemetry": True,
                "enable_performance_monitoring": True,
                "max_context_entries": 2000,
                "optimization_level": OptimizationLevel.ENHANCED,
            }
        }
        
        if self.environment in env_configs:
            for key, value in env_configs[self.environment].items():
                if hasattr(self._settings, key):
                    setattr(self._settings, key, value)
                    self.logger.debug(f"Applied {self.environment.value} override: {key}={value}")
    
    def _apply_env_var_overrides(self):
        """Apply environment variable overrides."""
        env_mappings = {
            # Core processing
            "PREPROCESSING_CONFIDENCE_THRESHOLD": ("confidence_threshold", float),
            "PREPROCESSING_TIMEOUT_MS": ("processing_timeout_ms", int),
            "PREPROCESSING_MAX_CLARIFICATIONS": ("max_clarification_iterations", int),
            
            # Feature toggles
            "PREPROCESSING_ENABLE_CLARIFICATION": ("enable_clarification", bool),
            "PREPROCESSING_ENABLE_OPTIMIZATION": ("enable_optimization", bool),
            "PREPROCESSING_ENABLE_CONTEXT_LEARNING": ("enable_context_learning", bool),
            "PREPROCESSING_TRACK_HISTORY": ("track_conversation_history", bool),
            
            # Optimization
            "PREPROCESSING_OPTIMIZATION_LEVEL": ("optimization_level", str),
            "PREPROCESSING_PRESERVE_STYLE": ("preserve_user_style", bool),
            "PREPROCESSING_ADD_EXAMPLES": ("add_examples", bool),
            
            # Resources
            "PREPROCESSING_MAX_CONTEXTS": ("max_context_entries", int),
            "PREPROCESSING_MAX_TURNS": ("max_session_turns", int),
            "PREPROCESSING_MAX_SESSIONS": ("max_concurrent_sessions", int),
            
            # Monitoring
            "PREPROCESSING_LOG_LEVEL": ("log_level", str),
            "PREPROCESSING_DEBUG_MODE": ("enable_debug_mode", bool),
            "PREPROCESSING_ENABLE_TELEMETRY": ("enable_telemetry", bool),
            
            # Quality
            "PREPROCESSING_MIN_CONFIDENCE": ("min_confidence_for_delegation", float),
            "PREPROCESSING_MAX_AMBIGUOUS": ("max_ambiguous_terms_threshold", int),
        }
        
        for env_var, (attr_name, type_func) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    if type_func == bool:
                        parsed_value = env_value.lower() in ('true', '1', 'yes', 'on')
                    elif type_func == str and attr_name == "optimization_level":
                        parsed_value = OptimizationLevel(env_value.lower())
                    else:
                        parsed_value = type_func(env_value)
                    
                    if hasattr(self._settings, attr_name):
                        setattr(self._settings, attr_name, parsed_value)
                        self.logger.debug(f"Applied env var override: {attr_name}={parsed_value}")
                    
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Invalid env var value for {env_var}: {env_value} - {e}")
    
    @property
    def settings(self) -> PreprocessingSettings:
        """Get current settings."""
        return self._settings
    
    def update_setting(self, key: str, value: Any) -> bool:
        """Update a single setting at runtime."""
        if not hasattr(self._settings, key):
            self.logger.warning(f"Unknown setting key: {key}")
            return False
        
        try:
            # Validate the new value
            old_value = getattr(self._settings, key)
            setattr(self._settings, key, value)
            
            # Try to recreate settings to trigger validation
            test_settings = PreprocessingSettings(**asdict(self._settings))
            
            self.logger.info(f"Updated setting {key}: {old_value} -> {value}")
            return True
            
        except Exception as e:
            # Revert on failure
            setattr(self._settings, key, old_value)
            self.logger.error(f"Failed to update setting {key}: {e}")
            return False
    
    def update_settings(self, updates: Dict[str, Any]) -> Dict[str, bool]:
        """Update multiple settings at once."""
        results = {}
        
        for key, value in updates.items():
            results[key] = self.update_setting(key, value)
        
        return results
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get environment and configuration info."""
        return {
            "environment": self.environment.value,
            "settings": asdict(self._settings),
            "environment_variables_used": [
                var for var in os.environ
                if var.startswith("PREPROCESSING_")
            ]
        }
    
    def validate_settings(self) -> Dict[str, Any]:
        """Validate current settings and return status."""
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Check confidence threshold
        if self._settings.confidence_threshold < 0.7:
            validation_results["warnings"].append(
                "Low confidence threshold may result in frequent clarifications"
            )
        
        if self._settings.confidence_threshold > 0.95:
            validation_results["warnings"].append(
                "Very high confidence threshold may skip beneficial clarifications"
            )
        
        # Check timeout settings
        if self._settings.processing_timeout_ms < 2000:
            validation_results["warnings"].append(
                "Short processing timeout may cause failures on complex requests"
            )
        
        # Check resource limits
        if self._settings.max_context_entries < 100:
            validation_results["warnings"].append(
                "Low context entry limit may reduce learning effectiveness"
            )
        
        if self._settings.max_concurrent_sessions < 10:
            validation_results["warnings"].append(
                "Low session limit may cause performance issues under load"
            )
        
        # Check feature combinations
        if not self._settings.enable_clarification and self._settings.confidence_threshold > 0.8:
            validation_results["warnings"].append(
                "High confidence threshold with clarification disabled may miss ambiguous requests"
            )
        
        if not self._settings.enable_context_learning and self._settings.track_conversation_history:
            validation_results["warnings"].append(
                "Conversation tracking without learning may waste resources"
            )
        
        # Production-specific checks
        if self.environment == EnvironmentType.PRODUCTION:
            if self._settings.enable_debug_mode:
                validation_results["warnings"].append(
                    "Debug mode enabled in production environment"
                )
            
            if self._settings.log_level == "DEBUG":
                validation_results["warnings"].append(
                    "Debug logging enabled in production may impact performance"
                )
        
        validation_results["valid"] = len(validation_results["errors"]) == 0
        
        return validation_results
    
    def reset_to_defaults(self):
        """Reset configuration to defaults."""
        self._settings = self._load_base_config()
        self._apply_environment_overrides()
        self._apply_env_var_overrides()
        
        self.logger.info("Configuration reset to defaults")
    
    def export_config(self) -> Dict[str, Any]:
        """Export current configuration for backup/transfer."""
        return {
            "environment": self.environment.value,
            "settings": asdict(self._settings),
            "export_timestamp": os.time.time(),
            "version": "1.0.0"
        }
    
    def import_config(self, config_data: Dict[str, Any]) -> bool:
        """Import configuration from backup/transfer."""
        try:
            if "settings" not in config_data:
                raise ValueError("Invalid config data format")
            
            # Create new settings from imported data
            new_settings = PreprocessingSettings(**config_data["settings"])
            
            # Validate new settings
            self._settings = new_settings
            validation = self.validate_settings()
            
            if not validation["valid"]:
                raise ValueError(f"Invalid configuration: {validation['errors']}")
            
            self.logger.info("Configuration imported successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to import configuration: {e}")
            return False


# Global configuration manager instance
_config_manager: Optional[ConfigurationManager] = None


def get_config_manager(environment: Optional[EnvironmentType] = None) -> ConfigurationManager:
    """Get or create the global configuration manager."""
    global _config_manager
    
    if _config_manager is None or (environment and _config_manager.environment != environment):
        if environment is None:
            # Detect environment from environment variable
            env_name = os.getenv("PREPROCESSING_ENVIRONMENT", "development").lower()
            environment = EnvironmentType(env_name)
        
        _config_manager = ConfigurationManager(environment)
    
    return _config_manager


def get_settings() -> PreprocessingSettings:
    """Get current preprocessing settings."""
    return get_config_manager().settings


def update_setting(key: str, value: Any) -> bool:
    """Update a preprocessing setting."""
    return get_config_manager().update_setting(key, value)