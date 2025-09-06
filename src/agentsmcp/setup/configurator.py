"""Auto-Configuration System for AgentsMCP.

Generates intelligent, project-specific configuration based on detected environment:
- Creates sensible defaults for common development setups  
- Configures provider settings based on available credentials
- Sets appropriate security settings for environment (dev vs prod)
- Enables relevant features based on detected tools
- Optimizes performance settings for system capabilities
"""

from __future__ import annotations

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import logging

from .environment_detector import EnvironmentProfile
from ..config import Config
from ..paths import default_user_config_path

logger = logging.getLogger(__name__)


@dataclass 
class ConfigurationTemplate:
    """Template for generating configuration."""
    name: str
    description: str
    provider: str
    model: str
    api_key_required: bool
    features: List[str]
    security_level: str
    performance_profile: str
    estimated_cost_per_month: float
    

class AutoConfigurator:
    """Intelligent auto-configuration system."""
    
    # Configuration templates for different scenarios
    TEMPLATES = {
        "development": ConfigurationTemplate(
            name="Development",
            description="Best for local development with cost awareness",
            provider="openai", 
            model="gpt-4o-mini",
            api_key_required=True,
            features=["cost_tracking", "debug_mode", "local_rag"],
            security_level="relaxed",
            performance_profile="balanced",
            estimated_cost_per_month=5.0
        ),
        "production": ConfigurationTemplate(
            name="Production", 
            description="Optimized for production workloads",
            provider="openai",
            model="gpt-4o",
            api_key_required=True,
            features=["cost_tracking", "audit_logging", "rate_limiting"],
            security_level="strict", 
            performance_profile="high_performance",
            estimated_cost_per_month=50.0
        ),
        "local_only": ConfigurationTemplate(
            name="Local Only",
            description="Uses only local models, no external API calls", 
            provider="ollama",
            model="llama3.1:8b",
            api_key_required=False,
            features=["privacy_mode", "offline_capable"],
            security_level="maximum",
            performance_profile="resource_aware",
            estimated_cost_per_month=0.0
        ),
        "hybrid": ConfigurationTemplate(
            name="Hybrid",
            description="Mix of local and cloud models for optimal cost/performance",
            provider="openai",
            model="gpt-4o-mini", 
            api_key_required=True,
            features=["cost_tracking", "model_switching", "local_fallback"],
            security_level="balanced",
            performance_profile="cost_optimized",
            estimated_cost_per_month=15.0
        )
    }
    
    def __init__(self, environment: EnvironmentProfile):
        """
        Initialize auto-configurator with environment profile.
        
        Args:
            environment: Environment profile from detection phase
        """
        self.environment = environment
        
    def generate_configuration(self, preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate optimal configuration based on environment and preferences.
        
        Args:
            preferences: Optional user preferences to override defaults
            
        Returns:
            Complete configuration dictionary ready for use
        """
        logger.info("Generating auto-configuration...")
        
        # Select base template
        template = self._select_optimal_template()
        
        # Generate base configuration
        config = self._build_base_config(template)
        
        # Apply environment-specific optimizations
        config = self._apply_environment_optimizations(config)
        
        # Apply user preferences if provided
        if preferences:
            config = self._apply_user_preferences(config, preferences)
            
        # Validate and finalize configuration
        config = self._validate_and_finalize(config)
        
        return config
    
    def _select_optimal_template(self) -> ConfigurationTemplate:
        """Select the best template based on environment analysis."""
        env = self.environment
        
        # Local-only scenario
        if (env.tools.ollama_available and env.tools.ollama_running and 
            not env.api_keys.detected_providers):
            return self.TEMPLATES["local_only"]
            
        # Production scenario
        if env.installation_mode == "production":
            return self.TEMPLATES["production"]
            
        # Hybrid scenario (has both local and cloud capabilities)
        if (env.tools.ollama_available and 
            len(env.api_keys.detected_providers) > 0):
            return self.TEMPLATES["hybrid"]
            
        # Default to development
        return self.TEMPLATES["development"]
    
    def _build_base_config(self, template: ConfigurationTemplate) -> Dict[str, Any]:
        """Build base configuration from template."""
        config = {
            # Core settings
            "version": "1.0",
            "template_used": template.name,
            "generated_at": time.time(),
            
            # Provider configuration
            "orchestration": {
                "provider": template.provider,
                "model": template.model,
                "mode": "simple" if template.name == "development" else "complex"
            },
            
            # Features
            "features": {
                feature: True for feature in template.features
            },
            
            # Security
            "security": {
                "level": template.security_level,
                "api_key_validation": template.api_key_required,
                "rate_limiting": "rate_limiting" in template.features,
                "audit_logging": "audit_logging" in template.features
            },
            
            # Performance
            "performance": {
                "profile": template.performance_profile,
                "max_concurrent_requests": self._calculate_max_concurrent(),
                "timeout_seconds": self._calculate_timeout(),
                "memory_limit_mb": self._calculate_memory_limit()
            },
            
            # Cost management
            "cost": {
                "tracking_enabled": "cost_tracking" in template.features,
                "monthly_budget": template.estimated_cost_per_month,
                "alert_threshold": 0.8,
                "provider_cost_per_token": self._get_provider_costs(template.provider)
            }
        }
        
        return config
    
    def _apply_environment_optimizations(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply optimizations based on detected environment."""
        env = self.environment
        
        # API key configuration
        if env.api_keys.detected_providers:
            config["api_keys"] = {}
            for provider in env.api_keys.detected_providers:
                key = getattr(env.api_keys, f"{provider}_key")
                if key:
                    config["api_keys"][provider] = key
        
        # MCP server configuration
        if env.mcp_servers.config_files:
            config["mcp"] = {
                "servers": [],
                "config_files": [str(f) for f in env.mcp_servers.config_files]
            }
        
        # System-specific optimizations
        if env.system.memory_gb > 16:
            config["performance"]["memory_limit_mb"] = min(8192, int(env.system.memory_gb * 1024 * 0.5))
        elif env.system.memory_gb < 4:
            config["performance"]["memory_limit_mb"] = 1024
            config["performance"]["max_concurrent_requests"] = 2
            
        # Platform-specific settings
        if env.system.os_name == "Windows":
            config["system"] = {
                "shell": "powershell" if env.system.shell == "powershell" else "cmd",
                "path_separator": "\\",
                "line_endings": "crlf"
            }
        else:
            config["system"] = {
                "shell": env.system.shell,
                "path_separator": "/", 
                "line_endings": "lf"
            }
        
        # Terminal capabilities
        config["ui"] = {
            "colors": env.system.supports_colors,
            "unicode": env.system.supports_unicode,
            "theme": "dark" if env.system.supports_colors else "minimal"
        }
        
        # Project-specific settings
        if env.project.detected_project_type == "nodejs":
            config["integrations"] = {
                "npm": True,
                "package_json": True
            }
        elif env.project.detected_project_type == "python":
            config["integrations"] = {
                "pip": True,
                "requirements_txt": env.project.has_requirements_txt,
                "pyproject_toml": env.project.has_pyproject_toml
            }
        
        # Docker integration
        if env.tools.docker_available:
            config["orchestration"]["docker"] = {
                "enabled": True,
                "use_containers": env.project.has_dockerfile
            }
        
        return config
    
    def _apply_user_preferences(self, config: Dict[str, Any], preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Apply user preferences to override defaults."""
        # Deep merge preferences into config
        def deep_merge(base: Dict, updates: Dict) -> Dict:
            for key, value in updates.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
            return base
        
        return deep_merge(config, preferences)
    
    def _validate_and_finalize(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration and apply final adjustments."""
        # Ensure required fields
        required_fields = ["orchestration", "features", "security", "performance"]
        for field in required_fields:
            if field not in config:
                logger.warning(f"Missing required config field: {field}")
                config[field] = {}
        
        # Validate cost settings
        if config.get("cost", {}).get("monthly_budget", 0) > 1000:
            logger.warning("High monthly budget detected, adding confirmation requirement")
            config["cost"]["require_confirmation"] = True
        
        # Security validations
        if config.get("security", {}).get("level") == "strict":
            config["security"]["require_api_key_validation"] = True
            config["security"]["disable_debug_mode"] = True
        
        return config
    
    def _calculate_max_concurrent(self) -> int:
        """Calculate optimal max concurrent requests based on system."""
        memory_gb = self.environment.system.memory_gb
        cpu_count = self.environment.system.cpu_count
        
        if memory_gb >= 16 and cpu_count >= 8:
            return 10
        elif memory_gb >= 8 and cpu_count >= 4:
            return 5
        else:
            return 2
    
    def _calculate_timeout(self) -> int:
        """Calculate appropriate timeout based on environment."""
        if self.environment.installation_mode == "production":
            return 30  # Shorter timeout for production
        return 120  # Longer timeout for development
    
    def _calculate_memory_limit(self) -> int:
        """Calculate memory limit in MB."""
        total_memory_gb = self.environment.system.memory_gb
        
        if total_memory_gb >= 16:
            return 4096  # 4GB limit
        elif total_memory_gb >= 8:
            return 2048  # 2GB limit
        else:
            return 1024  # 1GB limit
    
    def _get_provider_costs(self, provider: str) -> Dict[str, float]:
        """Get cost information for provider."""
        costs = {
            "openai": {
                "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
                "gpt-4o": {"input": 0.0025, "output": 0.01},
                "gpt-4-turbo": {"input": 0.001, "output": 0.002}
            },
            "anthropic": {
                "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
                "claude-3-sonnet": {"input": 0.003, "output": 0.015},
                "claude-3-opus": {"input": 0.015, "output": 0.075}
            },
            "ollama": {
                "default": {"input": 0.0, "output": 0.0}  # Local models are free
            }
        }
        
        return costs.get(provider, {"default": {"input": 0.001, "output": 0.002}})
    
    def save_configuration(self, config: Dict[str, Any], path: Optional[Path] = None) -> Path:
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary to save
            path: Optional custom path, defaults to user config path
            
        Returns:
            Path where configuration was saved
        """
        if path is None:
            path = default_user_config_path()
        
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as YAML for human readability
        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config, f, default_flow_style=False, indent=2, sort_keys=False)
        
        logger.info(f"Configuration saved to {path}")
        return path
    
    def suggest_improvements(self, config: Dict[str, Any]) -> List[str]:
        """
        Suggest configuration improvements based on environment.
        
        Args:
            config: Current configuration to analyze
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        env = self.environment
        
        # Cost optimization suggestions
        if config.get("cost", {}).get("monthly_budget", 0) > 50:
            if not env.tools.ollama_available:
                suggestions.append("Consider installing Ollama for cost-effective local models")
        
        # Performance suggestions
        if env.system.memory_gb > 8 and config.get("performance", {}).get("memory_limit_mb", 0) < 2048:
            suggestions.append("You can increase memory_limit_mb for better performance")
        
        # Security suggestions  
        if config.get("security", {}).get("level") == "relaxed" and env.installation_mode == "production":
            suggestions.append("Consider using 'strict' security level for production")
        
        # Feature suggestions
        if env.project.is_git_repo and not config.get("integrations", {}).get("git"):
            suggestions.append("Enable Git integration for better project context")
        
        return suggestions


def generate_configuration(environment: EnvironmentProfile, preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function for generating configuration.
    
    Args:
        environment: Environment profile from detection
        preferences: Optional user preferences
        
    Returns:
        Generated configuration dictionary
    """
    configurator = AutoConfigurator(environment)
    return configurator.generate_configuration(preferences)


# Import time for timestamps
import time