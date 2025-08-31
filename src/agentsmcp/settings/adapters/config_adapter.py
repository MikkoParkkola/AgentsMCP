"""
Adapter for integrating with existing AgentsMCP configuration system.

This adapter bridges the new settings management system
with the existing configuration loading and management.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List
import json
import yaml
from pathlib import Path

from ...config import get_config, ConfigLoader, DEFAULT_CONFIG
from ..domain.entities import SettingsNode, SettingsHierarchy, UserProfile
from ..domain.value_objects import SettingsLevel, SettingValue, SettingType


class AgentsMCPConfigAdapter:
    """
    Adapter for bridging with existing AgentsMCP configuration system.
    
    This adapter allows the new settings system to read from and write to
    the existing configuration files and environment variables.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path.home() / ".agentsmcp"
        self.config_loader = ConfigLoader()
        
        # Ensure config directory exists
        self.config_path.mkdir(exist_ok=True)
    
    async def import_existing_config(self, user_id: str) -> SettingsHierarchy:
        """
        Import existing AgentsMCP configuration into the new settings system.
        
        Creates a hierarchy that mirrors the current configuration structure.
        """
        # Load existing configuration
        config = get_config()
        
        # Create hierarchy
        hierarchy = SettingsHierarchy(
            name="AgentsMCP Configuration",
        )
        
        # Create global settings node
        global_node = SettingsNode(
            level=SettingsLevel.GLOBAL,
            name="Global Settings"
        )
        
        # Convert config to settings
        self._convert_config_to_settings(config, global_node)
        
        hierarchy.add_node(global_node)
        hierarchy.root_node_id = global_node.id
        
        # Create user-specific node if user preferences exist
        user_config_file = self.config_path / "user_config.yaml"
        if user_config_file.exists():
            user_node = await self._load_user_config(user_config_file, global_node.id)
            hierarchy.add_node(user_node)
        
        return hierarchy
    
    async def export_to_legacy_format(self, hierarchy: SettingsHierarchy,
                                    format: str = "yaml") -> Dict[str, Any]:
        """
        Export settings hierarchy to legacy AgentsMCP configuration format.
        """
        # Get all effective settings from hierarchy
        if not hierarchy.root_node_id:
            return {}
        
        root_node = hierarchy.get_node(hierarchy.root_node_id)
        if not root_node:
            return {}
        
        # Convert settings to legacy format
        legacy_config = {}
        
        for key, setting in root_node.settings.items():
            legacy_key = self._map_setting_key_to_legacy(key)
            legacy_value = self._convert_setting_to_legacy_value(setting)
            
            # Use dot notation to create nested structure
            self._set_nested_value(legacy_config, legacy_key, legacy_value)
        
        return legacy_config
    
    async def sync_with_environment(self, node: SettingsNode) -> None:
        """
        Sync settings node with environment variables.
        
        This allows environment variables to override settings values
        following the existing AgentsMCP pattern.
        """
        import os
        
        # Map of setting keys to environment variable names
        env_mappings = {
            "openai_api_key": "OPENAI_API_KEY",
            "anthropic_api_key": "ANTHROPIC_API_KEY",
            "model_name": "DEFAULT_MODEL",
            "max_tokens": "MAX_TOKENS",
            "temperature": "TEMPERATURE",
            "debug": "DEBUG",
        }
        
        for setting_key, env_var in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert environment string to appropriate type
                setting_type, converted_value = self._convert_env_value(env_value)
                
                setting_value = SettingValue(
                    value=converted_value,
                    type=setting_type,
                    source_level=SettingsLevel.GLOBAL,
                    metadata={"source": "environment", "env_var": env_var}
                )
                
                node.set_setting(setting_key, setting_value)
    
    async def validate_against_schema(self, settings: Dict[str, Any]) -> List[str]:
        """
        Validate settings against existing AgentsMCP configuration schema.
        """
        errors = []
        
        # Use existing validation logic from AgentsMCP config system
        try:
            # This would use the existing config validation
            # For now, basic validation
            required_fields = ["model_name"]
            
            for field in required_fields:
                if field not in settings:
                    errors.append(f"Required field '{field}' is missing")
            
        except Exception as e:
            errors.append(f"Configuration validation failed: {str(e)}")
        
        return errors
    
    def _convert_config_to_settings(self, config: Any, node: SettingsNode) -> None:
        """Convert configuration object to settings in a node."""
        if hasattr(config, '__dict__'):
            config_dict = config.__dict__
        elif isinstance(config, dict):
            config_dict = config
        else:
            return
        
        for key, value in config_dict.items():
            if key.startswith('_'):  # Skip private attributes
                continue
            
            setting_type = self._infer_setting_type(value)
            setting_value = SettingValue(
                value=value,
                type=setting_type,
                source_level=node.level,
                metadata={"source": "legacy_config"}
            )
            
            node.set_setting(key, setting_value)
    
    async def _load_user_config(self, config_file: Path, parent_id: str) -> SettingsNode:
        """Load user-specific configuration from file."""
        user_node = SettingsNode(
            level=SettingsLevel.USER,
            name="User Settings",
            parent_id=parent_id
        )
        
        try:
            with open(config_file, 'r') as f:
                user_config = yaml.safe_load(f)
            
            if user_config:
                for key, value in user_config.items():
                    setting_type = self._infer_setting_type(value)
                    setting_value = SettingValue(
                        value=value,
                        type=setting_type,
                        source_level=SettingsLevel.USER,
                        metadata={"source": "user_config_file"}
                    )
                    
                    user_node.set_setting(key, setting_value)
        
        except Exception as e:
            # Log error but continue
            pass
        
        return user_node
    
    def _map_setting_key_to_legacy(self, key: str) -> str:
        """Map new setting key to legacy configuration key."""
        # Mapping table for backwards compatibility
        key_mappings = {
            "openai_api_key": "providers.openai.api_key",
            "anthropic_api_key": "providers.anthropic.api_key",
            "model_name": "default_model",
            "max_tokens": "generation.max_tokens",
            "temperature": "generation.temperature",
        }
        
        return key_mappings.get(key, key)
    
    def _convert_setting_to_legacy_value(self, setting: SettingValue) -> Any:
        """Convert setting value to legacy format."""
        # Handle special cases for backwards compatibility
        if setting.type == SettingType.SECRET:
            return "***ENCRYPTED***"  # Don't export actual secrets
        
        return setting.value
    
    def _set_nested_value(self, config_dict: Dict[str, Any], key_path: str, value: Any) -> None:
        """Set nested dictionary value using dot notation."""
        keys = key_path.split('.')
        current = config_dict
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _convert_env_value(self, env_value: str) -> tuple[SettingType, Any]:
        """Convert environment variable string to typed value."""
        # Try to infer type from string value
        env_value = env_value.strip()
        
        # Boolean values
        if env_value.lower() in ('true', 'false'):
            return SettingType.BOOLEAN, env_value.lower() == 'true'
        
        # Integer values
        if env_value.isdigit():
            return SettingType.INTEGER, int(env_value)
        
        # Float values
        try:
            float_val = float(env_value)
            if '.' in env_value:
                return SettingType.FLOAT, float_val
        except ValueError:
            pass
        
        # JSON values (arrays/objects)
        if env_value.startswith(('{', '[')):
            try:
                json_val = json.loads(env_value)
                if isinstance(json_val, list):
                    return SettingType.ARRAY, json_val
                elif isinstance(json_val, dict):
                    return SettingType.OBJECT, json_val
            except json.JSONDecodeError:
                pass
        
        # Default to string
        return SettingType.STRING, env_value
    
    def _infer_setting_type(self, value: Any) -> SettingType:
        """Infer setting type from value."""
        if isinstance(value, bool):
            return SettingType.BOOLEAN
        elif isinstance(value, int):
            return SettingType.INTEGER
        elif isinstance(value, float):
            return SettingType.FLOAT
        elif isinstance(value, (list, tuple)):
            return SettingType.ARRAY
        elif isinstance(value, dict):
            return SettingType.OBJECT
        else:
            return SettingType.STRING