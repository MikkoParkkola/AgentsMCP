"""
Validation service for settings and agents.

Provides comprehensive validation capabilities with real-time
validation and smart suggestions.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import re
import asyncio

from ..domain.entities import (
    SettingsNode,
    SettingsHierarchy,
    AgentDefinition,
    AgentInstance,
    UserProfile,
)
from ..domain.value_objects import (
    SettingValue,
    SettingType,
    ValidationRule,
    SettingsLevel,
    AgentStatus,
    SettingsValidationError,
)
from ..domain.repositories import (
    SettingsRepository,
    AgentRepository,
    CacheRepository,
)
from ..domain.services import SettingsValidationService
from ..events.validation_events import (
    ValidationCompletedEvent,
    ValidationFailedEvent,
    SmartSuggestionGeneratedEvent,
)
from ..events.event_publisher import EventPublisher


class ValidationService:
    """
    Main application service for validation operations.
    
    Provides real-time validation, smart suggestions, and
    comprehensive validation reporting.
    """
    
    def __init__(self,
                 settings_repository: SettingsRepository,
                 agent_repository: AgentRepository,
                 cache_repository: CacheRepository,
                 event_publisher: EventPublisher):
        self.settings_repo = settings_repository
        self.agent_repo = agent_repository
        self.cache_repo = cache_repository
        self.event_publisher = event_publisher
        
        # Domain services
        self.validation_service = SettingsValidationService()
        
        # Smart validation patterns
        self.smart_patterns = self._init_smart_patterns()
    
    async def validate_setting_real_time(self, user_id: str, node_id: str,
                                       key: str, value: Any) -> Dict[str, Any]:
        """
        Perform real-time validation of a setting value.
        
        Returns immediate validation results with suggestions.
        """
        node = await self.settings_repo.get_node(node_id)
        if not node:
            return {"valid": False, "errors": ["Node not found"]}
        
        # Create temporary setting value for validation
        setting_type = self._infer_setting_type(value)
        setting_value = SettingValue(
            value=value,
            type=setting_type,
            source_level=node.level
        )
        
        # Get hierarchy and validation rules
        hierarchy = await self._get_hierarchy_for_node(node_id)
        if not hierarchy:
            return {"valid": False, "errors": ["Hierarchy not found"]}
        
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": [],
            "auto_fixes": [],
            "dependencies": [],
            "conflicts": []
        }
        
        # Basic validation rules
        validation_errors = []
        for rule in hierarchy.validation_rules:
            if rule.setting_key == key or rule.setting_key == "*":
                is_valid, message = rule.validate(setting_value)
                if not is_valid:
                    error_info = {
                        "rule_id": rule.id,
                        "message": message,
                        "severity": rule.severity
                    }
                    if rule.severity == "error":
                        result["errors"].append(error_info)
                        result["valid"] = False
                    else:
                        result["warnings"].append(error_info)
                    validation_errors.append(error_info)
        
        # Smart pattern validation
        smart_results = await self._apply_smart_patterns(key, value, node, hierarchy)
        result["suggestions"].extend(smart_results.get("suggestions", []))
        result["auto_fixes"].extend(smart_results.get("auto_fixes", []))
        
        # Check for dependencies
        dependencies = await self._check_setting_dependencies(key, value, node, hierarchy)
        result["dependencies"] = dependencies
        
        # Check for conflicts
        conflicts = await self._check_setting_conflicts(key, value, node, hierarchy)
        result["conflicts"] = conflicts
        
        # Generate suggestions if there are errors
        if not result["valid"] and not result["suggestions"]:
            suggestions = await self._generate_smart_suggestions(key, value, validation_errors)
            result["suggestions"] = suggestions
        
        # Publish validation event
        if result["valid"]:
            event = ValidationCompletedEvent(
                node_id=node_id,
                user_id=user_id,
                setting_key=key,
                validation_result=result
            )
        else:
            event = ValidationFailedEvent(
                node_id=node_id,
                user_id=user_id,
                setting_key=key,
                errors=result["errors"],
                suggestions=result["suggestions"]
            )
        
        await self.event_publisher.publish(event)
        
        return result
    
    async def validate_settings_bulk(self, user_id: str, node_id: str,
                                   settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate multiple settings in bulk with cross-setting validation.
        """
        node = await self.settings_repo.get_node(node_id)
        hierarchy = await self._get_hierarchy_for_node(node_id)
        
        if not node or not hierarchy:
            return {"valid": False, "errors": ["Node or hierarchy not found"]}
        
        results = {
            "valid": True,
            "setting_results": {},
            "cross_setting_errors": [],
            "overall_suggestions": [],
            "validation_summary": {
                "total_settings": len(settings),
                "valid_settings": 0,
                "invalid_settings": 0,
                "warnings": 0
            }
        }
        
        # Validate each setting individually
        for key, value in settings.items():
            setting_result = await self.validate_setting_real_time(user_id, node_id, key, value)
            results["setting_results"][key] = setting_result
            
            if setting_result["valid"]:
                results["validation_summary"]["valid_settings"] += 1
            else:
                results["validation_summary"]["invalid_settings"] += 1
                results["valid"] = False
            
            results["validation_summary"]["warnings"] += len(setting_result.get("warnings", []))
        
        # Cross-setting validation
        cross_errors = await self._validate_cross_settings(settings, node, hierarchy)
        results["cross_setting_errors"] = cross_errors
        
        if cross_errors:
            results["valid"] = False
        
        # Generate overall suggestions
        if not results["valid"]:
            overall_suggestions = await self._generate_bulk_suggestions(results)
            results["overall_suggestions"] = overall_suggestions
        
        return results
    
    async def validate_agent_configuration(self, user_id: str, agent_id: str,
                                         configuration: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate agent configuration against its schema and requirements.
        """
        agent = await self.agent_repo.get_agent_definition(agent_id)
        if not agent:
            return {"valid": False, "errors": ["Agent not found"]}
        
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": [],
            "schema_validation": {},
            "capability_validation": {},
            "security_validation": {}
        }
        
        # Schema validation
        schema_errors = agent.validate_settings(configuration)
        if schema_errors:
            result["valid"] = False
            result["errors"].extend([{"type": "schema", "message": error} for error in schema_errors])
        
        # Validate against agent's settings schema
        for key, value in configuration.items():
            if key in agent.settings_schema:
                schema = agent.settings_schema[key]
                schema_result = await self._validate_against_schema(key, value, schema)
                result["schema_validation"][key] = schema_result
                
                if not schema_result["valid"]:
                    result["valid"] = False
        
        # Capability validation
        capability_result = await self._validate_agent_capabilities(agent, configuration)
        result["capability_validation"] = capability_result
        
        # Security validation
        security_result = await self._validate_agent_security(agent, configuration)
        result["security_validation"] = security_result
        
        if not capability_result["valid"] or not security_result["valid"]:
            result["valid"] = False
        
        return result
    
    async def get_smart_suggestions(self, user_id: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate smart configuration suggestions based on context.
        """
        suggestions = []
        
        # Context-based suggestions
        if "node_id" in context:
            node_suggestions = await self._get_node_suggestions(context["node_id"])
            suggestions.extend(node_suggestions)
        
        if "agent_id" in context:
            agent_suggestions = await self._get_agent_suggestions(context["agent_id"])
            suggestions.extend(agent_suggestions)
        
        # User history-based suggestions
        user_suggestions = await self._get_user_pattern_suggestions(user_id)
        suggestions.extend(user_suggestions)
        
        # Popular patterns suggestions
        popular_suggestions = await self._get_popular_pattern_suggestions()
        suggestions.extend(popular_suggestions)
        
        # Sort by relevance
        suggestions.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        # Publish suggestion event
        if suggestions:
            event = SmartSuggestionGeneratedEvent(
                user_id=user_id,
                context=context,
                suggestions=suggestions[:10]  # Limit to top 10
            )
            await self.event_publisher.publish(event)
        
        return suggestions[:10]  # Return top 10 suggestions
    
    async def validate_hierarchy_consistency(self, hierarchy_id: str) -> Dict[str, Any]:
        """
        Validate the consistency and integrity of a settings hierarchy.
        """
        hierarchy = await self.settings_repo.get_hierarchy(hierarchy_id)
        if not hierarchy:
            return {"valid": False, "errors": ["Hierarchy not found"]}
        
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "structural_issues": [],
            "orphaned_nodes": [],
            "circular_references": [],
            "level_inconsistencies": []
        }
        
        # Use domain service for hierarchy validation
        validation_errors = self.validation_service.validate_hierarchy_integrity(hierarchy)
        
        for error in validation_errors:
            if error["type"] == "orphaned_nodes":
                result["orphaned_nodes"] = error["details"]
                result["warnings"].append(error)
            elif error["type"] == "circular_references":
                result["circular_references"] = error["details"]
                result["errors"].append(error)
                result["valid"] = False
            elif error["type"] == "level_inconsistencies":
                result["level_inconsistencies"] = error["details"]
                result["warnings"].append(error)
        
        return result
    
    def _init_smart_patterns(self) -> Dict[str, Any]:
        """Initialize smart validation patterns."""
        return {
            "api_keys": {
                "patterns": [
                    r"^sk-[a-zA-Z0-9]{48}$",  # OpenAI
                    r"^[a-zA-Z0-9-_]{40}$",   # Generic 40-char
                ],
                "suggestions": [
                    "API keys should be stored securely",
                    "Consider using environment variables",
                    "Ensure API key has appropriate permissions"
                ]
            },
            "urls": {
                "patterns": [
                    r"^https?://[^\s/$.?#].[^\s]*$"
                ],
                "suggestions": [
                    "Use HTTPS URLs for security",
                    "Validate URL accessibility",
                    "Check for proper URL encoding"
                ]
            },
            "model_names": {
                "common_values": [
                    "gpt-4", "gpt-3.5-turbo", "claude-3-sonnet",
                    "llama2", "mistral-7b", "gemini-pro"
                ],
                "suggestions": [
                    "Use well-tested model names",
                    "Check model availability",
                    "Consider model capabilities and costs"
                ]
            }
        }
    
    async def _apply_smart_patterns(self, key: str, value: Any,
                                   node: SettingsNode, hierarchy: SettingsHierarchy) -> Dict[str, Any]:
        """Apply smart validation patterns."""
        results = {"suggestions": [], "auto_fixes": []}
        
        value_str = str(value)
        
        # API key pattern
        if "api" in key.lower() and "key" in key.lower():
            patterns = self.smart_patterns["api_keys"]["patterns"]
            if not any(re.match(pattern, value_str) for pattern in patterns):
                results["suggestions"].extend(self.smart_patterns["api_keys"]["suggestions"])
        
        # URL pattern
        if "url" in key.lower() or "endpoint" in key.lower():
            pattern = self.smart_patterns["urls"]["patterns"][0]
            if not re.match(pattern, value_str):
                results["suggestions"].append("Invalid URL format")
                if value_str.startswith("http://"):
                    results["auto_fixes"].append({
                        "type": "url_https",
                        "description": "Convert HTTP to HTTPS",
                        "old_value": value_str,
                        "new_value": value_str.replace("http://", "https://", 1)
                    })
        
        # Model name suggestions
        if "model" in key.lower():
            common_models = self.smart_patterns["model_names"]["common_values"]
            if value_str not in common_models:
                results["suggestions"].append(
                    f"Consider using a common model: {', '.join(common_models[:3])}"
                )
        
        return results
    
    async def _check_setting_dependencies(self, key: str, value: Any,
                                         node: SettingsNode, hierarchy: SettingsHierarchy) -> List[Dict[str, Any]]:
        """Check for setting dependencies."""
        dependencies = []
        
        # Example: API key requires endpoint URL
        if key == "api_key" and not node.get_setting("api_url"):
            dependencies.append({
                "type": "required_setting",
                "message": "API key requires api_url to be set",
                "required_setting": "api_url"
            })
        
        # Example: Model-specific settings
        if key == "model_name" and value in ["gpt-4", "gpt-3.5-turbo"]:
            if not node.get_setting("openai_api_key"):
                dependencies.append({
                    "type": "required_setting",
                    "message": "OpenAI models require openai_api_key",
                    "required_setting": "openai_api_key"
                })
        
        return dependencies
    
    async def _check_setting_conflicts(self, key: str, value: Any,
                                      node: SettingsNode, hierarchy: SettingsHierarchy) -> List[Dict[str, Any]]:
        """Check for setting conflicts."""
        conflicts = []
        
        # Check against existing settings in node
        for existing_key, existing_setting in node.settings.items():
            if existing_key != key:
                conflict = self._detect_setting_conflict(key, value, existing_key, existing_setting.value)
                if conflict:
                    conflicts.append(conflict)
        
        return conflicts
    
    def _detect_setting_conflict(self, key1: str, value1: Any, key2: str, value2: Any) -> Optional[Dict[str, Any]]:
        """Detect conflicts between two settings."""
        # Example: conflicting provider settings
        if key1 == "provider" and key2 == "model_name":
            if value1 == "openai" and str(value2).startswith("claude"):
                return {
                    "type": "provider_model_mismatch",
                    "message": f"Provider '{value1}' conflicts with model '{value2}'",
                    "conflicting_keys": [key1, key2]
                }
        
        return None
    
    async def _generate_smart_suggestions(self, key: str, value: Any,
                                        validation_errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate smart suggestions for fixing validation errors."""
        suggestions = []
        
        for error in validation_errors:
            rule_id = error.get("rule_id", "")
            
            if "api_key_format" in rule_id:
                suggestions.append({
                    "type": "format_fix",
                    "description": "API key should contain only letters, numbers, underscores, and hyphens",
                    "example": "sk-1234567890abcdef1234567890abcdef12345678"
                })
            
            elif "timeout_range" in rule_id:
                suggestions.append({
                    "type": "range_fix",
                    "description": "Timeout should be between 1 and 300 seconds",
                    "suggested_value": max(1, min(300, int(value) if str(value).isdigit() else 30))
                })
        
        return suggestions
    
    async def _validate_cross_settings(self, settings: Dict[str, Any],
                                      node: SettingsNode, hierarchy: SettingsHierarchy) -> List[Dict[str, Any]]:
        """Validate relationships between multiple settings."""
        errors = []
        
        # Provider-model consistency
        provider = settings.get("provider")
        model = settings.get("model_name")
        
        if provider and model:
            if provider == "openai" and not str(model).startswith(("gpt-", "text-", "davinci")):
                errors.append({
                    "type": "provider_model_mismatch",
                    "message": f"Model '{model}' is not compatible with provider '{provider}'"
                })
        
        # Required combinations
        if settings.get("use_custom_endpoint") and not settings.get("custom_endpoint_url"):
            errors.append({
                "type": "missing_required_combination",
                "message": "Custom endpoint URL is required when use_custom_endpoint is true"
            })
        
        return errors
    
    async def _validate_against_schema(self, key: str, value: Any,
                                     schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a value against a schema definition."""
        result = {"valid": True, "errors": [], "warnings": []}
        
        # Type validation
        expected_type = schema.get("type")
        if expected_type and not isinstance(value, type(expected_type)):
            result["valid"] = False
            result["errors"].append(f"Expected {expected_type}, got {type(value).__name__}")
        
        # Range validation
        if "min" in schema and value < schema["min"]:
            result["valid"] = False
            result["errors"].append(f"Value must be >= {schema['min']}")
        
        if "max" in schema and value > schema["max"]:
            result["valid"] = False
            result["errors"].append(f"Value must be <= {schema['max']}")
        
        # Enum validation
        if "enum" in schema and value not in schema["enum"]:
            result["valid"] = False
            result["errors"].append(f"Value must be one of: {schema['enum']}")
        
        # Deprecation warning
        if schema.get("deprecated"):
            result["warnings"].append(f"Setting '{key}' is deprecated")
        
        return result
    
    async def _validate_agent_capabilities(self, agent: AgentDefinition,
                                         configuration: Dict[str, Any]) -> Dict[str, Any]:
        """Validate agent capabilities against configuration."""
        return {"valid": True, "errors": [], "warnings": []}
    
    async def _validate_agent_security(self, agent: AgentDefinition,
                                     configuration: Dict[str, Any]) -> Dict[str, Any]:
        """Validate agent security requirements."""
        result = {"valid": True, "errors": [], "warnings": []}
        
        # Check for sensitive data in non-encrypted fields
        for key, value in configuration.items():
            if any(sensitive in key.lower() for sensitive in ["key", "secret", "password", "token"]):
                if key not in agent.settings_schema or not agent.settings_schema[key].get("encrypted", False):
                    result["warnings"].append(f"Sensitive setting '{key}' should be encrypted")
        
        return result
    
    async def _get_hierarchy_for_node(self, node_id: str) -> Optional[SettingsHierarchy]:
        """Get the hierarchy containing a specific node."""
        # This would be implemented based on your repository design
        # Placeholder implementation
        return None
    
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
    
    async def _get_node_suggestions(self, node_id: str) -> List[Dict[str, Any]]:
        """Get suggestions specific to a node."""
        return []
    
    async def _get_agent_suggestions(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get suggestions specific to an agent."""
        return []
    
    async def _get_user_pattern_suggestions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get suggestions based on user's usage patterns."""
        return []
    
    async def _get_popular_pattern_suggestions(self) -> List[Dict[str, Any]]:
        """Get suggestions based on popular configuration patterns."""
        return []
    
    async def _generate_bulk_suggestions(self, validation_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate suggestions for bulk validation results."""
        return []