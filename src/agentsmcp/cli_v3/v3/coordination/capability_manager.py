"""Capability manager for cross-modal interface coordination.

This module manages interface capabilities, feature availability matrices,
and dynamic capability adjustment based on environment conditions.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set
from datetime import datetime, timezone

from ..models.coordination_models import (
    InterfaceMode,
    CapabilityType, 
    Feature,
    CapabilityQuery,
    CapabilityInfo,
    CapabilityMismatchError
)

logger = logging.getLogger(__name__)


class CapabilityManager:
    """Manages interface capabilities and feature availability."""
    
    def __init__(self):
        """Initialize capability manager."""
        self._capabilities: Dict[InterfaceMode, CapabilityInfo] = {}
        self._feature_registry: Dict[str, Feature] = {}
        self._environment_cache: Dict[str, any] = {}
        self._last_detection: Optional[datetime] = None
        self._detection_lock = asyncio.Lock()
        
        # Initialize default capabilities
        self._initialize_default_capabilities()
    
    def _initialize_default_capabilities(self) -> None:
        """Initialize default capability definitions."""
        # CLI capabilities
        cli_features = [
            Feature(
                name="command_line_args",
                display_name="Command Line Arguments",
                description="Process command line arguments and flags",
                capability_type=CapabilityType.INPUT,
                performance_impact=0.0,
                availability={InterfaceMode.CLI: True}
            ),
            Feature(
                name="text_output",
                display_name="Text Output",
                description="Display structured text output",
                capability_type=CapabilityType.OUTPUT,
                performance_impact=0.1,
                availability={InterfaceMode.CLI: True}
            ),
            Feature(
                name="batch_processing",
                display_name="Batch Processing",
                description="Execute multiple commands in sequence",
                capability_type=CapabilityType.INTERACTION,
                performance_impact=0.2,
                availability={InterfaceMode.CLI: True, InterfaceMode.API: True}
            ),
            Feature(
                name="exit_codes",
                display_name="Exit Codes",
                description="Return structured exit codes for automation",
                capability_type=CapabilityType.OUTPUT,
                performance_impact=0.0,
                availability={InterfaceMode.CLI: True, InterfaceMode.API: True}
            )
        ]
        
        # TUI capabilities
        tui_features = [
            Feature(
                name="interactive_ui",
                display_name="Interactive UI",
                description="Real-time interactive terminal interface",
                capability_type=CapabilityType.INTERACTION,
                required_permissions={"tty_access"},
                performance_impact=0.3,
                availability={InterfaceMode.TUI: True}
            ),
            Feature(
                name="keyboard_shortcuts",
                display_name="Keyboard Shortcuts",
                description="Keyboard-based navigation and commands",
                capability_type=CapabilityType.INPUT,
                dependencies={"interactive_ui"},
                performance_impact=0.1,
                availability={InterfaceMode.TUI: True}
            ),
            Feature(
                name="mouse_support",
                display_name="Mouse Support", 
                description="Mouse-based interaction in terminal",
                capability_type=CapabilityType.INPUT,
                dependencies={"interactive_ui"},
                performance_impact=0.2,
                availability={InterfaceMode.TUI: True}
            ),
            Feature(
                name="real_time_updates",
                display_name="Real-time Updates",
                description="Live updating display components",
                capability_type=CapabilityType.OUTPUT,
                dependencies={"interactive_ui"},
                performance_impact=0.4,
                availability={InterfaceMode.TUI: True, InterfaceMode.WEB_UI: True}
            ),
            Feature(
                name="progress_indicators",
                display_name="Progress Indicators",
                description="Visual progress bars and spinners",
                capability_type=CapabilityType.OUTPUT,
                performance_impact=0.2,
                availability={InterfaceMode.TUI: True, InterfaceMode.WEB_UI: True}
            )
        ]
        
        # WebUI capabilities  
        webui_features = [
            Feature(
                name="rich_ui",
                display_name="Rich Web UI",
                description="Full-featured web interface with rich components",
                capability_type=CapabilityType.INTERACTION,
                required_permissions={"web_server"},
                performance_impact=0.5,
                availability={InterfaceMode.WEB_UI: True}
            ),
            Feature(
                name="file_uploads",
                display_name="File Uploads", 
                description="Upload files through web interface",
                capability_type=CapabilityType.INPUT,
                dependencies={"rich_ui"},
                required_permissions={"file_write"},
                performance_impact=0.6,
                availability={InterfaceMode.WEB_UI: True}
            ),
            Feature(
                name="collaborative_editing",
                display_name="Collaborative Editing",
                description="Multiple users editing simultaneously",
                capability_type=CapabilityType.INTERACTION,
                dependencies={"rich_ui"},
                required_permissions={"multi_user"},
                performance_impact=0.8,
                availability={InterfaceMode.WEB_UI: True}
            ),
            Feature(
                name="data_visualization",
                display_name="Data Visualization",
                description="Interactive charts and graphs",
                capability_type=CapabilityType.OUTPUT,
                dependencies={"rich_ui"},
                performance_impact=0.7,
                availability={InterfaceMode.WEB_UI: True}
            )
        ]
        
        # API capabilities
        api_features = [
            Feature(
                name="rest_api",
                display_name="REST API",
                description="RESTful HTTP API endpoints",
                capability_type=CapabilityType.INTERACTION,
                required_permissions={"api_server"},
                performance_impact=0.3,
                availability={InterfaceMode.API: True}
            ),
            Feature(
                name="webhooks",
                display_name="Webhooks",
                description="HTTP callback notifications",
                capability_type=CapabilityType.OUTPUT,
                dependencies={"rest_api"},
                required_permissions={"network_outbound"},
                performance_impact=0.2,
                availability={InterfaceMode.API: True}
            ),
            Feature(
                name="bulk_operations",
                display_name="Bulk Operations",
                description="Process multiple operations in single request",
                capability_type=CapabilityType.INTERACTION,
                dependencies={"rest_api"},
                performance_impact=0.4,
                availability={InterfaceMode.API: True}
            ),
            Feature(
                name="structured_responses",
                display_name="Structured Responses",
                description="JSON/XML structured response formats",
                capability_type=CapabilityType.OUTPUT,
                performance_impact=0.1,
                availability={InterfaceMode.API: True, InterfaceMode.CLI: True}
            )
        ]
        
        # Register all features
        all_features = cli_features + tui_features + webui_features + api_features
        for feature in all_features:
            self._feature_registry[feature.name] = feature
        
        # Build capability info for each interface
        for mode in InterfaceMode:
            available_features = [f for f in all_features if f.availability.get(mode, False)]
            
            # Group capabilities by type
            capabilities_by_type = {}
            for cap_type in CapabilityType:
                capabilities_by_type[cap_type] = [
                    f.name for f in available_features 
                    if f.capability_type == cap_type
                ]
            
            # Performance profile
            performance_profile = {
                "average_impact": sum(f.performance_impact for f in available_features) / len(available_features) if available_features else 0.0,
                "max_impact": max(f.performance_impact for f in available_features) if available_features else 0.0,
                "feature_count": len(available_features)
            }
            
            # Limitations and recommendations
            limitations = []
            recommended_for = []
            
            if mode == InterfaceMode.CLI:
                limitations = ["No real-time interaction", "Text-only output", "Single command execution"]
                recommended_for = ["Automation scripts", "CI/CD pipelines", "Quick commands"]
            elif mode == InterfaceMode.TUI:
                limitations = ["Terminal dependency", "Limited graphics", "Local access only"]
                recommended_for = ["Interactive workflows", "Real-time monitoring", "Power users"]
            elif mode == InterfaceMode.WEB_UI:
                limitations = ["Network dependency", "Browser requirement", "Higher resource usage"]
                recommended_for = ["Rich visualizations", "Collaboration", "Complex workflows"]
            elif mode == InterfaceMode.API:
                limitations = ["Programmatic access only", "No user interface", "Network dependency"]
                recommended_for = ["System integration", "Automation", "Third-party applications"]
            
            self._capabilities[mode] = CapabilityInfo(
                interface=mode,
                available_features=available_features,
                supported_capabilities=capabilities_by_type,
                performance_profile=performance_profile,
                limitations=limitations,
                recommended_for=recommended_for
            )
    
    async def detect_environment_capabilities(self) -> Dict[str, any]:
        """Detect available environment capabilities."""
        async with self._detection_lock:
            # Cache detection for 5 minutes
            now = datetime.now(timezone.utc)
            if (self._last_detection and 
                (now - self._last_detection).total_seconds() < 300):
                return self._environment_cache.copy()
            
            logger.debug("Detecting environment capabilities")
            capabilities = {}
            
            # Check TTY availability for TUI
            try:
                import sys
                capabilities["has_tty"] = sys.stdout.isatty() and sys.stdin.isatty()
            except Exception:
                capabilities["has_tty"] = False
            
            # Check terminal capabilities
            try:
                import os
                capabilities["term_type"] = os.environ.get("TERM", "unknown")
                capabilities["term_colors"] = os.environ.get("COLORTERM") is not None
                capabilities["term_size"] = None
                if capabilities["has_tty"]:
                    capabilities["term_size"] = os.get_terminal_size()
            except Exception:
                capabilities["term_colors"] = False
                capabilities["term_size"] = None
            
            # Check network capabilities
            try:
                import socket
                # Test basic network connectivity
                with socket.create_connection(("8.8.8.8", 53), timeout=1):
                    capabilities["network_available"] = True
            except Exception:
                capabilities["network_available"] = False
            
            # Check file system permissions
            try:
                import tempfile
                with tempfile.NamedTemporaryFile() as f:
                    capabilities["file_write"] = True
            except Exception:
                capabilities["file_write"] = False
            
            # Check if running in container
            try:
                with open("/proc/1/cgroup", "r") as f:
                    cgroup_content = f.read()
                    capabilities["in_container"] = "docker" in cgroup_content or "containerd" in cgroup_content
            except Exception:
                capabilities["in_container"] = False
            
            # Check available memory
            try:
                import psutil
                memory = psutil.virtual_memory()
                capabilities["available_memory_mb"] = memory.available // (1024 * 1024)
                capabilities["cpu_count"] = psutil.cpu_count()
            except Exception:
                capabilities["available_memory_mb"] = None
                capabilities["cpu_count"] = None
            
            self._environment_cache = capabilities
            self._last_detection = now
            
            logger.info(f"Environment capabilities detected: {capabilities}")
            return capabilities.copy()
    
    async def query_capabilities(self, query: CapabilityQuery) -> CapabilityInfo:
        """Query interface capabilities."""
        logger.debug(f"Querying capabilities for {query.interface}")
        
        # Get base capability info
        if query.interface not in self._capabilities:
            raise CapabilityMismatchError(f"Interface mode {query.interface} not supported")
        
        capability_info = self._capabilities[query.interface].model_copy(deep=True)
        
        # Filter by specific feature if requested
        if query.feature:
            if query.feature not in self._feature_registry:
                raise CapabilityMismatchError(f"Feature {query.feature} not found")
            
            feature = self._feature_registry[query.feature]
            if not feature.availability.get(query.interface, False):
                raise CapabilityMismatchError(
                    f"Feature {query.feature} not available in {query.interface}"
                )
            
            capability_info.available_features = [feature]
        
        # Filter by capability category
        if query.category:
            capability_info.available_features = [
                f for f in capability_info.available_features
                if f.capability_type == query.category
            ]
        
        # Include dependencies if requested
        if query.include_dependencies:
            features_with_deps = set()
            for feature in capability_info.available_features:
                features_with_deps.add(feature.name)
                features_with_deps.update(feature.dependencies)
            
            capability_info.available_features = [
                f for f in self._feature_registry.values()
                if f.name in features_with_deps and 
                f.availability.get(query.interface, False)
            ]
        
        # Check permissions if requested
        if query.check_permissions and query.user_id:
            # In a real implementation, this would check user permissions
            # For now, we'll assume all permissions are granted
            pass
        
        # Update with current environment capabilities
        env_capabilities = await self.detect_environment_capabilities()
        
        # Filter features based on environment
        available_features = []
        for feature in capability_info.available_features:
            if self._is_feature_available_in_environment(feature, env_capabilities):
                available_features.append(feature)
        
        capability_info.available_features = available_features
        capability_info.last_updated = datetime.now(timezone.utc)
        
        return capability_info
    
    def _is_feature_available_in_environment(
        self, 
        feature: Feature, 
        env_capabilities: Dict[str, any]
    ) -> bool:
        """Check if feature is available in current environment."""
        # Check basic requirements
        if feature.capability_type == CapabilityType.INTERACTION:
            if feature.name == "interactive_ui" and not env_capabilities.get("has_tty", False):
                return False
            if feature.name == "rich_ui" and not env_capabilities.get("network_available", False):
                return False
        
        # Check permissions (simplified check)
        if "file_write" in feature.required_permissions:
            if not env_capabilities.get("file_write", False):
                return False
        
        if "network_outbound" in feature.required_permissions:
            if not env_capabilities.get("network_available", False):
                return False
        
        return True
    
    async def get_mode_capabilities(self) -> Dict[InterfaceMode, List[Feature]]:
        """Get capability matrix for all interface modes."""
        result = {}
        for mode in InterfaceMode:
            query = CapabilityQuery(interface=mode, check_permissions=False)
            capability_info = await self.query_capabilities(query)
            result[mode] = capability_info.available_features
        
        return result
    
    def get_feature_dependencies(self, feature_name: str) -> Set[str]:
        """Get all dependencies for a feature."""
        if feature_name not in self._feature_registry:
            return set()
        
        feature = self._feature_registry[feature_name]
        dependencies = set(feature.dependencies)
        
        # Recursively get dependencies of dependencies
        for dep in list(dependencies):
            dependencies.update(self.get_feature_dependencies(dep))
        
        return dependencies
    
    def validate_feature_compatibility(
        self, 
        features: List[str], 
        interface: InterfaceMode
    ) -> List[str]:
        """Validate that features are compatible with interface mode."""
        incompatible = []
        
        for feature_name in features:
            if feature_name not in self._feature_registry:
                incompatible.append(f"Unknown feature: {feature_name}")
                continue
            
            feature = self._feature_registry[feature_name]
            if not feature.availability.get(interface, False):
                incompatible.append(
                    f"Feature {feature_name} not available in {interface}"
                )
        
        return incompatible
    
    async def get_performance_impact(
        self, 
        features: List[str], 
        interface: InterfaceMode
    ) -> float:
        """Calculate total performance impact for feature set."""
        total_impact = 0.0
        feature_count = 0
        
        for feature_name in features:
            if feature_name in self._feature_registry:
                feature = self._feature_registry[feature_name]
                if feature.availability.get(interface, False):
                    total_impact += feature.performance_impact
                    feature_count += 1
        
        # Apply diminishing returns for multiple features
        if feature_count > 1:
            total_impact *= (1.0 - 0.1 * (feature_count - 1))
        
        return min(total_impact, 1.0)
    
    def register_custom_feature(self, feature: Feature) -> None:
        """Register a custom feature."""
        logger.info(f"Registering custom feature: {feature.name}")
        self._feature_registry[feature.name] = feature
        
        # Update capability info for affected interfaces
        for interface, available in feature.availability.items():
            if available and interface in self._capabilities:
                self._capabilities[interface].available_features.append(feature)
                
                # Update capability grouping
                cap_type = feature.capability_type
                if cap_type not in self._capabilities[interface].supported_capabilities:
                    self._capabilities[interface].supported_capabilities[cap_type] = []
                self._capabilities[interface].supported_capabilities[cap_type].append(feature.name)
    
    async def graceful_degradation(
        self, 
        requested_features: List[str],
        target_interface: InterfaceMode
    ) -> List[str]:
        """Find alternative features when requested features are unavailable."""
        available_features = []
        
        # Get current environment capabilities
        env_capabilities = await self.detect_environment_capabilities()
        
        for feature_name in requested_features:
            if feature_name not in self._feature_registry:
                logger.warning(f"Unknown feature requested: {feature_name}")
                continue
            
            feature = self._feature_registry[feature_name]
            
            # Check if feature is available
            if (feature.availability.get(target_interface, False) and
                self._is_feature_available_in_environment(feature, env_capabilities)):
                available_features.append(feature_name)
            else:
                # Look for alternative features
                alternative = self._find_alternative_feature(feature, target_interface, env_capabilities)
                if alternative:
                    available_features.append(alternative)
                    logger.info(f"Using {alternative} as alternative to {feature_name}")
        
        return available_features
    
    def _find_alternative_feature(
        self, 
        original_feature: Feature, 
        interface: InterfaceMode,
        env_capabilities: Dict[str, any]
    ) -> Optional[str]:
        """Find alternative feature with similar capability."""
        # Simple heuristic: find features of same type that are available
        for feature_name, feature in self._feature_registry.items():
            if (feature.capability_type == original_feature.capability_type and
                feature.availability.get(interface, False) and
                self._is_feature_available_in_environment(feature, env_capabilities)):
                return feature_name
        
        return None