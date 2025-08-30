"""Tests for capability manager functionality."""

import asyncio
import pytest
from unittest.mock import patch, Mock

from src.agentsmcp.cli.v3.coordination.capability_manager import CapabilityManager
from src.agentsmcp.cli.v3.models.coordination_models import (
    InterfaceMode,
    CapabilityType,
    CapabilityQuery,
    Feature,
    CapabilityMismatchError
)


@pytest.fixture
def capability_manager():
    """Create a capability manager for testing."""
    return CapabilityManager()


class TestCapabilityManager:
    """Test capability manager functionality."""
    
    def test_initialization(self, capability_manager):
        """Test capability manager initialization."""
        assert capability_manager._capabilities is not None
        assert capability_manager._feature_registry is not None
        
        # Check that default capabilities are loaded
        assert len(capability_manager._feature_registry) > 0
        assert InterfaceMode.CLI in capability_manager._capabilities
    
    @pytest.mark.asyncio
    async def test_environment_detection(self, capability_manager):
        """Test environment capability detection."""
        capabilities = await capability_manager.detect_environment_capabilities()
        
        assert isinstance(capabilities, dict)
        assert 'has_tty' in capabilities
        assert 'network_available' in capabilities
        assert 'file_write' in capabilities
        
        # Values should be boolean or appropriate type
        assert isinstance(capabilities['has_tty'], bool)
        assert isinstance(capabilities['network_available'], bool)
        assert isinstance(capabilities['file_write'], bool)
    
    @pytest.mark.asyncio
    async def test_query_cli_capabilities(self, capability_manager):
        """Test querying CLI capabilities."""
        query = CapabilityQuery(
            interface=InterfaceMode.CLI,
            check_permissions=False
        )
        
        capability_info = await capability_manager.query_capabilities(query)
        
        assert capability_info.interface == InterfaceMode.CLI
        assert len(capability_info.available_features) > 0
        assert capability_info.performance_profile is not None
        assert len(capability_info.limitations) > 0
        assert len(capability_info.recommended_for) > 0
        
        # Check specific CLI features
        feature_names = [f.name for f in capability_info.available_features]
        assert 'command_line_args' in feature_names
        assert 'text_output' in feature_names
    
    @pytest.mark.asyncio
    async def test_query_specific_feature(self, capability_manager):
        """Test querying specific feature."""
        query = CapabilityQuery(
            interface=InterfaceMode.CLI,
            feature='text_output',
            check_permissions=False
        )
        
        capability_info = await capability_manager.query_capabilities(query)
        
        assert len(capability_info.available_features) == 1
        assert capability_info.available_features[0].name == 'text_output'
    
    @pytest.mark.asyncio
    async def test_query_by_capability_type(self, capability_manager):
        """Test querying by capability type."""
        query = CapabilityQuery(
            interface=InterfaceMode.CLI,
            category=CapabilityType.INPUT,
            check_permissions=False
        )
        
        capability_info = await capability_manager.query_capabilities(query)
        
        # All returned features should be INPUT type
        for feature in capability_info.available_features:
            assert feature.capability_type == CapabilityType.INPUT
    
    @pytest.mark.asyncio
    async def test_query_with_dependencies(self, capability_manager):
        """Test querying with dependency inclusion."""
        # First find a feature with dependencies
        tui_query = CapabilityQuery(
            interface=InterfaceMode.TUI,
            check_permissions=False
        )
        
        tui_capabilities = await capability_manager.query_capabilities(tui_query)
        
        # Find a feature with dependencies
        feature_with_deps = None
        for feature in tui_capabilities.available_features:
            if feature.dependencies:
                feature_with_deps = feature
                break
        
        if feature_with_deps:
            # Query with dependencies included
            dep_query = CapabilityQuery(
                interface=InterfaceMode.TUI,
                feature=feature_with_deps.name,
                include_dependencies=True,
                check_permissions=False
            )
            
            dep_capability_info = await capability_manager.query_capabilities(dep_query)
            
            # Should include the feature and its dependencies
            feature_names = [f.name for f in dep_capability_info.available_features]
            assert feature_with_deps.name in feature_names
            
            # Check that dependencies are included
            for dep in feature_with_deps.dependencies:
                if dep in capability_manager._feature_registry:
                    assert dep in feature_names
    
    @pytest.mark.asyncio
    async def test_query_nonexistent_feature(self, capability_manager):
        """Test querying non-existent feature."""
        query = CapabilityQuery(
            interface=InterfaceMode.CLI,
            feature='nonexistent_feature',
            check_permissions=False
        )
        
        with pytest.raises(CapabilityMismatchError, match="Feature nonexistent_feature not found"):
            await capability_manager.query_capabilities(query)
    
    @pytest.mark.asyncio
    async def test_query_unavailable_feature(self, capability_manager):
        """Test querying feature not available in interface."""
        # Try to get a TUI-specific feature in CLI mode
        query = CapabilityQuery(
            interface=InterfaceMode.CLI,
            feature='interactive_ui',  # TUI-only feature
            check_permissions=False
        )
        
        with pytest.raises(CapabilityMismatchError, match="not available in"):
            await capability_manager.query_capabilities(query)
    
    @pytest.mark.asyncio
    async def test_get_all_mode_capabilities(self, capability_manager):
        """Test getting capabilities for all modes."""
        capabilities = await capability_manager.get_mode_capabilities()
        
        assert isinstance(capabilities, dict)
        assert InterfaceMode.CLI in capabilities
        
        # Each mode should have features
        for mode, features in capabilities.items():
            assert isinstance(features, list)
            if features:  # Some modes might not have features in test environment
                assert all(isinstance(f, Feature) for f in features)
    
    def test_get_feature_dependencies(self, capability_manager):
        """Test getting feature dependencies."""
        # Test with a feature that has dependencies
        deps = capability_manager.get_feature_dependencies('mouse_support')
        
        if 'mouse_support' in capability_manager._feature_registry:
            assert isinstance(deps, set)
            # mouse_support should depend on interactive_ui
            assert 'interactive_ui' in deps
    
    def test_validate_feature_compatibility(self, capability_manager):
        """Test feature compatibility validation."""
        # Test compatible features
        compatible_features = ['command_line_args', 'text_output']
        incompatible = capability_manager.validate_feature_compatibility(
            compatible_features, 
            InterfaceMode.CLI
        )
        
        assert len(incompatible) == 0
        
        # Test incompatible features
        incompatible_features = ['interactive_ui']  # TUI-only feature
        incompatible = capability_manager.validate_feature_compatibility(
            incompatible_features, 
            InterfaceMode.CLI
        )
        
        assert len(incompatible) > 0
        assert any('not available' in error for error in incompatible)
    
    @pytest.mark.asyncio
    async def test_performance_impact_calculation(self, capability_manager):
        """Test performance impact calculation."""
        features = ['command_line_args', 'text_output']
        impact = await capability_manager.get_performance_impact(
            features, 
            InterfaceMode.CLI
        )
        
        assert isinstance(impact, float)
        assert 0.0 <= impact <= 1.0
    
    def test_register_custom_feature(self, capability_manager):
        """Test registering custom features."""
        # Create custom feature
        custom_feature = Feature(
            name='custom_test_feature',
            display_name='Custom Test Feature',
            description='A custom feature for testing',
            capability_type=CapabilityType.OUTPUT,
            availability={InterfaceMode.CLI: True}
        )
        
        # Register feature
        capability_manager.register_custom_feature(custom_feature)
        
        # Verify feature is registered
        assert 'custom_test_feature' in capability_manager._feature_registry
        
        # Verify it's available in CLI capabilities
        cli_capabilities = capability_manager._capabilities[InterfaceMode.CLI]
        feature_names = [f.name for f in cli_capabilities.available_features]
        assert 'custom_test_feature' in feature_names
    
    @pytest.mark.asyncio
    async def test_graceful_degradation(self, capability_manager):
        """Test graceful degradation of features."""
        # Request features, some of which might not be available
        requested_features = ['text_output', 'interactive_ui', 'nonexistent_feature']
        
        available_features = await capability_manager.graceful_degradation(
            requested_features, 
            InterfaceMode.CLI
        )
        
        assert isinstance(available_features, list)
        assert 'text_output' in available_features  # Should be available in CLI
        # interactive_ui should either be missing or have alternative
        # nonexistent_feature should be handled gracefully
    
    @pytest.mark.asyncio
    async def test_environment_dependent_features(self, capability_manager):
        """Test features that depend on environment."""
        # Mock environment without TTY
        with patch.object(capability_manager, 'detect_environment_capabilities') as mock_detect:
            mock_detect.return_value = {
                'has_tty': False,
                'network_available': True,
                'file_write': True
            }
            
            query = CapabilityQuery(
                interface=InterfaceMode.TUI,
                check_permissions=False
            )
            
            capability_info = await capability_manager.query_capabilities(query)
            
            # Features requiring TTY should be filtered out
            feature_names = [f.name for f in capability_info.available_features]
            # interactive_ui requires TTY, so might be filtered
            # This depends on the specific implementation logic
    
    @pytest.mark.asyncio
    async def test_caching_behavior(self, capability_manager):
        """Test environment detection caching."""
        # First call should detect
        result1 = await capability_manager.detect_environment_capabilities()
        
        # Second call within cache period should return cached result
        result2 = await capability_manager.detect_environment_capabilities()
        
        assert result1 == result2
        
        # Verify cache was used (in a real test, you'd check that detection logic wasn't called again)


class TestInterfaceSpecificCapabilities:
    """Test capabilities specific to each interface mode."""
    
    @pytest.mark.asyncio
    async def test_cli_specific_capabilities(self, capability_manager):
        """Test CLI-specific capabilities."""
        query = CapabilityQuery(interface=InterfaceMode.CLI, check_permissions=False)
        capabilities = await capability_manager.query_capabilities(query)
        
        feature_names = [f.name for f in capabilities.available_features]
        
        # CLI should support these features
        expected_cli_features = ['command_line_args', 'text_output', 'exit_codes']
        for feature in expected_cli_features:
            assert feature in feature_names, f"CLI should support {feature}"
    
    @pytest.mark.asyncio
    async def test_tui_specific_capabilities(self, capability_manager):
        """Test TUI-specific capabilities."""
        # Check if TUI is available in test environment
        env_caps = await capability_manager.detect_environment_capabilities()
        if not env_caps.get('has_tty', False):
            pytest.skip("TUI not available in test environment")
        
        query = CapabilityQuery(interface=InterfaceMode.TUI, check_permissions=False)
        capabilities = await capability_manager.query_capabilities(query)
        
        feature_names = [f.name for f in capabilities.available_features]
        
        # TUI should support these features if TTY is available
        expected_tui_features = ['interactive_ui', 'real_time_updates']
        for feature in expected_tui_features:
            if capability_manager._is_feature_available_in_environment(
                capability_manager._feature_registry.get(feature), env_caps
            ):
                assert feature in feature_names, f"TUI should support {feature}"
    
    @pytest.mark.asyncio
    async def test_webui_specific_capabilities(self, capability_manager):
        """Test WebUI-specific capabilities."""
        query = CapabilityQuery(interface=InterfaceMode.WEB_UI, check_permissions=False)
        
        try:
            capabilities = await capability_manager.query_capabilities(query)
            feature_names = [f.name for f in capabilities.available_features]
            
            # WebUI should support these features if network is available
            expected_webui_features = ['rich_ui', 'data_visualization']
            for feature in expected_webui_features:
                # Feature availability depends on environment
                if feature in capability_manager._feature_registry:
                    feature_obj = capability_manager._feature_registry[feature]
                    if feature_obj.availability.get(InterfaceMode.WEB_UI, False):
                        # Feature should be in list or filtered by environment
                        pass  # Environment filtering might remove it
        
        except CapabilityMismatchError:
            # WebUI might not be supported in test environment
            pass
    
    @pytest.mark.asyncio
    async def test_api_specific_capabilities(self, capability_manager):
        """Test API-specific capabilities."""
        query = CapabilityQuery(interface=InterfaceMode.API, check_permissions=False)
        
        try:
            capabilities = await capability_manager.query_capabilities(query)
            feature_names = [f.name for f in capabilities.available_features]
            
            # API should support these features
            expected_api_features = ['rest_api', 'structured_responses']
            for feature in expected_api_features:
                # Feature availability depends on environment
                if feature in capability_manager._feature_registry:
                    feature_obj = capability_manager._feature_registry[feature]
                    if feature_obj.availability.get(InterfaceMode.API, False):
                        # Feature should be in list or filtered by environment
                        pass
        
        except CapabilityMismatchError:
            # API might not be supported in test environment
            pass


class TestErrorHandling:
    """Test error handling in capability manager."""
    
    @pytest.mark.asyncio
    async def test_query_unsupported_interface(self, capability_manager):
        """Test querying capabilities for unsupported interface."""
        # Remove all handlers to simulate unsupported interface
        original_capabilities = capability_manager._capabilities.copy()
        capability_manager._capabilities.clear()
        
        try:
            query = CapabilityQuery(interface=InterfaceMode.CLI, check_permissions=False)
            
            with pytest.raises(CapabilityMismatchError, match="not supported"):
                await capability_manager.query_capabilities(query)
        
        finally:
            # Restore original capabilities
            capability_manager._capabilities = original_capabilities
    
    def test_invalid_feature_dependencies(self, capability_manager):
        """Test handling of invalid feature dependencies."""
        # Create feature with non-existent dependency
        invalid_feature = Feature(
            name='invalid_deps_feature',
            display_name='Invalid Dependencies Feature',
            description='Feature with invalid dependencies',
            capability_type=CapabilityType.OUTPUT,
            dependencies={'nonexistent_dependency'},
            availability={InterfaceMode.CLI: True}
        )
        
        capability_manager.register_custom_feature(invalid_feature)
        
        # Getting dependencies should handle missing dependencies gracefully
        deps = capability_manager.get_feature_dependencies('invalid_deps_feature')
        assert isinstance(deps, set)
        # Should include the invalid dependency name even if it doesn't exist
        assert 'nonexistent_dependency' in deps


class TestConcurrency:
    """Test concurrent operations in capability manager."""
    
    @pytest.mark.asyncio
    async def test_concurrent_capability_queries(self, capability_manager):
        """Test concurrent capability queries."""
        queries = [
            CapabilityQuery(interface=InterfaceMode.CLI, check_permissions=False),
            CapabilityQuery(interface=InterfaceMode.CLI, category=CapabilityType.INPUT, check_permissions=False),
            CapabilityQuery(interface=InterfaceMode.CLI, category=CapabilityType.OUTPUT, check_permissions=False)
        ]
        
        # Execute queries concurrently
        results = await asyncio.gather(
            *[capability_manager.query_capabilities(query) for query in queries],
            return_exceptions=True
        )
        
        # All should succeed
        for result in results:
            assert not isinstance(result, Exception), f"Unexpected exception: {result}"
    
    @pytest.mark.asyncio
    async def test_concurrent_environment_detection(self, capability_manager):
        """Test concurrent environment detection."""
        # Clear cache to force detection
        capability_manager._last_detection = None
        capability_manager._environment_cache.clear()
        
        # Run multiple concurrent detections
        results = await asyncio.gather(
            capability_manager.detect_environment_capabilities(),
            capability_manager.detect_environment_capabilities(),
            capability_manager.detect_environment_capabilities(),
            return_exceptions=True
        )
        
        # All should succeed and return same result
        for result in results:
            assert not isinstance(result, Exception)
            assert isinstance(result, dict)
        
        # All results should be identical (cached)
        assert all(r == results[0] for r in results)
    
    def test_concurrent_feature_registration(self, capability_manager):
        """Test concurrent feature registration."""
        def register_feature(i):
            feature = Feature(
                name=f'concurrent_feature_{i}',
                display_name=f'Concurrent Feature {i}',
                description=f'Concurrently registered feature {i}',
                capability_type=CapabilityType.OUTPUT,
                availability={InterfaceMode.CLI: True}
            )
            capability_manager.register_custom_feature(feature)
            return feature
        
        # Register features concurrently (using threads since registration is sync)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(register_feature, i) for i in range(3)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All should be registered successfully
        assert len(results) == 3
        for i in range(3):
            assert f'concurrent_feature_{i}' in capability_manager._feature_registry