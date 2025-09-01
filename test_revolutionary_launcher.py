#!/usr/bin/env python3
"""
Test script for Revolutionary TUI Launcher system

This script tests the revolutionary launcher components to ensure they work
correctly with capability detection and graceful fallbacks.
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add the AgentsMCP source directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agentsmcp.ui.v2.revolutionary_launcher import RevolutionaryLauncher, launch_revolutionary_tui
from agentsmcp.ui.v2.feature_activation_manager import FeatureActivationManager, FeatureLevel
from agentsmcp.ui.v2.tui_entry_point_adapter import TUIEntryPointAdapter, launch_adaptive_tui
from agentsmcp.ui.cli_app import CLIConfig

# Configure logging for testing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_feature_activation_manager():
    """Test the Feature Activation Manager."""
    print("🔍 Testing Feature Activation Manager...")
    
    manager = FeatureActivationManager()
    
    # Test initialization
    init_result = await manager.initialize()
    print(f"   ✓ Initialization: {'✅ Success' if init_result else '❌ Failed'}")
    
    if init_result:
        # Test capability detection
        capabilities = await manager.detect_capabilities()
        print(f"   ✓ Capabilities detected: {len(capabilities)} items")
        
        # Print key capabilities
        terminal_type = capabilities.get('terminal_type', 'unknown')
        performance_tier = capabilities.get('performance_tier', 'unknown')
        colors = capabilities.get('colors', 0)
        
        print(f"     • Terminal: {terminal_type}")
        print(f"     • Performance: {performance_tier}")
        print(f"     • Colors: {colors}")
        
        # Test feature level determination
        feature_level = manager.determine_feature_level(capabilities)
        print(f"   ✓ Determined feature level: {feature_level.name}")
        
        # Test recommendations
        recommendations = manager.get_feature_recommendations(feature_level)
        print(f"   ✓ Feature recommendations: {len(recommendations)} features")
        
        await manager.cleanup()
    
    print("   🎯 Feature Activation Manager test complete\n")
    return init_result


async def test_tui_entry_point_adapter():
    """Test the TUI Entry Point Adapter."""
    print("🔗 Testing TUI Entry Point Adapter...")
    
    adapter = TUIEntryPointAdapter()
    
    # Test environment validation
    validation = await adapter.validate_tui_environment()
    print(f"   ✓ Environment validation: {'✅ Valid' if validation['valid'] else '❌ Invalid'}")
    
    if validation['warnings']:
        print(f"     • Warnings: {len(validation['warnings'])}")
        for warning in validation['warnings']:
            print(f"       - {warning}")
    
    if validation['errors']:
        print(f"     • Errors: {len(validation['errors'])}")
        for error in validation['errors']:
            print(f"       - {error}")
    
    # Test launch recommendations
    recommendations = adapter.get_launch_recommendations()
    print(f"   ✓ Launch recommendations:")
    print(f"     • Recommended method: {recommendations['recommended_method']}")
    print(f"     • Revolutionary available: {recommendations['revolutionary_available']}")
    print(f"     • Fallback available: {recommendations['fallback_available']}")
    
    if recommendations['compatibility_notes']:
        print(f"     • Compatibility notes:")
        for note in recommendations['compatibility_notes']:
            print(f"       - {note}")
    
    # Test environment configuration
    adapter.configure_environment_for_mode('basic')
    print("   ✓ Environment configured for basic mode")
    
    await adapter.cleanup()
    print("   🎯 TUI Entry Point Adapter test complete\n")
    return validation['valid']


async def test_revolutionary_launcher():
    """Test the Revolutionary Launcher."""
    print("🚀 Testing Revolutionary Launcher...")
    
    # Create test CLI config
    cli_config = CLIConfig()
    launcher = RevolutionaryLauncher(cli_config)
    
    # Test feature detection initialization
    init_result = await launcher._initialize_feature_detection()
    print(f"   ✓ Feature detection init: {'✅ Success' if init_result else '❌ Failed'}")
    
    if init_result:
        # Test feature level determination
        feature_level = await launcher._determine_feature_level()
        print(f"   ✓ Feature level determined: {feature_level.name}")
        
        # Test capability summary logging
        if launcher.feature_manager._capabilities_cache:
            capabilities = launcher.feature_manager._capabilities_cache
            launcher._log_capability_summary(capabilities, feature_level)
            print("   ✓ Capability summary logged")
    
    await launcher._cleanup()
    print("   🎯 Revolutionary Launcher test complete\n")
    return init_result


async def test_launch_functions():
    """Test the convenience launch functions."""
    print("🎪 Testing Launch Functions...")
    
    # Test without actually launching (would start interactive TUI)
    print("   ℹ️  Launch functions available:")
    print("     • launch_revolutionary_tui()")
    print("     • launch_adaptive_tui()")
    print("   ✓ Launch functions imported successfully")
    
    # Test CLI config creation
    cli_config = CLIConfig()
    print(f"   ✓ CLI config created: {type(cli_config).__name__}")
    
    print("   🎯 Launch Functions test complete\n")
    return True


async def main():
    """Run all tests."""
    print("🧪 Revolutionary TUI Launcher Test Suite")
    print("=" * 50)
    
    results = []
    
    # Test each component
    try:
        results.append(await test_feature_activation_manager())
        results.append(await test_tui_entry_point_adapter())
        results.append(await test_revolutionary_launcher())
        results.append(await test_launch_functions())
        
    except Exception as e:
        logger.exception(f"Test suite failed: {e}")
        results.append(False)
    
    # Summary
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Revolutionary TUI Launcher is ready.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))