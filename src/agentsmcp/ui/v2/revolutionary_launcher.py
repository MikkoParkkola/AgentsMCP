"""
Revolutionary TUI Launcher - Main entry point for feature-rich TUI experience

This module provides the main launcher that conditionally loads revolutionary TUI components,
maintaining 100% backward compatibility while providing enhanced features when available.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from .feature_activation_manager import FeatureActivationManager, FeatureLevel
except ImportError:
    # Fallback classes if feature manager not available
    from enum import Enum
    
    class FeatureLevel(Enum):
        BASIC = "basic"
        ENHANCED = "enhanced"
        REVOLUTIONARY = "revolutionary"
        ULTRA = "ultra"
    
    class FeatureActivationManager:
        async def initialize(self):
            return True
        
        async def detect_capabilities(self):
            return {"terminal_type": "basic", "performance_tier": "medium", "colors": 256}
        
        def determine_feature_level(self, capabilities):
            return FeatureLevel.ENHANCED

try:
    from .tui_entry_point_adapter import TUIEntryPointAdapter
except ImportError:
    class TUIEntryPointAdapter:
        async def launch_basic_tui(self, cli_config):
            from .fixed_working_tui import launch_fixed_working_tui
            return await launch_fixed_working_tui()

from ..cli_app import CLIConfig

logger = logging.getLogger(__name__)


class RevolutionaryLauncher:
    """
    Main revolutionary TUI launcher providing progressive enhancement.
    
    This launcher:
    - Detects terminal capabilities and system performance
    - Activates appropriate feature level (Basic → Enhanced → Revolutionary → Ultra)
    - Provides graceful fallback to basic TUI on any failures
    - Maintains complete backward compatibility
    """
    
    def __init__(self, cli_config: Optional[CLIConfig] = None):
        """Initialize the revolutionary launcher.
        
        Args:
            cli_config: CLI configuration, uses defaults if None
        """
        self.cli_config = cli_config or CLIConfig()
        self.feature_manager = FeatureActivationManager()
        self.entry_adapter = TUIEntryPointAdapter()
        self.initialized = False
        self._shutdown_event = asyncio.Event()
        
        # Component references for cleanup
        self._active_components = []
        
        logger.debug("Revolutionary launcher initialized")
    
    async def launch(self) -> int:
        """
        Launch the revolutionary TUI with automatic capability detection.
        
        Returns:
            Exit code (0 for success, 1 for error)
        """
        try:
            logger.info("🚀 Starting Revolutionary TUI Launcher")
            
            # Phase 1: Initialize feature detection
            if not await self._initialize_feature_detection():
                logger.warning("Feature detection failed, falling back to basic TUI")
                return await self._launch_fallback_tui()
            
            # Phase 2: Determine optimal feature level
            feature_level = await self._determine_feature_level()
            logger.info(f"Determined feature level: {feature_level.name}")
            
            # Phase 3: Launch appropriate TUI implementation
            return await self._launch_tui_for_level(feature_level)
            
        except Exception as e:
            logger.exception(f"Revolutionary launcher failed: {e}")
            # Always fallback gracefully
            return await self._launch_fallback_tui()
        finally:
            await self._cleanup()
    
    async def _initialize_feature_detection(self) -> bool:
        """Initialize the feature activation manager.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Timeout feature detection to prevent hanging
            await asyncio.wait_for(
                self.feature_manager.initialize(),
                timeout=2.0
            )
            self.initialized = True
            return True
            
        except asyncio.TimeoutError:
            logger.warning("Feature detection timed out")
            return False
        except Exception as e:
            logger.warning(f"Feature detection initialization failed: {e}")
            return False
    
    async def _determine_feature_level(self) -> FeatureLevel:
        """Determine the appropriate feature level based on capabilities.
        
        Returns:
            The optimal feature level for this environment
        """
        if not self.initialized:
            return FeatureLevel.BASIC
        
        try:
            # Get capabilities with timeout
            capabilities = await asyncio.wait_for(
                self.feature_manager.detect_capabilities(),
                timeout=1.0
            )
            
            # Determine feature level based on capabilities
            feature_level = self.feature_manager.determine_feature_level(capabilities)
            
            # Log capability summary for debugging
            self._log_capability_summary(capabilities, feature_level)
            
            return feature_level
            
        except asyncio.TimeoutError:
            logger.warning("Capability detection timed out, using basic level")
            return FeatureLevel.BASIC
        except Exception as e:
            logger.warning(f"Capability detection failed: {e}, using basic level")
            return FeatureLevel.BASIC
    
    def _log_capability_summary(self, capabilities: Dict[str, Any], level: FeatureLevel):
        """Log a summary of detected capabilities."""
        try:
            terminal_type = capabilities.get('terminal_type', 'unknown')
            colors = capabilities.get('colors', 0)
            performance_tier = capabilities.get('performance_tier', 'unknown')
            
            logger.info(f"Capabilities detected:")
            logger.info(f"  Terminal: {terminal_type}")
            logger.info(f"  Colors: {colors}")
            logger.info(f"  Performance: {performance_tier}")
            logger.info(f"  Selected level: {level.name}")
            
        except Exception:
            # Don't fail on logging issues
            pass
    
    async def _launch_tui_for_level(self, level: FeatureLevel) -> int:
        """Launch the appropriate TUI implementation for the given feature level.
        
        Args:
            level: The feature level to implement
            
        Returns:
            Exit code
        """
        try:
            if level == FeatureLevel.ULTRA:
                return await self._launch_ultra_tui()
            elif level == FeatureLevel.REVOLUTIONARY:
                return await self._launch_revolutionary_tui()
            elif level == FeatureLevel.ENHANCED:
                return await self._launch_enhanced_tui()
            else:
                return await self._launch_basic_tui()
                
        except Exception as e:
            logger.warning(f"Failed to launch {level.name} TUI: {e}")
            # Cascade down to next level
            if level != FeatureLevel.BASIC:
                return await self._launch_tui_for_level(FeatureLevel.BASIC)
            else:
                return await self._launch_fallback_tui()
    
    async def _launch_ultra_tui(self) -> int:
        """Launch Ultra-level TUI with all revolutionary features."""
        logger.info("🌟 Launching Ultra TUI with full revolutionary features")
        
        try:
            # Try to load revolutionary integration layer
            try:
                # Try full integration layer first
                try:
                    from ..components.revolutionary_integration_layer import RevolutionaryIntegrationLayer
                except ImportError:
                    # Use simplified version if full version not available
                    from ..components.revolutionary_integration_layer_simple import RevolutionaryIntegrationLayer
                
                # Try to load TUI enhancements
                tui_enhancements = None
                try:
                    from ..components.revolutionary_tui_enhancements import RevolutionaryTUIEnhancements
                    tui_enhancements = RevolutionaryTUIEnhancements()
                    await asyncio.wait_for(tui_enhancements.initialize(), timeout=3.0)
                    self._active_components.append(tui_enhancements)
                except (ImportError, asyncio.TimeoutError) as e:
                    logger.debug(f"TUI enhancements not available: {e}")
                
                # Create a basic event system for components that need it
                try:
                    from ..v2.event_system import AsyncEventSystem
                    event_system = AsyncEventSystem()
                except ImportError:
                    event_system = None
                
                # Initialize integration layer
                integration_layer = RevolutionaryIntegrationLayer(event_system)
                await asyncio.wait_for(integration_layer.initialize(), timeout=2.0)
                self._active_components.append(integration_layer)
                
                # Launch enhanced main app with revolutionary components
                from .main_app import MainTUIApp
                app = MainTUIApp(self.cli_config)
                
                # Inject revolutionary components if app supports them
                if hasattr(app, 'integration_layer'):
                    app.integration_layer = integration_layer
                if tui_enhancements and hasattr(app, 'tui_enhancements'):
                    app.tui_enhancements = tui_enhancements
                
                return await app.run()
                
            except (ImportError, asyncio.TimeoutError) as e:
                # Revolutionary components not available, use enhanced mode
                logger.info(f"Revolutionary components not available ({e}), using enhanced TUI")
                return await self._launch_enhanced_tui()
            
        except asyncio.TimeoutError:
            logger.warning("Revolutionary component initialization timed out")
            return await self._launch_enhanced_tui()
        except Exception as e:
            logger.warning(f"Ultra TUI launch failed: {e}")
            return await self._launch_enhanced_tui()
    
    async def _launch_revolutionary_tui(self) -> int:
        """Launch Revolutionary-level TUI with core enhancements."""
        logger.info("🔥 Launching Revolutionary TUI with core enhancements")
        
        try:
            # Try to load partial revolutionary features
            try:
                tui_enhancements = None
                try:
                    from ..components.revolutionary_tui_enhancements import RevolutionaryTUIEnhancements
                    tui_enhancements = RevolutionaryTUIEnhancements()
                    await asyncio.wait_for(tui_enhancements.initialize(), timeout=2.0)
                    self._active_components.append(tui_enhancements)
                except (ImportError, asyncio.TimeoutError) as e:
                    logger.debug(f"TUI enhancements not available: {e}")
                
                # Launch main app with or without enhancements
                from .main_app import MainTUIApp
                app = MainTUIApp(self.cli_config)
                if tui_enhancements and hasattr(app, 'tui_enhancements'):
                    app.tui_enhancements = tui_enhancements
                
                return await app.run()
                
            except ImportError:
                logger.info("Revolutionary components not available, using enhanced TUI")
                return await self._launch_enhanced_tui()
            
        except asyncio.TimeoutError:
            logger.warning("Revolutionary enhancement initialization timed out")
            return await self._launch_enhanced_tui()
        except Exception as e:
            logger.warning(f"Revolutionary TUI launch failed: {e}")
            return await self._launch_enhanced_tui()
    
    async def _launch_enhanced_tui(self) -> int:
        """Launch Enhanced-level TUI with improved features."""
        logger.info("✨ Launching Enhanced TUI with improved features")
        
        try:
            # Try to use the main TUI app
            try:
                from .main_app import MainTUIApp
                app = MainTUIApp(self.cli_config)
                return await app.run()
            except ImportError:
                # MainTUIApp not available, use basic launcher
                logger.info("MainTUIApp not available, using TUI launcher")
                from .main_app import launch_main_tui
                return await launch_main_tui(self.cli_config)
            
        except Exception as e:
            logger.warning(f"Enhanced TUI launch failed: {e}")
            return await self._launch_basic_tui()
    
    async def _launch_basic_tui(self) -> int:
        """Launch Basic-level TUI using the fixed working implementation."""
        logger.info("🔧 Launching Basic TUI using fixed working implementation")
        
        try:
            # First try the entry point adapter
            try:
                return await self.entry_adapter.launch_basic_tui(self.cli_config)
            except:
                # Direct fallback to fixed working TUI
                logger.info("Using direct fallback to fixed working TUI")
                from .fixed_working_tui import launch_fixed_working_tui
                return await launch_fixed_working_tui()
            
        except Exception as e:
            logger.warning(f"Basic TUI launch failed: {e}")
            return await self._launch_fallback_tui()
    
    async def _launch_fallback_tui(self) -> int:
        """Launch the most basic fallback TUI implementation."""
        logger.info("🆘 Launching fallback TUI - last resort")
        
        try:
            # Use the fixed working TUI directly as last resort
            try:
                from .fixed_working_tui import launch_fixed_working_tui
                return await launch_fixed_working_tui()
            except ImportError:
                # Even the fixed working TUI isn't available, try main_app
                from .main_app import launch_main_tui
                return await launch_main_tui(self.cli_config)
            
        except Exception as e:
            logger.error(f"Fallback TUI launch failed: {e}")
            # Ultimate fallback: basic error message
            print("❌ TUI launch failed. Please use CLI mode with: agentsmcp simple \"<your message>\"")
            return 1
    
    async def shutdown(self):
        """Initiate graceful shutdown."""
        logger.info("Revolutionary launcher shutting down")
        self._shutdown_event.set()
        await self._cleanup()
    
    async def _cleanup(self):
        """Clean up all active components."""
        for component in reversed(self._active_components):
            try:
                if hasattr(component, 'cleanup'):
                    await component.cleanup()
            except Exception as e:
                logger.warning(f"Component cleanup failed: {e}")
        
        self._active_components.clear()
        logger.debug("Revolutionary launcher cleanup complete")


# Convenience function for external integration
async def launch_revolutionary_tui(cli_config: Optional[CLIConfig] = None) -> int:
    """
    Launch the revolutionary TUI system with automatic feature detection.
    
    This is the main entry point for the revolutionary TUI experience.
    It automatically detects terminal capabilities and system performance
    to provide the best possible TUI experience.
    
    Args:
        cli_config: CLI configuration, uses defaults if None
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    launcher = RevolutionaryLauncher(cli_config)
    return await launcher.launch()


# Direct execution support for testing
if __name__ == "__main__":
    async def main():
        """Direct execution for testing."""
        try:
            return await launch_revolutionary_tui()
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return 0
    
    sys.exit(asyncio.run(main()))