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
            print("🚀 Starting Revolutionary TUI Launcher")
            logger.info("🚀 Starting Revolutionary TUI Launcher")
            
            # Phase 1: Initialize feature detection
            print("Phase 1: Initializing feature detection...")
            if not await self._initialize_feature_detection():
                print("⚠️ Feature detection failed, falling back to basic TUI")
                logger.warning("Feature detection failed, falling back to basic TUI")
                return await self._launch_fallback_tui()
            
            # Phase 2: Determine optimal feature level
            print("Phase 2: Determining feature level...")
            feature_level = await self._determine_feature_level()
            print(f"✅ Determined feature level: {feature_level.name}")
            logger.info(f"Determined feature level: {feature_level.name}")
            
            # Phase 3: Launch appropriate TUI implementation
            print(f"Phase 3: Launching {feature_level.name} TUI...")
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
        print("🌟 Launching Ultra TUI with full revolutionary features")
        logger.info("🌟 Launching Ultra TUI with full revolutionary features")
        
        try:
            # For ultra-level, use the Revolutionary TUI Interface with all features
            print("Ultra mode: Using Revolutionary TUI Interface with all features...")
            return await self._launch_revolutionary_tui()
            
        except Exception as e:
            print(f"❌ Ultra TUI launch failed: {e}")
            logger.warning(f"Ultra TUI launch failed: {e}")
            print("Falling back to enhanced TUI...")
            return await self._launch_enhanced_tui()
    
    async def _launch_revolutionary_tui(self) -> int:
        """Launch Revolutionary-level TUI with core enhancements."""
        print("🔥 Launching Revolutionary TUI with core enhancements")
        logger.info("🔥 Launching Revolutionary TUI with core enhancements")
        
        try:
            # Import the Revolutionary TUI Interface
            print("Importing Revolutionary TUI Interface...")
            from .revolutionary_tui_interface import RevolutionaryTUIInterface, create_revolutionary_interface
            
            print("Initializing Revolutionary TUI Interface...")
            logger.info("Initializing Revolutionary TUI Interface...")
            
            # Try to initialize orchestrator integration
            orchestrator_integration = None
            try:
                from .orchestrator_integration import initialize_orchestrator_integration
                orchestrator_integration = await initialize_orchestrator_integration()
            except Exception as e:
                logger.warning(f"Orchestrator integration failed, continuing without: {e}")
            
            # Initialize revolutionary components
            revolutionary_components = {}
            try:
                from ..components.symphony_dashboard import SymphonyDashboard
                from ..components.ai_command_composer import AICommandComposer
                from .event_system import AsyncEventSystem
                
                event_system = AsyncEventSystem()
                await event_system.initialize()
                revolutionary_components = {
                    'symphony_dashboard': SymphonyDashboard(event_system),
                    'ai_command_composer': AICommandComposer(event_system),
                    'event_system': event_system
                }
                logger.info("Revolutionary components initialized successfully")
                
            except Exception as e:
                logger.warning(f"Some revolutionary components failed to initialize: {e}")
            
            # Create the Revolutionary TUI Interface
            revolutionary_interface = await create_revolutionary_interface(
                cli_config=self.cli_config,
                orchestrator_integration=orchestrator_integration,
                revolutionary_components=revolutionary_components
            )
            
            self._active_components.append(revolutionary_interface)
            
            logger.info("🚀 Revolutionary TUI Interface created successfully - launching...")
            
            # Launch the Revolutionary TUI Interface
            return await revolutionary_interface.run()
            
        except Exception as e:
            print(f"❌ Revolutionary TUI Interface launch failed: {e}")
            logger.warning(f"Revolutionary TUI Interface launch failed: {e}")
            print("Falling back to enhanced TUI...")
            logger.info("Falling back to enhanced TUI...")
            return await self._launch_enhanced_tui()
    
    async def _launch_enhanced_tui(self) -> int:
        """Launch Enhanced-level TUI with improved features."""
        logger.info("✨ Launching Enhanced TUI with improved features")
        
        try:
            # Try to use Revolutionary TUI Interface in enhanced mode
            from .revolutionary_tui_interface import RevolutionaryTUIInterface, create_revolutionary_interface
            
            logger.info("Launching Revolutionary TUI Interface in enhanced mode...")
            
            # Initialize orchestrator integration
            orchestrator_integration = None
            try:
                from .orchestrator_integration import initialize_orchestrator_integration
                orchestrator_integration = await initialize_orchestrator_integration()
            except Exception as e:
                logger.warning(f"Orchestrator integration failed: {e}")
            
            # Create Revolutionary interface with basic components
            revolutionary_interface = await create_revolutionary_interface(
                cli_config=self.cli_config,
                orchestrator_integration=orchestrator_integration,
                revolutionary_components={}  # Minimal components for enhanced mode
            )
            
            self._active_components.append(revolutionary_interface)
            
            # Launch the interface
            logger.info("Enhanced TUI (Revolutionary interface) initialized successfully")
            return await revolutionary_interface.run()
            
        except Exception as e:
            logger.warning(f"Enhanced TUI (Revolutionary interface) launch failed: {e}")
            logger.info("Falling back to basic TUI...")
            return await self._launch_basic_tui()
    
    async def _launch_basic_tui(self) -> int:
        """Launch Basic-level TUI using the fixed working implementation."""
        logger.info("🔧 Launching Basic TUI using fixed working implementation")
        
        try:
            # Direct fallback to fixed working TUI - skip adapter to avoid complexity
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
                print("⚠️  Revolutionary TUI failed, launching fixed working TUI...")
                return await launch_fixed_working_tui()
            except ImportError as import_err:
                logger.error(f"Fixed working TUI not available: {import_err}")
                # Ultimate fallback: basic error message
                print("❌ TUI launch failed completely. All TUI implementations unavailable.")
                print("   Please use CLI mode with: agentsmcp --mode interactive")
                return 1
            
        except Exception as e:
            logger.error(f"Fallback TUI launch failed: {e}")
            # Ultimate fallback: basic error message with user guidance
            print("❌ TUI launch failed completely due to system error.")
            print(f"   Error: {str(e)}")
            print("   Please use CLI mode with: agentsmcp --mode interactive")
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