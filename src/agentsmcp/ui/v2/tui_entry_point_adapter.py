"""
TUI Entry Point Adapter - Bridge between CLI system and revolutionary integration

This module provides an adapter that bridges the existing CLI system with the
revolutionary TUI integration layer, maintaining compatibility while enabling
enhanced features when available.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Optional, Any, Dict
from pathlib import Path

from ..cli_app import CLIConfig

logger = logging.getLogger(__name__)


class TUIEntryPointAdapter:
    """
    Adapter for bridging CLI system with revolutionary TUI integration.
    
    This adapter:
    - Preserves all existing CLI functionality and command-line arguments
    - Modifies how TUI launches to use revolutionary system when available
    - Provides graceful error handling with fallback to basic mode
    - Maintains complete backward compatibility
    """
    
    def __init__(self):
        """Initialize the TUI entry point adapter."""
        self.cli_config: Optional[CLIConfig] = None
        self.integration_available = False
        self.fallback_mode = False
        
        logger.debug("TUI entry point adapter initialized")
    
    async def launch_basic_tui(self, cli_config: Optional[CLIConfig] = None) -> int:
        """
        Launch basic TUI using the fixed working implementation.
        
        This is the reliable fallback that always works.
        
        Args:
            cli_config: CLI configuration to use
            
        Returns:
            Exit code (0 for success, 1 for error)
        """
        self.cli_config = cli_config or CLIConfig()
        
        try:
            logger.info("🔧 Launching basic TUI via entry point adapter")
            
            # Use the fixed working TUI as the reliable base implementation
            from .fixed_working_tui import launch_fixed_working_tui
            return await launch_fixed_working_tui()
            
        except Exception as e:
            logger.error(f"Basic TUI launch failed: {e}")
            return await self._launch_emergency_fallback()
    
    async def launch_with_revolutionary_features(self, cli_config: Optional[CLIConfig] = None) -> int:
        """
        Launch TUI with revolutionary features when available.
        
        Args:
            cli_config: CLI configuration to use
            
        Returns:
            Exit code (0 for success, 1 for error)
        """
        self.cli_config = cli_config or CLIConfig()
        
        try:
            logger.info("🚀 Attempting to launch TUI with revolutionary features")
            
            # Check if revolutionary components are available
            if await self._check_revolutionary_availability():
                return await self._launch_revolutionary_integration()
            else:
                logger.info("Revolutionary features not available, using basic TUI")
                return await self.launch_basic_tui(cli_config)
                
        except Exception as e:
            logger.warning(f"Revolutionary TUI launch failed: {e}")
            return await self.launch_basic_tui(cli_config)
    
    async def _check_revolutionary_availability(self) -> bool:
        """Check if revolutionary components are available and functional."""
        try:
            # Try to import revolutionary components
            from ..components.revolutionary_integration_layer import RevolutionaryIntegrationLayer
            from ..components.revolutionary_tui_enhancements import RevolutionaryTUIEnhancements
            
            # Quick health check
            integration_layer = RevolutionaryIntegrationLayer()
            if hasattr(integration_layer, 'health_check'):
                health_ok = await asyncio.wait_for(integration_layer.health_check(), timeout=0.5)
                if not health_ok:
                    logger.warning("Revolutionary integration layer health check failed")
                    return False
            
            self.integration_available = True
            logger.debug("Revolutionary components are available and healthy")
            return True
            
        except ImportError as e:
            logger.debug(f"Revolutionary components not available: {e}")
            return False
        except Exception as e:
            logger.warning(f"Revolutionary component health check failed: {e}")
            return False
    
    async def _launch_revolutionary_integration(self) -> int:
        """Launch the TUI with full revolutionary integration."""
        try:
            from ..components.revolutionary_integration_layer import RevolutionaryIntegrationLayer
            from ..components.revolutionary_tui_enhancements import RevolutionaryTUIEnhancements
            
            # Initialize revolutionary components
            integration_layer = RevolutionaryIntegrationLayer()
            tui_enhancements = RevolutionaryTUIEnhancements()
            
            # Initialize components with timeout
            await asyncio.wait_for(integration_layer.initialize(), timeout=2.0)
            await asyncio.wait_for(tui_enhancements.initialize(), timeout=2.0)
            
            # Launch the enhanced main TUI app
            from .main_app import MainTUIApp
            app = MainTUIApp(self.cli_config)
            
            # Inject revolutionary components
            app.integration_layer = integration_layer
            app.tui_enhancements = tui_enhancements
            
            logger.info("🌟 Successfully launched TUI with revolutionary features")
            return await app.run()
            
        except asyncio.TimeoutError:
            logger.warning("Revolutionary component initialization timed out")
            raise
        except Exception as e:
            logger.warning(f"Revolutionary integration launch failed: {e}")
            raise
    
    async def launch_compatible_tui(self, cli_config: Optional[CLIConfig] = None) -> int:
        """
        Launch TUI with maximum compatibility mode.
        
        This method ensures the TUI works across all terminal types and
        system configurations by using the most compatible implementation.
        
        Args:
            cli_config: CLI configuration to use
            
        Returns:
            Exit code (0 for success, 1 for error)
        """
        self.cli_config = cli_config or CLIConfig()
        
        try:
            logger.info("🔄 Launching TUI in maximum compatibility mode")
            
            # Force minimal mode for maximum compatibility
            os.environ['AGENTS_TUI_V2_MINIMAL'] = '1'
            os.environ['AGENTS_TUI_V2_FORCE_RAW_INPUT'] = '1'
            os.environ['AGENTS_TUI_SUPPRESS_TIPS'] = '1'
            
            # Use basic TUI implementation
            return await self.launch_basic_tui(cli_config)
            
        except Exception as e:
            logger.error(f"Compatible TUI launch failed: {e}")
            return await self._launch_emergency_fallback()
    
    async def _launch_emergency_fallback(self) -> int:
        """Launch the most basic emergency fallback TUI."""
        try:
            logger.warning("🆘 Launching emergency fallback TUI")
            
            # Try the most basic fixed working TUI
            from .fixed_working_tui import launch_fixed_working_tui
            return await launch_fixed_working_tui()
            
        except Exception as e:
            logger.error(f"Emergency fallback failed: {e}")
            # Ultimate fallback: print error and exit
            print("❌ TUI system failed completely. Please use CLI mode:")
            print("   agentsmcp simple \"<your message>\"")
            print("   agentsmcp interactive --legacy")
            return 1
    
    def get_launch_recommendations(self) -> Dict[str, Any]:
        """
        Get recommendations for TUI launch based on current environment.
        
        Returns:
            Dictionary with launch recommendations
        """
        recommendations = {
            'recommended_method': 'basic',
            'fallback_available': True,
            'revolutionary_available': self.integration_available,
            'compatibility_notes': []
        }
        
        try:
            # Check terminal type
            term = os.getenv('TERM', '').lower()
            term_program = os.getenv('TERM_PROGRAM', '').lower()
            
            # Recommendations based on terminal
            if 'kitty' in term or 'kitty' in term_program:
                recommendations['recommended_method'] = 'revolutionary'
                recommendations['compatibility_notes'].append('Kitty terminal supports all features')
            
            elif 'iterm' in term_program:
                recommendations['recommended_method'] = 'revolutionary' if self.integration_available else 'enhanced'
                recommendations['compatibility_notes'].append('iTerm2 supports enhanced features')
            
            elif 'vscode' in term_program:
                recommendations['recommended_method'] = 'enhanced'
                recommendations['compatibility_notes'].append('VS Code terminal has some limitations')
            
            elif 'tmux' in os.getenv('TMUX', ''):
                recommendations['recommended_method'] = 'basic'
                recommendations['compatibility_notes'].append('tmux may interfere with advanced features')
            
            else:
                recommendations['recommended_method'] = 'basic'
                recommendations['compatibility_notes'].append('Using basic mode for compatibility')
            
            # Check for environment flags that affect recommendations
            if os.getenv('AGENTS_TUI_V2_MINIMAL') == '1':
                recommendations['recommended_method'] = 'basic'
                recommendations['compatibility_notes'].append('Forced minimal mode via environment')
            
            # Check if we're in a CI environment
            ci_indicators = ['CI', 'CONTINUOUS_INTEGRATION', 'GITHUB_ACTIONS', 'GITLAB_CI']
            if any(os.getenv(var) for var in ci_indicators):
                recommendations['recommended_method'] = 'compatible'
                recommendations['compatibility_notes'].append('CI environment detected - using compatible mode')
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Failed to generate launch recommendations: {e}")
            return {
                'recommended_method': 'basic',
                'fallback_available': True,
                'revolutionary_available': False,
                'compatibility_notes': ['Error generating recommendations, using basic mode']
            }
    
    def configure_environment_for_mode(self, mode: str) -> None:
        """
        Configure environment variables for the specified TUI mode.
        
        Args:
            mode: TUI mode ('basic', 'enhanced', 'revolutionary', 'compatible')
        """
        try:
            # Clear previous TUI environment settings
            tui_env_vars = [
                'AGENTS_TUI_V2_MINIMAL',
                'AGENTS_TUI_V2_FORCE_RAW_INPUT', 
                'AGENTS_TUI_V2_BACKEND',
                'AGENTS_TUI_V2_DEBUG',
                'AGENTS_TUI_ENABLE_V2',
                'AGENTS_TUI_V2_NO_FALLBACK',
                'AGENTS_TUI_SUPPRESS_TIPS'
            ]
            
            # Configure based on mode
            if mode == 'basic':
                os.environ['AGENTS_TUI_V2_MINIMAL'] = '1'
                os.environ['AGENTS_TUI_V2_FORCE_RAW_INPUT'] = '1'
                os.environ['AGENTS_TUI_V2_BACKEND'] = '1'
                os.environ['AGENTS_TUI_SUPPRESS_TIPS'] = '1'
                
            elif mode == 'enhanced':
                os.environ['AGENTS_TUI_ENABLE_V2'] = '1'
                os.environ['AGENTS_TUI_V2_BACKEND'] = '1'
                os.environ['AGENTS_TUI_SUPPRESS_TIPS'] = '1'
                
            elif mode == 'revolutionary':
                os.environ['AGENTS_TUI_ENABLE_V2'] = '1'
                os.environ['AGENTS_TUI_V2_BACKEND'] = '1'
                os.environ['AGENTS_TUI_V2_NO_FALLBACK'] = '0'  # Allow fallback if needed
                
            elif mode == 'compatible':
                os.environ['AGENTS_TUI_V2_MINIMAL'] = '1'
                os.environ['AGENTS_TUI_V2_FORCE_RAW_INPUT'] = '1'
                os.environ['AGENTS_TUI_SUPPRESS_TIPS'] = '1'
            
            logger.debug(f"Configured environment for {mode} mode")
            
        except Exception as e:
            logger.warning(f"Failed to configure environment for {mode} mode: {e}")
    
    async def validate_tui_environment(self) -> Dict[str, Any]:
        """
        Validate the current environment for TUI launch.
        
        Returns:
            Dictionary with validation results
        """
        validation = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        try:
            # Check if we have a TTY
            if not sys.stdout.isatty() or not sys.stdin.isatty():
                validation['errors'].append('Not running in a TTY environment')
                validation['valid'] = False
            
            # Check terminal size
            try:
                import shutil
                size = shutil.get_terminal_size()
                if size.columns < 40 or size.lines < 10:
                    validation['warnings'].append(f'Small terminal size: {size.columns}x{size.lines}')
                    validation['recommendations'].append('Consider using a larger terminal window')
            except OSError:
                validation['warnings'].append('Cannot determine terminal size')
            
            # Check for conflicting environment variables
            conflicting_vars = []
            if os.getenv('PAGER'):
                conflicting_vars.append('PAGER')
            if os.getenv('LESS'):
                conflicting_vars.append('LESS')
            
            if conflicting_vars:
                validation['warnings'].append(f'Potentially conflicting environment: {", ".join(conflicting_vars)}')
            
            # Check Python version compatibility
            if sys.version_info < (3, 8):
                validation['errors'].append(f'Python {sys.version_info.major}.{sys.version_info.minor} not supported (requires 3.8+)')
                validation['valid'] = False
            
            # Check for required modules
            try:
                import termios
                import tty
            except ImportError as e:
                validation['errors'].append(f'Missing required module: {e}')
                validation['valid'] = False
            
            return validation
            
        except Exception as e:
            logger.warning(f"Environment validation failed: {e}")
            return {
                'valid': False,
                'warnings': [],
                'errors': [f'Validation failed: {e}'],
                'recommendations': ['Try using basic CLI mode instead']
            }
    
    async def cleanup(self):
        """Clean up adapter resources."""
        logger.debug("TUI entry point adapter cleanup")
        self.cli_config = None
        self.integration_available = False


# Convenience functions for external integration
async def launch_adaptive_tui(cli_config: Optional[CLIConfig] = None) -> int:
    """
    Launch TUI with adaptive feature selection.
    
    This function automatically selects the best TUI implementation
    based on terminal capabilities and system performance.
    
    Args:
        cli_config: CLI configuration to use
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    adapter = TUIEntryPointAdapter()
    
    # Get launch recommendations
    recommendations = adapter.get_launch_recommendations()
    recommended_method = recommendations['recommended_method']
    
    # Launch using recommended method
    if recommended_method == 'revolutionary':
        return await adapter.launch_with_revolutionary_features(cli_config)
    elif recommended_method == 'compatible':
        return await adapter.launch_compatible_tui(cli_config)
    else:
        return await adapter.launch_basic_tui(cli_config)


async def launch_safe_tui(cli_config: Optional[CLIConfig] = None) -> int:
    """
    Launch TUI in safe mode with maximum compatibility.
    
    This function prioritizes reliability over features.
    
    Args:
        cli_config: CLI configuration to use
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    adapter = TUIEntryPointAdapter()
    return await adapter.launch_compatible_tui(cli_config)


# Direct execution support for testing
if __name__ == "__main__":
    async def main():
        """Direct execution for testing."""
        adapter = TUIEntryPointAdapter()
        
        # Validate environment first
        validation = await adapter.validate_tui_environment()
        if not validation['valid']:
            print("❌ Environment validation failed:")
            for error in validation['errors']:
                print(f"   • {error}")
            return 1
        
        if validation['warnings']:
            print("⚠️  Environment warnings:")
            for warning in validation['warnings']:
                print(f"   • {warning}")
        
        # Get recommendations
        recommendations = adapter.get_launch_recommendations()
        print(f"🎯 Recommended launch method: {recommendations['recommended_method']}")
        
        # Launch TUI
        return await launch_adaptive_tui()
    
    sys.exit(asyncio.run(main()))