"""
Feature Activation Manager - Progressive enhancement and capability detection

This module manages the detection of terminal capabilities, system performance,
and accessibility features to determine the appropriate TUI feature level.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import shutil
import sys
from enum import Enum
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class FeatureLevel(Enum):
    """TUI feature levels from basic to ultra-advanced."""
    BASIC = "basic"           # Basic chat interface with minimal features
    ENHANCED = "enhanced"     # Improved UX with better input handling
    REVOLUTIONARY = "revolutionary"  # Revolutionary features with rich UI
    ULTRA = "ultra"           # Full feature set with all enhancements


class TerminalType(Enum):
    """Terminal emulator types with their capabilities."""
    KITTY = "kitty"           # Advanced terminal with full feature support
    ITERM2 = "iterm2"         # macOS terminal with good capabilities
    VSCODE = "vscode"         # VS Code integrated terminal
    TMUX = "tmux"             # Terminal multiplexer
    SCREEN = "screen"         # GNU Screen
    BASIC = "basic"           # Basic terminal emulator
    UNKNOWN = "unknown"       # Unknown or unsupported terminal


class PerformanceTier(Enum):
    """System performance tiers for resource allocation."""
    LOW = "low"               # Limited resources, prefer lightweight features
    MEDIUM = "medium"         # Moderate resources, balanced approach
    HIGH = "high"             # Good resources, enable more features
    ULTRA = "ultra"           # Excellent resources, full feature set


class FeatureActivationManager:
    """
    Manages progressive enhancement and feature activation.
    
    This class detects terminal capabilities, system performance, and
    accessibility requirements to determine the optimal TUI feature level.
    """
    
    def __init__(self):
        """Initialize the feature activation manager."""
        self.initialized = False
        self._capabilities_cache = None
        self._feature_level_cache = None
        
        # Feature flags from environment
        self.env_flags = self._load_environment_flags()
        
        logger.debug("Feature activation manager created")
    
    def _load_environment_flags(self) -> Dict[str, str]:
        """Load TUI-related environment flags for feature control."""
        flags = {}
        
        # TUI v2 flags
        tui_flags = [
            'AGENTS_TUI_V2_MINIMAL',
            'AGENTS_TUI_V2_FORCE_RAW_INPUT',
            'AGENTS_TUI_V2_BACKEND',
            'AGENTS_TUI_V2_DEBUG',
            'AGENTS_TUI_V2_INPUT_LINES',
            'AGENTS_TUI_V2_WHEEL_LINES',
            'AGENTS_TUI_V2_CARET_CHAR',
            'AGENTS_TUI_V2_SPINNER_MIN_MS',
            'AGENTS_TUI_V2_POLL_MS',
            'AGENTS_TUI_V2_MOUSE',
            'AGENTS_TUI_V2_QUICK_QUIT',
            'AGENTS_TUI_ENABLE_V2',
            'AGENTS_TUI_V2_NO_FALLBACK',
            'AGENTS_TUI_SUPPRESS_TIPS'
        ]
        
        for flag in tui_flags:
            value = os.getenv(flag, '')
            if value:
                flags[flag] = value
        
        # Terminal detection flags
        term_flags = [
            'TERM',
            'TERM_PROGRAM', 
            'TERM_PROGRAM_VERSION',
            'COLORTERM',
            'KITTY_WINDOW_ID',
            'ITERM_SESSION_ID',
            'TMUX',
            'STY'  # GNU Screen
        ]
        
        for flag in term_flags:
            value = os.getenv(flag, '')
            if value:
                flags[flag] = value
        
        return flags
    
    async def initialize(self) -> bool:
        """Initialize the feature activation manager.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.debug("Initializing feature activation manager")
            
            # Check if we can detect basic terminal info
            if not self._can_detect_terminal():
                logger.warning("Cannot detect terminal capabilities")
                return False
            
            self.initialized = True
            logger.debug("Feature activation manager initialized successfully")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to initialize feature activation manager: {e}")
            return False
    
    def _can_detect_terminal(self) -> bool:
        """Check if we can perform basic terminal detection."""
        try:
            # Check if we have a TTY
            if not sys.stdout.isatty() or not sys.stdin.isatty():
                logger.debug("Not running in a TTY")
                return False
            
            # Check if we can get terminal size
            try:
                shutil.get_terminal_size()
                return True
            except OSError:
                logger.debug("Cannot get terminal size")
                return False
                
        except Exception as e:
            logger.debug(f"Terminal detection check failed: {e}")
            return False
    
    async def detect_capabilities(self) -> Dict[str, Any]:
        """
        Detect terminal and system capabilities.
        
        Returns:
            Dictionary containing capability information
        """
        if not self.initialized:
            await self.initialize()
        
        # Use cached result if available
        if self._capabilities_cache is not None:
            return self._capabilities_cache
        
        try:
            capabilities = {}
            
            # Terminal detection
            terminal_info = await self._detect_terminal_type()
            capabilities.update(terminal_info)
            
            # System performance detection
            performance_info = await self._detect_system_performance()
            capabilities.update(performance_info)
            
            # Accessibility features
            accessibility_info = self._detect_accessibility_features()
            capabilities.update(accessibility_info)
            
            # Color support
            color_info = self._detect_color_support()
            capabilities.update(color_info)
            
            # Feature overrides from environment
            override_info = self._check_environment_overrides()
            capabilities.update(override_info)
            
            # Cache the result
            self._capabilities_cache = capabilities
            
            logger.debug(f"Detected capabilities: {capabilities}")
            return capabilities
            
        except Exception as e:
            logger.warning(f"Capability detection failed: {e}")
            return self._get_fallback_capabilities()
    
    async def _detect_terminal_type(self) -> Dict[str, Any]:
        """Detect the terminal emulator type and capabilities."""
        terminal_info = {
            'terminal_type': TerminalType.UNKNOWN.value,
            'terminal_name': 'unknown',
            'terminal_version': '',
            'width': 80,
            'height': 24
        }
        
        try:
            # Get terminal size
            try:
                size = shutil.get_terminal_size()
                terminal_info['width'] = size.columns
                terminal_info['height'] = size.lines
            except OSError:
                pass
            
            # Detect terminal type from environment variables
            term = self.env_flags.get('TERM', '').lower()
            term_program = self.env_flags.get('TERM_PROGRAM', '').lower()
            term_version = self.env_flags.get('TERM_PROGRAM_VERSION', '')
            
            # Kitty terminal
            if 'kitty' in term or 'kitty' in term_program or 'KITTY_WINDOW_ID' in self.env_flags:
                terminal_info.update({
                    'terminal_type': TerminalType.KITTY.value,
                    'terminal_name': 'Kitty',
                    'terminal_version': term_version,
                    'supports_images': True,
                    'supports_hyperlinks': True,
                    'supports_notifications': True
                })
            
            # iTerm2
            elif 'iterm' in term_program or 'ITERM_SESSION_ID' in self.env_flags:
                terminal_info.update({
                    'terminal_type': TerminalType.ITERM2.value,
                    'terminal_name': 'iTerm2',
                    'terminal_version': term_version,
                    'supports_images': True,
                    'supports_hyperlinks': True
                })
            
            # VS Code integrated terminal
            elif 'vscode' in term_program or 'code' in term_program:
                terminal_info.update({
                    'terminal_type': TerminalType.VSCODE.value,
                    'terminal_name': 'VS Code',
                    'terminal_version': term_version,
                    'supports_hyperlinks': True
                })
            
            # Tmux
            elif 'TMUX' in self.env_flags or 'tmux' in term:
                terminal_info.update({
                    'terminal_type': TerminalType.TMUX.value,
                    'terminal_name': 'tmux',
                    'terminal_version': self.env_flags.get('TMUX', '').split(',')[0] if 'TMUX' in self.env_flags else ''
                })
            
            # GNU Screen
            elif 'STY' in self.env_flags or 'screen' in term:
                terminal_info.update({
                    'terminal_type': TerminalType.SCREEN.value,
                    'terminal_name': 'GNU Screen'
                })
            
            # Basic terminal detection
            elif term in ('xterm', 'xterm-256color', 'xterm-color'):
                terminal_info.update({
                    'terminal_type': TerminalType.BASIC.value,
                    'terminal_name': 'XTerm compatible'
                })
            
            else:
                terminal_info.update({
                    'terminal_type': TerminalType.BASIC.value,
                    'terminal_name': f'Basic ({term})'
                })
            
            logger.debug(f"Detected terminal: {terminal_info}")
            return terminal_info
            
        except Exception as e:
            logger.warning(f"Terminal detection failed: {e}")
            return terminal_info
    
    async def _detect_system_performance(self) -> Dict[str, Any]:
        """Detect system performance characteristics."""
        performance_info = {
            'performance_tier': PerformanceTier.MEDIUM.value,
            'cpu_count': 1,
            'memory_gb': 0,
            'platform': platform.system().lower()
        }
        
        try:
            # CPU core count
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            performance_info['cpu_count'] = cpu_count
            
            # Memory detection (approximate)
            memory_gb = 0
            try:
                if platform.system() == 'Darwin':  # macOS
                    import subprocess
                    result = subprocess.run(['sysctl', '-n', 'hw.memsize'], 
                                           capture_output=True, text=True, timeout=1)
                    if result.returncode == 0:
                        memory_bytes = int(result.stdout.strip())
                        memory_gb = memory_bytes / (1024 ** 3)
                elif platform.system() == 'Linux':
                    with open('/proc/meminfo', 'r') as f:
                        for line in f:
                            if line.startswith('MemTotal:'):
                                memory_kb = int(line.split()[1])
                                memory_gb = memory_kb / (1024 ** 2)
                                break
            except Exception:
                # Memory detection is optional
                pass
            
            performance_info['memory_gb'] = round(memory_gb, 1)
            
            # Determine performance tier
            if cpu_count >= 8 and memory_gb >= 16:
                performance_info['performance_tier'] = PerformanceTier.ULTRA.value
            elif cpu_count >= 4 and memory_gb >= 8:
                performance_info['performance_tier'] = PerformanceTier.HIGH.value
            elif cpu_count >= 2 and memory_gb >= 4:
                performance_info['performance_tier'] = PerformanceTier.MEDIUM.value
            else:
                performance_info['performance_tier'] = PerformanceTier.LOW.value
            
            logger.debug(f"Detected performance: {performance_info}")
            return performance_info
            
        except Exception as e:
            logger.warning(f"Performance detection failed: {e}")
            return performance_info
    
    def _detect_accessibility_features(self) -> Dict[str, Any]:
        """Detect accessibility requirements."""
        accessibility_info = {
            'screen_reader': False,
            'high_contrast': False,
            'reduced_motion': False
        }
        
        try:
            # Check for common screen reader indicators
            if any(env in os.environ for env in ['NVDA', 'JAWS', 'DRAGON']):
                accessibility_info['screen_reader'] = True
            
            # Check for accessibility preferences (macOS)
            if platform.system() == 'Darwin':
                try:
                    import subprocess
                    # Check for VoiceOver
                    result = subprocess.run(['defaults', 'read', 'com.apple.universalaccess', 'voiceOverOnOffKey'], 
                                           capture_output=True, timeout=1)
                    if result.returncode == 0:
                        accessibility_info['screen_reader'] = True
                except Exception:
                    pass
            
            logger.debug(f"Detected accessibility features: {accessibility_info}")
            return accessibility_info
            
        except Exception as e:
            logger.warning(f"Accessibility detection failed: {e}")
            return accessibility_info
    
    def _detect_color_support(self) -> Dict[str, Any]:
        """Detect terminal color support capabilities."""
        color_info = {
            'colors': 8,
            'true_color': False,
            'color_support': 'basic'
        }
        
        try:
            term = self.env_flags.get('TERM', '')
            colorterm = self.env_flags.get('COLORTERM', '')
            
            # True color support
            if colorterm in ('truecolor', '24bit') or '24bit' in term:
                color_info.update({
                    'colors': 16777216,
                    'true_color': True,
                    'color_support': 'truecolor'
                })
            # 256 color support
            elif '256color' in term or '256' in colorterm:
                color_info.update({
                    'colors': 256,
                    'color_support': '256color'
                })
            # Basic 16 colors
            elif 'color' in term:
                color_info.update({
                    'colors': 16,
                    'color_support': '16color'
                })
            # Monochrome
            else:
                color_info.update({
                    'colors': 2,
                    'color_support': 'monochrome'
                })
            
            logger.debug(f"Detected color support: {color_info}")
            return color_info
            
        except Exception as e:
            logger.warning(f"Color detection failed: {e}")
            return color_info
    
    def _check_environment_overrides(self) -> Dict[str, Any]:
        """Check for environment variable overrides."""
        overrides = {}
        
        try:
            # Force minimal mode
            if self.env_flags.get('AGENTS_TUI_V2_MINIMAL') == '1':
                overrides['force_minimal'] = True
            
            # Force raw input
            if self.env_flags.get('AGENTS_TUI_V2_FORCE_RAW_INPUT') == '1':
                overrides['force_raw_input'] = True
            
            # Debug mode
            if self.env_flags.get('AGENTS_TUI_V2_DEBUG') == '1':
                overrides['debug_mode'] = True
            
            # Backend disabled
            if self.env_flags.get('AGENTS_TUI_V2_BACKEND') == '0':
                overrides['backend_disabled'] = True
            
            # Custom input lines
            try:
                input_lines = int(self.env_flags.get('AGENTS_TUI_V2_INPUT_LINES', '0'))
                if input_lines > 0:
                    overrides['custom_input_lines'] = input_lines
            except ValueError:
                pass
            
            logger.debug(f"Environment overrides: {overrides}")
            return overrides
            
        except Exception as e:
            logger.warning(f"Environment override check failed: {e}")
            return overrides
    
    def determine_feature_level(self, capabilities: Dict[str, Any]) -> FeatureLevel:
        """
        Determine the appropriate feature level based on capabilities.
        
        Args:
            capabilities: Detected capability information
            
        Returns:
            The optimal feature level
        """
        if self._feature_level_cache is not None:
            return self._feature_level_cache
        
        try:
            # Check for force minimal override
            if capabilities.get('force_minimal', False):
                self._feature_level_cache = FeatureLevel.BASIC
                return FeatureLevel.BASIC
            
            # Check accessibility requirements
            if capabilities.get('screen_reader', False):
                # Screen readers work better with basic interfaces
                self._feature_level_cache = FeatureLevel.ENHANCED
                return FeatureLevel.ENHANCED
            
            # Get key metrics
            terminal_type = capabilities.get('terminal_type', 'unknown')
            performance_tier = capabilities.get('performance_tier', 'medium')
            colors = capabilities.get('colors', 8)
            
            # Ultra level: Kitty/iTerm2 + high performance + true color
            if (terminal_type in ['kitty', 'iterm2'] and 
                performance_tier in ['ultra', 'high'] and 
                colors >= 256):
                self._feature_level_cache = FeatureLevel.ULTRA
                return FeatureLevel.ULTRA
            
            # Revolutionary level: Good terminal + decent performance
            if (terminal_type in ['kitty', 'iterm2', 'vscode'] and 
                performance_tier in ['ultra', 'high', 'medium'] and 
                colors >= 256):
                self._feature_level_cache = FeatureLevel.REVOLUTIONARY
                return FeatureLevel.REVOLUTIONARY
            
            # Enhanced level: Basic terminal + reasonable performance
            if (terminal_type != 'unknown' and 
                performance_tier in ['ultra', 'high', 'medium'] and 
                colors >= 16):
                self._feature_level_cache = FeatureLevel.ENHANCED
                return FeatureLevel.ENHANCED
            
            # Basic level: Everything else
            self._feature_level_cache = FeatureLevel.BASIC
            return FeatureLevel.BASIC
            
        except Exception as e:
            logger.warning(f"Feature level determination failed: {e}")
            return FeatureLevel.BASIC
    
    def _get_fallback_capabilities(self) -> Dict[str, Any]:
        """Get minimal fallback capabilities when detection fails."""
        return {
            'terminal_type': TerminalType.BASIC.value,
            'terminal_name': 'Basic Terminal',
            'performance_tier': PerformanceTier.LOW.value,
            'colors': 8,
            'width': 80,
            'height': 24,
            'color_support': 'basic',
            'screen_reader': False,
            'force_minimal': True
        }
    
    def get_feature_recommendations(self, level: FeatureLevel) -> Dict[str, Any]:
        """
        Get feature recommendations for a specific feature level.
        
        Args:
            level: The feature level to get recommendations for
            
        Returns:
            Dictionary of recommended features and settings
        """
        base_features = {
            'input_echo': True,
            'history_support': True,
            'basic_commands': True,
            'error_handling': True
        }
        
        if level == FeatureLevel.BASIC:
            return base_features
        
        elif level == FeatureLevel.ENHANCED:
            return {
                **base_features,
                'multiline_input': True,
                'syntax_highlighting': False,
                'auto_completion': False,
                'status_bar': True,
                'scroll_support': True
            }
        
        elif level == FeatureLevel.REVOLUTIONARY:
            return {
                **base_features,
                'multiline_input': True,
                'syntax_highlighting': True,
                'auto_completion': True,
                'status_bar': True,
                'scroll_support': True,
                'rich_formatting': True,
                'progress_indicators': True,
                'keyboard_shortcuts': True
            }
        
        elif level == FeatureLevel.ULTRA:
            return {
                **base_features,
                'multiline_input': True,
                'syntax_highlighting': True,
                'auto_completion': True,
                'status_bar': True,
                'scroll_support': True,
                'rich_formatting': True,
                'progress_indicators': True,
                'keyboard_shortcuts': True,
                'mouse_support': True,
                'split_panes': True,
                'themes': True,
                'animations': True,
                'advanced_search': True
            }
        
        return base_features
    
    async def cleanup(self):
        """Clean up resources."""
        logger.debug("Feature activation manager cleanup")
        self._capabilities_cache = None
        self._feature_level_cache = None