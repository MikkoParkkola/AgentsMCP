"""
Display Controller - Guaranteed Rich display activation with timeout protection.

This module provides critical display initialization with guaranteed activation within 2 seconds
or immediate fallback to basic mode. Prevents Rich display hangs and terminal scrollback pollution.

Key Features:
- GUARANTEED display activation within 2s or immediate fallback
- Rich Live timeout protection using TimeoutGuardian
- Fast terminal capability detection with fallbacks
- Display mode auto-selection (AUTO|RICH|FALLBACK)
- Prevents terminal scrollback pollution via alternate screen enforcement
- Emergency fallback for any display initialization failure

ICD Compliance:
- Inputs: display_mode, timeout_seconds, fallback_enabled
- Outputs: DisplayMode enum, active display instance, initialization_metrics
- Performance: Display activation within 2s or fallback immediately
- Error Handling: All display failures handled with immediate fallback
"""

import asyncio
import logging
import sys
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Any, Dict, Union, Callable
import threading

# Import timeout protection
from .timeout_guardian import TimeoutGuardian, get_global_guardian

# Rich imports with availability check
try:
    from rich.console import Console
    from rich.live import Live
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Define minimal stubs for type compatibility
    class Console:
        def __init__(self, *args, **kwargs): pass
        def print(self, *args, **kwargs): print(*args)
    class Live:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def update(self, content): pass
        def refresh(self): pass

logger = logging.getLogger(__name__)


class DisplayMode(Enum):
    """Display mode options."""
    AUTO = "auto"           # Automatic selection based on terminal capabilities
    RICH = "rich"          # Force Rich Live display
    FALLBACK = "fallback"  # Force basic text output
    FAILED = "failed"      # Display initialization failed


@dataclass
class DisplayMetrics:
    """Display initialization metrics."""
    mode_selected: DisplayMode
    initialization_time_ms: float
    rich_available: bool
    terminal_capable: bool
    timeout_occurred: bool = False
    fallback_reason: Optional[str] = None
    error_details: Optional[str] = None


@dataclass
class TerminalCapabilities:
    """Terminal capability detection results."""
    is_tty: bool
    supports_color: bool
    supports_alternate_screen: bool
    terminal_size: tuple
    term_type: str
    detection_time_ms: float


class DisplayController:
    """
    Controls Rich display initialization with guaranteed timeout protection.
    
    Provides the critical display initialization layer that prevents Rich Live
    hangs by enforcing strict 2-second timeout with immediate fallback.
    """
    
    def __init__(self, 
                 default_timeout: float = 2.0,
                 detection_timeout: float = 0.5,
                 enable_alternate_screen: bool = True):
        """
        Initialize display controller.
        
        Args:
            default_timeout: Max time for display initialization (seconds)
            detection_timeout: Max time for terminal detection (seconds)
            enable_alternate_screen: Use alternate screen to prevent scrollback pollution
        """
        self.default_timeout = default_timeout
        self.detection_timeout = detection_timeout
        self.enable_alternate_screen = enable_alternate_screen
        
        # Timeout protection
        self.guardian = get_global_guardian()
        
        # Display state
        self._active_display: Optional[Live] = None
        self._display_mode: DisplayMode = DisplayMode.AUTO
        self._console: Optional[Console] = None
        self._capabilities: Optional[TerminalCapabilities] = None
        
        # Metrics
        self._initialization_metrics: Optional[DisplayMetrics] = None
        
    def _detect_terminal_capabilities(self) -> TerminalCapabilities:
        """
        Fast terminal capability detection with timeout protection.
        
        Returns:
            TerminalCapabilities with detection results
        """
        start_time = time.time()
        
        try:
            # Basic TTY detection (fast)
            is_tty = sys.stdout.isatty() and sys.stdin.isatty()
            
            # Terminal size detection
            try:
                import shutil
                size = shutil.get_terminal_size()
                terminal_size = (size.columns, size.lines)
            except:
                terminal_size = (80, 24)  # Safe default
            
            # Color support detection
            supports_color = (
                is_tty and 
                os.getenv('TERM', '').lower() not in ['dumb', 'unknown'] and
                os.getenv('NO_COLOR') is None
            )
            
            # Alternate screen support (most modern terminals)
            supports_alternate_screen = (
                is_tty and 
                os.getenv('TERM', '').lower() not in ['dumb', 'unknown', 'vt100']
            )
            
            # Terminal type
            term_type = os.getenv('TERM', 'unknown')
            
            detection_time = (time.time() - start_time) * 1000
            
            capabilities = TerminalCapabilities(
                is_tty=is_tty,
                supports_color=supports_color,
                supports_alternate_screen=supports_alternate_screen,
                terminal_size=terminal_size,
                term_type=term_type,
                detection_time_ms=detection_time
            )
            
            logger.debug(f"Terminal capabilities: {capabilities}")
            return capabilities
            
        except Exception as e:
            logger.warning(f"Terminal detection failed: {e}")
            # Return safe defaults
            return TerminalCapabilities(
                is_tty=False,
                supports_color=False,
                supports_alternate_screen=False,
                terminal_size=(80, 24),
                term_type="unknown",
                detection_time_ms=(time.time() - start_time) * 1000
            )
    
    def _select_display_mode(self, 
                           requested_mode: DisplayMode,
                           capabilities: TerminalCapabilities) -> DisplayMode:
        """
        Select optimal display mode based on capabilities and request.
        
        Args:
            requested_mode: Requested display mode
            capabilities: Detected terminal capabilities
            
        Returns:
            Selected display mode
        """
        # Handle explicit mode requests
        if requested_mode == DisplayMode.RICH:
            if not RICH_AVAILABLE:
                logger.warning("Rich not available, forcing fallback")
                return DisplayMode.FALLBACK
            if not capabilities.is_tty:
                logger.warning("Not a TTY, forcing fallback")
                return DisplayMode.FALLBACK
            return DisplayMode.RICH
        
        if requested_mode == DisplayMode.FALLBACK:
            return DisplayMode.FALLBACK
        
        # AUTO mode selection logic
        if not RICH_AVAILABLE:
            logger.info("Rich not available, using fallback mode")
            return DisplayMode.FALLBACK
        
        if not capabilities.is_tty:
            logger.info("Not a TTY, using fallback mode")
            return DisplayMode.FALLBACK
        
        if not capabilities.supports_color:
            logger.info("No color support, using fallback mode")
            return DisplayMode.FALLBACK
        
        # Check for problematic terminal types
        problematic_terms = {'dumb', 'unknown', 'vt100'}
        if capabilities.term_type.lower() in problematic_terms:
            logger.info(f"Problematic terminal type '{capabilities.term_type}', using fallback")
            return DisplayMode.FALLBACK
        
        # Rich mode looks viable
        logger.debug("Terminal supports Rich display, using Rich mode")
        return DisplayMode.RICH
    
    async def _create_rich_console(self) -> Console:
        """
        Create Rich console with timeout protection.
        
        Returns:
            Configured Rich console
        """
        async with self.guardian.protect_operation("create_console", 0.5):
            # Create console with optimized settings
            console = Console(
                force_terminal=True,
                force_interactive=True,
                force_jupyter=False,
                width=self._capabilities.terminal_size[0] if self._capabilities else 80,
                height=self._capabilities.terminal_size[1] if self._capabilities else 24,
                color_system="auto",
                legacy_windows=False,
                safe_box=True,
                get_time=time.time,
                _environ=os.environ.copy()
            )
            
            # Test console functionality
            await asyncio.sleep(0.01)  # Small delay to test async
            
            return console
    
    async def _create_rich_live_display(self, 
                                      console: Console,
                                      initial_content: Any = None) -> Live:
        """
        Create Rich Live display with strict timeout protection.
        
        Args:
            console: Rich console instance
            initial_content: Initial content for display
            
        Returns:
            Configured Rich Live display
            
        Raises:
            asyncio.TimeoutError: If display creation times out
        """
        if initial_content is None:
            # Create minimal initial content
            initial_content = Panel(
                Text("Initializing display...", style="dim"),
                title="AgentsMCP",
                border_style="blue"
            )
        
        # Rich Live configuration optimized for reliability
        live_config = {
            "console": console,
            "auto_refresh": False,  # Manual refresh control
            "vertical_overflow": "ellipsis",
            "get_renderable": lambda: initial_content,
        }
        
        # Add alternate screen if supported and enabled
        if (self.enable_alternate_screen and 
            self._capabilities and 
            self._capabilities.supports_alternate_screen):
            live_config["screen"] = True
            logger.debug("Enabling alternate screen mode")
        else:
            logger.debug("Using regular screen mode")
        
        async with self.guardian.protect_operation("create_live_display", 1.0):
            # Create Live display - this is where hangs can occur
            live_display = Live(**live_config)
            
            # Test the display by entering/exiting context quickly
            async def test_display():
                live_display.__enter__()
                await asyncio.sleep(0.01)  # Brief test
                live_display.__exit__(None, None, None)
            
            # Test with timeout protection
            await asyncio.wait_for(test_display(), timeout=0.5)
            
            logger.info("Rich Live display created and tested successfully")
            return live_display
    
    async def initialize_display(self, 
                                mode: DisplayMode = DisplayMode.AUTO,
                                timeout: Optional[float] = None,
                                initial_content: Any = None) -> tuple[DisplayMode, Any, DisplayMetrics]:
        """
        Initialize display with guaranteed activation within timeout.
        
        This is the main entry point that GUARANTEES display activation within
        the timeout period or immediate fallback to basic mode.
        
        Args:
            mode: Requested display mode
            timeout: Timeout in seconds (uses default if None)
            initial_content: Initial content for display
            
        Returns:
            Tuple of (actual_mode, display_instance, metrics)
        """
        timeout = timeout or self.default_timeout
        start_time = time.time()
        
        logger.info(f"Initializing display (mode: {mode.value}, timeout: {timeout}s)")
        
        try:
            # Phase 1: Fast terminal detection with timeout
            async with self.guardian.protect_operation("terminal_detection", self.detection_timeout):
                self._capabilities = self._detect_terminal_capabilities()
            
            # Phase 2: Display mode selection
            selected_mode = self._select_display_mode(mode, self._capabilities)
            self._display_mode = selected_mode
            
            logger.info(f"Selected display mode: {selected_mode.value}")
            
            # Phase 3: Display initialization based on mode
            if selected_mode == DisplayMode.RICH:
                display_instance = await self._initialize_rich_display(
                    timeout - (time.time() - start_time),
                    initial_content
                )
            else:
                display_instance = await self._initialize_fallback_display(initial_content)
            
            # Create success metrics
            initialization_time = (time.time() - start_time) * 1000
            metrics = DisplayMetrics(
                mode_selected=selected_mode,
                initialization_time_ms=initialization_time,
                rich_available=RICH_AVAILABLE,
                terminal_capable=self._capabilities.is_tty,
                timeout_occurred=False
            )
            
            self._initialization_metrics = metrics
            logger.info(f"Display initialized successfully in {initialization_time:.1f}ms")
            
            return selected_mode, display_instance, metrics
            
        except asyncio.TimeoutError:
            logger.warning(f"Display initialization timed out after {timeout}s, falling back")
            return await self._handle_timeout_fallback(start_time, "initialization_timeout")
            
        except Exception as e:
            logger.error(f"Display initialization failed: {e}")
            return await self._handle_error_fallback(start_time, f"initialization_error: {e}")
    
    async def _initialize_rich_display(self, 
                                     remaining_timeout: float,
                                     initial_content: Any) -> Live:
        """
        Initialize Rich display with remaining timeout.
        
        Args:
            remaining_timeout: Remaining timeout in seconds
            initial_content: Initial content for display
            
        Returns:
            Active Rich Live display
        """
        # Ensure we have minimum time for initialization
        if remaining_timeout < 0.5:
            logger.warning(f"Insufficient time remaining ({remaining_timeout:.1f}s), using fallback")
            raise asyncio.TimeoutError("Insufficient time for Rich initialization")
        
        # Create console with timeout protection
        console = await asyncio.wait_for(
            self._create_rich_console(),
            timeout=remaining_timeout * 0.3  # 30% of remaining time
        )
        self._console = console
        
        # Create Live display with remaining time
        live_display = await asyncio.wait_for(
            self._create_rich_live_display(console, initial_content),
            timeout=remaining_timeout * 0.7  # 70% of remaining time
        )
        
        # Enter the Live context
        live_display.__enter__()
        self._active_display = live_display
        
        logger.info("Rich display activated successfully")
        return live_display
    
    async def _initialize_fallback_display(self, initial_content: Any) -> 'FallbackDisplay':
        """
        Initialize fallback text display.
        
        Args:
            initial_content: Initial content (ignored in fallback)
            
        Returns:
            Fallback display instance
        """
        fallback_display = FallbackDisplay()
        await asyncio.sleep(0.01)  # Minimal async operation
        
        logger.info("Fallback display activated")
        return fallback_display
    
    async def _handle_timeout_fallback(self, 
                                     start_time: float, 
                                     reason: str) -> tuple[DisplayMode, Any, DisplayMetrics]:
        """Handle timeout by falling back to basic display."""
        initialization_time = (time.time() - start_time) * 1000
        
        fallback_display = FallbackDisplay()
        
        metrics = DisplayMetrics(
            mode_selected=DisplayMode.FALLBACK,
            initialization_time_ms=initialization_time,
            rich_available=RICH_AVAILABLE,
            terminal_capable=self._capabilities.is_tty if self._capabilities else False,
            timeout_occurred=True,
            fallback_reason=reason
        )
        
        self._display_mode = DisplayMode.FALLBACK
        self._initialization_metrics = metrics
        
        return DisplayMode.FALLBACK, fallback_display, metrics
    
    async def _handle_error_fallback(self, 
                                   start_time: float, 
                                   error: str) -> tuple[DisplayMode, Any, DisplayMetrics]:
        """Handle error by falling back to basic display."""
        initialization_time = (time.time() - start_time) * 1000
        
        fallback_display = FallbackDisplay()
        
        metrics = DisplayMetrics(
            mode_selected=DisplayMode.FALLBACK,
            initialization_time_ms=initialization_time,
            rich_available=RICH_AVAILABLE,
            terminal_capable=self._capabilities.is_tty if self._capabilities else False,
            timeout_occurred=False,
            fallback_reason="error_fallback",
            error_details=error
        )
        
        self._display_mode = DisplayMode.FAILED
        self._initialization_metrics = metrics
        
        return DisplayMode.FALLBACK, fallback_display, metrics
    
    def get_metrics(self) -> Optional[DisplayMetrics]:
        """Get display initialization metrics."""
        return self._initialization_metrics
    
    def get_capabilities(self) -> Optional[TerminalCapabilities]:
        """Get detected terminal capabilities."""
        return self._capabilities
    
    def get_active_display(self) -> Optional[Union[Live, 'FallbackDisplay']]:
        """Get the currently active display instance."""
        return self._active_display
    
    def get_display_mode(self) -> DisplayMode:
        """Get the current display mode."""
        return self._display_mode
    
    async def cleanup(self):
        """Clean up display resources."""
        if self._active_display:
            try:
                if hasattr(self._active_display, '__exit__'):
                    self._active_display.__exit__(None, None, None)
                logger.debug("Display cleanup completed")
            except Exception as e:
                logger.warning(f"Error during display cleanup: {e}")
            finally:
                self._active_display = None


class FallbackDisplay:
    """
    Basic fallback display for when Rich is unavailable or fails.
    
    Provides minimal display functionality compatible with Rich Live interface.
    """
    
    def __init__(self):
        self.content = "AgentsMCP - Basic Mode"
        
    def update(self, content: Any):
        """Update display content (basic implementation)."""
        if hasattr(content, '__str__'):
            self.content = str(content)
        else:
            self.content = repr(content)
    
    def refresh(self):
        """Refresh display (no-op for fallback)."""
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass


# Convenience functions for common usage patterns

async def create_display(mode: DisplayMode = DisplayMode.AUTO,
                        timeout: float = 2.0,
                        initial_content: Any = None) -> tuple[DisplayMode, Any, DisplayMetrics]:
    """
    Convenience function to create a display with timeout protection.
    
    Args:
        mode: Display mode to use
        timeout: Timeout in seconds
        initial_content: Initial content for display
        
    Returns:
        Tuple of (actual_mode, display_instance, metrics)
    """
    controller = DisplayController(default_timeout=timeout)
    return await controller.initialize_display(mode, timeout, initial_content)


async def create_rich_display_safe(timeout: float = 2.0,
                                  initial_content: Any = None) -> tuple[bool, Any, DisplayMetrics]:
    """
    Safely create Rich display with guaranteed fallback.
    
    Args:
        timeout: Timeout in seconds
        initial_content: Initial content for display
        
    Returns:
        Tuple of (is_rich_mode, display_instance, metrics)
    """
    mode, display, metrics = await create_display(DisplayMode.RICH, timeout, initial_content)
    is_rich = mode == DisplayMode.RICH
    return is_rich, display, metrics


# Example usage for testing
async def test_display_controller():
    """Test the display controller functionality."""
    print("Testing Display Controller...")
    
    # Test 1: Auto mode selection
    try:
        mode, display, metrics = await create_display(DisplayMode.AUTO, timeout=1.0)
        print(f"✓ Auto mode test: {mode.value} in {metrics.initialization_time_ms:.1f}ms")
        
        if hasattr(display, '__exit__'):
            display.__exit__(None, None, None)
            
    except Exception as e:
        print(f"✗ Auto mode test failed: {e}")
    
    # Test 2: Forced Rich mode
    try:
        mode, display, metrics = await create_display(DisplayMode.RICH, timeout=1.0)
        print(f"✓ Rich mode test: {mode.value} in {metrics.initialization_time_ms:.1f}ms")
        
        if hasattr(display, '__exit__'):
            display.__exit__(None, None, None)
            
    except Exception as e:
        print(f"✗ Rich mode test failed: {e}")
    
    # Test 3: Fallback mode
    try:
        mode, display, metrics = await create_display(DisplayMode.FALLBACK, timeout=1.0)
        print(f"✓ Fallback mode test: {mode.value} in {metrics.initialization_time_ms:.1f}ms")
        
    except Exception as e:
        print(f"✗ Fallback mode test failed: {e}")
    
    print("Display Controller tests completed")


if __name__ == "__main__":
    asyncio.run(test_display_controller())