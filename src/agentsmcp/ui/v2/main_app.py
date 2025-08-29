"""
Main TUI Application Entry Point - Complete v2 TUI System Integration

This module provides the main application entry point that integrates all v2 TUI components
into a complete working system, replacing the broken v1 TUI with reliable functionality.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import signal
from typing import Optional, Any, Dict
from pathlib import Path

from .application_controller import ApplicationController, ApplicationConfig, ApplicationState
from .terminal_manager import create_terminal_manager, TerminalType
from .event_system import create_event_system, Event, EventType
from .input_handler import InputHandler, InputEvent, InputEventType
from .display_renderer import DisplayRenderer
from .layout_engine import create_standard_tui_layout
from .themes import create_theme_manager, detect_preferred_scheme, ColorMode
from .chat_interface import create_chat_interface, ChatInterfaceConfig
from .keyboard_processor import KeyboardProcessor, ShortcutContext
from ..cli_app import CLIConfig

logger = logging.getLogger(__name__)


class MainTUIApp:
    """Main TUI Application - Complete v2 system integration.
    
    This class coordinates all v2 components to provide a working TUI that:
    - Shows typed characters immediately (fixes original typing issue)
    - Prevents scrollback pollution (fixes original display issue) 
    - Provides clean chat interface with real-time input/output
    - Handles exit commands (/quit) and Ctrl+C gracefully
    """
    
    def __init__(self, cli_config: Optional[CLIConfig] = None):
        """Initialize the main TUI application.
        
        Args:
            cli_config: CLI configuration, if None uses defaults
        """
        self.cli_config = cli_config or CLIConfig()
        self.app_controller: Optional[ApplicationController] = None
        self.running = False
        self._shutdown_event = asyncio.Event()
        
        # Component references
        self.terminal_manager = None
        self.event_system = None
        self.input_handler = None
        self.display_renderer = None
        self.layout_engine = None
        self.theme_manager = None
        self.chat_interface = None
        self.keyboard_processor = None
        
    async def initialize(self) -> bool:
        """Initialize all TUI components in correct order with performance optimization.
        
        Returns:
            True if initialization successful, False otherwise
        """
        import time
        start_time = time.time()
        
        try:
            logger.info("Initializing TUI v2 system...")
            
            # PERFORMANCE: Fast concurrent initialization where possible
            init_start = time.time()
            
            # 1. Terminal capabilities and management (critical path)
            self.terminal_manager = create_terminal_manager()
            term_init_start = time.time()
            if not await asyncio.wait_for(self.terminal_manager.initialize(), timeout=1.0):
                logger.error("Failed to initialize terminal manager")
                return False
            term_init_time = time.time() - term_init_start
            
            # Check terminal compatibility
            caps = self.terminal_manager.get_capabilities()
            if caps is None or caps.type == TerminalType.UNKNOWN:
                logger.error("Unknown terminal type detected or initialization failed")
                return False
                
            logger.debug(f"Terminal init: {term_init_time*1000:.1f}ms")
            
            # 2. Parallel initialization of independent components
            async def init_event_system():
                self.event_system = create_event_system()
                await self.event_system.initialize()
            
            async def init_theme_manager():
                color_scheme = self._determine_color_scheme(caps.colors)
                self.theme_manager = create_theme_manager(self.terminal_manager)
                self.theme_manager.set_color_scheme(color_scheme)
            
            # PERFORMANCE: Initialize event system and theme concurrently
            await asyncio.gather(
                init_event_system(),
                init_theme_manager()
            )
            
            # 3. Display renderer (depends on terminal manager)
            display_init_start = time.time()
            self.display_renderer = DisplayRenderer(
                terminal_manager=self.terminal_manager
            )
            await asyncio.wait_for(self.display_renderer.initialize(), timeout=1.0)
            display_init_time = time.time() - display_init_start
            logger.debug(f"Display renderer init: {display_init_time*1000:.1f}ms")
            
            # 4. Layout engine (lightweight, can be synchronous)
            layout_start = time.time()
            self.layout_engine, self.layout_nodes = create_standard_tui_layout(
                terminal_manager=self.terminal_manager
            )
            layout_time = time.time() - layout_start
            logger.debug(f"Layout engine init: {layout_time*1000:.1f}ms")
            
            # 5. Input handling and keyboard processing (can be concurrent)
            async def init_input_handler():
                self.input_handler = InputHandler(
                    terminal_manager=self.terminal_manager,
                    event_system=self.event_system
                )
                await self.input_handler.initialize()
            
            async def init_keyboard_processor():
                # Keyboard processor depends on input handler, so wait
                await init_input_handler()
                self.keyboard_processor = KeyboardProcessor(
                    input_handler=self.input_handler,
                    event_system=self.event_system
                )
                await self.keyboard_processor.initialize()
            
            # Initialize input components
            input_start = time.time()
            await init_keyboard_processor()
            input_time = time.time() - input_start
            logger.debug(f"Input system init: {input_time*1000:.1f}ms")
            
            # 6. Application controller (performance optimized config)
            app_config = ApplicationConfig(
                enable_auto_save=False,  # PERFORMANCE: Disable auto-save during init
                graceful_shutdown_timeout=2.0,  # PERFORMANCE: Faster shutdown
                debug_mode=False
            )
            
            app_start = time.time()
            self.app_controller = ApplicationController(
                config=app_config,
                terminal_manager=self.terminal_manager,
                event_system=self.event_system
            )
            if not await asyncio.wait_for(self.app_controller.startup(), timeout=2.0):
                logger.error("Failed to startup application controller")
                return False
            app_time = time.time() - app_start
            logger.debug(f"Application controller init: {app_time*1000:.1f}ms")
            
            # 7. Chat interface (can be deferred until first use)
            chat_start = time.time()
            await self._initialize_chat_interface_deferred()
            chat_time = time.time() - chat_start
            logger.debug(f"Chat interface init: {chat_time*1000:.1f}ms")
            
            # 8. Setup event handlers (lightweight)
            await self._setup_event_handlers()
            
            # 9. Setup signal handlers (synchronous)
            self._setup_signal_handlers()
            
            # Mark as running after successful initialization
            self.running = True
            
            total_time = time.time() - start_time
            logger.info(f"TUI v2 system initialized successfully in {total_time*1000:.1f}ms")
            
            # PERFORMANCE: Log if initialization is slower than target
            if total_time > 0.5:  # 500ms target
                logger.warning(f"Startup time {total_time*1000:.1f}ms exceeds 500ms target")
            
            return True
            
        except asyncio.TimeoutError as e:
            logger.error(f"TUI initialization timed out: {e}")
            await self.cleanup()
            return False
        except Exception as e:
            logger.exception(f"Failed to initialize TUI v2 system: {e}")
            await self.cleanup()
            return False
    
    async def _initialize_chat_interface_deferred(self):
        """Initialize chat interface with minimal configuration for fast startup."""
        # PERFORMANCE: Minimal config for faster startup
        chat_config = ChatInterfaceConfig(
            enable_history_search=False,  # Enable later
            enable_multiline=True,
            enable_commands=True,
            max_history_messages=100  # Lower limit initially
        )
        
        self.chat_interface = create_chat_interface(
            application_controller=self.app_controller,
            config=chat_config
        )
        await self.chat_interface.initialize()
    
    def _determine_color_scheme(self, colors: int) -> str:
        """Determine appropriate color scheme based on terminal capabilities."""
        if self.cli_config.theme_mode == "light":
            return "light"
        elif self.cli_config.theme_mode == "dark":
            return "dark"
        else:
            # Auto-detect based on terminal and environment
            return detect_preferred_scheme()
    
    async def _setup_event_handlers(self):
        """Setup event handlers for component integration."""
        
        # Keyboard event handling - handles input, commands, and shortcuts
        async def handle_keyboard_event(event: Event):
            if event.event_type == EventType.KEYBOARD:
                key_data = event.data
                
                # Handle quit commands
                if key_data.get('key') == 'c-c' or key_data.get('key') == 'c-d':  # Ctrl+C or Ctrl+D
                    await self.shutdown()
                elif key_data.get('type') == 'command':
                    command = key_data.get('command', '').strip().lower()
                    if command in ['/quit', '/exit', '/q']:
                        await self.shutdown()
                    else:
                        # Pass to chat interface for handling
                        if self.chat_interface and hasattr(self.chat_interface, 'handle_command'):
                            await self.chat_interface.handle_command(command)
                else:
                    # Pass to chat interface for input handling
                    if self.chat_interface and hasattr(self.chat_interface, 'handle_input_event'):
                        await self.chat_interface.handle_input_event(key_data)
        
        # Application event handling - for startup, shutdown, etc.
        async def handle_application_event(event: Event):
            if event.event_type == EventType.APPLICATION:
                action = event.data.get('action')
                if action == 'shutdown':
                    await self.shutdown()
        
        # Register event handlers
        await self.event_system.subscribe(EventType.KEYBOARD, handle_keyboard_event)
        await self.event_system.subscribe(EventType.APPLICATION, handle_application_event)
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def run(self) -> int:
        """Run the main TUI application.
        
        Returns:
            Exit code (0 for success, 1 for error)
        """
        try:
            if not await self.initialize():
                return 1
            
            self.running = True
            logger.info("Starting TUI v2 main application...")
            
            # Display initial interface
            self.display_renderer.clear_all_regions()
            logger.info("TUI v2 interface initialized and ready")
            
            # Main event loop - wait for shutdown
            await self._shutdown_event.wait()
            
            logger.info("TUI v2 application shutting down...")
            return 0
            
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
            return 0
        except Exception as e:
            logger.exception(f"Unexpected error in main application: {e}")
            return 1
        finally:
            await self.cleanup()
    
    async def shutdown(self):
        """Initiate graceful shutdown of the application."""
        if not self.running:
            return
            
        logger.info("Initiating TUI v2 shutdown...")
        self.running = False
        
        # Stop application controller first
        if self.app_controller:
            await self.app_controller.shutdown()
        
        # Signal main loop to exit
        self._shutdown_event.set()
    
    async def cleanup(self):
        """Cleanup all components and resources."""
        logger.info("Cleaning up TUI v2 components...")
        
        # Clean up in reverse order of initialization
        components = [
            self.app_controller,
            self.chat_interface,
            self.keyboard_processor,
            self.input_handler,
            self.display_renderer,
            self.layout_engine,
            self.theme_manager,
            self.event_system,
            self.terminal_manager
        ]
        
        for component in components:
            if component and hasattr(component, 'cleanup'):
                try:
                    await component.cleanup()
                except Exception as e:
                    logger.warning(f"Error during component cleanup: {e}")
        
        logger.info("TUI v2 cleanup complete")


class TUILauncher:
    """Launcher class for integrating with existing CLI infrastructure."""
    
    def __init__(self):
        self.app: Optional[MainTUIApp] = None
    
    async def launch_tui(self, cli_config: Optional[CLIConfig] = None) -> int:
        """Launch the TUI with fallback handling.
        
        Args:
            cli_config: CLI configuration
            
        Returns:
            Exit code
        """
        try:
            # Try to launch v2 TUI
            self.app = MainTUIApp(cli_config)
            return await self.app.run()
            
        except Exception as e:
            logger.exception(f"Failed to launch TUI v2: {e}")
            
            # Fallback to v1 TUI if v2 fails
            try:
                logger.warning("Falling back to legacy TUI v1...")
                return await self._fallback_to_v1(cli_config)
            except Exception as fallback_error:
                logger.exception(f"Fallback to v1 also failed: {fallback_error}")
                print("❌ Both TUI v2 and v1 failed to start")
                print(f"Error: {e}")
                return 1
    
    async def _fallback_to_v1(self, cli_config: Optional[CLIConfig]) -> int:
        """Fallback to v1 TUI system.
        
        Args:
            cli_config: CLI configuration
            
        Returns:
            Exit code
        """
        try:
            from ..modern_tui import ModernTUI, TUIConfig
            
            # Convert CLI config to TUI config
            tui_config = TUIConfig(
                theme=cli_config.theme_mode if cli_config else "auto",
                show_welcome=cli_config.show_welcome if cli_config else True,
                agent_type=cli_config.agent_type if cli_config else "ollama-turbo-coding"
            )
            
            tui = ModernTUI(tui_config)
            await tui.run()
            return 0
            
        except Exception as e:
            logger.exception(f"V1 fallback failed: {e}")
            raise


# Convenience function for CLI integration
async def launch_main_tui(cli_config: Optional[CLIConfig] = None) -> int:
    """Launch the main TUI application with CLI integration.
    
    Args:
        cli_config: CLI configuration, uses defaults if None
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    launcher = TUILauncher()
    return await launcher.launch_tui(cli_config)


# Direct execution support
if __name__ == "__main__":
    async def main():
        app = MainTUIApp()
        return await app.run()
    
    sys.exit(asyncio.run(main()))