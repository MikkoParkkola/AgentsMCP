"""
Application Controller - Main application state management and coordination.

Handles the application lifecycle, command processing, view transitions,
and graceful error handling for the TUI interface.
"""

import asyncio
import logging
import sys
from typing import Optional, Dict, Any, Callable, Set, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from contextlib import asynccontextmanager

from .event_system import AsyncEventSystem, Event, EventType, EventHandler
from .terminal_manager import TerminalManager
from .display_renderer import DisplayRenderer
from .input_handler import InputHandler
from .component_registry import ComponentRegistry
from .keyboard_processor import KeyboardProcessor
from .status_manager import StatusManager, SystemState

logger = logging.getLogger(__name__)


class ApplicationState(Enum):
    """Application lifecycle states."""
    STARTING = "starting"
    RUNNING = "running"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class ApplicationConfig:
    """Configuration for the application controller."""
    enable_auto_save: bool = True
    graceful_shutdown_timeout: float = 5.0
    component_cleanup_timeout: float = 2.0
    enable_crash_recovery: bool = True
    max_error_count: int = 10
    debug_mode: bool = False


@dataclass 
class ViewContext:
    """Context information for a view."""
    name: str
    data: Dict[str, Any] = field(default_factory=dict)
    active: bool = False
    timestamp: datetime = field(default_factory=datetime.now)


class ApplicationController:
    """
    Main application state management and coordination.
    
    Manages the application lifecycle, coordinates between components,
    processes commands, and handles clean shutdown.
    """
    
    def __init__(self, 
                 event_system: Optional[AsyncEventSystem] = None,
                 terminal_manager: Optional[TerminalManager] = None,
                 config: Optional[ApplicationConfig] = None):
        """Initialize the application controller."""
        self.event_system = event_system or AsyncEventSystem()
        self.terminal_manager = terminal_manager or TerminalManager()
        self.config = config or ApplicationConfig()
        
        # Core components
        self.display_renderer: Optional[DisplayRenderer] = None
        self.input_handler: Optional[InputHandler] = None
        self.component_registry: Optional[ComponentRegistry] = None
        self.keyboard_processor: Optional[KeyboardProcessor] = None
        self.status_manager: Optional[StatusManager] = None
        
        # Application state
        self._state = ApplicationState.STOPPED
        self._running = False
        self._shutdown_requested = False
        self._error_count = 0
        self._startup_time: Optional[datetime] = None
        
        # Views and navigation
        self._views: Dict[str, ViewContext] = {}
        self._current_view: Optional[str] = None
        self._view_stack: List[str] = []
        
        # Command processing
        self._commands: Dict[str, Callable] = {}
        self._command_aliases: Dict[str, str] = {}
        self._command_history: List[Dict[str, Any]] = []
        
        # Event handlers
        self._event_handlers: Set[EventHandler] = set()
        
        self._setup_core_commands()
        self._setup_event_handlers()
    
    def _setup_core_commands(self):
        """Setup core application commands."""
        self._commands = {
            'quit': self._cmd_quit,
            'exit': self._cmd_quit,  # Alias for quit
            'help': self._cmd_help,
            'status': self._cmd_status,
            'debug': self._cmd_debug,
            'clear': self._cmd_clear,
            'restart': self._cmd_restart,
        }
        
        # Setup aliases
        self._command_aliases = {
            'q': 'quit',
            'h': 'help',
            '?': 'help',
            'cls': 'clear',
        }
    
    def _setup_event_handlers(self):
        """Setup event handlers for the application."""
        # Create application event handler
        app_handler = ApplicationEventHandler(self)
        self._event_handlers.add(app_handler)
        self.event_system.add_handler(EventType.APPLICATION, app_handler)
        
        # Keyboard event handler for global shortcuts
        keyboard_handler = GlobalKeyboardHandler(self)
        self._event_handlers.add(keyboard_handler)
        self.event_system.add_handler(EventType.KEYBOARD, keyboard_handler)
    
    async def startup(self) -> bool:
        """
        Start the application and initialize all components.
        
        Returns:
            True if startup successful, False otherwise
        """
        if self._state != ApplicationState.STOPPED:
            logger.warning(f"Cannot start application in {self._state} state")
            return False
        
        try:
            self._state = ApplicationState.STARTING
            self._startup_time = datetime.now()
            logger.info("Starting application...")
            
            # Start event system
            await self.event_system.start()
            
            # Initialize status manager first for tracking
            self.status_manager = StatusManager(self.event_system)
            if not await self.status_manager.initialize():
                logger.error("Failed to initialize status manager")
                self._state = ApplicationState.ERROR
                await self._cleanup_on_error()
                return False
            
            await self.status_manager.set_status(SystemState.STARTING, "Initializing display system...")
            
            # Initialize core components
            self.display_renderer = DisplayRenderer(self.terminal_manager)
            if not await self.display_renderer.initialize():
                logger.error("Failed to initialize display renderer")
                await self.status_manager.set_error("Display renderer initialization failed")
                self._state = ApplicationState.ERROR
                await self._cleanup_on_error()
                return False
            
            await self.status_manager.set_status(SystemState.STARTING, "Initializing input system...")
            
            self.input_handler = InputHandler()
            if not self.input_handler.is_available():
                logger.warning("Input handler has limited capabilities")
                await self.status_manager.set_warning("Input handler has limited capabilities")
            
            await self.status_manager.set_status(SystemState.STARTING, "Setting up components...")
            
            self.component_registry = ComponentRegistry(
                self.event_system,
                self.display_renderer
            )
            
            self.keyboard_processor = KeyboardProcessor(
                self.input_handler,
                self.event_system
            )
            
            await self.status_manager.set_status(SystemState.STARTING, "Configuring keyboard shortcuts...")
            
            # Configure keyboard processor with application commands
            await self.keyboard_processor.initialize()
            self._setup_keyboard_shortcuts()
            
            # Update context information
            self.status_manager.update_context(
                agent_name="AgentsMCP",
                connection_status="Local"
            )
            
            # Emit startup event
            startup_event = Event(
                event_type=EventType.APPLICATION,
                data={
                    'action': 'startup',
                    'timestamp': self._startup_time
                }
            )
            await self.event_system.emit_event(startup_event)
            
            self._state = ApplicationState.RUNNING
            self._running = True
            await self.status_manager.set_status(SystemState.READY, "System ready for use")
            logger.info("Application started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start application: {e}")
            self._state = ApplicationState.ERROR
            await self._cleanup_on_error()
            return False
    
    async def shutdown(self, graceful: bool = True) -> bool:
        """
        Shutdown the application gracefully.
        
        Args:
            graceful: Whether to perform graceful shutdown
            
        Returns:
            True if shutdown successful
        """
        if self._state in (ApplicationState.SHUTTING_DOWN, ApplicationState.STOPPED):
            return True
        
        try:
            self._state = ApplicationState.SHUTTING_DOWN
            self._shutdown_requested = True
            self._running = False
            
            logger.info("Shutting down application...")
            
            # Emit shutdown event
            shutdown_event = Event(
                event_type=EventType.APPLICATION,
                data={
                    'action': 'shutdown',
                    'graceful': graceful
                }
            )
            await self.event_system.emit_event(shutdown_event)
            
            if graceful:
                # Graceful shutdown with timeout
                try:
                    await asyncio.wait_for(
                        self._graceful_shutdown(),
                        timeout=self.config.graceful_shutdown_timeout
                    )
                except asyncio.TimeoutError:
                    logger.warning("Graceful shutdown timed out, forcing cleanup")
                    await self._force_shutdown()
            else:
                await self._force_shutdown()
            
            self._state = ApplicationState.STOPPED
            logger.info("Application shutdown complete")
            return True
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            await self._force_shutdown()
            self._state = ApplicationState.ERROR
            return False
    
    async def _graceful_shutdown(self):
        """Perform graceful shutdown of all components."""
        # SECURITY: Cleanup components in reverse order with error isolation
        cleanup_errors = []
        
        try:
            if self.keyboard_processor:
                try:
                    await asyncio.wait_for(
                        self.keyboard_processor.cleanup(), 
                        timeout=self.config.component_cleanup_timeout
                    )
                except Exception as e:
                    cleanup_errors.append(f"KeyboardProcessor cleanup failed: {e}")
        except Exception as e:
            cleanup_errors.append(f"KeyboardProcessor cleanup error: {e}")
        
        try:
            if self.component_registry:
                try:
                    await asyncio.wait_for(
                        self.component_registry.cleanup_all_components(),
                        timeout=self.config.component_cleanup_timeout
                    )
                except Exception as e:
                    cleanup_errors.append(f"ComponentRegistry cleanup failed: {e}")
        except Exception as e:
            cleanup_errors.append(f"ComponentRegistry cleanup error: {e}")
        
        try:
            if self.input_handler:
                # Input handler has synchronous cleanup
                if hasattr(self.input_handler, 'cleanup'):
                    await self.input_handler.cleanup()
                elif hasattr(self.input_handler, 'stop'):
                    self.input_handler.stop()
        except Exception as e:
            cleanup_errors.append(f"InputHandler cleanup failed: {e}")
        
        try:
            if self.display_renderer:
                await asyncio.wait_for(
                    self.display_renderer.cleanup(),
                    timeout=self.config.component_cleanup_timeout
                )
        except Exception as e:
            cleanup_errors.append(f"DisplayRenderer cleanup failed: {e}")
        
        try:
            # Stop event system last
            await asyncio.wait_for(
                self.event_system.stop(),
                timeout=self.config.component_cleanup_timeout
            )
        except Exception as e:
            cleanup_errors.append(f"EventSystem cleanup failed: {e}")
        
        # Log cleanup errors but don't raise
        if cleanup_errors:
            for error in cleanup_errors:
                logger.error(error)
    
    async def _force_shutdown(self):
        """Force immediate shutdown."""
        try:
            if self.display_renderer:
                self.display_renderer.cleanup()
        except:
            pass
        
        try:
            if self.input_handler:
                self.input_handler.stop()
        except:
            pass
        
        try:
            await self.event_system.stop()
        except:
            pass
    
    async def _cleanup_on_error(self):
        """Cleanup after error during startup."""
        try:
            await self._force_shutdown()
        except:
            pass
    
    def _setup_keyboard_shortcuts(self):
        """Setup global keyboard shortcuts."""
        if not self.keyboard_processor:
            return
        
        from .keyboard_processor import KeySequence, ShortcutContext
        
        # Ctrl+C for graceful quit
        self.keyboard_processor.add_shortcut(
            KeySequence(['c'], {'ctrl'}), 
            self._handle_ctrl_c,
            ShortcutContext.GLOBAL,
            "Graceful quit"
        )
        
        # Ctrl+D for quick quit  
        self.keyboard_processor.add_shortcut(
            KeySequence(['d'], {'ctrl'}), 
            self._handle_ctrl_d,
            ShortcutContext.GLOBAL,
            "Quick quit"
        )
        
        # F1 for help
        self.keyboard_processor.add_shortcut(
            KeySequence(['f1']), 
            self._handle_help_key,
            ShortcutContext.GLOBAL,
            "Show help"
        )
    
    async def _handle_ctrl_c(self, event: Event) -> bool:
        """Handle Ctrl+C gracefully."""
        logger.info("Received Ctrl+C, initiating graceful shutdown")
        await self.shutdown(graceful=True)
        return True
    
    async def _handle_ctrl_d(self, event: Event) -> bool:
        """Handle Ctrl+D for quick exit."""
        await self.shutdown(graceful=False)
        return True
    
    async def _handle_help_key(self, event: Event) -> bool:
        """Handle F1 help key."""
        await self.process_command('help')
        return True
    
    async def process_command(self, command_line: str) -> Dict[str, Any]:
        """
        Process a command line input with security validation.
        
        Args:
            command_line: Raw command input
            
        Returns:
            Command execution result
        """
        # Initialize history entry at the beginning
        history_entry = {
            'command': command_line,
            'timestamp': datetime.now(),
            'success': False,
            'error': 'Command not processed'
        }
        
        if not command_line or not command_line.strip():
            return {'success': True, 'result': ''}
        
        # SECURITY: Input sanitization
        sanitized_command = self._sanitize_command_input(command_line)
        if not sanitized_command:
            history_entry['error'] = 'Invalid command input'
            return {'success': False, 'error': 'Invalid command input'}
        
        try:
            # Parse command with length limits
            parts = sanitized_command.strip().split()
            if not parts:
                return {'success': True, 'result': ''}
            
            command = parts[0].lower()
            args = parts[1:]
            
            # Update history entry with parsed args
            history_entry['args'] = args
            
            # SECURITY: Validate command structure
            if not self._validate_command_structure(command, args):
                history_entry['error'] = 'Invalid command format'
                return {'success': False, 'error': 'Invalid command format'}
            
            # Check aliases
            if command in self._command_aliases:
                command = self._command_aliases[command]
            
            # Execute command with timeout protection
            if command in self._commands:
                handler = self._commands[command]
                try:
                    if asyncio.iscoroutinefunction(handler):
                        result = await asyncio.wait_for(
                            handler(*args), 
                            timeout=30.0  # SECURITY: 30s command timeout
                        )
                    else:
                        result = handler(*args)
                    
                    history_entry['success'] = True
                    history_entry['result'] = result
                    
                    return {
                        'success': True,
                        'result': result,
                        'command': command,
                        'args': args
                    }
                except asyncio.TimeoutError:
                    error_msg = f"Command '{command}' timed out"
                    history_entry['success'] = False
                    history_entry['error'] = error_msg
                    return {'success': False, 'error': error_msg, 'command': command}
            else:
                error_msg = f"Unknown command: {command}"
                history_entry['success'] = False
                history_entry['error'] = error_msg
                
                return {
                    'success': False,
                    'error': error_msg,
                    'command': command
                }
        
        except Exception as e:
            logger.error(f"Error processing command '{command_line}': {e}")
            return {
                'success': False,
                'error': str(e),
                'command': command_line
            }
        
        finally:
            # Always add to history
            self._command_history.append(history_entry)
            # Keep history size manageable
            if len(self._command_history) > 100:
                self._command_history = self._command_history[-100:]
    
    def _sanitize_command_input(self, command_line: str) -> str:
        """
        Sanitize command input to prevent terminal injection attacks.
        
        Args:
            command_line: Raw command input
            
        Returns:
            Sanitized command input or empty string if invalid
        """
        if not command_line:
            return ''
        
        # SECURITY: Length limit to prevent DoS
        if len(command_line) > 1024:
            logger.warning("Command input too long, truncating")
            command_line = command_line[:1024]
        
        # SECURITY: Remove dangerous control sequences
        dangerous_sequences = [
            '\x1b',  # ESC character
            '\x07',  # Bell
            '\x00',  # Null byte  
            '\r\n',  # CRLF injection
            '\n\r',  # LFCR injection
        ]
        
        for seq in dangerous_sequences:
            command_line = command_line.replace(seq, '')
        
        # SECURITY: Remove non-printable characters except space and tab
        sanitized = ''.join(c for c in command_line if c.isprintable() or c in ' \t')
        
        return sanitized.strip()
    
    def _validate_command_structure(self, command: str, args: List[str]) -> bool:
        """
        Validate command structure for security.
        
        Args:
            command: Command name
            args: Command arguments
            
        Returns:
            True if command structure is valid
        """
        # SECURITY: Validate command name
        if not command or len(command) > 50:
            return False
        
        # SECURITY: Only allow alphanumeric commands with limited special chars
        if not all(c.isalnum() or c in '-_.' for c in command):
            return False
        
        # SECURITY: Validate arguments
        if len(args) > 20:  # Max 20 arguments
            return False
        
        for arg in args:
            if len(arg) > 256:  # Max 256 chars per argument
                return False
        
        return True
    
    # Command handlers
    async def _cmd_quit(self, *args) -> str:
        """Handle quit/exit command."""
        self._shutdown_requested = True
        await self.shutdown(graceful=True)
        return "Shutting down..."
    
    def _cmd_help(self, *args) -> str:
        """Handle help command with enhanced formatting."""
        if args and args[0] in self._commands:
            # Help for specific command
            return f"Help for command: {args[0]}"
        
        # Get terminal width for formatting
        width = 80
        if self.display_renderer and self.display_renderer.terminal_manager:
            try:
                caps = self.display_renderer.terminal_manager.detect_capabilities()
                width = min(caps.width, 90)
            except:
                pass
        
        help_sections = []
        
        # Header
        if self.display_renderer:
            help_sections.append(self.display_renderer.format_section_header(
                "AgentsMCP System Commands", width, "double"
            ))
        else:
            help_sections.append("=== AgentsMCP System Commands ===")
        
        # Available commands with descriptions
        command_descriptions = {
            'quit': '🚪 Exit the application gracefully',
            'exit': '🚪 Exit the application gracefully (alias for quit)',
            'help': '❓ Show this help information', 
            'status': '📊 Display detailed system status and diagnostics',
            'debug': '🐛 Show debug information for troubleshooting',
            'clear': '🧹 Clear the display and reset interface',
            'restart': '🔄 Restart the application system'
        }
        
        command_items = []
        for cmd in sorted(self._commands.keys()):
            description = command_descriptions.get(cmd, f"Execute {cmd} command")
            command_items.append(f"{description}")
        
        if self.display_renderer:
            help_sections.append(self.display_renderer.format_list_items(command_items, width, "▶"))
        else:
            help_sections.append("\n".join(f"  • {item}" for item in command_items))
        
        # Quick shortcuts
        if self.display_renderer:
            help_sections.append("\n" + self.display_renderer.format_section_header(
                "⌨️ Quick Shortcuts", width, "single"
            ))
            
            shortcut_items = [
                "Ctrl+C - Graceful shutdown",
                "Ctrl+D - Quick exit", 
                "F1 - Show help system"
            ]
            
            help_sections.append(self.display_renderer.format_list_items(shortcut_items, width, "⌨️"))
        
        # Footer
        if self.display_renderer:
            footer_box = self.display_renderer.format_message_box(
                "💡 TIP: Use specific command names for detailed help. "
                "For full chat interface help, switch to chat mode.",
                width, "info"
            )
            help_sections.append("\n" + footer_box)
        else:
            help_sections.append("\nTIP: Use specific command names for detailed help.")
        
        return "\n".join(help_sections)
    
    def _cmd_status(self, *args) -> str:
        """Handle status command with enhanced formatting."""
        # Get terminal width for formatting
        width = 80
        if self.display_renderer and self.display_renderer.terminal_manager:
            try:
                caps = self.display_renderer.terminal_manager.detect_capabilities()
                width = min(caps.width, 100)
            except:
                pass
        
        status_sections = []
        
        # Header
        if self.display_renderer:
            status_sections.append(self.display_renderer.format_section_header(
                "📊 System Status Report", width, "double"
            ))
        else:
            status_sections.append("=== System Status Report ===")
        
        # Application status
        uptime = self.status_manager.get_uptime() if self.status_manager else "Unknown"
        app_status_items = [
            f"🔧 State: {self._state.value}",
            f"⏱️ Uptime: {uptime}",
            f"👁️ Current View: {self._current_view or 'None'}",
            f"⚠️ Error Count: {self._error_count}"
        ]
        
        if self.display_renderer:
            status_sections.append("\n" + self.display_renderer.format_section_header(
                "🏠 Application Status", width, "single"
            ))
            status_sections.append(self.display_renderer.format_list_items(app_status_items, width, "▶"))
        else:
            status_sections.append("\nApplication Status:")
            status_sections.append("\n".join(f"  {item}" for item in app_status_items))
        
        # Component status
        components = []
        if self.component_registry:
            components = list(self.component_registry.get_registered_components().keys())
        
        component_items = [
            f"🎛️ Display Renderer: {'✅ Active' if self.display_renderer else '❌ Not Available'}",
            f"⌨️ Input Handler: {'✅ Active' if self.input_handler and self.input_handler.is_available() else '❌ Limited'}",
            f"📋 Component Registry: {'✅ Active' if self.component_registry else '❌ Not Available'}",
            f"🔤 Keyboard Processor: {'✅ Active' if self.keyboard_processor else '❌ Not Available'}",
            f"📊 Status Manager: {'✅ Active' if self.status_manager else '❌ Not Available'}",
            f"📦 Total Components: {len(components)}"
        ]
        
        if self.display_renderer:
            status_sections.append("\n" + self.display_renderer.format_section_header(
                "🧩 Component Status", width, "single"  
            ))
            status_sections.append(self.display_renderer.format_list_items(component_items, width, "▶"))
        else:
            status_sections.append("\nComponent Status:")
            status_sections.append("\n".join(f"  {item}" for item in component_items))
        
        # Status manager details
        if self.status_manager:
            stats = self.status_manager.get_stats()
            status_items = [
                f"🎯 Current Status: {stats['current_state']} - {stats['current_message']}",
                f"❌ Errors: {stats['error_count']} | ⚠️ Warnings: {stats['warning_count']}",
                f"🔄 Updates: {stats['update_count']} | 📊 Subscribers: {stats['status_subscribers']}",
                f"📈 History Size: {stats['history_size']}"
            ]
            
            if self.display_renderer:
                status_sections.append("\n" + self.display_renderer.format_section_header(
                    "📊 Status Manager Details", width, "single"
                ))
                status_sections.append(self.display_renderer.format_list_items(status_items, width, "▶"))
            else:
                status_sections.append("\nStatus Manager Details:")
                status_sections.append("\n".join(f"  {item}" for item in status_items))
        
        # Memory and performance info
        import psutil
        import os
        try:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()
            
            perf_items = [
                f"💾 Memory Usage: {memory_mb:.1f} MB",
                f"🖥️ CPU Usage: {cpu_percent:.1f}%",
                f"🔗 Open Files: {process.num_fds() if hasattr(process, 'num_fds') else 'N/A'}"
            ]
            
            if self.display_renderer:
                status_sections.append("\n" + self.display_renderer.format_section_header(
                    "⚡ Performance Metrics", width, "single"
                ))
                status_sections.append(self.display_renderer.format_list_items(perf_items, width, "▶"))
        except ImportError:
            # psutil not available
            pass
        except Exception as e:
            logger.warning(f"Could not get performance metrics: {e}")
        
        return "\n".join(status_sections)
    
    def _cmd_debug(self, *args) -> str:
        """Handle debug command."""
        debug_info = {
            'state': self._state.value,
            'running': self._running,
            'views': list(self._views.keys()),
            'current_view': self._current_view,
            'components': len(self.component_registry.get_registered_components()) if self.component_registry else 0,
            'event_stats': self.event_system.get_stats()
        }
        
        return f"Debug Information:\n{debug_info}"
    
    def _cmd_clear(self, *args) -> str:
        """Handle clear command."""
        if self.display_renderer:
            self.display_renderer.clear_all_regions()
        return "Screen cleared"
    
    async def _cmd_restart(self, *args) -> str:
        """Handle restart command."""
        await self.shutdown(graceful=True)
        await asyncio.sleep(0.1)  # Brief pause
        await self.startup()
        return "Application restarted"
    
    # View management
    def register_view(self, name: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Register a new view."""
        if name in self._views:
            logger.warning(f"View {name} already registered")
            return False
        
        self._views[name] = ViewContext(
            name=name,
            data=context or {}
        )
        logger.debug(f"Registered view: {name}")
        return True
    
    async def switch_to_view(self, view_name: str, push_current: bool = True) -> bool:
        """Switch to a different view."""
        if view_name not in self._views:
            logger.error(f"View {view_name} not registered")
            return False
        
        # Deactivate current view
        if self._current_view:
            if push_current:
                self._view_stack.append(self._current_view)
            self._views[self._current_view].active = False
        
        # Activate new view
        self._current_view = view_name
        self._views[view_name].active = True
        
        # Emit view change event
        view_event = Event(
            event_type=EventType.APPLICATION,
            data={
                'action': 'view_change',
                'old_view': self._view_stack[-1] if self._view_stack else None,
                'new_view': view_name
            }
        )
        await self.event_system.emit_event(view_event)
        
        logger.debug(f"Switched to view: {view_name}")
        return True
    
    async def go_back(self) -> bool:
        """Return to previous view."""
        if not self._view_stack:
            return False
        
        previous_view = self._view_stack.pop()
        return await self.switch_to_view(previous_view, push_current=False)
    
    # State and error handling
    def is_running(self) -> bool:
        """Check if application is running."""
        return self._running and self._state == ApplicationState.RUNNING
    
    def get_state(self) -> ApplicationState:
        """Get current application state."""
        return self._state
    
    def report_error(self, error: Exception, context: Optional[str] = None) -> bool:
        """Report an application error."""
        self._error_count += 1
        logger.error(f"Application error {self._error_count}: {error}")
        
        if context:
            logger.error(f"Error context: {context}")
        
        # Check if we should transition to error state
        if self._error_count >= self.config.max_error_count:
            logger.critical("Maximum error count reached, transitioning to error state")
            self._state = ApplicationState.ERROR
            return False
        
        return True
    
    async def run(self) -> int:
        """
        Run the application main loop.
        
        Returns:
            Exit code (0 for success, non-zero for error)
        """
        try:
            if not await self.startup():
                return 1
            
            # Main application loop - just keep running until shutdown
            while self.is_running():
                try:
                    await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                    
                    # Handle any pending events
                    # The actual work is done by event handlers and components
                    
                except KeyboardInterrupt:
                    logger.info("Received keyboard interrupt")
                    break
                except Exception as e:
                    if not self.report_error(e, "main_loop"):
                        break
            
            # Shutdown
            await self.shutdown(graceful=True)
            return 0 if self._state == ApplicationState.STOPPED else 1
            
        except Exception as e:
            logger.error(f"Fatal error in application run: {e}")
            await self._force_shutdown()
            return 1
    
    @asynccontextmanager
    async def app_context(self):
        """Async context manager for application lifecycle."""
        try:
            if not await self.startup():
                raise RuntimeError("Failed to start application")
            yield self
        finally:
            await self.shutdown(graceful=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get application statistics."""
        return {
            'state': self._state.value,
            'uptime': (datetime.now() - self._startup_time).total_seconds() if self._startup_time else 0,
            'views': len(self._views),
            'current_view': self._current_view,
            'commands': len(self._commands),
            'command_history': len(self._command_history),
            'error_count': self._error_count,
            'running': self._running
        }


class ApplicationEventHandler(EventHandler):
    """Event handler for application-level events."""
    
    def __init__(self, controller: ApplicationController):
        super().__init__("ApplicationEventHandler")
        self.controller = controller
    
    async def handle_event(self, event: Event) -> bool:
        """Handle application events."""
        if event.event_type != EventType.APPLICATION:
            return False
        
        action = event.data.get('action')
        if not action:
            return False
        
        if action == 'shutdown_request':
            await self.controller.shutdown(graceful=True)
            return True
        elif action == 'restart_request':
            await self.controller._cmd_restart()
            return True
        elif action == 'error_report':
            error = event.data.get('error')
            context = event.data.get('context')
            if error:
                self.controller.report_error(Exception(error), context)
            return True
        
        return False


class GlobalKeyboardHandler(EventHandler):
    """Global keyboard event handler for application shortcuts."""
    
    def __init__(self, controller: ApplicationController):
        super().__init__("GlobalKeyboardHandler")
        self.controller = controller
    
    async def handle_event(self, event: Event) -> bool:
        """Handle global keyboard shortcuts."""
        if event.event_type != EventType.KEYBOARD:
            return False
        
        key = event.data.get('key')
        if not key:
            return False
        
        # Let the keyboard processor handle shortcuts
        # This handler is for any additional global processing
        return False