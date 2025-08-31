"""
Chat Interface - Main chat UI coordination component.

This component coordinates the chat input and history components to provide
a complete chat interface that integrates with the existing AgentsMCP backend.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .event_system import AsyncEventSystem, Event, EventType
from .application_controller import ApplicationController
from .display_renderer import DisplayRenderer
from .input_handler import InputHandler
from .keyboard_processor import KeyboardProcessor
from .layout_engine import LayoutEngine
from .components.chat_input import ChatInput, ChatInputEvent, create_chat_input
from .components.chat_history import ChatHistory, ChatMessage, MessageRole, MessageStatus, create_chat_history

logger = logging.getLogger(__name__)


class ChatState(Enum):
    """State of the chat interface."""
    IDLE = "idle"
    WAITING_INPUT = "waiting_input"
    PROCESSING = "processing"
    STREAMING = "streaming"
    ERROR = "error"


@dataclass
class ChatInterfaceConfig:
    """Configuration for the chat interface."""
    enable_history_search: bool = True
    enable_multiline: bool = True
    enable_commands: bool = True
    auto_scroll_history: bool = True
    show_timestamps: bool = True
    max_history_messages: int = 10000
    typing_timeout: float = 30.0  # Timeout for typing indicator


class ChatInterface:
    """
    Main chat interface that coordinates input, history, and backend integration.
    
    This component provides the complete chat experience by:
    1. Coordinating chat input and history components
    2. Integrating with AgentsMCP conversation backend
    3. Managing layout and display
    4. Handling commands (/quit, /help, etc.)
    5. Providing status indicators and feedback
    """
    
    def __init__(self,
                 application_controller: ApplicationController,
                 config: Optional[ChatInterfaceConfig] = None,
                 status_manager=None,
                 display_renderer=None):
        """Initialize the chat interface."""
        self.app_controller = application_controller
        self.config = config or ChatInterfaceConfig()
        
        # Get references to core systems
        self.event_system = application_controller.event_system
        self.display_renderer = display_renderer or application_controller.display_renderer
        self.input_handler = application_controller.input_handler
        self.status_manager = status_manager or application_controller.status_manager
        self.keyboard_processor = application_controller.keyboard_processor
        
        # Chat components
        self.chat_input: Optional[ChatInput] = None
        self.chat_history: Optional[ChatHistory] = None
        
        # Layout and display
        self.layout_engine: Optional[LayoutEngine] = None
        self.display_region: Optional[Dict[str, Any]] = None
        
        # Chat state
        self.state = ChatState.IDLE
        self.current_message_id: Optional[str] = None
        self.typing_task: Optional[asyncio.Task] = None
        
        # Backend integration
        self.conversation_manager: Optional[Any] = None  # Will be imported dynamically
        self.orchestrator_integration: Optional[Any] = None  # Orchestrator integration for strict communication isolation
        
        # Status and feedback
        self.status_message = ""
        self.last_error: Optional[str] = None
        self._spinner_task: Optional[asyncio.Task] = None
        self._spinner_active: bool = False
        # Input-line spinner (mirrors status spinner)
        self._input_spinner_char: str = ''
        
        # Callbacks
        self._callbacks: Dict[str, Callable] = {}
        
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the chat interface and all components."""
        if self._initialized:
            return True
        
        try:
            logger.info("Initializing chat interface...")
            
            # Initialize backend connection
            await self._initialize_conversation_backend()
            
            # Create and initialize chat components
            self.chat_input = create_chat_input(
                self.event_system,
                self.input_handler,
                self.keyboard_processor
            )
            
            self.chat_history = create_chat_history(
                self.event_system,
                self.display_renderer,
                self.config.max_history_messages
            )
            
            # Initialize components
            if not await self.chat_input.initialize():
                logger.error("Failed to initialize chat input")
                return False
            
            if not await self.chat_history.initialize():
                logger.error("Failed to initialize chat history")
                return False
            
            # Setup layout
            await self._setup_layout()
            
            # Connect component events
            self._connect_component_events()
            
            # Setup keyboard shortcuts
            await self._setup_chat_shortcuts()
            
            # Add welcome message
            await self._add_welcome_message()
            
            # Register with application controller
            self.app_controller.register_view("chat", {"interface": self})
            
            self._initialized = True
            logger.info("Chat interface initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize chat interface: {e}")
            await self._cleanup_on_error()
            return False
    
    async def _initialize_conversation_backend(self):
        """Initialize connection to orchestrated conversation backend."""
        try:
            # Use orchestrator integration for strict communication isolation
            from .orchestrator_integration import initialize_orchestrator_integration
            
            # Initialize orchestrator integration
            self.orchestrator_integration = await initialize_orchestrator_integration(
                command_interface=None,  # Will be handled via application controller
                theme_manager=None,
                agent_manager=None
            )
            
            # Keep reference to orchestrated conversation manager
            self.conversation_manager = self.orchestrator_integration.orchestrated_conversation
            
            logger.info("Orchestrated conversation backend initialized - strict communication isolation active")
            
        except Exception as e:
            logger.warning(f"Could not initialize orchestrated conversation backend: {e}")
            # Fallback to regular conversation manager
            try:
                from ...conversation.conversation import ConversationManager
                self.conversation_manager = ConversationManager(
                    command_interface=None,
                    theme_manager=None,
                    agent_manager=None
                )
                logger.info("Fallback conversation backend initialized")
            except Exception as fallback_error:
                logger.error(f"Fallback conversation backend failed: {fallback_error}")
                # Continue without backend - basic functionality will still work
    
    async def _setup_layout(self):
        """Setup the chat interface layout."""
        if not self.display_renderer:
            logger.warning("No display renderer available, using fallback layout")
            return
        
        try:
            # Get terminal capabilities
            caps = self.display_renderer.terminal_manager.detect_capabilities()

            # Stabilize: single-line input + single-line status
            status_height = 1
            input_height = 1
            # History fills remaining area
            history_height = max(3, caps.height - (status_height + input_height))
            
            self.display_region = {
                "history": {
                    "start_line": 0,
                    "end_line": history_height - 1,
                    "width": caps.width
                },
                "input": {
                    "start_line": history_height,
                    "end_line": history_height + input_height - 1,
                    "width": caps.width
                },
                "status": {
                    "start_line": history_height + input_height,
                    "end_line": history_height + input_height,
                    "width": caps.width
                }
            }
            
            logger.debug(f"Chat layout configured: {history_height} history + {input_height} input lines")
            # Define renderer regions for input and status as well
            try:
                # Input region: single-line just above status
                self.display_renderer.define_region(
                    "chat_input",
                    x=0,
                    y=history_height,
                    width=caps.width,
                    height=1
                )
                # Status bar region: single bottom line
                self.display_renderer.define_region(
                    "status_bar",
                    x=0,
                    y=history_height + input_height,
                    width=caps.width,
                    height=1
                )
            except Exception as e:
                logger.warning(f"Failed to define input/status regions: {e}")
            
        except Exception as e:
            logger.error(f"Error setting up chat layout: {e}")
    
    def _connect_component_events(self):
        """Connect events between components."""
        if self.chat_input:
            # Handle input submission
            self.chat_input.add_callback("submit", self._handle_input_submit)
            self.chat_input.add_callback("text_change", self._handle_input_change)
            self.chat_input.add_callback("mode_change", self._handle_input_mode_change)
        
        if self.chat_history:
            # Handle history events
            self.chat_history.add_callback("message_added", self._handle_message_added)
            self.chat_history.add_callback("message_updated", self._handle_message_updated)
    
    async def _setup_chat_shortcuts(self):
        """Setup chat-specific keyboard shortcuts."""
        if not self.keyboard_processor:
            return
        
        from .keyboard_processor import KeySequence, ShortcutContext
        
        # Ctrl+L to clear chat history
        self.keyboard_processor.add_shortcut(
            KeySequence(['l'], {'ctrl'}),
            self._handle_clear_history,
            ShortcutContext.CHAT,
            "Clear chat history"
        )
        
        # Page Up/Down for history scrolling
        self.keyboard_processor.add_shortcut(
            KeySequence(['page_up']),
            self._handle_scroll_up,
            ShortcutContext.CHAT,
            "Scroll history up"
        )
        
        self.keyboard_processor.add_shortcut(
            KeySequence(['page_down']),
            self._handle_scroll_down,
            ShortcutContext.CHAT,
            "Scroll history down"
        )
        
        # Ctrl+F for search
        self.keyboard_processor.add_shortcut(
            KeySequence(['f'], {'ctrl'}),
            self._handle_search,
            ShortcutContext.CHAT,
            "Search chat history"
        )
    
    async def _add_welcome_message(self):
        """Add welcome message to chat history."""
        if self.chat_history:
            welcome_text = """Welcome to AgentsMCP Chat Interface!

🤖 You can chat naturally with AI agents
💬 Type your message and press Enter
📝 Use Shift+Enter for multi-line input
⚙️ Commands start with / (try /help)
🔍 Ctrl+F to search, Ctrl+L to clear
❌ /quit or Ctrl+C to exit

Start typing to begin..."""
            
            await self.chat_history.add_message(
                welcome_text,
                MessageRole.SYSTEM,
                metadata={"welcome": True}
            )
    
    async def activate(self):
        """Activate the chat interface."""
        if not self._initialized:
            logger.error("Cannot activate uninitialized chat interface")
            return False
        
        try:
            # Activate components
            if self.chat_input:
                await self.chat_input.activate()
                # Disable direct stdout echo and printing; render via renderer
                try:
                    if hasattr(self.chat_input, 'input_handler') and self.chat_input.input_handler:
                        self.chat_input.input_handler.set_echo(False)
                    if hasattr(self.chat_input, '_immediate_display'):
                        self.chat_input._immediate_display = False
                except Exception:
                    pass
            
            if self.chat_history:
                self.chat_history.set_visibility(True)
            
            # Switch to chat view
            await self.app_controller.switch_to_view("chat")

            # Set initial state
            self.state = ChatState.WAITING_INPUT
            await self._update_status("Ready for input")
            # Initial input line paint
            await self._handle_input_change(ChatInputEvent(event_type="text_change", text=self.chat_input.state.text))

            # Ensure keyboard processor routes keys to text input component
            try:
                if self.keyboard_processor:
                    self.keyboard_processor.enter_text_input_mode("chat_input")
            except Exception:
                pass
            
            logger.info("Chat interface activated")
            return True
            
        except Exception as e:
            logger.error(f"Failed to activate chat interface: {e}")
            return False
    
    async def deactivate(self):
        """Deactivate the chat interface."""
        try:
            # Deactivate components
            if self.chat_input:
                await self.chat_input.deactivate()
            
            if self.chat_history:
                self.chat_history.set_visibility(False)
            
            # Cancel any ongoing operations
            if self.typing_task and not self.typing_task.done():
                self.typing_task.cancel()
            
            self.state = ChatState.IDLE

            # Leave text input mode
            try:
                if self.keyboard_processor:
                    self.keyboard_processor.exit_text_input_mode()
            except Exception:
                pass
            
            logger.info("Chat interface deactivated")
            
        except Exception as e:
            logger.error(f"Error deactivating chat interface: {e}")
    
    async def _handle_input_submit(self, event: ChatInputEvent):
        """Handle input submission from chat input component."""
        text = event.text.strip()
        is_command = event.data and event.data.get("is_command", False)
        
        if not text:
            return
        
        logger.info(f"Input submitted: '{text}' (command: {is_command})")
        
        # Add user message to history
        user_msg_id = await self.chat_history.add_message(text, MessageRole.USER)
        
        # Handle commands
        if is_command:
            await self._handle_command(text)
            return
        
        # Process regular chat message
        await self._process_chat_message(text, user_msg_id)
        # Refresh input line after submission
        await self._handle_input_change(ChatInputEvent(event_type="text_change", text=self.chat_input.state.text))
    
    async def _handle_input_change(self, event: ChatInputEvent):
        """Handle input text changes by updating the input region via the renderer."""
        try:
            if not self.display_renderer or not self.display_region:
                return
            # Compose single-line input view with cursor
            full_text = self.chat_input.state.text
            pos = self.chat_input.state.cursor_position
            # Add spinner char to prompt when active
            prompt = (f"{self._input_spinner_char} " if self._input_spinner_char else "") + self.chat_input.prompt_text
            cursor_char = self.chat_input.cursor_char if (self.chat_input.state.show_cursor and getattr(self.chat_input, '_cursor_visible', True)) else ""
            if cursor_char:
                display_text = full_text[:pos] + cursor_char + full_text[pos:]
            else:
                display_text = full_text
            caps = self.display_renderer.terminal_manager.detect_capabilities()
            width = caps.width
            line = f"{prompt}{display_text}"
            if len(line) < width:
                line = line + (" " * (width - len(line)))
            self.display_renderer.update_region("chat_input", line[:width], force=True)
        except Exception as e:
            logger.debug(f"Input change render failed: {e}")
    
    async def _handle_input_mode_change(self, event: ChatInputEvent):
        """Handle input mode changes."""
        mode = event.mode.value if event.mode else "unknown"
        await self._update_status(f"Input mode: {mode}")
    
    async def _handle_command(self, command_text: str):
        """Handle chat commands."""
        parts = command_text[1:].split()  # Remove / and split
        if not parts:
            return
        
        cmd = parts[0].lower()
        args = parts[1:]
        
        try:
            if cmd in ["quit", "exit", "q"]:
                await self._handle_quit_command()
            elif cmd in ["help", "h", "?"]:
                await self._handle_help_command()
            elif cmd in ["clear", "cls", "c"]:
                await self._handle_clear_command()
            elif cmd == "status":
                await self._handle_status_command()
            elif cmd == "search" and args:
                await self._handle_search_command(" ".join(args))
            else:
                # Try to delegate to application controller
                result = await self.app_controller.process_command(command_text[1:])
                if result.get("success"):
                    response = result.get("result", "Command executed")
                    await self.chat_history.add_message(response, MessageRole.SYSTEM)
                else:
                    error = result.get("error", "Unknown command")
                    await self._display_error_with_recovery("Command Error", error, [
                        "1. Check the command spelling and try again",
                        "2. Use /help to see all available commands",
                        "3. Try the command without arguments for usage info"
                    ])
                    
        except Exception as e:
            logger.error(f"Error handling command '{cmd}': {e}")
            await self._display_error_with_recovery("Command Execution Error", str(e), [
                "1. Check system status with /status command",
                "2. Try restarting the application with /restart",
                "3. Report the issue if it persists",
                "4. Use /debug for detailed system information"
            ])
    
    async def _handle_quit_command(self):
        """Handle quit command."""
        await self.chat_history.add_message("Goodbye! 👋", MessageRole.SYSTEM)
        await asyncio.sleep(0.5)  # Brief pause to show message
        await self.app_controller.shutdown(graceful=True)
    
    async def _handle_help_command(self):
        """Handle help command with enhanced categorized help system."""
        # Use display renderer formatting if available
        width = 80
        if self.display_renderer and self.display_renderer.terminal_manager:
            try:
                caps = self.display_renderer.terminal_manager.detect_capabilities()
                width = min(caps.width, 100)  # Max width for readability
            except:
                pass
        
        help_sections = []
        
        # Header
        help_sections.append(self.display_renderer.format_section_header(
            "AgentsMCP TUI v2 - Help System", width, "double"
        ))
        
        # Commands section
        command_items = [
            "🚀 /quit or /exit - Exit the application gracefully",
            "🚀 /clear or /cls - Clear chat history and start fresh", 
            "🚀 /help or /h - Show this comprehensive help system",
            "🚀 /status - Display detailed system status and diagnostics",
            "🚀 /search <query> - Search through chat history",
            "🚀 /restart - Restart the application",
            "🚀 /debug - Show debug information for troubleshooting"
        ]
        
        help_sections.append("\n" + self.display_renderer.format_section_header(
            "🚀 Commands", width, "single"
        ))
        help_sections.append(self.display_renderer.format_list_items(command_items, width))
        
        # AI Interaction section  
        ai_items = [
            "💬 Chat naturally - Ask questions in plain English",
            "💬 Multi-turn conversations - Context is maintained",
            "💬 Ask for explanations, code help, or analysis",
            "💬 Request system information or configuration help",
            "💬 Get assistance with AgentsMCP features and usage"
        ]
        
        help_sections.append("\n" + self.display_renderer.format_section_header(
            "🤖 AI Interaction", width, "single"  
        ))
        help_sections.append(self.display_renderer.format_list_items(ai_items, width))
        
        # Shortcuts section
        shortcut_items = [
            "⌨️ Enter - Send message or execute command",
            "⌨️ Shift+Enter - Multi-line input mode",
            "⌨️ ↑/↓ arrows - Navigate through input history", 
            "⌨️ Ctrl+C - Graceful exit or cancel current operation",
            "⌨️ Ctrl+D - Quick exit",
            "⌨️ Ctrl+L - Clear chat history instantly",
            "⌨️ Ctrl+F - Search chat history",
            "⌨️ Page Up/Down - Scroll through chat history",
            "⌨️ F1 - Show this help system"
        ]
        
        help_sections.append("\n" + self.display_renderer.format_section_header(
            "⚙️ Keyboard Shortcuts", width, "single"
        ))
        help_sections.append(self.display_renderer.format_list_items(shortcut_items, width))
        
        # System section
        system_items = [
            "📊 Status bar shows current system state with icons",
            "📊 Context information displays active agent and model",
            "📊 Error messages include recovery steps and guidance",
            "📊 All operations are logged for debugging purposes",
            "📊 Terminal compatibility mode adapts to your environment"
        ]
        
        help_sections.append("\n" + self.display_renderer.format_section_header(
            "📊 System Information", width, "single"
        ))
        help_sections.append(self.display_renderer.format_list_items(system_items, width))
        
        # Usage examples
        example_queries = [
            "\"What's the current system status?\"",
            "\"Show me the available models and agents\"", 
            "\"Help me configure AgentsMCP for my project\"",
            "\"Analyze the errors in the current conversation\"",
            "\"What are the best practices for using this system?\"",
            "\"Explain how the MCP protocol works\""
        ]
        
        help_sections.append("\n" + self.display_renderer.format_section_header(
            "💡 Example Queries", width, "single"
        ))
        help_sections.append(self.display_renderer.format_list_items(example_queries, width))
        
        # Footer with tips
        tips_box = self.display_renderer.format_message_box(
            "💡 TIP: Start typing to begin a conversation, or use /help <topic> for specific help sections. "
            "The system will adapt to your terminal's capabilities automatically.",
            width, "info"
        )
        help_sections.append("\n" + tips_box)
        
        # Combine all sections
        complete_help = "\n".join(help_sections)
        
        await self.chat_history.add_message(complete_help, MessageRole.SYSTEM)
    
    async def _handle_clear_command(self):
        """Handle clear command."""
        if self.chat_history:
            self.chat_history.clear_history()
            await self._add_welcome_message()
    
    async def _handle_status_command(self):
        """Handle status command."""
        stats = self.get_stats()
        
        status_text = f"""📊 **Chat Interface Status**

**State:** {self.state.value}
**Messages:** {stats['message_count']}
**Input Active:** {stats['input_active']}
**Backend:** {'Connected' if self.conversation_manager else 'Not Available'}
**Memory Usage:** {stats['memory_usage_mb']:.1f} MB
**Uptime:** {stats.get('uptime', 'Unknown')}

**Components:**
• Chat Input: {'✅ Active' if stats['input_active'] else '❌ Inactive'}
• Chat History: {'✅ Visible' if stats['history_visible'] else '❌ Hidden'}
• Display Renderer: {'✅ Available' if self.display_renderer else '❌ Not Available'}
• Conversation Backend: {'✅ Connected' if self.conversation_manager else '❌ Not Available'}
"""
        
        await self.chat_history.add_message(status_text, MessageRole.SYSTEM)
    
    async def _handle_search_command(self, query: str):
        """Handle search command."""
        if not self.chat_history:
            return
        
        results = await self.chat_history.search_messages(query)
        
        if results:
            await self.chat_history.add_message(
                f"🔍 Found {len(results)} messages matching '{query}'. Use Ctrl+F to navigate.",
                MessageRole.SYSTEM
            )
        else:
            await self.chat_history.add_message(
                f"🔍 No messages found matching '{query}'.",
                MessageRole.SYSTEM
            )
    
    async def _process_chat_message(self, text: str, user_msg_id: str):
        """Process a regular chat message."""
        if not self.conversation_manager:
            await self._display_error_with_recovery(
                "Backend Connection Error",
                "The chat backend is not available. AI conversation features are currently unavailable.",
                [
                    "1. Check system status with /status command",
                    "2. Restart the application with /restart",
                    "3. Use commands (starting with /) for basic functionality",
                    "4. Verify network connectivity if using remote services"
                ]
            )
            return
        
        try:
            self.state = ChatState.PROCESSING
            await self._update_status("Processing...")
            # Start a lightweight spinner that updates the status bar and input line
            try:
                if self.display_renderer and self.display_region:
                    self._start_status_spinner()
            except Exception:
                pass
            
            # Create AI response message  
            ai_msg_id = await self.chat_history.add_message(
                "Thinking...",
                MessageRole.AI,
                status=MessageStatus.STREAMING
            )
            self.current_message_id = ai_msg_id
            
            # Start timeout task
            self.typing_task = asyncio.create_task(
                asyncio.sleep(self.config.typing_timeout)
            )
            
            try:
                # Process with conversation manager
                response = await asyncio.wait_for(
                    self.conversation_manager.process_input(text),
                    timeout=self.config.typing_timeout
                )
                
                # Update AI message with response
                await self.chat_history.update_message(
                    ai_msg_id,
                    content=response,
                    status=MessageStatus.COMPLETE
                )
                
                self.state = ChatState.WAITING_INPUT
                await self._update_status("Ready")

            except asyncio.TimeoutError:
                await self.chat_history.update_message(
                    ai_msg_id,
                    content="Response processing timed out.",
                    status=MessageStatus.ERROR
                )
                
                await self._display_error_with_recovery(
                    "Processing Timeout",
                    f"The AI response took longer than {self.config.typing_timeout} seconds to process.",
                    [
                        "1. Try asking a simpler or shorter question",
                        "2. Check system status with /status",
                        "3. Verify network connectivity for remote models",
                        "4. Consider restarting with /restart if timeouts persist"
                    ]
                )
                
                self.state = ChatState.ERROR
                await self._update_status("Timeout error")

        except Exception as e:
            logger.error(f"Error processing chat message: {e}")
            
            if self.current_message_id:
                await self.chat_history.update_message(
                    self.current_message_id,
                    content="An error occurred while processing your request.",
                    status=MessageStatus.ERROR
                )
            
            # Provide detailed error information with recovery steps
            await self._display_error_with_recovery(
                "Chat Processing Error",
                f"An unexpected error occurred: {str(e)}",
                [
                    "1. Check if the error persists with a different question",
                    "2. Use /status to check system health",
                    "3. Try restarting the application with /restart",
                    "4. Review system logs with /debug",
                    "5. Report persistent errors to support"
                ]
            )
            
            self.state = ChatState.ERROR
            await self._update_status(f"Error: {str(e)}")
        
        finally:
            # Stop spinner
            try:
                self._stop_status_spinner()
            except Exception:
                pass
            # Cancel timeout task
            if self.typing_task and not self.typing_task.done():
                self.typing_task.cancel()
            self.current_message_id = None

    def _start_status_spinner(self):
        if self._spinner_active:
            return
        self._spinner_active = True
        frames = ['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏']

        async def _spin():
            idx = 0
            try:
                while self._spinner_active:
                    ch = frames[idx % len(frames)]
                    idx += 1
                    # Update status bar region directly for responsiveness (single line)
                    try:
                        caps = self.display_renderer.terminal_manager.detect_capabilities()
                        content = f"{ch} Processing…  |  /help  |  /quit"
                        if len(content) < caps.width:
                            content = content + (" " * (caps.width - len(content)))
                        self.display_renderer.update_region('status_bar', content[:caps.width], force=True)
                        # Mirror spinner in the prompt line
                        self._input_spinner_char = ch
                        await self._handle_input_change(ChatInputEvent(event_type="text_change", text=self.chat_input.state.text))
                    except Exception:
                        pass
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                pass

        self._spinner_task = asyncio.create_task(_spin())

    def _stop_status_spinner(self):
        self._spinner_active = False
        if self._spinner_task:
            try:
                self._spinner_task.cancel()
            except Exception:
                pass
            self._spinner_task = None
        # Clear prompt spinner
        self._input_spinner_char = ''
        try:
            # Force a final input-line repaint without spinner
            asyncio.create_task(self._handle_input_change(ChatInputEvent(event_type="text_change", text=self.chat_input.state.text)))
        except Exception:
            pass
    
    async def _handle_message_added(self, message: ChatMessage):
        """Handle message added to history."""
        if "message_added" in self._callbacks:
            await self._callbacks["message_added"](message)
    
    async def _handle_message_updated(self, message: ChatMessage):
        """Handle message updated in history."""
        if "message_updated" in self._callbacks:
            await self._callbacks["message_updated"](message)
    
    # Keyboard shortcut handlers
    async def _handle_clear_history(self, event: Event) -> bool:
        """Handle Ctrl+L to clear history."""
        await self._handle_clear_command()
        return True
    
    async def _handle_scroll_up(self, event: Event) -> bool:
        """Handle Page Up to scroll history up."""
        if self.chat_history:
            self.chat_history.scroll_up(1)
        return True
    
    async def _handle_scroll_down(self, event: Event) -> bool:
        """Handle Page Down to scroll history down."""
        if self.chat_history:
            self.chat_history.scroll_down(1)
        return True
    
    async def _handle_search(self, event: Event) -> bool:
        """Handle Ctrl+F to search."""
        # For now, show search help - in future could open search UI
        await self.chat_history.add_message(
            "🔍 Use `/search <query>` to search chat history.",
            MessageRole.SYSTEM
        )
        return True
    
    async def _display_error_with_recovery(self, title: str, error_message: str, recovery_steps: List[str]):
        """Display professional error message with recovery guidance."""
        # Get terminal width for formatting
        width = 80
        if self.display_renderer and self.display_renderer.terminal_manager:
            try:
                caps = self.display_renderer.terminal_manager.detect_capabilities()
                width = min(caps.width, 90)
            except:
                pass
        
        # Format the error message
        error_sections = []
        
        # Error header
        error_sections.append(self.display_renderer.format_section_header(
            f"❌ {title}", width, "single"
        ))
        
        # Error description
        error_box = self.display_renderer.format_message_box(
            error_message, width, "error"
        )
        error_sections.append(error_box)
        
        # Recovery steps
        if recovery_steps:
            error_sections.append("\n" + self.display_renderer.format_section_header(
                "🔧 Recovery Steps", width, "single"
            ))
            error_sections.append(self.display_renderer.format_list_items(recovery_steps, width, "→"))
        
        # Support information
        support_box = self.display_renderer.format_message_box(
            "Need additional help? Use '/status' for system diagnostics or '/help' for comprehensive guidance.",
            width, "info"
        )
        error_sections.append("\n" + support_box)
        
        # Combine and send
        complete_error = "\n".join(error_sections)
        await self.chat_history.add_message(complete_error, MessageRole.ERROR)
    
    async def _display_system_check_failure(self, component: str, details: str):
        """Display system check failure with diagnostic guidance."""
        width = 80
        if self.display_renderer and self.display_renderer.terminal_manager:
            try:
                caps = self.display_renderer.terminal_manager.detect_capabilities()  
                width = min(caps.width, 90)
            except:
                pass
        
        check_sections = []
        
        # Header
        check_sections.append(self.display_renderer.format_section_header(
            f"⚠️ System Check - {component} Issue", width, "single"
        ))
        
        # Problem description
        problem_box = self.display_renderer.format_message_box(
            f"Issue detected with {component}: {details}", width, "warning"
        )
        check_sections.append(problem_box)
        
        # System diagnostics
        diagnostic_steps = [
            "1. Check system status: /status",
            "2. Verify system resources and connectivity",
            "3. Review recent error logs: /debug", 
            "4. Restart affected components: /restart",
            "5. Contact support if issue persists"
        ]
        
        check_sections.append("\n" + self.display_renderer.format_section_header(
            "🔍 Diagnostic Steps", width, "single"
        ))
        check_sections.append(self.display_renderer.format_list_items(diagnostic_steps, width, "▶"))
        
        # Send to chat
        complete_check = "\n".join(check_sections)
        await self.chat_history.add_message(complete_check, MessageRole.ERROR)
    
    async def _update_status(self, message: str):
        """Update status message and paint status bar."""
        self.status_message = message

        # Try to paint a single-line status bar with simple, helpful affordances
        try:
            if self.display_renderer:
                caps = self.display_renderer.terminal_manager.detect_capabilities()
                width = caps.width
                content = f"{message}  |  /help  |  /quit"
                if len(content) < width:
                    content = content + (" " * (width - len(content)))
                self.display_renderer.define_region("status_bar", 0, max(0, caps.height - 1), width, 1)
                self.display_renderer.update_region("status_bar", content[:width], force=True)
        except Exception:
            pass
        
        # Emit status update event
        event = Event(
            event_type=EventType.CUSTOM,
            data={
                "component": "chat_interface",
                "action": "status_update",
                "message": message,
                "state": self.state.value
            }
        )
        await self.event_system.emit_event(event)
        
        logger.debug(f"Chat status: {message}")
    
    async def _cleanup_on_error(self):
        """Cleanup on initialization error."""
        try:
            if self.chat_input:
                await self.chat_input.cleanup()
            if self.chat_history:
                await self.chat_history.cleanup()
        except:
            pass
    
    def add_callback(self, event_type: str, callback: Callable):
        """Add callback for specific event types."""
        self._callbacks[event_type] = callback
    
    def remove_callback(self, event_type: str):
        """Remove callback for specific event types."""
        if event_type in self._callbacks:
            del self._callbacks[event_type]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chat interface statistics."""
        stats = {
            "state": self.state.value,
            "initialized": self._initialized,
            "status_message": self.status_message,
            "last_error": self.last_error,
            "backend_connected": self.conversation_manager is not None
        }
        
        if self.chat_input:
            input_stats = self.chat_input.get_stats()
            stats.update({
                "input_active": input_stats["active"],
                "input_text_length": input_stats["text_length"]
            })
        else:
            stats.update({
                "input_active": False,
                "input_text_length": 0
            })
        
        if self.chat_history:
            history_stats = self.chat_history.get_stats()
            stats.update({
                "message_count": history_stats["message_count"],
                "history_visible": history_stats["visible"],
                "memory_usage_mb": history_stats["memory_usage_mb"]
            })
        else:
            stats.update({
                "message_count": 0,
                "history_visible": False,
                "memory_usage_mb": 0
            })
        
        return stats
    
    async def cleanup(self):
        """Cleanup the chat interface."""
        await self.deactivate()
        
        if self.chat_input:
            await self.chat_input.cleanup()
        
        if self.chat_history:
            await self.chat_history.cleanup()
        
        self._callbacks.clear()
        self._initialized = False
        
        logger.info("Chat interface cleaned up")


# Utility function for easy instantiation
def create_chat_interface(application_controller: ApplicationController,
                         config: Optional[ChatInterfaceConfig] = None,
                         status_manager=None,
                         display_renderer=None) -> ChatInterface:
    """Create and return a new ChatInterface instance."""
    return ChatInterface(application_controller, config, status_manager, display_renderer)
