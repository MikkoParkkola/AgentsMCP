"""Main TUI launcher - integrates all v3 architecture components."""

import asyncio
import sys
import signal
import logging
from typing import Optional
from .terminal_capabilities import detect_terminal_capabilities
from .ui_renderer_base import ProgressiveRenderer
from .plain_cli_renderer import PlainCLIRenderer, SimpleTUIRenderer
from .console_renderer import ConsoleRenderer
from .chat_engine import ChatEngine, ChatMessage


class TUILauncher:
    """Main launcher for the progressive enhancement TUI system."""
    
    def __init__(self):
        self.capabilities = None
        self.progressive_renderer = None
        self.current_renderer = None
        self.chat_engine = None
        self.running = False
        self._cleanup_called = False  # Guard against multiple cleanup calls
        self._goodbye_shown = False   # Ensure single goodbye message
        
    def initialize(self) -> bool:
        """Initialize V3 TUI launcher - PHASE 2 COMPLETE: Rich renderer with proper EOF handling."""
        # Initialize TUI with minimal output
        try:
            # COMPLETE LOG SUPPRESSION: Absolutely no DEBUG/INFO messages during TUI
            # This must be done BEFORE any LLM components are imported/initialized
            conversation_loggers = [
                'agentsmcp.conversation.llm_client',
                'agentsmcp.conversation.conversation', 
                'agentsmcp.conversation.orchestrated_conversation',
                'agentsmcp.conversation.dispatcher',
                'agentsmcp.conversation.structured_processor',
                'agentsmcp.conversation',  # Parent logger as fallback
                'agentsmcp.llm',  # LLM module
                'agentsmcp',  # Root agentsmcp logger
                ''  # Root logger itself
            ]
            
            # Set to level 100 (higher than CRITICAL=50) to suppress EVERYTHING
            for logger_name in conversation_loggers:
                logger = logging.getLogger(logger_name)
                logger.setLevel(100)  # Suppress ALL messages including CRITICAL
                logger.propagate = False  # Prevent propagation
                # Remove all handlers to ensure no output
                for handler in logger.handlers[:]:
                    logger.removeHandler(handler)
                # Add null handler to absorb any messages
                logger.addHandler(logging.NullHandler())
                
            # Set root logger to maximum suppression
            root_logger = logging.getLogger()
            root_logger.setLevel(100)  # Higher than CRITICAL
            
            # Remove ALL handlers from root logger
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
                
            # Add ONLY a null handler to completely absorb all logging
            root_logger.addHandler(logging.NullHandler())
            
            # Disable logging module entirely for maximum suppression
            logging.disable(logging.CRITICAL)  # Disable all logging below level 50
            
            # Detect terminal capabilities
            from .terminal_capabilities import detect_terminal_capabilities
            self.capabilities = detect_terminal_capabilities()
            
            # Initialize renderer system
            self.progressive_renderer = ProgressiveRenderer(self.capabilities)
            
            # Register renderers
            self.progressive_renderer.register_renderer("plain", PlainCLIRenderer, priority=10)
            
            if self.capabilities.is_tty and self.capabilities.supports_rich:
                # Console renderer available
                self.progressive_renderer.register_renderer("console", ConsoleRenderer, priority=20)
            else:
                # Console renderer not available
                pass
            
            # Select renderer
            self.current_renderer = self.progressive_renderer.select_best_renderer()
            if not self.current_renderer:
                print("❌ TUI initialization failed")
                return False
            
            renderer_name = self.current_renderer.__class__.__name__
            # Renderer selected
            
            # Initialize chat engine
            self.chat_engine = ChatEngine()
            
            # Set up callbacks
            self.chat_engine.set_callbacks(
                status_callback=self._on_status_change,
                message_callback=self._on_new_message,
                error_callback=self._on_error
            )
            
            return True
            
        except Exception as e:
            print(f"❌ V3: Critical initialization failure: {e}")
            import traceback
            print("🔍 V3: Full initialization error traceback:")
            traceback.print_exc()
            return False
    def _on_status_change(self, status: str) -> None:
        """Handle status change from chat engine, including streaming updates and detailed progress."""
        if status.startswith("streaming_update:"):
            # Handle streaming response update
            content = status[17:]  # Remove "streaming_update:" prefix
            self._handle_streaming_update(content)
        elif self.current_renderer and hasattr(self.current_renderer, 'show_status'):
            # Rich renderer - use Rich status display with enhanced formatting
            self.current_renderer.show_status(status)
        elif status and status != "Ready":  # Avoid spamming "Ready" status
            # Enhanced status display for plain renderer
            self._display_enhanced_status(status)
    
    def _display_enhanced_status(self, status: str) -> None:
        """Display enhanced status for plain renderer with better formatting."""
        try:
            # Parse different types of status messages for better display
            if "[" in status and "]" in status:
                # Status with timing information
                parts = status.rsplit(" [", 1)
                if len(parts) == 2:
                    message = parts[0]
                    timing = "[" + parts[1]
                    print(f"⏳ {message} {timing}")
                else:
                    print(f"⏳ {status}")
            elif status.startswith("🔍") or status.startswith("🛠️") or status.startswith("📊"):
                # Already has emoji, just display
                print(f"  {status}")
            else:
                # Basic status, add icon
                print(f"⏳ {status}")
                
        except Exception:
            # Fallback to simple display
            print(f"⏳ {status}")
    
    def _on_new_message(self, message: ChatMessage) -> None:
        """Handle new message from chat engine."""
        from datetime import datetime
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        
        if self.current_renderer and hasattr(self.current_renderer, 'display_chat_message'):
            # Rich renderer - use Rich message display with timestamp
            self.current_renderer.display_chat_message(message.role.value, message.content, timestamp)
        else:
            # Plain renderer fallback with timestamp
            role_symbols = {"user": "👤", "assistant": "🤖", "system": "ℹ️"}
            symbol = role_symbols.get(message.role.value, "❓")
            print(f"{timestamp} {symbol} {message.role.value.title()}: {message.content}")
    
    def _handle_streaming_update(self, content: str) -> None:
        """Handle streaming response update."""
        try:
            if self.current_renderer and hasattr(self.current_renderer, 'handle_streaming_update'):
                self.current_renderer.handle_streaming_update(content)
            else:
                # Fallback for renderers without streaming support
                # Clear current line and show updated content
                print(f"\r🤖 AI: {content[:100]}{'...' if len(content) > 100 else ''}", end="", flush=True)
        except Exception as e:
            print(f"\nStreaming update error: {e}")
    
    def _on_error(self, error: str) -> None:
        """Handle error from chat engine."""
        if self.current_renderer:
            self.current_renderer.show_error(error)
        else:
            print(f"❌ Error: {error}")
    
    async def run_main_loop(self) -> int:
        """Run the V3 TUI main interaction loop with proven EOF handling."""
        try:
            if not self.initialize():
                return 1
            
            # Ready to start
            
            # Set up signal handlers
            self._setup_signal_handlers()
            
            # Show welcome message using the renderer
            if self.current_renderer and hasattr(self.current_renderer, 'show_welcome'):
                self.current_renderer.show_welcome()
            
            if self.current_renderer and hasattr(self.current_renderer, 'show_ready'):
                self.current_renderer.show_ready()
            
            self.running = True
            
            # Main interaction loop with proven EOF handling
            while self.running:
                try:
                    # Direct input handling with proper error recovery
                    user_input = self.current_renderer.handle_input()
                    
                    if user_input:
                        # Process input through chat engine instead of simple echo
                        should_continue = await self.chat_engine.process_input(user_input)
                        if not should_continue:
                            self._show_goodbye()
                            break
                    
                    # Brief pause to prevent busy waiting
                    await asyncio.sleep(0.01)
                    
                except KeyboardInterrupt:
                    print("\n⚠️ Received Ctrl+C - shutting down gracefully...")
                    self._show_goodbye()
                    break
                except Exception as e:
                    print(f"❌ Error in main loop: {e}")
                    break
            
            return 0
            
        except Exception as e:
            print(f"❌ V3: Fatal error in main loop: {e}")
            import traceback
            print("🔍 V3: Full main loop error traceback:")
            traceback.print_exc()
            return 1
        finally:
            self._cleanup()
    
    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            print(f"\n⚠️ Received signal {signum} - shutting down...")
            self.running = False
            
            # Force cleanup immediately on signal to prevent hanging
            try:
                self._cleanup()
            except Exception as e:
                print(f"⚠️ Signal cleanup error: {e}")
            
            # Exit after cleanup attempt
            import sys
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _show_welcome(self) -> None:
        """Show welcome message with V3 system identification."""
        renderer_name = self.current_renderer.__class__.__name__ if self.current_renderer else "Unknown"
        welcome_msg = f"""
ℹ️ 🤖 Welcome to AI Command Composer!

This TUI automatically adapts to your terminal capabilities:
• Plain text mode for maximum compatibility
• Simple TUI mode for basic terminal features  
• Rich TUI mode for full-featured experience

Type your message and press Enter to chat.
Type /help to see available commands.
Type /quit to exit.

Let's start chatting!
        """.strip()
        
        if self.current_renderer:
            self.current_renderer.show_message(welcome_msg, "info")
        else:
            print("❌ V3 TUI: No renderer available for welcome message!")
    
    def cleanup(self) -> None:
        """Public cleanup method for external use."""
        self._cleanup()
    
    def _show_goodbye(self) -> None:
        """Show a single goodbye message with proper positioning."""
        if self._goodbye_shown:
            return
        self._goodbye_shown = True
        
        # Use the renderer's goodbye method if available
        if self.current_renderer and hasattr(self.current_renderer, 'show_goodbye'):
            self.current_renderer.show_goodbye()
        else:
            # Fallback
            print("👋 Goodbye!")    
    def _cleanup(self) -> None:
        """Clean up resources."""
        if self._cleanup_called:
            return  # Prevent multiple cleanup calls
        self._cleanup_called = True
        
        try:
            # Clean up chat engine first
            if self.chat_engine and hasattr(self.chat_engine, 'cleanup'):
                # Run async cleanup in a safe way
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                    # We're in an async context, schedule the cleanup
                    asyncio.create_task(self.chat_engine.cleanup())
                except RuntimeError:
                    # No running loop, create a new one for cleanup
                    try:
                        asyncio.run(self.chat_engine.cleanup())
                    except Exception:
                        pass  # Ignore cleanup errors if we can't run async
                        
            if self.current_renderer:
                self.current_renderer.cleanup()
            if self.progressive_renderer:
                self.progressive_renderer.cleanup()
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")


async def launch_tui() -> int:
    """Launch the V3 TUI application."""
    try:
        launcher = TUILauncher()
        result = await launcher.run_main_loop()
        return result
    except KeyboardInterrupt:
        return 0  # Clean exit on Ctrl+C
    except Exception as e:
        print(f"❌ TUI failed to start: {e}")
        return 1


def main() -> int:
    """Main entry point for the TUI."""
    try:
        return asyncio.run(launch_tui())
    except KeyboardInterrupt:
        # No goodbye here - let TUI launcher handle it
        return 0
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())