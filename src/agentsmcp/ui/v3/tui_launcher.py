"""Main TUI launcher - integrates all v3 architecture components."""

import asyncio
import sys
import signal
from typing import Optional
from .terminal_capabilities import detect_terminal_capabilities
from .ui_renderer_base import ProgressiveRenderer
from .plain_cli_renderer import PlainCLIRenderer, SimpleTUIRenderer
from .rich_tui_renderer import RichTUIRenderer
from .chat_engine import ChatEngine, ChatMessage


class TUILauncher:
    """Main launcher for the progressive enhancement TUI system."""
    
    def __init__(self):
        self.capabilities = None
        self.progressive_renderer = None
        self.current_renderer = None
        self.chat_engine = None
        self.running = False
        
    def initialize(self) -> bool:
        """Initialize the TUI launcher."""
        try:
            # Detect terminal capabilities
            self.capabilities = detect_terminal_capabilities()
            print(f"🔍 Terminal capabilities detected:")
            print(f"  TTY: {self.capabilities.is_tty}")
            print(f"  Size: {self.capabilities.width}x{self.capabilities.height}")
            print(f"  Colors: {self.capabilities.supports_colors}")
            print(f"  Unicode: {self.capabilities.supports_unicode}")
            print(f"  Rich: {self.capabilities.supports_rich}")
            print()
            
            # Initialize progressive renderer system
            self.progressive_renderer = ProgressiveRenderer(self.capabilities)
            
            # Register renderers in priority order (highest priority first)
            self.progressive_renderer.register_renderer("rich", RichTUIRenderer, priority=30)
            self.progressive_renderer.register_renderer("simple", SimpleTUIRenderer, priority=20)
            self.progressive_renderer.register_renderer("plain", PlainCLIRenderer, priority=10)
            
            # Select best available renderer
            self.current_renderer = self.progressive_renderer.select_best_renderer()
            if not self.current_renderer:
                print("❌ Failed to initialize any UI renderer!")
                return False
            
            renderer_name = self.current_renderer.__class__.__name__
            print(f"✅ Selected renderer: {renderer_name}")
            print()
            
            # Initialize chat engine
            self.chat_engine = ChatEngine()
            
            # Set up callbacks for renderer updates
            self.chat_engine.set_callbacks(
                status_callback=self._on_status_change,
                message_callback=self._on_new_message,
                error_callback=self._on_error
            )
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize TUI launcher: {e}")
            return False
    
    def _on_status_change(self, status: str) -> None:
        """Handle status change from chat engine."""
        if self.current_renderer:
            self.current_renderer.update_state(status_message=status)
            self.current_renderer.render_frame()
    
    def _on_new_message(self, message: ChatMessage) -> None:
        """Handle new message from chat engine."""
        if self.current_renderer:
            # Add message to renderer's message list
            messages = self.current_renderer.state.messages[:]
            messages.append(message.to_dict())
            self.current_renderer.update_state(messages=messages)
            self.current_renderer.render_frame()
    
    def _on_error(self, error: str) -> None:
        """Handle error from chat engine."""
        if self.current_renderer:
            self.current_renderer.show_error(error)
    
    async def run_main_loop(self) -> int:
        """Run the main TUI interaction loop."""
        try:
            if not self.initialize():
                return 1
            
            # Set up signal handlers
            self._setup_signal_handlers()
            
            # Show welcome message
            self._show_welcome()
            
            self.running = True
            
            # Main interaction loop
            while self.running:
                try:
                    # Render current frame
                    self.current_renderer.render_frame()
                    
                    # Handle user input (non-blocking or short timeout)
                    user_input = self.current_renderer.handle_input()
                    
                    if user_input:
                        # Process input through chat engine
                        should_continue = await self.chat_engine.process_input(user_input)
                        
                        if not should_continue:
                            print("👋 Goodbye!")
                            break
                    
                    # Small sleep to prevent busy waiting
                    await asyncio.sleep(0.1)
                    
                except KeyboardInterrupt:
                    print("\n⚠️ Received Ctrl+C - shutting down gracefully...")
                    break
                except Exception as e:
                    print(f"❌ Error in main loop: {e}")
                    break
            
            return 0
            
        except Exception as e:
            print(f"❌ Fatal error in main loop: {e}")
            return 1
        finally:
            self._cleanup()
    
    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            print(f"\n⚠️ Received signal {signum} - shutting down...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _show_welcome(self) -> None:
        """Show welcome message."""
        welcome_msg = """
🤖 Welcome to AI Command Composer!

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
    
    def cleanup(self) -> None:
        """Public cleanup method for external use."""
        self._cleanup()
    
    def _cleanup(self) -> None:
        """Clean up resources."""
        try:
            if self.current_renderer:
                self.current_renderer.cleanup()
            if self.progressive_renderer:
                self.progressive_renderer.cleanup()
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")


async def launch_tui() -> int:
    """Launch the TUI application."""
    launcher = TUILauncher()
    return await launcher.run_main_loop()


def main() -> int:
    """Main entry point for the TUI."""
    try:
        return asyncio.run(launch_tui())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return 0
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())