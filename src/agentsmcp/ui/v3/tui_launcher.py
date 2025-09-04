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
        """Initialize V3 TUI launcher - PHASE 2 COMPLETE: Rich renderer with proper EOF handling."""
        print("🔧 V3 TUI: Starting initialization with proven EOF handling fixes...")
        try:
            # Detect actual terminal capabilities properly
            print("🔧 V3: Detecting terminal capabilities...")
            from .terminal_capabilities import detect_terminal_capabilities
            self.capabilities = detect_terminal_capabilities()
            
            print(f"✓ Terminal capabilities: TTY={self.capabilities.is_tty}, Rich={self.capabilities.supports_rich}")
            
            # Initialize progressive renderer system
            print("🔧 V3: Initializing progressive renderer system...")
            self.progressive_renderer = ProgressiveRenderer(self.capabilities)
            
            # Register both Plain and Rich renderers (both have proper EOF handling now)
            print("🔧 V3: Registering Plain CLI renderer as reliable fallback...")
            self.progressive_renderer.register_renderer("plain", PlainCLIRenderer, priority=10)
            
            if self.capabilities.is_tty and self.capabilities.supports_rich:
                print("🔧 V3: Registering Rich TUI renderer with proven EOF handling...")
                self.progressive_renderer.register_renderer("rich", RichTUIRenderer, priority=20)
            else:
                print("🔧 V3: Rich disabled - not a TTY or Rich not supported")
            
            # Select best available renderer
            print("🔧 V3: Selecting best available renderer...")
            self.current_renderer = self.progressive_renderer.select_best_renderer()
            if not self.current_renderer:
                print("❌ V3: No renderer could be initialized!")
                return False
            
            renderer_name = self.current_renderer.__class__.__name__
            print(f"✅ Selected renderer: {renderer_name}")
            
            # Initialize chat engine
            self.chat_engine = ChatEngine()
            
            # Skip callbacks for now - keep simple until core functionality is solid
            print("🔧 V3: Skipping complex callbacks for streamlined version")
            
            return True
            
        except Exception as e:
            print(f"❌ V3: Critical initialization failure: {e}")
            import traceback
            print("🔍 V3: Full initialization error traceback:")
            traceback.print_exc()
            return False
    
    def _on_status_change(self, status: str) -> None:
        """Handle status change from chat engine."""
        if self.current_renderer:
            self.current_renderer.update_state(status_message=status)
            # Only render frame when state actually changes
            self.current_renderer.render_frame()
    
    def _on_new_message(self, message: ChatMessage) -> None:
        """Handle new message from chat engine."""
        if self.current_renderer:
            # Add message to renderer's message list
            messages = self.current_renderer.state.messages[:]
            messages.append(message.to_dict())
            self.current_renderer.update_state(messages=messages)
            # Only render frame when new content is added
            self.current_renderer.render_frame()
    
    def _on_error(self, error: str) -> None:
        """Handle error from chat engine."""
        if self.current_renderer:
            self.current_renderer.show_error(error)
    
    async def run_main_loop(self) -> int:
        """Run the V3 TUI main interaction loop with proven EOF handling."""
        try:
            if not self.initialize():
                return 1
            
            print("🔧 V3: Starting main interaction loop...")
            
            # Set up signal handlers
            self._setup_signal_handlers()
            
            # Simple welcome message
            print("🤖 AgentsMCP TUI (V3 Architecture)")
            print("=" * 50)
            print("Type your message and press Enter.")
            print("Type '/quit' to exit.")
            print()
            
            self.running = True
            
            # Main interaction loop with proven EOF handling
            while self.running:
                try:
                    # Direct input handling with proper error recovery
                    user_input = self.current_renderer.handle_input()
                    
                    if user_input:
                        # Handle quit command directly
                        if user_input.lower().strip() in ['/quit', '/exit', 'quit', 'exit']:
                            print("👋 Goodbye!")
                            break
                        
                        # Simple echo with renderer-appropriate formatting
                        renderer_name = self.current_renderer.__class__.__name__
                        if hasattr(self.current_renderer, 'console'):
                            # Rich renderer - use Rich formatting
                            self.current_renderer.console.print(f"[green]Echo:[/green] {user_input}")
                        else:
                            # Plain renderer - use plain text
                            print(f"Echo: {user_input}")
                        
                        # TODO: Add back chat engine processing once basic input works
                        # should_continue = await self.chat_engine.process_input(user_input)
                        # if not should_continue:
                        #     break
                    
                    # Brief pause to prevent busy waiting
                    await asyncio.sleep(0.01)
                    
                except KeyboardInterrupt:
                    print("\n⚠️ Received Ctrl+C - shutting down gracefully...")
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
    """Launch the V3 TUI application with enhanced error reporting."""
    print("🚀 V3 TUI: Starting launch_tui()...")
    try:
        launcher = TUILauncher()
        print("✅ V3 TUI: TUILauncher created successfully")
        result = await launcher.run_main_loop()
        print(f"✅ V3 TUI: Main loop completed with result: {result}")
        return result
    except Exception as e:
        print(f"❌ V3 TUI: Launch failed: {e}")
        import traceback
        print("🔍 V3 TUI: Full launch error traceback:")
        traceback.print_exc()
        print("   This error needs to be fixed in V3 system")
        return 1


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