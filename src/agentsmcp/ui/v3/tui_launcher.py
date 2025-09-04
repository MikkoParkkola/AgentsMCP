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
        """Initialize V3 TUI launcher - PHASE 2: Enable Rich renderer with minimal features."""
        print("🔧 PHASE 2 V3 TUI: Starting minimal Rich renderer initialization...")
        try:
            # Detect actual terminal capabilities but enable Rich with TTY detection
            print("🔧 PHASE 2: Detecting terminal capabilities...")
            from .terminal_capabilities import detect_terminal_capabilities, TerminalCapabilities
            detected_capabilities = detect_terminal_capabilities()
            
            # PHASE 2: Force TTY mode for testing (TEMPORARY)
            # This allows us to test Rich renderer even when not in real TTY
            print("🔧 PHASE 2: FORCING TTY mode for testing minimal Rich renderer...")
            self.capabilities = TerminalCapabilities(
                is_tty=True,  # Force TTY mode
                width=detected_capabilities.width,
                height=detected_capabilities.height,
                supports_colors=True,  # Force color support
                supports_unicode=True,
                supports_rich=True,   # Force Rich support
                is_fast_terminal=True,
                max_refresh_rate=5,
                force_plain=False,
                force_simple=False
            )
            
            # PHASE 2: Enable Rich but only in real TTY environments
            # This isolates whether Rich itself or specific Rich features cause issues
            print(f"✓ Terminal capabilities (FORCED): TTY={self.capabilities.is_tty}, Rich={self.capabilities.supports_rich}")
            
            # Initialize progressive renderer system
            print("🔧 PHASE 2: Initializing progressive renderer system...")
            self.progressive_renderer = ProgressiveRenderer(self.capabilities)
            
            # PHASE 2: Register both Plain and MINIMAL Rich renderer
            print("🔧 PHASE 2: Registering Plain CLI renderer as fallback...")
            self.progressive_renderer.register_renderer("plain", PlainCLIRenderer, priority=10)
            
            if self.capabilities.is_tty and self.capabilities.supports_rich:
                print("🔧 PHASE 2: Registering MINIMAL Rich TUI renderer (no complex features)...")
                self.progressive_renderer.register_renderer("rich", RichTUIRenderer, priority=20)
            else:
                print("🔧 PHASE 2: Rich disabled - not a TTY or Rich not supported")
            
            # Select best available renderer
            print("🔧 PHASE 2: Selecting best available renderer...")
            self.current_renderer = self.progressive_renderer.select_best_renderer()
            if not self.current_renderer:
                print("❌ PHASE 2: No renderer could be initialized!")
                return False
            
            renderer_name = self.current_renderer.__class__.__name__
            print(f"✅ Selected renderer: {renderer_name}")
            
            # Initialize chat engine
            self.chat_engine = ChatEngine()
            
            # NO callbacks for bare-bones version - too complex
            print("🔧 BARE-BONES V3: Skipping complex callbacks for minimal version")
            
            return True
            
        except Exception as e:
            print(f"❌ BARE-BONES V3: Critical initialization failure: {e}")
            import traceback
            print("🔍 BARE-BONES V3: Full initialization error traceback:")
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
        """Run the BARE-BONES main TUI interaction loop."""
        try:
            if not self.initialize():
                return 1
            
            print("🔧 BARE-BONES V3: Starting minimal main loop...")
            
            # Set up signal handlers
            self._setup_signal_handlers()
            
            # Simple welcome message (no complex rendering)
            print("🤖 BARE-BONES AgentsMCP TUI")
            print("=" * 50)
            print("Type your message and press Enter.")
            print("Type '/quit' to exit.")
            print()
            
            self.running = True
            
            # SIMPLIFIED Main interaction loop - no complex rendering or async processing
            while self.running:
                try:
                    # Direct input handling - no complex state tracking
                    user_input = self.current_renderer.handle_input()
                    
                    if user_input:
                        # Handle quit command directly
                        if user_input.lower().strip() in ['/quit', '/exit', 'quit', 'exit']:
                            print("👋 Goodbye!")
                            break
                        
                        # PHASE 2: Simple echo with renderer-appropriate formatting
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
                    
                    # NO sleep for busy waiting - keep it simple for now
                    
                except KeyboardInterrupt:
                    print("\n⚠️ Received Ctrl+C - shutting down gracefully...")
                    break
                except Exception as e:
                    print(f"❌ Error in main loop: {e}")
                    break
            
            return 0
            
        except Exception as e:
            print(f"❌ BARE-BONES V3: Fatal error in main loop: {e}")
            import traceback
            print("🔍 BARE-BONES V3: Full main loop error traceback:")
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