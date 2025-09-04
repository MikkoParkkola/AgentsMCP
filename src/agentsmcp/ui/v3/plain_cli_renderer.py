"""Plain text CLI renderer - works everywhere, minimal features."""

import sys
import threading
from typing import Optional
from .ui_renderer_base import UIRenderer


class PlainCLIRenderer(UIRenderer):
    """Minimal plain text CLI renderer that works in any environment."""
    
    def __init__(self, capabilities):
        super().__init__(capabilities)
        self._input_lock = threading.Lock()
        self._last_prompt_shown = False
        self._streaming_active = False  # Track if currently streaming
        self._current_streaming_content = ""  # Current streaming message
        
    def initialize(self) -> bool:
        """Initialize plain CLI renderer."""
        try:
            print("🤖 AI Command Composer - Plain Text Mode")
            print("=" * 50)
            print("Commands: /quit, /help, /clear")
            print()
            return True
        except Exception as e:
            print(f"Failed to initialize plain CLI: {e}")
            return False
    
    def cleanup(self) -> None:
        """Clean up plain CLI renderer."""
        # Prevent multiple cleanup calls
        if getattr(self, '_cleanup_called', False):
            return
        self._cleanup_called = True
        
        try:
            # Plain CLI cleanup - NO goodbye message here
            # Let the TUI launcher handle the single goodbye message
            pass
        except Exception:
            pass  # Ignore cleanup errors
    
    def render_frame(self) -> None:
        """Render current state in plain text."""
        try:
            # Only show prompt if not already shown
            if not self._last_prompt_shown and not self.state.is_processing:
                if self.state.status_message:
                    print(f"Status: {self.state.status_message}")
                
                print(f"💬 > {self.state.current_input}", end="", flush=True)
                self._last_prompt_shown = True
                
        except Exception as e:
            print(f"Render error: {e}")
    
    def handle_input(self) -> Optional[str]:
        """Handle user input in BARE-BONES plain CLI mode."""
        try:
            # SIMPLIFIED input handling - no complex state management
            # Just use standard input() function - most reliable
            try:
                user_input = input("> ").strip()
                return user_input if user_input else None
            except (EOFError, KeyboardInterrupt):
                return "/quit"
            except Exception as e:
                print(f"Input error: {e}")
                # Return /quit on persistent errors to avoid infinite loops
                return "/quit"
                
        except Exception as e:
            print(f"Critical input error: {e}")
            # Return /quit on critical errors to avoid infinite loops
            return "/quit"
    
    def show_message(self, message: str, level: str = "info") -> None:
        """Show a message in plain text."""
        try:
            prefix = {
                "info": "ℹ️",
                "success": "✅", 
                "warning": "⚠️",
                "error": "❌"
            }.get(level, "ℹ️")
            
            print(f"{prefix} {message}")
            
        except Exception as e:
            print(f"Message display error: {e}")
    
    def show_error(self, error: str) -> None:
        """Show an error message in plain text."""
        self.show_message(error, "error")
    
    def handle_streaming_update(self, content: str) -> None:
        """Handle real-time streaming updates in plain text."""
        try:
            # First streaming update - initialize
            if not self._streaming_active:
                self._streaming_active = True
                self._current_streaming_content = ""
                print("🤖 AI (streaming): ", end="", flush=True)
            
            # Update content
            self._current_streaming_content = content
            
            # For plain CLI, we'll show a truncated version during streaming
            if len(content) > 100:
                display_content = content[:97] + "..."
            else:
                display_content = content
            
            # Use carriage return to overwrite the line
            print(f"\r🤖 AI (streaming): {display_content}", end="", flush=True)
            
        except Exception as e:
            print(f"\nStreaming update error: {e}")
    
    def display_chat_message(self, role: str, content: str, timestamp: str = None) -> None:
        """Display a chat message with plain text formatting."""
        try:
            # If we were streaming and this is the final assistant message
            if self._streaming_active and role == "assistant":
                # Finalize the streaming display
                print()  # New line to finish streaming
                self._streaming_active = False
                self._current_streaming_content = ""
                
                # Don't display the message again - it's already been shown during streaming
                return
            
            # Format messages based on role with emojis and timestamp
            role_symbols = {
                "user": "👤",
                "assistant": "🤖", 
                "system": "ℹ️"
            }
            symbol = role_symbols.get(role, "❓")
            role_name = role.title()
            
            if timestamp:
                print(f"{timestamp} {symbol} {role_name}: {content}")
            else:
                # Fallback without timestamp
                print(f"{symbol} {role_name}: {content}")
        except Exception as e:
            print(f"Chat message display error: {e}")
    
    def show_status(self, status: str) -> None:
        """Show status message in plain text."""
        try:
            if status and status != "Ready":  # Avoid spamming "Ready" status
                print(f"⏳ {status}")
        except Exception as e:
            print(f"Status display error: {e}")
    
    def _show_help(self) -> None:
        """Show help information."""
        help_text = """
Available Commands:
  /help     - Show this help message
  /quit     - Exit the application
  /clear    - Clear the screen
  
  Just type your message and press Enter to chat!
        """
        print(help_text.strip())


class SimpleTUIRenderer(UIRenderer):
    """Simple TUI renderer using basic terminal features."""
    
    def __init__(self, capabilities):
        super().__init__(capabilities)
        self._input_buffer = ""
        self._cursor_pos = 0
        self._screen_height = capabilities.height
        self._screen_width = capabilities.width
        
    def initialize(self) -> bool:
        """Initialize simple TUI renderer."""
        try:
            if self.capabilities.is_tty:
                # Clear screen and position cursor - but be less aggressive
                print("\033[2J\033[H", end="", flush=True)  # Clear screen, go to top
                print()  # Add some breathing room
            
            self._draw_header()
            
            if self.capabilities.is_tty:
                print()  # Add space before input area
            
            return True
        except Exception as e:
            print(f"Failed to initialize simple TUI: {e}")
            return False
    
    def cleanup(self) -> None:
        """Clean up simple TUI renderer."""
        # Prevent multiple cleanup calls
        if getattr(self, '_cleanup_called', False):
            return
        self._cleanup_called = True
        
        try:
            if self.capabilities.is_tty:
                print("\033[2J\033[H", end="")  # Clear screen
            print("Goodbye! 👋")
        except Exception:
            pass
    
    def render_frame(self) -> None:
        """Render current frame in simple TUI mode."""
        try:
            if not self.capabilities.is_tty:
                return  # Fallback to plain mode behavior
            
            # For SimpleTUI, minimize rendering to avoid conflicts
            # Only show status if there's an important message
            if self.state.status_message and self.state.status_message.strip():
                print(f"\n{self.state.status_message}")
                
        except Exception as e:
            print(f"Render error: {e}")
    
    def handle_input(self) -> Optional[str]:
        """Handle user input in simple TUI mode."""
        try:
            if not self.capabilities.is_tty:
                # Fallback to plain input
                user_input = input("💬 > ").strip()
                return user_input if user_input else None
            
            # For TTY mode, use blocking input with proper line handling
            # This is simpler and more reliable than character-by-character input
            try:
                # Position cursor at input line
                input_line = self._screen_height - 1
                print(f"\033[{input_line};1H\033[K💬 > ", end="", flush=True)
                
                # Use blocking input - this works correctly with terminal
                user_input = input("").strip()
                
                if user_input:
                    # Update state for potential rendering
                    self.state.current_input = ""  # Clear since we got the input
                    return user_input
                
                return None
                
            except (EOFError, KeyboardInterrupt):
                return "/quit"
            
        except Exception as e:
            print(f"Input error: {e}")
            # Ultimate fallback to basic input
            try:
                return input("💬 > ").strip() or None
            except (EOFError, KeyboardInterrupt):
                return "/quit"
    
    def show_message(self, message: str, level: str = "info") -> None:
        """Show a message in simple TUI."""
        try:
            prefix = {
                "info": "ℹ️",
                "success": "✅",
                "warning": "⚠️", 
                "error": "❌"
            }.get(level, "ℹ️")
            
            # Move to message area and display
            if self.capabilities.is_tty:
                print(f"\033[{self._screen_height-3};1H\033[K{prefix} {message}")
                print(f"\033[{self._screen_height-1};1H", end="", flush=True)
            else:
                print(f"{prefix} {message}")
                
        except Exception as e:
            print(f"Message display error: {e}")
    
    def show_error(self, error: str) -> None:
        """Show an error in simple TUI."""
        self.show_message(error, "error")
    
    def _draw_header(self) -> None:
        """Draw TUI header."""
        try:
            if self.capabilities.supports_colors:
                print("\033[1;34m", end="")  # Bold blue
            
            header = "🤖 AI Command Composer - Simple TUI"
            separator = "─" * self._screen_width if self.capabilities.supports_unicode else "=" * self._screen_width
            
            print(f"\033[1;1H{header}")
            print(f"\033[2;1H{separator}")
            
            if self.capabilities.supports_colors:
                print("\033[0m", end="")  # Reset colors
                
        except Exception as e:
            print(f"Header draw error: {e}")
    
    def _draw_input_area(self) -> None:
        """Draw input area."""
        try:
            # For SimpleTUI, we keep this minimal to avoid cursor conflicts
            # The actual input prompt is handled in handle_input()
            if not self.capabilities.is_tty:
                # Non-TTY fallback
                if self.state.current_input:
                    print(f"💬 > {self.state.current_input}", end="", flush=True)
                    
        except Exception as e:
            print(f"Input area draw error: {e}")
    
    def _draw_status(self) -> None:
        """Draw status area."""
        try:
            status_line = self._screen_height - 2
            
            if self.capabilities.is_tty:
                print(f"\033[{status_line};1H\033[K", end="")
                if self.state.is_processing:
                    print("⏳ Processing...", end="")
                elif self.state.status_message:
                    print(f"ℹ️ {self.state.status_message}", end="")
                    
        except Exception as e:
            print(f"Status draw error: {e}")