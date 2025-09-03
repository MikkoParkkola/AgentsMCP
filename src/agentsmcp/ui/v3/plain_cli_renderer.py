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
        try:
            print()
            print("Goodbye! 👋")
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
        """Handle user input in plain CLI mode."""
        with self._input_lock:
            try:
                if self.state.is_processing:
                    return None
                
                # Clear current input display
                if self._last_prompt_shown:
                    print()  # Move to next line
                    self._last_prompt_shown = False
                
                # Get input from user
                try:
                    user_input = input("💬 > ").strip()
                except (EOFError, KeyboardInterrupt):
                    return "/quit"
                
                if user_input:
                    self.state.current_input = ""  # Clear input buffer
                    return user_input
                
                return None
                
            except Exception as e:
                print(f"Input error: {e}")
                return None
    
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
            # Clear screen and position cursor
            if self.capabilities.is_tty:
                print("\033[2J\033[H", end="")  # Clear screen, go to top
            
            self._draw_header()
            self._draw_input_area()
            return True
        except Exception as e:
            print(f"Failed to initialize simple TUI: {e}")
            return False
    
    def cleanup(self) -> None:
        """Clean up simple TUI renderer."""
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
            
            # Save cursor position
            print("\033[s", end="")
            
            # Update input area
            self._draw_input_area()
            
            # Show status if available
            if self.state.status_message:
                self._draw_status()
            
            # Restore cursor position
            print("\033[u", end="", flush=True)
            
        except Exception as e:
            print(f"Render error: {e}")
    
    def handle_input(self) -> Optional[str]:
        """Handle user input in simple TUI mode."""
        try:
            import select
            
            if not self.capabilities.is_tty:
                # Fallback to plain input
                return input("💬 > ").strip() or None
            
            # Check if input is available (non-blocking)
            if select.select([sys.stdin], [], [], 0.1)[0]:
                char = sys.stdin.read(1)
                
                if ord(char) == 13:  # Enter key
                    result = self._input_buffer.strip()
                    self._input_buffer = ""
                    self._cursor_pos = 0
                    self.state.current_input = ""
                    return result if result else None
                    
                elif ord(char) == 127:  # Backspace
                    if self._cursor_pos > 0:
                        self._input_buffer = (
                            self._input_buffer[:self._cursor_pos-1] + 
                            self._input_buffer[self._cursor_pos:]
                        )
                        self._cursor_pos -= 1
                        self.state.current_input = self._input_buffer
                        
                elif ord(char) >= 32:  # Printable character
                    self._input_buffer = (
                        self._input_buffer[:self._cursor_pos] + 
                        char + 
                        self._input_buffer[self._cursor_pos:]
                    )
                    self._cursor_pos += 1
                    self.state.current_input = self._input_buffer
            
            return None
            
        except Exception as e:
            print(f"Input error: {e}")
            # Fallback to blocking input
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
            input_line = self._screen_height - 1
            
            # Clear input line
            if self.capabilities.is_tty:
                print(f"\033[{input_line};1H\033[K", end="")
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