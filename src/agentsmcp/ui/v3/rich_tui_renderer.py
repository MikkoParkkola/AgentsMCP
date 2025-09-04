"""Rich TUI renderer - PHASE 2: MINIMAL features to isolate input issues."""

import sys
from typing import Optional
from rich.console import Console
from .ui_renderer_base import UIRenderer


class RichTUIRenderer(UIRenderer):
    """PHASE 2: Minimal Rich-based TUI renderer - no complex features."""
    
    def __init__(self, capabilities):
        super().__init__(capabilities)
        self.console = None
        self._cleanup_called = False  # Guard against multiple cleanup calls
        # PHASE 2: Remove all complex state tracking that might interfere with input
        # No Live display, no layouts, no raw terminal mode, no cursor tracking
        
    def initialize(self) -> bool:
        """PHASE 2: Initialize MINIMAL Rich TUI - just console output, no Live display."""
        try:
            # Check if Rich should work in this environment
            if not self.capabilities.supports_rich:
                return False
            
            # PHASE 2: Initialize ONLY basic Rich console - no Live display, no layouts
            self.console = Console(
                force_terminal=self.capabilities.is_tty,
                color_system="auto" if self.capabilities.supports_colors else None
            )
            
            # PHASE 2: Show simple Rich welcome (no complex layouts)
            self.console.print("🤖 [bold blue]Rich Console TUI (Simple Mode)[/bold blue]")
            self.console.print("[dim]PHASE 2: Testing minimal Rich features - no Live display or raw terminal mode[/dim]")
            
            return True
            
        except Exception as e:
            print(f"Failed to initialize minimal Rich TUI: {e}")
            return False
    
    def cleanup(self) -> None:
        """PHASE 2: Minimal cleanup - no complex terminal restoration."""
        if self._cleanup_called:
            return  # Prevent multiple cleanup calls
        self._cleanup_called = True
        
        try:
            # Rich renderer cleanup - NO goodbye message here
            # Let the TUI launcher handle the single goodbye message
            pass
        except Exception:
            pass  # Ignore cleanup errors
    
    def render_frame(self) -> None:
        """PHASE 2: No frame rendering - using simple console prints."""
        # PHASE 2: No Live display, no frame rendering - keep it simple
        pass
    
    def handle_input(self) -> Optional[str]:
        """PHASE 2: Use standard input() like PlainCLI - no raw terminal mode."""
        try:
            if self.state.is_processing:
                return None
            
            # PHASE 2: Use standard input() function like PlainCLI but with Rich formatting
            self.console.print("💬 [yellow]>[/yellow] ", end="")
            user_input = input().strip()
            return user_input if user_input else None
                    
        except KeyboardInterrupt:
            return "/quit"
        except EOFError:
            # EOF reached - no more input available (e.g., piped input finished)
            # Return /quit to gracefully exit instead of infinite error loop
            return "/quit"
        except Exception as e:
            if self.console:
                self.console.print(f"[red]Input error: {e}[/red]")
            # Return /quit on persistent errors to avoid infinite loops
            return "/quit"
    
    def show_message(self, message: str, level: str = "info") -> None:
        """PHASE 2: Show a simple Rich formatted message."""
        try:
            colors = {
                "info": "blue",
                "success": "green", 
                "warning": "yellow",
                "error": "red"
            }
            color = colors.get(level, "blue")
            
            # PHASE 2: Direct console print instead of complex state management
            if self.console:
                self.console.print(f"[{color}]{message}[/{color}]")
            
        except Exception as e:
            print(f"Message display error: {e}")
    
    def show_error(self, error: str) -> None:
        """Show an error in Rich TUI."""
        self.show_message(error, "error")
    
    # PHASE 2: Remove all complex layout and message rendering
    # We're using simple console.print() instead