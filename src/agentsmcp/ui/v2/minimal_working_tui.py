"""
Minimal Working TUI - Emergency Fix for Immediate Usability

This provides a simple, working TUI that fixes the core typing issue
by using basic terminal input/output without complex frameworks.
"""

import asyncio
import sys
import os
import signal
from typing import Optional
import termios
import tty


class MinimalTUI:
    """Minimal TUI that works immediately - characters appear as typed."""
    
    def __init__(self):
        self.running = False
        self.input_buffer = ""
        self.original_settings = None
        
    def setup_terminal(self):
        """Setup terminal for immediate character input."""
        if sys.stdin.isatty():
            try:
                # Save original terminal settings
                self.original_settings = termios.tcgetattr(sys.stdin.fileno())
                # Set raw mode for immediate character input
                tty.setraw(sys.stdin.fileno())
                # Hide cursor initially
                sys.stdout.write('\033[?25l')
                sys.stdout.flush()
                return True
            except Exception as e:
                print(f"Warning: Could not setup terminal: {e}")
                return False
        return False
    
    def restore_terminal(self):
        """Restore original terminal settings."""
        if self.original_settings and sys.stdin.isatty():
            try:
                # Show cursor
                sys.stdout.write('\033[?25h')
                # Restore terminal settings
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self.original_settings)
                sys.stdout.flush()
            except Exception as e:
                print(f"Warning: Could not restore terminal: {e}")
    
    def clear_screen(self):
        """Clear screen and show prompt."""
        sys.stdout.write('\033[2J\033[H')  # Clear screen and move to top
        sys.stdout.write('🚀 AgentsMCP - Minimal TUI (Fixed Input)\n')
        sys.stdout.write('─' * 50 + '\n')
        sys.stdout.write('Type your message (Ctrl+C to exit, /quit to quit):\n')
        sys.stdout.write('> ')
        sys.stdout.flush()
    
    async def handle_input(self):
        """Handle keyboard input with immediate echo."""
        loop = asyncio.get_event_loop()
        
        while self.running:
            try:
                # Read one character (non-blocking)
                if sys.stdin.isatty():
                    char = await loop.run_in_executor(None, sys.stdin.read, 1)
                else:
                    # Fallback for non-TTY
                    line = await loop.run_in_executor(None, input)
                    await self.process_line(line)
                    continue
                
                # Handle special characters
                if ord(char) == 3:  # Ctrl+C
                    break
                elif ord(char) == 13 or ord(char) == 10:  # Enter
                    sys.stdout.write('\n')
                    await self.process_line(self.input_buffer)
                    self.input_buffer = ""
                    sys.stdout.write('> ')
                elif ord(char) == 127 or ord(char) == 8:  # Backspace
                    if self.input_buffer:
                        self.input_buffer = self.input_buffer[:-1]
                        sys.stdout.write('\b \b')  # Move back, write space, move back
                elif ord(char) >= 32:  # Printable characters
                    self.input_buffer += char
                    sys.stdout.write(char)  # IMMEDIATE ECHO - this fixes the typing issue!
                
                sys.stdout.flush()
                
            except Exception as e:
                print(f"\nError in input handling: {e}")
                break
    
    async def process_line(self, line: str):
        """Process a complete line of input."""
        line = line.strip()
        
        if line.lower() in ['/quit', '/exit', 'quit', 'exit']:
            self.running = False
            sys.stdout.write('\n👋 Goodbye!\n')
            return
        
        if line.lower() == '/help':
            sys.stdout.write('\nCommands:\n')
            sys.stdout.write('  /help  - Show this help\n')
            sys.stdout.write('  /quit  - Exit TUI\n')
            sys.stdout.write('  Ctrl+C - Exit TUI\n')
            sys.stdout.write('\nJust type normally - characters appear immediately!\n')
            return
        
        if line:
            # Echo the message (simulate agent response)
            sys.stdout.write(f'\n🤖 Agent: I received your message: "{line}"\n')
            sys.stdout.write('💭 (This is a minimal TUI - full agent integration coming soon)\n')
    
    async def run(self):
        """Run the minimal TUI."""
        self.running = True
        
        # Setup signal handlers
        def signal_handler(signum, frame):
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Setup terminal
        terminal_setup = self.setup_terminal()
        
        try:
            self.clear_screen()
            await self.handle_input()
        
        except KeyboardInterrupt:
            sys.stdout.write('\n👋 Goodbye!\n')
        
        finally:
            if terminal_setup:
                self.restore_terminal()
    
    def __del__(self):
        """Ensure terminal is restored on cleanup."""
        if hasattr(self, 'original_settings') and self.original_settings:
            self.restore_terminal()


async def launch_minimal_tui():
    """Launch the minimal TUI - emergency fix for typing issues."""
    tui = MinimalTUI()
    await tui.run()
    return 0


if __name__ == "__main__":
    # Direct execution support
    asyncio.run(launch_minimal_tui())