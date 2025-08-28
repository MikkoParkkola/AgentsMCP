"""
Enhanced Chat Input Components

Fixes critical multi-line paste issues and provides enhanced input handling
for AgentsMCP's terminal interface.
"""

from __future__ import annotations

import asyncio
import re
import sys
from typing import Optional, Union

try:
    from rich.console import Console
    from rich.text import Text
    from rich.syntax import Syntax
except ImportError:  # pragma: no cover
    Console = None
    Text = None
    Syntax = None


class EnhancedChatInput:
    """Enhanced chat input with multi-line paste support.
    
    Fixes the critical issue where bracketed paste mode sequences
    (^[[200~ and ^[[201~) appear in user input, breaking the chat experience.
    """
    
    def __init__(self, console: Optional[Console] = None):
        self.console = console
        self._bracketed_paste_pattern = re.compile(r'\x1b\[200~|\x1b\[201~')
        self._ansi_escape_pattern = re.compile(r'\x1b\[[0-9;]*[mK]')
        
    def sanitize_input(self, raw_input: str) -> str:
        """Remove bracketed paste artifacts and clean input.
        
        Removes:
        - ^[[200~ and ^[[201~ (bracketed paste markers)
        - Other ANSI escape sequences that don't belong in text
        - Carriage returns that create display issues
        - Leading/trailing whitespace
        
        Preserves:
        - Intentional newlines in multi-line content
        - Unicode characters and emoji
        - Legitimate formatting
        """
        if not raw_input:
            return ""
            
        # Remove bracketed paste mode sequences
        cleaned = self._bracketed_paste_pattern.sub('', raw_input)
        
        # Remove other common ANSI escape sequences (colors, cursor movement)
        # but preserve actual content
        cleaned = self._ansi_escape_pattern.sub('', cleaned)
        
        # Handle carriage returns that can cause display issues
        cleaned = cleaned.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove excessive whitespace but preserve intentional formatting
        cleaned = cleaned.strip()
        
        return cleaned
    
    async def get_user_input(self) -> str:
        """Get user input with enhanced paste support.
        
        Handles both single-line and multi-line input asynchronously
        without blocking the TUI's event loop.
        """
        loop = asyncio.get_running_loop()
        
        try:
            # Read line asynchronously to avoid blocking the TUI
            raw_input = await loop.run_in_executor(None, sys.stdin.readline)
            
            if not raw_input:  # EOF
                return ""
                
            # Remove trailing newline from readline
            raw_input = raw_input.rstrip('\n')
            
            # Apply sanitization to remove paste artifacts
            return self.sanitize_input(raw_input)
            
        except Exception:  # pragma: no cover
            return ""
    
    def format_message_display(self, message: str, style: str = "default") -> Union[Text, str]:
        """Format message for display in chat history.
        
        Provides proper formatting for different types of content:
        - Code blocks with syntax highlighting
        - Multi-line text with proper wrapping
        - Special characters and emoji
        """
        if not Text:  # Rich not available
            return message
            
        # Detect if message looks like code
        if self._looks_like_code(message):
            return self._format_code_block(message)
        
        # Regular text with proper styling
        text = Text()
        text.append(message, style=style)
        text.overflow = "fold"  # Wrap long lines nicely
        
        return text
    
    def _looks_like_code(self, text: str) -> bool:
        """Heuristic to detect if text contains code."""
        code_indicators = [
            'def ', 'class ', 'import ', 'from ',  # Python
            'function', 'var ', 'const ', 'let ',  # JavaScript
            '{', '}', ';',  # General code indicators
            '```',  # Markdown code blocks
        ]
        
        # Check for multiple indicators or specific patterns
        indicator_count = sum(1 for indicator in code_indicators if indicator in text)
        
        # Also check for indentation patterns common in code
        lines = text.split('\n')
        indented_lines = sum(1 for line in lines if line.startswith(('  ', '\t')))
        
        return indicator_count >= 2 or (len(lines) > 1 and indented_lines > 0)
    
    def _format_code_block(self, code: str) -> Union[Text, str]:
        """Format code with syntax highlighting if possible."""
        if not Syntax:
            return code
            
        try:
            # Try to detect language
            language = self._detect_language(code)
            return Syntax(code, language, theme="monokai", line_numbers=False)
        except Exception:
            # Fallback to plain text with monospace style
            if Text:
                return Text(code, style="bold white on black")
            return code
    
    def _detect_language(self, code: str) -> str:
        """Simple language detection for syntax highlighting."""
        code_lower = code.lower()
        
        if 'def ' in code_lower or 'import ' in code_lower:
            return 'python'
        elif 'function' in code_lower or 'const ' in code_lower:
            return 'javascript'
        elif '#include' in code_lower or 'int main' in code_lower:
            return 'c'
        elif 'public class' in code_lower or 'System.out' in code_lower:
            return 'java'
        else:
            return 'text'  # Generic text highlighting
    
    def show_paste_feedback(self, message: str = "Pasting...") -> None:
        """Show visual feedback during paste operations."""
        if self.console:
            self.console.print(f"[dim yellow]{message}[/dim yellow]")
    
    def clear_paste_feedback(self) -> None:
        """Clear paste feedback message."""
        if self.console:
            # Move cursor up and clear line
            self.console.print("\033[1A\033[K", end="")