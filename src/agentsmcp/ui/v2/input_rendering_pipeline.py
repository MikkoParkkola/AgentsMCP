"""
Input Rendering Pipeline - Immediate character visibility and proper input display rendering.

Provides real-time input rendering with immediate character feedback to eliminate
slow input visibility issues. Handles cursor management, input validation, and
multi-line input modes with optimized rendering performance.

ICD Compliance:
- Inputs: current_input, cursor_position, input_mode, prompt_text
- Outputs: rendered_input, display_changed, input_complete
- Performance: Input rendering within 5ms for immediate feedback
- Key Functions: Real-time character rendering, cursor management, input validation
"""

import asyncio
import time
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List, Callable, Tuple, Union
from datetime import datetime
import threading

try:
    from rich.text import Text
    from rich.style import Style
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def sanitize_control_characters(text: str) -> str:
    """
    Sanitize control characters that could be used for injection attacks.
    
    THREAT: Control character injection
    MITIGATION: Remove or replace dangerous control characters
    """
    if not text:
        return text
    
    # Remove dangerous control characters but keep safe ones
    # Keep: tab (0x09), newline (0x0A), carriage return (0x0D)
    # Remove: other control chars (0x00-0x08, 0x0B, 0x0C, 0x0E-0x1F, 0x7F)
    sanitized = ""
    for char in text:
        code = ord(char)
        if code < 32:
            # Allow tab, newline, carriage return
            if code in (9, 10, 13):
                sanitized += char
            else:
                # Replace other control characters with safe placeholder
                sanitized += '?'
        elif code == 127:  # DEL character
            sanitized += '?'
        else:
            sanitized += char
    
    return sanitized


def sanitize_ansi_escape_sequences(text: str) -> str:
    """
    Remove ANSI escape sequences that could interfere with display.
    
    THREAT: ANSI escape sequence injection
    MITIGATION: Strip dangerous escape sequences
    """
    # Pattern to match ANSI escape sequences
    ansi_pattern = re.compile(r'\x1b\[[0-9;]*[mGKHJABCDEF]')
    return ansi_pattern.sub('', text)


class InputMode(Enum):
    """Input modes for different interaction types."""
    SINGLE_LINE = "single_line"      # Single line input
    MULTI_LINE = "multi_line"        # Multi-line input
    PASSWORD = "password"            # Password input (masked)
    COMMAND = "command"              # Command input with completion
    SEARCH = "search"                # Search input with highlighting


class CursorStyle(Enum):
    """Cursor display styles."""
    BLOCK = "block"                  # Block cursor █
    UNDERLINE = "underline"          # Underline cursor _
    BAR = "bar"                      # Bar cursor |
    NONE = "none"                    # No cursor


@dataclass
class InputState:
    """Current state of input."""
    text: str
    cursor_position: int
    selection_start: Optional[int] = None
    selection_end: Optional[int] = None
    is_modified: bool = False
    is_valid: bool = True
    validation_message: Optional[str] = None


@dataclass
class RenderResult:
    """Result of input rendering operation."""
    rendered_input: Union[str, 'Text']
    display_changed: bool
    input_complete: bool
    cursor_visible: bool = True
    performance_ms: float = 0.0
    render_method: str = "unknown"
    lines_count: int = 1
    cursor_line: int = 0
    cursor_column: int = 0


class InputValidator:
    """Input validation interface."""
    
    def __init__(self, 
                 min_length: int = 0,
                 max_length: Optional[int] = None,
                 pattern: Optional[str] = None,
                 custom_validator: Optional[Callable[[str], Tuple[bool, Optional[str]]]] = None):
        """
        Initialize input validator.
        
        Args:
            min_length: Minimum input length
            max_length: Maximum input length  
            pattern: Regex pattern to match
            custom_validator: Custom validation function
        """
        self.min_length = min_length
        self.max_length = max_length
        self.pattern = pattern
        self.custom_validator = custom_validator
    
    def validate(self, text: str) -> Tuple[bool, Optional[str]]:
        """Validate input text."""
        # Length validation
        if len(text) < self.min_length:
            return False, f"Minimum length is {self.min_length}"
        
        if self.max_length and len(text) > self.max_length:
            return False, f"Maximum length is {self.max_length}"
        
        # Pattern validation
        if self.pattern:
            import re
            if not re.match(self.pattern, text):
                return False, "Invalid format"
        
        # Custom validation
        if self.custom_validator:
            return self.custom_validator(text)
        
        return True, None


class InputRenderingPipeline:
    """
    High-performance input rendering pipeline for immediate character feedback.
    
    Provides real-time input rendering with cursor management and multi-mode support
    to eliminate input lag and ensure immediate character visibility.
    """
    
    def __init__(self):
        """Initialize the input rendering pipeline."""
        self._lock = threading.RLock()
        
        # Rendering state
        self._current_state = InputState(text="", cursor_position=0)
        self._previous_state = InputState(text="", cursor_position=0)
        self._last_render_result: Optional[RenderResult] = None
        
        # Configuration
        self._input_mode = InputMode.SINGLE_LINE
        self._cursor_style = CursorStyle.BAR
        self._prompt_text = ""
        self._placeholder_text = ""
        self._max_width = 80
        self._max_height = 10
        
        # Performance optimization
        self._render_cache: Dict[str, RenderResult] = {}
        self._cache_max_size = 50
        self._performance_target_ms = 5.0  # ICD requirement
        
        # Style configuration
        if RICH_AVAILABLE:
            self._styles = {
                'input': Style(color="white"),
                'cursor': Style(color="white", bgcolor="white"),
                'placeholder': Style(color="bright_black"),  # Use bright_black instead of gray
                'prompt': Style(color="green", bold=True),
                'error': Style(color="red"),
                'selection': Style(bgcolor="blue")
            }
        else:
            # Fallback styles for non-Rich environments
            self._styles = {
                'input': "white",
                'cursor': "white",
                'placeholder': "bright_black",
                'prompt': "green",
                'error': "red",
                'selection': "blue"
            }
        
        # Cursor blinking
        self._cursor_visible = True
        self._cursor_blink_interval = 0.5  # seconds
        self._last_cursor_toggle = time.time()
        
        # Input validation
        self._validator: Optional[InputValidator] = None
        
        # Performance tracking
        self._operation_times: Dict[str, float] = {}
        self._render_count = 0
        self._cache_hits = 0
    
    async def render_input(self,
                          current_input: str,
                          cursor_position: int,
                          input_mode: InputMode = InputMode.SINGLE_LINE,
                          prompt_text: str = "") -> RenderResult:
        """
        Render input with immediate character feedback.
        
        Args:
            current_input: Current input text
            cursor_position: Position of cursor in text
            input_mode: Input mode (single/multi-line, etc.)
            prompt_text: Prompt text to display
            
        Returns:
            RenderResult with rendered input and metadata
        """
        start_time = time.time()
        
        try:
            # Update state with input sanitization
            with self._lock:
                # THREAT: Control character injection via input
                # MITIGATION: Sanitize input before processing
                sanitized_input = sanitize_ansi_escape_sequences(
                    sanitize_control_characters(current_input)
                )
                sanitized_prompt = sanitize_ansi_escape_sequences(
                    sanitize_control_characters(prompt_text)
                )
                
                self._current_state = InputState(
                    text=sanitized_input,
                    cursor_position=max(0, min(cursor_position, len(sanitized_input)))
                )
                self._input_mode = input_mode
                self._prompt_text = sanitized_prompt
                
                # Check if display actually changed
                display_changed = self._has_display_changed()
                
                # Check cache if no changes
                if not display_changed:
                    cache_key = self._generate_cache_key()
                    if cache_key in self._render_cache:
                        cached_result = self._render_cache[cache_key]
                        cached_result.performance_ms = (time.time() - start_time) * 1000
                        self._cache_hits += 1
                        return cached_result
                
                # Validate input
                await self._validate_input()
                
                # Render based on mode
                if input_mode == InputMode.SINGLE_LINE:
                    result = await self._render_single_line()
                elif input_mode == InputMode.MULTI_LINE:
                    result = await self._render_multi_line()
                elif input_mode == InputMode.PASSWORD:
                    result = await self._render_password()
                elif input_mode == InputMode.COMMAND:
                    result = await self._render_command()
                elif input_mode == InputMode.SEARCH:
                    result = await self._render_search()
                else:
                    result = await self._render_single_line()  # Fallback
                
                # Update metadata
                result.display_changed = display_changed
                result.input_complete = self._is_input_complete()
                result.performance_ms = (time.time() - start_time) * 1000
                
                # Performance check (ICD requirement: 5ms)
                if result.performance_ms > self._performance_target_ms:
                    # Log performance warning but don't fail
                    pass
                
                # Cache result
                if len(self._render_cache) < self._cache_max_size:
                    cache_key = self._generate_cache_key()
                    self._render_cache[cache_key] = result
                
                # Update tracking
                self._last_render_result = result
                self._previous_state = InputState(
                    text=self._current_state.text,
                    cursor_position=self._current_state.cursor_position
                )
                self._render_count += 1
                self._operation_times['render_input'] = result.performance_ms
                
                return result
                
        except Exception as e:
            # Fallback rendering on error
            operation_time = (time.time() - start_time) * 1000
            
            return RenderResult(
                rendered_input=f"{prompt_text}{current_input}",
                display_changed=True,
                input_complete=False,
                performance_ms=operation_time,
                render_method="error_fallback"
            )
    
    async def _render_single_line(self) -> RenderResult:
        """Render single-line input."""
        
        text = self._current_state.text
        cursor_pos = self._current_state.cursor_position
        
        # Calculate display boundaries
        if len(text) <= self._max_width - len(self._prompt_text) - 2:
            # Text fits entirely
            display_text = text
            display_cursor_pos = cursor_pos
            scroll_offset = 0
        else:
            # Need scrolling
            available_width = self._max_width - len(self._prompt_text) - 2
            if cursor_pos < available_width // 2:
                # Cursor near start
                scroll_offset = 0
            elif cursor_pos > len(text) - available_width // 2:
                # Cursor near end
                scroll_offset = len(text) - available_width
            else:
                # Cursor in middle
                scroll_offset = cursor_pos - available_width // 2
            
            display_text = text[scroll_offset:scroll_offset + available_width]
            display_cursor_pos = cursor_pos - scroll_offset
        
        # Build rendered output
        if RICH_AVAILABLE:
            rendered = self._render_with_rich_single_line(
                display_text, display_cursor_pos, scroll_offset > 0
            )
        else:
            rendered = self._render_plain_single_line(
                display_text, display_cursor_pos, scroll_offset > 0
            )
        
        return RenderResult(
            rendered_input=rendered,
            display_changed=True,
            input_complete=False,
            cursor_visible=self._should_show_cursor(),
            render_method="single_line",
            lines_count=1,
            cursor_line=0,
            cursor_column=display_cursor_pos
        )
    
    async def _render_multi_line(self) -> RenderResult:
        """Render multi-line input."""
        
        text = self._current_state.text
        cursor_pos = self._current_state.cursor_position
        
        # Split into lines
        lines = text.split('\n')
        
        # Find cursor position in lines
        cursor_line = 0
        cursor_column = cursor_pos
        char_count = 0
        
        for i, line in enumerate(lines):
            if char_count + len(line) >= cursor_pos:
                cursor_line = i
                cursor_column = cursor_pos - char_count
                break
            char_count += len(line) + 1  # +1 for newline
        
        # Handle height constraints
        if len(lines) > self._max_height:
            # Calculate visible range
            if cursor_line < self._max_height // 2:
                start_line = 0
            elif cursor_line > len(lines) - self._max_height // 2:
                start_line = len(lines) - self._max_height
            else:
                start_line = cursor_line - self._max_height // 2
            
            visible_lines = lines[start_line:start_line + self._max_height]
            visual_cursor_line = cursor_line - start_line
        else:
            visible_lines = lines
            visual_cursor_line = cursor_line
        
        # Build rendered output
        if RICH_AVAILABLE:
            rendered = self._render_with_rich_multi_line(
                visible_lines, visual_cursor_line, cursor_column
            )
        else:
            rendered = self._render_plain_multi_line(
                visible_lines, visual_cursor_line, cursor_column
            )
        
        return RenderResult(
            rendered_input=rendered,
            display_changed=True,
            input_complete=False,
            cursor_visible=self._should_show_cursor(),
            render_method="multi_line",
            lines_count=len(visible_lines),
            cursor_line=visual_cursor_line,
            cursor_column=cursor_column
        )
    
    async def _render_password(self) -> RenderResult:
        """Render password input with masking."""
        
        text = self._current_state.text
        cursor_pos = self._current_state.cursor_position
        
        # Mask the text
        masked_text = '•' * len(text)
        
        # Calculate display boundaries (similar to single line)
        if len(masked_text) <= self._max_width - len(self._prompt_text) - 2:
            display_text = masked_text
            display_cursor_pos = cursor_pos
        else:
            # Scrolling for long passwords
            available_width = self._max_width - len(self._prompt_text) - 2
            if cursor_pos > available_width:
                scroll_offset = cursor_pos - available_width + 1
                display_text = masked_text[scroll_offset:scroll_offset + available_width]
                display_cursor_pos = available_width - 1
            else:
                display_text = masked_text[:available_width]
                display_cursor_pos = cursor_pos
        
        # Build rendered output
        if RICH_AVAILABLE:
            rendered = self._render_with_rich_password(display_text, display_cursor_pos)
        else:
            rendered = self._render_plain_password(display_text, display_cursor_pos)
        
        return RenderResult(
            rendered_input=rendered,
            display_changed=True,
            input_complete=False,
            cursor_visible=self._should_show_cursor(),
            render_method="password",
            lines_count=1,
            cursor_line=0,
            cursor_column=display_cursor_pos
        )
    
    async def _render_command(self) -> RenderResult:
        """Render command input with potential completion hints."""
        # For now, render like single line with command styling
        result = await self._render_single_line()
        result.render_method = "command"
        return result
    
    async def _render_search(self) -> RenderResult:
        """Render search input with highlighting."""
        # For now, render like single line with search styling
        result = await self._render_single_line()
        result.render_method = "search"
        return result
    
    def _render_with_rich_single_line(self, 
                                     display_text: str, 
                                     cursor_pos: int,
                                     is_scrolled: bool) -> 'Text':
        """Render single line with Rich formatting."""
        
        rendered = Text()
        
        # Add prompt
        if self._prompt_text:
            rendered.append(self._prompt_text, style=self._styles['prompt'])
        
        # Add scroll indicator
        if is_scrolled:
            rendered.append("◀ ", style=self._styles['input'])
        
        # Add text with cursor
        if display_text:
            # Text before cursor
            if cursor_pos > 0:
                rendered.append(display_text[:cursor_pos], style=self._styles['input'])
            
            # Cursor
            if self._should_show_cursor() and cursor_pos < len(display_text):
                cursor_char = display_text[cursor_pos]
                rendered.append(cursor_char, style=self._styles['cursor'])
                
                # Text after cursor
                if cursor_pos + 1 < len(display_text):
                    rendered.append(display_text[cursor_pos + 1:], style=self._styles['input'])
            else:
                # Cursor at end or invisible
                if cursor_pos < len(display_text):
                    rendered.append(display_text[cursor_pos:], style=self._styles['input'])
                
                # Show cursor at end
                if self._should_show_cursor() and cursor_pos >= len(display_text):
                    rendered.append(" ", style=self._styles['cursor'])
        else:
            # Empty input - show cursor or placeholder
            if self._placeholder_text and not self._current_state.text:
                rendered.append(self._placeholder_text, style=self._styles['placeholder'])
            elif self._should_show_cursor():
                rendered.append(" ", style=self._styles['cursor'])
        
        return rendered
    
    def _render_with_rich_multi_line(self,
                                    lines: List[str],
                                    cursor_line: int,
                                    cursor_column: int) -> 'Text':
        """Render multi-line with Rich formatting."""
        
        rendered = Text()
        
        for i, line in enumerate(lines):
            # Add line number or prompt for first line
            if i == 0 and self._prompt_text:
                rendered.append(f"{self._prompt_text} ", style=self._styles['prompt'])
            else:
                rendered.append("  ", style=self._styles['input'])  # Indent continuation lines
            
            # Add line content with cursor if this is cursor line
            if i == cursor_line and self._should_show_cursor():
                # Text before cursor
                if cursor_column > 0:
                    rendered.append(line[:cursor_column], style=self._styles['input'])
                
                # Cursor
                if cursor_column < len(line):
                    cursor_char = line[cursor_column]
                    rendered.append(cursor_char, style=self._styles['cursor'])
                    
                    # Text after cursor
                    if cursor_column + 1 < len(line):
                        rendered.append(line[cursor_column + 1:], style=self._styles['input'])
                else:
                    # Cursor at end of line
                    rendered.append(" ", style=self._styles['cursor'])
            else:
                # Regular line without cursor
                rendered.append(line, style=self._styles['input'])
            
            # Add newline except for last line
            if i < len(lines) - 1:
                rendered.append("\n")
        
        return rendered
    
    def _render_with_rich_password(self, display_text: str, cursor_pos: int) -> 'Text':
        """Render password with Rich formatting."""
        
        rendered = Text()
        
        # Add prompt
        if self._prompt_text:
            rendered.append(self._prompt_text, style=self._styles['prompt'])
        
        # Add masked text with cursor
        if display_text:
            # Text before cursor
            if cursor_pos > 0:
                rendered.append(display_text[:cursor_pos], style=self._styles['input'])
            
            # Cursor
            if self._should_show_cursor() and cursor_pos < len(display_text):
                cursor_char = display_text[cursor_pos]
                rendered.append(cursor_char, style=self._styles['cursor'])
                
                # Text after cursor
                if cursor_pos + 1 < len(display_text):
                    rendered.append(display_text[cursor_pos + 1:], style=self._styles['input'])
            else:
                # Cursor at end or invisible
                if cursor_pos < len(display_text):
                    rendered.append(display_text[cursor_pos:], style=self._styles['input'])
                
                # Show cursor at end
                if self._should_show_cursor() and cursor_pos >= len(display_text):
                    rendered.append("•", style=self._styles['cursor'])
        elif self._should_show_cursor():
            # Empty password - show cursor
            rendered.append("•", style=self._styles['cursor'])
        
        return rendered
    
    def _render_plain_single_line(self, 
                                 display_text: str, 
                                 cursor_pos: int,
                                 is_scrolled: bool) -> str:
        """Render single line without Rich formatting."""
        
        result = ""
        
        # Add prompt
        if self._prompt_text:
            result += self._prompt_text
        
        # Add scroll indicator
        if is_scrolled:
            result += "◀ "
        
        # Add text with cursor
        if display_text:
            # Text before cursor
            result += display_text[:cursor_pos]
            
            # Cursor (using pipe character)
            if self._should_show_cursor():
                if cursor_pos < len(display_text):
                    # Cursor over character - use brackets
                    result += f"[{display_text[cursor_pos]}]"
                    result += display_text[cursor_pos + 1:]
                else:
                    # Cursor at end
                    result += "|"
            else:
                # No cursor
                result += display_text[cursor_pos:]
        elif self._should_show_cursor():
            # Empty input with cursor
            result += "|"
        
        return result
    
    def _render_plain_multi_line(self,
                                lines: List[str],
                                cursor_line: int,
                                cursor_column: int) -> str:
        """Render multi-line without Rich formatting."""
        
        result_lines = []
        
        for i, line in enumerate(lines):
            line_result = ""
            
            # Add line prefix
            if i == 0 and self._prompt_text:
                line_result += f"{self._prompt_text} "
            else:
                line_result += "  "  # Indent continuation lines
            
            # Add line content with cursor if this is cursor line
            if i == cursor_line and self._should_show_cursor():
                # Text before cursor
                line_result += line[:cursor_column]
                
                # Cursor
                if cursor_column < len(line):
                    line_result += f"[{line[cursor_column]}]"
                    line_result += line[cursor_column + 1:]
                else:
                    # Cursor at end of line
                    line_result += "|"
            else:
                # Regular line without cursor
                line_result += line
            
            result_lines.append(line_result)
        
        return "\n".join(result_lines)
    
    def _render_plain_password(self, display_text: str, cursor_pos: int) -> str:
        """Render password without Rich formatting."""
        
        result = ""
        
        # Add prompt
        if self._prompt_text:
            result += self._prompt_text
        
        # Add masked text with cursor
        if display_text:
            # Text before cursor
            result += display_text[:cursor_pos]
            
            # Cursor
            if self._should_show_cursor():
                if cursor_pos < len(display_text):
                    result += f"[{display_text[cursor_pos]}]"
                    result += display_text[cursor_pos + 1:]
                else:
                    result += "|"
            else:
                result += display_text[cursor_pos:]
        elif self._should_show_cursor():
            # Empty password with cursor
            result += "|"
        
        return result
    
    def _has_display_changed(self) -> bool:
        """Check if display has changed since last render."""
        # If this is the first render (no previous render result), always consider it changed
        if self._last_render_result is None:
            return True
            
        return (
            self._current_state.text != self._previous_state.text or
            self._current_state.cursor_position != self._previous_state.cursor_position or
            self._cursor_blink_changed()
        )
    
    def _cursor_blink_changed(self) -> bool:
        """Check if cursor blink state has changed."""
        current_time = time.time()
        if current_time - self._last_cursor_toggle >= self._cursor_blink_interval:
            self._cursor_visible = not self._cursor_visible
            self._last_cursor_toggle = current_time
            return True
        return False
    
    def _should_show_cursor(self) -> bool:
        """Determine if cursor should be visible."""
        return self._cursor_visible and self._cursor_style != CursorStyle.NONE
    
    async def _validate_input(self) -> None:
        """Validate current input."""
        if self._validator:
            is_valid, message = self._validator.validate(self._current_state.text)
            self._current_state.is_valid = is_valid
            self._current_state.validation_message = message
        else:
            self._current_state.is_valid = True
            self._current_state.validation_message = None
    
    def _is_input_complete(self) -> bool:
        """Check if input is complete (for Enter key detection)."""
        # This would typically be set by external input handler
        # For now, return False as we don't handle input completion here
        return False
    
    def _generate_cache_key(self) -> str:
        """Generate cache key for render result."""
        return f"{self._current_state.text}_{self._current_state.cursor_position}_{self._input_mode.value}_{self._cursor_visible}"
    
    def configure(self,
                 max_width: int = 80,
                 max_height: int = 10,
                 cursor_style: CursorStyle = CursorStyle.BAR,
                 placeholder_text: str = "",
                 validator: Optional[InputValidator] = None) -> None:
        """Configure the input rendering pipeline."""
        with self._lock:
            self._max_width = max_width
            self._max_height = max_height
            self._cursor_style = cursor_style
            self._placeholder_text = placeholder_text
            self._validator = validator
            
            # Clear cache on configuration change
            self._render_cache.clear()
    
    def set_style(self, 
                 style_name: str, 
                 style: Union[str, 'Style']) -> None:
        """Set a style for input rendering."""
        if RICH_AVAILABLE and isinstance(style, str):
            style = Style.parse(style)
        
        self._styles[style_name] = style
        self._render_cache.clear()  # Clear cache on style change
    
    def clear_cache(self) -> None:
        """Clear render cache."""
        with self._lock:
            self._render_cache.clear()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for rendering operations."""
        cache_hit_rate = self._cache_hits / max(self._render_count, 1) * 100
        
        return {
            'operation_times': dict(self._operation_times),
            'render_count': self._render_count,
            'cache_hits': self._cache_hits,
            'cache_hit_rate_percent': cache_hit_rate,
            'cache_size': len(self._render_cache),
            'performance_target_ms': self._performance_target_ms,
            'cursor_visible': self._cursor_visible
        }
    
    def render_immediate_feedback(self, char: str, current_input: str, cursor_position: int) -> bool:
        """
        Immediate feedback for character input - non-async for input thread compatibility.
        
        Args:
            char: Character that was just typed
            current_input: Full current input string
            cursor_position: Current cursor position
            
        Returns:
            bool: True if rendering succeeded
        """
        try:
            # For immediate feedback, we don't need complex rendering
            # Just ensure the input state is updated correctly
            with self._lock:
                self._current_state = InputState(
                    text=current_input,
                    cursor_position=cursor_position
                )
            return True
        except Exception:
            return False
    
    def render_deletion_feedback(self, current_input: str, cursor_position: int) -> bool:
        """
        Immediate feedback for backspace/deletion - non-async for input thread compatibility.
        
        Args:
            current_input: Full current input string after deletion
            cursor_position: Current cursor position after deletion
            
        Returns:
            bool: True if rendering succeeded
        """
        try:
            # For deletion feedback, we don't need complex rendering
            # Just ensure the input state is updated correctly
            with self._lock:
                self._current_state = InputState(
                    text=current_input,
                    cursor_position=cursor_position
                )
            return True
        except Exception:
            return False


# Convenience functions for easy usage
async def render_input_simple(text: str, 
                             cursor_pos: int,
                             prompt: str = "",
                             width: int = 80) -> str:
    """Simple input rendering function."""
    pipeline = InputRenderingPipeline()
    pipeline.configure(max_width=width)
    
    result = await pipeline.render_input(
        current_input=text,
        cursor_position=cursor_pos,
        prompt_text=prompt
    )
    
    if isinstance(result.rendered_input, str):
        return result.rendered_input
    else:
        return result.rendered_input.plain if hasattr(result.rendered_input, 'plain') else str(result.rendered_input)


async def render_with_immediate_feedback(text: str,
                                       cursor_pos: int,
                                       prompt: str = "") -> Tuple[str, bool]:
    """Render input optimized for immediate character feedback."""
    pipeline = InputRenderingPipeline()
    
    result = await pipeline.render_input(
        current_input=text,
        cursor_position=cursor_pos,
        prompt_text=prompt
    )
    
    rendered_text = (
        result.rendered_input.plain
        if hasattr(result.rendered_input, 'plain')
        else str(result.rendered_input)
    )
    
    return rendered_text, result.performance_ms <= 5.0  # Met performance target