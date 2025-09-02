"""
Text Layout Engine - Proper text wrapping and layout to eliminate dotted line issues.

Provides advanced text layout algorithms to prevent the dotted line continuation
characters (...) and layout corruption that occurs with Rich Text overflow patterns.
Implements smart wrapping, container-aware layout, and overflow detection.

ICD Compliance:
- Inputs: text_content, container_width, wrap_mode, overflow_handling
- Outputs: laid_out_text, actual_dimensions, overflow_occurred
- Performance: Text layout within 10ms for 1000 characters
- Key Functions: Smart text wrapping, container-aware layout, overflow detection
"""

import asyncio
import time
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any, Union
from datetime import datetime
import re

try:
    from rich.text import Text
    from rich.console import Console
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class WrapMode(Enum):
    """Text wrapping modes."""
    NONE = "none"           # No wrapping
    WORD = "word"           # Word-boundary wrapping
    CHAR = "char"           # Character-level wrapping  
    SMART = "smart"         # Intelligent wrapping with hyphenation
    ADAPTIVE = "adaptive"   # Adaptive based on content type


class OverflowHandling(Enum):
    """How to handle text overflow."""
    CLIP = "clip"           # Clip text at boundary
    WRAP = "wrap"           # Force wrap to next line
    ELLIPSIS = "ellipsis"   # Add ellipsis (but not dotted ...)
    SCROLL = "scroll"       # Enable scrolling
    ADAPTIVE = "adaptive"   # Choose best option automatically


@dataclass
class TextDimensions:
    """Dimensions of laid out text."""
    width: int
    height: int
    actual_width: int       # Real width used
    actual_height: int      # Real height used
    lines_count: int        # Number of lines
    max_line_width: int     # Width of longest line
    timestamp: datetime


@dataclass
class LayoutResult:
    """Result of text layout operation."""
    laid_out_text: Union[str, 'Text']
    actual_dimensions: TextDimensions
    overflow_occurred: bool
    clipped_content: Optional[str] = None
    performance_ms: float = 0.0
    layout_method: str = "unknown"


class TextLayoutEngine:
    """
    Advanced text layout engine that eliminates dotted line issues.
    
    Provides smart text wrapping and container-aware layout to prevent
    the (...) continuation characters that cause layout corruption in TUIs.
    """
    
    def __init__(self, console: Optional['Console'] = None):
        """Initialize the text layout engine."""
        self._console = console
        
        # Performance tracking
        self._operation_times: Dict[str, float] = {}
        self._layout_cache: Dict[str, LayoutResult] = {}
        self._cache_max_size = 100
        
        # Unicode handling
        self._unicode_enabled = True
        self._emoji_width_cache: Dict[str, int] = {}
        
        # Layout algorithms
        self._word_break_chars = set(' \t\n\r\f\v')
        self._hyphen_chars = set('-–—')
        self._punctuation_chars = set('.,;:!?')
        
        # Performance thresholds (ICD requirement: 10ms for 1000 chars)
        self._max_layout_time_ms = 10.0
        self._chars_per_ms_target = 100.0  # 1000 chars in 10ms
        
    async def layout_text(self,
                         text_content: str,
                         container_width: int,
                         wrap_mode: WrapMode = WrapMode.SMART,
                         overflow_handling: OverflowHandling = OverflowHandling.ADAPTIVE,
                         max_height: Optional[int] = None,
                         preserve_formatting: bool = True) -> LayoutResult:
        """
        Layout text within container constraints.
        
        Args:
            text_content: Text to layout
            container_width: Maximum width available
            wrap_mode: How to wrap text
            overflow_handling: How to handle overflow
            max_height: Maximum height constraint
            preserve_formatting: Whether to preserve Rich formatting
            
        Returns:
            LayoutResult with laid out text and dimensions
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            if container_width <= 0:
                container_width = 80  # Default fallback
            
            if not text_content:
                return LayoutResult(
                    laid_out_text="",
                    actual_dimensions=TextDimensions(
                        width=container_width,
                        height=1,
                        actual_width=0,
                        actual_height=1,
                        lines_count=1,
                        max_line_width=0,
                        timestamp=datetime.now()
                    ),
                    overflow_occurred=False,
                    performance_ms=0.0,
                    layout_method="empty"
                )
            
            # Check cache first
            cache_key = self._generate_cache_key(
                text_content, container_width, wrap_mode, overflow_handling, max_height
            )
            
            if cache_key in self._layout_cache:
                cached_result = self._layout_cache[cache_key]
                cached_result.performance_ms = (time.time() - start_time) * 1000
                return cached_result
            
            # Choose layout strategy based on content and constraints
            strategy = await self._choose_layout_strategy(
                text_content, container_width, wrap_mode, overflow_handling
            )
            
            # Apply layout strategy
            if strategy == "rich":
                result = await self._layout_with_rich(
                    text_content, container_width, wrap_mode, overflow_handling, max_height
                )
            elif strategy == "smart_wrap":
                result = await self._layout_with_smart_wrap(
                    text_content, container_width, wrap_mode, overflow_handling, max_height
                )
            elif strategy == "simple_wrap":
                result = await self._layout_with_simple_wrap(
                    text_content, container_width, wrap_mode, overflow_handling, max_height
                )
            else:
                # Fallback to safe layout
                result = await self._layout_safe_fallback(
                    text_content, container_width, max_height
                )
            
            # Performance check (ICD requirement)
            operation_time = (time.time() - start_time) * 1000
            result.performance_ms = operation_time
            
            if operation_time > self._max_layout_time_ms:
                # Log performance warning but don't fail
                pass
            
            # Cache result if reasonable size
            if len(self._layout_cache) < self._cache_max_size:
                self._layout_cache[cache_key] = result
            
            # Track performance metrics
            self._operation_times['layout_text'] = operation_time
            
            return result
            
        except Exception as e:
            # Fallback to safe layout on any error
            operation_time = (time.time() - start_time) * 1000
            
            return LayoutResult(
                laid_out_text=self._safe_truncate(text_content, container_width),
                actual_dimensions=TextDimensions(
                    width=container_width,
                    height=1,
                    actual_width=min(len(text_content), container_width),
                    actual_height=1,
                    lines_count=1,
                    max_line_width=min(len(text_content), container_width),
                    timestamp=datetime.now()
                ),
                overflow_occurred=len(text_content) > container_width,
                performance_ms=operation_time,
                layout_method="error_fallback"
            )
    
    async def _choose_layout_strategy(self,
                                     text: str,
                                     width: int,
                                     wrap_mode: WrapMode,
                                     overflow_handling: OverflowHandling) -> str:
        """Choose the best layout strategy for the given content."""
        
        # Simple heuristics for strategy selection
        if RICH_AVAILABLE and self._console and len(text) > 100:
            return "rich"
        elif wrap_mode == WrapMode.SMART:
            return "smart_wrap"
        elif wrap_mode in (WrapMode.WORD, WrapMode.CHAR, WrapMode.NONE):
            return "simple_wrap"
        elif wrap_mode == WrapMode.ADAPTIVE:
            return "smart_wrap"  # Use smart wrap for adaptive
        else:
            return "simple_wrap"  # Default to simple wrap instead of safe fallback
    
    async def _layout_with_rich(self,
                               text: str,
                               width: int,
                               wrap_mode: WrapMode,
                               overflow_handling: OverflowHandling,
                               max_height: Optional[int]) -> LayoutResult:
        """Layout text using Rich Text capabilities."""
        
        try:
            if not RICH_AVAILABLE or not self._console:
                return await self._layout_safe_fallback(text, width, max_height)
            
            # Create Rich Text object
            rich_text = Text(text)
            
            # Configure Rich rendering options
            console_options = self._console.options.copy()
            console_options.max_width = width
            console_options.soft_wrap = wrap_mode != WrapMode.NONE
            
            # CRITICAL: Disable overflow patterns that cause dotted lines
            console_options.overflow = "crop"  # Never use "fold" which causes ...
            
            # Render with Rich
            rendered_lines = rich_text.wrap(
                console=self._console,
                width=width,
                no_wrap=(wrap_mode == WrapMode.NONE),
                tab_size=4
            )
            
            # Calculate actual dimensions
            lines = list(rendered_lines)
            actual_height = len(lines)
            max_line_width = max(len(line.plain) for line in lines) if lines else 0
            
            # Handle height overflow
            overflow_occurred = False
            if max_height and actual_height > max_height:
                overflow_occurred = True
                if overflow_handling == OverflowHandling.CLIP:
                    lines = lines[:max_height]
                    actual_height = max_height
                elif overflow_handling == OverflowHandling.ELLIPSIS:
                    lines = lines[:max_height-1]
                    lines.append(Text("▼ More content below"))
                    actual_height = max_height
            
            # Combine lines back to Rich Text
            final_text = Text()
            for i, line in enumerate(lines):
                final_text.append(line)
                if i < len(lines) - 1:
                    final_text.append("\n")
            
            return LayoutResult(
                laid_out_text=final_text,
                actual_dimensions=TextDimensions(
                    width=width,
                    height=max_height or actual_height,
                    actual_width=max_line_width,
                    actual_height=actual_height,
                    lines_count=len(lines),
                    max_line_width=max_line_width,
                    timestamp=datetime.now()
                ),
                overflow_occurred=overflow_occurred,
                layout_method="rich"
            )
            
        except Exception:
            # Fallback to safe layout on Rich errors
            return await self._layout_safe_fallback(text, width, max_height)
    
    async def _layout_with_smart_wrap(self,
                                     text: str,
                                     width: int,
                                     wrap_mode: WrapMode,
                                     overflow_handling: OverflowHandling,
                                     max_height: Optional[int]) -> LayoutResult:
        """Layout text with smart wrapping algorithms."""
        
        lines = []
        
        # First, split by explicit newlines to preserve line structure
        paragraphs = text.split('\n')
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                # Empty line
                lines.append("")
                continue
                
            # Process each paragraph for wrapping
            current_line = ""
            words = paragraph.split()
            
            for word in words:
                if not current_line:
                    current_line = word
                else:
                    test_line = current_line + " " + word
                    
                    if self._get_display_width(test_line) <= width:
                        current_line = test_line
                    else:
                        # Line would be too long, wrap
                        lines.append(current_line)
                        current_line = word
                        
                        # If single word is too long, break it
                        if self._get_display_width(current_line) > width:
                            wrapped_word = self._wrap_long_word(current_line, width)
                            lines.extend(wrapped_word[:-1])
                            current_line = wrapped_word[-1] if wrapped_word else ""
            
            # Add final line of paragraph
            if current_line:
                lines.append(current_line)
        
        # Handle empty result
        if not lines:
            lines = [""]
        
        # Calculate dimensions
        actual_height = len(lines)
        max_line_width = max(self._get_display_width(line) for line in lines)
        
        # Handle overflow
        overflow_occurred = False
        if max_height and actual_height > max_height:
            overflow_occurred = True
            if overflow_handling == OverflowHandling.CLIP:
                lines = lines[:max_height]
                actual_height = max_height
            elif overflow_handling == OverflowHandling.ELLIPSIS:
                if max_height > 0:
                    lines = lines[:max_height-1]
                    lines.append("▼ More content below")
                    actual_height = max_height
                else:
                    lines = ["▼ More content below"]
                    actual_height = 1
            elif overflow_handling == OverflowHandling.ADAPTIVE:
                # Choose best strategy: clip if lots of overflow, ellipsis if moderate
                if actual_height > max_height * 2:
                    # Lots of overflow - clip
                    lines = lines[:max_height]
                    actual_height = max_height
                else:
                    # Moderate overflow - ellipsis
                    if max_height > 0:
                        lines = lines[:max_height-1]
                        lines.append("▼ More content below")
                        actual_height = max_height
                    else:
                        lines = ["▼ More content below"]
                        actual_height = 1
        
        # Join lines
        final_text = "\n".join(lines)
        
        return LayoutResult(
            laid_out_text=final_text,
            actual_dimensions=TextDimensions(
                width=width,
                height=max_height or actual_height,
                actual_width=max_line_width,
                actual_height=actual_height,
                lines_count=len(lines),
                max_line_width=max_line_width,
                timestamp=datetime.now()
            ),
            overflow_occurred=overflow_occurred,
            layout_method="smart_wrap"
        )
    
    async def _layout_with_simple_wrap(self,
                                      text: str,
                                      width: int,
                                      wrap_mode: WrapMode,
                                      overflow_handling: OverflowHandling,
                                      max_height: Optional[int]) -> LayoutResult:
        """Layout text with simple wrapping."""
        
        lines = []
        
        if wrap_mode == WrapMode.NONE:
            # No wrapping - split only on existing newlines, keep long lines intact
            lines = text.split('\n')
        elif wrap_mode == WrapMode.WORD:
            # Word wrapping
            for paragraph in text.split('\n'):
                if not paragraph:
                    lines.append("")
                    continue
                    
                words = paragraph.split()
                current_line = ""
                
                for word in words:
                    if not current_line:
                        current_line = word
                    elif len(current_line) + 1 + len(word) <= width:
                        current_line += " " + word
                    else:
                        lines.append(current_line)
                        current_line = word
                
                if current_line:
                    lines.append(current_line)
        elif wrap_mode == WrapMode.CHAR:
            # Character wrapping
            for paragraph in text.split('\n'):
                if not paragraph:
                    lines.append("")
                    continue
                    
                while paragraph:
                    if len(paragraph) <= width:
                        lines.append(paragraph)
                        break
                    else:
                        lines.append(paragraph[:width])
                        paragraph = paragraph[width:]
        
        # Handle empty result
        if not lines:
            lines = [""]
        
        # Calculate dimensions
        actual_height = len(lines)
        max_line_width = max(len(line) for line in lines)
        
        # Handle overflow
        overflow_occurred = False
        if max_height and actual_height > max_height:
            overflow_occurred = True
            if overflow_handling == OverflowHandling.CLIP:
                lines = lines[:max_height]
                actual_height = max_height
            elif overflow_handling == OverflowHandling.ELLIPSIS:
                if max_height > 0:
                    lines = lines[:max_height-1]  
                    lines.append("▼ More content below")
                    actual_height = max_height
                else:
                    lines = ["▼ More content below"]
                    actual_height = 1
            elif overflow_handling == OverflowHandling.ADAPTIVE:
                # Choose best strategy: clip if lots of overflow, ellipsis if moderate
                if actual_height > max_height * 2:
                    # Lots of overflow - clip
                    lines = lines[:max_height]
                    actual_height = max_height
                else:
                    # Moderate overflow - ellipsis
                    if max_height > 0:
                        lines = lines[:max_height-1]
                        lines.append("▼ More content below")
                        actual_height = max_height
                    else:
                        lines = ["▼ More content below"]
                        actual_height = 1
        
        # Join lines
        final_text = "\n".join(lines)
        
        return LayoutResult(
            laid_out_text=final_text,
            actual_dimensions=TextDimensions(
                width=width,
                height=max_height or actual_height,
                actual_width=max_line_width,
                actual_height=actual_height,
                lines_count=len(lines),
                max_line_width=max_line_width,
                timestamp=datetime.now()
            ),
            overflow_occurred=overflow_occurred,
            layout_method="simple_wrap"
        )
    
    async def _layout_safe_fallback(self,
                                   text: str,
                                   width: int,
                                   max_height: Optional[int]) -> LayoutResult:
        """Safe fallback layout that never fails."""
        
        # Simple character-based truncation that never produces dotted lines
        lines = []
        
        for paragraph in text.split('\n'):
            if len(paragraph) <= width:
                lines.append(paragraph)
            else:
                # Break long lines but avoid ellipsis/dots
                while paragraph:
                    if len(paragraph) <= width:
                        lines.append(paragraph)
                        break
                    else:
                        lines.append(paragraph[:width])
                        paragraph = paragraph[width:]
        
        if not lines:
            lines = [""]
        
        # Handle height constraint
        overflow_occurred = False
        if max_height and len(lines) > max_height:
            overflow_occurred = True
            lines = lines[:max_height]
        
        actual_height = len(lines)
        max_line_width = max(len(line) for line in lines) if lines else 0
        
        final_text = "\n".join(lines)
        
        return LayoutResult(
            laid_out_text=final_text,
            actual_dimensions=TextDimensions(
                width=width,
                height=max_height or actual_height,
                actual_width=max_line_width,
                actual_height=actual_height,
                lines_count=len(lines),
                max_line_width=max_line_width,
                timestamp=datetime.now()
            ),
            overflow_occurred=overflow_occurred,
            layout_method="safe_fallback"
        )
    
    def _smart_split_words(self, text: str) -> List[str]:
        """Split text into words intelligently."""
        # Handle multiple whitespace types and keep word boundaries
        words = re.findall(r'\S+|\s+', text)
        return [word for word in words if word.strip()]
    
    def _wrap_long_word(self, word: str, width: int) -> List[str]:
        """Wrap a word that's longer than the container width."""
        if width <= 0:
            return [word]
        
        wrapped = []
        while len(word) > width:
            wrapped.append(word[:width])
            word = word[width:]
        
        if word:
            wrapped.append(word)
        
        return wrapped if wrapped else [word]
    
    def _get_display_width(self, text: str) -> int:
        """Get the display width of text (handling Unicode)."""
        if not self._unicode_enabled:
            return len(text)
        
        width = 0
        for char in text:
            if char in self._emoji_width_cache:
                width += self._emoji_width_cache[char]
            else:
                # Handle wide characters (CJK, emojis)
                char_width = self._calculate_char_width(char)
                self._emoji_width_cache[char] = char_width
                width += char_width
        
        return width
    
    def _calculate_char_width(self, char: str) -> int:
        """Calculate display width of a single character."""
        # Basic width calculation
        if unicodedata.east_asian_width(char) in 'WF':
            return 2  # Wide or fullwidth characters
        elif unicodedata.combining(char):
            return 0  # Combining characters
        else:
            return 1  # Normal characters
    
    def _safe_truncate(self, text: str, max_length: int) -> str:
        """Safely truncate text without ellipsis."""
        if len(text) <= max_length:
            return text
        
        # Find good break point
        truncated = text[:max_length]
        
        # Try to break on word boundary if close
        last_space = truncated.rfind(' ')
        if last_space > max_length * 0.8:  # If space is in last 20%
            truncated = truncated[:last_space]
        
        return truncated
    
    def _generate_cache_key(self, 
                           text: str, 
                           width: int,
                           wrap_mode: WrapMode,
                           overflow_handling: OverflowHandling,
                           max_height: Optional[int]) -> str:
        """Generate cache key for layout result."""
        # Create a hash-based key to avoid long text in keys
        import hashlib
        
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
        return f"{text_hash}_{width}_{wrap_mode.value}_{overflow_handling.value}_{max_height}"
    
    def clear_cache(self) -> None:
        """Clear layout cache."""
        self._layout_cache.clear()
        self._emoji_width_cache.clear()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for layout operations."""
        return {
            'operation_times': dict(self._operation_times),
            'cache_size': len(self._layout_cache),
            'emoji_cache_size': len(self._emoji_width_cache),
            'max_layout_time_ms': self._max_layout_time_ms,
            'chars_per_ms_target': self._chars_per_ms_target
        }


# Convenience functions for easy usage
async def layout_text_simple(text: str, 
                            width: int, 
                            wrap: bool = True,
                            max_height: Optional[int] = None) -> str:
    """Simple text layout function."""
    engine = TextLayoutEngine()
    
    wrap_mode = WrapMode.WORD if wrap else WrapMode.NONE
    result = await engine.layout_text(
        text_content=text,
        container_width=width,
        wrap_mode=wrap_mode,
        max_height=max_height
    )
    
    if isinstance(result.laid_out_text, str):
        return result.laid_out_text
    else:
        # Handle Rich Text object
        return result.laid_out_text.plain if hasattr(result.laid_out_text, 'plain') else str(result.laid_out_text)


async def eliminate_dotted_lines(text: str, width: int) -> str:
    """Eliminate any dotted line patterns from text."""
    engine = TextLayoutEngine()
    
    result = await engine.layout_text(
        text_content=text,
        container_width=width,
        wrap_mode=WrapMode.SMART,
        overflow_handling=OverflowHandling.WRAP
    )
    
    laid_out = result.laid_out_text
    if isinstance(laid_out, str):
        # Remove any remaining ellipsis patterns
        laid_out = re.sub(r'\.{3,}', '', laid_out)
        laid_out = re.sub(r'…+', '', laid_out)
        return laid_out
    else:
        return laid_out.plain if hasattr(laid_out, 'plain') else str(laid_out)