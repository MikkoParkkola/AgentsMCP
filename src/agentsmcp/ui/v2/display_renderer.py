"""
Display renderer with direct terminal control.

Clean terminal output without scrollback pollution - the #1 user complaint.
Uses direct cursor control instead of Rich Live for fixed screen positions 
with in-place updates.
"""

import sys
import os
import hashlib
import logging
import asyncio
import threading
import time
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from .terminal_manager import TerminalManager, TerminalCapabilities


logger = logging.getLogger(__name__)


class RenderMode(Enum):
    """Rendering modes for different terminal capabilities."""
    FULL_SCREEN = "full_screen"     # Full cursor control, alternate screen
    IN_PLACE = "in_place"           # In-place updates, no alternate screen
    LINE_BASED = "line_based"       # Line-by-line output, minimal control
    FALLBACK = "fallback"           # Basic text output only


@dataclass
class RenderRegion:
    """Defines a rendering region on the terminal."""
    name: str
    x: int
    y: int
    width: int
    height: int
    content_hash: str = ""
    last_content: str = ""
    dirty: bool = True


class TerminalState(NamedTuple):
    """Current terminal state."""
    width: int
    height: int
    cursor_x: int
    cursor_y: int


class DisplayRenderer:
    """
    Clean terminal display renderer without scrollback pollution.
    
    Provides direct cursor control for fixed screen positions with
    change detection to prevent redundant renders.
    """
    
    def __init__(self, terminal_manager: Optional[TerminalManager] = None):
        """Initialize the display renderer."""
        self.terminal_manager = terminal_manager or TerminalManager()
        self._regions: Dict[str, RenderRegion] = {}
        self._render_mode = RenderMode.FALLBACK
        self._initialized = False
        self._alternate_screen = False
        self._original_cursor_pos: Optional[Tuple[int, int]] = None
        self._last_terminal_state: Optional[TerminalState] = None
        
        # Output stream
        self._output = sys.stdout
        
        # Performance tracking
        self._stats = {
            'renders_total': 0,
            'renders_skipped': 0,
            'regions_updated': 0,
            'last_render_time': 0.0
        }
    
    async def initialize(self) -> bool:
        """
        Initialize the display renderer with race condition protection.
        
        Returns:
            True if initialization successful
        """
        # THREAT: Race condition during initialization
        # MITIGATION: Use asyncio Lock for atomic initialization
        if not hasattr(self, '_init_lock'):
            self._init_lock = asyncio.Lock()
        
        async with self._init_lock:
            if self._initialized:
                return True
            
            try:
                # SECURITY: Validate terminal manager before use
                if not self.terminal_manager:
                    logger.error("Terminal manager not available")
                    return False
                
                # Ensure terminal manager is initialized with timeout
                if hasattr(self.terminal_manager, 'initialize') and hasattr(self.terminal_manager, '_initialized'):
                    if not self.terminal_manager._initialized:
                        try:
                            init_result = await asyncio.wait_for(
                                self.terminal_manager.initialize(), 
                                timeout=2.0  # PERFORMANCE: 2s timeout for initialization
                            )
                            if not init_result:
                                logger.error("Terminal manager initialization failed")
                                return False
                        except asyncio.TimeoutError:
                            logger.error("Terminal manager initialization timed out")
                            return False
                
                # SECURITY: Validate capabilities before proceeding
                caps = self.terminal_manager.detect_capabilities()
                if not caps:
                    logger.error("Unable to detect terminal capabilities")
                    return False
                
                self._render_mode = self._determine_render_mode(caps)
                
                if self._render_mode in (RenderMode.FULL_SCREEN, RenderMode.IN_PLACE):
                    self._setup_terminal_control()
                
                self._initialized = True
                logger.debug(f"Display renderer initialized in {self._render_mode.value} mode")
                return True
                
            except Exception as e:
                logger.error(f"Failed to initialize display renderer: {e}")
                self._render_mode = RenderMode.FALLBACK
                # SECURITY: Clear partial state on failure
                self._initialized = False
                return False
    
    def cleanup_sync(self):
        """Synchronous cleanup - for context managers and immediate cleanup."""
        if not self._initialized:
            return
            
        try:
            if self._alternate_screen:
                self._exit_alternate_screen()
            
            if self._original_cursor_pos:
                self._move_cursor(*self._original_cursor_pos)
            
            # Show cursor
            self._output.write('\033[?25h')
            self._output.flush()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        self._initialized = False
        self._alternate_screen = False
    
    def _determine_render_mode(self, caps: TerminalCapabilities) -> RenderMode:
        """Determine the best rendering mode for the terminal."""
        # CRITICAL FIX: Force FULL_SCREEN mode when alternate screen is available
        # to prevent scrollback flooding - this was the #1 user complaint
        if caps.cursor_control and caps.alternate_screen and caps.width >= 60:
            return RenderMode.FULL_SCREEN
        elif caps.cursor_control and caps.width >= 40:
            # Use IN_PLACE for terminals that support cursor control but not alternate screen
            return RenderMode.IN_PLACE
        elif caps.width >= 20:
            return RenderMode.LINE_BASED
        else:
            return RenderMode.FALLBACK
    
    def _setup_terminal_control(self):
        """Setup terminal for cursor control."""
        if self._render_mode == RenderMode.FULL_SCREEN:
            # Enter alternate screen to avoid scrollback pollution
            self._enter_alternate_screen()
        
        # Hide cursor during rendering
        self._output.write('\033[?25l')
        
        # Save current cursor position
        self._output.write('\033[s')
        self._output.flush()
    
    def _enter_alternate_screen(self):
        """Enter alternate screen buffer."""
        if not self._alternate_screen:
            self._output.write('\033[?1049h')  # Save screen and enter alternate
            self._output.flush()
            self._alternate_screen = True
    
    def _exit_alternate_screen(self):
        """Exit alternate screen buffer."""
        if self._alternate_screen:
            self._output.write('\033[?1049l')  # Restore screen and exit alternate
            self._output.flush()
            self._alternate_screen = False
    
    def _move_cursor(self, x: int, y: int):
        """Move cursor to specific position."""
        self._output.write(f'\033[{y + 1};{x + 1}H')  # 1-based positioning
    
    def _clear_region(self, region: RenderRegion):
        """Clear a specific region."""
        for row in range(region.height):
            self._move_cursor(region.x, region.y + row)
            self._output.write(' ' * region.width)
    
    def _hash_content(self, content: str) -> str:
        """Generate hash for content change detection."""
        # PERFORMANCE: Use faster hash algorithm for real-time updates
        return str(hash(content))[:8]
    
    def define_region(self, name: str, x: int, y: int, width: int, height: int) -> bool:
        """
        Define a rendering region.
        
        Args:
            name: Unique region name
            x, y: Top-left position
            width, height: Region dimensions
            
        Returns:
            True if region was defined successfully
        """
        if not self._initialized:
            logger.warning("Renderer not initialized")
            return False
        
        # Validate region bounds
        term_width, term_height = self.terminal_manager.get_size()
        
        if x < 0 or y < 0 or x + width > term_width or y + height > term_height:
            logger.warning(f"Region {name} bounds invalid: ({x},{y},{width},{height}) for terminal {term_width}x{term_height}")
            return False
        
        region = RenderRegion(
            name=name,
            x=x,
            y=y,
            width=width,
            height=height
        )
        
        self._regions[name] = region
        logger.debug(f"Defined region {name}: {x},{y} {width}x{height}")
        return True
    
    def update_region(self, name: str, content: str, force: bool = False) -> bool:
        """
        Update content in a region with change detection and performance monitoring.
        
        Args:
            name: Region name
            content: New content
            force: Force update even if content unchanged
            
        Returns:
            True if region was updated
        """
        if not self._initialized:
            return False
        
        region = self._regions.get(name)
        if not region:
            logger.warning(f"Region {name} not defined")
            return False
        
        # PERFORMANCE: Early exit for empty content updates
        if not content and not region.last_content:
            return False
        
        # SECURITY: Validate content size to prevent memory exhaustion
        if len(content) > 1024 * 1024:  # 1MB limit
            logger.warning(f"Content too large for region {name}: {len(content)} bytes")
            return False
        
        # Change detection with performance tracking
        start_time = time.time()
        content_hash = self._hash_content(content)
        
        if not force and content_hash == region.content_hash:
            self._stats['renders_skipped'] += 1
            return False
        
        # Update region
        region.content_hash = content_hash
        region.last_content = content
        region.dirty = True
        
        result = self._render_region(region, content)
        
        # PERFORMANCE: Track hash computation time
        hash_time = time.time() - start_time
        if hash_time > 0.001:  # Log if hashing takes >1ms
            logger.debug(f"Slow content hashing for {name}: {hash_time*1000:.1f}ms")
        
        return result
    
    def _render_region(self, region: RenderRegion, content: str) -> bool:
        """Render content to a specific region."""
        try:
            if self._render_mode == RenderMode.FALLBACK:
                return self._render_fallback(region, content)
            elif self._render_mode == RenderMode.LINE_BASED:
                return self._render_line_based(region, content)
            else:
                return self._render_positioned(region, content)
                
        except Exception as e:
            logger.error(f"Error rendering region {region.name}: {e}")
            return False
    
    def _render_positioned(self, region: RenderRegion, content: str) -> bool:
        """Render with positioned cursor control."""
        # Avoid clearing the entire region to reduce flicker; instead pad each
        # line to full width and ensure we write exactly region.height rows.
        raw_lines = content.split('\n') if content is not None else []
        width = region.width
        height = region.height
        
        for i in range(height):
            line = raw_lines[i] if i < len(raw_lines) else ""
            if len(line) < width:
                line = line + (" " * (width - len(line)))
            else:
                line = line[:width]
            self._move_cursor(region.x, region.y + i)
            self._output.write(line)
        
        self._output.flush()
        region.dirty = False
        self._stats['regions_updated'] += 1
        return True
    
    def _render_line_based(self, region: RenderRegion, content: str) -> bool:
        """Render in line-based mode (minimal cursor control)."""
        # In line-based mode, we print with clear separators
        # but avoid complex positioning
        
        if region.name == "status":
            # Status gets a separator
            self._output.write(f"\n--- {region.name.upper()} ---\n")
        
        self._output.write(content)
        self._output.write('\n')
        self._output.flush()
        
        region.dirty = False
        self._stats['regions_updated'] += 1
        return True
    
    def _render_fallback(self, region: RenderRegion, content: str) -> bool:
        """Fallback rendering (basic text output)."""
        # Just output the content with a simple separator
        if content.strip():
            self._output.write(f"[{region.name}] {content}\n")
            self._output.flush()
        
        region.dirty = False
        self._stats['regions_updated'] += 1
        return True
    
    def render_all(self, force: bool = False) -> int:
        """
        Render all dirty regions.
        
        Args:
            force: Force render all regions
            
        Returns:
            Number of regions rendered
        """
        if not self._initialized:
            return 0
        
        rendered_count = 0
        start_time = datetime.now()
        
        try:
            for region in self._regions.values():
                if force or region.dirty:
                    if self._render_region(region, region.last_content):
                        rendered_count += 1
            
            self._stats['renders_total'] += 1
            self._stats['last_render_time'] = (datetime.now() - start_time).total_seconds()
            
        except Exception as e:
            logger.error(f"Error in render_all: {e}")
        
        return rendered_count
    
    def clear_all_regions(self):
        """Clear all regions."""
        if not self._initialized:
            return
        
        if self._render_mode in (RenderMode.FULL_SCREEN, RenderMode.IN_PLACE):
            # Clear screen
            self._output.write('\033[2J')  # Clear entire screen
            self._output.write('\033[H')   # Move cursor to home
            self._output.flush()
        
        # Mark all regions as dirty
        for region in self._regions.values():
            region.dirty = True
            region.content_hash = ""
            region.last_content = ""
    
    def get_region_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a region."""
        region = self._regions.get(name)
        if not region:
            return None
        
        return {
            'name': region.name,
            'x': region.x,
            'y': region.y,
            'width': region.width,
            'height': region.height,
            'dirty': region.dirty,
            'content_length': len(region.last_content),
            'content_hash': region.content_hash
        }
    
    def format_status_bar(self, content: str, width: int = 80) -> str:
        """Format content as a status bar with borders."""
        if width < 10:
            return content[:width] if content else ""
        
        # Create bordered status bar
        top_border = "┌" + "─" * (width - 2) + "┐"
        content_line = f"│ {content:<{width - 4}} │"
        bottom_border = "└" + "─" * (width - 2) + "┘"
        
        return "\n".join([top_border, content_line, bottom_border])
    
    def format_section_header(self, title: str, width: int = 80, style: str = "double") -> str:
        """Format a section header with borders."""
        if width < len(title) + 4:
            return title
        
        if style == "double":
            # Double line style for main sections
            top = "╔" + "═" * (width - 2) + "╗"
            middle = f"║ {title:^{width - 4}} ║"
            bottom = "╚" + "═" * (width - 2) + "╝"
        elif style == "single":
            # Single line style for subsections  
            top = "┌" + "─" * (width - 2) + "┐"
            middle = f"│ {title:^{width - 4}} │"
            bottom = "└" + "─" * (width - 2) + "┘"
        else:
            # Simple style
            line = "─" * width
            middle = f" {title} "
            return f"{line}\n{middle}\n{line}"
        
        return "\n".join([top, middle, bottom])
    
    def format_message_box(self, content: str, width: int = 80, box_type: str = "info") -> str:
        """Format content in a message box with appropriate styling."""
        if width < 10:
            return content
        
        # Choose box styling based on type
        if box_type == "error":
            icon = "❌"
            border_char = "═"
            corner_tl, corner_tr = "╔", "╗"
            corner_bl, corner_br = "╚", "╝"
            side_char = "║"
        elif box_type == "warning":
            icon = "⚠️"
            border_char = "─"
            corner_tl, corner_tr = "┌", "┐"
            corner_bl, corner_br = "└", "┘"
            side_char = "│"
        elif box_type == "success":
            icon = "✅"
            border_char = "─"
            corner_tl, corner_tr = "┌", "┐"
            corner_bl, corner_br = "└", "┘"
            side_char = "│"
        else:  # info
            icon = "ℹ️"
            border_char = "─"
            corner_tl, corner_tr = "┌", "┐"
            corner_bl, corner_br = "└", "┘"
            side_char = "│"
        
        # Split content into lines and wrap if needed
        lines = []
        for line in content.split('\n'):
            if len(line) <= width - 6:  # Account for icon, borders, and padding
                lines.append(line)
            else:
                # Simple word wrapping
                words = line.split()
                current_line = ""
                for word in words:
                    if len(current_line + word + 1) <= width - 6:
                        current_line += word + " " if current_line else word
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = word
                if current_line:
                    lines.append(current_line)
        
        # Build the box
        result_lines = []
        
        # Top border
        result_lines.append(corner_tl + border_char * (width - 2) + corner_tr)
        
        # Content lines with icon on first line
        for i, line in enumerate(lines):
            if i == 0:
                content_text = f"{icon} {line}"
            else:
                content_text = f"  {line}"
            
            padded_content = f"{side_char} {content_text:<{width - 4}} {side_char}"
            result_lines.append(padded_content)
        
        # Bottom border
        result_lines.append(corner_bl + border_char * (width - 2) + corner_br)
        
        return "\n".join(result_lines)
    
    def format_list_items(self, items: List[str], width: int = 80, bullet: str = "•") -> str:
        """Format a list of items with consistent indentation."""
        if not items:
            return ""
        
        formatted_lines = []
        for item in items:
            if len(item) <= width - 4:
                formatted_lines.append(f"  {bullet} {item}")
            else:
                # Wrap long items
                words = item.split()
                current_line = ""
                for word in words:
                    if len(current_line + word + 1) <= width - 6:
                        current_line += word + " " if current_line else word
                    else:
                        if current_line:
                            if not formatted_lines or not formatted_lines[-1].startswith(f"  {bullet}"):
                                formatted_lines.append(f"  {bullet} {current_line}")
                            else:
                                formatted_lines.append(f"    {current_line}")
                        current_line = word
                if current_line:
                    if not formatted_lines:
                        formatted_lines.append(f"  {bullet} {current_line}")
                    else:
                        formatted_lines.append(f"    {current_line}")
        
        return "\n".join(formatted_lines)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get renderer statistics."""
        return {
            'initialized': self._initialized,
            'render_mode': self._render_mode.value,
            'alternate_screen': self._alternate_screen,
            'regions_count': len(self._regions),
            'regions_dirty': sum(1 for r in self._regions.values() if r.dirty),
            **self._stats
        }
    
    def handle_resize(self):
        """Handle terminal resize."""
        if not self._initialized:
            return
        
        # Force refresh of terminal capabilities
        self.terminal_manager.refresh_capabilities()
        
        # Clear screen and mark all regions dirty
        self.clear_all_regions()
        
        logger.debug("Handled terminal resize")
    
    def __enter__(self):
        """Context manager entry - synchronous wrapper."""
        # THREAT: Event loop conflicts
        # MITIGATION: Check for existing event loop
        try:
            loop = asyncio.get_running_loop()
            # If there's a running loop, we can't use asyncio.run()
            raise RuntimeError("DisplayRenderer context manager cannot be used with active event loop")
        except RuntimeError as e:
            if "no running event loop" not in str(e):
                raise
            # No running loop, safe to proceed
            result = asyncio.run(self.initialize())
            if not result:
                raise RuntimeError("Failed to initialize DisplayRenderer")
            return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - synchronous cleanup."""
        # THREAT: Event loop conflicts in cleanup
        # MITIGATION: Use synchronous cleanup method
        self.cleanup_sync()
    
    async def cleanup(self):
        """
        Async cleanup the display renderer.
        
        Restores terminal state and clears all regions.
        """
        try:
            if not self._initialized:
                return
            
            # Clear all regions first  
            self.clear_all_regions()
            
            # Use synchronous cleanup for terminal operations
            self.cleanup_sync()
            
            # Reset additional state
            self._regions.clear()
            self._last_terminal_state = None
            
            logger.debug("Display renderer async cleanup completed")
            
        except Exception as e:
            logger.warning(f"Error during display renderer cleanup: {e}")