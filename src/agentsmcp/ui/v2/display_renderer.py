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
    
    def initialize(self) -> bool:
        """
        Initialize the display renderer.
        
        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True
        
        try:
            caps = self.terminal_manager.detect_capabilities()
            self._render_mode = self._determine_render_mode(caps)
            
            if self._render_mode in (RenderMode.FULL_SCREEN, RenderMode.IN_PLACE):
                self._setup_terminal_control()
            
            self._initialized = True
            logger.debug(f"Display renderer initialized in {self._render_mode.value} mode")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize display renderer: {e}")
            self._render_mode = RenderMode.FALLBACK
            return False
    
    def cleanup(self):
        """Clean up terminal state."""
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
        if not caps.interactive:
            return RenderMode.FALLBACK
        
        if caps.cursor_control and caps.alternate_screen and caps.width >= 60:
            return RenderMode.FULL_SCREEN
        elif caps.cursor_control and caps.width >= 40:
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
        return hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
    
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
        Update content in a region with change detection.
        
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
        
        # Change detection
        content_hash = self._hash_content(content)
        if not force and content_hash == region.content_hash:
            self._stats['renders_skipped'] += 1
            return False
        
        # Update region
        region.content_hash = content_hash
        region.last_content = content
        region.dirty = True
        
        return self._render_region(region, content)
    
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
        lines = content.split('\n')
        
        # Clear region first
        self._clear_region(region)
        
        # Render content line by line
        for i, line in enumerate(lines[:region.height]):
            if i >= region.height:
                break
            
            self._move_cursor(region.x, region.y + i)
            
            # Truncate line if too long
            display_line = line[:region.width]
            self._output.write(display_line)
        
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
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()