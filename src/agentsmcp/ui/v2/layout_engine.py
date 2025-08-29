"""
Simple layout system for TUI components.

Basic box model for positioning components like input field, chat history, 
and status bar. Responsive to terminal size changes with clean layout calculation.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

from .terminal_manager import TerminalManager


logger = logging.getLogger(__name__)


class LayoutDirection(Enum):
    """Layout direction options."""
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"


class Alignment(Enum):
    """Alignment options."""
    START = "start"
    CENTER = "center"
    END = "end"
    STRETCH = "stretch"


class SizeUnit(Enum):
    """Size unit types."""
    PIXELS = "px"      # Absolute characters
    PERCENT = "pct"    # Percentage of available space
    AUTO = "auto"      # Auto-sized to content


@dataclass
class Size:
    """Represents a size with value and unit."""
    value: int
    unit: SizeUnit = SizeUnit.PIXELS
    
    @classmethod
    def pixels(cls, value: int) -> 'Size':
        """Create pixel-based size."""
        return cls(value, SizeUnit.PIXELS)
    
    @classmethod
    def percent(cls, value: int) -> 'Size':
        """Create percentage-based size."""
        return cls(value, SizeUnit.PERCENT)
    
    @classmethod
    def auto(cls) -> 'Size':
        """Create auto-sized."""
        return cls(0, SizeUnit.AUTO)


@dataclass
class Padding:
    """Represents padding around content."""
    top: int = 0
    right: int = 0
    bottom: int = 0
    left: int = 0
    
    @classmethod
    def all(cls, value: int) -> 'Padding':
        """Create uniform padding."""
        return cls(value, value, value, value)
    
    @classmethod
    def symmetric(cls, vertical: int, horizontal: int) -> 'Padding':
        """Create symmetric padding."""
        return cls(vertical, horizontal, vertical, horizontal)


class Rectangle(NamedTuple):
    """Represents a rectangle position and size."""
    x: int
    y: int
    width: int
    height: int
    
    @property
    def right(self) -> int:
        """Right edge position."""
        return self.x + self.width
    
    @property
    def bottom(self) -> int:
        """Bottom edge position."""
        return self.y + self.height
    
    def contains_point(self, x: int, y: int) -> bool:
        """Check if point is within rectangle."""
        return (self.x <= x < self.right and 
                self.y <= y < self.bottom)
    
    def intersects(self, other: 'Rectangle') -> bool:
        """Check if this rectangle intersects with another."""
        return (self.x < other.right and self.right > other.x and
                self.y < other.bottom and self.bottom > other.y)


class LayoutNode(ABC):
    """Base class for layout nodes."""
    
    def __init__(self, 
                 name: str,
                 width: Size = Size.auto(),
                 height: Size = Size.auto(),
                 padding: Optional[Padding] = None):
        """Initialize layout node."""
        self.name = name
        self.width = width
        self.height = height
        self.padding = padding or Padding()
        self.parent: Optional['ContainerNode'] = None
        self._computed_rect: Optional[Rectangle] = None
        self._min_width = 1
        self._min_height = 1
    
    @abstractmethod
    def get_content_size(self, available_width: int, available_height: int) -> Tuple[int, int]:
        """Get the natural content size."""
        pass
    
    def compute_size(self, available_width: int, available_height: int) -> Tuple[int, int]:
        """Compute the actual size given available space."""
        content_width, content_height = self.get_content_size(available_width, available_height)
        
        # Add padding
        total_width = content_width + self.padding.left + self.padding.right
        total_height = content_height + self.padding.top + self.padding.bottom
        
        # Apply width constraints
        if self.width.unit == SizeUnit.PIXELS:
            final_width = max(self._min_width, self.width.value)
        elif self.width.unit == SizeUnit.PERCENT:
            final_width = max(self._min_width, (available_width * self.width.value) // 100)
        else:  # AUTO
            final_width = max(self._min_width, min(total_width, available_width))
        
        # Apply height constraints
        if self.height.unit == SizeUnit.PIXELS:
            final_height = max(self._min_height, self.height.value)
        elif self.height.unit == SizeUnit.PERCENT:
            final_height = max(self._min_height, (available_height * self.height.value) // 100)
        else:  # AUTO
            final_height = max(self._min_height, min(total_height, available_height))
        
        return final_width, final_height
    
    def set_computed_rect(self, rect: Rectangle):
        """Set the computed rectangle for this node."""
        self._computed_rect = rect
    
    def get_computed_rect(self) -> Optional[Rectangle]:
        """Get the computed rectangle."""
        return self._computed_rect
    
    def get_content_rect(self) -> Optional[Rectangle]:
        """Get the content rectangle (minus padding)."""
        if not self._computed_rect:
            return None
        
        return Rectangle(
            x=self._computed_rect.x + self.padding.left,
            y=self._computed_rect.y + self.padding.top,
            width=max(0, self._computed_rect.width - self.padding.left - self.padding.right),
            height=max(0, self._computed_rect.height - self.padding.top - self.padding.bottom)
        )


class TextNode(LayoutNode):
    """Layout node for text content."""
    
    def __init__(self, 
                 name: str,
                 content: str = "",
                 **kwargs):
        """Initialize text node."""
        super().__init__(name, **kwargs)
        self.content = content
    
    def get_content_size(self, available_width: int, available_height: int) -> Tuple[int, int]:
        """Get text content size."""
        if not self.content:
            return 0, 0
        
        lines = self.content.split('\n')
        
        # Calculate width (longest line)
        content_width = max(len(line) for line in lines) if lines else 0
        content_height = len(lines)
        
        return content_width, content_height
    
    def set_content(self, content: str):
        """Update text content."""
        self.content = content


class ContainerNode(LayoutNode):
    """Layout node that contains other nodes."""
    
    def __init__(self,
                 name: str,
                 direction: LayoutDirection = LayoutDirection.VERTICAL,
                 alignment: Alignment = Alignment.START,
                 gap: int = 0,
                 **kwargs):
        """Initialize container node."""
        super().__init__(name, **kwargs)
        self.direction = direction
        self.alignment = alignment
        self.gap = gap
        self.children: List[LayoutNode] = []
    
    def add_child(self, child: LayoutNode):
        """Add a child node."""
        child.parent = self
        self.children.append(child)
    
    def remove_child(self, child: LayoutNode):
        """Remove a child node."""
        if child in self.children:
            child.parent = None
            self.children.remove(child)
    
    def get_content_size(self, available_width: int, available_height: int) -> Tuple[int, int]:
        """Get container content size based on children."""
        if not self.children:
            return 0, 0
        
        if self.direction == LayoutDirection.VERTICAL:
            total_width = 0
            total_height = 0
            
            for child in self.children:
                child_width, child_height = child.compute_size(available_width, available_height)
                total_width = max(total_width, child_width)
                total_height += child_height
            
            # Add gaps
            if len(self.children) > 1:
                total_height += self.gap * (len(self.children) - 1)
            
            return total_width, total_height
        
        else:  # HORIZONTAL
            total_width = 0
            total_height = 0
            
            for child in self.children:
                child_width, child_height = child.compute_size(available_width, available_height)
                total_width += child_width
                total_height = max(total_height, child_height)
            
            # Add gaps
            if len(self.children) > 1:
                total_width += self.gap * (len(self.children) - 1)
            
            return total_width, total_height
    
    def layout_children(self, content_rect: Rectangle):
        """Layout children within the content rectangle."""
        if not self.children:
            return
        
        if self.direction == LayoutDirection.VERTICAL:
            self._layout_children_vertical(content_rect)
        else:
            self._layout_children_horizontal(content_rect)
    
    def _layout_children_vertical(self, rect: Rectangle):
        """Layout children vertically."""
        current_y = rect.y
        available_height = rect.height
        
        # Calculate total gap space
        total_gap = self.gap * max(0, len(self.children) - 1)
        available_height -= total_gap
        
        for child in self.children:
            child_width, child_height = child.compute_size(rect.width, available_height)
            
            # Apply alignment for width
            child_x = rect.x
            if self.alignment == Alignment.CENTER:
                child_x = rect.x + (rect.width - child_width) // 2
            elif self.alignment == Alignment.END:
                child_x = rect.x + rect.width - child_width
            elif self.alignment == Alignment.STRETCH:
                child_width = rect.width
            
            # Ensure child stays within bounds
            child_width = min(child_width, rect.width)
            child_height = min(child_height, available_height)
            
            child_rect = Rectangle(child_x, current_y, child_width, child_height)
            child.set_computed_rect(child_rect)
            
            # If child is also a container, layout its children
            if isinstance(child, ContainerNode):
                content_rect = child.get_content_rect()
                if content_rect:
                    child.layout_children(content_rect)
            
            current_y += child_height + self.gap
            available_height -= child_height
    
    def _layout_children_horizontal(self, rect: Rectangle):
        """Layout children horizontally."""
        current_x = rect.x
        available_width = rect.width
        
        # Calculate total gap space
        total_gap = self.gap * max(0, len(self.children) - 1)
        available_width -= total_gap
        
        for child in self.children:
            child_width, child_height = child.compute_size(available_width, rect.height)
            
            # Apply alignment for height
            child_y = rect.y
            if self.alignment == Alignment.CENTER:
                child_y = rect.y + (rect.height - child_height) // 2
            elif self.alignment == Alignment.END:
                child_y = rect.y + rect.height - child_height
            elif self.alignment == Alignment.STRETCH:
                child_height = rect.height
            
            # Ensure child stays within bounds
            child_width = min(child_width, available_width)
            child_height = min(child_height, rect.height)
            
            child_rect = Rectangle(current_x, child_y, child_width, child_height)
            child.set_computed_rect(child_rect)
            
            # If child is also a container, layout its children
            if isinstance(child, ContainerNode):
                content_rect = child.get_content_rect()
                if content_rect:
                    child.layout_children(content_rect)
            
            current_x += child_width + self.gap
            available_width -= child_width


class LayoutEngine:
    """
    Simple layout system for TUI components.
    
    Provides basic box model positioning for components with responsive
    layout calculation and terminal size handling.
    """
    
    def __init__(self, terminal_manager: Optional[TerminalManager] = None):
        """Initialize layout engine."""
        self.terminal_manager = terminal_manager or TerminalManager()
        self.root: Optional[LayoutNode] = None
        self._last_terminal_size: Optional[Tuple[int, int]] = None
        self._layout_cache: Dict[str, Rectangle] = {}
        
    def set_root(self, root: LayoutNode):
        """Set the root layout node."""
        self.root = root
        self._invalidate_layout()
    
    def _invalidate_layout(self):
        """Invalidate layout cache."""
        self._layout_cache.clear()
        self._last_terminal_size = None
    
    def compute_layout(self, force: bool = False) -> bool:
        """
        Compute layout for all nodes.
        
        Args:
            force: Force recomputation even if terminal size unchanged
            
        Returns:
            True if layout was computed
        """
        if not self.root:
            return False
        
        terminal_width, terminal_height = self.terminal_manager.get_size()
        current_size = (terminal_width, terminal_height)
        
        # Check if recomputation needed
        if not force and current_size == self._last_terminal_size:
            return False
        
        try:
            # Compute root size and position
            root_width, root_height = self.root.compute_size(terminal_width, terminal_height)
            
            # Position root at origin
            root_rect = Rectangle(0, 0, root_width, root_height)
            self.root.set_computed_rect(root_rect)
            
            # Layout children if root is a container
            if isinstance(self.root, ContainerNode):
                content_rect = self.root.get_content_rect()
                if content_rect:
                    self.root.layout_children(content_rect)
            
            self._last_terminal_size = current_size
            logger.debug(f"Layout computed for {terminal_width}x{terminal_height}")
            return True
            
        except Exception as e:
            logger.error(f"Error computing layout: {e}")
            return False
    
    def find_node(self, name: str) -> Optional[LayoutNode]:
        """Find a node by name."""
        if not self.root:
            return None
        
        def _search(node: LayoutNode) -> Optional[LayoutNode]:
            if node.name == name:
                return node
            
            if isinstance(node, ContainerNode):
                for child in node.children:
                    result = _search(child)
                    if result:
                        return result
            
            return None
        
        return _search(self.root)
    
    def get_node_rect(self, name: str) -> Optional[Rectangle]:
        """Get computed rectangle for a named node."""
        node = self.find_node(name)
        return node.get_computed_rect() if node else None
    
    def get_node_content_rect(self, name: str) -> Optional[Rectangle]:
        """Get content rectangle for a named node."""
        node = self.find_node(name)
        return node.get_content_rect() if node else None
    
    def handle_resize(self):
        """Handle terminal resize."""
        self.terminal_manager.refresh_capabilities()
        self._invalidate_layout()
        self.compute_layout(force=True)
    
    def get_layout_info(self) -> Dict[str, Any]:
        """Get comprehensive layout information."""
        if not self.root:
            return {'nodes': 0, 'terminal_size': self.terminal_manager.get_size()}
        
        def _collect_node_info(node: LayoutNode) -> Dict[str, Any]:
            info = {
                'name': node.name,
                'type': node.__class__.__name__,
                'computed_rect': node.get_computed_rect(),
                'content_rect': node.get_content_rect()
            }
            
            if isinstance(node, ContainerNode):
                info['direction'] = node.direction.value
                info['children'] = [_collect_node_info(child) for child in node.children]
            elif isinstance(node, TextNode):
                info['content_length'] = len(node.content)
            
            return info
        
        return {
            'terminal_size': self.terminal_manager.get_size(),
            'last_computed_size': self._last_terminal_size,
            'root': _collect_node_info(self.root)
        }


def create_standard_tui_layout(terminal_manager: Optional[TerminalManager] = None) -> Tuple[LayoutEngine, Dict[str, str]]:
    """
    Create a standard TUI layout with common components.
    
    Returns:
        Tuple of (layout_engine, node_names_dict)
    """
    engine = LayoutEngine(terminal_manager)
    
    # Root container
    root = ContainerNode(
        name="root",
        direction=LayoutDirection.VERTICAL,
        width=Size.percent(100),
        height=Size.percent(100)
    )
    
    # Status bar at top
    status_bar = TextNode(
        name="status_bar",
        height=Size.pixels(1),
        width=Size.percent(100),
        padding=Padding.symmetric(0, 1)
    )
    
    # Chat history (main content area)
    chat_history = TextNode(
        name="chat_history",
        height=Size.percent(80),  # Most of the screen
        width=Size.percent(100),
        padding=Padding.all(1)
    )
    
    # Input area container
    input_container = ContainerNode(
        name="input_container",
        direction=LayoutDirection.VERTICAL,
        height=Size.percent(20),
        width=Size.percent(100),
        gap=1
    )
    
    # Input field
    input_field = TextNode(
        name="input_field",
        height=Size.pixels(3),
        width=Size.percent(100),
        padding=Padding.symmetric(1, 2)
    )
    
    # Input status line
    input_status = TextNode(
        name="input_status",
        height=Size.pixels(1),
        width=Size.percent(100),
        padding=Padding.symmetric(0, 2)
    )
    
    # Build layout tree
    input_container.add_child(input_field)
    input_container.add_child(input_status)
    
    root.add_child(status_bar)
    root.add_child(chat_history)
    root.add_child(input_container)
    
    engine.set_root(root)
    
    # Return node names for easy reference
    node_names = {
        'root': 'root',
        'status_bar': 'status_bar',
        'chat_history': 'chat_history',
        'input_container': 'input_container',
        'input_field': 'input_field',
        'input_status': 'input_status'
    }
    
    return engine, node_names