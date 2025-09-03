"""Base classes and interfaces for UI renderers."""

from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict
from dataclasses import dataclass


@dataclass
class UIState:
    """Unified UI state structure."""
    
    # Input state
    current_input: str = ""
    input_cursor_pos: int = 0
    
    # Chat state
    messages: List[Dict[str, Any]] = None
    is_processing: bool = False
    status_message: str = ""
    
    # UI state
    show_help: bool = False
    show_debug: bool = False
    
    def __post_init__(self):
        if self.messages is None:
            self.messages = []


class UIRenderer(ABC):
    """Base class for all UI renderers."""
    
    def __init__(self, capabilities):
        self.capabilities = capabilities
        self.state = UIState()
        self._active = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the renderer. Return True if successful."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up renderer resources."""
        pass
    
    @abstractmethod
    def render_frame(self) -> None:
        """Render a single frame with current state."""
        pass
    
    @abstractmethod
    def handle_input(self) -> Optional[str]:
        """Handle user input. Return command if ready, None otherwise."""
        pass
    
    @abstractmethod
    def show_message(self, message: str, level: str = "info") -> None:
        """Show a message to the user."""
        pass
    
    @abstractmethod
    def show_error(self, error: str) -> None:
        """Show an error message."""
        pass
    
    def update_state(self, **kwargs) -> None:
        """Update UI state with new values."""
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
    
    def is_active(self) -> bool:
        """Check if renderer is currently active."""
        return self._active
    
    def activate(self) -> bool:
        """Activate the renderer."""
        if self.initialize():
            self._active = True
            return True
        return False
    
    def deactivate(self) -> None:
        """Deactivate the renderer."""
        if self._active:
            self.cleanup()
            self._active = False


class ProgressiveRenderer:
    """Manager for progressive UI enhancement."""
    
    def __init__(self, capabilities):
        self.capabilities = capabilities
        self.renderers = {}
        self.current_renderer = None
    
    def register_renderer(self, name: str, renderer_class, priority: int = 0):
        """Register a renderer class with priority (higher = preferred)."""
        self.renderers[name] = {
            'class': renderer_class,
            'priority': priority,
            'instance': None
        }
    
    def select_best_renderer(self) -> Optional[UIRenderer]:
        """Select the best available renderer for current capabilities."""
        # Sort by priority (highest first)
        sorted_renderers = sorted(
            self.renderers.items(),
            key=lambda x: x[1]['priority'],
            reverse=True
        )
        
        for name, info in sorted_renderers:
            try:
                # Create renderer instance if not exists
                if info['instance'] is None:
                    info['instance'] = info['class'](self.capabilities)
                
                renderer = info['instance']
                
                # Test if renderer can activate
                if renderer.activate():
                    self.current_renderer = renderer
                    return renderer
                
            except Exception as e:
                # Log error and try next renderer
                print(f"Failed to activate {name} renderer: {e}")
                continue
        
        return None
    
    def get_current_renderer(self) -> Optional[UIRenderer]:
        """Get the currently active renderer."""
        return self.current_renderer
    
    def cleanup(self) -> None:
        """Clean up all renderers."""
        if self.current_renderer:
            self.current_renderer.deactivate()
            self.current_renderer = None
        
        for info in self.renderers.values():
            if info['instance']:
                try:
                    info['instance'].cleanup()
                except Exception:
                    pass  # Ignore cleanup errors