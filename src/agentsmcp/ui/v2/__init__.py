"""
Core TUI systems v2 - Simple, reliable terminal interface components.

This package provides the foundational systems for robust terminal user interface:
- terminal_manager: Terminal capability detection without Rich dependencies
- event_system: Simple async event handling
- input_handler: Reliable keyboard input using prompt_toolkit
- display_renderer: Clean terminal output without scrollback pollution
- layout_engine: Simple layout system for TUI components
- themes: Color and styling system with accessibility support
- chat_interface: Complete chat interface with real-time input and history
- application_controller: Main application coordination and lifecycle management

Designed for simplicity and reliability over complex features.
"""

__version__ = "2.0.0"

# Import core components
from .terminal_manager import TerminalManager, TerminalCapabilities, TerminalType
from .event_system import AsyncEventSystem, Event, EventType, EventHandler, KeyboardEventHandler
from .input_handler import InputHandler, InputEvent, InputEventType
from .display_renderer import DisplayRenderer, RenderMode, RenderRegion
from .layout_engine import LayoutEngine, LayoutNode, ContainerNode, TextNode, Size, Padding, Rectangle
from .themes import ThemeManager, ColorMode, BorderStyle, ColorScheme
from .keyboard_processor import KeyboardProcessor, KeySequence, ShortcutContext
from .application_controller import ApplicationController, ApplicationState, ApplicationConfig

# Chat interface components
from .chat_interface import ChatInterface, ChatInterfaceConfig, create_chat_interface
from .components import ChatInput, ChatHistory, ChatMessage, MessageRole, create_chat_input, create_chat_history

# Main application
from .main_app import TUILauncher, launch_main_tui

# Convenience functions
from .terminal_manager import create_terminal_manager, get_terminal_manager
from .event_system import create_event_system, get_event_system
from .layout_engine import create_standard_tui_layout
from .themes import create_theme_manager, detect_preferred_scheme

__all__ = [
    # Core classes
    'TerminalManager', 'TerminalCapabilities', 'TerminalType',
    'AsyncEventSystem', 'Event', 'EventType', 'EventHandler', 'KeyboardEventHandler',
    'InputHandler', 'InputEvent', 'InputEventType',
    'DisplayRenderer', 'RenderMode', 'RenderRegion',
    'LayoutEngine', 'LayoutNode', 'ContainerNode', 'TextNode', 'Size', 'Padding', 'Rectangle',
    'ThemeManager', 'ColorMode', 'BorderStyle', 'ColorScheme',
    'KeyboardProcessor', 'KeySequence', 'ShortcutContext',
    'ApplicationController', 'ApplicationState', 'ApplicationConfig',
    
    # Chat interface components
    'ChatInterface', 'ChatInterfaceConfig', 'create_chat_interface',
    'ChatInput', 'ChatHistory', 'ChatMessage', 'MessageRole',
    'create_chat_input', 'create_chat_history',
    
    # Main application
    'TUILauncher', 'launch_main_tui',
    
    # Convenience functions
    'create_terminal_manager', 'get_terminal_manager',
    'create_event_system', 'get_event_system',
    'create_standard_tui_layout',
    'create_theme_manager', 'detect_preferred_scheme'
]