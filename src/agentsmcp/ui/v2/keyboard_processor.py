"""
Keyboard Processor - High-level keyboard shortcut processing.

Converts raw key events to application actions, provides context-aware command
interpretation, and handles shortcut conflict resolution.
"""

import asyncio
import logging
from typing import Dict, Optional, List, Callable, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from datetime import datetime, timedelta

from .input_handler import InputHandler, InputEvent, InputEventType
from .event_system import AsyncEventSystem, Event, EventType

logger = logging.getLogger(__name__)


class ShortcutContext(Enum):
    """Context for keyboard shortcuts."""
    GLOBAL = "global"           # Available everywhere
    INPUT = "input"            # Text input contexts
    CHAT = "chat"              # Chat/conversation contexts
    MENU = "menu"              # Menu navigation contexts
    DIALOG = "dialog"          # Modal dialog contexts


@dataclass
class KeySequence:
    """Represents a key sequence for shortcuts."""
    keys: List[str]
    modifiers: Set[str] = field(default_factory=set)
    timeout: float = 2.0  # Maximum time between keys in sequence
    
    def __str__(self) -> str:
        """String representation of the key sequence."""
        mod_str = "+".join(sorted(self.modifiers)) 
        key_str = "-".join(self.keys)
        return f"{mod_str}+{key_str}" if mod_str else key_str
    
    def matches(self, keys: List[str], modifiers: Set[str]) -> bool:
        """Check if this sequence matches the given keys and modifiers."""
        return self.keys == keys and self.modifiers == modifiers


@dataclass
class ShortcutBinding:
    """Represents a keyboard shortcut binding."""
    sequence: KeySequence
    handler: Callable[[Event], Any]
    context: ShortcutContext = ShortcutContext.GLOBAL
    description: str = ""
    enabled: bool = True
    priority: int = 0  # Higher priority shortcuts take precedence
    created_at: datetime = field(default_factory=datetime.now)


class SequenceState:
    """Tracks the state of multi-key sequence input."""
    
    def __init__(self):
        self.active_keys: List[str] = []
        self.active_modifiers: Set[str] = set()
        self.last_key_time: Optional[datetime] = None
        self.timeout: float = 2.0
    
    def add_key(self, key: str, modifiers: Set[str] = None):
        """Add a key to the current sequence."""
        self.active_keys.append(key)
        if modifiers:
            self.active_modifiers.update(modifiers)
        self.last_key_time = datetime.now()
    
    def is_expired(self) -> bool:
        """Check if the sequence has expired."""
        if not self.last_key_time:
            return False
        return datetime.now() - self.last_key_time > timedelta(seconds=self.timeout)
    
    def clear(self):
        """Clear the sequence state."""
        self.active_keys.clear()
        self.active_modifiers.clear()
        self.last_key_time = None
    
    def get_sequence(self) -> KeySequence:
        """Get the current sequence."""
        return KeySequence(
            keys=self.active_keys.copy(),
            modifiers=self.active_modifiers.copy(),
            timeout=self.timeout
        )


class KeyboardProcessor:
    """
    High-level keyboard shortcut processing.
    
    Converts raw key events to application actions with context awareness
    and conflict resolution.
    """
    
    def __init__(self, 
                 input_handler: InputHandler,
                 event_system: AsyncEventSystem):
        self.input_handler = input_handler
        self.event_system = event_system
        
        # Shortcut management
        self._shortcuts: Dict[str, List[ShortcutBinding]] = defaultdict(list)
        self._global_shortcuts: List[ShortcutBinding] = []
        self._context_stack: List[ShortcutContext] = [ShortcutContext.GLOBAL]
        
        # Sequence processing
        self._sequence_state = SequenceState()
        self._processing_sequence = False
        
        # Text input handling
        self._text_input_mode = False
        self._text_input_component: Optional[str] = None
        self._bypass_shortcuts: Set[str] = set()
        
        # Conflict resolution
        self._conflict_resolution = "priority"  # "priority" or "context"
        
        # Statistics
        self._stats = {
            'shortcuts_processed': 0,
            'sequences_completed': 0,
            'sequences_timeout': 0,
            'conflicts_resolved': 0,
            'text_input_handled': 0
        }
        
        # State
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the keyboard processor."""
        if self._initialized:
            return True
        
        try:
            # Set up input handler callbacks
            if self.input_handler.is_available():
                self.input_handler.add_key_handler('*', self._handle_key_input)
            
            self._setup_default_shortcuts()
            self._initialized = True
            
            logger.info("Keyboard processor initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize keyboard processor: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup the keyboard processor."""
        try:
            # Remove all shortcuts
            self._shortcuts.clear()
            self._global_shortcuts.clear()
            
            # Clear state
            self._sequence_state.clear()
            self._context_stack = [ShortcutContext.GLOBAL]
            
            self._initialized = False
            logger.debug("Keyboard processor cleaned up")
            
        except Exception as e:
            logger.error(f"Error during keyboard processor cleanup: {e}")
    
    def _setup_default_shortcuts(self):
        """Setup default keyboard shortcuts."""
        # Navigation shortcuts
        self.add_shortcut(
            KeySequence(['tab']),
            self._handle_tab_navigation,
            ShortcutContext.GLOBAL,
            "Navigate to next component"
        )
        
        self.add_shortcut(
            KeySequence(['tab'], {'shift'}),
            self._handle_shift_tab_navigation,
            ShortcutContext.GLOBAL,
            "Navigate to previous component"
        )
        
        # Text input shortcuts
        self.add_shortcut(
            KeySequence(['enter']),
            self._handle_enter,
            ShortcutContext.INPUT,
            "Confirm text input"
        )
        
        self.add_shortcut(
            KeySequence(['escape']),
            self._handle_escape,
            ShortcutContext.INPUT,
            "Cancel text input"
        )
        
        # Global shortcuts
        self.add_shortcut(
            KeySequence(['f1']),
            self._handle_help,
            ShortcutContext.GLOBAL,
            "Show help"
        )
        
        # Multi-key sequence example
        self.add_shortcut(
            KeySequence(['x', 'x'], {'ctrl'}),
            self._handle_copy_sequence,
            ShortcutContext.GLOBAL,
            "Copy selection"
        )
    
    def add_shortcut(self,
                    sequence: KeySequence,
                    handler: Callable[[Event], Any],
                    context: ShortcutContext = ShortcutContext.GLOBAL,
                    description: str = "",
                    priority: int = 0) -> bool:
        """
        Add a keyboard shortcut.
        
        Args:
            sequence: Key sequence for the shortcut
            handler: Function to call when shortcut is triggered
            context: Context where shortcut is active
            description: Human-readable description
            priority: Priority level (higher = more important)
            
        Returns:
            True if shortcut was added successfully
        """
        try:
            binding = ShortcutBinding(
                sequence=sequence,
                handler=handler,
                context=context,
                description=description,
                priority=priority
            )
            
            # Add to appropriate collection
            sequence_str = str(sequence)
            if context == ShortcutContext.GLOBAL:
                self._global_shortcuts.append(binding)
            
            self._shortcuts[sequence_str].append(binding)
            
            # Sort by priority (highest first)
            self._shortcuts[sequence_str].sort(key=lambda b: b.priority, reverse=True)
            
            logger.debug(f"Added shortcut: {sequence_str} -> {description}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding shortcut: {e}")
            return False
    
    def remove_shortcut(self, sequence: KeySequence, context: ShortcutContext = None) -> bool:
        """Remove a keyboard shortcut."""
        sequence_str = str(sequence)
        
        if sequence_str not in self._shortcuts:
            return False
        
        bindings = self._shortcuts[sequence_str]
        
        if context is None:
            # Remove all bindings for this sequence
            del self._shortcuts[sequence_str]
            # Also remove from global shortcuts
            self._global_shortcuts = [b for b in self._global_shortcuts 
                                     if str(b.sequence) != sequence_str]
        else:
            # Remove only bindings with matching context
            self._shortcuts[sequence_str] = [b for b in bindings if b.context != context]
            if not self._shortcuts[sequence_str]:
                del self._shortcuts[sequence_str]
        
        logger.debug(f"Removed shortcut: {sequence_str}")
        return True
    
    def push_context(self, context: ShortcutContext):
        """Push a new shortcut context onto the stack."""
        self._context_stack.append(context)
        logger.debug(f"Pushed context: {context.value}")
    
    def pop_context(self) -> Optional[ShortcutContext]:
        """Pop the current context from the stack."""
        if len(self._context_stack) > 1:  # Always keep global context
            context = self._context_stack.pop()
            logger.debug(f"Popped context: {context.value}")
            return context
        return None
    
    def get_current_context(self) -> ShortcutContext:
        """Get the current shortcut context."""
        return self._context_stack[-1]
    
    def enter_text_input_mode(self, component_name: Optional[str] = None):
        """Enter text input mode."""
        self._text_input_mode = True
        self._text_input_component = component_name
        self.push_context(ShortcutContext.INPUT)
        logger.debug(f"Entered text input mode for component: {component_name}")
    
    def exit_text_input_mode(self):
        """Exit text input mode."""
        self._text_input_mode = False
        self._text_input_component = None
        if self.get_current_context() == ShortcutContext.INPUT:
            self.pop_context()
        logger.debug("Exited text input mode")
    
    async def _handle_key_input(self, input_event: InputEvent):
        """Handle raw key input from the input handler."""
        try:
            # Convert input event to our event format
            event = Event(
                event_type=EventType.KEYBOARD,
                data={
                    'key': input_event.key,
                    'character': input_event.character,
                    'ctrl': input_event.ctrl,
                    'alt': input_event.alt,
                    'shift': input_event.shift,
                    'event_type': input_event.event_type.value,
                    'input_event': input_event
                }
            )
            
            # Process the key event
            await self._process_key_event(event)
            
        except Exception as e:
            logger.error(f"Error handling key input: {e}")
    
    async def _process_key_event(self, event: Event) -> bool:
        """Process a keyboard event."""
        try:
            # Check if sequence has expired
            if self._sequence_state.is_expired():
                self._sequence_state.clear()
                self._processing_sequence = False
            
            # Extract key information
            key = event.data.get('key')
            character = event.data.get('character') 
            
            # Determine the effective key
            effective_key = key or character
            if not effective_key:
                return False
            
            # Get modifiers
            modifiers = set()
            if event.data.get('ctrl'):
                modifiers.add('ctrl')
            if event.data.get('alt'):
                modifiers.add('alt')
            if event.data.get('shift'):
                modifiers.add('shift')
            
            # Handle text input mode specially
            if self._text_input_mode and self._should_pass_to_text_input(effective_key, modifiers):
                await self._handle_text_input(event)
                return True
            
            # Add to sequence state
            self._sequence_state.add_key(effective_key, modifiers)
            current_sequence = self._sequence_state.get_sequence()
            
            # Try to find matching shortcut
            binding = self._find_matching_shortcut(current_sequence)
            
            if binding:
                # Execute the shortcut
                await self._execute_shortcut(binding, event)
                self._sequence_state.clear()
                self._processing_sequence = False
                self._stats['shortcuts_processed'] += 1
                return True
            
            # Check if this could be part of a longer sequence
            if self._could_be_sequence_start(current_sequence):
                self._processing_sequence = True
                return True
            
            # No shortcut found, clear sequence and pass through
            self._sequence_state.clear()
            self._processing_sequence = False
            
            # In text input mode, pass unhandled keys to text input
            if self._text_input_mode:
                await self._handle_text_input(event)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error processing key event: {e}")
            return False
    
    def _should_pass_to_text_input(self, key: str, modifiers: Set[str]) -> bool:
        """Determine if key should be passed to text input."""
        # Always handle certain control keys as shortcuts
        control_keys = {'tab', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12'}
        
        if key in control_keys:
            return False
        
        # Handle Ctrl combinations as shortcuts (except a few exceptions)
        if 'ctrl' in modifiers:
            text_input_ctrl_keys = {'a', 'x', 'c', 'v', 'z', 'y'}  # Common text editing
            return key in text_input_ctrl_keys
        
        # Regular characters and basic editing keys go to text input
        return True
    
    def _find_matching_shortcut(self, sequence: KeySequence) -> Optional[ShortcutBinding]:
        """Find a shortcut binding that matches the sequence."""
        sequence_str = str(sequence)
        
        if sequence_str not in self._shortcuts:
            return None
        
        bindings = self._shortcuts[sequence_str]
        current_context = self.get_current_context()
        
        # Find best matching binding based on context and priority
        best_binding = None
        best_is_exact_context = False
        
        for binding in bindings:
            if not binding.enabled:
                continue
            
            is_exact_context = binding.context == current_context
            is_global = binding.context == ShortcutContext.GLOBAL
            
            # Decision logic:
            # 1. Exact context match always beats global (regardless of priority)
            # 2. Among exact context matches, highest priority wins  
            # 3. Among global matches, highest priority wins
            # 4. Global only wins if no exact context match exists
            
            should_replace = False
            
            if not best_binding:
                # No existing binding - take this one
                should_replace = True
            elif is_exact_context and not best_is_exact_context:
                # Exact context beats global, regardless of priority
                should_replace = True  
            elif is_exact_context and best_is_exact_context:
                # Both exact context - compare priority
                should_replace = binding.priority > best_binding.priority
            elif is_global and not best_is_exact_context:
                # Both global - compare priority
                should_replace = binding.priority > best_binding.priority
            # If current is global and best is exact context, never replace
            
            if should_replace:
                best_binding = binding
                best_is_exact_context = is_exact_context
        
        if best_binding and len(bindings) > 1:
            self._stats['conflicts_resolved'] += 1
        
        return best_binding
    
    def _could_be_sequence_start(self, current_sequence: KeySequence) -> bool:
        """Check if current sequence could be the start of a longer sequence."""
        current_keys = current_sequence.keys
        
        for sequence_str, bindings in self._shortcuts.items():
            for binding in bindings:
                if not binding.enabled:
                    continue
                
                # Ensure binding.sequence is a KeySequence object
                if not hasattr(binding.sequence, 'keys'):
                    logger.warning(f"Skipping invalid binding sequence: {binding.sequence}")
                    continue
                
                binding_keys = binding.sequence.keys
                
                # Check if current sequence is a prefix of this binding
                if (len(current_keys) < len(binding_keys) and 
                    binding_keys[:len(current_keys)] == current_keys):
                    return True
        
        return False
    
    async def _execute_shortcut(self, binding: ShortcutBinding, event: Event):
        """Execute a shortcut binding."""
        try:
            # Add binding info to event data
            event.data['shortcut_binding'] = binding
            event.data['shortcut_description'] = binding.description
            
            # Execute the handler
            if asyncio.iscoroutinefunction(binding.handler):
                await binding.handler(event)
            else:
                binding.handler(event)
            
            logger.debug(f"Executed shortcut: {binding.description}")
            
        except Exception as e:
            logger.error(f"Error executing shortcut {binding.description}: {e}")
    
    async def _handle_text_input(self, event: Event):
        """Handle text input in input mode."""
        # Emit event for text input components
        text_event = Event(
            event_type=EventType.KEYBOARD,
            data={
                **event.data,
                'text_input_mode': True,
                'target_component': self._text_input_component
            }
        )
        
        await self.event_system.emit_event(text_event)
        self._stats['text_input_handled'] += 1
    
    # Default shortcut handlers
    async def _handle_tab_navigation(self, event: Event):
        """Handle Tab key for component navigation."""
        nav_event = Event(
            event_type=EventType.APPLICATION,
            data={'action': 'focus_next_component'}
        )
        await self.event_system.emit_event(nav_event)
    
    async def _handle_shift_tab_navigation(self, event: Event):
        """Handle Shift+Tab for reverse navigation."""
        nav_event = Event(
            event_type=EventType.APPLICATION,
            data={'action': 'focus_previous_component'}
        )
        await self.event_system.emit_event(nav_event)
    
    async def _handle_enter(self, event: Event):
        """Handle Enter key in input context."""
        if self._text_input_mode:
            confirm_event = Event(
                event_type=EventType.APPLICATION,
                data={
                    'action': 'confirm_input',
                    'component': self._text_input_component
                }
            )
            await self.event_system.emit_event(confirm_event)
    
    async def _handle_escape(self, event: Event):
        """Handle Escape key."""
        if self._text_input_mode:
            # Exit text input mode
            self.exit_text_input_mode()
            
            cancel_event = Event(
                event_type=EventType.APPLICATION,
                data={
                    'action': 'cancel_input',
                    'component': self._text_input_component
                }
            )
            await self.event_system.emit_event(cancel_event)
        else:
            # General escape action
            escape_event = Event(
                event_type=EventType.APPLICATION,
                data={'action': 'escape_pressed'}
            )
            await self.event_system.emit_event(escape_event)
    
    async def _handle_help(self, event: Event):
        """Handle F1 help key."""
        help_event = Event(
            event_type=EventType.APPLICATION,
            data={'action': 'show_help'}
        )
        await self.event_system.emit_event(help_event)
    
    async def _handle_copy_sequence(self, event: Event):
        """Handle Ctrl+X Ctrl+C copy sequence."""
        copy_event = Event(
            event_type=EventType.APPLICATION,
            data={'action': 'copy_selection'}
        )
        await self.event_system.emit_event(copy_event)
        self._stats['sequences_completed'] += 1
    
    def get_shortcuts_for_context(self, context: ShortcutContext = None) -> List[ShortcutBinding]:
        """Get all shortcuts for a specific context."""
        if context is None:
            context = self.get_current_context()
        
        shortcuts = []
        for bindings in self._shortcuts.values():
            for binding in bindings:
                if binding.context == context or binding.context == ShortcutContext.GLOBAL:
                    shortcuts.append(binding)
        
        return sorted(shortcuts, key=lambda b: b.priority, reverse=True)
    
    def get_shortcut_help(self, context: ShortcutContext = None) -> str:
        """Get help text for shortcuts in the current context."""
        shortcuts = self.get_shortcuts_for_context(context)
        
        if not shortcuts:
            return "No shortcuts available in current context."
        
        lines = ["Available shortcuts:"]
        for binding in shortcuts:
            if binding.description:
                lines.append(f"  {binding.sequence}: {binding.description}")
        
        return "\n".join(lines)
    
    def is_text_input_mode(self) -> bool:
        """Check if we're in text input mode."""
        return self._text_input_mode
    
    def get_sequence_state(self) -> Dict[str, Any]:
        """Get current sequence state information."""
        return {
            'active_keys': self._sequence_state.active_keys.copy(),
            'active_modifiers': list(self._sequence_state.active_modifiers),
            'processing_sequence': self._processing_sequence,
            'last_key_time': self._sequence_state.last_key_time,
            'expired': self._sequence_state.is_expired()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get keyboard processor statistics."""
        return {
            **self._stats,
            'shortcuts_registered': sum(len(bindings) for bindings in self._shortcuts.values()),
            'global_shortcuts': len(self._global_shortcuts),
            'current_context': self.get_current_context().value,
            'context_stack_depth': len(self._context_stack),
            'text_input_mode': self._text_input_mode,
            'processing_sequence': self._processing_sequence
        }
    
    def debug_shortcuts(self) -> str:
        """Get debug information about registered shortcuts."""
        lines = ["Registered shortcuts:"]
        
        for sequence_str, bindings in sorted(self._shortcuts.items()):
            lines.append(f"  {sequence_str}:")
            for binding in bindings:
                lines.append(f"    Context: {binding.context.value}, "
                           f"Priority: {binding.priority}, "
                           f"Enabled: {binding.enabled}")
                if binding.description:
                    lines.append(f"    Description: {binding.description}")
        
        return "\n".join(lines)


# Utility functions for creating common key sequences
def create_simple_key(key: str) -> KeySequence:
    """Create a simple single-key sequence."""
    return KeySequence([key])

def create_ctrl_key(key: str) -> KeySequence:
    """Create a Ctrl+key sequence."""
    return KeySequence([key], {'ctrl'})

def create_alt_key(key: str) -> KeySequence:
    """Create an Alt+key sequence."""
    return KeySequence([key], {'alt'})

def create_shift_key(key: str) -> KeySequence:
    """Create a Shift+key sequence."""
    return KeySequence([key], {'shift'})

def create_key_sequence(keys: List[str], modifiers: Set[str] = None) -> KeySequence:
    """Create a multi-key sequence."""
    return KeySequence(keys, modifiers or set())