"""
Tests for KeyboardProcessor - High-level keyboard shortcut processing.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from agentsmcp.ui.v2.keyboard_processor import (
    KeyboardProcessor, KeySequence, ShortcutBinding, ShortcutContext,
    SequenceState, create_simple_key, create_ctrl_key, create_alt_key,
    create_shift_key, create_key_sequence
)
from agentsmcp.ui.v2.input_handler import InputHandler, InputEvent, InputEventType
from agentsmcp.ui.v2.event_system import AsyncEventSystem, Event, EventType


@pytest.fixture
async def event_system():
    """Create event system for testing."""
    system = AsyncEventSystem()
    await system.start()
    yield system
    await system.stop()


@pytest.fixture
def input_handler():
    """Create mock input handler."""
    handler = Mock(spec=InputHandler)
    handler.is_available.return_value = True
    handler.add_key_handler = Mock()
    return handler


@pytest.fixture
def processor(input_handler, event_system):
    """Create keyboard processor for testing."""
    return KeyboardProcessor(input_handler, event_system)


class TestKeySequence:
    """Test KeySequence functionality."""
    
    def test_key_sequence_creation(self):
        """Test creating key sequences."""
        seq = KeySequence(['a'])
        assert seq.keys == ['a']
        assert len(seq.modifiers) == 0
        
        seq_with_mods = KeySequence(['c'], {'ctrl', 'shift'})
        assert seq_with_mods.keys == ['c']
        assert seq_with_mods.modifiers == {'ctrl', 'shift'}
    
    def test_key_sequence_string_representation(self):
        """Test string representation of key sequences."""
        seq = KeySequence(['a'])
        assert str(seq) == 'a'
        
        seq_with_mod = KeySequence(['c'], {'ctrl'})
        assert str(seq_with_mod) == 'ctrl+c'
        
        multi_mod = KeySequence(['x'], {'ctrl', 'shift'})
        # Order should be sorted
        assert 'ctrl' in str(multi_mod)
        assert 'shift' in str(multi_mod)
        assert str(multi_mod).endswith('+x')
        
        multi_key = KeySequence(['c-x', 'c-c'])
        assert str(multi_key) == 'c-x-c-c'
    
    def test_key_sequence_matching(self):
        """Test key sequence matching."""
        seq = KeySequence(['a'], {'ctrl'})
        
        assert seq.matches(['a'], {'ctrl'})
        assert not seq.matches(['a'], set())
        assert not seq.matches(['b'], {'ctrl'})
        assert not seq.matches(['a'], {'shift'})


class TestSequenceState:
    """Test SequenceState functionality."""
    
    def test_sequence_state_initialization(self):
        """Test sequence state initialization."""
        state = SequenceState()
        
        assert len(state.active_keys) == 0
        assert len(state.active_modifiers) == 0
        assert state.last_key_time is None
    
    def test_add_key(self):
        """Test adding keys to sequence state."""
        state = SequenceState()
        
        state.add_key('a', {'ctrl'})
        
        assert state.active_keys == ['a']
        assert state.active_modifiers == {'ctrl'}
        assert state.last_key_time is not None
    
    def test_sequence_expiration(self):
        """Test sequence expiration."""
        state = SequenceState()
        state.timeout = 0.1  # Very short timeout
        
        # Fresh state should not be expired
        assert not state.is_expired()
        
        # Add a key
        state.add_key('a')
        assert not state.is_expired()
        
        # Manually set old timestamp
        state.last_key_time = datetime.now() - timedelta(seconds=1)
        assert state.is_expired()
    
    def test_clear_sequence(self):
        """Test clearing sequence state."""
        state = SequenceState()
        
        state.add_key('a', {'ctrl'})
        assert len(state.active_keys) > 0
        
        state.clear()
        
        assert len(state.active_keys) == 0
        assert len(state.active_modifiers) == 0
        assert state.last_key_time is None


class TestKeyboardProcessor:
    """Test KeyboardProcessor functionality."""
    
    async def test_initialization(self, processor):
        """Test processor initialization."""
        assert not processor._initialized
        
        result = await processor.initialize()
        
        assert result is True
        assert processor._initialized
        assert len(processor._shortcuts) > 0  # Default shortcuts should be added
    
    async def test_add_shortcut(self, processor):
        """Test adding keyboard shortcuts."""
        await processor.initialize()
        
        # Mock handler
        handler = AsyncMock(return_value=True)
        
        # Add shortcut
        sequence = KeySequence(['f'], {'ctrl'})
        result = processor.add_shortcut(
            sequence,
            handler,
            ShortcutContext.GLOBAL,
            "Test shortcut"
        )
        
        assert result is True
        sequence_str = str(sequence)
        assert sequence_str in processor._shortcuts
        assert len(processor._shortcuts[sequence_str]) == 1
    
    async def test_remove_shortcut(self, processor):
        """Test removing keyboard shortcuts."""
        await processor.initialize()
        
        # Add and then remove shortcut
        handler = Mock()
        sequence = KeySequence(['f'], {'ctrl'})
        
        processor.add_shortcut(sequence, handler)
        assert str(sequence) in processor._shortcuts
        
        result = processor.remove_shortcut(sequence)
        
        assert result is True
        assert str(sequence) not in processor._shortcuts
    
    async def test_context_management(self, processor):
        """Test shortcut context management."""
        await processor.initialize()
        
        # Initial context should be global
        assert processor.get_current_context() == ShortcutContext.GLOBAL
        
        # Push new context
        processor.push_context(ShortcutContext.INPUT)
        assert processor.get_current_context() == ShortcutContext.INPUT
        
        # Pop context
        popped = processor.pop_context()
        assert popped == ShortcutContext.INPUT
        assert processor.get_current_context() == ShortcutContext.GLOBAL
        
        # Can't pop global context
        popped = processor.pop_context()
        assert popped is None
        assert processor.get_current_context() == ShortcutContext.GLOBAL
    
    async def test_text_input_mode(self, processor):
        """Test text input mode functionality."""
        await processor.initialize()
        
        assert not processor.is_text_input_mode()
        
        # Enter text input mode
        processor.enter_text_input_mode("test_component")
        
        assert processor.is_text_input_mode()
        assert processor._text_input_component == "test_component"
        assert processor.get_current_context() == ShortcutContext.INPUT
        
        # Exit text input mode
        processor.exit_text_input_mode()
        
        assert not processor.is_text_input_mode()
        assert processor._text_input_component is None
        assert processor.get_current_context() == ShortcutContext.GLOBAL
    
    async def test_simple_shortcut_processing(self, processor, event_system):
        """Test processing simple keyboard shortcuts."""
        await processor.initialize()
        
        # Add test shortcut
        handled_events = []
        
        async def test_handler(event):
            handled_events.append(event)
            return True
        
        sequence = KeySequence(['f'], {'ctrl'})
        processor.add_shortcut(sequence, test_handler, ShortcutContext.GLOBAL, "Test")
        
        # Create keyboard event
        event = Event(
            event_type=EventType.KEYBOARD,
            data={
                'key': 'f',
                'ctrl': True,
                'character': None
            }
        )
        
        # Process the event
        result = await processor._process_key_event(event)
        
        assert result is True
        assert len(handled_events) == 1
        assert processor._sequence_state.active_keys == []  # Should be cleared
    
    async def test_multi_key_sequence(self, processor):
        """Test multi-key sequence processing."""
        await processor.initialize()
        
        # Add multi-key shortcut (Ctrl+X Ctrl+C)
        handled_events = []
        
        async def test_handler(event):
            handled_events.append(event)
            return True
        
        sequence = KeySequence(['x', 'c'], {'ctrl'})
        processor.add_shortcut(sequence, test_handler, ShortcutContext.GLOBAL, "Multi-key test")
        
        # Send first key
        event1 = Event(
            event_type=EventType.KEYBOARD,
            data={
                'key': 'x',
                'ctrl': True
            }
        )
        
        result1 = await processor._process_key_event(event1)
        
        # Should be processing sequence but not complete yet
        assert result1 is True
        assert processor._processing_sequence is True
        assert len(handled_events) == 0
        
        # Send second key
        event2 = Event(
            event_type=EventType.KEYBOARD,
            data={
                'key': 'c',
                'ctrl': True
            }
        )
        
        result2 = await processor._process_key_event(event2)
        
        # Should complete the sequence
        assert result2 is True
        assert processor._processing_sequence is False
        assert len(handled_events) == 1
    
    async def test_sequence_timeout(self, processor):
        """Test sequence timeout handling."""
        await processor.initialize()
        
        # Add multi-key shortcut
        sequence = KeySequence(['a', 'b'])
        handler = AsyncMock()
        processor.add_shortcut(sequence, handler)
        
        # Start sequence
        event1 = Event(
            event_type=EventType.KEYBOARD,
            data={'key': 'a'}
        )
        
        await processor._process_key_event(event1)
        assert processor._processing_sequence is True
        
        # Manually expire the sequence
        processor._sequence_state.last_key_time = datetime.now() - timedelta(seconds=10)
        
        # Next key should clear expired sequence
        event2 = Event(
            event_type=EventType.KEYBOARD,
            data={'key': 'c'}  # Different key
        )
        
        await processor._process_key_event(event2)
        
        # Sequence should be cleared
        assert len(processor._sequence_state.active_keys) == 0
        assert processor._processing_sequence is False
        handler.assert_not_called()
    
    async def test_text_input_handling(self, processor, event_system):
        """Test text input mode event handling."""
        await processor.initialize()
        
        # Enter text input mode
        processor.enter_text_input_mode("test_input")
        
        # Mock the event system to capture emitted events
        emitted_events = []
        original_emit = event_system.emit_event
        
        async def mock_emit(event):
            emitted_events.append(event)
            return await original_emit(event)
        
        event_system.emit_event = mock_emit
        
        # Send regular character
        event = Event(
            event_type=EventType.KEYBOARD,
            data={
                'character': 'a',
                'key': None
            }
        )
        
        result = await processor._process_key_event(event)
        
        assert result is True
        # Should emit text input event
        assert len(emitted_events) == 1
        assert emitted_events[0].data['text_input_mode'] is True
        assert emitted_events[0].data['target_component'] == "test_input"
    
    async def test_shortcut_priority(self, processor):
        """Test shortcut priority handling."""
        await processor.initialize()
        
        # Add two shortcuts with same sequence but different priorities
        handled_events = []
        
        async def low_priority_handler(event):
            handled_events.append("low")
            return True
        
        async def high_priority_handler(event):
            handled_events.append("high")
            return True
        
        sequence = KeySequence(['f'], {'ctrl'})
        
        processor.add_shortcut(sequence, low_priority_handler, priority=1)
        processor.add_shortcut(sequence, high_priority_handler, priority=5)
        
        # Send event
        event = Event(
            event_type=EventType.KEYBOARD,
            data={
                'key': 'f',
                'ctrl': True
            }
        )
        
        await processor._process_key_event(event)
        
        # High priority handler should be called
        assert len(handled_events) == 1
        assert handled_events[0] == "high"
    
    async def test_context_based_shortcuts(self, processor):
        """Test context-based shortcut resolution."""
        await processor.initialize()
        
        # Ensure we're not in text input mode
        if processor.is_text_input_mode():
            processor.exit_text_input_mode()
        
        handled_events = []
        
        async def global_handler(event):
            handled_events.append("global")
            return True
        
        async def input_handler(event):
            handled_events.append("input")
            return True
        
        sequence = KeySequence(['f2'])
        
        # Add shortcuts for different contexts
        processor.add_shortcut(sequence, global_handler, ShortcutContext.GLOBAL)
        processor.add_shortcut(sequence, input_handler, ShortcutContext.INPUT)
        
        # Debug - check the shortcuts were added
        print(f"F2 shortcuts: {processor._shortcuts.get('f2', [])}")
        for binding in processor._shortcuts.get('f2', []):
            print(f"  Binding: context={binding.context}, enabled={binding.enabled}, priority={binding.priority}")
        
        # Test in global context
        event = Event(
            event_type=EventType.KEYBOARD,
            data={'key': 'f2'}
        )
        
        await processor._process_key_event(event)
        assert handled_events[-1] == "global"
        
        # Switch to input context
        processor.push_context(ShortcutContext.INPUT)
        
        # Debug
        print(f"Context stack: {processor._context_stack}")
        print(f"Current context: {processor.get_current_context()}")
        
        await processor._process_key_event(event)
        print(f"Handled events after context switch: {handled_events}")
        assert len(handled_events) == 2  # We should have both events now
        assert handled_events[-1] == "input"
    
    def test_shortcut_help_generation(self, processor):
        """Test shortcut help text generation."""
        # Add some shortcuts with descriptions
        seq1 = KeySequence(['f1'])
        seq2 = KeySequence(['h'], {'ctrl'})
        
        processor.add_shortcut(seq1, Mock(), description="Show help")
        processor.add_shortcut(seq2, Mock(), description="Open file")
        
        help_text = processor.get_shortcut_help()
        
        assert "Available shortcuts:" in help_text
        assert "Show help" in help_text
        assert "Open file" in help_text
    
    def test_should_pass_to_text_input(self, processor):
        """Test text input passthrough logic."""
        # Control keys should not pass to text input
        assert not processor._should_pass_to_text_input('f1', set())
        assert not processor._should_pass_to_text_input('tab', set())
        
        # Regular characters should pass
        assert processor._should_pass_to_text_input('a', set())
        assert processor._should_pass_to_text_input('1', set())
        
        # Common text editing Ctrl combinations should pass
        assert processor._should_pass_to_text_input('a', {'ctrl'})  # Select all
        assert processor._should_pass_to_text_input('c', {'ctrl'})  # Copy
        assert processor._should_pass_to_text_input('v', {'ctrl'})  # Paste
        
        # Other Ctrl combinations should not pass
        assert not processor._should_pass_to_text_input('f', {'ctrl'})
        assert not processor._should_pass_to_text_input('n', {'ctrl'})
    
    def test_get_shortcuts_for_context(self, processor):
        """Test getting shortcuts for specific context."""
        # Add shortcuts for different contexts
        global_seq = KeySequence(['f1'])
        input_seq = KeySequence(['enter'])
        
        processor.add_shortcut(global_seq, Mock(), ShortcutContext.GLOBAL, "Global help")
        processor.add_shortcut(input_seq, Mock(), ShortcutContext.INPUT, "Confirm input")
        
        # Get global context shortcuts
        global_shortcuts = processor.get_shortcuts_for_context(ShortcutContext.GLOBAL)
        assert len(global_shortcuts) >= 1
        
        # Get input context shortcuts (should include global ones too)
        input_shortcuts = processor.get_shortcuts_for_context(ShortcutContext.INPUT)
        assert len(input_shortcuts) >= 2  # Input-specific + global
    
    def test_stats_collection(self, processor):
        """Test statistics collection."""
        stats = processor.get_stats()
        
        assert 'shortcuts_processed' in stats
        assert 'shortcuts_registered' in stats
        assert 'current_context' in stats
        assert 'text_input_mode' in stats
        assert 'processing_sequence' in stats
    
    def test_debug_shortcuts(self, processor):
        """Test debug information generation."""
        # Add a test shortcut
        sequence = KeySequence(['f'], {'ctrl'})
        processor.add_shortcut(sequence, Mock(), description="Test shortcut")
        
        debug_info = processor.debug_shortcuts()
        
        assert "Registered shortcuts:" in debug_info
        assert "ctrl+f" in debug_info
        assert "Test shortcut" in debug_info
    
    async def test_cleanup(self, processor):
        """Test processor cleanup."""
        await processor.initialize()
        
        # Add some shortcuts
        processor.add_shortcut(KeySequence(['a']), Mock())
        assert len(processor._shortcuts) > 0
        
        # Cleanup
        await processor.cleanup()
        
        assert not processor._initialized
        assert len(processor._shortcuts) == 0
        assert len(processor._global_shortcuts) == 0


class TestDefaultShortcuts:
    """Test default keyboard shortcuts."""
    
    async def test_tab_navigation_shortcut(self, processor, event_system):
        """Test default Tab navigation shortcut."""
        await processor.initialize()
        
        # Mock event emission
        emitted_events = []
        original_emit = event_system.emit_event
        
        async def mock_emit(event):
            emitted_events.append(event)
            return True
        
        event_system.emit_event = mock_emit
        
        # Send Tab key
        event = Event(
            event_type=EventType.KEYBOARD,
            data={'key': 'tab'}
        )
        
        await processor._process_key_event(event)
        
        # Should emit focus next event
        assert len(emitted_events) == 1
        assert emitted_events[0].event_type == EventType.APPLICATION
        assert emitted_events[0].data['action'] == 'focus_next_component'
    
    async def test_help_shortcut(self, processor, event_system):
        """Test F1 help shortcut."""
        await processor.initialize()
        
        emitted_events = []
        
        async def mock_emit(event):
            emitted_events.append(event)
            return True
        
        event_system.emit_event = mock_emit
        
        # Send F1 key
        event = Event(
            event_type=EventType.KEYBOARD,
            data={'key': 'f1'}
        )
        
        await processor._process_key_event(event)
        
        # Should emit show help event
        assert len(emitted_events) == 1
        assert emitted_events[0].data['action'] == 'show_help'


class TestUtilityFunctions:
    """Test utility functions for creating key sequences."""
    
    def test_create_simple_key(self):
        """Test create_simple_key utility."""
        seq = create_simple_key('a')
        
        assert seq.keys == ['a']
        assert len(seq.modifiers) == 0
    
    def test_create_ctrl_key(self):
        """Test create_ctrl_key utility."""
        seq = create_ctrl_key('c')
        
        assert seq.keys == ['c']
        assert seq.modifiers == {'ctrl'}
    
    def test_create_alt_key(self):
        """Test create_alt_key utility."""
        seq = create_alt_key('f4')
        
        assert seq.keys == ['f4']
        assert seq.modifiers == {'alt'}
    
    def test_create_shift_key(self):
        """Test create_shift_key utility."""
        seq = create_shift_key('tab')
        
        assert seq.keys == ['tab']
        assert seq.modifiers == {'shift'}
    
    def test_create_key_sequence(self):
        """Test create_key_sequence utility."""
        seq = create_key_sequence(['c-x', 'c-c'], {'ctrl'})
        
        assert seq.keys == ['c-x', 'c-c']
        assert seq.modifiers == {'ctrl'}
        
        # Test without modifiers
        seq2 = create_key_sequence(['a', 'b'])
        
        assert seq2.keys == ['a', 'b']
        assert len(seq2.modifiers) == 0


@pytest.mark.asyncio
async def test_keyboard_processor_narrow_terminal():
    """Test keyboard processor behavior with narrow terminal."""
    event_system = AsyncEventSystem()
    await event_system.start()
    
    try:
        input_handler = Mock(spec=InputHandler)
        input_handler.is_available.return_value = True
        
        processor = KeyboardProcessor(input_handler, event_system)
        await processor.initialize()
        
        # Should work normally even with narrow terminal
        # (keyboard processing is not dependent on terminal width)
        assert processor._initialized
        assert len(processor._shortcuts) > 0
        
    finally:
        await event_system.stop()


@pytest.mark.asyncio
async def test_keyboard_processor_wide_terminal():
    """Test keyboard processor behavior with wide terminal."""
    event_system = AsyncEventSystem()
    await event_system.start()
    
    try:
        input_handler = Mock(spec=InputHandler)
        input_handler.is_available.return_value = True
        
        processor = KeyboardProcessor(input_handler, event_system)
        await processor.initialize()
        
        # Should work optimally with wide terminal
        assert processor._initialized
        assert len(processor._shortcuts) > 0
        
        # Can add more complex shortcuts for wide terminal usage
        complex_sequence = KeySequence(['c-x', 'c-c', 'c-v'])
        result = processor.add_shortcut(complex_sequence, Mock(), description="Complex shortcut")
        assert result is True
        
    finally:
        await event_system.stop()