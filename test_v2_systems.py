#!/usr/bin/env python3
"""
Test script for the v2 TUI core systems.

Tests the input handler, terminal manager, and event system
to verify they work correctly in real terminal environments.
"""

import asyncio
import sys
import os

# Add src to path so we can import agentsmcp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agentsmcp.ui.v2.input_handler import InputHandler, InputEvent, InputEventType
from agentsmcp.ui.v2.terminal_manager import TerminalManager
from agentsmcp.ui.v2.event_system import AsyncEventSystem, Event, EventType, KeyboardEventHandler


async def test_terminal_manager():
    """Test terminal manager functionality."""
    print("=== Testing Terminal Manager ===")
    
    tm = TerminalManager()
    caps = tm.detect_capabilities()
    
    print(f"Terminal Type: {caps.type.value}")
    print(f"Interactive: {caps.interactive}")
    print(f"Size: {caps.width}x{caps.height}")
    print(f"Colors: {caps.colors}")
    print(f"Unicode: {caps.unicode_support}")
    
    # Print full info
    print("\n--- Full Terminal Info ---")
    tm.print_terminal_info()
    print()


async def test_event_system():
    """Test event system functionality."""
    print("=== Testing Event System ===")
    
    event_system = AsyncEventSystem()
    
    # Create a simple event handler
    class TestHandler:
        def __init__(self):
            self.events_received = []
            
        async def handle_event(self, event: Event) -> bool:
            self.events_received.append(event)
            print(f"Received event: {event.event_type.value} with data: {event.data}")
            return True
            
        def can_handle(self, event: Event) -> bool:
            return True
    
    handler = TestHandler()
    
    # Add handler
    from agentsmcp.ui.v2.event_system import EventHandler
    class WrappedHandler(EventHandler):
        def __init__(self, inner):
            super().__init__("TestHandler")
            self.inner = inner
            
        async def handle_event(self, event: Event) -> bool:
            return await self.inner.handle_event(event)
            
        def can_handle(self, event: Event) -> bool:
            return self.inner.can_handle(event)
    
    wrapped_handler = WrappedHandler(handler)
    event_system.add_handler(EventType.APPLICATION, wrapped_handler)
    
    # Start event system
    await event_system.start()
    
    # Emit test events
    test_event = Event(
        event_type=EventType.APPLICATION,
        data={"test": "data", "message": "Hello from event system!"}
    )
    
    await event_system.emit_event(test_event)
    
    # Give it a moment to process
    await asyncio.sleep(0.1)
    
    # Check stats
    stats = event_system.get_stats()
    print(f"Event system stats: {stats}")
    
    # Stop event system
    await event_system.stop()
    
    print(f"Handler received {len(handler.events_received)} events")
    print()


async def test_input_handler_basic():
    """Test basic input handler functionality."""
    print("=== Testing Input Handler (Basic) ===")
    
    input_handler = InputHandler()
    
    print(f"Input handler available: {input_handler.is_available()}")
    print(f"Capabilities: {input_handler.get_capabilities()}")
    
    if input_handler.is_available():
        print("Input handler is available and ready!")
    else:
        print("Input handler not available (probably missing prompt_toolkit)")
    
    print()


async def test_input_handler_interactive():
    """Test interactive input handler functionality."""
    print("=== Testing Input Handler (Interactive) ===")
    
    input_handler = InputHandler()
    
    if not input_handler.is_available():
        print("Input handler not available - skipping interactive test")
        return
    
    # Test single key functionality (if available)
    print("Testing single key input (non-blocking)...")
    key = input_handler.get_single_key(timeout=1.0)
    if key:
        print(f"Got key: {repr(key)}")
    else:
        print("No key pressed (timeout)")
    
    print()


async def main():
    """Main test function."""
    print("Testing AgentsMCP UI v2 Core Systems")
    print("=" * 40)
    
    # Test all systems
    await test_terminal_manager()
    await test_event_system()
    await test_input_handler_basic()
    await test_input_handler_interactive()
    
    print("All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())