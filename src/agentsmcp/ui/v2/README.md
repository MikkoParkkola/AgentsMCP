# UI v2 Core Systems

This directory contains the v2 implementation of the core TUI (Terminal User Interface) systems for AgentsMCP. These systems are designed for **simplicity** and **reliability** over complex features.

## Architecture

The v2 systems consist of three core modules:

### 1. `input_handler.py`
**Robust keyboard input using prompt_toolkit**

- **Purpose**: Reliable key detection that works in real TTY environments
- **Key Features**:
  - Immediate character echo capability
  - Works where the current v1 system fails
  - No complex mode switching - single reliable approach
  - Proper handling of Ctrl+C, arrows, and special keys
- **Dependencies**: `prompt_toolkit` (with graceful fallback)

### 2. `terminal_manager.py`
**Terminal capability detection without Rich dependencies**

- **Purpose**: Clean TTY detection and terminal capability queries
- **Key Features**:
  - Actually works in real terminal environments
  - Terminal dimensions and capability detection
  - Safe fallback modes for non-interactive environments
  - No dependency on Rich or other heavy libraries
- **Dependencies**: Standard library only

### 3. `event_system.py`
**Simple async event handling**

- **Purpose**: Clean event propagation without blocking or deadlocks
- **Key Features**:
  - Keyboard events, resize events, application events
  - Clean event propagation with timeout prevention
  - Handler timeout prevention to avoid blocking
  - Simple async/await interface
- **Dependencies**: Standard library `asyncio`

## Design Principles

1. **Simplicity over Complexity**: Each system does one thing well
2. **Reliability over Features**: Works consistently across environments
3. **Graceful Degradation**: Falls back cleanly when capabilities aren't available
4. **No Blocking**: All operations are non-blocking or have timeouts
5. **Memory Safe**: Uses weak references to prevent memory leaks

## Critical Success Criteria

✅ **Input handler provides immediate key feedback** (no blind typing)  
✅ **Terminal manager works in actual terminal environments**  
✅ **Event system does not block or cause deadlocks**

## Usage Examples

### Basic Terminal Detection
```python
from agentsmcp.ui.v2.terminal_manager import TerminalManager

tm = TerminalManager()
caps = tm.detect_capabilities()

print(f"Interactive: {caps.interactive}")
print(f"Size: {caps.width}x{caps.height}")
print(f"Colors: {caps.colors}")
```

### Simple Event Handling
```python
from agentsmcp.ui.v2.event_system import AsyncEventSystem, Event, EventType

event_system = AsyncEventSystem()
await event_system.start()

# Emit an event
event = Event(event_type=EventType.APPLICATION, data={"message": "Hello"})
await event_system.emit_event(event)

await event_system.stop()
```

### Input with Immediate Echo
```python
from agentsmcp.ui.v2.input_handler import InputHandler

handler = InputHandler()
if handler.is_available():
    handler.set_echo(True)  # Enable immediate character echo
    # Set up key bindings...
```

## Integration

See `integration_example.py` for a complete example of how all three systems work together in a simple interactive application.

## Testing

Run the test script to verify all systems work correctly:

```bash
python test_v2_systems.py
```

This tests:
- Terminal capability detection
- Event system functionality  
- Input handler availability and basic operation

## Migration from v1

The v2 systems replace the problematic parts of the v1 UI implementation:

- **Replaces**: Complex TTY detection in `keyboard_input.py`
- **Replaces**: Rich-dependent terminal handling in `modern_tui.py`
- **Provides**: Clean event system to replace callback spaghetti

The v2 systems can be used alongside existing v1 code during migration.