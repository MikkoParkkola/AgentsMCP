# TUI Multiline Paste Fix

## Problem Statement

The AgentsMCP TUI had a critical issue where multiline copy-paste operations were treated as multiple separate inputs instead of a single input. This caused:

1. **Fragmented Input**: Each line of pasted content was submitted as a separate message
2. **Poor User Experience**: Users couldn't paste code blocks or multiline text effectively  
3. **Loss of Context**: Multiline content lost its structure when split across multiple inputs

## Root Cause Analysis

The TUI input handlers processed characters one-by-one through `_handle_character_input()`, treating each Enter keypress as an immediate submission trigger. When users pasted multiline content:

1. Terminal sends characters rapidly in sequence
2. Each character is processed individually 
3. Enter characters (`\n`) trigger immediate input submission
4. Multiline content gets split at line boundaries

## Solution Implementation

### Paste Detection Algorithm

The fix implements intelligent paste detection using timing-based heuristics:

```python
# Paste detection thresholds
paste_threshold_ms = 50   # Characters arriving within 50ms suggest paste
paste_timeout_ms = 200    # Wait 200ms after last character before completion
```

### Key Components

#### 1. Timing-Based Detection
- Monitor time between character arrivals
- Characters arriving within 50ms threshold are considered paste candidates
- Maintains paste state until 200ms timeout without new characters

#### 2. Paste Buffer Management
- Characters during paste operations accumulate in `paste_buffer`
- Enter keypresses during paste are treated as content (not submission triggers)
- Buffer is inserted as complete block when paste operation completes

#### 3. Async Completion Handling
- Uses asyncio tasks to handle paste timeout detection
- Cancels and reschedules completion tasks as new characters arrive
- Ensures paste completes even if user stops typing

### Implementation Details

#### Chat Input Component (`src/agentsmcp/ui/v2/components/chat_input.py`)

Added paste state to `ChatInputState`:
```python
# Paste detection state
paste_buffer: str = ""
paste_start_time: float = 0
is_pasting: bool = False
```

Modified `_handle_character_input()` to route through paste detection:
```python
def _handle_character_input(self, event: InputEvent):
    current_time = time.time() * 1000
    if self._is_paste_event(current_time):
        self._handle_paste_character(char, current_time)
    else:
        self._handle_normal_character(char)
```

#### Fixed Working TUI (`src/agentsmcp/ui/v2/fixed_working_tui.py`)

Applied same paste detection logic to the direct terminal interface:
- Added timing-based paste detection
- Modified Enter key handling to respect paste state
- Implemented paste completion with proper terminal output

## Benefits

### ✅ Unified Input Handling
- Multiline paste now treated as single input block
- Maintains original formatting and structure
- No more fragmented submissions

### ✅ Backward Compatibility  
- Normal typing behavior unchanged
- No impact on existing workflows
- Graceful fallback for edge cases

### ✅ Cross-Platform Support
- Works across different terminal emulators
- No terminal-specific escape sequence dependencies
- Consistent behavior on macOS, Linux, Windows

### ✅ Performance Optimized
- Minimal overhead for normal typing
- Efficient async task management
- Proper cleanup and resource management

## Usage Example

**Before Fix:**
```
User pastes:
def hello():
    print("Hello")
    return True

Results in 3 separate inputs:
> def hello():
> print("Hello") 
> return True
```

**After Fix:**  
```
User pastes:
def hello():
    print("Hello")
    return True

Results in 1 complete input:
> def hello():
    print("Hello")
    return True
```

## Testing

Run the test script to verify paste detection:
```bash
python test_paste_fix.py
```

The test will:
1. Launch the fixed TUI
2. Provide multiline content for copy-paste testing
3. Verify paste is treated as single input

## Technical Configuration

### Timing Thresholds
- `paste_threshold_ms = 50`: Characters within 50ms are paste candidates
- `paste_timeout_ms = 200`: Wait 200ms for paste completion

### Adjustable Parameters
These can be tuned based on system performance and user preferences:
```python
# For slower systems, increase thresholds
self._paste_threshold_ms = 100
self._paste_timeout_ms = 300

# For faster/more responsive detection
self._paste_threshold_ms = 30  
self._paste_timeout_ms = 150
```

## Edge Cases Handled

1. **Mixed Typing and Paste**: Normal typing can interrupt paste operations
2. **Partial Pastes**: Incomplete paste operations are handled gracefully  
3. **Terminal Resizing**: Paste state is preserved during terminal events
4. **Keyboard Interrupts**: Proper cleanup of paste state on exit
5. **Empty Pastes**: Zero-length paste operations are ignored

## Future Improvements

1. **Bracketed Paste Mode**: Add support for `ESC[200~` / `ESC[201~` sequences
2. **Paste Size Limits**: Add configurable limits for large paste operations
3. **Visual Feedback**: Show paste operation status to users
4. **Configuration**: Allow users to adjust timing thresholds

## Files Modified

- `src/agentsmcp/ui/v2/components/chat_input.py` - Main chat input component
- `src/agentsmcp/ui/v2/fixed_working_tui.py` - Direct terminal TUI
- `test_paste_fix.py` - Test verification script
- `docs/TUI_PASTE_FIX.md` - This documentation