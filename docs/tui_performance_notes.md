# TUI Performance Optimizations

## Overview
The Modern TUI has been optimized to reduce excessive refresh cycles and improve terminal performance.

## Key Optimizations Implemented

### 1. Rich Live Refresh Rate
- **Before**: Default ~20-30 FPS (excessive for terminal UI)
- **After**: 3 FPS (responsive but efficient)
- **Code**: `refresh_per_second=3` parameter in Live context manager

### 2. Smart Debouncing System
Section-specific debounce timeouts to prevent excessive refreshes:

| Section | Debounce Time | Reason |
|---------|---------------|---------|
| Footer (input) | 0.1s | Real-time typing feedback |
| Content (chat) | 0.15s | Moderate frequency for conversation flow |
| Header (status) | 0.3s | Status info can update less frequently |
| Sidebar (navigation) | 0.5s | Rarely changes, can be slow |

### 3. Enhanced Content Change Detection
- Normalizes whitespace in content comparison
- Only triggers refresh when content actually changes
- Prevents identical frame renders

### 4. Typing Activity Detection
- Tracks user typing activity with 2-second timeout
- Uses shorter debounce (0.05s) during active typing
- Uses longer debounce (0.15s) during idle periods

### 5. Meaningful Status Change Detection
- Only refreshes header for significant changes:
  - Cost changes > 1 cent
  - Agent count changes
  - Model or connection status changes
- Ignores minor fluctuations that don't affect UX

### 6. Navigation Key Optimization
- Arrow keys and navigation don't trigger constant refreshes
- Only content-changing keys trigger visual updates

## Performance Impact

### Before Optimizations:
- High refresh rate (~20-30 FPS) even when idle
- Excessive terminal scrollback with repeated identical frames
- Frequent refreshes on every keystroke
- Status updates triggered refreshes for minor changes

### After Optimizations:
- Efficient 3 FPS refresh rate
- Clean terminal scrollback with minimal redundant renders
- Smart refresh only when content actually changes
- Adaptive refresh rates based on user activity

## Usage
The optimizations are transparent to users. The TUI maintains its responsive feel while being much more efficient with system resources and terminal rendering.

## Debugging
If you need to monitor refresh behavior:
1. Check `_last_dirty_time` dict for section-specific debounce tracking
2. Monitor `_refresh_event` for actual refresh triggers
3. Use `_is_user_typing()` to check typing detection
4. Inspect `_status_has_meaningful_change()` for status update logic