# TUI Issues Report

This document records the functional and usability issues discovered during manual exploratory testing of the AgentsMCP terminal user interface (TUI).  Each entry includes a brief description, reproducible steps, severity, and optional visual aids.

---

## 1. Mouse‑wheel events not consumed
- **Description**: The `KeyboardInput` class correctly parses SGR mouse‑wheel sequences and returns `MouseEvent.SCROLL_UP` / `SCROLL_DOWN`, but no UI component processes these events. Consequently, scrolling the mouse wheel has no effect on any visible pane.
- **Steps to reproduce**:
  1. Launch the TUI (`python -m agentsmcp.ui.modern_tui`).
  2. Navigate to a scrollable view (e.g., the dashboard or a long command‑history list).
  3. Use the mouse wheel.
  4. Observe that the content does not move.
- **Severity**: **Major** – Users expect mouse scrolling to work in a modern TUI; the lack reduces usability noticeably.
- **Visual aid**: *N/A (no UI change to capture)*

---

## 2. Terminal resize does not trigger UI refresh
- **Description**: Resizing the terminal window does not cause the TUI to recompute its layout. Panels may become truncated or overflow, leading to unreadable output.
- **Steps to reproduce**:
  1. Start the TUI.
  2. Resize the terminal (drag the window border or use `stty cols/rows`).
  3. Notice that the UI retains the old width/height.
- **Severity**: **Major** – A responsive layout is essential for a good user experience, especially on varying terminal sizes.
- **Visual aid**: *N/A*

---

## 3. Backspace edge‑case with Unicode graphemes
- **Description**: The line buffer stores characters as individual Python `str` elements. Deleting a Unicode grapheme that consists of multiple code points (e.g., emojis or accented characters) removes only the last code point, leaving a broken visual glyph.
- **Steps to reproduce**:
  1. Launch the TUI.
  2. Type an emoji, e.g., `😀` (or a composed character like `é`).
  3. Press Backspace.
  4. Observe that the glyph is partially removed, showing a stray accent or box.
- **Severity**: **Minor** – Affects only users who type complex Unicode characters, but still a correctness issue.
- **Visual aid**: *N/A*

---

## 4. No SIGWINCH handler for graceful resize
- **Description**: The TUI does not register a signal handler for `SIGWINCH`. Even if a resize handler were added, the lack of the signal handler prevents any automatic response.
- **Steps to reproduce**: Same as Issue #2; the underlying cause is the missing signal registration.
- **Severity**: **Major** (same impact as Issue #2).
- **Visual aid**: *N/A*

---

## 5. Incomplete mouse‑click handling
- **Description**: `MouseEvent` enum defines `CLICK_LEFT` and `CLICK_RIGHT`, but there is no logic that maps clicks to UI actions (e.g., selecting a menu entry).
- **Steps to reproduce**:
  1. Launch the TUI.
  2. Click inside a scrollable pane.
  3. No observable effect occurs.
- **Severity**: **Minor** – Feature not yet implemented; does not break core functionality.
- **Visual aid**: *N/A*

---

## 6. Inconsistent colour usage
- **Description**: Some UI components embed raw ANSI escape codes while others use `ThemeManager`. This can cause colour mismatches when switching themes.
- **Steps to reproduce**:
  1. Run the TUI with the default theme.
  2. Switch to a different theme via `/theme dark`.
  3. Observe that a few elements retain the previous colour scheme.
- **Severity**: **Minor** – Cosmetic but reduces visual consistency.
- **Visual aid**: *N/A*

---

## 7. Potential terminal raw‑mode leak on forced termination
- **Description**: If the process receives `SIGTERM` or crashes, `KeyboardInput.close()` may not execute, leaving the terminal in raw mode.
- **Steps to reproduce**:
  1. Start the TUI.
  2. From another shell, send `kill -TERM <pid>`.
  3. Return to the original terminal; input behaves erratically (no echo, no line buffering).
- **Severity**: **Critical** – Leaves the user’s terminal unusable until a reset.
- **Visual aid**: *N/A*

---

## 8. History navigation limited to `readline`
- **Description**: Command history works only while the prompt is a single line. Multi‑line inputs (e.g., `/edit` or pasted blocks) do not support arrow‑up navigation.
- **Steps to reproduce**:
  1. Execute `/edit` to open the multi‑line editor.
  2. Press the up‑arrow key.
  3. No history navigation occurs.
- **Severity**: **Minor** – A usability limitation for power users.
- **Visual aid**: *N/A*

---

*End of report.*
