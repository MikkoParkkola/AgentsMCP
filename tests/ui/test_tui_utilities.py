"""
Utility classes and functions for TUI testing.

Provides mock terminals, console capture mechanisms, and analysis tools
for comprehensive TUI testing.
"""

import io
import sys
import time
import threading
from typing import List, Dict, Tuple, Any, Optional
from contextlib import contextmanager
from dataclasses import dataclass
from unittest.mock import Mock, patch


@dataclass
class TerminalState:
    """Represents terminal state at a point in time."""
    cursor_row: int
    cursor_col: int
    screen_width: int
    screen_height: int
    content: List[str]
    timestamp: float


class TerminalSimulator:
    """
    Advanced terminal simulator for testing cursor positioning and screen management.
    
    This class simulates a real terminal's behavior including:
    - Cursor positioning with row/column tracking
    - ANSI escape sequence processing
    - Screen buffering and scrolling
    - Character insertion and deletion
    """
    
    def __init__(self, width: int = 80, height: int = 24):
        self.width = width
        self.height = height
        self.cursor_row = 0
        self.cursor_col = 0
        self.screen_buffer = [' ' * width for _ in range(height)]
        self.history = []
        self.in_escape_sequence = False
        self.escape_buffer = ""
        
    def write(self, text: str) -> None:
        """Process text input and update terminal state."""
        for char in text:
            self._process_char(char)
    
    def _process_char(self, char: str) -> None:
        """Process a single character."""
        if self.in_escape_sequence:
            self._process_escape_char(char)
        elif char == '\033':  # ESC
            self.in_escape_sequence = True
            self.escape_buffer = '\033'
        elif char == '\r':
            self.cursor_col = 0
        elif char == '\n':
            self._newline()
        elif char == '\b':
            self._backspace()
        elif char == '\t':
            self._tab()
        elif ord(char) >= 32:  # Printable character
            self._insert_char(char)
    
    def _process_escape_char(self, char: str) -> None:
        """Process character within ANSI escape sequence."""
        self.escape_buffer += char
        
        # Check for sequence termination
        if char.isalpha() or char in ['~', '@']:
            self._execute_escape_sequence()
            self.in_escape_sequence = False
            self.escape_buffer = ""
    
    def _execute_escape_sequence(self) -> None:
        """Execute completed ANSI escape sequence."""
        seq = self.escape_buffer
        
        if seq == '\033[2J':  # Clear screen
            self.screen_buffer = [' ' * self.width for _ in range(self.height)]
        elif seq == '\033[H':  # Move to home position
            self.cursor_row = 0
            self.cursor_col = 0
        elif seq.startswith('\033[') and seq.endswith('H'):  # Move cursor
            # Parse position (simplified)
            try:
                coords = seq[2:-1].split(';')
                if len(coords) == 2:
                    self.cursor_row = max(0, min(int(coords[0]) - 1, self.height - 1))
                    self.cursor_col = max(0, min(int(coords[1]) - 1, self.width - 1))
            except ValueError:
                pass
    
    def _newline(self) -> None:
        """Handle newline character."""
        self.cursor_row += 1
        self.cursor_col = 0
        if self.cursor_row >= self.height:
            self._scroll_up()
            self.cursor_row = self.height - 1
    
    def _backspace(self) -> None:
        """Handle backspace character."""
        if self.cursor_col > 0:
            self.cursor_col -= 1
            self._set_char_at_cursor(' ')
    
    def _tab(self) -> None:
        """Handle tab character."""
        tab_stop = ((self.cursor_col // 8) + 1) * 8
        self.cursor_col = min(tab_stop, self.width - 1)
    
    def _insert_char(self, char: str) -> None:
        """Insert character at current cursor position."""
        self._set_char_at_cursor(char)
        self.cursor_col += 1
        if self.cursor_col >= self.width:
            self._newline()
    
    def _set_char_at_cursor(self, char: str) -> None:
        """Set character at current cursor position."""
        if 0 <= self.cursor_row < self.height:
            line = list(self.screen_buffer[self.cursor_row])
            if 0 <= self.cursor_col < self.width:
                line[self.cursor_col] = char
                self.screen_buffer[self.cursor_row] = ''.join(line)
    
    def _scroll_up(self) -> None:
        """Scroll screen content up by one line."""
        self.screen_buffer = self.screen_buffer[1:] + [' ' * self.width]
    
    def get_cursor_position(self) -> Tuple[int, int]:
        """Get current cursor position."""
        return self.cursor_row, self.cursor_col
    
    def get_screen_content(self) -> List[str]:
        """Get current screen content."""
        return [line.rstrip() for line in self.screen_buffer]
    
    def get_line(self, row: int) -> str:
        """Get content of specific line."""
        if 0 <= row < self.height:
            return self.screen_buffer[row].rstrip()
        return ""
    
    def save_state(self) -> TerminalState:
        """Save current terminal state."""
        return TerminalState(
            cursor_row=self.cursor_row,
            cursor_col=self.cursor_col,
            screen_width=self.width,
            screen_height=self.height,
            content=self.get_screen_content().copy(),
            timestamp=time.time()
        )


class AlignmentAnalyzer:
    """
    Analyze terminal output for alignment issues and patterns.
    """
    
    @staticmethod
    def analyze_progressive_indentation(lines: List[str]) -> List[Dict[str, Any]]:
        """Analyze lines for progressive indentation issues."""
        issues = []
        prompt_lines = []
        
        # Find all prompt lines
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith('>'):
                leading_spaces = len(line) - len(stripped)
                prompt_lines.append({
                    'line_number': i,
                    'content': line,
                    'leading_spaces': leading_spaces,
                    'position': line.find('>')
                })
        
        # Check for progressive indentation
        if len(prompt_lines) > 1:
            baseline_position = prompt_lines[0]['position']
            for i, prompt_info in enumerate(prompt_lines[1:], 1):
                if prompt_info['position'] > baseline_position:
                    issues.append({
                        'type': 'progressive_indentation',
                        'line_number': prompt_info['line_number'],
                        'expected_position': baseline_position,
                        'actual_position': prompt_info['position'],
                        'drift': prompt_info['position'] - baseline_position
                    })
        
        return issues
    
    @staticmethod
    def analyze_response_formatting(lines: List[str]) -> List[Dict[str, Any]]:
        """Analyze response formatting consistency."""
        issues = []
        in_response = False
        response_start = None
        
        for i, line in enumerate(lines):
            if '🤖 Agent:' in line:
                in_response = True
                response_start = i
                continue
            elif line.strip().startswith('>') or line.strip() in ['', '/help', '/quit', '/clear']:
                in_response = False
                continue
            
            if in_response and response_start is not None:
                # Check continuation line formatting
                if line.strip() and not line.startswith('         '):
                    # Skip the first line after "🤖 Agent:" 
                    if i != response_start + 1:
                        issues.append({
                            'type': 'inconsistent_response_formatting',
                            'line_number': i,
                            'content': line,
                            'expected_indent': 9,
                            'actual_indent': len(line) - len(line.lstrip())
                        })
        
        return issues
    
    @staticmethod
    def analyze_cursor_consistency(terminal_states: List[TerminalState]) -> List[Dict[str, Any]]:
        """Analyze cursor position consistency across states."""
        issues = []
        
        for i in range(1, len(terminal_states)):
            prev_state = terminal_states[i-1]
            curr_state = terminal_states[i]
            
            # Check for unexpected cursor jumps
            row_diff = abs(curr_state.cursor_row - prev_state.cursor_row)
            col_diff = abs(curr_state.cursor_col - prev_state.cursor_col)
            
            if row_diff > 5 or col_diff > 20:  # Thresholds for "unexpected" jumps
                issues.append({
                    'type': 'unexpected_cursor_jump',
                    'from_state': i-1,
                    'to_state': i,
                    'row_change': curr_state.cursor_row - prev_state.cursor_row,
                    'col_change': curr_state.cursor_col - prev_state.cursor_col,
                    'time_delta': curr_state.timestamp - prev_state.timestamp
                })
        
        return issues


class OutputPatternMatcher:
    """
    Match and verify specific output patterns in TUI output.
    """
    
    def __init__(self):
        self.patterns = {
            'prompt': r'>\s*',
            'thinking': r'🤔 Thinking\.\.\.',
            'response': r'🤖 Agent:',
            'error': r'❌ Error:',
            'help': r'📚 Commands:',
            'clear_confirmation': r'🧹 Conversation history cleared!'
        }
    
    def find_pattern_occurrences(self, lines: List[str], pattern_name: str) -> List[Dict[str, Any]]:
        """Find all occurrences of a specific pattern."""
        import re
        
        if pattern_name not in self.patterns:
            return []
        
        pattern = self.patterns[pattern_name]
        occurrences = []
        
        for i, line in enumerate(lines):
            if re.search(pattern, line):
                occurrences.append({
                    'line_number': i,
                    'content': line,
                    'pattern': pattern_name
                })
        
        return occurrences
    
    def verify_conversation_flow(self, lines: List[str]) -> Dict[str, Any]:
        """Verify that conversation flow follows expected patterns."""
        flow_analysis = {
            'valid': True,
            'issues': [],
            'flow_sequence': []
        }
        
        current_state = 'initial'
        
        for i, line in enumerate(lines):
            if '>' in line and line.strip().startswith('>'):
                flow_analysis['flow_sequence'].append(('prompt', i))
                if current_state not in ['initial', 'response_complete', 'command_complete']:
                    flow_analysis['issues'].append({
                        'type': 'unexpected_prompt',
                        'line_number': i,
                        'previous_state': current_state
                    })
                current_state = 'waiting_input'
                
            elif '🤔 Thinking...' in line:
                flow_analysis['flow_sequence'].append(('thinking', i))
                if current_state != 'processing':
                    current_state = 'processing'
                    
            elif '🤖 Agent:' in line:
                flow_analysis['flow_sequence'].append(('response', i))
                if current_state != 'processing':
                    flow_analysis['issues'].append({
                        'type': 'response_without_thinking',
                        'line_number': i
                    })
                current_state = 'responding'
                
            elif line.strip() and current_state == 'responding':
                # Continue response
                continue
            elif line.strip() == '':
                if current_state == 'responding':
                    current_state = 'response_complete'
                    
        return flow_analysis


@contextmanager
def mock_terminal_environment(width: int = 80, height: int = 24):
    """Context manager for mocking terminal environment."""
    terminal = TerminalSimulator(width, height)
    original_stdout = sys.stdout
    
    class MockStdout:
        def write(self, text):
            terminal.write(text)
            return len(text)
        
        def flush(self):
            pass
            
        def isatty(self):
            return True
    
    mock_stdout = MockStdout()
    
    try:
        sys.stdout = mock_stdout
        yield terminal
    finally:
        sys.stdout = original_stdout


class PerformanceBenchmarker:
    """
    Benchmark TUI operations for performance analysis.
    """
    
    def __init__(self):
        self.measurements = {}
    
    @contextmanager
    def measure(self, operation_name: str):
        """Context manager to measure operation time."""
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            if operation_name not in self.measurements:
                self.measurements[operation_name] = []
            self.measurements[operation_name].append(duration)
    
    def get_statistics(self, operation_name: str) -> Dict[str, float]:
        """Get performance statistics for an operation."""
        if operation_name not in self.measurements:
            return {}
        
        measurements = self.measurements[operation_name]
        return {
            'count': len(measurements),
            'total': sum(measurements),
            'average': sum(measurements) / len(measurements),
            'min': min(measurements),
            'max': max(measurements),
            'median': sorted(measurements)[len(measurements) // 2]
        }
    
    def check_performance_degradation(self, operation_name: str, threshold: float = 2.0) -> bool:
        """Check if performance has degraded over time."""
        if operation_name not in self.measurements or len(self.measurements[operation_name]) < 10:
            return False
        
        measurements = self.measurements[operation_name]
        early_avg = sum(measurements[:5]) / 5
        late_avg = sum(measurements[-5:]) / 5
        
        return late_avg > early_avg * threshold


# Common test fixtures and utilities
def create_test_responses() -> List[str]:
    """Create a set of test responses for mocking."""
    return [
        "Simple response",
        "Multi-line response\nWith second line\nAnd third line",
        "Response with code:\n```python\nprint('hello')\n```",
        "Very long response that might wrap across terminal lines. " * 10,
        "Response with special characters: !@#$%^&*() 🌟 你好"
    ]


def create_test_inputs() -> List[str]:
    """Create a set of test inputs."""
    return [
        "hello",
        "write a python function",
        "/help",
        "/clear", 
        "",
        "   ",
        "input with special chars !@#$%^&*()",
        "very long input that might exceed normal terminal width boundaries and test wrapping behavior",
        "unicode input 🚀 你好 🌟"
    ]


if __name__ == "__main__":
    # Basic self-test
    terminal = TerminalSimulator()
    terminal.write("Test> Hello World\n")
    terminal.write("Response: This is a test\n")
    
    content = terminal.get_screen_content()
    print("Terminal content:")
    for i, line in enumerate(content[:5]):  # Show first 5 lines
        print(f"{i}: '{line}'")
    
    print(f"Cursor at: {terminal.get_cursor_position()}")