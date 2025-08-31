import os
import pytest
from agentsmcp.ui.keyboard_input import KeyboardInput, KeyCode, MouseEvent

@pytest.fixture
def pipe_keyboard():
    # Create a pipe; read end will be injected into KeyboardInput, write end used by test
    r_fd, w_fd = os.pipe()
    yield r_fd, w_fd
    os.close(r_fd)
    os.close(w_fd)

def build_keyboard(r_fd):
    kb = KeyboardInput()
    # Force KeyboardInput to use the pipe's read end instead of real tty
    kb._fd = r_fd
    kb._fd_file = os.fdopen(r_fd, 'rb', buffering=0)
    kb._orig_settings = None  # No termios on pipe
    return kb

def test_backspace_deletes_character(pipe_keyboard):
    r_fd, w_fd = pipe_keyboard
    kb = build_keyboard(r_fd)
    # Simulate typing "abc"
    os.write(w_fd, b'abc')
    for expected_char in ('a', 'b', 'c'):
        key, char, mode, mouse = kb.get_key(timeout=0.1)
        assert char == expected_char
        assert key is None
        assert mouse is None
    # Now send Backspace byte
    os.write(w_fd, b'\x7f')
    key, char, mode, mouse = kb.get_key(timeout=0.1)
    assert key == KeyCode.BACKSPACE
    assert char is None
    assert mouse is None
    # Buffer should now contain "ab"
    assert kb.get_current_line() == 'ab'

def test_mouse_scroll_up(pipe_keyboard):
    r_fd, w_fd = pipe_keyboard
    kb = build_keyboard(r_fd)
    # SGR scroll‑up sequence: ESC [ < 64 ; x ; y M
    os.write(w_fd, b'\x1b[<64;10;20M')
    key, char, mode, mouse = kb.get_key(timeout=0.1)
    assert mouse == MouseEvent.SCROLL_UP
    assert key is None and char is None

def test_mouse_scroll_down(pipe_keyboard):
    r_fd, w_fd = pipe_keyboard
    kb = build_keyboard(r_fd)
    # SGR scroll‑down sequence: ESC [ < 65 ; x ; y M
    os.write(w_fd, b'\x1b[<65;5;3M')
    key, char, mode, mouse = kb.get_key(timeout=0.1)
    assert mouse == MouseEvent.SCROLL_DOWN
    assert key is None and char is None
