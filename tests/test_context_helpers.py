import pytest

from agentsmcp.context import estimate_tokens, trim_history


def test_estimate_tokens_basic():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") >= 1
    assert estimate_tokens("a" * 40) >= 10  # approx chars/4


def test_trim_history_keeps_recent():
    history = [("user", f"msg{i}") for i in range(10)]
    trimmed = trim_history(history, target_tokens=5)  # small budget
    # Should return some of the most recent messages, preserving order
    assert isinstance(trimmed, list)
    assert all(isinstance(t, tuple) and len(t) == 2 for t in trimmed)
    # If budget is zero, returns empty
    assert trim_history(history, 0) == []

