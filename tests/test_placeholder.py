"""Tests for placeholder module."""

from agentsmcp.placeholder import add


def test_add() -> None:
    """Ensure add returns the sum of operands."""
    assert add(2, 3) == 5
