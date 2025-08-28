# tests/utils.py
"""
Utility helpers used by many test modules.
"""

import json
from pathlib import Path
from typing import Any, Callable

import pytest


def read_json(path: Path) -> Any:
    """Read a JSON file and return the parsed Python object."""
    return json.loads(path.read_text())


def assert_json_equal(actual: Any, expected: Any):
    """Recursively compare two JSON‑compatible structures."""
    assert actual == expected, f"JSON structures differ:\n{actual}\n!=\n{expected}"


@pytest.fixture
def async_dummy():
    """
    Helper that creates a simple coroutine that returns its argument.
    Useful for patching async functions.
    """
    async def _dummy(value):
        return value

    return _dummy


def raise_if_called(*_args, **_kwargs):
    """Utility used for "should‑not‑be‑called" patches."""
    raise AssertionError("This mocked function should not have been called!")