"""Context window management helpers (X1).

Provides rough token estimation and deterministic trimming of chat history
while preserving recency. Keep it simple and fast.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


def estimate_tokens(text: str) -> int:
    """Very rough token estimate (chars/4 fallback).

    This heuristic keeps us deterministic without external deps.
    """
    if not text:
        return 0
    # Approximate: average 4 chars per token across mixed English/code
    return max(1, len(text) // 4)


def trim_history(history: Sequence[Tuple[str, str]], target_tokens: int) -> List[Tuple[str, str]]:
    """Trim history to fit within target_tokens, keeping most recent first.

    - History is a list of (role, text)
    - Returns a subsequence preserving order, biased to recency
    """
    if target_tokens <= 0 or not history:
        return []

    selected: List[Tuple[str, str]] = []
    total = 0
    # Walk from end to start, then reverse to preserve chronological order
    for role, text in reversed(history):
        cost = estimate_tokens(text) + 2  # small overhead per message
        if total + cost > target_tokens:
            break
        selected.append((role, text))
        total += cost
    return list(reversed(selected))


__all__ = ["estimate_tokens", "trim_history"]

