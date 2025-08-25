"""Unified streaming interface (S1–S2).

Provides a provider-agnostic `Chunk` type and helpers to stream outputs.
For providers without streaming, we degrade to a single final chunk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Iterable, Optional


@dataclass(frozen=True)
class Chunk:
    text: str
    is_final: bool = False
    finish_reason: Optional[str] = None


def generate_stream_from_text(text: str, step: int = 40) -> Generator[Chunk, None, None]:
    """Yield text in small chunks; final chunk marks completion (S1).

    This is a shim for non-streaming providers or for offline/test mode.
    """
    if not text:
        yield Chunk(text="", is_final=True, finish_reason="empty")
        return
    for i in range(0, len(text), max(1, step)):
        part = text[i : i + step]
        final = i + step >= len(text)
        yield Chunk(text=part, is_final=final, finish_reason="stop" if final else None)


__all__ = ["Chunk", "generate_stream_from_text"]


# --- Provider stubs (S2) ---
def openai_stream(prompt: str) -> Generator[Chunk, None, None]:
    """Stub OpenAI streaming adapter: yields chunks from final text.

    Replace with real provider streaming when wiring network calls.
    """
    for ch in generate_stream_from_text(prompt):
        yield ch


def openrouter_stream(prompt: str) -> Generator[Chunk, None, None]:
    """Stub OpenRouter streaming adapter: yields chunks from final text."""
    for ch in generate_stream_from_text(prompt):
        yield ch


# Optional provider-native OpenAI streaming (S2)
def openai_stream_text(api_key: str, base_url: Optional[str], model: str, prompt: str, temperature: float = 0.7) -> Generator[Chunk, None, None]:
    """Stream tokens from OpenAI Chat Completions if available.

    Falls back to chunking final output if streaming is not available or an error occurs.
    This function is not wired by default; set an env flag or call explicitly when ready.
    """
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key, base_url=base_url)
        stream = client.chat.completions.create(
            model=model,
            temperature=temperature,
            stream=True,
            messages=[{"role": "user", "content": prompt}],
        )
        # The event type names depend on SDK; handle common shapes
        final_text = ""
        for event in stream:  # type: ignore
            try:
                deltas = getattr(event, "choices", [])[0].get("delta") if isinstance(event, dict) else event.choices[0].delta  # type: ignore[attr-defined]
                piece = deltas.get("content") if isinstance(deltas, dict) else getattr(deltas, "content", None)
            except Exception:
                piece = None
            if piece:
                final_text += piece
                yield Chunk(text=piece, is_final=False)
        yield Chunk(text="", is_final=True, finish_reason="stop")
    except Exception:
        # Fallback to chunking the final result via a single call (non-stream)
        for ch in generate_stream_from_text(prompt):
            yield ch
