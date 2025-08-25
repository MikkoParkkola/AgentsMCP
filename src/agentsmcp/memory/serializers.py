"""
Context serialization helpers.

The context structure is assumed to be JSON‑serialisable (i.e. it contains
strings, numbers, dicts, lists …).  For larger payloads a small zlib
compression is applied to keep the round‑trip below the 100 ms latency
target while staying CPU‑light.

The helper functions are idempotent – passing already compressed data
raises an informative error.
"""

from __future__ import annotations

import json
import zlib
from typing import Any, Dict

__all__ = [
    "serialize_context",
    "deserialize_context",
]


def serialize_context(context: Dict[str, Any], compress: bool = True) -> bytes:
    """
    Convert a context dictionary to a compressed JSON `bytes` blob.

    Args:
        context: A JSON‑serialisable dictionary describing the agent state.
        compress: If *True*, the JSON string is compressed with `zlib`.
                  Defaults to *True* – use *False* only for debugging.

    Returns:
        A `bytes` object ready to be stored in Redis or PostgreSQL.

    Raises:
        TypeError: If the input cannot be serialised to JSON.
        ValueError: If compression is requested but the implementation
                    cannot handle the data size.
    """
    try:
        json_bytes = json.dumps(context, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TypeError("Context must be JSON‑serialisable") from exc

    if compress:
        return zlib.compress(json_bytes, level=9)
    return json_bytes


def deserialize_context(data: bytes, compressed: bool = True) -> Dict[str, Any]:
    """
    Reverse :func:`serialize_context` – convert a bytes blob back to a plain
    Python `dict`.

    Args:
        data: The bytes retrieved from storage.
        compressed: Whether :data:`data` is compressed.  The default mirrors
                    the production path.

    Returns:
        The original context dictionary.

    Raises:
        OSError: If the data could not be decompressed (corrupt stream).
        json.JSONDecodeError: If the data is not valid JSON.
    """
    if compressed:
        try:
            data = zlib.decompress(data)
        except zlib.error as exc:
            raise OSError("Failed to decompress context data") from exc

    try:
        return json.loads(data.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise json.JSONDecodeError("Context is not valid JSON", doc=str(data), pos=0) from exc