from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncGenerator, Dict, List


class EventBus:
    """Simple in-process pub/sub for server events (WUI1)."""

    def __init__(self) -> None:
        self._subscribers: List[asyncio.Queue] = []
        self._lock = asyncio.Lock()

    async def publish(self, event: Dict[str, Any]) -> None:
        async with self._lock:
            for q in list(self._subscribers):
                # best effort; drop if queue is closed
                try:
                    q.put_nowait(event)
                except Exception:
                    pass

    async def subscribe(self) -> AsyncGenerator[str, None]:
        q: asyncio.Queue = asyncio.Queue()
        async with self._lock:
            self._subscribers.append(q)
        try:
            while True:
                ev = await q.get()
                yield f"data: {json.dumps(ev)}\n\n"
        finally:
            async with self._lock:
                if q in self._subscribers:
                    self._subscribers.remove(q)


__all__ = ["EventBus"]

