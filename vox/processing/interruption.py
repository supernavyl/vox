"""Barge-in interruption handling.

When the user starts speaking over the assistant ("barge-in"), the in-flight
LLM/TTS stream must stop *now*. :class:`InterruptionHandler` wraps an async text
stream and tears it down promptly the moment an interruption is signalled,
surfacing it as :class:`asyncio.CancelledError` so callers unwind cleanly.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator


class InterruptionHandler:
    """Coordinate barge-in cancellation across a streaming turn."""

    def __init__(self) -> None:
        """Create a handler armed for the first turn (not interrupted)."""
        self._event = asyncio.Event()

    def signal(self) -> None:
        """Mark the current turn as interrupted (barge-in detected)."""
        self._event.set()

    def clear(self) -> None:
        """Re-arm for the next turn, clearing any prior interruption."""
        self._event.clear()

    @property
    def is_interrupted(self) -> bool:
        """True once :meth:`signal` has fired and before :meth:`clear`."""
        return self._event.is_set()

    async def wrap_stream(self, stream: AsyncIterator[str]) -> AsyncIterator[str]:
        """Yield from ``stream`` until interrupted, then raise ``asyncio.CancelledError``.

        The interruption flag is checked both before and after each item so a
        barge-in arriving mid-yield aborts before the next chunk is produced.
        """
        if self._event.is_set():
            raise asyncio.CancelledError("interrupted before stream start")

        async for item in stream:
            if self._event.is_set():
                raise asyncio.CancelledError("interrupted mid-stream")
            yield item
            if self._event.is_set():
                raise asyncio.CancelledError("interrupted after item")
