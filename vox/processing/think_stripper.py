"""Streaming ``<think>...</think>`` remover.

A small state machine that strips reasoning spans from a token stream, even when
the open/close tags are split across chunk boundaries. The invariant it
guarantees is::

    "".join(stripper.filter_chunk(c) for c in chunks) + stripper.flush()
        == joined_stream_with_think_spans_removed
"""

from __future__ import annotations


class ThinkStripper:
    """Remove ``<think>...</think>`` spans from a streamed text feed."""

    def __init__(self, *, open_tag: str = "<think>", close_tag: str = "</think>") -> None:
        """Configure the open/close tags that delimit hidden reasoning."""
        if not open_tag or not close_tag:
            raise ValueError("open_tag and close_tag must be non-empty")
        self._open_tag = open_tag
        self._close_tag = close_tag
        self._inside = False
        self._buffer = ""

    def filter_chunk(self, chunk: str) -> str:
        """Return the visible portion of ``chunk`` with think-spans removed.

        Holds back up to ``len(tag) - 1`` trailing characters that *could* be the
        start of a tag split across this and the next chunk, so a partial tag is
        never emitted as visible text.
        """
        if not chunk:
            return ""

        self._buffer += chunk
        out: list[str] = []

        while self._buffer:
            if self._inside:
                idx = self._buffer.find(self._close_tag)
                if idx == -1:
                    # Keep only enough tail to detect a split close_tag next chunk.
                    self._buffer = self._keep_partial_tail(self._close_tag)
                    break
                # Drop the hidden span and the close tag; resume visible mode.
                self._buffer = self._buffer[idx + len(self._close_tag) :]
                self._inside = False
                continue

            idx = self._buffer.find(self._open_tag)
            if idx == -1:
                emit, self._buffer = self._split_safe_tail(self._open_tag)
                if emit:
                    out.append(emit)
                break

            # Emit everything before the open tag, then enter hidden mode.
            out.append(self._buffer[:idx])
            self._buffer = self._buffer[idx + len(self._open_tag) :]
            self._inside = True

        return "".join(out)

    def flush(self) -> str:
        """Emit any safely-buffered visible text and reset to a clean state.

        Text still held because it could have started a tag is now emitted (the
        stream has ended, so no completion is possible) — unless we are mid-span,
        in which case the unterminated reasoning is dropped.
        """
        if self._inside:
            self._buffer = ""
            self._inside = False
            return ""
        remainder = self._buffer
        self._buffer = ""
        return remainder

    def reset(self) -> None:
        """Clear all state between turns."""
        self._inside = False
        self._buffer = ""

    def _split_safe_tail(self, tag: str) -> tuple[str, str]:
        """Split the buffer into (emittable, held) given no full ``tag`` was found.

        The held portion is the longest proper suffix of the buffer that is also
        a proper prefix of ``tag`` — i.e. the only bytes that could still grow
        into ``tag`` once more chunks arrive.
        """
        max_keep = min(len(tag) - 1, len(self._buffer))
        for keep in range(max_keep, 0, -1):
            if tag.startswith(self._buffer[-keep:]):
                return self._buffer[:-keep], self._buffer[-keep:]
        return self._buffer, ""

    def _keep_partial_tail(self, tag: str) -> str:
        """While inside a span, keep only the suffix that could begin ``tag``."""
        max_keep = min(len(tag) - 1, len(self._buffer))
        for keep in range(max_keep, 0, -1):
            if tag.startswith(self._buffer[-keep:]):
                return self._buffer[-keep:]
        return ""
