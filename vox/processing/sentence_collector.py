"""Streaming sentence collector.

Accumulates LLM tokens and emits complete sentences as soon as a real sentence
boundary appears, guarding against the classic false-split traps: abbreviations,
decimals, ellipses, URLs and email addresses. Sentences are emitted eagerly so
downstream TTS can begin synthesizing the first clause while the model is still
producing the rest of the turn.
"""

from __future__ import annotations

import re

# Abbreviations whose trailing period must NOT end a sentence (lower-cased,
# trailing dots stripped for matching).
_ABBREVIATIONS: frozenset[str] = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "st",
        "vs",
        "etc",
        "inc",
        "ltd",
        "co",
        "corp",
        "dept",
        "fig",
        "no",
        "vol",
        "e.g",
        "i.e",
        "a.m",
        "p.m",
        "u.s",
        "u.k",
        "u.n",
    }
)

# Sentence-final punctuation, including the single-character ellipsis.
_SENTENCE_FINAL: frozenset[str] = frozenset({".", "!", "?", "…"})

# A candidate boundary: one-or-more final punctuation marks (so "?!" or "..."
# collapse to one boundary) followed by whitespace/newline or end of string.
_BOUNDARY_RE = re.compile(r"[.!?…]+(?=\s|$)")

# Trailing word (letters/digits/dots) immediately before a boundary, used to
# detect abbreviations and decimals.
_TRAILING_TOKEN_RE = re.compile(r"([A-Za-z0-9][A-Za-z0-9.]*)$")

# A URL or email sitting at the tail of the buffer; dots inside it are never
# sentence boundaries.
_URL_EMAIL_TAIL_RE = re.compile(
    r"(?:https?://|www\.)\S*$|[^\s@]+@[^\s@]+$",
    re.IGNORECASE,
)


class SentenceCollector:
    """Collect streamed tokens and yield complete sentences at safe boundaries."""

    def __init__(self, *, max_chars: int = 200) -> None:
        """Create a collector that force-splits any clause exceeding ``max_chars``."""
        self._max_chars = max_chars
        self._buffer = ""

    def feed(self, token: str) -> list[str]:
        """Append ``token`` to the buffer and return any complete sentences now available.

        Returns a (possibly empty) list of trimmed, non-empty sentences. The
        remainder stays buffered until its own boundary arrives or :meth:`flush`
        is called.
        """
        if not token:
            return self._drain_oversized([])

        self._buffer += token
        sentences: list[str] = []

        while True:
            cut = self._next_boundary(self._buffer)
            if cut is None:
                break
            head, self._buffer = self._buffer[:cut], self._buffer[cut:]
            stripped = head.strip()
            if stripped:
                sentences.append(stripped)

        return self._drain_oversized(sentences)

    def flush(self) -> str:
        """Return whatever remains in the buffer (trimmed), clearing it."""
        remainder = self._buffer.strip()
        self._buffer = ""
        return remainder

    def reset(self) -> None:
        """Discard all buffered text."""
        self._buffer = ""

    def _drain_oversized(self, sentences: list[str]) -> list[str]:
        """Force-split the buffer at the last space while it exceeds ``max_chars``."""
        while len(self._buffer) > self._max_chars:
            window = self._buffer[: self._max_chars]
            split_at = window.rfind(" ")
            if split_at <= 0:
                # No space to break on — emit the whole oversized window.
                split_at = self._max_chars
            head, self._buffer = self._buffer[:split_at], self._buffer[split_at:]
            self._buffer = self._buffer.lstrip()
            stripped = head.strip()
            if stripped:
                sentences.append(stripped)
        return sentences

    def _next_boundary(self, text: str) -> int | None:
        """Return the index just past the first real sentence boundary, or None.

        Considers newline boundaries and punctuation boundaries, rejecting
        false positives from abbreviations, decimals, and URLs/emails.
        """
        candidate: int | None = None

        newline = text.find("\n")
        if newline != -1:
            candidate = newline + 1

        for match in _BOUNDARY_RE.finditer(text):
            end = match.end()
            if candidate is not None and end >= candidate:
                # A newline before this punctuation already wins.
                break
            if self._is_false_split(text, match.start()):
                continue
            if end == len(text) and self._is_provisional_tail(text, match.start()):
                # The mark sits at the very end with no whitespace yet; the next
                # token might reveal it as a decimal/abbreviation. Defer the split.
                continue
            candidate = end
            break

        return candidate

    def _is_provisional_tail(self, text: str, dot_index: int) -> bool:
        """Return True if an end-of-buffer mark could still resolve to a false split.

        Only fires when the punctuation is the final character (matched via ``$``
        with no trailing whitespace). A digit immediately before the dot may
        become a decimal once the next chunk arrives; a short alpha token may
        become a known abbreviation. In both cases we wait for more input.
        """
        before = text[dot_index - 1 : dot_index]
        if before.isdigit():
            return True
        match = _TRAILING_TOKEN_RE.search(text[:dot_index])
        if match is None:
            return False
        normalized = match.group(1).rstrip(".").lower()
        return any(abbr.startswith(normalized) for abbr in _ABBREVIATIONS)

    def _is_false_split(self, text: str, dot_index: int) -> bool:
        """Return True if punctuation at ``dot_index`` is NOT a sentence end."""
        head = text[:dot_index]

        # Inside a URL or email at the tail before the punctuation.
        if _URL_EMAIL_TAIL_RE.search(head):
            return True

        match = _TRAILING_TOKEN_RE.search(head)
        if match is None:
            return False
        token = match.group(1)

        # Decimal number: a digit precedes the dot and a digit follows it.
        after = text[dot_index + 1 : dot_index + 2]
        if token[-1:].isdigit() and after.isdigit():
            return True

        # Known abbreviation (compare without trailing dots, case-insensitive).
        normalized = token.rstrip(".").lower()
        if normalized in _ABBREVIATIONS:
            return True

        # Single capital initial like "A." or "J." (people's initials).
        if len(normalized) == 1 and token[0].isalpha():
            return True

        return False
