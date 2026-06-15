"""Anti-parrot filter.

Stops the assistant from echoing the user's own words back at them — the single
most grating failure mode of a real-time voice agent. Two complementary tools:

* :meth:`AntiParrot.filter_chunk` — streaming n-gram scrubber that removes any
  run of the user's words (longest first) from the assistant stream without ever
  rewinding already-spoken audio.
* :meth:`AntiParrot.sanitize_opening` — sentence-level guard that drops a whole
  opening line if it is just a paraphrase/mirror of what the user said.

Ported from the LIS voice organism.
"""

from __future__ import annotations

import re

# Mirror lead-ins to strip from an opening sentence before overlap scoring.
_LEAD_IN_RE = re.compile(
    r"^\s*(?:"
    r"you(?:'re| are)\s+saying|"
    r"what\s+i\s+hear\s+is|"
    r"so\s+you(?:'re| are)\s+(?:saying|asking)|"
    r"it\s+sounds\s+like|"
    r"if\s+i\s+understand(?:\s+you)?(?:\s+correctly)?|"
    r"so(?:,)?\s+you\s+want|"
    r"what\s+you(?:'re| are)\s+(?:saying|asking)\s+is"
    r")[\s,:-]*",
    re.IGNORECASE,
)

_WORD_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


class AntiParrot:
    """Remove echoes of the user's words from the assistant's reply stream."""

    def __init__(
        self,
        *,
        min_ngram: int = 3,
        max_ngram: int = 8,
        open_overlap: float = 0.85,
        short_overlap: float = 0.95,
    ) -> None:
        """Configure n-gram window and the long/short Jaccard-overlap thresholds."""
        if min_ngram < 1 or max_ngram < min_ngram:
            raise ValueError("require 1 <= min_ngram <= max_ngram")
        self._min_ngram = min_ngram
        self._max_ngram = max_ngram
        self._open_overlap = open_overlap
        self._short_overlap = short_overlap
        self._emitted = ""

    def filter_chunk(self, chunk: str, user_text: str) -> str:
        """Scrub user-echoing n-grams from the running stream; return only the new tail.

        ``(emitted + chunk)`` is scrubbed of the user's word-level n-grams
        (case-insensitive, word-boundary, longest n-grams first), whitespace is
        collapsed, and only the portion *after* what was already emitted is
        returned — already-spoken audio is never rewound.
        """
        if not chunk:
            return ""

        combined = self._emitted + chunk
        scrubbed = self._scrub_ngrams(combined, user_text)

        # The already-emitted prefix is immutable. Return the new tail only.
        if scrubbed.startswith(self._emitted):
            tail = scrubbed[len(self._emitted) :]
        else:
            # Scrubbing changed an already-emitted region (rare); fall back to the
            # longest common prefix so we still never rewind.
            common = self._common_prefix_len(scrubbed, self._emitted)
            tail = scrubbed[common:]

        self._emitted = scrubbed
        return tail

    def sanitize_opening(self, sentence: str, user_text: str) -> str:
        """Drop an opening sentence that merely mirrors the user; else clean it.

        Strips mirror lead-ins, then compares Jaccard word-overlap of the
        remainder against ``user_text``. Returns ``""`` to drop, otherwise the
        de-leadin'd sentence.
        """
        cleaned = _LEAD_IN_RE.sub("", sentence).strip()
        if not cleaned:
            return ""

        overlap = self._jaccard(cleaned, user_text)
        tokens = len(self._words(cleaned))

        if tokens >= 6 and overlap >= self._open_overlap:
            return ""
        if tokens <= 5 and overlap >= self._short_overlap:
            return ""
        return cleaned

    def reset(self) -> None:
        """Clear the running emitted-text buffer between turns."""
        self._emitted = ""

    def _scrub_ngrams(self, text: str, user_text: str) -> str:
        """Remove user word-level n-grams from ``text`` (longest first), collapse spaces."""
        user_words = self._words(user_text)
        if not user_words:
            # Collapse whitespace runs but do NOT strip: this buffer is still
            # streaming, and stripping the trailing space eats inter-word gaps.
            return _WS_RE.sub(" ", text)

        result = text
        upper = min(self._max_ngram, len(user_words))
        for n in range(upper, self._min_ngram - 1, -1):
            seen: set[str] = set()
            for i in range(len(user_words) - n + 1):
                phrase = " ".join(user_words[i : i + n])
                if phrase in seen:
                    continue
                seen.add(phrase)
                pattern = re.compile(
                    r"\b" + r"\s+".join(re.escape(w) for w in phrase.split(" ")) + r"\b",
                    re.IGNORECASE,
                )
                result = pattern.sub(" ", result)

        # Collapse whitespace runs, but do not strip — see note above.
        return _WS_RE.sub(" ", result)

    def _jaccard(self, a: str, b: str) -> float:
        """Jaccard overlap of the two texts' lower-cased word sets (0.0 if either empty)."""
        set_a = set(self._words(a))
        set_b = set(self._words(b))
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        if union == 0:
            return 0.0
        return intersection / union

    @staticmethod
    def _words(text: str) -> list[str]:
        """Lower-cased word tokens of ``text``."""
        return [m.group(0).lower() for m in _WORD_RE.finditer(text)]

    @staticmethod
    def _common_prefix_len(a: str, b: str) -> int:
        """Length of the longest shared prefix of ``a`` and ``b``."""
        limit = min(len(a), len(b))
        i = 0
        while i < limit and a[i] == b[i]:
            i += 1
        return i
