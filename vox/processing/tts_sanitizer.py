"""TTS text sanitizer.

LLMs leak paralinguistic stage directions ("[chuckles]", "*sighs softly*",
"(pauses)") and inline voice tags ("<voice:nicole>", "voice: sky") into their
prose. Spoken aloud by a TTS engine these become embarrassing literal readings
of "chuckles" or, worse, the model voicing its own control tokens. This module
strips them before synthesis.
"""

from __future__ import annotations

import re

# Action verbs/adverbs that mark a wrapped span as a stage direction rather than
# meaningful content. Kept deliberately voice-acting flavored.
_ACTION_WORDS: frozenset[str] = frozenset(
    {
        "laughs",
        "laughing",
        "laugh",
        "chuckles",
        "chuckling",
        "chuckle",
        "giggles",
        "sighs",
        "sighing",
        "sigh",
        "gasps",
        "groans",
        "grunts",
        "sniffles",
        "clears",
        "coughs",
        "whispers",
        "whispering",
        "mutters",
        "murmurs",
        "shouts",
        "yells",
        "screams",
        "cries",
        "sobs",
        "pauses",
        "pause",
        "pausing",
        "breathes",
        "breathing",
        "inhales",
        "exhales",
        "smiles",
        "smiling",
        "grins",
        "winks",
        "nods",
        "shrugs",
        "softly",
        "quietly",
        "loudly",
        "gently",
        "warmly",
        "nervously",
        "sarcastically",
        "excitedly",
        "hesitantly",
        "thoughtfully",
        "cheerfully",
        "sadly",
        "angrily",
        "throat",
        "beat",
    }
)

# One word inside a wrapper, lower-cased, used to decide if it is an action span.
_INNER_WORD_RE = re.compile(r"[a-z]+", re.IGNORECASE)

# [bracketed] spans — almost always stage directions; removed unconditionally.
_BRACKET_RE = re.compile(r"\[[^\]]*\]")

# *asterisk* and (paren) spans — removed only when they wrap an action word.
_ASTERISK_RE = re.compile(r"\*([^*]+)\*")
_PAREN_RE = re.compile(r"\(([^()]*)\)")

# Voice tags: <voice:nicole>, <voice : af_nicole>, and bare "voice: sky".
_VOICE_TAG_ANGLED_RE = re.compile(r"<\s*voice\s*:\s*[A-Za-z0-9_\-]+\s*>", re.IGNORECASE)
_VOICE_TAG_BARE_RE = re.compile(r"\bvoice\s*:\s*[A-Za-z0-9_\-]+", re.IGNORECASE)

# Cleanup passes.
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.!?;:…])")
_MULTISPACE_RE = re.compile(r"[ \t]{2,}")


def _is_action_span(inner: str) -> bool:
    """Return True if ``inner`` (a wrapped span's text) reads as a stage direction."""
    words = [m.group(0).lower() for m in _INNER_WORD_RE.finditer(inner)]
    if not words:
        return False
    return any(word in _ACTION_WORDS for word in words)


def _strip_if_action(match: re.Match[str]) -> str:
    """Drop a ``*..*``/``(..)`` span iff it wraps an action word, else keep it verbatim."""
    inner = match.group(1)
    if _is_action_span(inner):
        return " "
    return match.group(0)


def sanitize_tts_text(text: str) -> str:
    """Strip paralinguistic stage directions and voice tags from ``text`` for TTS.

    Removes ``[bracketed]`` directions outright, ``*asterisk*`` and
    ``(parenthetical)`` spans when they wrap an action verb/adverb, and inline
    voice tags such as ``<voice:nicole>`` or ``voice: sky``. Collapses the
    resulting whitespace and fixes spacing before punctuation.
    """
    if not text:
        return ""

    cleaned = _VOICE_TAG_ANGLED_RE.sub(" ", text)
    cleaned = _VOICE_TAG_BARE_RE.sub(" ", cleaned)
    cleaned = _BRACKET_RE.sub(" ", cleaned)
    cleaned = _ASTERISK_RE.sub(_strip_if_action, cleaned)
    cleaned = _PAREN_RE.sub(_strip_if_action, cleaned)

    cleaned = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", cleaned)
    cleaned = _MULTISPACE_RE.sub(" ", cleaned)
    return cleaned.strip()
