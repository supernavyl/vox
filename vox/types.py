"""Shared, dependency-free data types for the voice engine.

Audio convention throughout `vox`: float32, mono, 1-D ``numpy`` arrays. Each
stage advertises its own sample rate; resampling happens only at the I/O edges
(microphone, speaker).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Role(str, Enum):
    """Conversation roles, aligned with every major chat-completions API."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(slots=True)
class Message:
    """One turn of conversation handed to an :class:`~vox.providers.base.LLMProvider`."""

    role: Role
    content: str

    def as_dict(self) -> dict[str, str]:
        """Render to the ``{"role": ..., "content": ...}`` shape every chat API expects."""
        return {"role": self.role.value, "content": self.content}


class ConversationState(str, Enum):
    """Coarse lifecycle of a single conversational turn."""

    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
