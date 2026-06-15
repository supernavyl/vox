"""Provider contracts — the seam that lets `vox` wrap *any* LLM, STT, or TTS.

These four ABCs are the entire coupling surface of the engine. The pipeline and
conversation loop know nothing beyond them; swapping a backend is implementing
one class. Audio is always float32 mono 1-D ``numpy``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence

import numpy as np

from vox.types import Message


class LLMProvider(ABC):
    """Streams assistant text token-by-token. The one interface for *any* LLM."""

    model: str

    @abstractmethod
    def stream(
        self, messages: Sequence[Message], *, system: str | None = None
    ) -> AsyncIterator[str]:
        """Yield response text chunks for ``messages``.

        Implemented as an ``async def`` generator (``async def stream(...): ... yield``).
        ``system`` is the system prompt; backends that take it as a first message
        should prepend it themselves.
        """
        raise NotImplementedError

    async def aclose(self) -> None:
        """Release any network resources. Safe to call more than once."""


class STTProvider(ABC):
    """Speech-to-text over a finished utterance."""

    @abstractmethod
    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Transcribe a mono float32 utterance, returning stripped text ("" if silence)."""
        raise NotImplementedError

    async def aclose(self) -> None:
        """Release model resources."""


class TTSProvider(ABC):
    """Text-to-speech for a single sentence/clause."""

    sample_rate: int

    @abstractmethod
    async def synthesize(self, text: str) -> np.ndarray:
        """Synthesize ``text`` to mono float32 audio at :attr:`sample_rate`."""
        raise NotImplementedError

    async def aclose(self) -> None:
        """Release model resources."""


class VADProvider(ABC):
    """Voice-activity detection over a single frame."""

    @abstractmethod
    def is_speech(self, frame: np.ndarray, sample_rate: int) -> bool:
        """Return True if ``frame`` (mono float32) contains speech."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset any internal state between utterances."""
