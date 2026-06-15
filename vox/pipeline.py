"""The streaming voice pipeline: LLM tokens → clean sentences → Kokoro audio.

`process_streaming` is the heart of the engine. It yields ``(sentence, audio)``
the instant each sentence is synthesized, so playback starts on the first
sentence rather than waiting for the whole reply — and it runs every token
through the think-stripper, anti-parrot, and TTS sanitizer first.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

import numpy as np

from vox.processing.anti_parrot import AntiParrot
from vox.processing.interruption import InterruptionHandler
from vox.processing.sentence_collector import SentenceCollector
from vox.processing.think_stripper import ThinkStripper
from vox.processing.tts_sanitizer import sanitize_tts_text
from vox.providers.base import LLMProvider, TTSProvider
from vox.types import Message, Role


class VoicePipeline:
    """Composes an LLM and a TTS provider with the streaming text filters."""

    def __init__(
        self,
        llm: LLMProvider,
        tts: TTSProvider,
        *,
        system_prompt: str,
        interruption: InterruptionHandler,
        think_stripper: ThinkStripper | None = None,
        anti_parrot: AntiParrot | None = None,
        sentence_collector: SentenceCollector | None = None,
    ) -> None:
        self._llm = llm
        self._tts = tts
        self._system_prompt = system_prompt
        self._interruption = interruption
        self._think = think_stripper or ThinkStripper()
        self._parrot = anti_parrot or AntiParrot()
        self._sentences = sentence_collector or SentenceCollector()

    async def process_streaming(
        self, user_text: str, history: Sequence[Message]
    ) -> AsyncIterator[tuple[str, np.ndarray]]:
        """Yield ``(spoken_text, audio)`` per sentence as the LLM streams its reply."""
        self._think.reset()
        self._parrot.reset()
        self._sentences.reset()

        messages = [*history, Message(Role.USER, user_text)]
        token_stream = self._llm.stream(messages, system=self._system_prompt)
        guarded = self._interruption.wrap_stream(token_stream)

        first = True
        async for token in guarded:
            visible = self._think.filter_chunk(token)
            if not visible:
                continue
            safe = self._parrot.filter_chunk(visible, user_text)
            if not safe:
                continue
            for sentence in self._sentences.feed(safe):
                spoken = self._finalize(sentence, user_text, first)
                if spoken is None:
                    continue
                first = False
                audio = await self._tts.synthesize(spoken)
                if audio.size:
                    yield spoken, audio

        # Drain anything buffered in the filters at end-of-stream.
        tail = self._think.flush()
        if tail:
            tail = self._parrot.filter_chunk(tail, user_text)
            for sentence in self._sentences.feed(tail):
                spoken = self._finalize(sentence, user_text, first)
                if spoken is None:
                    continue
                first = False
                audio = await self._tts.synthesize(spoken)
                if audio.size:
                    yield spoken, audio

        leftover = self._sentences.flush().strip()
        if leftover:
            spoken = self._finalize(leftover, user_text, first)
            if spoken:
                audio = await self._tts.synthesize(spoken)
                if audio.size:
                    yield spoken, audio

    def _finalize(self, sentence: str, user_text: str, first: bool) -> str | None:
        """Apply opening anti-parrot (first sentence only) + TTS sanitization.

        Returns the speakable text, or None if the sentence should be dropped.
        """
        if first:
            sentence = self._parrot.sanitize_opening(sentence, user_text)
            if not sentence:
                return None
        spoken = sanitize_tts_text(sentence).strip()
        return spoken or None
