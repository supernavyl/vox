"""The full voice engine: wire providers, run the listen ↔ respond loop.

Two concurrent tasks share state through an utterance queue and a coarse
:class:`ConversationState`:

* **listen loop** — reads mic frames, runs VAD endpointing, and (while the
  assistant is speaking) watches for sustained speech to fire barge-in.
* **respond loop** — pulls a finished utterance, transcribes it, streams the
  LLM reply through the pipeline, and plays each sentence as it lands.

Barge-in works best with OS-level acoustic echo cancellation (e.g. a PipeWire
echo-cancelled mic source); without it, sustained-speech gating still prevents
most self-interruption from the engine's own voice.
"""

from __future__ import annotations

import asyncio
import contextlib

import numpy as np

from vox.audio.microphone import MicrophoneStream
from vox.audio.speaker import Speaker
from vox.audio.vad import make_vad
from vox.config import VoxConfig
from vox.pipeline import VoicePipeline
from vox.processing.interruption import InterruptionHandler
from vox.providers.base import LLMProvider, STTProvider, TTSProvider
from vox.providers.llm.spec import parse_llm_spec
from vox.providers.stt.faster_whisper import FasterWhisperSTT
from vox.providers.tts.kokoro import KokoroTTS
from vox.types import ConversationState, Message, Role

_SAMPLE_RATE = 16_000
_BLOCKSIZE = 512
_FRAME_MS = 1000.0 * _BLOCKSIZE / _SAMPLE_RATE
_BARGEIN_HOLD_MS = 220.0  # sustained speech needed to interrupt the assistant


class VoiceEngine:
    """A complete, runnable voice conversation over any configured LLM."""

    def __init__(
        self,
        config: VoxConfig | None = None,
        *,
        llm: LLMProvider | None = None,
        stt: STTProvider | None = None,
        tts: TTSProvider | None = None,
    ) -> None:
        self.config = config or VoxConfig()
        self._llm = llm or parse_llm_spec(self.config.llm, max_tokens=self.config.max_tokens)
        self._stt = stt or FasterWhisperSTT(
            model=self.config.stt_model,
            device=self.config.stt_device,
            compute_type=self.config.stt_compute_type,
            language=self.config.stt_language,
        )
        self._tts = tts or KokoroTTS(
            voice=self.config.tts_voice,
            lang=self.config.tts_lang,
            device=self.config.tts_device,
            speed=self.config.tts_speed,
        )
        self._vad = make_vad(self.config.vad_threshold)
        self._mic = MicrophoneStream(
            sample_rate=_SAMPLE_RATE,
            device=self.config.input_device,
            blocksize=_BLOCKSIZE,
        )
        self._speaker = Speaker(device=self.config.output_device)
        self._interrupt = InterruptionHandler()
        self._pipeline = VoicePipeline(
            self._llm,
            self._tts,
            system_prompt=self.config.system_prompt,
            interruption=self._interrupt,
        )

        self._state = ConversationState.LISTENING
        self._history: list[Message] = []
        self._utterances: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self._utt: list[np.ndarray] = []
        self._has_speech = False
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._bargein_ms = 0.0

    async def run(self) -> None:
        """Start listening and conversing until cancelled (Ctrl-C)."""
        await self._mic.start()
        listen = asyncio.create_task(self._listen_loop(), name="vox-listen")
        respond = asyncio.create_task(self._respond_loop(), name="vox-respond")
        try:
            await asyncio.gather(listen, respond)
        finally:
            for task in (listen, respond):
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            await self._shutdown()

    # -- listening -----------------------------------------------------------

    async def _listen_loop(self) -> None:
        async for frame in self._mic.frames():
            speech = self._vad.is_speech(frame, _SAMPLE_RATE)

            if self._state is ConversationState.SPEAKING:
                self._watch_bargein(speech)
                continue
            if self._state is not ConversationState.LISTENING:
                continue  # THINKING — ignore the mic until the reply starts

            self._accumulate(frame, speech)

    def _watch_bargein(self, speech: bool) -> None:
        if not speech:
            self._bargein_ms = 0.0
            return
        self._bargein_ms += _FRAME_MS
        if self._bargein_ms >= _BARGEIN_HOLD_MS and not self._interrupt.is_interrupted:
            self._interrupt.signal()
            self._speaker.cancel()

    def _accumulate(self, frame: np.ndarray, speech: bool) -> None:
        if speech:
            self._utt.append(frame)
            self._has_speech = True
            self._speech_ms += _FRAME_MS
            self._silence_ms = 0.0
            return
        if not self._has_speech:
            return
        # Trailing silence after speech — count toward the endpoint.
        self._utt.append(frame)
        self._silence_ms += _FRAME_MS
        if self._silence_ms < self.config.silence_ms:
            return

        utterance = np.concatenate(self._utt) if self._utt else np.zeros(0, dtype=np.float32)
        long_enough = self._speech_ms >= self.config.min_speech_ms
        self._reset_utterance()
        if long_enough:
            self._state = ConversationState.THINKING
            self._utterances.put_nowait(utterance)

    def _reset_utterance(self) -> None:
        self._utt = []
        self._has_speech = False
        self._speech_ms = 0.0
        self._silence_ms = 0.0

    # -- responding ----------------------------------------------------------

    async def _respond_loop(self) -> None:
        while True:
            utterance = await self._utterances.get()
            try:
                await self._handle_utterance(utterance)
            finally:
                self._utterances.task_done()

    async def _handle_utterance(self, utterance: np.ndarray) -> None:
        text = (await self._stt.transcribe(utterance, _SAMPLE_RATE)).strip()
        if not text:
            self._state = ConversationState.LISTENING
            return

        self._on_transcript(text)
        self._history.append(Message(Role.USER, text))
        self._state = ConversationState.SPEAKING
        self._interrupt.clear()
        self._mic.mute()  # damp our own voice from re-entering STT during synth gaps

        spoken_parts: list[str] = []
        try:
            # History excludes the current user turn — the pipeline appends it.
            async for sentence, audio in self._pipeline.process_streaming(text, self._history[:-1]):
                if self._interrupt.is_interrupted:
                    break
                spoken_parts.append(sentence)
                self._on_assistant(sentence)
                self._mic.unmute()  # listen for barge-in while actually playing
                await self._speaker.play(audio, self._tts.sample_rate)
                if self._interrupt.is_interrupted:
                    break
        except asyncio.CancelledError:
            pass  # barge-in raised through wrap_stream

        reply = " ".join(spoken_parts).strip()
        if reply and not self._interrupt.is_interrupted:
            self._history.append(Message(Role.ASSISTANT, reply))
        self._trim_history()

        await asyncio.sleep(self.config.post_speak_mute_ms / 1000.0)
        self._mic.unmute()
        self._reset_utterance()
        self._state = ConversationState.LISTENING

    def _trim_history(self) -> None:
        max_msgs = self.config.max_history_turns * 2
        if len(self._history) > max_msgs:
            self._history = self._history[-max_msgs:]

    # -- hooks (overridable for UIs/logging) ---------------------------------

    def _on_transcript(self, text: str) -> None:
        print(f"\n\033[36myou:\033[0m {text}")

    def _on_assistant(self, sentence: str) -> None:
        print(f"\033[35mvox:\033[0m {sentence}")

    async def _shutdown(self) -> None:
        with contextlib.suppress(Exception):
            await self._mic.stop()
        with contextlib.suppress(Exception):
            await self._speaker.close()
        for provider in (self._llm, self._stt, self._tts):
            with contextlib.suppress(Exception):
                await provider.aclose()
