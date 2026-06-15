"""Kokoro TTS backend — lightweight 24 kHz neural text-to-speech.

Lazy-loads the ``KPipeline`` on the first :meth:`KokoroTTS.synthesize` so import
and construction stay cheap. The blocking synthesis runs in a thread-pool
executor. CUDA presence is probed via ``ctranslate2`` (we avoid importing
``torch`` directly, since it may be absent), with a graceful CPU fallback.
"""

from __future__ import annotations

import asyncio
import warnings

import numpy as np

from vox.providers.base import TTSProvider


class KokoroTTS(TTSProvider):
    """Synthesizes single clauses with a lazily loaded Kokoro ``KPipeline``."""

    sample_rate = 24_000  # Kokoro emits 24 kHz audio.

    def __init__(
        self,
        *,
        voice: str = "af_nicole",
        lang: str = "a",
        device: str = "auto",
        speed: float = 1.0,
    ) -> None:
        """Configure the backend without loading the pipeline yet.

        Args:
            voice: Kokoro voice id (e.g. ``"af_nicole"``).
            lang: Kokoro ``lang_code`` (e.g. ``"a"`` for American English).
            device: ``"auto"`` resolves to ``"cuda"`` when a CUDA device is
                visible, else ``"cpu"``.
            speed: Speaking-rate multiplier.
        """
        self._voice = voice
        self._lang = lang
        self._device = device
        self._speed = speed
        self._pipeline: object | None = None

    def _resolve_device(self) -> str:
        """Return the concrete device, probing CUDA via ctranslate2 when ``auto``."""
        if self._device != "auto":
            return self._device

        try:
            import ctranslate2

            return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception as exc:  # noqa: BLE001 — probe is best-effort; cpu is safe.
            warnings.warn(
                f"CUDA probe failed ({exc!r}); falling back to cpu for Kokoro.",
                stacklevel=2,
            )
            return "cpu"

    def _ensure_pipeline(self) -> object:
        """Lazily construct and cache the ``KPipeline`` on first use."""
        if self._pipeline is not None:
            return self._pipeline

        from kokoro import KPipeline

        device = self._resolve_device()
        try:
            self._pipeline = KPipeline(lang_code=self._lang, device=device)
        except Exception as exc:  # noqa: BLE001 — fall back to cpu on any GPU error.
            if device == "cpu":
                raise
            warnings.warn(
                f"Kokoro KPipeline on {device} failed ({exc!r}); falling back to cpu.",
                stacklevel=2,
            )
            self._pipeline = KPipeline(lang_code=self._lang, device="cpu")
        return self._pipeline

    async def synthesize(self, text: str) -> np.ndarray:
        """Synthesize ``text`` to mono float32 audio at :attr:`sample_rate`.

        Returns an empty float32 array for blank input. The blocking pipeline
        runs in the default executor; chunks are concatenated into one 1-D array.
        """
        if not text.strip():
            return np.zeros(0, dtype=np.float32)

        pipeline = self._ensure_pipeline()
        voice = self._voice
        speed = self._speed

        def _blocking() -> np.ndarray:
            chunks: list[np.ndarray] = []
            for item in pipeline(text, voice=voice, speed=speed):  # type: ignore[operator]
                chunk = item[2]
                chunks.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
            if not chunks:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(chunks).astype(np.float32, copy=False)

        return await asyncio.get_running_loop().run_in_executor(None, _blocking)

    async def aclose(self) -> None:
        """No network resources to release."""
