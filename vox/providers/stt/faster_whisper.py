"""faster-whisper STT backend — CTranslate2-accelerated Whisper transcription.

Lazy-loads the model on the first :meth:`FasterWhisperSTT.transcribe` so importing
this module (and constructing the provider) stays cheap. The blocking
``model.transcribe`` runs in a thread-pool executor so it never stalls the loop.

We deliberately avoid importing ``torch`` (it may be absent); CUDA presence is
probed through ``ctranslate2``, which ships with faster-whisper.
"""

from __future__ import annotations

import asyncio

import numpy as np

from vox.providers.base import STTProvider

_TARGET_SAMPLE_RATE = 16_000
# Below this peak amplitude an utterance is treated as silence and skipped.
_SILENCE_PEAK = 1e-4


class FasterWhisperSTT(STTProvider):
    """Transcribes finished utterances with a lazily loaded ``WhisperModel``."""

    def __init__(
        self,
        model: str = "distil-large-v3",
        *,
        device: str = "auto",
        compute_type: str = "auto",
        language: str = "en",
        beam_size: int = 5,
    ) -> None:
        """Configure the backend without loading the model yet.

        Args:
            model: faster-whisper model id or local path.
            device: ``"auto"`` resolves to ``"cuda"`` when a CUDA device is
                visible, else ``"cpu"``.
            compute_type: ``"auto"`` resolves to ``"float16"`` on cuda, ``"int8"``
                on cpu.
            language: Forced decode language; falsy disables forcing.
            beam_size: Decoder beam width.
        """
        self._model_name = model
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._beam_size = beam_size
        self._model: object | None = None

    def _resolve_device(self) -> str:
        """Return the concrete device, probing CUDA via ctranslate2 when ``auto``."""
        if self._device != "auto":
            return self._device

        import ctranslate2

        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"

    def _resolve_compute_type(self, device: str) -> str:
        """Return the concrete compute type, defaulting by device when ``auto``."""
        if self._compute_type != "auto":
            return self._compute_type
        return "float16" if device == "cuda" else "int8"

    def _ensure_model(self) -> object:
        """Lazily construct and cache the ``WhisperModel`` on first use."""
        if self._model is not None:
            return self._model

        from faster_whisper import WhisperModel

        device = self._resolve_device()
        compute_type = self._resolve_compute_type(device)
        self._model = WhisperModel(self._model_name, device=device, compute_type=compute_type)
        return self._model

    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Transcribe a mono float32 utterance, returning stripped text.

        Returns ``""`` for empty or all-silence input. Resamples to 16 kHz when
        ``sample_rate`` differs. The blocking decode runs in the default executor.
        """
        audio = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0 or float(np.max(np.abs(audio))) < _SILENCE_PEAK:
            return ""

        if sample_rate != _TARGET_SAMPLE_RATE:
            from scipy.signal import resample_poly

            audio = resample_poly(audio, _TARGET_SAMPLE_RATE, sample_rate).astype(
                np.float32, copy=False
            )

        audio16k = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
        model = self._ensure_model()
        language = self._language or None
        beam_size = self._beam_size

        def _blocking() -> str:
            segments, _info = model.transcribe(  # type: ignore[attr-defined]
                audio16k,
                beam_size=beam_size,
                language=language,
                vad_filter=True,
            )
            return "".join(segment.text for segment in segments).strip()

        return await asyncio.get_running_loop().run_in_executor(None, _blocking)

    async def aclose(self) -> None:
        """No network resources to release."""
