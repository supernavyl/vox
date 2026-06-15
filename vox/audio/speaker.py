"""Speaker playback — block-by-block output with barge-in cancellation.

Opens a ``sounddevice.OutputStream`` lazily on the first :meth:`Speaker.play`
(never at import or ``__init__``, so the module loads on headless hosts). Input
audio is resampled to the device rate, then written in small blocks inside a
``run_in_executor`` loop that checks a ``threading.Event`` between writes, so
:meth:`cancel` (barge-in) halts playback within tens of milliseconds. We avoid
``sd.play``/``sd.stop`` deliberately — the explicit write loop sidesteps the
PortAudio start/stop race those convenience calls expose.
"""

from __future__ import annotations

import asyncio
import threading

import numpy as np
from scipy.signal import resample_poly

# Frames written per blocking ``stream.write`` — small enough that a cancel
# between blocks stops audio quickly, large enough to avoid underflow churn.
_WRITE_BLOCK_FRAMES = 1024


class Speaker:
    """Cancellable audio sink with a lazily opened PortAudio output stream."""

    def __init__(self, *, device: int | None = None) -> None:
        """Configure output without touching any audio device.

        Args:
            device: PortAudio output device index, or None for the default.
        """
        self.device = device
        self._stream: object | None = None
        self._device_rate: int = 0
        self._channels: int = 1
        self._cancel = threading.Event()
        self._lock = asyncio.Lock()

    async def play(self, audio: np.ndarray, sample_rate: int) -> None:
        """Play ``audio``, resampling to the device rate, honoring :meth:`cancel`.

        Concurrent calls are serialized; a fresh call clears any prior cancel.

        Args:
            audio: Mono float32 samples, 1-D.
            sample_rate: Sample rate of ``audio``.
        """
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return

        async with self._lock:
            self._cancel.clear()
            self._ensure_stream()

            playable = self._resample(samples, sample_rate)
            shaped = self._to_device_channels(playable)

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._write_blocking, shaped)

    def _ensure_stream(self) -> None:
        """Open the output stream at the device default rate on first use."""
        if self._stream is not None:
            return

        import sounddevice as sd

        info = sd.query_devices(self.device, "output")
        self._device_rate = int(info["default_samplerate"])
        # Honor mono when the device allows it; duplicate to stereo otherwise.
        max_out = int(info["max_output_channels"])
        self._channels = 1 if max_out >= 1 and max_out < 2 else min(2, max_out)
        if self._channels < 1:
            self._channels = 1

        stream = sd.OutputStream(
            samplerate=self._device_rate,
            device=self.device,
            channels=self._channels,
            dtype="float32",
        )
        stream.start()
        self._stream = stream

    def _resample(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Resample mono ``samples`` from ``sample_rate`` to the device rate.

        Args:
            samples: Mono float32 input.
            sample_rate: Input sample rate.

        Returns:
            Float32 mono samples at the device rate (unchanged when equal).
        """
        if sample_rate == self._device_rate:
            return samples
        out = resample_poly(samples, self._device_rate, sample_rate)
        return np.asarray(out, dtype=np.float32)

    def _to_device_channels(self, mono: np.ndarray) -> np.ndarray:
        """Shape mono audio to the stream's channel count.

        Args:
            mono: 1-D float32 samples.

        Returns:
            ``(n,)`` for a mono device, else ``(n, channels)`` with the mono
            signal duplicated across channels.
        """
        if self._channels <= 1:
            return mono
        return np.repeat(mono[:, np.newaxis], self._channels, axis=1)

    def _write_blocking(self, audio: np.ndarray) -> None:
        """Write ``audio`` block-by-block, bailing out when cancelled.

        Runs inside an executor thread (``stream.write`` blocks on PortAudio).

        Args:
            audio: Float32 samples already at the device rate and channel count.
        """
        stream = self._stream
        if stream is None:
            return

        total = audio.shape[0]
        for start in range(0, total, _WRITE_BLOCK_FRAMES):
            if self._cancel.is_set():
                return
            block = audio[start : start + _WRITE_BLOCK_FRAMES]
            stream.write(block)

    def cancel(self) -> None:
        """Stop the current playback immediately (barge-in)."""
        self._cancel.set()

    async def close(self) -> None:
        """Cancel playback and release the output stream. Safe to call twice."""
        self._cancel.set()
        async with self._lock:
            stream = self._stream
            self._stream = None
            if stream is not None:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._close_blocking, stream)

    @staticmethod
    def _close_blocking(stream: object) -> None:
        """Stop and close ``stream`` in an executor thread.

        Args:
            stream: The ``sounddevice.OutputStream`` to tear down.
        """
        stream.stop()  # type: ignore[attr-defined]
        stream.close()  # type: ignore[attr-defined]
