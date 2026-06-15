"""Microphone capture — PortAudio in, resampled float32 frames out.

Opens a ``sounddevice.InputStream`` at the device's native rate inside
:meth:`MicrophoneStream.start` (never at import or ``__init__``, so the module
loads on headless hosts with no audio device). The PortAudio callback fires on a
C thread and hands frames to the event loop via ``loop.call_soon_threadsafe``;
:meth:`frames` drains an :class:`asyncio.Queue` and yields ``blocksize``-ish
float32 mono blocks resampled to ``sample_rate``. While muted (e.g. during our
own TTS) incoming audio is dropped so we never transcribe our own voice.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import numpy as np
from scipy.signal import resample_poly

# Pushed onto the queue by stop() to end the frames() generator cleanly.
_SENTINEL: object = object()
# Bound the queue so a stalled consumer can't grow memory without limit.
_QUEUE_MAXSIZE = 64


class MicrophoneStream:
    """Async microphone source yielding resampled float32 mono frames."""

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        device: int | None = None,
        blocksize: int = 512,
    ) -> None:
        """Configure capture without touching any audio device.

        Args:
            sample_rate: Target rate for yielded frames (device audio is
                resampled to this).
            device: PortAudio input device index, or None for the default.
            blocksize: Approximate samples per yielded block at ``sample_rate``.
        """
        self.sample_rate = sample_rate
        self.device = device
        self.blocksize = blocksize
        self.muted: bool = False

        self._device_rate: int = sample_rate
        self._stream: object | None = None
        self._queue: asyncio.Queue[object] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running: bool = False

    async def start(self) -> None:
        """Open the input stream at the device rate and begin capturing.

        Idempotent: a second call while already running is a no-op.

        Raises:
            sounddevice.PortAudioError: If no input device can be opened.
        """
        if self._running:
            return

        import sounddevice as sd

        info = sd.query_devices(self.device, "input")
        self._device_rate = int(info["default_samplerate"])

        # Capture at the native input rate so PortAudio never resamples for us;
        # blocksize is in *device* samples scaled from the target blocksize.
        device_blocksize = max(1, round(self.blocksize * self._device_rate / self.sample_rate))

        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
        self._running = True

        self._stream = sd.InputStream(
            samplerate=self._device_rate,
            device=self.device,
            channels=1,
            dtype="float32",
            blocksize=device_blocksize,
            callback=self._callback,
        )
        self._stream.start()

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,  # noqa: ARG002 — PortAudio callback signature
        time: object,  # noqa: ARG002 — PortAudio callback signature
        status: object,
    ) -> None:
        """PortAudio C-thread callback: marshal one block onto the loop queue.

        Args:
            indata: ``(frames, channels)`` float32 block from PortAudio.
            frames: Sample count (unused; ``indata`` carries the shape).
            time: PortAudio timing struct (unused).
            status: Over/underflow flags; dropouts are tolerated silently.
        """
        if not self._running or self.muted:
            return  # drop audio while muted so our own TTS can't echo back

        # Copy out of PortAudio's reused buffer before crossing the thread.
        mono = np.asarray(indata, dtype=np.float32).reshape(-1).copy()
        loop = self._loop
        if loop is None:
            return
        loop.call_soon_threadsafe(self._enqueue, mono)

    def _enqueue(self, block: np.ndarray) -> None:
        """Resample a captured block and place it on the queue (loop thread).

        Args:
            block: Native-rate float32 mono samples from the callback.
        """
        queue = self._queue
        if queue is None or not self._running:
            return

        resampled = self._resample(block)
        try:
            queue.put_nowait(resampled)
        except asyncio.QueueFull:
            pass  # consumer is behind; drop the oldest-equivalent frame

    def _resample(self, block: np.ndarray) -> np.ndarray:
        """Resample ``block`` from the device rate to ``sample_rate``.

        Args:
            block: Native-rate float32 mono samples.

        Returns:
            Float32 mono samples at ``sample_rate`` (unchanged when rates match).
        """
        if self._device_rate == self.sample_rate or block.size == 0:
            return block
        out = resample_poly(block, self.sample_rate, self._device_rate)
        return np.asarray(out, dtype=np.float32)

    async def stop(self) -> None:
        """Stop capture, close the stream, and end :meth:`frames`.

        Idempotent and safe to call after a failed :meth:`start`.
        """
        if not self._running:
            return
        self._running = False

        stream = self._stream
        self._stream = None
        if stream is not None:
            stream.stop()
            stream.close()

        # Unblock any consumer parked on the queue.
        queue = self._queue
        loop = self._loop
        if queue is not None and loop is not None:
            loop.call_soon_threadsafe(self._put_sentinel)

    def _put_sentinel(self) -> None:
        """Push the end sentinel onto the queue (loop thread)."""
        queue = self._queue
        if queue is None:
            return
        try:
            queue.put_nowait(_SENTINEL)
        except asyncio.QueueFull:
            # Make room: the consumer is ending regardless.
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                queue.put_nowait(_SENTINEL)
            except asyncio.QueueFull:
                return

    async def frames(self) -> AsyncIterator[np.ndarray]:
        """Yield resampled float32 mono blocks until :meth:`stop` is called.

        Yields:
            ``blocksize``-ish 1-D float32 arrays at ``sample_rate``.
        """
        queue = self._queue
        if queue is None:
            return  # start() was never called

        while True:
            item = await queue.get()
            if item is _SENTINEL:
                return
            yield item  # type: ignore[misc]  # non-sentinel items are ndarrays

    def mute(self) -> None:
        """Drop incoming audio (echo guard during our own TTS playback)."""
        self.muted = True

    def unmute(self) -> None:
        """Resume passing captured audio to :meth:`frames`."""
        self.muted = False
