"""Voice-activity detection — energy default, optional Silero upgrade.

:class:`EnergyVAD` is dependency-free (RMS + zero-crossing rate) and is the live
default whenever ``silero_vad`` is absent. :class:`SileroVAD` lazily imports the
neural model and only engages on the 512-sample / 16 kHz frames it expects,
falling back to energy detection otherwise. :func:`make_vad` is the single
entry point the engine calls: it prefers Silero and degrades to energy on
:class:`ImportError`, so importing this module never requires the optional dep.
"""

from __future__ import annotations

import numpy as np

from vox.providers.base import VADProvider

# Silero operates on fixed-size frames at a fixed rate; off-spec frames are
# routed to the energy detector instead of being silently dropped.
_SILERO_FRAME_SAMPLES = 512
_SILERO_SAMPLE_RATE = 16_000

# Energy-floor mapping: a 0..1 ``threshold`` knob maps onto an RMS floor. 0.0 is
# very permissive (~0.002), 1.0 is strict (~0.012). Tuned for float32 [-1, 1].
_ENERGY_FLOOR_BASE = 0.002
_ENERGY_FLOOR_SPAN = 0.010

# Speech has many polarity changes per frame; pure DC offset or low hum has very
# few. This minimum zero-crossing *rate* (crossings / sample) rejects them.
_MIN_ZERO_CROSSING_RATE = 0.02


class EnergyVAD(VADProvider):
    """RMS-plus-zero-crossing speech detector with no external dependencies."""

    def __init__(self, threshold: float = 0.5) -> None:
        """Configure the detector.

        Args:
            threshold: Sensitivity in ``[0, 1]``; higher demands louder speech.
                Mapped to an RMS energy floor (see module constants).
        """
        clamped = min(max(threshold, 0.0), 1.0)
        self.threshold = clamped
        self._energy_floor = _ENERGY_FLOOR_BASE + clamped * _ENERGY_FLOOR_SPAN

    def is_speech(self, frame: np.ndarray, sample_rate: int) -> bool:
        """Return True when ``frame`` clears both the energy and ZCR gates.

        Args:
            frame: Mono float32 audio, 1-D.
            sample_rate: Frame sample rate (unused by the energy heuristic but
                part of the :class:`VADProvider` contract).

        Returns:
            True only if RMS energy exceeds the floor *and* the zero-crossing
            rate clears the minimum (rejecting DC offset and low hum).
        """
        if frame.size == 0:
            return False

        samples = np.asarray(frame, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return False

        rms = float(np.sqrt(np.mean(np.square(samples))))
        if rms < self._energy_floor:
            return False

        # Crossings of zero, normalised by sample count → rate in [0, 1].
        signs = np.signbit(samples)
        crossings = int(np.count_nonzero(signs[1:] != signs[:-1]))
        zero_crossing_rate = crossings / samples.size
        return zero_crossing_rate >= _MIN_ZERO_CROSSING_RATE

    def reset(self) -> None:
        """No-op; the energy detector keeps no per-utterance state."""


class SileroVAD(VADProvider):
    """Neural VAD backed by ``silero_vad``, with an energy fallback."""

    def __init__(self, threshold: float = 0.5) -> None:
        """Load the Silero model eagerly so failure surfaces at construction.

        Args:
            threshold: Speech-probability cutoff in ``[0, 1]`` passed to Silero.

        Raises:
            ImportError: If ``silero_vad`` (or its runtime) is unavailable, so
                :func:`make_vad` can degrade to :class:`EnergyVAD`.
        """
        try:
            from silero_vad import load_silero_vad
        except ImportError as exc:  # optional dependency
            raise ImportError("silero_vad is not installed; use EnergyVAD instead") from exc

        self.threshold = min(max(threshold, 0.0), 1.0)
        self._model = load_silero_vad()
        # Off-spec frames (wrong length/rate) are scored by this detector.
        self._energy = EnergyVAD(self.threshold)

    def is_speech(self, frame: np.ndarray, sample_rate: int) -> bool:
        """Score a 512-sample / 16 kHz frame with Silero, else fall back.

        Args:
            frame: Mono float32 audio, 1-D.
            sample_rate: Frame sample rate; must be 16 kHz for the neural path.

        Returns:
            True when speech probability exceeds the threshold (neural path), or
            the energy detector's verdict for any off-spec frame.
        """
        if frame.size == 0:
            return False

        samples = np.asarray(frame, dtype=np.float32).reshape(-1)
        if samples.size != _SILERO_FRAME_SAMPLES or sample_rate != _SILERO_SAMPLE_RATE:
            return self._energy.is_speech(samples, sample_rate)

        import torch

        with torch.no_grad():
            tensor = torch.from_numpy(samples)
            probability = float(self._model(tensor, _SILERO_SAMPLE_RATE).item())
        return probability >= self.threshold

    def reset(self) -> None:
        """Reset Silero's recurrent state between utterances."""
        reset_states = getattr(self._model, "reset_states", None)
        if callable(reset_states):
            reset_states()


def make_vad(threshold: float = 0.5) -> VADProvider:
    """Build the best available VAD: Silero if installed, else energy.

    Args:
        threshold: Sensitivity in ``[0, 1]`` forwarded to the chosen backend.

    Returns:
        A :class:`SileroVAD` when ``silero_vad`` is importable, otherwise a
        :class:`EnergyVAD`. This is the constructor the engine calls.
    """
    try:
        return SileroVAD(threshold)
    except ImportError:
        return EnergyVAD(threshold)
