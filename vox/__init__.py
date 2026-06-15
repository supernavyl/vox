"""vox — a state-of-the-art local voice engine (Kokoro TTS) that wraps any LLM."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vox.config import VoxConfig

__version__ = "0.1.0"

if TYPE_CHECKING:
    from vox.conversation import VoiceEngine

__all__ = ["VoxConfig", "VoiceEngine", "__version__"]


def __getattr__(name: str) -> object:
    # Lazy so `import vox` doesn't pull in sounddevice / model runtimes.
    if name == "VoiceEngine":
        from vox.conversation import VoiceEngine

        return VoiceEngine
    raise AttributeError(f"module 'vox' has no attribute {name!r}")
