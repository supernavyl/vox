"""Runtime configuration for the voice engine.

Plain dataclass + ``from_env`` so the engine has zero config-framework
dependency. Every field maps to a ``VOX_*`` environment variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

_DEFAULT_SYSTEM_PROMPT = (
    "You are a warm, concise voice assistant. Speak naturally, in short "
    "conversational sentences. Never use markdown, lists, emoji, or stage "
    "directions — your words are spoken aloud."
)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw else default


def _env_opt_int(name: str) -> int | None:
    raw = os.environ.get(name)
    return int(raw) if raw else None


@dataclass(slots=True)
class VoxConfig:
    """All knobs for one voice session."""

    # --- LLM ----------------------------------------------------------------
    # An llm-spec string: "provider:model[@base_url]".
    #   ollama:llama3.2 · ollama:qwen3:8b · openai:gpt-4o · anthropic:claude-opus-4-8
    #   openai-compat:my-model@http://localhost:8080/v1
    llm: str = "ollama:llama3.2"
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT
    max_tokens: int = 1024
    max_history_turns: int = 12

    # --- STT (faster-whisper) ----------------------------------------------
    stt_model: str = "distil-large-v3"
    stt_device: str = "auto"  # auto | cpu | cuda
    stt_compute_type: str = "auto"  # auto | int8 | float16
    stt_language: str = "en"

    # --- TTS (Kokoro) -------------------------------------------------------
    tts_voice: str = "af_nicole"
    tts_lang: str = "a"  # Kokoro lang code ('a' = American English)
    tts_device: str = "auto"  # auto | cpu | cuda
    tts_speed: float = 1.0

    # --- VAD / endpointing --------------------------------------------------
    vad_threshold: float = 0.5  # Silero probability OR scaled energy threshold
    silence_ms: int = 700  # trailing silence that closes a turn
    min_speech_ms: int = 250  # shorter utterances are dropped as noise
    post_speak_mute_ms: int = 350  # mic-mute after we stop talking (echo guard)

    # --- audio I/O ----------------------------------------------------------
    input_device: int | None = None
    output_device: int | None = None

    @classmethod
    def from_env(cls) -> VoxConfig:
        """Build a config from ``VOX_*`` environment variables, falling back to defaults."""
        return cls(
            llm=os.environ.get("VOX_LLM", cls.llm),
            system_prompt=os.environ.get("VOX_SYSTEM_PROMPT", _DEFAULT_SYSTEM_PROMPT),
            max_tokens=_env_int("VOX_MAX_TOKENS", cls.max_tokens),
            max_history_turns=_env_int("VOX_MAX_HISTORY_TURNS", cls.max_history_turns),
            stt_model=os.environ.get("VOX_STT_MODEL", cls.stt_model),
            stt_device=os.environ.get("VOX_STT_DEVICE", cls.stt_device),
            stt_compute_type=os.environ.get("VOX_STT_COMPUTE_TYPE", cls.stt_compute_type),
            stt_language=os.environ.get("VOX_STT_LANGUAGE", cls.stt_language),
            tts_voice=os.environ.get("VOX_TTS_VOICE", cls.tts_voice),
            tts_lang=os.environ.get("VOX_TTS_LANG", cls.tts_lang),
            tts_device=os.environ.get("VOX_TTS_DEVICE", cls.tts_device),
            tts_speed=_env_float("VOX_TTS_SPEED", cls.tts_speed),
            vad_threshold=_env_float("VOX_VAD_THRESHOLD", cls.vad_threshold),
            silence_ms=_env_int("VOX_SILENCE_MS", cls.silence_ms),
            min_speech_ms=_env_int("VOX_MIN_SPEECH_MS", cls.min_speech_ms),
            post_speak_mute_ms=_env_int("VOX_POST_SPEAK_MUTE_MS", cls.post_speak_mute_ms),
            input_device=_env_opt_int("VOX_INPUT_DEVICE"),
            output_device=_env_opt_int("VOX_OUTPUT_DEVICE"),
        )
