"""Model-free integration tests for the streaming pipeline and the LLM seam.

Run: ``PYTHONPATH=. python3 tests/test_pipeline.py`` (or under pytest).
Uses fake LLM/TTS providers so it needs no network, no models, and no audio.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence

import numpy as np

from vox.pipeline import VoicePipeline
from vox.processing.interruption import InterruptionHandler
from vox.providers.base import LLMProvider, TTSProvider
from vox.providers.llm.spec import available_providers, parse_llm_spec
from vox.types import Message


class FakeLLM(LLMProvider):
    """Streams a scripted reply, character by character."""

    model = "fake"

    def __init__(self, reply: str) -> None:
        self._reply = reply

    async def stream(
        self, messages: Sequence[Message], *, system: str | None = None
    ) -> AsyncIterator[str]:
        for ch in self._reply:
            yield ch

    async def aclose(self) -> None:
        return None


class FakeTTS(TTSProvider):
    """Returns a fixed-length tone per sentence so audio size is deterministic."""

    sample_rate = 24000

    async def synthesize(self, text: str) -> np.ndarray:
        return np.ones(len(text) * 10, dtype=np.float32)

    async def aclose(self) -> None:
        return None


def _run(coro):
    return asyncio.run(coro)


async def _collect(pipeline: VoicePipeline, user_text: str) -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = []
    async for sentence, audio in pipeline.process_streaming(user_text, []):
        out.append((sentence, audio))
    return out


def test_think_strip_and_sentence_split() -> None:
    llm = FakeLLM("<think>secret reasoning</think>It's sunny and warm today. Enjoy your afternoon!")
    pipe = VoicePipeline(
        llm, FakeTTS(), system_prompt="be nice", interruption=InterruptionHandler()
    )
    results = _run(_collect(pipe, "what's the weather"))

    spoken = [s for s, _ in results]
    assert spoken, "expected at least one sentence"
    joined = " ".join(spoken)
    assert "<think>" not in joined and "secret" not in joined, joined
    assert "sunny" in joined.lower(), joined
    assert all(audio.size > 0 for _, audio in results), "every sentence must produce audio"
    assert len(results) >= 2, f"expected the reply to split into >=2 sentences, got {spoken}"
    print("OK think-strip + split:", spoken)


def test_barge_in_stops_stream() -> None:
    interrupt = InterruptionHandler()
    llm = FakeLLM("One. Two. Three. Four. Five. Six. Seven. Eight.")
    pipe = VoicePipeline(llm, FakeTTS(), system_prompt="", interruption=interrupt)

    async def scenario() -> int:
        count = 0
        async for _sentence, _audio in pipe.process_streaming("count", []):
            count += 1
            interrupt.signal()  # barge in after the first sentence
        return count

    try:
        produced = _run(scenario())
    except asyncio.CancelledError:
        produced = 1  # wrap_stream raised through — that is a valid stop
    assert produced <= 2, f"barge-in should stop quickly, produced {produced}"
    print("OK barge-in stopped after", produced, "sentence(s)")


def test_llm_spec_seam() -> None:
    assert "ollama" in available_providers()
    assert "anthropic" in available_providers()
    assert "openai-compat" in available_providers()

    for bad in ("", "nocolon", "bogusprov:model"):
        try:
            parse_llm_spec(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for spec {bad!r}")

    # anthropic SDK is installed here, so this constructs a real provider object.
    provider = parse_llm_spec("anthropic:claude-opus-4-8")
    assert provider.model == "claude-opus-4-8"
    print("OK llm-spec seam: parsed anthropic:claude-opus-4-8 ->", type(provider).__name__)


if __name__ == "__main__":
    test_think_strip_and_sentence_split()
    test_barge_in_stops_stream()
    test_llm_spec_seam()
    print("\nall pipeline smoke tests passed")
