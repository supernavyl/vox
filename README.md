# vox

A state-of-the-art **local voice engine** — Kokoro TTS, faster-whisper STT — that wraps **any LLM** behind one string.

```
mic → VAD → endpoint → STT → [ any LLM ] → sentence-stream → sanitize → Kokoro → speaker
                                  ▲                                          │
                                  └──────────── barge-in (interrupt) ◀───────┘
```

Distilled from two real-time voice organisms (LIS + Emily): Emily's clean provider/pipeline architecture as the spine, LIS's sanitizer "secret sauce" (anti-parrot, think-stripping, TTS sanitization), best-of-breed VAD and streaming from both.

## Wrap any LLM with one string

```
ollama:llama3.2                                  # local Ollama
ollama:qwen3:8b                                  # reasoning model (think-tags stripped automatically)
openai:gpt-4o                                    # OpenAI
anthropic:claude-opus-4-8                        # Claude
groq:llama-3.3-70b-versatile                     # Groq (OpenAI-compatible)
openai-compat:my-model@http://localhost:8080/v1  # vLLM / llama.cpp / TabbyAPI / LM Studio
```

## Quick start

```bash
uv pip install -e '.[ollama]'      # or .[openai], .[anthropic], .[all]
vox talk --llm ollama:llama3.2     # start talking
vox say "hello, world"             # TTS smoke test
vox devices                        # list audio devices
```

```python
import asyncio
from vox import VoiceEngine, VoxConfig

async def main() -> None:
    engine = VoiceEngine(VoxConfig(llm="anthropic:claude-opus-4-8"))
    await engine.run()

asyncio.run(main())
```

## What makes it state-of-the-art

- **Any-LLM seam** — one `LLMProvider.stream()` ABC; backends are ~50 lines each.
- **Streaming end-to-end** — playback starts on the *first* synthesized sentence, not the full reply.
- **Barge-in** — talk over the assistant and it stops instantly (`InterruptionHandler.wrap_stream`).
- **Anti-parrot** — n-gram + Jaccard echo removal so the model never repeats you back.
- **Think-stripping** — `<think>…</think>` from reasoning models never reaches the speaker.
- **TTS sanitization** — `[laughs]`, `*sighs*`, `<voice:…>` stripped before synthesis.
- **Kokoro 24 kHz TTS**, faster-whisper STT, Silero VAD with a zero-dep energy fallback.

## Configuration

Every `VoxConfig` field has a `VOX_*` env var (`VOX_LLM`, `VOX_TTS_VOICE`, `VOX_SILENCE_MS`, …). See `vox/config.py`.

## License

MIT.
