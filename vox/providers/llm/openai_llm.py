"""OpenAI-compatible LLM backend — async streaming via the official ``openai`` SDK.

Generic on purpose: with ``base_url`` set, the same class reaches vLLM,
llama.cpp, TabbyAPI, LM Studio, Groq, Together, OpenRouter, and friends — any
server speaking the OpenAI chat-completions protocol. ``base_url=None`` targets
the official OpenAI endpoint. Local servers often need no real key, so a
harmless placeholder is used when none is supplied.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Sequence

from vox.providers.base import LLMProvider
from vox.types import Message


class OpenAILLM(LLMProvider):
    """Streams responses from any OpenAI-compatible server via ``chat.completions.create``."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 1024,
        **_: object,
    ) -> None:
        """Build the backend, lazily importing ``openai`` so the module stays import-clean.

        ``base_url`` selects the endpoint (``None`` → official OpenAI). ``api_key``
        falls back to ``OPENAI_API_KEY`` and finally to a ``sk-noauth`` placeholder
        so keyless local servers work. Extra keyword options are accepted and
        ignored for spec-factory passthrough.
        """
        from openai import AsyncOpenAI

        self.model = model
        self._max_tokens = max_tokens
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY") or "sk-noauth"
        self._client = AsyncOpenAI(base_url=base_url, api_key=resolved_key)

    async def stream(
        self, messages: Sequence[Message], *, system: str | None = None
    ) -> AsyncIterator[str]:
        """Yield assistant text chunks for ``messages``, prepending ``system`` as a message."""
        msgs: list[dict[str, str]] = [{"role": "system", "content": system}] if system else []
        msgs += [m.as_dict() for m in messages]

        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=msgs,  # type: ignore[arg-type]
            stream=True,
            max_tokens=self._max_tokens,
        )
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    async def aclose(self) -> None:
        """Release the underlying httpx client. Safe to call repeatedly."""
        await self._client.close()
