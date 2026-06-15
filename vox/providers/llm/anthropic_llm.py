"""Anthropic (Claude) LLM backend — async streaming via the official SDK.

Defaults to ``claude-opus-4-8`` (current most-capable Opus). For a voice loop we
keep thinking off and pass no sampling params, which keeps the request valid
across the whole 4.7/4.8/Fable family (those reject ``temperature``/``top_p`` and
``budget_tokens``) and minimises time-to-first-token.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from vox.providers.base import LLMProvider
from vox.types import Message, Role


class AnthropicLLM(LLMProvider):
    """Streams Claude responses through ``AsyncAnthropic.messages.stream``."""

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        from anthropic import AsyncAnthropic

        self.model = model
        self._max_tokens = max_tokens
        # Both default to None → SDK resolves api_key from ANTHROPIC_API_KEY and
        # base_url from ANTHROPIC_BASE_URL / the public endpoint.
        self._client = AsyncAnthropic(api_key=api_key, base_url=base_url)

    async def stream(
        self, messages: Sequence[Message], *, system: str | None = None
    ) -> AsyncIterator[str]:
        # Anthropic carries the system prompt as a top-level param, not a message.
        api_messages = [m.as_dict() for m in messages if m.role in (Role.USER, Role.ASSISTANT)]
        kwargs: dict[str, object] = {
            "model": self.model,
            "max_tokens": self._max_tokens,
            "messages": api_messages,
        }
        if system:
            kwargs["system"] = system

        async with self._client.messages.stream(**kwargs) as stream:  # type: ignore[arg-type]
            async for text in stream.text_stream:
                yield text

    async def aclose(self) -> None:
        await self._client.close()
