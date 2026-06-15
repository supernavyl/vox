"""Ollama LLM backend — async streaming via the official ``ollama`` SDK.

Targets a local Ollama daemon (default ``http://localhost:11434``); pass
``base_url`` to reach a remote host. The system prompt is carried as a leading
``system`` message, which is how Ollama's chat endpoint expects it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from vox.providers.base import LLMProvider
from vox.types import Message


class OllamaLLM(LLMProvider):
    """Streams responses from an Ollama daemon through ``AsyncClient.chat``."""

    def __init__(self, model: str, *, base_url: str | None = None, **_: object) -> None:
        """Build the backend, lazily importing ``ollama`` so the module stays import-clean.

        ``base_url`` overrides the daemon host (e.g. ``http://gpu-box:11434``);
        ``None`` lets the SDK resolve the default localhost endpoint. Any extra
        keyword options are accepted and ignored for spec-factory passthrough.
        """
        from ollama import AsyncClient

        self.model = model
        self._client = AsyncClient(host=base_url) if base_url else AsyncClient()

    async def stream(
        self, messages: Sequence[Message], *, system: str | None = None
    ) -> AsyncIterator[str]:
        """Yield assistant text chunks for ``messages``, prepending ``system`` as a message."""
        msgs: list[dict[str, str]] = [{"role": "system", "content": system}] if system else []
        msgs += [m.as_dict() for m in messages]

        resp = await self._client.chat(model=self.model, messages=msgs, stream=True)
        async for part in resp:
            chunk = part["message"]["content"]
            if chunk:
                yield chunk

    async def aclose(self) -> None:
        """Best-effort release of the underlying httpx client. Safe to call repeatedly."""
        try:
            await self._client._client.aclose()
        except Exception:  # noqa: BLE001 - cleanup is best-effort across SDK versions
            pass
