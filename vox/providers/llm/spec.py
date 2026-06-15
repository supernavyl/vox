"""The "wrap any LLM" seam: turn one string into an :class:`LLMProvider`.

Spec grammar::

    provider:model[@base_url]

Examples::

    ollama:llama3.2
    ollama:qwen3:8b                       # colons in the model are fine
    openai:gpt-4o
    anthropic:claude-opus-4-8
    groq:llama-3.3-70b-versatile
    openai-compat:my-model@http://localhost:8080/v1   # vLLM / llama.cpp / TabbyAPI / LM Studio

`openai-compat` (alias: `openai`) reaches any OpenAI-compatible server, which is
how most local and hosted inference stacks expose themselves — so "any LLM" is
almost always one of: a named provider, or `openai-compat:<model>@<url>`.
"""

from __future__ import annotations

from collections.abc import Callable

from vox.providers.base import LLMProvider

# Well-known OpenAI-compatible endpoints, so users type a short provider name.
_OPENAI_COMPAT_BASE_URLS: dict[str, str] = {
    "groq": "https://api.groq.com/openai/v1",
    "together": "https://api.together.xyz/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "mistral": "https://api.mistral.ai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}


def _make_ollama(model: str, base_url: str | None, **opts: object) -> LLMProvider:
    from vox.providers.llm.ollama_llm import OllamaLLM

    return OllamaLLM(model=model, base_url=base_url, **opts)  # type: ignore[arg-type]


def _make_openai(model: str, base_url: str | None, **opts: object) -> LLMProvider:
    from vox.providers.llm.openai_llm import OpenAILLM

    return OpenAILLM(model=model, base_url=base_url, **opts)  # type: ignore[arg-type]


def _make_anthropic(model: str, base_url: str | None, **opts: object) -> LLMProvider:
    from vox.providers.llm.anthropic_llm import AnthropicLLM

    return AnthropicLLM(model=model, base_url=base_url, **opts)  # type: ignore[arg-type]


# provider name -> (constructor, default base_url)
_REGISTRY: dict[str, tuple[Callable[..., LLMProvider], str | None]] = {
    "ollama": (_make_ollama, None),
    "openai": (_make_openai, None),
    "openai-compat": (_make_openai, None),
    "anthropic": (_make_anthropic, None),
    "claude": (_make_anthropic, None),
    **{name: (_make_openai, url) for name, url in _OPENAI_COMPAT_BASE_URLS.items()},
}


def parse_llm_spec(spec: str, **opts: object) -> LLMProvider:
    """Resolve an llm-spec string to a ready :class:`LLMProvider`.

    Extra ``opts`` (e.g. ``api_key``, ``max_tokens``) pass through to the backend.
    Raises ``ValueError`` on an unknown provider or malformed spec.
    """
    spec = spec.strip()
    if not spec:
        raise ValueError("empty llm spec")

    # Split off an optional @base_url first; everything before it is provider:model.
    head, _, base_url = spec.partition("@")
    provider, sep, model = head.partition(":")
    if not sep or not model:
        raise ValueError(f"malformed llm spec {spec!r}; expected 'provider:model[@base_url]'")

    entry = _REGISTRY.get(provider.lower())
    if entry is None:
        known = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"unknown llm provider {provider!r}; known: {known}")

    constructor, default_base_url = entry
    return constructor(model, base_url or default_base_url, **opts)


def available_providers() -> list[str]:
    """Names accepted as the ``provider`` half of an llm spec."""
    return sorted(_REGISTRY)
