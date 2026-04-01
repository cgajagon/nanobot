"""Instrumented LLM provider wrapper for Langfuse generation tracing.

Wraps an ``LLMProvider`` to create Langfuse GENERATION observations
around every ``chat()`` and ``stream_chat()`` call.  Wired at the
composition root (``agent_factory.py``) so all LLM calls — agent loop,
micro-extraction, recovery, delegation — are automatically traced.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from nanobot.observability.langfuse import generation_span
from nanobot.providers.base import LLMProvider, LLMResponse, StreamChunk


def _update_generation(
    obs: Any,
    usage: dict[str, int],
    output: str | None = None,
) -> None:
    """Update a GENERATION observation with usage and optional output."""
    kwargs: dict[str, Any] = {
        "usage_details": {
            "input": usage.get("prompt_tokens", 0),
            "output": usage.get("completion_tokens", 0),
        },
    }
    cache_create = usage.get("cache_creation_input_tokens", 0)
    cache_read = usage.get("cache_read_input_tokens", 0)
    if cache_create:
        kwargs["usage_details"]["cache_creation_input_tokens"] = cache_create
    if cache_read:
        kwargs["usage_details"]["cache_read_input_tokens"] = cache_read
    if output is not None:
        kwargs["output"] = output[:500]
    try:
        obs.update(**kwargs)
    except Exception:  # crash-barrier: usage recording must never break the agent
        pass


class InstrumentedProvider(LLMProvider):
    """Langfuse-instrumented LLM provider wrapper.

    Delegates all calls to an inner ``LLMProvider`` while wrapping them
    in ``generation_span`` context managers that produce GENERATION
    observations with model, usage, and cost data.
    """

    def __init__(self, inner: LLMProvider) -> None:
        # Do not call super().__init__() — we delegate everything to inner.
        self._inner = inner

    @property
    def api_key(self) -> str | None:  # type: ignore[override]
        return self._inner.api_key

    @property
    def api_base(self) -> str | None:  # type: ignore[override]
        return self._inner.api_base

    def get_default_model(self) -> str:
        return self._inner.get_default_model()

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        effective_model = model or self._inner.get_default_model()
        async with generation_span(
            name="llm_generation",
            model=effective_model,
            model_parameters={"temperature": temperature, "max_tokens": max_tokens},
        ) as obs:
            response = await self._inner.chat(
                messages=messages,
                tools=tools,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                metadata=metadata,
            )
            if obs is not None:
                _update_generation(obs, response.usage, response.content)
            return response

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        effective_model = model or self._inner.get_default_model()
        async with generation_span(
            name="llm_generation",
            model=effective_model,
            model_parameters={"temperature": temperature, "max_tokens": max_tokens},
        ) as obs:
            last_usage: dict[str, int] = {}
            try:
                async for chunk in self._inner.stream_chat(
                    messages=messages,
                    tools=tools,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    metadata=metadata,
                ):
                    if chunk.usage:
                        last_usage = chunk.usage
                    yield chunk
            finally:
                if obs is not None and last_usage:
                    _update_generation(obs, last_usage)

    async def aclose(self) -> None:
        await self._inner.aclose()
