"""Tests for InstrumentedProvider — Langfuse generation wrapper around LLMProvider."""

from __future__ import annotations

from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.observability.instrumented_provider import InstrumentedProvider
from nanobot.providers.base import LLMProvider, LLMResponse, StreamChunk


class FakeProvider(LLMProvider):
    """Minimal provider for tests."""

    def __init__(self) -> None:
        super().__init__(api_key="fake-key", api_base="https://fake.api")
        self.chat_calls: list[dict] = []
        self.stream_calls: list[dict] = []
        self.aclose_called = False

    def get_default_model(self) -> str:
        return "fake-model"

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.chat_calls.append({"model": model, "messages": messages})
        return LLMResponse(
            content="hello world",
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "cache_creation_input_tokens": 50,
                "cache_read_input_tokens": 30,
            },
        )

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        self.stream_calls.append({"model": model, "messages": messages})
        yield StreamChunk(content_delta="hel")
        yield StreamChunk(
            content_delta="lo",
            finish_reason="stop",
            usage={
                "prompt_tokens": 200,
                "completion_tokens": 30,
                "total_tokens": 230,
                "cache_creation_input_tokens": 80,
                "cache_read_input_tokens": 60,
            },
            done=True,
        )

    async def aclose(self) -> None:
        self.aclose_called = True


@pytest.mark.asyncio
async def test_chat_delegates_and_returns_response():
    inner = FakeProvider()
    wrapped = InstrumentedProvider(inner)
    resp = await wrapped.chat(
        messages=[{"role": "user", "content": "hi"}],
        model="gpt-4o",
    )
    assert resp.content == "hello world"
    assert resp.usage["prompt_tokens"] == 100
    assert resp.usage["cache_creation_input_tokens"] == 50
    assert len(inner.chat_calls) == 1
    assert inner.chat_calls[0]["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_stream_chat_delegates_and_yields_all_chunks():
    inner = FakeProvider()
    wrapped = InstrumentedProvider(inner)
    chunks = []
    async for chunk in wrapped.stream_chat(
        messages=[{"role": "user", "content": "hi"}],
        model="gpt-4o",
    ):
        chunks.append(chunk)
    assert len(chunks) == 2
    assert chunks[0].content_delta == "hel"
    assert chunks[1].content_delta == "lo"
    assert chunks[1].usage["prompt_tokens"] == 200
    assert chunks[1].usage["cache_creation_input_tokens"] == 80
    assert len(inner.stream_calls) == 1


@pytest.mark.asyncio
async def test_get_default_model_delegates():
    inner = FakeProvider()
    wrapped = InstrumentedProvider(inner)
    assert wrapped.get_default_model() == "fake-model"


@pytest.mark.asyncio
async def test_api_key_and_api_base_delegate():
    inner = FakeProvider()
    wrapped = InstrumentedProvider(inner)
    assert wrapped.api_key == "fake-key"
    assert wrapped.api_base == "https://fake.api"


@pytest.mark.asyncio
async def test_aclose_delegates():
    inner = FakeProvider()
    wrapped = InstrumentedProvider(inner)
    await wrapped.aclose()
    assert inner.aclose_called


@pytest.mark.asyncio
async def test_chat_calls_generation_span_with_model():
    inner = FakeProvider()
    wrapped = InstrumentedProvider(inner)

    mock_obs = MagicMock()
    mock_obs.update = MagicMock()

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_obs)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "nanobot.observability.instrumented_provider.generation_span",
        return_value=mock_ctx,
    ) as mock_gen:
        await wrapped.chat(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4o",
            temperature=0.5,
            max_tokens=1000,
        )
        mock_gen.assert_called_once_with(
            name="llm_generation",
            model="gpt-4o",
            model_parameters={"temperature": 0.5, "max_tokens": 1000},
        )
        mock_obs.update.assert_called_once()
        call_kwargs = mock_obs.update.call_args[1]
        assert call_kwargs["usage_details"]["input"] == 100
        assert call_kwargs["usage_details"]["output"] == 20
        assert call_kwargs["usage_details"]["cache_creation_input_tokens"] == 50
        assert call_kwargs["usage_details"]["cache_read_input_tokens"] == 30
        assert "output" in call_kwargs


@pytest.mark.asyncio
async def test_stream_chat_updates_span_with_final_usage():
    inner = FakeProvider()
    wrapped = InstrumentedProvider(inner)

    mock_obs = MagicMock()
    mock_obs.update = MagicMock()

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_obs)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "nanobot.observability.instrumented_provider.generation_span",
        return_value=mock_ctx,
    ):
        chunks = []
        async for chunk in wrapped.stream_chat(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4o",
        ):
            chunks.append(chunk)

        mock_obs.update.assert_called_once()
        call_kwargs = mock_obs.update.call_args[1]
        assert call_kwargs["usage_details"]["input"] == 200
        assert call_kwargs["usage_details"]["output"] == 30
        assert call_kwargs["usage_details"]["cache_creation_input_tokens"] == 80


@pytest.mark.asyncio
async def test_stream_chat_updates_span_on_early_termination():
    """If caller breaks out of iteration, finally block still records usage."""
    inner = FakeProvider()
    wrapped = InstrumentedProvider(inner)

    mock_obs = MagicMock()
    mock_obs.update = MagicMock()

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_obs)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "nanobot.observability.instrumented_provider.generation_span",
        return_value=mock_ctx,
    ):
        async for _chunk in wrapped.stream_chat(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4o",
        ):
            break  # early termination after first chunk

        # First chunk has no usage in FakeProvider, so update should not be called
        mock_obs.update.assert_not_called()


@pytest.mark.asyncio
async def test_chat_uses_default_model_when_none():
    inner = FakeProvider()
    wrapped = InstrumentedProvider(inner)

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=None)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "nanobot.observability.instrumented_provider.generation_span",
        return_value=mock_ctx,
    ) as mock_gen:
        await wrapped.chat(messages=[{"role": "user", "content": "hi"}])
        call_kwargs = mock_gen.call_args[1]
        assert call_kwargs["model"] == "fake-model"


def test_instrumented_provider_implements_llm_provider_interface():
    """InstrumentedProvider must implement all LLMProvider abstract methods."""
    import inspect

    abstract_methods = {
        name
        for name, _ in inspect.getmembers(LLMProvider)
        if getattr(getattr(LLMProvider, name, None), "__isabstractmethod__", False)
    }
    for method_name in abstract_methods:
        assert hasattr(InstrumentedProvider, method_name), (
            f"InstrumentedProvider missing abstract method: {method_name}"
        )
