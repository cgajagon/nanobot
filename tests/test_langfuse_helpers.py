"""Tests for Langfuse span helper context managers."""

from __future__ import annotations

import pytest

from nanobot.observability.langfuse import generation_span, retriever_span


@pytest.mark.asyncio
async def test_generation_span_yields_none_when_disabled():
    """generation_span is a no-op when Langfuse is disabled."""
    async with generation_span(name="test", model="gpt-4o") as obs:
        assert obs is None


@pytest.mark.asyncio
async def test_generation_span_accepts_model_parameters():
    """generation_span accepts model_parameters without error."""
    async with generation_span(
        name="test",
        model="gpt-4o",
        model_parameters={"temperature": 0.7, "max_tokens": 1000},
    ) as obs:
        assert obs is None


@pytest.mark.asyncio
async def test_retriever_span_yields_none_when_disabled():
    """retriever_span is a no-op when Langfuse is disabled."""
    async with retriever_span(name="test") as obs:
        assert obs is None


@pytest.mark.asyncio
async def test_retriever_span_accepts_input_and_metadata():
    """retriever_span accepts input and metadata without error."""
    async with retriever_span(
        name="test",
        input={"query": "hello", "top_k": 5},
        metadata={"source": "test"},
    ) as obs:
        assert obs is None
