# Langfuse InstrumentedProvider — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the 99.6% token/cost tracking gap by replacing litellm's broken OTEL auto-instrumentation with an `InstrumentedProvider` wrapper that creates manual Langfuse GENERATION observations.

**Architecture:** `InstrumentedProvider` extends `LLMProvider`, delegates all calls to an inner provider, and wraps `chat()`/`stream_chat()` in `generation_span` context managers. Wired at the composition root (`agent_factory.py`) before any subsystem receives the provider. Also adds `retriever_span` around memory retrieval, removes litellm OTEL workarounds, and indexes batch tool metadata.

**Tech Stack:** Langfuse v4.0.1 Python SDK (OTEL-based), litellm 1.82.0, Python 3.10+

**Design Spec:** `docs/superpowers/specs/2026-04-01-langfuse-instrumented-provider-design.md`

---

## File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `nanobot/observability/instrumented_provider.py` | LLMProvider wrapper creating Langfuse GENERATION spans |
| Modify | `nanobot/observability/langfuse.py` | Remove litellm OTEL callback; add `generation_span` + `retriever_span` |
| Modify | `nanobot/observability/__init__.py` | Export `InstrumentedProvider` |
| Modify | `nanobot/agent/agent_factory.py` | Wire `InstrumentedProvider` at step 4.5 |
| Modify | `nanobot/memory/read/retriever.py` | Add `retriever_span` around `retrieve()` |
| Modify | `nanobot/agent/turn_runner.py` | Index batch metadata keys by iteration |
| Modify | `.claude/rules/architecture.md` | Document `observability/` → `providers/base.py` import |
| Create | `tests/test_instrumented_provider.py` | Tests for InstrumentedProvider |
| Create | `tests/test_langfuse_helpers.py` | Tests for generation_span/retriever_span |

---

### Task 1: Remove litellm OTEL callback and `_EndedSpanFilter` from `langfuse.py`

This goes first because it frees LOC headroom and eliminates the broken instrumentation
before the replacement is wired.

**Files:**
- Modify: `nanobot/observability/langfuse.py:96-165`

- [ ] **Step 1: Remove the litellm OTEL callback block (lines 96–132)**

In `nanobot/observability/langfuse.py`, replace lines 96–132 (the entire `try` block that
registers `"otel"` on litellm callbacks and monkey-patches `_maybe_log_raw_request`) with:

```python
        # NOTE: litellm's "otel" callback is NOT registered.
        # InstrumentedProvider (observability/instrumented_provider.py) handles
        # all LLM call tracing via manual Langfuse GENERATION observations.
        # The litellm OTEL callback had a bug where async streaming dispatched
        # to litellm._async_success_callback (empty), causing 0 tokens/cost.
```

- [ ] **Step 2: Remove the `_EndedSpanFilter` class and registration (lines 154–162)**

In the log-filter `try` block (starting at line 134), remove only the `_EndedSpanFilter`
class definition and its registration. Keep `_ProxyFilter` and `_SpanCtxFilter`.

The block starting at line 134 should become:

```python
        # Suppress benign warnings from litellm/langfuse loggers.
        try:
            import logging

            # litellm warns "Proxy Server is not installed" on first LLM call
            # when the optional proxy package is absent.
            class _ProxyFilter(logging.Filter):
                def filter(self, record: logging.LogRecord) -> bool:
                    return "Proxy Server is not installed" not in record.getMessage()

            logging.getLogger("LiteLLM").addFilter(_ProxyFilter())

            # Langfuse may warn "No active span in current context" briefly
            # before the trace_request context manager is entered.
            class _SpanCtxFilter(logging.Filter):
                def filter(self, record: logging.LogRecord) -> bool:
                    return "No active span in current context" not in record.getMessage()

            logging.getLogger("langfuse").addFilter(_SpanCtxFilter())
        except Exception as exc:  # crash-barrier: filter setup is optional
            logger.debug("Log filter setup failed: {}", exc)
```

- [ ] **Step 3: Run lint and typecheck**

Run: `make lint && make typecheck`
Expected: PASS

- [ ] **Step 4: Run tests**

Run: `make test`
Expected: PASS — no existing tests depend on litellm OTEL callback registration.

- [ ] **Step 5: Commit**

```bash
git add nanobot/observability/langfuse.py
git commit -m "fix(observability): remove broken litellm OTEL callback and workarounds

litellm 1.82.0 dispatches async streaming callbacks to
litellm._async_success_callback (empty) while 'otel' was registered
only on litellm.success_callback. This caused 99.6% of tokens/cost
to be invisible in Langfuse. InstrumentedProvider (next commit) will
handle all LLM tracing. Also removes _EndedSpanFilter and the
_maybe_log_raw_request monkey-patch, both no longer needed."
```

---

### Task 2: Add `generation_span` and `retriever_span` helpers to `langfuse.py`

**Files:**
- Modify: `nanobot/observability/langfuse.py` (append after `span()` at end of file)
- Create: `tests/test_langfuse_helpers.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_langfuse_helpers.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_langfuse_helpers.py -v`
Expected: FAIL with `ImportError: cannot import name 'generation_span'`

- [ ] **Step 3: Implement `generation_span` and `retriever_span`**

Append to the end of `nanobot/observability/langfuse.py` (after the `span()` function):

```python


@contextlib.asynccontextmanager
async def generation_span(
    *,
    name: str,
    model: str | None = None,
    model_parameters: dict[str, Any] | None = None,
) -> AsyncIterator[Any]:
    """Create a Langfuse GENERATION observation for an LLM call.

    Yields the observation object (or ``None`` when disabled).
    The caller should call ``obs.update(output=..., usage_details=...)``
    before exiting to record the response and token counts.

    Unlike ``tool_span`` and ``span``, this creates a GENERATION-type
    observation that Langfuse renders with model/token/cost tracking.
    """
    if not _enabled or _client is None:
        yield None
        return

    try:
        kwargs: dict[str, Any] = {"name": name, "as_type": "generation"}
        if model is not None:
            kwargs["model"] = model
        if model_parameters is not None:
            kwargs["model_parameters"] = model_parameters
        with _client.start_as_current_observation(**kwargs) as obs:
            yield obs
    except Exception:  # crash-barrier: tracing must never break the agent
        logger.opt(exception=True).warning("Langfuse generation_span failed")
        yield None


@contextlib.asynccontextmanager
async def retriever_span(
    *,
    name: str,
    input: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> AsyncIterator[Any]:
    """Create a Langfuse RETRIEVER observation for a RAG retrieval operation.

    Yields the observation object (or ``None`` when disabled).
    The caller should call ``obs.update(output=..., metadata=...)``
    before exiting.
    """
    if not _enabled or _client is None:
        yield None
        return

    try:
        with _client.start_as_current_observation(
            name=name,
            as_type="retriever",
            input=input,
            metadata=metadata,
        ) as obs:
            yield obs
    except Exception:  # crash-barrier: tracing must never break the agent
        logger.opt(exception=True).warning("Langfuse retriever_span failed")
        yield None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_langfuse_helpers.py -v`
Expected: PASS (all 4 tests — Langfuse is disabled in test env so yields None)

- [ ] **Step 5: Run lint and typecheck**

Run: `make lint && make typecheck`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add nanobot/observability/langfuse.py tests/test_langfuse_helpers.py
git commit -m "feat(observability): add generation_span and retriever_span helpers

Langfuse v4 GENERATION-type observations need start_as_current_observation
with as_type='generation'. Add dedicated helpers consistent with existing
tool_span and span pattern. retriever_span uses as_type='retriever' for
RAG retrieval tracing. Both are no-ops when Langfuse is disabled."
```

---

### Task 3: Create `InstrumentedProvider` wrapper

**Files:**
- Create: `nanobot/observability/instrumented_provider.py`
- Modify: `nanobot/observability/__init__.py`
- Create: `tests/test_instrumented_provider.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_instrumented_provider.py`:

```python
"""Tests for InstrumentedProvider — Langfuse generation wrapper around LLMProvider."""
from __future__ import annotations

from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.observability.instrumented_provider import InstrumentedProvider
from nanobot.providers.base import LLMProvider, LLMResponse, StreamChunk, ToolCallRequest


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


# -- Delegation tests (Langfuse disabled — generation_span yields None) --


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


# -- Span creation tests (mock generation_span to verify it's called) --


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
        # Verify obs.update was called with usage including cache metrics
        mock_obs.update.assert_called_once()
        call_kwargs = mock_obs.update.call_args[1]
        assert call_kwargs["usage_details"]["input"] == 100
        assert call_kwargs["usage_details"]["output"] == 20
        assert call_kwargs["usage_details"]["cache_creation_input_tokens"] == 50
        assert call_kwargs["usage_details"]["cache_read_input_tokens"] == 30
        assert "output" in call_kwargs  # content truncated


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

        # Verify obs.update was called with usage from final chunk
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
        async for chunk in wrapped.stream_chat(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4o",
        ):
            break  # early termination after first chunk

        # obs.update should NOT be called — first chunk has no usage
        # (usage only comes on final chunk in FakeProvider)
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


# -- Contract test --


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_instrumented_provider.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nanobot.observability.instrumented_provider'`

- [ ] **Step 3: Implement `InstrumentedProvider`**

Create `nanobot/observability/instrumented_provider.py`:

```python
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
```

- [ ] **Step 4: Update `__init__.py` exports**

Replace the contents of `nanobot/observability/__init__.py` with:

```python
"""Observability: tracing, instrumentation, and metrics."""

from __future__ import annotations

__all__: list[str] = ["InstrumentedProvider"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_instrumented_provider.py -v`
Expected: PASS (all 11 tests)

- [ ] **Step 6: Run lint and typecheck**

Run: `make lint && make typecheck`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add nanobot/observability/instrumented_provider.py nanobot/observability/__init__.py tests/test_instrumented_provider.py
git commit -m "feat(observability): add InstrumentedProvider for Langfuse generation tracing

Wraps LLMProvider to create GENERATION observations around every chat()
and stream_chat() call using Langfuse v4 start_as_current_observation.
Captures model, usage_details (prompt/completion/cache tokens), and
output. Uses try/finally in stream_chat to handle early termination.
Includes contract test for LLMProvider interface completeness."
```

---

### Task 4: Wire `InstrumentedProvider` in `agent_factory.py`

**Files:**
- Modify: `nanobot/agent/agent_factory.py:278-279`

- [ ] **Step 1: Add InstrumentedProvider wrapping at step 4.5**

In `nanobot/agent/agent_factory.py`, insert between line 277 (end of ContextBuilder
construction) and line 279 (`# 5. Construct SessionManager`):

```python

    # 4.5. Wrap provider with Langfuse generation tracing
    from nanobot.observability.instrumented_provider import InstrumentedProvider

    provider = InstrumentedProvider(provider)

```

The resulting code around the insertion point should read:

```python
    # 4. Construct ContextBuilder
    context = ContextBuilder(
        config.workspace_path,
        memory=memory,
        memory_config=config.memory if config.memory_enabled else None,
        strategy_store=strategy_store,
    )

    # 4.5. Wrap provider with Langfuse generation tracing
    from nanobot.observability.instrumented_provider import InstrumentedProvider

    provider = InstrumentedProvider(provider)

    # 5. Construct SessionManager
    sessions = session_manager or _SessionManager(config.workspace_path)
```

- [ ] **Step 2: Run existing tests to verify nothing breaks**

Run: `make test`
Expected: PASS — InstrumentedProvider is a transparent wrapper. All existing tests
that use providers via mocks or FakeProvider are unaffected because the wrapper
delegates all calls.

- [ ] **Step 3: Run lint and typecheck**

Run: `make lint && make typecheck`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add nanobot/agent/agent_factory.py
git commit -m "feat(agent): wire InstrumentedProvider at composition root

Wrap the raw LLM provider with InstrumentedProvider in build_agent()
at step 4.5 — before _build_tools (step 6) and all other subsystems.
This ensures all LLM calls (agent loop, micro-extraction, strategy
extraction, consolidation, tool caching) produce Langfuse GENERATION
observations with model/token/cost data."
```

---

### Task 5: Add retriever span to memory retrieval

**Files:**
- Modify: `nanobot/memory/read/retriever.py:57-74`

- [ ] **Step 1: Wrap `retrieve()` body in `retriever_span`**

Replace the `retrieve()` method (lines 57–74) in `nanobot/memory/read/retriever.py` with:

```python
    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 6,
    ) -> list[RetrievedMemory]:
        from nanobot.observability.langfuse import retriever_span

        self._graph_aug.reset_cache()
        t0 = time.monotonic()

        async with retriever_span(
            name="memory_retrieve",
            input={"query": query, "top_k": top_k},
        ) as obs:
            # Unified path: vector + FTS5 + RRF when db and embedder are injected
            if self._db is not None and self._embedder is not None:
                results = await self._retrieve_unified(
                    query,
                    top_k=top_k,
                    t0=t0,
                )
            else:
                results = []

            if obs is not None:
                obs.update(
                    output=f"{len(results)} results",
                    metadata={
                        "result_count": len(results),
                        "duration_ms": round((time.monotonic() - t0) * 1000),
                    },
                )

            return results
```

- [ ] **Step 2: Run tests**

Run: `make test`
Expected: PASS

- [ ] **Step 3: Run lint and typecheck**

Run: `make lint && make typecheck`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add nanobot/memory/read/retriever.py
git commit -m "feat(memory): add Langfuse retriever span to memory retrieval

Memory retrieval (vector+FTS5+RRF) was invisible in Langfuse traces —
a 1-2s gap between trace start and first LLM call. Now wrapped in a
retriever_span with query, result count, and duration metadata."
```

---

### Task 6: Index batch tool metadata keys in `turn_runner.py`

**Files:**
- Modify: `nanobot/agent/turn_runner.py:496-502`

- [ ] **Step 1: Change flat keys to iteration-indexed keys**

In `nanobot/agent/turn_runner.py`, replace lines 496–502:

```python
        update_current_span(
            metadata={
                "batch_tools": [tc.name for tc in response.tool_calls],
                "batch_any_failed": any(not a.success for a in latest_attempts),
                "batch_duration_ms": round(elapsed_ms),
            }
        )
```

With:

```python
        update_current_span(
            metadata={
                f"batch_{state.iteration}_tools": [tc.name for tc in response.tool_calls],
                f"batch_{state.iteration}_failed": any(
                    not a.success for a in latest_attempts
                ),
                f"batch_{state.iteration}_ms": round(elapsed_ms),
            }
        )
```

- [ ] **Step 2: Run tests**

Run: `make test`
Expected: PASS

- [ ] **Step 3: Run lint and typecheck**

Run: `make lint && make typecheck`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add nanobot/agent/turn_runner.py
git commit -m "fix(observability): use indexed keys for batch tool metadata

Each tool batch call to update_current_span overwrote the previous
batch's data. Use batch_{iteration}_tools/failed/ms keys so all
batches are visible in the Langfuse trace metadata."
```

---

### Task 7: Update architecture documentation

**Files:**
- Modify: `.claude/rules/architecture.md:61`

- [ ] **Step 1: Update Import DAG**

In `.claude/rules/architecture.md`, change line 61 from:

```
providers/base.py              ← imported by agent/ (via Protocol)
```

To:

```
providers/base.py              ← imported by agent/ (via Protocol), observability/
```

- [ ] **Step 2: Run doc check**

Run: `make check`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add .claude/rules/architecture.md
git commit -m "docs(architecture): add observability → providers/base.py to Import DAG

InstrumentedProvider in observability/ imports LLMProvider, LLMResponse,
and StreamChunk from providers/base.py. This import is allowed by
check_imports.py but was not reflected in the architecture doc."
```

---

### Task 8: Full validation

- [ ] **Step 1: Run full check suite**

Run: `make check`
Expected: PASS (lint + typecheck + import-check + structure-check + prompt-check + phase-todo-check + doc-check)

- [ ] **Step 2: Run tests with coverage**

Run: `make test-cov`
Expected: PASS with coverage ≥ 85%

- [ ] **Step 3: Verify import boundaries**

Run: `make import-check`
Expected: PASS — `InstrumentedProvider` is in `observability/` (allowed to import
from `providers/`). `agent_factory.py` imports from `observability/` (allowed).

- [ ] **Step 4: Verify structure limits**

Run: `make structure-check`
Expected: PASS — `observability/` has 5 files (≤ 15), `__init__.py` has 1 export (≤ 12),
`instrumented_provider.py` is ~60 LOC (≤ 500).

- [ ] **Step 5: Verify no stale references**

Run:
```bash
grep -rn "litellm.success_callback\|_maybe_log_raw_request\|_EndedSpanFilter" nanobot/ --include="*.py"
```
Expected: zero matches.

- [ ] **Step 6: Commit any final fixes if needed**
