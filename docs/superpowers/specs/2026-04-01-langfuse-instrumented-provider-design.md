# Langfuse InstrumentedProvider — Design Spec

> Replace litellm's broken auto-instrumentation with a provider wrapper that
> creates manual GENERATION observations using Langfuse v4's
> `start_as_current_observation(as_type="generation")` API.

**Date:** 2026-04-01
**Status:** Draft
**Supersedes:** `docs/superpowers/plans/2026-03-28-langfuse-streaming-tracing.md`
(original plan — assumptions outdated by code changes since March 28)

---

## Problem

litellm 1.82.0 registers OTEL callbacks on `litellm.success_callback` (sync).
Async streaming dispatches to `litellm._async_success_callback` (empty). The sync
fallback runs in a thread with no `standard_logging_object`, so `set_attributes`
fails silently. Result: every streaming LLM call produces a Langfuse GENERATION
observation with 0 tokens, 0 cost, no model name.

Since the agent uses streaming for all main-loop calls, **99.6% of cost data is
invisible** in Langfuse.

Additional gaps:
- Memory retrieval (1–2s per turn) is invisible in traces.
- Batch tool metadata keys overwrite each iteration (flat `batch_tools` key).

## Solution: InstrumentedProvider Wrapper

Wrap `LLMProvider` with an `InstrumentedProvider` in `observability/` that creates
Langfuse GENERATION observations around every `chat()` and `stream_chat()` call.
Wired at the composition root (`agent_factory.py`) so all LLM calls — agent loop,
micro-extraction, strategy extraction, consolidation — are traced.

Simultaneously remove litellm's broken `"otel"` callback and associated workarounds.

### Why This Approach (Approach A — Provider Wrapper)

Five approaches were evaluated. See the analysis conversation for full details.

| Approach | Verdict | Reason |
|----------|---------|--------|
| **A. InstrumentedProvider** | **Chosen** | Covers all call sites, clean separation, composition root pattern |
| B. Instrument StreamingLLMCaller | Rejected | Misses MicroExtractor, StrategyExtractor, consolidation calls |
| C. Fix litellm callback registration | Rejected | Race condition in litellm internals, fragile to upgrades |
| D. Langfuse @observe decorator | Rejected | Doesn't handle streaming (captures generator object, not chunks) |
| E. Custom litellm callback class | Rejected | Complex, coupled to litellm internals, same race condition |

---

## Architecture Compliance

Verified against all rule documents. Full audit in conversation context.

| Rule | Check | Status |
|------|-------|--------|
| Import direction (`check_imports.py`) | `observability/` → `providers.base`: not forbidden | PASS |
| Composition root | `InstrumentedProvider()` constructed in `agent_factory.py` | PASS |
| No cross-package instantiation | `agent_factory.py` is COMPOSITION_ROOT (exempt) | PASS |
| Single ownership | `InstrumentedProvider` is pure instrumentation, no domain logic | PASS |
| File size (500 LOC hard) | `langfuse.py`: 391→~370, `instrumented_provider.py`: ~60, all under | PASS |
| Package file count (15) | `observability/`: 4→5 files | PASS |
| `__init__.py` exports (12) | 0→1 export | PASS |
| Prohibited patterns | No catch-alls, no shims, no domain logic in loop | PASS |
| Import DAG documentation | **Gap**: `observability/` → `providers/base.py` not documented | Fix included |

---

## Deviations from Original Plan (2026-03-28)

The original plan was written against code as of March 28. Significant work has
merged since. These deviations were identified during deep code analysis:

### 1. langfuse.py LOC — Not a Problem

**Plan assumed:** 477 LOC, near 500 limit, needs separate file for helpers.
**Reality:** `check_structure.py` counts code-only LOC = 391. Removing OTEL callback
(~50 code lines) and adding helpers (~30 code lines) lands at ~370. Well under 500.

### 2. Cache Metrics Already Captured

**Plan assumed:** `_parse_response()` only captures `prompt_tokens`/`completion_tokens`.
**Reality:** Both `_parse_response()` (line 365–370) and the streaming path (line 481–486)
already capture `cache_creation_input_tokens` and `cache_read_input_tokens`. `TurnResult`
propagates them. `MessageProcessor` logs them.
**Implication:** `InstrumentedProvider` must forward the **full** usage dict to Langfuse,
not just prompt/completion tokens.

### 3. retriever.py Return Type Changed

**Plan assumed:** `retrieve()` returns `list[dict]`.
**Reality:** Returns `list[RetrievedMemory]` (typed dataclass).
**Implication:** Span metadata uses `len(results)` which works for both. No code impact.

### 4. turn_runner.py LOC — Not a Blocker

**Plan assumed:** Might be over 500 LOC limit.
**Reality:** 559 code-only LOC — advisory warning only, not hard gate. Task 6 is a
net-zero change (rename 3 key strings). No extraction needed.

---

## File Changes

| Action | Path | Responsibility | LOC delta |
|--------|------|---------------|-----------|
| Modify | `nanobot/observability/langfuse.py` | Remove litellm OTEL callback + workarounds; add `generation_span` and `retriever_span` | ~-20 net |
| Create | `nanobot/observability/instrumented_provider.py` | LLMProvider wrapper creating GENERATION observations | ~60 |
| Modify | `nanobot/observability/__init__.py` | Export `InstrumentedProvider` | +1 line |
| Modify | `nanobot/agent/agent_factory.py` | Wire `InstrumentedProvider` around provider | +3 lines |
| Modify | `nanobot/memory/read/retriever.py` | Add `retriever_span` around retrieval | +12 lines |
| Modify | `nanobot/agent/turn_runner.py` | Index batch metadata keys by iteration | +3/-3 |
| Modify | `.claude/rules/architecture.md` | Add `observability/` → `providers/base.py` to Import DAG | +1 line |
| Create | `tests/test_instrumented_provider.py` | Unit tests for wrapper | ~120 |
| Create | `tests/test_langfuse_helpers.py` | Unit tests for generation_span/retriever_span | ~30 |

---

## Detailed Design

### 1. `generation_span` and `retriever_span` helpers

Added to `langfuse.py` after the existing `span()` context manager (line 477).
Follow the identical pattern of `tool_span` and `span`: async context manager,
no-op when disabled, crash-barrier on exceptions.

```python
@contextlib.asynccontextmanager
async def generation_span(
    *,
    name: str,
    model: str | None = None,
    model_parameters: dict[str, Any] | None = None,
) -> AsyncIterator[Any]:
    """Create a Langfuse GENERATION observation for an LLM call.

    Yields the observation object (or None when disabled).
    The caller should call obs.update(output=..., usage_details=...)
    before exiting to record the response and token counts.
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
    """Create a Langfuse RETRIEVER observation for a RAG retrieval operation."""
    if not _enabled or _client is None:
        yield None
        return

    try:
        with _client.start_as_current_observation(
            name=name, as_type="retriever", input=input, metadata=metadata,
        ) as obs:
            yield obs
    except Exception:  # crash-barrier: tracing must never break the agent
        logger.opt(exception=True).warning("Langfuse retriever_span failed")
        yield None
```

**Design notes:**
- `generation_span` does NOT accept `input` — message arrays are large and would
  bloat Langfuse storage. The observation captures model, parameters, and usage only.
- `retriever_span` accepts `input` (the query string) and `metadata` — these are
  small and useful for debugging retrieval quality.
- Both follow the existing `tool_span`/`span` pattern exactly.

### 2. `InstrumentedProvider` wrapper

New file: `observability/instrumented_provider.py`.

```python
class InstrumentedProvider(LLMProvider):
    """Langfuse-instrumented LLM provider wrapper.

    Delegates all calls to an inner LLMProvider while wrapping them in
    generation_span context managers that produce GENERATION observations
    with model, usage, and cost data.
    """

    def __init__(self, inner: LLMProvider) -> None:
        # Skip super().__init__() — we delegate everything to inner.
        # Store api_key/api_base as properties delegating to inner so that
        # any code reading provider.api_key gets the real value.
        self._inner = inner

    @property
    def api_key(self) -> str | None:
        return self._inner.api_key

    @property
    def api_base(self) -> str | None:
        return self._inner.api_base

    def get_default_model(self) -> str:
        return self._inner.get_default_model()

    async def chat(self, messages, tools=None, model=None,
                   max_tokens=4096, temperature=0.7, metadata=None):
        effective_model = model or self._inner.get_default_model()
        async with generation_span(
            name="llm_generation",
            model=effective_model,
            model_parameters={"temperature": temperature, "max_tokens": max_tokens},
        ) as obs:
            response = await self._inner.chat(
                messages=messages, tools=tools, model=model,
                max_tokens=max_tokens, temperature=temperature, metadata=metadata,
            )
            if obs is not None:
                _update_generation(obs, response.usage, response.content)
            return response

    async def stream_chat(self, messages, tools=None, model=None,
                          max_tokens=4096, temperature=0.7, metadata=None):
        effective_model = model or self._inner.get_default_model()
        async with generation_span(
            name="llm_generation",
            model=effective_model,
            model_parameters={"temperature": temperature, "max_tokens": max_tokens},
        ) as obs:
            last_usage: dict[str, int] = {}
            try:
                async for chunk in self._inner.stream_chat(
                    messages=messages, tools=tools, model=model,
                    max_tokens=max_tokens, temperature=temperature,
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

**Shared helper (module-level):**

```python
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
    # Forward cache metrics when present
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
```

**Design decisions:**

- **`super().__init__()` skipped.** `LLMProvider.__init__` sets `self.api_key` and
  `self.api_base` as instance attributes. We delegate via properties instead so
  callers reading `provider.api_key` get the inner provider's value.
- **`try/finally` in `stream_chat`.** If the caller breaks out of iteration early
  (e.g., error handling), the finally block ensures usage is still recorded on the
  generation span. Without this, early termination would produce a span with 0 usage.
- **`_update_generation` as module-level helper.** Shared between `chat()` and
  `stream_chat()` to avoid duplication. Not a method on the class because it only
  needs the observation and usage dict, not `self`.
- **Cache metrics forwarded.** `cache_creation_input_tokens` and
  `cache_read_input_tokens` are included in `usage_details` when non-zero. These
  are already captured by `LiteLLMProvider` in both `_parse_response()` and the
  streaming path.
- **Output truncated to 500 chars.** For `chat()` only. Streaming doesn't accumulate
  full content (chunks are yielded individually), so output is omitted there. This
  matches the plan's approach and avoids bloating Langfuse storage.

### 3. Wiring in `agent_factory.py`

In `build_agent()`, wrap the provider **before any subsystem receives it**. Insert
between step 4 (ContextBuilder, line 277) and step 5 (SessionManager, line 280):

```python
    # 4.5. Wrap provider with Langfuse generation tracing
    from nanobot.observability.instrumented_provider import InstrumentedProvider

    provider = InstrumentedProvider(provider)
```

**Placement is critical.** The provider is passed to `_build_tools()` at step 6 —
which forwards it to `MissionManager`, `ToolResultCache`, and tool registration.
Later it's passed to `StreamingLLMCaller` (step 9), `MicroExtractor` (step 13),
`StrategyExtractor` (step 13.2), and `MessageProcessor` (step 13.5). Wrapping at
step 4.5 ensures ALL subsystems receive the instrumented provider.

### 4. Remove litellm OTEL callback from `langfuse.py`

In `init_langfuse()`, remove the block from line 96 to line 132 that:
1. Appends `"otel"` to `litellm.success_callback` and `litellm.failure_callback`
2. Sets `USE_OTEL_LITELLM_REQUEST_SPAN` env var
3. Monkey-patches `_OtelCls._maybe_log_raw_request`

Also remove the `_EndedSpanFilter` class and its registration (lines 157–162).
These warnings were caused by litellm's broken streaming spans.

Keep `_ProxyFilter` and `_SpanCtxFilter` — they suppress unrelated benign warnings
that still occur.

Replace with a comment:

```python
        # NOTE: litellm's "otel" callback is NOT registered.
        # InstrumentedProvider handles all LLM call tracing via manual
        # Langfuse GENERATION observations. The litellm OTEL callback
        # had a bug where async streaming dispatched to
        # litellm._async_success_callback (empty), causing 0 tokens/cost.
```

### 5. Retriever span in `memory/read/retriever.py`

Wrap `retrieve()` body in a `retriever_span`:

```python
async def retrieve(self, query: str, *, top_k: int = 6) -> list[RetrievedMemory]:
    from nanobot.observability.langfuse import retriever_span

    self._graph_aug.reset_cache()
    t0 = time.monotonic()

    async with retriever_span(
        name="memory_retrieve",
        input={"query": query, "top_k": top_k},
    ) as obs:
        if self._db is not None and self._embedder is not None:
            results = await self._retrieve_unified(query, top_k=top_k, t0=t0)
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

**Import is deferred** (inside method body) to avoid adding `observability` to the
module-level imports of `memory/read/retriever.py`. This follows the existing pattern
used by `retriever.py` for other optional imports.

### 6. Indexed batch metadata in `turn_runner.py`

Change flat keys to iteration-indexed keys (line 496–502):

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

Net-zero LOC change.

### 7. Architecture documentation update

In `.claude/rules/architecture.md`, update the Import DAG (line 61):

```
providers/base.py              ← imported by agent/ (via Protocol), observability/
```

This documents the new dependency from `observability/instrumented_provider.py` to
`providers/base.py`, which is allowed by `check_imports.py` but was not reflected
in the architecture doc.

---

## Testing Strategy

### `tests/test_langfuse_helpers.py` (~30 LOC)

Test no-op behavior when Langfuse is disabled (the default in test env):

```python
@pytest.mark.asyncio
async def test_generation_span_yields_none_when_disabled():
    async with generation_span(name="test", model="gpt-4o") as obs:
        assert obs is None

@pytest.mark.asyncio
async def test_retriever_span_yields_none_when_disabled():
    async with retriever_span(name="test") as obs:
        assert obs is None
```

### `tests/test_instrumented_provider.py` (~120 LOC)

Uses a `FakeProvider` (in-memory, no network) to verify:

1. **`chat()` delegates and returns response** — inner provider called,
   response passed through unchanged including full usage dict.
2. **`stream_chat()` delegates and yields all chunks** — all chunks forwarded,
   usage from final chunk preserved.
3. **`chat()` creates generation_span when Langfuse enabled** — mock
   `generation_span`, verify called with correct model and parameters.
4. **`stream_chat()` creates generation_span and updates usage** — mock
   `generation_span`, verify `obs.update()` called with `usage_details`
   including cache metrics.
5. **`stream_chat()` updates span on early termination** — break after first
   chunk, verify `obs.update()` still called via finally block.
6. **`get_default_model()` delegates** — returns inner provider's model.
7. **`api_key`/`api_base` properties delegate** — return inner values.
8. **`aclose()` delegates** — inner provider's `aclose()` called.

### Contract test: interface completeness

Verify `InstrumentedProvider` implements all `LLMProvider` abstract methods:

```python
def test_instrumented_provider_implements_llm_provider_interface():
    """InstrumentedProvider must implement all LLMProvider abstract methods."""
    import inspect
    from nanobot.providers.base import LLMProvider
    from nanobot.observability.instrumented_provider import InstrumentedProvider

    abstract_methods = {
        name for name, _ in inspect.getmembers(LLMProvider)
        if getattr(getattr(LLMProvider, name, None), "__isabstractmethod__", False)
    }
    for method_name in abstract_methods:
        assert hasattr(InstrumentedProvider, method_name), (
            f"InstrumentedProvider missing abstract method: {method_name}"
        )
```

This test catches future `LLMProvider` interface drift — if a new abstract method
is added and `InstrumentedProvider` doesn't implement it, this test fails.

---

## What This Does NOT Change

- **litellm itself** — no patches, no custom callbacks, no reliance on litellm internals.
- **LiteLLMProvider** — no modifications to the concrete provider.
- **StreamingLLMCaller** — no modifications. It receives the wrapped provider transparently.
- **Rate limiter** — unaffected. It's injected into StreamingLLMCaller, not the provider.
- **Cache control** — `_apply_cache_control()` runs inside `LiteLLMProvider`, which is
  the inner provider. Cache metrics flow up through `usage` dicts as before.
- **MicroExtractor/StrategyExtractor** — no code changes. They receive the wrapped
  provider from `build_agent()` and their LLM calls are automatically traced.

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Langfuse `usage_details` format changes | Low | Medium | Crash-barrier in `_update_generation` prevents breakage |
| LLMProvider adds new abstract method | Low | Low | Contract test catches immediately |
| OTEL context leak from generation span | Very low | Medium | Span closes before tool execution (verified in analysis) |
| Langfuse disabled in production | N/A | None | All helpers are no-ops when disabled |
| Performance overhead of wrapper | Negligible | None | One async context manager + dict copy per LLM call |
