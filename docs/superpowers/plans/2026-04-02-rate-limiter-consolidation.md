# Rate Limiter Consolidation Fix

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent background consolidation from competing with the main agent for Anthropic API rate limits, and increase retry resilience so transient rate limit errors don't kill the turn.

**Architecture:** Two independent fixes. Fix A adds an optional `rate_limiter` parameter to `ConsolidationOrchestrator` so it calls `wait_if_needed()` before and `record()` after the LLM call — keeping the rate limiter in `nanobot/agent/` (boundary-safe). Fix B increases the error retry threshold from 3 to 5 and max backoff from 10s to 30s in `TurnRunner._handle_llm_error()`.

**Tech Stack:** Python 3.10+, pytest

**Worktree:** `../nanobot-fix-rate-limiter` (branch `fix/rate-limiter-consolidation`)

---

## File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Modify | `nanobot/agent/consolidation.py` | Accept optional rate limiter, call wait/record around LLM calls |
| Modify | `nanobot/agent/agent_factory.py` | Move rate limiter construction before `_wire_memory`, pass to consolidator |
| Modify | `nanobot/agent/turn_runner.py:592,598` | Change error threshold 3→5, max backoff 10→30 |
| Modify | `tests/test_consolidation.py` | Add tests for rate limiter integration |
| Modify | `tests/test_cli_retry_path.py` | Update retry count expectations |

---

### Task 1: Add rate limiter support to ConsolidationOrchestrator

**Files:**
- Modify: `tests/test_consolidation.py`
- Modify: `nanobot/agent/consolidation.py:29-47,123-148`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_consolidation.py` after the existing imports (line 6):

```python
import asyncio

import pytest
```

Then add at the end of the file (after the `TestArchiveFnOnFailure` class):

```python
class TestRateLimiterIntegration:
    """Consolidation respects the shared rate limiter."""

    async def test_submit_calls_wait_if_needed_before_consolidate(self):
        """Rate limiter wait is called before the LLM-backed consolidation."""
        rl = AsyncMock()
        rl.wait_if_needed = AsyncMock(return_value=0.0)
        rl.record = MagicMock()

        memory = MagicMock()
        memory.consolidate = AsyncMock(return_value=True)

        orch = ConsolidationOrchestrator(
            memory=memory,
            archive_fn=MagicMock(),
            max_concurrent=2,
            memory_window=50,
            enable_contradiction_check=True,
            rate_limiter=rl,
        )
        session = MagicMock()
        session.messages = []
        async with orch:
            orch.submit("s", session, MagicMock(), "m")
        rl.wait_if_needed.assert_awaited_once()
        memory.consolidate.assert_called_once()

    async def test_submit_records_tokens_after_consolidate(self):
        """Rate limiter records estimated tokens after consolidation completes."""
        rl = AsyncMock()
        rl.wait_if_needed = AsyncMock(return_value=0.0)
        rl.record = MagicMock()

        memory = MagicMock()
        memory.consolidate = AsyncMock(return_value=True)

        orch = ConsolidationOrchestrator(
            memory=memory,
            archive_fn=MagicMock(),
            max_concurrent=2,
            memory_window=50,
            enable_contradiction_check=True,
            rate_limiter=rl,
        )
        session = MagicMock()
        session.messages = [{"role": "user", "content": "hello " * 500}]
        async with orch:
            orch.submit("s", session, MagicMock(), "m")
        rl.record.assert_called_once()
        recorded_tokens = rl.record.call_args[0][0]
        assert recorded_tokens > 0

    async def test_submit_without_rate_limiter_still_works(self):
        """Consolidation without a rate limiter works as before (backward compat)."""
        memory = MagicMock()
        memory.consolidate = AsyncMock(return_value=True)

        orch = ConsolidationOrchestrator(
            memory=memory,
            archive_fn=MagicMock(),
            max_concurrent=2,
            memory_window=50,
            enable_contradiction_check=True,
        )
        session = MagicMock()
        session.messages = []
        async with orch:
            orch.submit("s", session, MagicMock(), "m")
        memory.consolidate.assert_called_once()

    async def test_consolidate_and_wait_also_respects_rate_limiter(self):
        """The blocking consolidate_and_wait path also checks the rate limiter."""
        rl = AsyncMock()
        rl.wait_if_needed = AsyncMock(return_value=0.0)
        rl.record = MagicMock()

        memory = MagicMock()
        memory.consolidate = AsyncMock(return_value=True)

        orch = ConsolidationOrchestrator(
            memory=memory,
            archive_fn=MagicMock(),
            max_concurrent=2,
            memory_window=50,
            enable_contradiction_check=True,
            rate_limiter=rl,
        )
        session = MagicMock()
        session.messages = []
        async with orch:
            await orch.consolidate_and_wait("s", session, MagicMock(), "m")
        rl.wait_if_needed.assert_awaited_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_consolidation.py::TestRateLimiterIntegration -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'rate_limiter'`

- [ ] **Step 3: Implement rate limiter support in ConsolidationOrchestrator**

In `nanobot/agent/consolidation.py`, add the import at line 15 (after `from loguru import logger`):

```python
from nanobot.context.compression import estimate_messages_tokens
```

Update `__init__` (line 29) to accept the rate limiter. Add after the `enable_contradiction_check` parameter:

```python
        rate_limiter: Any | None = None,
```

Store it after line 42 (`self._enable_contradiction_check = enable_contradiction_check`):

```python
        self._rate_limiter = rate_limiter
```

Add a private helper method after `_get_or_create_lock` (after line 121):

```python
    async def _rate_limit_guard(self, session: Session) -> None:
        """Wait for rate limit headroom before a consolidation LLM call."""
        if self._rate_limiter is None:
            return
        await self._rate_limiter.wait_if_needed()

    def _rate_limit_record(self, session: Session) -> None:
        """Record estimated token usage after a consolidation LLM call."""
        if self._rate_limiter is None:
            return
        # Estimate: consolidation sends a digest of session messages.
        # The digest is ~600 chars per message (truncated in compression.py:301).
        # Use the same heuristic as estimate_messages_tokens for consistency.
        estimated = estimate_messages_tokens(session.messages) // 4
        # Floor at 2000 tokens (system prompt + tool schema overhead)
        self._rate_limiter.record(max(2000, estimated))
```

Update `_run` method (line 123) to call the guard before and record after:

Replace lines 131-142:
```python
        try:
            async with self._sem:
                lock = self._get_or_create_lock(session_key)
                async with lock:
                    try:
                        await self._memory.consolidate(
                            session,
                            provider,
                            model,
                            memory_window=self._memory_window,
                            enable_contradiction_check=self._enable_contradiction_check,
                        )
                    except Exception:  # crash-barrier: consolidation failure
                        logger.exception("Consolidation failed for {}; archiving", session_key)
                        if self._archive_fn is not None:
                            self._archive_fn(list(session.messages))
```

With:
```python
        try:
            async with self._sem:
                lock = self._get_or_create_lock(session_key)
                async with lock:
                    try:
                        await self._rate_limit_guard(session)
                        await self._memory.consolidate(
                            session,
                            provider,
                            model,
                            memory_window=self._memory_window,
                            enable_contradiction_check=self._enable_contradiction_check,
                        )
                        self._rate_limit_record(session)
                    except Exception:  # crash-barrier: consolidation failure
                        logger.exception("Consolidation failed for {}; archiving", session_key)
                        if self._archive_fn is not None:
                            self._archive_fn(list(session.messages))
```

Update `consolidate_and_wait` method (line 88). Replace lines 101-110:
```python
        lock = self._get_or_create_lock(session_key)
        async with lock:
            return await self._memory.consolidate(
                session,
                provider,
                model,
                memory_window=self._memory_window,
                enable_contradiction_check=self._enable_contradiction_check,
                archive_all=archive_all,
            )
```

With:
```python
        lock = self._get_or_create_lock(session_key)
        async with lock:
            await self._rate_limit_guard(session)
            result = await self._memory.consolidate(
                session,
                provider,
                model,
                memory_window=self._memory_window,
                enable_contradiction_check=self._enable_contradiction_check,
                archive_all=archive_all,
            )
            self._rate_limit_record(session)
            return result
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_consolidation.py -v`
Expected: PASS (all existing + 4 new tests)

- [ ] **Step 5: Run lint and typecheck**

Run: `make lint && make typecheck`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add nanobot/agent/consolidation.py tests/test_consolidation.py
git commit -m "fix(agent): make consolidation respect shared rate limiter

Background consolidation was calling provider.chat() directly, bypassing
the rate limiter. When consolidation and the main agent turn hit the
Anthropic API simultaneously, rate limit errors crashed the user's turn.

ConsolidationOrchestrator now accepts an optional rate_limiter and calls
wait_if_needed() before and record() after LLM calls."
```

---

### Task 2: Wire rate limiter to consolidator in agent_factory

**Files:**
- Modify: `nanobot/agent/agent_factory.py:155-192,303-310`

- [ ] **Step 1: Move rate limiter construction before `_wire_memory`**

In `nanobot/agent/agent_factory.py`, the rate limiter is currently constructed at step 8.5 (lines 305-310), after `_wire_memory` at step 7 (line 303). Move it before.

Find lines 302-310:
```python
    # 7. Wire memory
    consolidator = _wire_memory(context=context, config=config)

    # 8.5. Construct RateLimiter for Anthropic models
    from nanobot.providers.rate_limiter import RateLimiter as _RateLimiter

    _rate_limiter: _RateLimiter | None = None
    if "anthropic/" in model or "claude" in model.lower():
        _rate_limiter = _RateLimiter(tokens_per_minute=50_000)
```

Replace with:
```python
    # 7. Construct RateLimiter for Anthropic models (before memory wiring)
    from nanobot.providers.rate_limiter import RateLimiter as _RateLimiter

    _rate_limiter: _RateLimiter | None = None
    if "anthropic/" in model or "claude" in model.lower():
        _rate_limiter = _RateLimiter(tokens_per_minute=50_000)

    # 7.5. Wire memory (with rate limiter for consolidation)
    consolidator = _wire_memory(context=context, config=config, rate_limiter=_rate_limiter)
```

- [ ] **Step 2: Update `_wire_memory` to accept and pass rate limiter**

Update the function signature at line 155. Replace:
```python
def _wire_memory(
    context: ContextBuilder,
    config: AgentConfig,
) -> ConsolidationOrchestrator:
```

With:
```python
def _wire_memory(
    context: ContextBuilder,
    config: AgentConfig,
    rate_limiter: Any | None = None,
) -> ConsolidationOrchestrator:
```

Update the `ConsolidationOrchestrator` construction (line 186). Replace:
```python
    assert context.memory is not None  # always injected by build_agent
    return ConsolidationOrchestrator(
        memory=context.memory,
        archive_fn=_archive,
        max_concurrent=3,
        memory_window=config.memory.window,
        enable_contradiction_check=config.memory.enable_contradiction_check,
    )
```

With:
```python
    assert context.memory is not None  # always injected by build_agent
    return ConsolidationOrchestrator(
        memory=context.memory,
        archive_fn=_archive,
        max_concurrent=3,
        memory_window=config.memory.window,
        enable_contradiction_check=config.memory.enable_contradiction_check,
        rate_limiter=rate_limiter,
    )
```

- [ ] **Step 3: Run checks**

Run: `make lint && make typecheck`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add nanobot/agent/agent_factory.py
git commit -m "fix(agent): wire rate limiter to consolidation orchestrator

Moves rate limiter construction before _wire_memory so it can be passed
to the ConsolidationOrchestrator. Both background (submit) and blocking
(consolidate_and_wait) paths now share the rate limiter with the main
agent turn."
```

---

### Task 3: Increase retry resilience in TurnRunner

**Files:**
- Modify: `nanobot/agent/turn_runner.py:592,598`
- Modify: `tests/test_cli_retry_path.py:38-54`

- [ ] **Step 1: Update the retry test**

In `tests/test_cli_retry_path.py`, the test `test_llm_error_three_times_returns_fallback_message` (line 38) sends exactly 3 error responses and expects the fallback. Update it to send 5.

Replace lines 38-54:
```python
async def test_llm_error_three_times_returns_fallback_message() -> None:
    """Three consecutive errors return the fallback message without crashing."""
    provider = ScriptedProvider([error_response(), error_response(), error_response()])
    received: list[ProgressEvent] = []

    async def tracking(event: ProgressEvent) -> None:
        received.append(event)

    loop = make_agent_loop(provider)
    result = await loop.process_direct("hello", on_progress=tracking)

    assert "trouble reaching the language model" in result
    retry_signals = [
        e for e in received if isinstance(e, StatusEvent) and e.status_code == "retrying"
    ]
    # Signals on attempt 1 and 2; attempt 3 breaks the loop
    assert len(retry_signals) == 2
```

With:
```python
async def test_llm_error_five_times_returns_fallback_message() -> None:
    """Five consecutive errors return the fallback message without crashing."""
    provider = ScriptedProvider([
        error_response(), error_response(), error_response(),
        error_response(), error_response(),
    ])
    received: list[ProgressEvent] = []

    async def tracking(event: ProgressEvent) -> None:
        received.append(event)

    loop = make_agent_loop(provider)
    result = await loop.process_direct("hello", on_progress=tracking)

    assert "trouble reaching the language model" in result
    retry_signals = [
        e for e in received if isinstance(e, StatusEvent) and e.status_code == "retrying"
    ]
    # Signals on attempts 1-4; attempt 5 breaks the loop
    assert len(retry_signals) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_retry_path.py::test_llm_error_five_times_returns_fallback_message -v`
Expected: FAIL — `assert len(retry_signals) == 4` fails (currently 2 because the loop breaks after 3)

- [ ] **Step 3: Update turn_runner constants**

In `nanobot/agent/turn_runner.py`, update line 592. Replace:
```python
            if state.consecutive_errors >= 3:
```
With:
```python
            if state.consecutive_errors >= 5:
```

Update line 598. Replace:
```python
            await asyncio.sleep(min(2**state.consecutive_errors, 10))
```
With:
```python
            await asyncio.sleep(min(2**state.consecutive_errors, 30))
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_cli_retry_path.py -v`
Expected: PASS (both tests)

- [ ] **Step 5: Run full check**

Run: `make check && make test`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add nanobot/agent/turn_runner.py tests/test_cli_retry_path.py
git commit -m "fix(agent): increase LLM error retry resilience (3→5, 10s→30s)

The previous 3-retry / 10s-max-backoff was too aggressive for rate limit
errors, where the API window is 60 seconds. Increasing to 5 retries with
30s max backoff gives rate windows time to drain before giving up."
```

---

## Verification

After all tasks:

1. `make check && make test` — all pass
2. Review the wiring: rate limiter is constructed once in `agent_factory.py`, shared between `StreamingLLMCaller` (main turns) and `ConsolidationOrchestrator` (background consolidation)
3. Both `submit()` and `consolidate_and_wait()` paths call `wait_if_needed()` and `record()`
4. The retry path now tolerates up to 5 consecutive errors with up to 30s backoff
