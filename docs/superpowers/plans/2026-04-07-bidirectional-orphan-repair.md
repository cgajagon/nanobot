# Tool-Message Orphan Prevention and Repair — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix orphaned tool_result messages that crash the Anthropic API with 400 errors, and stop retrying non-retryable errors.

**Architecture:** Prevention via boundary-aware slicing in `get_history()`, single-authority reverse orphan repair in `providers/sanitize.py`, independent tail orphan fix in compression, and distinct `finish_reason` values for non-retryable LLM errors.

**Tech Stack:** Python 3.10+, pytest, pytest-asyncio

**Spec:** `docs/superpowers/specs/2026-04-07-bidirectional-orphan-repair-design.md`

---

### Task 1: Extract `_sanitize_messages()` to `providers/sanitize.py`

Extract the existing function before modifying it. This is a pure refactor — no behavior change.

**Files:**
- Create: `nanobot/providers/sanitize.py`
- Modify: `nanobot/providers/litellm_provider.py:210-245`
- Test: `tests/test_litellm_provider.py` (existing tests must still pass)

- [ ] **Step 1: Create `nanobot/providers/sanitize.py` with the extracted function**

```python
"""Message sanitization for LLM providers.

Pure functions that normalize message lists before sending to LLM APIs.
Repairs orphaned tool_calls and tool_results to prevent API validation errors.
"""

from __future__ import annotations

from typing import Any

# Keys that LLM APIs accept in message dicts. Everything else is stripped.
_ALLOWED_MSG_KEYS: frozenset[str] = frozenset(
    {"role", "content", "tool_calls", "tool_call_id", "name"}
)


def sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip non-standard keys, ensure content key, and repair orphaned tool_calls.

    This is the single authoritative location for message sanitization before
    LLM API calls. Called by LiteLLMProvider.chat() and stream_chat().
    """
    sanitized = []
    for msg in messages:
        clean = {k: v for k, v in msg.items() if k in _ALLOWED_MSG_KEYS}
        # Strict providers require "content" even when assistant only has tool_calls
        if clean.get("role") == "assistant" and "content" not in clean:
            clean["content"] = None
        sanitized.append(clean)

    # Forward repair: strip assistant tool_calls that lack matching tool results.
    # This handles mid-turn crashes where tool execution was interrupted.
    tool_result_ids: set[str] = {
        m["tool_call_id"]
        for m in sanitized
        if m.get("role") == "tool" and "tool_call_id" in m
    }
    repaired = []
    for msg in sanitized:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            orphaned = [
                tc
                for tc in msg["tool_calls"]
                if tc.get("id") and tc["id"] not in tool_result_ids
            ]
            if orphaned:
                valid = [tc for tc in msg["tool_calls"] if tc["id"] in tool_result_ids]
                if valid:
                    repaired.append({**msg, "tool_calls": valid})
                else:
                    repaired.append(
                        {k: v for k, v in msg.items() if k != "tool_calls"}
                    )
                continue
        repaired.append(msg)
    return repaired
```

- [ ] **Step 2: Update `litellm_provider.py` to import from the new module**

In `nanobot/providers/litellm_provider.py`, replace the `_sanitize_messages` static method and the `_ALLOWED_MSG_KEYS` constant with an import:

Remove lines 14-18 (`_ALLOWED_MSG_KEYS` definition) and lines 210-245 (`_sanitize_messages` method).

Add this import near the top:

```python
from nanobot.providers.sanitize import sanitize_messages
```

Replace all calls from `self._sanitize_messages(messages)` to `sanitize_messages(messages)`. There are two call sites:
- In `chat()` (~line 270): `messages, tools = self._apply_cache_control(sanitize_messages(messages), tools)`
- In `stream_chat()` (~line 410): `messages = sanitize_messages(messages)`

- [ ] **Step 3: Run existing tests to confirm no regression**

Run: `pytest tests/test_litellm_provider.py -v`
Expected: All 30+ existing tests PASS — no behavior change.

- [ ] **Step 4: Run structural checks**

Run: `make lint && make typecheck && make import-check && make structure-check`
Expected: All pass. `litellm_provider.py` drops ~35 LOC (from 535 to ~500). `sanitize.py` is ~55 LOC.

- [ ] **Step 5: Commit**

```bash
git add nanobot/providers/sanitize.py nanobot/providers/litellm_provider.py
git commit -m "refactor(providers): extract sanitize_messages to providers/sanitize.py

Prepares for adding reverse orphan repair. No behavior change.
Pure extraction of _sanitize_messages() and _ALLOWED_MSG_KEYS."
```

---

### Task 2: Add reverse orphan repair to `sanitize_messages()`

Add the missing reverse direction: strip tool results without matching assistant tool_calls.

**Files:**
- Modify: `nanobot/providers/sanitize.py`
- Test: `tests/test_litellm_provider.py`

- [ ] **Step 1: Write failing tests**

Add these tests to `tests/test_litellm_provider.py`:

```python
def test_sanitize_strips_orphaned_tool_result():
    """Tool result without matching assistant tool_call is stripped."""
    messages = [
        {"role": "user", "content": "hello"},
        # No assistant with tool_calls for "orphan_id"
        {"role": "tool", "tool_call_id": "orphan_id", "name": "exec", "content": "result"},
        {"role": "assistant", "content": "response"},
    ]
    result = sanitize_messages(messages)
    roles = [m["role"] for m in result]
    assert "tool" not in roles, "Orphaned tool result should be stripped"
    assert len(result) == 2  # user + assistant


def test_sanitize_keeps_paired_tool_result():
    """Tool result with matching assistant tool_call is preserved."""
    messages = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "tc_1", "type": "function", "function": {"name": "exec", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "tc_1", "name": "exec", "content": "result"},
        {"role": "assistant", "content": "done"},
    ]
    result = sanitize_messages(messages)
    tool_msgs = [m for m in result if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "tc_1"


def test_sanitize_bidirectional_repair():
    """Both orphaned tool_calls and orphaned tool_results are stripped."""
    messages = [
        {"role": "user", "content": "hello"},
        # Orphaned tool result (no matching assistant)
        {"role": "tool", "tool_call_id": "orphan_result", "name": "exec", "content": "r1"},
        # Assistant with orphaned tool_call (no matching result)
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "orphan_call", "type": "function", "function": {"name": "read", "arguments": "{}"}}
            ],
        },
        # Valid pair
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "valid_1", "type": "function", "function": {"name": "exec", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "valid_1", "name": "exec", "content": "r2"},
    ]
    result = sanitize_messages(messages)
    # Orphaned tool result stripped, orphaned tool_call stripped, valid pair kept
    tool_msgs = [m for m in result if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "valid_1"
    # The orphaned assistant should have tool_calls removed
    for m in result:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            ids = [tc["id"] for tc in m["tool_calls"]]
            assert "orphan_call" not in ids
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_litellm_provider.py::test_sanitize_strips_orphaned_tool_result tests/test_litellm_provider.py::test_sanitize_bidirectional_repair -v`
Expected: FAIL — orphaned tool results are not yet stripped.

- [ ] **Step 3: Add reverse orphan repair to `sanitize_messages()`**

In `nanobot/providers/sanitize.py`, add the reverse repair after the existing forward repair loop. Replace the `return repaired` at the end with:

```python
    # Reverse repair: strip tool results without matching assistant tool_calls.
    # This catches orphans from session truncation, compression, or corrupted data.
    assistant_tc_ids: set[str] = set()
    for msg in repaired:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if tc.get("id"):
                    assistant_tc_ids.add(tc["id"])

    return [
        msg
        for msg in repaired
        if not (
            msg.get("role") == "tool"
            and msg.get("tool_call_id")
            and msg["tool_call_id"] not in assistant_tc_ids
        )
    ]
```

- [ ] **Step 4: Run all sanitize tests**

Run: `pytest tests/test_litellm_provider.py -v -k "sanitize"`
Expected: All pass, including the three new tests.

- [ ] **Step 5: Run full test suite for provider**

Run: `pytest tests/test_litellm_provider.py -v`
Expected: All 30+ tests pass.

- [ ] **Step 6: Commit**

```bash
git add nanobot/providers/sanitize.py tests/test_litellm_provider.py
git commit -m "fix(providers): add reverse orphan repair to sanitize_messages

Strip tool result messages whose tool_call_id has no matching assistant
tool_calls entry. This is the single authoritative repair point for
orphaned tool results, catching truncation, compression, and corruption."
```

---

### Task 3: Boundary-aware slicing in `get_history()`

Replace the raw-index truncation with boundary-aware slicing that prevents orphans structurally.

**Files:**
- Modify: `nanobot/session/manager.py:66-111`
- Test: `tests/test_session_manager.py`

- [ ] **Step 1: Write failing tests**

Add these tests to `tests/test_session_manager.py` inside the `TestSession` class:

```python
def test_get_history_boundary_slicing_finds_user(self):
    """Window walks back to nearest user message for clean boundary."""
    session = Session(key="test")
    # 10 messages: user at index 0, then 9 tool-heavy messages
    session.messages = [
        {"role": "user", "content": "start"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "tc_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "tc_1", "name": "f", "content": "r1"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "tc_2", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "tc_2", "name": "f", "content": "r2"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "tc_3", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "tc_3", "name": "f", "content": "r3"},
        {"role": "assistant", "content": "final"},
    ]
    # max_messages=5 would start at index 3, walk back to index 0 (user)
    history = session.get_history(max_messages=5)
    assert history[0]["role"] == "user"
    # No orphaned tool results
    tool_ids = {m["tool_call_id"] for m in history if m.get("role") == "tool"}
    asst_ids = set()
    for m in history:
        if m.get("tool_calls"):
            for tc in m["tool_calls"]:
                asst_ids.add(tc["id"])
    assert tool_ids <= asst_ids, "All tool results must have matching tool_calls"

def test_get_history_boundary_slicing_finds_standalone_assistant(self):
    """Window walks back to standalone assistant when no user in range."""
    session = Session(key="test")
    session.messages = [
        {"role": "assistant", "content": "standalone"},  # clean boundary
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "tc_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "tc_1", "name": "f", "content": "r1"},
        {"role": "assistant", "content": "done"},
    ]
    history = session.get_history(max_messages=3)
    # Should walk back to index 0 (standalone assistant)
    assert history[0]["role"] == "assistant"
    assert history[0]["content"] == "standalone"

def test_get_history_boundary_no_clean_boundary_returns_empty(self):
    """No user or standalone assistant → empty history."""
    session = Session(key="test")
    # All messages are tool-call cycles with no user or standalone assistant
    session.messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "tc_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "tc_1", "name": "f", "content": "r1"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "tc_2", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "tc_2", "name": "f", "content": "r2"},
    ]
    history = session.get_history(max_messages=2)
    # Forward scan finds assistant at index 2 — but it has tool_calls
    # No standalone assistant either — returns empty
    assert history == []

def test_get_history_boundary_preserves_complete_cycles(self):
    """Boundary slicing never orphans tool results."""
    session = Session(key="test")
    # Simulate the incident: 39 messages, window=25
    msgs = [{"role": "user", "content": "start"}]
    for i in range(15):
        tc_id = f"tc_{i}"
        msgs.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": tc_id, "type": "function", "function": {"name": "exec", "arguments": "{}"}}],
        })
        msgs.append({"role": "tool", "tool_call_id": tc_id, "name": "exec", "content": f"result_{i}"})
    msgs.append({"role": "assistant", "content": "final answer"})
    session.messages = msgs  # 33 messages

    history = session.get_history(max_messages=25)
    # Verify no orphans: every tool result has a matching tool_call
    tool_ids = {m["tool_call_id"] for m in history if m.get("role") == "tool"}
    asst_ids = set()
    for m in history:
        if m.get("tool_calls"):
            for tc in m["tool_calls"]:
                asst_ids.add(tc["id"])
    assert tool_ids <= asst_ids, f"Orphaned tool results: {tool_ids - asst_ids}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_session_manager.py::TestSession::test_get_history_boundary_slicing_finds_user tests/test_session_manager.py::TestSession::test_get_history_boundary_preserves_complete_cycles -v`
Expected: FAIL — current code produces orphaned tool results.

- [ ] **Step 3: Implement boundary-aware slicing**

In `nanobot/session/manager.py`, replace lines 68-75 of `get_history()`:

```python
        unconsolidated = self.messages[self.last_consolidated :]
        sliced = unconsolidated[-max_messages:]

        # Drop leading non-user messages to avoid orphaned tool_result blocks
        for i, m in enumerate(sliced):
            if m.get("role") == "user":
                sliced = sliced[i:]
                break
```

With:

```python
        unconsolidated = self.messages[self.last_consolidated :]

        # Boundary-aware slicing: find a clean message boundary to avoid
        # orphaning tool results whose matching assistant tool_calls are
        # outside the window. A "clean boundary" is a user message or a
        # standalone assistant (no tool_calls) — neither is mid-cycle.
        target_start = max(0, len(unconsolidated) - max_messages)
        boundary = target_start
        for i in range(target_start, -1, -1):
            m = unconsolidated[i]
            role = m.get("role")
            if role == "user":
                boundary = i
                break
            if role == "assistant" and not m.get("tool_calls"):
                boundary = i
                break
        else:
            # No clean boundary behind target — scan forward for any
            # standalone assistant after target_start
            found = False
            for i in range(target_start, len(unconsolidated)):
                m = unconsolidated[i]
                if m.get("role") == "user":
                    boundary = i
                    found = True
                    break
                if m.get("role") == "assistant" and not m.get("tool_calls"):
                    boundary = i
                    found = True
                    break
            if not found:
                boundary = len(unconsolidated)  # yields empty slice

        sliced = unconsolidated[boundary:]
```

- [ ] **Step 4: Run all session tests**

Run: `pytest tests/test_session_manager.py -v`
Expected: All tests pass, including the 4 new ones and 8 existing ones.

- [ ] **Step 5: Run structural checks**

Run: `make lint && make typecheck`
Expected: Pass.

- [ ] **Step 6: Commit**

```bash
git add nanobot/session/manager.py tests/test_session_manager.py
git commit -m "fix(session): boundary-aware slicing in get_history()

Replace raw-index truncation with boundary-aware slicing that walks
backward from the target index to the nearest user message or standalone
assistant. Prevents orphaned tool_result messages structurally rather
than repairing them after the fact."
```

---

### Task 4: Compression tail orphan verification

Add tail orphan verification after `_paired_drop_tools()` in both compression callers.

**Files:**
- Modify: `nanobot/context/compression.py:190-194` and `273-277`
- Test: `tests/test_compression_coherence.py`

- [ ] **Step 1: Write failing tests**

Add these tests to `tests/test_compression_coherence.py`, in a new class after `TestPairedDropTools`:

```python
class TestTailOrphanVerification:
    """Tests for tail orphan protection after _paired_drop_tools."""

    def test_tail_result_with_call_in_tail_preserved(self):
        """Tool result in tail with matching call in tail is kept."""
        middle = [{"role": "user", "content": "old"}]
        tail = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "tc_1", "type": "function", "function": {"name": "f"}}],
            },
            {"role": "tool", "tool_call_id": "tc_1", "content": "result"},
        ]
        from nanobot.context.compression import _strip_tail_orphans

        cleaned = _strip_tail_orphans(middle, tail)
        assert len([m for m in cleaned if m.get("role") == "tool"]) == 1

    def test_tail_result_with_call_in_middle_preserved(self):
        """Tool result in tail with matching call still in middle is kept."""
        middle = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "tc_1", "type": "function", "function": {"name": "f"}}],
            },
        ]
        tail = [
            {"role": "tool", "tool_call_id": "tc_1", "content": "result"},
            {"role": "user", "content": "next"},
        ]
        from nanobot.context.compression import _strip_tail_orphans

        cleaned = _strip_tail_orphans(middle, tail)
        tool_msgs = [m for m in cleaned if m.get("role") == "tool"]
        assert len(tool_msgs) == 1

    def test_tail_result_with_call_nowhere_stripped(self):
        """Tool result in tail with no matching call anywhere is stripped."""
        middle = [{"role": "user", "content": "old"}]
        tail = [
            {"role": "tool", "tool_call_id": "orphan_tc", "content": "result"},
            {"role": "user", "content": "next"},
        ]
        from nanobot.context.compression import _strip_tail_orphans

        cleaned = _strip_tail_orphans(middle, tail)
        tool_msgs = [m for m in cleaned if m.get("role") == "tool"]
        assert len(tool_msgs) == 0

    def test_tail_non_tool_messages_untouched(self):
        """Non-tool messages in tail are never stripped."""
        middle = []
        tail = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
        from nanobot.context.compression import _strip_tail_orphans

        cleaned = _strip_tail_orphans(middle, tail)
        assert len(cleaned) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_compression_coherence.py::TestTailOrphanVerification -v`
Expected: FAIL — `_strip_tail_orphans` does not exist yet.

- [ ] **Step 3: Implement `_strip_tail_orphans` and wire into callers**

In `nanobot/context/compression.py`, add this function (before `_paired_drop_tools`):

```python
def _strip_tail_orphans(
    middle: list[dict[str, Any]],
    tail: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Strip tool results from *tail* whose tool_call is not in middle or tail.

    After ``_paired_drop_tools`` filters the middle and Phase 3 may
    drop/summarize it entirely, tool results in the tail can become orphaned
    if their matching assistant tool_calls was in the (now-dropped) middle.
    """
    all_call_ids: set[str] = set()
    for m in middle + tail:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                if tc.get("id"):
                    all_call_ids.add(tc["id"])

    return [
        m
        for m in tail
        if not (
            m.get("role") == "tool"
            and m.get("tool_call_id")
            and m["tool_call_id"] not in all_call_ids
        )
    ]
```

Then wire it into both callers:

In `compress_context()` (sync), after line 190 (`middle = _paired_drop_tools(middle, tail)`), add:

```python
        tail = _strip_tail_orphans(middle, tail)
```

In `summarize_and_compress()` (async), after line 273 (`middle_no_tools = _paired_drop_tools(middle, tail)`), add:

```python
    tail = _strip_tail_orphans(middle_no_tools, tail)
```

- [ ] **Step 4: Run all compression tests**

Run: `pytest tests/test_compression_coherence.py -v`
Expected: All tests pass (4 new + 17 existing).

- [ ] **Step 5: Commit**

```bash
git add nanobot/context/compression.py tests/test_compression_coherence.py
git commit -m "fix(context): add tail orphan verification after _paired_drop_tools

Strip tool results from the tail whose matching assistant tool_calls
is only in the middle (which Phase 3 may drop/summarize). Prevents
orphaned tool_results from reaching the LLM API."
```

---

### Task 5: Non-retryable error classification in provider

Set distinct `finish_reason` values for non-retryable errors.

**Files:**
- Modify: `nanobot/providers/litellm_provider.py:314-336` and `517-525`
- Test: `tests/test_litellm_provider.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_litellm_provider.py`:

```python
@pytest.mark.asyncio
async def test_chat_invalid_request_sets_finish_reason(monkeypatch):
    """BadRequestError sets finish_reason='invalid_request', not 'error'."""
    import litellm

    async def mock_acompletion(**kwargs):
        raise litellm.BadRequestError(
            message="invalid_request_error: orphaned tool_result",
            model="test",
            llm_provider="anthropic",
        )

    monkeypatch.setattr("nanobot.providers.litellm_provider.acompletion", mock_acompletion)
    provider = LiteLLMProvider(api_keys={"anthropic": "sk-test"})
    result = await provider.chat(
        messages=[{"role": "user", "content": "hi"}],
        model="anthropic/claude-haiku-4-5",
    )
    assert result.finish_reason == "invalid_request"


@pytest.mark.asyncio
async def test_chat_auth_error_sets_finish_reason(monkeypatch):
    """AuthenticationError sets finish_reason='auth_error', not 'error'."""
    import litellm

    async def mock_acompletion(**kwargs):
        raise litellm.AuthenticationError(
            message="invalid api key",
            model="test",
            llm_provider="anthropic",
        )

    monkeypatch.setattr("nanobot.providers.litellm_provider.acompletion", mock_acompletion)
    provider = LiteLLMProvider(api_keys={"anthropic": "sk-test"})
    result = await provider.chat(
        messages=[{"role": "user", "content": "hi"}],
        model="anthropic/claude-haiku-4-5",
    )
    assert result.finish_reason == "auth_error"


@pytest.mark.asyncio
async def test_chat_generic_error_keeps_finish_reason_error(monkeypatch):
    """Unknown exceptions keep finish_reason='error' (existing behavior)."""
    async def mock_acompletion(**kwargs):
        raise ConnectionError("network down")

    monkeypatch.setattr("nanobot.providers.litellm_provider.acompletion", mock_acompletion)
    provider = LiteLLMProvider(api_keys={"anthropic": "sk-test"})
    result = await provider.chat(
        messages=[{"role": "user", "content": "hi"}],
        model="anthropic/claude-haiku-4-5",
    )
    assert result.finish_reason == "error"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_litellm_provider.py::test_chat_invalid_request_sets_finish_reason tests/test_litellm_provider.py::test_chat_auth_error_sets_finish_reason -v`
Expected: FAIL — both return `finish_reason="error"` currently.

- [ ] **Step 3: Add error classification to `chat()` exception handler**

In `nanobot/providers/litellm_provider.py`, replace the except block in `chat()` (around lines 328-336):

```python
        except Exception as e:  # crash-barrier: LLM provider errors are varied
            elapsed = time.monotonic() - t0
            llm_calls_total.labels(model=model, role="chat", success="False").inc()
            llm_latency_seconds.labels(model=model, role="chat").observe(elapsed)
            error_str = str(e)
            exc_name = type(e).__name__
            # Classify into distinct finish_reason values so the turn runner
            # can fail fast on non-retryable errors without inspecting content.
            if "BadRequestError" in exc_name or "invalid_request" in error_str:
                finish = "invalid_request"
            elif "AuthenticationError" in exc_name or "401" in error_str:
                finish = "auth_error"
            else:
                finish = "error"
            return LLMResponse(
                content=f"Error calling LLM: {error_str}",
                finish_reason=finish,
            )
```

- [ ] **Step 4: Add the same classification to `stream_chat()` exception handler**

In `stream_chat()` (around lines 517-525), apply the same pattern:

```python
        except Exception as e:  # crash-barrier: streaming errors are varied
            elapsed = time.monotonic() - t0
            llm_calls_total.labels(model=resolved, role="stream", success="False").inc()
            llm_latency_seconds.labels(model=resolved, role="stream").observe(elapsed)
            error_str = str(e)
            exc_name = type(e).__name__
            if "BadRequestError" in exc_name or "invalid_request" in error_str:
                finish = "invalid_request"
            elif "AuthenticationError" in exc_name or "401" in error_str:
                finish = "auth_error"
            else:
                finish = "error"
            yield StreamChunk(
                content_delta=f"Error calling LLM: {e}",
                finish_reason=finish,
                done=True,
            )
```

- [ ] **Step 5: Run all provider tests**

Run: `pytest tests/test_litellm_provider.py -v`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add nanobot/providers/litellm_provider.py tests/test_litellm_provider.py
git commit -m "fix(providers): distinct finish_reason for non-retryable errors

Set finish_reason='invalid_request' for BadRequestError and
'auth_error' for AuthenticationError instead of generic 'error'.
Enables the turn runner to fail fast on deterministic failures."
```

---

### Task 6: Fail-fast branches in `_handle_llm_error()`

Add mechanical branches for `invalid_request` and `auth_error` finish reasons.

**Files:**
- Modify: `nanobot/agent/turn_runner.py:580-623` and `626-667`
- Test: `tests/test_turn_runner.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_turn_runner.py`, in a new class:

```python
class TestLLMErrorClassification:
    """Tests for non-retryable error handling."""

    @pytest.mark.asyncio
    async def test_invalid_request_fails_fast(self):
        """invalid_request finish_reason breaks immediately, no retries."""
        from nanobot.providers.base import LLMResponse

        responses = [
            LLMResponse(
                content="Error calling LLM: BadRequestError: orphaned tool_result",
                finish_reason="invalid_request",
            ),
        ]
        caller = ScriptedCaller(responses)
        runner = TurnRunner(
            llm_caller=caller,
            tool_executor=FakeExecutor(),
            guardrails=FakeGuardrails(),
            context=FakeContext(),
            config=_make_config(),
        )
        state = TurnState(
            messages=[{"role": "user", "content": "test"}],
            user_text="test",
        )
        result = await runner.run(state, on_progress=None)
        assert caller.call_count == 1, "Should not retry invalid_request"
        assert "error preparing the conversation" in result.content

    @pytest.mark.asyncio
    async def test_auth_error_fails_fast(self):
        """auth_error finish_reason breaks immediately, no retries."""
        from nanobot.providers.base import LLMResponse

        responses = [
            LLMResponse(
                content="Error calling LLM: AuthenticationError: invalid key",
                finish_reason="auth_error",
            ),
        ]
        caller = ScriptedCaller(responses)
        runner = TurnRunner(
            llm_caller=caller,
            tool_executor=FakeExecutor(),
            guardrails=FakeGuardrails(),
            context=FakeContext(),
            config=_make_config(),
        )
        state = TurnState(
            messages=[{"role": "user", "content": "test"}],
            user_text="test",
        )
        result = await runner.run(state, on_progress=None)
        assert caller.call_count == 1, "Should not retry auth_error"
        assert "Authentication" in result.content or "API key" in result.content

    @pytest.mark.asyncio
    async def test_generic_error_still_retries(self):
        """Generic 'error' finish_reason retries with backoff (existing behavior)."""
        from nanobot.providers.base import LLMResponse

        error_resp = LLMResponse(
            content="Error calling LLM: ConnectionError",
            finish_reason="error",
        )
        success_resp = LLMResponse(content="recovered", finish_reason="stop")
        responses = [error_resp, success_resp]
        caller = ScriptedCaller(responses)
        runner = TurnRunner(
            llm_caller=caller,
            tool_executor=FakeExecutor(),
            guardrails=FakeGuardrails(),
            context=FakeContext(),
            config=_make_config(),
        )
        state = TurnState(
            messages=[{"role": "user", "content": "test"}],
            user_text="test",
        )
        result = await runner.run(state, on_progress=None)
        assert caller.call_count == 2, "Generic errors should retry"
        assert result.content == "recovered"
```

Note: These tests use the existing `ScriptedCaller`, `FakeExecutor`, `FakeGuardrails`, `FakeContext`, and `_make_config` helpers already in `tests/test_turn_runner.py`. If `ScriptedCaller` doesn't track `call_count`, add a counter in the `call` method. Check the existing helper implementations before running.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_turn_runner.py::TestLLMErrorClassification -v`
Expected: FAIL — `invalid_request` and `auth_error` are not handled.

- [ ] **Step 3: Add fail-fast branches to `_handle_llm_error()`**

In `nanobot/agent/turn_runner.py`, in `_handle_llm_error()`, add these branches **before** the existing `if response.finish_reason == "error":` block (around line 587):

```python
        if response.finish_reason == "invalid_request":
            logger.warning("Non-retryable LLM error (invalid request): {}", response.content)
            fc = _build_error_with_progress(state, error_type="invalid_request")
            state.messages[:] = self._context.add_assistant_message(state.messages, fc)
            return "break", fc

        if response.finish_reason == "auth_error":
            logger.warning("Non-retryable LLM error (auth): {}", response.content)
            fc = _build_error_with_progress(state, error_type="auth_error")
            state.messages[:] = self._context.add_assistant_message(state.messages, fc)
            return "break", fc
```

- [ ] **Step 4: Add error messages to `_build_error_with_progress()`**

In the `_build_error_with_progress()` function, add cases before the existing `if not tool_summaries:` block:

```python
    if error_type == "invalid_request":
        return (
            "I encountered an error preparing the conversation for the language model. "
            "This usually resolves by starting a new conversation."
        )
    if error_type == "auth_error":
        return (
            "Authentication with the language model failed. "
            "Please check your API key configuration."
        )
```

- [ ] **Step 5: Run all turn runner tests**

Run: `pytest tests/test_turn_runner.py -v`
Expected: All tests pass (3 new + existing).

- [ ] **Step 6: Commit**

```bash
git add nanobot/agent/turn_runner.py tests/test_turn_runner.py
git commit -m "fix(agent): fail fast on non-retryable LLM errors

Add mechanical branches for 'invalid_request' and 'auth_error'
finish_reason values. No retries or backoff — immediate break with
descriptive error message. Generic 'error' retries unchanged."
```

---

### Task 7: Contract tests — pipeline invariant

End-to-end test verifying the full pipeline never produces orphaned tool results.

**Files:**
- Create: `tests/contract/test_message_pairing_contracts.py`

- [ ] **Step 1: Write the contract tests**

```python
"""Contract tests: tool-message pairing invariant across the full pipeline.

These tests verify that no combination of session truncation and provider
sanitization produces orphaned tool_result messages (tool results without
matching assistant tool_calls).
"""

from __future__ import annotations

from nanobot.providers.sanitize import sanitize_messages
from nanobot.session.manager import Session


def _assert_no_orphaned_tool_results(messages: list[dict]) -> None:
    """Assert that every tool result has a matching assistant tool_call."""
    assistant_tc_ids: set[str] = set()
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                if tc.get("id"):
                    assistant_tc_ids.add(tc["id"])

    for m in messages:
        if m.get("role") == "tool" and m.get("tool_call_id"):
            assert m["tool_call_id"] in assistant_tc_ids, (
                f"Orphaned tool_result: tool_call_id={m['tool_call_id']} "
                f"not in assistant tool_calls {assistant_tc_ids}"
            )


class TestMessagePairingInvariant:
    """The pipeline must never produce orphaned tool results."""

    def test_large_session_small_window(self):
        """39 messages, max_messages=25 — the exact incident scenario."""
        session = Session(key="test")
        session.messages = [
            {"role": "user", "content": "Set up cron job"},
        ]
        for i in range(15):
            tc_id = f"toolu_{i:04d}"
            session.messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": tc_id, "type": "function", "function": {"name": "exec", "arguments": "{}"}}
                ],
            })
            session.messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "name": "exec",
                "content": f"result_{i}",
            })
        session.messages.append({"role": "assistant", "content": "Done"})
        # 33 messages total

        history = session.get_history(max_messages=25)
        sanitized = sanitize_messages(history)
        _assert_no_orphaned_tool_results(sanitized)

    def test_all_tool_messages_no_user(self):
        """Session with only tool-call cycles, no user messages at all."""
        session = Session(key="test")
        session.messages = []
        for i in range(10):
            tc_id = f"tc_{i}"
            session.messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": tc_id, "type": "function", "function": {"name": "f", "arguments": "{}"}}
                ],
            })
            session.messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "name": "f",
                "content": f"r{i}",
            })
        # 20 messages, all tool cycles

        history = session.get_history(max_messages=8)
        sanitized = sanitize_messages(history)
        _assert_no_orphaned_tool_results(sanitized)

    def test_multi_tool_batch_split(self):
        """Assistant with multiple tool_calls split across the window boundary."""
        session = Session(key="test")
        session.messages = [
            {"role": "user", "content": "start"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "tc_a", "type": "function", "function": {"name": "f", "arguments": "{}"}},
                    {"id": "tc_b", "type": "function", "function": {"name": "g", "arguments": "{}"}},
                    {"id": "tc_c", "type": "function", "function": {"name": "h", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "tc_a", "name": "f", "content": "ra"},
            {"role": "tool", "tool_call_id": "tc_b", "name": "g", "content": "rb"},
            {"role": "tool", "tool_call_id": "tc_c", "name": "h", "content": "rc"},
            {"role": "assistant", "content": "final"},
        ]

        history = session.get_history(max_messages=4)
        sanitized = sanitize_messages(history)
        _assert_no_orphaned_tool_results(sanitized)
```

- [ ] **Step 2: Run contract tests**

Run: `pytest tests/contract/test_message_pairing_contracts.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/contract/test_message_pairing_contracts.py
git commit -m "test(contract): add message pairing invariant tests

Verify the full pipeline (get_history + sanitize_messages) never
produces orphaned tool_result messages, including the exact incident
scenario (39 messages, max_messages=25)."
```

---

### Task 8: Final validation

Run the full check suite to confirm everything is clean.

**Files:** None (validation only)

- [ ] **Step 1: Run make check**

Run: `make check`
Expected: All pass — lint, typecheck, import-check, structure-check.

- [ ] **Step 2: Run full test suite**

Run: `make test`
Expected: All tests pass.

- [ ] **Step 3: Run make pre-push**

Run: `make pre-push`
Expected: Full CI passes including coverage gate.

- [ ] **Step 4: Verify file sizes**

Check that no file exceeds 500 LOC:
```bash
wc -l nanobot/session/manager.py nanobot/providers/sanitize.py nanobot/providers/litellm_provider.py nanobot/context/compression.py nanobot/agent/turn_runner.py
```
Expected: All under 500 (except `turn_runner.py` which has a pre-existing `size-exception`).
