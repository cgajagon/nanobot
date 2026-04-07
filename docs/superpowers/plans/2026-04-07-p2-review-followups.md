# P2 Review Follow-ups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Address the 4 medium-priority items from the comprehensive review of the orphan-repair branch.

**Architecture:** Documentation updates + one small refactor + two new tests. All independent, no cross-dependencies.

**Tech Stack:** Python 3.10+, pytest, pytest-asyncio

---

### Task 1: Update cognitive-architecture.md LLM Error Handling section

The LLM Error Handling section (lines 228-247) only documents `finish_reason="error"` with retry+backoff. It does not mention the `invalid_request` or `auth_error` fail-fast paths added in PR #163.

**Files:**
- Modify: `.claude/rules/cognitive-architecture.md:228-247`

- [ ] **Step 1: Add non-retryable error documentation**

In `.claude/rules/cognitive-architecture.md`, find the LLM Error Handling section (line 228). After the paragraph ending with "preventing concurrent API competition." (line 247), add:

```markdown

**Non-retryable errors:** The provider classifies certain exceptions into
distinct `finish_reason` values that bypass retry entirely:

- `"invalid_request"` — 400 errors (malformed messages, orphaned tool results).
  Immediate break with user-facing message suggesting a new conversation.
- `"auth_error"` — 401 errors (invalid API key). Immediate break with
  configuration guidance.

These branches appear before the `"error"` branch in `_handle_llm_error()`
so they take precedence. Classification uses `isinstance` against litellm's
exception hierarchy (`litellm.BadRequestError`, `litellm.AuthenticationError`)
via `_classify_llm_error()` in `litellm_provider.py`.
```

- [ ] **Step 2: Run doc-check**

Run: `cd C:/Users/C95071414/Documents/nanobot-main/.worktrees/fix-p2-review-items && make doc-check`
Expected: Pass (no broken references).

- [ ] **Step 3: Commit**

```bash
git add .claude/rules/cognitive-architecture.md
git commit -m "docs: document non-retryable error handling in cognitive-architecture.md

Add invalid_request and auth_error finish_reason documentation to the
LLM Error Handling section. These fail-fast paths were added in PR #163."
```

---

### Task 2: Extract `_find_clean_boundary()` from `get_history()`

The `for/else` construct in `get_history()` (lines 79-103) is a readability trap — the `else` is 25 lines from its `for`, contains a nested `for` with a `found` flag. Extract to a helper function.

**Files:**
- Modify: `nanobot/session/manager.py:66-105`
- Test: `tests/test_session_manager.py` (existing tests must still pass)

- [ ] **Step 1: Add the helper function**

In `nanobot/session/manager.py`, add this function BEFORE the `Session` class (after the `_clamp_tool_id` function):

```python
def _find_clean_boundary(
    unconsolidated: list[dict[str, Any]], target_start: int
) -> int:
    """Find nearest clean message boundary for history slicing.

    A "clean boundary" is a user message or a standalone assistant (no
    tool_calls) — neither is mid-tool-cycle. Scans backward from
    *target_start* first, then forward if nothing found behind.

    Returns the index to slice from, or ``len(unconsolidated)`` if no
    clean boundary exists (yields an empty slice).
    """
    # Scan backward from target
    for i in range(target_start, -1, -1):
        m = unconsolidated[i]
        role = m.get("role")
        if role == "user" or (role == "assistant" and not m.get("tool_calls")):
            return i
    # No clean boundary behind — scan forward
    for i in range(target_start, len(unconsolidated)):
        m = unconsolidated[i]
        role = m.get("role")
        if role == "user" or (role == "assistant" and not m.get("tool_calls")):
            return i
    return len(unconsolidated)
```

- [ ] **Step 2: Replace the inline logic in `get_history()`**

Replace lines 77-105 (from `target_start = ...` through `sliced = unconsolidated[boundary:]`) with:

```python
        target_start = max(0, len(unconsolidated) - max_messages)
        boundary = _find_clean_boundary(unconsolidated, target_start)
        sliced = unconsolidated[boundary:]
```

- [ ] **Step 3: Also update the docstring**

Replace line 67's docstring:
```python
        """Return unconsolidated messages for LLM input, aligned to a user turn."""
```
With:
```python
        """Return unconsolidated messages for LLM input, sliced at a clean boundary."""
```

- [ ] **Step 4: Run all session tests**

Run: `cd C:/Users/C95071414/Documents/nanobot-main/.worktrees/fix-p2-review-items && python -m pytest tests/test_session_manager.py -v`
Expected: All 20 tests pass (no behavior change).

- [ ] **Step 5: Run lint and typecheck**

Run: `make lint && make typecheck`
Expected: Pass.

- [ ] **Step 6: Commit**

```bash
git add nanobot/session/manager.py
git commit -m "refactor(session): extract _find_clean_boundary from get_history

Replace for/else construct with a helper function that scans backward
then forward for a clean message boundary. Eliminates the found flag,
the nested for loop, and the duplicated role-checking logic."
```

---

### Task 3: Add `stream_chat` error classification test

The `_classify_llm_error()` function is tested via `chat()` but the `stream_chat()` integration path is not exercised.

**Files:**
- Modify: `tests/test_litellm_provider.py`

- [ ] **Step 1: Write the test**

Add this test to `tests/test_litellm_provider.py`, near the existing `test_stream_chat_success_and_error` test (around line 340):

```python
@pytest.mark.asyncio
async def test_stream_chat_invalid_request_sets_finish_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """stream_chat classifies BadRequestError as 'invalid_request'."""
    import litellm as _litellm

    async def mock_acompletion(**kwargs: Any) -> Any:
        raise _litellm.BadRequestError(
            message="invalid_request_error: orphaned tool_result",
            model="test",
            llm_provider="anthropic",
        )

    monkeypatch.setattr("nanobot.providers.litellm_provider.acompletion", mock_acompletion)
    provider = LiteLLMProvider(api_key="sk-test")
    chunks = [
        c
        async for c in provider.stream_chat(
            messages=[{"role": "user", "content": "hi"}], tools=None
        )
    ]
    assert chunks[-1].done is True
    assert chunks[-1].finish_reason == "invalid_request"
```

- [ ] **Step 2: Run the test**

Run: `cd C:/Users/C95071414/Documents/nanobot-main/.worktrees/fix-p2-review-items && python -m pytest tests/test_litellm_provider.py::test_stream_chat_invalid_request_sets_finish_reason -v`
Expected: PASS (the implementation already uses `_classify_llm_error` in `stream_chat`).

- [ ] **Step 3: Run full provider tests**

Run: `python -m pytest tests/test_litellm_provider.py -v`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_litellm_provider.py
git commit -m "test(providers): add stream_chat error classification test

Verify that stream_chat uses _classify_llm_error and sets
finish_reason='invalid_request' for BadRequestError, matching
the chat() behavior."
```

---

### Task 4: Add `test_id_remapping_preserves_pairing` test

The `_clamp_tool_id` remapping in `get_history()` is not tested for pairing preservation — if remapping only transforms one side (assistant tool_calls but not tool results), orphans would result.

**Files:**
- Modify: `tests/test_session_manager.py`

- [ ] **Step 1: Write the test**

Add this test to `tests/test_session_manager.py` inside the `TestSession` class:

```python
    def test_get_history_id_remapping_preserves_pairing(self):
        """Tool call IDs are remapped consistently in both assistant and tool messages."""
        session = Session(key="test")
        # Use a long ID (>40 chars) that will trigger _clamp_tool_id remapping
        long_id = "toolu_01ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdef"
        assert len(long_id) > 40, "ID must be long enough to trigger remapping"
        session.messages = [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": long_id,
                        "type": "function",
                        "function": {"name": "exec", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": long_id, "name": "exec", "content": "result"},
            {"role": "assistant", "content": "done"},
        ]
        history = session.get_history()
        # Find the remapped IDs
        asst_msg = next(m for m in history if m.get("tool_calls"))
        tool_msg = next(m for m in history if m.get("role") == "tool")
        remapped_call_id = asst_msg["tool_calls"][0]["id"]
        remapped_result_id = tool_msg["tool_call_id"]
        # The IDs should be remapped (different from original)
        assert remapped_call_id != long_id, "Long ID should be remapped"
        # The remapped IDs must match each other (pairing preserved)
        assert remapped_call_id == remapped_result_id, (
            f"Remapped IDs must match: call={remapped_call_id}, result={remapped_result_id}"
        )
```

- [ ] **Step 2: Run the test**

Run: `cd C:/Users/C95071414/Documents/nanobot-main/.worktrees/fix-p2-review-items && python -m pytest tests/test_session_manager.py::TestSession::test_get_history_id_remapping_preserves_pairing -v`
Expected: PASS (the existing `_clamp_tool_id` uses deterministic hashing, so both sides get the same short ID).

- [ ] **Step 3: Run all session tests**

Run: `python -m pytest tests/test_session_manager.py -v`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_session_manager.py
git commit -m "test(session): verify tool_call ID remapping preserves pairing

Add test that long tool_call IDs (>40 chars) are remapped consistently
in both assistant tool_calls and tool result messages, ensuring the
deterministic hash produces matching short IDs on both sides."
```

---

### Task 5: Final validation

**Files:** None (validation only)

- [ ] **Step 1: Run make check**

Run: `cd C:/Users/C95071414/Documents/nanobot-main/.worktrees/fix-p2-review-items && make check`
Expected: All pass.

- [ ] **Step 2: Run affected tests**

Run: `python -m pytest tests/test_session_manager.py tests/test_litellm_provider.py tests/contract/test_message_pairing_contracts.py -v`
Expected: All pass.
