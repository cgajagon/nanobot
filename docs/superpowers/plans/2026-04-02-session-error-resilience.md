# Session Error Resilience Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure conversation state survives LLM errors, and the agent can resume interrupted tasks on the next turn.

**Architecture:** Two independent changes in `agent/`: (1) defensive session persistence with pre-save + try/finally in `MessageProcessor`, (2) enriched error messages via a pure formatter in `TurnRunner`. No new files besides tests. No new abstractions.

**Tech Stack:** Python 3.10+, pytest, pytest-asyncio, ScriptedProvider for test mocking.

**Spec:** `docs/superpowers/specs/2026-04-02-session-error-resilience-design.md`

---

### Task 1: Enriched Error Message Formatter (TurnRunner)

**Files:**
- Create: `tests/test_error_message_formatter.py`
- Modify: `nanobot/agent/turn_runner.py:580-625`

- [ ] **Step 1: Write failing tests for the error formatter**

Create `tests/test_error_message_formatter.py`:

```python
"""Unit tests for _build_error_with_progress formatter."""

from __future__ import annotations

from nanobot.agent.turn_runner import _build_error_with_progress
from nanobot.agent.turn_types import TurnState


def _make_state(
    messages: list[dict] | None = None,
    user_text: str = "find DS10540",
) -> TurnState:
    return TurnState(
        messages=messages or [],
        user_text=user_text,
    )


class TestBuildErrorWithProgress:
    """Tests for _build_error_with_progress."""

    def test_generic_message_when_no_tools_ran(self):
        """Without tool results, returns the generic error message."""
        state = _make_state(
            messages=[
                {"role": "system", "content": "You are an assistant."},
                {"role": "user", "content": "hello"},
            ],
        )
        msg = _build_error_with_progress(state)
        assert "having trouble" in msg.lower()
        assert "Progress" not in msg

    def test_includes_tool_progress_when_tools_ran(self):
        """When tools ran this turn, message includes progress summary."""
        state = _make_state(
            messages=[
                {"role": "system", "content": "prompt"},
                {"role": "user", "content": "find DS10540"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "tc1"}]},
                {"role": "tool", "tool_call_id": "tc1", "name": "exec", "content": "results"},
                {"role": "tool", "tool_call_id": "tc2", "name": "read_file", "content": "data"},
            ],
            user_text="find DS10540",
        )
        msg = _build_error_with_progress(state)
        assert "Progress before the error" in msg
        assert "exec: success" in msg
        assert "read_file: success" in msg
        assert "find DS10540" in msg

    def test_caps_tool_summaries_at_five(self):
        """Tool summary is capped at 5 entries to keep the message concise."""
        tool_msgs = [
            {"role": "tool", "tool_call_id": f"tc{i}", "name": f"tool_{i}", "content": "ok"}
            for i in range(10)
        ]
        state = _make_state(
            messages=[
                {"role": "system", "content": "prompt"},
                {"role": "user", "content": "hello"},
                *tool_msgs,
            ],
        )
        msg = _build_error_with_progress(state)
        assert msg.count("- tool_") == 5

    def test_only_counts_current_turn_tools(self):
        """Tool results from previous turns (before last user msg) are excluded."""
        state = _make_state(
            messages=[
                {"role": "system", "content": "prompt"},
                {"role": "user", "content": "previous question"},
                {"role": "tool", "tool_call_id": "old", "name": "old_tool", "content": "old"},
                {"role": "assistant", "content": "previous answer"},
                {"role": "user", "content": "new question"},
                {"role": "tool", "tool_call_id": "new", "name": "new_tool", "content": "new"},
            ],
            user_text="new question",
        )
        msg = _build_error_with_progress(state)
        assert "new_tool" in msg
        assert "old_tool" not in msg

    def test_user_text_truncated_at_200_chars(self):
        """Long user text is truncated in the error message."""
        long_text = "x" * 300
        state = _make_state(
            messages=[
                {"role": "user", "content": long_text},
                {"role": "tool", "tool_call_id": "tc1", "name": "exec", "content": "ok"},
            ],
            user_text=long_text,
        )
        msg = _build_error_with_progress(state)
        # The quoted user text should be truncated
        assert "x" * 201 not in msg
        assert "x" * 200 in msg
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_error_message_formatter.py -v`
Expected: FAIL — `ImportError: cannot import name '_build_error_with_progress'`

- [ ] **Step 3: Implement the formatter in turn_runner.py**

Add the function before `_handle_llm_error` (around line 578 in `nanobot/agent/turn_runner.py`):

```python
def _build_error_with_progress(state: TurnState) -> str:
    """Build an error message that summarizes what was accomplished.

    Only considers tool results from the current turn — messages after the
    last user message — not the full history.
    """
    last_user_idx = 0
    for i, m in enumerate(state.messages):
        if m.get("role") == "user":
            last_user_idx = i

    tool_summaries: list[str] = []
    for m in state.messages[last_user_idx + 1 :]:
        if m.get("role") == "tool" and m.get("name"):
            status = "failed" if m.get("_error") else "success"
            tool_summaries.append(f"- {m['name']}: {status}")

    if not tool_summaries:
        return (
            "I'm having trouble reaching the language model right now. "
            "Please try again in a moment."
        )

    progress = "\n".join(tool_summaries[:5])
    return (
        "I was working on your request but the language model became unavailable.\n\n"
        f"Progress before the error:\n{progress}\n\n"
        f"Your message: \"{state.user_text[:200]}\"\n\n"
        "Please repeat your message or say \"continue\" to resume."
    )
```

Then replace the three hardcoded error strings in `_handle_llm_error`:

**Line 593** (finish_reason="error"): Replace:
```python
fc = "I'm having trouble reaching the language model right now. Please try again in a moment."
```
With:
```python
fc = _build_error_with_progress(state)
```

**Line 605** (content_filter): Replace:
```python
fc = "The AI provider's content filter blocked my response. Try rephrasing your question."
```
With:
```python
fc = _build_error_with_progress(state)
```

**Line 617-619** (length): Replace:
```python
fc = (
    "My response was too long and got cut off. Try asking a more specific question."
)
```
With:
```python
fc = _build_error_with_progress(state)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_error_message_formatter.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Run lint and typecheck**

Run: `make lint && make typecheck`
Expected: PASS with no errors

- [ ] **Step 6: Commit**

```bash
git add tests/test_error_message_formatter.py nanobot/agent/turn_runner.py
git commit -m "feat(agent): enriched error messages with progress summary

When the LLM fails after retries, the error message now includes which
tools ran successfully and quotes the user's original message, helping
the agent resume on the next turn."
```

---

### Task 2: Update Existing Error Message Assertions

**Files:**
- Modify: `tests/test_agent_loop.py:241`
- Modify: `tests/test_cli_retry_path.py:57`
- Modify: `tests/golden/test_golden_scenarios.py:435`
- Modify: `tests/test_no_answer_recovery.py:174`

These tests assert on hardcoded error message substrings that changed in Task 1.

- [ ] **Step 1: Update test_agent_loop.py assertion**

In `tests/test_agent_loop.py`, line 241, replace:
```python
assert "trouble" in result.content.lower() or "try again" in result.content.lower()
```
With:
```python
assert "trouble" in result.content.lower() or "language model" in result.content.lower()
```

The generic message (no tools ran) still contains "trouble". This assertion already passes since the ScriptedProvider errors have no tool calls preceding them.

- [ ] **Step 2: Update test_cli_retry_path.py assertion**

In `tests/test_cli_retry_path.py`, line 57, replace:
```python
assert "trouble reaching the language model" in result
```
With:
```python
assert "trouble" in result.lower() or "language model" in result.lower()
```

Same reasoning — no tool calls in these errors, so the generic message fires.

- [ ] **Step 3: Update golden scenario assertion**

In `tests/golden/test_golden_scenarios.py`, line 435, replace:
```python
assert "trouble" in result.content.lower() or "try again" in result.content.lower()
```
With:
```python
assert "trouble" in result.content.lower() or "language model" in result.content.lower()
```

- [ ] **Step 4: Verify content_filter assertion still passes**

In `tests/test_no_answer_recovery.py`, line 174:
```python
assert "content filter" in result.content.lower()
```

This assertion will FAIL because the formatter no longer mentions "content filter" — it uses the same progress-based message. Replace with:
```python
assert "trouble" in result.content.lower() or "language model" in result.content.lower()
```

- [ ] **Step 5: Run all modified test files**

Run: `pytest tests/test_agent_loop.py tests/test_cli_retry_path.py tests/golden/test_golden_scenarios.py tests/test_no_answer_recovery.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add tests/test_agent_loop.py tests/test_cli_retry_path.py tests/golden/test_golden_scenarios.py tests/test_no_answer_recovery.py
git commit -m "test: update error message assertions for enriched formatter

The error formatter now returns progress-aware messages. Tests that
checked for hardcoded substrings are updated to match the new format."
```

---

### Task 3: Defensive Persistence (MessageProcessor)

**Files:**
- Create: `tests/contract/test_session_persistence_on_error.py`
- Modify: `nanobot/agent/message_processor.py:209-333`

- [ ] **Step 1: Write failing contract tests**

Create `tests/contract/test_session_persistence_on_error.py`:

```python
"""Contract tests: session persistence survives LLM errors."""

from __future__ import annotations

from pathlib import Path

import pytest

from nanobot.bus.events import InboundMessage
from nanobot.providers.base import LLMResponse, ToolCallRequest
from tests.helpers import ScriptedProvider, _make_loop


def _make_inbound(text: str = "find DS10540") -> InboundMessage:
    return InboundMessage(
        channel="cli",
        chat_id="test-user",
        sender_id="user-1",
        content=text,
    )


class TestUserMessagePersistedBeforeOrchestrator:
    """User message must be on disk before the orchestrator runs."""

    async def test_user_message_survives_provider_exception(self, tmp_path: Path):
        """If the provider raises, the user message is still in the session."""

        class FailingProvider(ScriptedProvider):
            async def chat(self, *args, **kwargs):  # type: ignore[override]
                raise RuntimeError("connection reset")

        provider = FailingProvider([])
        loop = _make_loop(tmp_path, provider)

        with pytest.raises(RuntimeError, match="connection reset"):
            await loop._process_message(_make_inbound("important question"))

        # Reload session from disk
        session = loop._processor.sessions.get_or_create("cli:test-user")
        loop._processor.sessions.invalidate("cli:test-user")
        session = loop._processor.sessions.get_or_create("cli:test-user")

        user_msgs = [m for m in session.messages if m.get("role") == "user"]
        assert len(user_msgs) >= 1
        assert "important question" in user_msgs[-1]["content"]


class TestPartialStateSavedOnException:
    """Tool results accumulated before a crash are persisted."""

    async def test_tool_results_saved_when_second_llm_call_crashes(self, tmp_path: Path):
        """First LLM call returns tool_calls, second raises. Partial state saved."""
        call_count = 0

        class CrashOnSecondCall(ScriptedProvider):
            async def chat(self, *args, **kwargs):  # type: ignore[override]
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return LLMResponse(
                        content=None,
                        tool_calls=[
                            ToolCallRequest(
                                id="tc_1",
                                name="read_file",
                                arguments={"file_path": str(tmp_path / "data.txt")},
                            )
                        ],
                    )
                raise RuntimeError("API unavailable")

        # Create the file the tool will read
        (tmp_path / "data.txt").write_text("test data")

        provider = CrashOnSecondCall([])
        loop = _make_loop(tmp_path, provider)

        with pytest.raises(RuntimeError, match="API unavailable"):
            await loop._process_message(_make_inbound("read my file"))

        # Reload session from disk
        loop._processor.sessions.invalidate("cli:test-user")
        session = loop._processor.sessions.get_or_create("cli:test-user")

        # User message should be persisted
        user_msgs = [m for m in session.messages if m.get("role") == "user"]
        assert any("read my file" in m["content"] for m in user_msgs)

        # Tool result should be persisted (partial state from before crash)
        tool_msgs = [m for m in session.messages if m.get("role") == "tool"]
        assert len(tool_msgs) >= 1


class TestNoDuplicateUserMessage:
    """Pre-save + normal save must not produce duplicate user messages."""

    async def test_successful_turn_has_one_user_message(self, tmp_path: Path):
        """A normal successful turn stores exactly one user message."""
        provider = ScriptedProvider([LLMResponse(content="Hello!")])
        loop = _make_loop(tmp_path, provider)

        await loop._process_message(_make_inbound("hi there"))

        # Reload session from disk
        loop._processor.sessions.invalidate("cli:test-user")
        session = loop._processor.sessions.get_or_create("cli:test-user")

        user_msgs = [m for m in session.messages if m.get("role") == "user"]
        assert len(user_msgs) == 1
        assert "hi there" in user_msgs[0]["content"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/contract/test_session_persistence_on_error.py -v`
Expected: FAIL — `test_user_message_survives_provider_exception` fails because user message is not on disk. `test_successful_turn_has_one_user_message` may pass or fail depending on current behavior.

- [ ] **Step 3: Implement defensive persistence in message_processor.py**

In `nanobot/agent/message_processor.py`, make two changes:

**Change A — Pre-save user message (after line 216, before line 218):**

Add after the `initial_messages` construction (line 216) and before the canonical builder (line 218):

```python
        # Defensive persistence: save the user message to the session
        # immediately, before orchestration. This ensures the user's input
        # survives even if the orchestrator crashes with an unhandled exception.
        session.add_message("user", msg.content)
        self.sessions.save(session)
```

**Change B — Wrap orchestrator + post-processing in try/finally (lines 254-333):**

Replace the block from line 254 through line 333 with:

```python
        _skip = 1 + len(history) + 1  # system + history + pre-saved user msg
        all_msgs: list[dict[str, Any]] = []
        try:
            final_content, tools_used, all_msgs = await self._run_orchestrator(
                initial_messages,
                on_progress=((on_progress or _bus_progress) if self.config.streaming_enabled else None),
            )

            if final_content is None:
                _recovered = await self._attempt_recovery(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    all_msgs=all_msgs,
                )
                if isinstance(_recovered, str):
                    final_content = _recovered

            if final_content is None:
                _safe_msgs: list[dict[str, Any]] = all_msgs if isinstance(all_msgs, list) else []
                final_content = _build_no_answer_explanation(msg.content, _safe_msgs)
                _added = self.context.add_assistant_message(_safe_msgs, final_content)
                if isinstance(_added, list):
                    all_msgs = _added

            if not isinstance(final_content, str):
                final_content = str(final_content) if final_content else ""

            self._sync_token_counters()

            _update_span = (
                getattr(self._span_module, "update_current_span", update_current_span)
                if self._span_module is not None
                else update_current_span
            )
            _update_span(
                output=final_content[:500] if final_content else None,
                metadata={
                    "channel": msg.channel,
                    "sender": msg.sender_id,
                    "model": self.model,
                    "role": self.role_name,
                    "session_key": key,
                    "llm_calls": self._turn_llm_calls,
                    "prompt_tokens": self._turn_tokens_prompt,
                    "completion_tokens": self._turn_tokens_completion,
                    "cache_creation_tokens": self._turn_cache_creation_tokens,
                    "cache_read_tokens": self._turn_cache_read_tokens,
                    "duration_ms": round((time.monotonic() - t0_request) * 1000),
                },
            )

            preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
            logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)

            duration_ms = (time.monotonic() - t0_request) * 1000
            bind_trace().info(
                "request_complete | {ch}:{cid} | {dur:.0f}ms | model={mdl}"
                " | tools={tc} | len={rlen}"
                " | llm_calls={lc} | prompt_tokens={pt} | completion_tokens={ct}"
                " | cache_write={cw} | cache_read={cr}",
                ch=msg.channel,
                cid=msg.chat_id,
                dur=duration_ms,
                mdl=self.model,
                tc=len(tools_used),
                rlen=len(final_content),
                lc=self._turn_llm_calls,
                pt=self._turn_tokens_prompt,
                ct=self._turn_tokens_completion,
                cw=self._turn_cache_creation_tokens,
                cr=self._turn_cache_read_tokens,
            )

            if isinstance(all_msgs, list):
                self._save_turn(session, all_msgs, _skip)
            self.sessions.save(session)

        except Exception:
            # Error-path save: persist any partial state (tool results
            # accumulated before the crash). The pre-saved user message
            # is already on disk from the defensive save above.
            if isinstance(all_msgs, list) and len(all_msgs) > _skip:
                self._save_turn(session, all_msgs, _skip)
                self.sessions.save(session)
            raise
```

The key change to `_skip`: the old value was `1 + len(history)` which pointed at the user message in `all_msgs`. Since we pre-saved the user message, we add `+ 1` to skip past it, so `_save_turn` only extracts assistant/tool messages. This prevents the duplicate.

- [ ] **Step 4: Run contract tests to verify they pass**

Run: `pytest tests/contract/test_session_persistence_on_error.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Run the full test suite for regressions**

Run: `pytest tests/test_agent_loop.py tests/test_no_answer_recovery.py tests/test_cli_retry_path.py tests/golden/test_golden_scenarios.py tests/test_session_manager.py -v`
Expected: All PASS

- [ ] **Step 6: Run lint and typecheck**

Run: `make lint && make typecheck`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add nanobot/agent/message_processor.py tests/contract/test_session_persistence_on_error.py
git commit -m "feat(agent): defensive session persistence on LLM error

Pre-save the user message before orchestration and wrap the orchestrator
in try/finally to persist partial state (tool results) on unhandled
exceptions. Ensures conversation context survives LLM failures."
```

---

### Task 4: Full Validation

- [ ] **Step 1: Run make check**

Run: `make check`
Expected: All structural checks pass (lint + typecheck + import-check + structure-check + prompt-check + phase-todo-check + doc-check)

- [ ] **Step 2: Run make test**

Run: `make test`
Expected: All unit tests pass, no regressions

- [ ] **Step 3: Run make pre-push**

Run: `make pre-push`
Expected: Full CI passes including coverage gate (85%)

- [ ] **Step 4: Final commit if any fixes were needed**

Only if steps 1-3 required fixes. Use conventional commit format matching the fix type.
