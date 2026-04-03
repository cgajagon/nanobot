# Session Error Resilience Design

> Date: 2026-04-02
> Status: Approved
> Scope: Defensive session persistence + enriched error messages on LLM failure

## Problem

When the LLM fails mid-turn (API errors, rate limits, timeouts), the conversation
context degrades in two ways:

1. **Unhandled exceptions** (network crashes, provider errors that escape TurnRunner):
   The crash barrier in `AgentLoop.run()` catches the exception and sends a generic
   error to the user, but `MessageProcessor._process_message()` never reaches the
   session save at line 331-333. The current turn's user message and any partial tool
   results are lost.

2. **Handled LLM errors** (finish_reason="error" after retries): The session IS saved,
   but the error message is generic ("I'm having trouble reaching the language model")
   with no context about what the agent was doing. On the next turn, the agent sees
   the error in its history but cannot connect it to the interrupted task.

### Evidence

Langfuse session `web:eee4361b-7225-445c-8434-f0db302d5a7b` (2026-04-02):

- **Trace 1**: 10 LLM calls, 34 tool observations. Agent gathered DS10540 project
  data from Obsidian and produced a status report. Asked clarification questions.
- **Trace 2**: User replied with answers. First LLM call succeeded (read a daily
  note), then 2 more calls failed (0 input tokens — Anthropic API rejected them).
  Error message: "I'm having trouble reaching the language model."
- **Trace 3**: User said "Try again". Agent loaded 12K tokens of history but responded
  "Could you clarify what you'd like me to do?" — the conversational thread was lost.

## Solution

Two changes in `agent/`, zero new abstractions.

### Change 1: Defensive Persistence (MessageProcessor)

**File:** `nanobot/agent/message_processor.py`

**Pre-save user message** before calling `_run_orchestrator()`:

After building `initial_messages` (line 216) and before the orchestrator call (line 254),
persist the user message to the session immediately:

```python
session.add_message("user", msg.content)
self.sessions.save(session)
```

This ensures the user's input survives even if the orchestrator crashes.

**Error-path save** with try/finally around the orchestrator:

```python
all_msgs: list[dict[str, Any]] = []
try:
    final_content, tools_used, all_msgs = await self._run_orchestrator(...)
    # ... existing post-processing ...
    if isinstance(all_msgs, list):
        self._save_turn(session, all_msgs, 1 + len(history))
    self.sessions.save(session)
except Exception:
    if all_msgs:
        self._save_turn(session, all_msgs, 1 + len(history))
        self.sessions.save(session)
    raise
```

The `skip` parameter in `_save_turn` already accounts for the user message in
`initial_messages`, so assistant/tool messages are extracted without duplicating
the pre-saved user message.

**LOC impact:** ~15 lines modified in `message_processor.py`.

### Change 2: Enriched Error Messages (TurnRunner)

**File:** `nanobot/agent/turn_runner.py`

Replace hardcoded error strings in `_handle_llm_error` with a pure-function formatter
that summarizes progress from `state.messages`:

```python
def _build_error_with_progress(state: TurnState) -> str:
    """Build an error message that summarizes what was accomplished.

    Only considers tool results from the current turn — messages after the
    last user message — not the full history.
    """
    # Find the last user message index to scope to current turn only
    last_user_idx = 0
    for i, m in enumerate(state.messages):
        if m.get("role") == "user":
            last_user_idx = i

    tool_summaries = []
    for m in state.messages[last_user_idx + 1:]:
        if m.get("role") == "tool" and m.get("name"):
            status = "success" if not m.get("_error") else "failed"
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

Applied to all three error types: `finish_reason="error"` (line 593),
`content_filter` (line 605), and `length` (line 620).

**Architectural justification:** The loop stays mechanical — it calls a pure formatter
and delivers the string. The formatter reads `state.messages` (working memory) which
is already available. No domain logic added to the loop. The error message content is
the "volatile edge" (Pattern 6).

**LOC impact:** ~28 lines added to `turn_runner.py`.

### Prompt template: NOT included

The enriched error message is self-contained — it quotes the user's original message
and lists progress. Adding a section to `reasoning.md` would cost ~40-100 tokens on
every turn for a rare scenario. If weaker models struggle with resumption despite the
enriched error message, a prompt section can be added later (one-line `.md` change).

## Testing

### New: Contract test for persistence on error

**File:** `tests/contract/test_session_persistence_on_error.py`

```
test_user_message_persisted_before_orchestrator
  — ScriptedProvider raises on first call
  — Assert: session file contains user message after exception

test_partial_state_saved_on_exception
  — ScriptedProvider: first call returns tool_calls, raises on second call
  — Assert: session contains user msg + assistant + tool result

test_no_duplicate_user_message
  — ScriptedProvider: normal successful turn
  — Assert: session.messages has exactly one user message for this turn
```

### New: Unit test for error formatter

**File:** `tests/test_error_message_formatter.py`

```
test_error_with_tool_progress
  — state has tool results → message includes "Progress before the error"

test_error_without_tools_is_generic
  — state has no tool results → generic message

test_error_caps_tool_summaries_at_five
  — state has 10 tool results → only 5 shown
```

### Modified: Existing error tests

Tests in `test_agent_loop.py`, `test_no_answer_recovery.py`, and
`golden/test_golden_scenarios.py` that assert on error message substrings
(e.g., `"trouble"`) need assertion updates to match the new formatter output.

## Architectural Compliance

| Rule | Status |
|------|--------|
| Package boundaries | OK — both changes in `agent/`, no cross-boundary imports |
| Pattern 1: Loop Is Dumb | OK — formatter is a pure function, loop just delivers |
| Pattern 6: Stable Core / Volatile Edge | OK — error message content is volatile |
| Pattern 7: One File, One Reason to Change | OK — message_processor changes for persistence, turn_runner for error formatting |
| Pattern 14: Design for Deletion | OK — formatter deletable without touching loop logic |
| Single Pipeline constraint | OK — pre-save is a pipeline step in MessageProcessor |
| Composition root | OK — no new subsystems to wire |
| File size limits | OK — both files well within 500 LOC |
| No backward compatibility artifacts | OK — error message text changes in place |

## What This Does NOT Change

- Session JSONL format (same schema)
- `get_history()` and orphan repair logic
- Context builder, contributors, prompt templates
- Guardrails, tools, memory subsystem
- Consolidation pipeline
