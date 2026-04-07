# Tool-Message Orphan Prevention and Repair

> Spec for fixing orphaned tool_result messages via prevention + repair
> across three layers, plus non-retryable LLM error classification.
>
> Date: 2026-04-07
> Revised: 2026-04-07 (post architectural review)

## Problem Statement

When a session accumulates many tool-call messages (e.g., 39 messages from a
tool-heavy turn), `get_history(max_messages=25)` truncates the window in a way
that leaves **orphaned tool_result messages** — tool responses whose matching
assistant `tool_calls` entry was dropped by the truncation.

These orphaned messages are sent to the Anthropic API, which rejects them with:

```
invalid_request_error: messages.0.content.0: unexpected `tool_use_id` found
in `tool_result` blocks: toolu_01JAeGy5e4S8SKjP7c6S6p4S. Each `tool_result`
block must have a corresponding `tool_use` block in the previous message.
```

The error is deterministic (400 Bad Request), but `_handle_llm_error()` retries
it 5 times with exponential backoff (2s + 4s + 8s + 16s + 30s = ~60 seconds
wasted), then returns a misleading "trouble reaching the language model" message.

### Incident Details

- **Session:** `web:c6caeff4-bb3c-4a1f-8354-40c50f0c8a39`
- **Trace:** `c9189d18f65309d387cd4acde08af16e` (2026-04-07 01:30 UTC)
- **User message:** "Delete this cron job"
- **Session state:** 39 messages; `max_messages=25` window started at msg 15
  (a `tool` result for `load_skill`), whose matching assistant msg 14 was
  outside the window
- **Log evidence:** 5 identical `BadRequestError` entries in `nanobot.log`
  with exponential backoff timing

### Root Cause Analysis

Three layers of orphan repair exist in the codebase, all with the **same blind
spot** — they repair orphaned `tool_calls` (assistant→tool direction) but not
orphaned `tool_results` (tool→assistant direction):

| Layer | File | Forward repair | Reverse repair |
|-------|------|---------------|----------------|
| Session history | `session/manager.py:91-99` | ✓ Strips assistant tool_calls without results | ✗ Missing |
| Provider sanitization | `litellm_provider.py:221-245` | ✓ Strips assistant tool_calls without results | ✗ Missing |
| Compression | `compression.py:88-142` | ✓ Drops middle tool results not in tail | ✗ Missing for tail |

Additionally, `get_history()` has a **silent fallthrough** bug: when no user
message exists in the truncated window (lines 71-75), the alignment loop
completes without `break` and the function continues with orphaned messages.

## Approach

**Prevention + repair with single ownership per technique** (revised from
initial proposal after architectural review).

The initial proposal (Approach A) added identical reverse orphan repair at
all three layers. Architectural review identified this as duplicated logic
across three packages, violating the "single authoritative location" rule
(`architecture-constraints.md`). The revised approach uses **different
techniques at each layer**, each owning a distinct concern:

1. **Prevention** (`get_history()`) — boundary-aware slicing prevents orphans
   structurally. No orphans are created in the first place.
2. **Repair** (`_sanitize_messages()`) — single authoritative reverse orphan
   repair as last-resort safety net before the API call. Catches corruption
   from disk, compression edge cases, or bugs in the prevention layer.
3. **Independent fix** (compression) — tail orphan verification after
   `_paired_drop_tools()`. Compression creates orphans via a different
   mechanism (Phase 3 middle drop) that boundary-aware slicing cannot prevent.
4. **Error classification** (provider + turn runner) — distinct `finish_reason`
   values for non-retryable errors. Fail fast instead of retrying 400s.

### Architectural Review Findings Addressed

| Finding | Severity | Resolution |
|---------|----------|------------|
| Error classification in turn runner violates "dumb loop" principle | BLOCKER | Provider sets distinct `finish_reason` values; loop branches mechanically |
| `usage` dict typed `dict[str, int]`, string values break mypy | BLOCKER | Use distinct `finish_reason` instead of `usage` hack |
| Same algorithm at three layers violates single ownership | CONCERN | Different technique at each layer; reverse repair only in provider |
| `litellm_provider.py` exceeds 500 LOC after changes | CONCERN | Extract `_sanitize_messages()` to `providers/sanitize.py` |
| Error message references `/new` slash command | CONCERN | Generic wording without slash-command reference |
| No migration strategy for existing corrupted sessions | CONCERN | Documented: `get_history()` + `_sanitize_messages()` repair on access |
| Fix 1a fallback not traced through cascading repairs | CONCERN | Boundary-aware slicing replaces the fallback; traced below |
| Cross-cutting test file breaks convention | NITPICK | Tests placed in per-module files; contract tests remain cross-cutting |

### Alternatives Considered

**Approach A (Initial, revised): Bidirectional repair at all three layers.**
Defense-in-depth via identical algorithm at session, provider, and compression.
Rejected: violates single authoritative location rule. Duplicated logic will
drift when tool message format evolves.

**Approach B (Rejected): Single canonical repair in `_sanitize_messages()`.**
Single source of truth but no structural prevention. Session history would
contain invalid sequences only repaired at the last moment. Harder to debug.

**Approach C (Rejected): Structural prevention in `get_history()` only.**
Prevents orphans from slicing but not from compression. Still needs a safety
net somewhere downstream.

## Design

### Fix 1: `get_history()` — Boundary-Aware Slicing

**File:** `nanobot/session/manager.py`, `get_history()` method

Replace the current raw-index truncation + broken user-alignment with
**boundary-aware slicing** that finds a clean message boundary before
truncating. Messages in this system form tool-call cycles (assistant with
tool_calls, followed by N tool results). Slicing mid-cycle creates orphans.
The fix slices at cycle boundaries instead.

#### Current code (lines 68-75):

```python
sliced = unconsolidated[-max_messages:]

# Drop leading non-user messages to avoid orphaned tool_result blocks
for i, m in enumerate(sliced):
    if m.get("role") == "user":
        sliced = sliced[i:]
        break
```

#### Replacement:

```python
# Find the target start index, then walk backward to a clean boundary
target_start = max(0, len(unconsolidated) - max_messages)
boundary = target_start
for i in range(target_start, -1, -1):
    m = unconsolidated[i]
    role = m.get("role")
    if role == "user":
        boundary = i
        break
    if role == "assistant" and not m.get("tool_calls"):
        # Standalone assistant (no pending tool results) is also a clean boundary
        boundary = i
        break
else:
    # No clean boundary found — scan forward from target for any assistant
    for i in range(target_start, len(unconsolidated)):
        if unconsolidated[i].get("role") == "assistant":
            boundary = i
            break
    else:
        boundary = len(unconsolidated)  # yields empty slice

sliced = unconsolidated[boundary:]
```

**Why this works:** A `user` message or a standalone `assistant` message
(one without `tool_calls`) is never part of a tool-call cycle. Starting
from either guarantees no orphaned tool results precede it. Walking backward
from the target index preserves as many messages as possible while finding
a clean boundary.

**Walkthrough of the incident scenario:**

Session has 39 messages. `max_messages=25` → `target_start = 14`.
Message 14 is an assistant with `tool_calls=[load_skill]`. Walk backward:
- msg 14: assistant with tool_calls → not clean (pending results)
- msg 13: tool result → not clean
- msg 12: assistant with tool_calls → not clean
- ...
- msg 5: user → **clean boundary**

`sliced = unconsolidated[5:]` = 34 messages (more than 25, but valid).
The window is larger than `max_messages` but structurally sound. The
compression pipeline (which runs next in the turn runner) will handle
the size reduction while preserving pairings.

**Degenerate case:** If the backward scan reaches index 0 without finding
a clean boundary, the forward scan looks for any assistant message after
`target_start`. If that also fails, `boundary = len(unconsolidated)` yields
an empty slice. This means the entire unconsolidated history is one massive
uninterrupted tool-call chain — the LLM sees only system prompt + current
message. Better than a 400 error.

**Note:** The existing forward orphan repair (lines 91-99, strips
tool_calls from assistant messages that lack results) is preserved. It
handles a different edge case: mid-turn crashes that leave assistant
messages with tool_calls but no corresponding results.

### Fix 2: `_sanitize_messages()` — Reverse Orphan Repair (Single Authority)

**File:** `nanobot/providers/sanitize.py` (extracted from `litellm_provider.py`)

This is the **single authoritative location** for reverse orphan repair
in the codebase. It catches orphans from any source: corrupted sessions
loaded from disk, compression edge cases, or bugs in the prevention layer.

#### Extraction

Extract `_sanitize_messages()` from `litellm_provider.py` (lines 210-245)
into a new module `nanobot/providers/sanitize.py`. This addresses the 535
LOC concern — the provider file loses ~35 lines, the new file starts at
~60 lines (existing forward repair + new reverse repair).

The function remains a `@staticmethod` / module-level pure function. The
provider imports and calls it:

```python
from nanobot.providers.sanitize import sanitize_messages
```

No import boundary violations: `providers/sanitize.py` is within the
`providers/` package.

#### Reverse orphan repair

After the existing forward repair loop (which strips assistant tool_calls
without matching results), add a second pass:

```python
# Reverse: strip tool results without matching assistant tool_calls
assistant_tc_ids: set[str] = set()
for msg in repaired:
    if msg.get("role") == "assistant" and msg.get("tool_calls"):
        for tc in msg["tool_calls"]:
            if tc.get("id"):
                assistant_tc_ids.add(tc["id"])

final = [
    msg for msg in repaired
    if not (
        msg.get("role") == "tool"
        and msg.get("tool_call_id")
        and msg["tool_call_id"] not in assistant_tc_ids
    )
]
return final
```

### Fix 3: Compression Tail Orphan Protection

**File:** `nanobot/context/compression.py`

**Location:** In the callers (`_compress_sync()` and the async
`summarize_and_compress()`), after calling `_paired_drop_tools()`.

After Phase 2 returns the filtered middle, verify the tail for orphaned tool
results. Build a set of all tool_call IDs from assistant messages in both the
returned middle and the tail. Strip tail tool results whose `tool_call_id` is
not in that set.

**Why in the caller, not `_paired_drop_tools()`:** The function's contract is
"filter the middle given the tail." It does not mutate the tail. The tail
verification is a post-condition of the overall compression pipeline. This
keeps `_paired_drop_tools()` single-responsibility.

**Why this is not duplication of Fix 2:** Compression creates orphans via a
different mechanism — `_paired_drop_tools()` drops middle tool results, then
Phase 3 drops/summarizes the entire middle, which can remove assistant
messages whose tool results are in the tail. This is independent of session
truncation. The compression pipeline should produce clean output on its own,
not rely on the provider to repair its mess.

**Edge case addressed:**

```
Middle: [assistant tool_calls=[tc1, tc2]]  [tool result tc1]
Tail:   [tool result tc2]  [user: "next"]
```

After `_paired_drop_tools`, tc1 result is dropped and tc1 call is annotated
with `_result_omitted`. If Phase 3 then drops/summarizes the middle
(including the assistant message with tc2's call), the tail's tc2 result
becomes orphaned. The post-verification step catches this.

### Fix 4: Non-Retryable Error Classification

**Files:** `nanobot/providers/litellm_provider.py` and
`nanobot/agent/turn_runner.py`

#### 4a. Distinct `finish_reason` values (provider layer)

In both `chat()` and `stream_chat()` exception handlers, classify the
exception and set a **distinct `finish_reason`** instead of the generic
`"error"`:

```python
except Exception as e:
    error_str = str(e)
    exc_name = type(e).__name__

    # Classify into distinct finish_reason values
    if "BadRequestError" in exc_name or "invalid_request" in error_str:
        finish = "invalid_request"
    elif "AuthenticationError" in exc_name or "401" in error_str:
        finish = "auth_error"
    elif "RateLimitError" in exc_name or "429" in error_str:
        finish = "rate_limit"
    else:
        finish = "error"  # existing behavior for unknown errors

    return LLMResponse(
        content=f"Error calling LLM: {error_str}",
        finish_reason=finish,
    )
```

Same pattern for `stream_chat()`'s `StreamChunk`.

**Why `finish_reason` instead of a new field or `usage` dict:**
- `finish_reason` is already `str` typed — no type violations.
- `_handle_llm_error()` already branches on `finish_reason` values
  (`"error"`, `"content_filter"`, `"length"`) — this is the existing
  extension point.
- No changes to `LLMResponse` dataclass needed.
- The loop stays dumb — it reacts mechanically to finish reasons it
  receives, with no error taxonomy knowledge.

#### 4b. Fail fast in turn runner (mechanical branches)

In `_handle_llm_error()`, add two new branches following the existing
`content_filter` and `length` pattern:

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

These go **before** the existing `finish_reason == "error"` block so they
take precedence. No retry, no backoff — immediate break.

**Note on `rate_limit`:** The `"rate_limit"` finish reason still falls
through to the existing `"error"` retry logic. Rate limits are transient
and benefit from exponential backoff. A future refinement could give
`rate_limit` its own branch with more retries and longer backoff, but
the existing behavior is acceptable.

#### 4c. Better error message

Add cases to `_build_error_with_progress()`:

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

Generic wording — no slash-command references. The turn runner does not
know about session management commands.

## Migration Strategy

**Existing corrupted sessions are repaired transparently on next access.**

When a user reopens a session with orphaned messages:
1. `get_history()` (Fix 1) uses boundary-aware slicing, which either avoids
   the orphaned region entirely or starts from a clean boundary upstream.
2. If any orphans still reach the provider, `_sanitize_messages()` (Fix 2)
   strips them before the API call.

No separate migration script is needed. The repair is lazy and invisible
to the user. The persisted `.jsonl` file on disk retains the original
messages (including orphans), but they are harmlessly filtered on every
read. If the session is re-saved after a successful turn, the new messages
are valid — old orphans remain in the file but are never included in new
API calls.

## Test Plan

### Tests in per-module test files (following project convention):

**In `tests/test_session_manager.py` — TestGetHistoryBoundarySlicing:**
- `test_boundary_slicing_finds_user_message` — walks back to nearest user
- `test_boundary_slicing_finds_standalone_assistant` — walks back to non-tool-call assistant
- `test_boundary_slicing_no_clean_boundary_returns_empty` — degenerate case
- `test_boundary_slicing_preserves_complete_tool_cycles` — no orphans in output
- `test_forward_orphan_repair_still_works` — regression guard for existing behavior
- `test_id_remapping_preserves_pairing` — `_clamp_tool_id` doesn't break pairings

**In `tests/test_litellm_provider.py` — TestSanitizeMessagesOrphanRepair:**
- `test_orphaned_tool_result_stripped` — tool msg without matching assistant call → dropped
- `test_orphaned_tool_call_stripped` — existing forward repair still works
- `test_bidirectional_orphan_repair` — both directions in one message list
- `test_complete_pairs_untouched` — no repair needed → no mutations

**In `tests/test_compression_coherence.py` — TestCompressionTailOrphanRepair:**
- `test_tail_tool_result_with_call_in_middle` — preserved (before Phase 3)
- `test_tail_tool_result_with_call_nowhere` — stripped
- `test_tail_tool_result_with_call_in_tail` — preserved
- `test_existing_middle_drop_behavior` — regression guard

**In `tests/test_turn_runner.py` — TestLLMErrorClassification:**
- `test_invalid_request_fails_fast` — no retries, immediate break
- `test_auth_error_fails_fast` — no retries, immediate break
- `test_rate_limit_retries_with_backoff` — existing behavior preserved
- `test_unknown_error_retries_with_backoff` — existing behavior preserved
- `test_non_retryable_error_message_no_slash_commands` — generic wording

### Contract tests: `tests/contract/test_message_pairing_contracts.py`

- `test_full_pipeline_no_orphaned_tool_results` — large session, small window, verify invariant
- `test_full_pipeline_with_compression` — same + compression step, verify invariant

### Validation:

After all changes: `make check` (lint + typecheck + import-check +
structure-check + prompt-check) must pass. File sizes must remain within
limits (the `_sanitize_messages` extraction into `providers/sanitize.py`
addresses the `litellm_provider.py` 535 LOC concern).

**Total: ~22 new tests across 5 files. No changes to existing tests.**

## Files Modified

| File | Change | LOC delta |
|------|--------|-----------|
| `nanobot/session/manager.py` | Fix 1: boundary-aware slicing in `get_history()` | +15, -8 |
| `nanobot/providers/sanitize.py` | **New file.** Fix 2: extracted `sanitize_messages()` + reverse orphan repair | +65 |
| `nanobot/providers/litellm_provider.py` | Extract `_sanitize_messages()`, Fix 4a: distinct `finish_reason` values | -35, +20 |
| `nanobot/context/compression.py` | Fix 3: tail orphan verification in callers | +15 |
| `nanobot/agent/turn_runner.py` | Fix 4b + 4c: new `finish_reason` branches + error messages | +15 |
| `tests/test_session_manager.py` | Boundary slicing tests | +80 |
| `tests/test_litellm_provider.py` | Sanitize orphan repair tests | +60 |
| `tests/test_compression_coherence.py` | Tail orphan tests | +50 |
| `tests/test_turn_runner.py` | Error classification tests | +50 |
| `tests/contract/test_message_pairing_contracts.py` | **New file.** Pipeline invariant tests | +80 |

**Total: ~400 LOC added, ~43 removed, across 10 files. 2 new files created.**

## Risks

- **Boundary-aware slicing may return more than `max_messages`.** When the
  clean boundary is far behind the target index, the slice may be larger.
  This is acceptable — the compression pipeline (which runs immediately
  after in the turn runner) handles size reduction while preserving pairings.
  The `max_messages` parameter becomes a soft target, not a hard limit.

- **Dropping orphaned tool results changes what the LLM sees.** Intentional.
  An orphaned tool result without context is noise. The LLM cannot interpret
  a tool result when it has no record of requesting it.

- **Empty history from boundary-aware slicing.** If no clean boundary exists,
  `get_history()` returns `[]`. The LLM sees only system prompt + current
  message. Better than a 400 error.

- **Error classification via string parsing.** The `finish_reason` extraction
  parses exception type names and message substrings. Fragile if litellm
  changes exception class names. Mitigated by the `"error"` fallback —
  unrecognized errors still retry (existing behavior).

- **`providers/sanitize.py` new file.** Adds one file to the `providers/`
  package (currently 9 files; limit is 15). The file is small (~65 LOC),
  single-purpose, and avoids the catch-all `utils.py` anti-pattern.

## Deviations

| Spec Section | Deviation | Reason |
|---|---|---|
| Fix 4a code example | Implementation uses `isinstance(exc, litellm.BadRequestError)` instead of string parsing (`"BadRequestError" in exc_name`) | Type-safe matching against litellm's exception hierarchy; eliminates false-positive risk from substring collisions (e.g., "401" in rate-limit messages) |
| Fix 4a `rate_limit` | `rate_limit` finish_reason not implemented; rate limit errors fall through to generic `"error"` | Acceptable per spec ("existing behavior is acceptable"); reduces scope |
| Fix 4a duplication | Error classification extracted to shared `_classify_llm_error()` helper | Eliminates code duplication between `chat()` and `stream_chat()` exception handlers |
| Risks: "string parsing" | No longer applies | Replaced by `isinstance` checks per deviation above |
