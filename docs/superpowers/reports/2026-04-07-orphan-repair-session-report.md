# Session Report: Tool-Message Orphan Repair

> Date: 2026-04-07
> Duration: Full day session (investigation → design → implementation → review → merge)
> PRs: #163 (orphan repair, merged), #164 (P2 follow-ups, merged)

---

## 1. Incident Investigation

### Trigger

User reported an error from their last conversation with the nanobot agent.
Message: "Delete this cron job" — agent responded with "I'm having trouble
reaching the language model right now."

### Evidence Gathered

**Langfuse traces:**
- Trace `c9189d18` (2026-04-07 01:30 UTC): 5 LLM calls to claude-haiku-4-5,
  4 returned 0/0 tokens with exponential backoff timing (2s, 4s, 8s, 16s gaps).
- One concurrent haiku call (temperature=0.7, 8361/769 tokens) succeeded — this
  was a compression/consolidation call, not the main loop.
- gpt-4o-mini micro-extraction call also succeeded (different provider).
- Previous trace `d8b96b0b` (2 hours earlier, same session) succeeded with
  12 LLM calls and 244K prompt tokens — a massive status report generation.

**Session file** (`web_c6caeff4-bb3c-4a1f-8354-40c50f0c8a39.jsonl`):
- 39 messages from previous turns (user, assistant with tool_calls, tool results).
- User message "Delete this cron job" at line 41.
- Error response at line 42.

**Application log** (`nanobot.log`):
- 5 identical `BadRequestError` entries with the exact same error:
  ```
  AnthropicException - invalid_request_error: messages.0.content.0:
  unexpected `tool_use_id` found in `tool_result` blocks:
  toolu_01JAeGy5e4S8SKjP7c6S6p4S. Each `tool_result` block must have
  a corresponding `tool_use` block in the previous message.
  ```
- Exponential backoff timing confirmed: attempts at 21:30:30, :33, :38, :47, 21:31:05.

### Root Cause

`get_history(max_messages=25)` truncated the 39-message session starting at
message 15 (a `tool` result for `load_skill`). The matching assistant message
with the `load_skill` tool_call was at message 14 — **outside the window**.

The orphaned tool_result was sent to the Anthropic API, which validated that
each `tool_result` must have a corresponding `tool_use` in the preceding
assistant message. Validation failed → 400 Bad Request.

The error handler (`_handle_llm_error`) treated this as a transient error and
retried 5 times with exponential backoff (total ~60 seconds wasted), then
returned a misleading "trouble reaching the language model" message.

### Systematic Analysis

Three layers of orphan repair existed, all with the **same blind spot**:

| Layer | File | Repaired | Missing |
|-------|------|----------|---------|
| Session history | `session/manager.py:91-99` | Assistant tool_calls without results | Tool results without assistant tool_calls |
| Provider sanitization | `litellm_provider.py:221-245` | Assistant tool_calls without results | Tool results without assistant tool_calls |
| Compression | `compression.py:88-142` | Middle tool results not in tail | Tail tool results whose call is in middle |

Additionally, `get_history()` had a **silent fallthrough bug**: when no user
message existed in the truncated window (lines 71-75), the alignment loop
completed without `break` and the function continued with orphaned messages.

---

## 2. Design Process

### Initial Approach (Approach A)

Add identical bidirectional orphan repair (forward + reverse) at all three
layers. Defense-in-depth.

### Architectural Review Findings

Two architectural reviewers independently evaluated the design against the
project's rules:

**Blockers found:**
1. Error classification logic in turn runner violated the "dumb loop" principle
   (cognitive-architecture.md Pattern 1). The loop must react mechanically to
   `finish_reason` values, not inspect error content.
2. Piggybacking `error_type` on the `usage` dict violated the `dict[str, int]`
   type contract. Would fail mypy.

**Concerns found:**
3. Same algorithm at three layers violated "single authoritative location"
   (architecture-constraints.md). Duplicated logic, not defense-in-depth.
4. `litellm_provider.py` would exceed 500 LOC hard limit.
5. Error message referenced `/new` slash command — leaked CLI concern into
   turn runner.
6. No migration strategy for existing corrupted sessions.

### Revised Approach

**Different technique at each layer** (true defense-in-depth):

1. **Prevention** (`get_history()`) — boundary-aware slicing walks backward to
   find a clean message boundary. Prevents orphans structurally.
2. **Repair** (`_sanitize_messages()`) — single authoritative reverse orphan
   repair as last-resort safety net. Catches corruption, compression edge cases.
3. **Independent fix** (compression) — tail orphan verification after
   `_paired_drop_tools()`. Compression creates orphans via a different mechanism.
4. **Error classification** (provider) — distinct `finish_reason` values
   (`invalid_request`, `auth_error`). Turn runner branches mechanically.

---

## 3. Implementation

### PR #163: Tool-Message Orphan Prevention and Repair

**9 commits, ~185 net LOC across 5 source files + 5 test files.**

| Commit | Task | Description |
|--------|------|-------------|
| `13c4f8a` | 1 | Extract `sanitize_messages` to `providers/sanitize.py` (pure refactor) |
| `34d7213` | 2 | Add reverse orphan repair — strip tool results without matching assistant tool_calls |
| `1f871a5` | 3 | Boundary-aware slicing in `get_history()` — walks backward to clean boundary |
| `5848a2a` | 4 | `_strip_tail_orphans()` after `_paired_drop_tools()` in compression |
| `117ff51` | 5 | Distinct `finish_reason` values: `invalid_request`, `auth_error` |
| `44b940d` | 6 | Fail-fast branches in `_handle_llm_error()` — no retries for 400/401 |
| `e9fb87e` | 6+7 | Docstring fix + contract tests |
| `d7bb0cf` | Review fix | Replace string matching with `isinstance` for error classification |
| `6dde371` | P1 fix | Spec deviations doc, forward-scan test, full-pipeline contract test, `__all__` |

**Key implementation decisions:**

- **`_classify_llm_error()` uses `isinstance`** against `litellm.BadRequestError`
  and `litellm.AuthenticationError` instead of fragile string matching. This was
  a review-driven improvement — the original spec proposed string parsing, but
  the security review found `"401" in error_str` could false-positive on rate
  limit messages like "Retry after 1401ms".

- **`_find_clean_boundary()` extracted** (PR #164) from `get_history()` to
  eliminate a confusing `for/else` construct with a nested `found` flag. The
  function scans backward then forward for a user message or standalone
  assistant (no tool_calls).

- **Provider sanitization is the single authoritative repair point.** Session
  prevention and compression tail-orphan stripping are different techniques
  at different layers, not duplicated logic.

### PR #164: P2 Review Follow-ups

**3 commits: docs update, refactor, 2 new tests.**

| Commit | Description |
|--------|-------------|
| `27d8191` | Extract `_find_clean_boundary()` from `get_history()` + stream_chat test |
| `912abed` | Document `invalid_request`/`auth_error` in cognitive-architecture.md |
| `f3b75a0` | Add ID remapping pairing test for `_clamp_tool_id` |

---

## 4. Comprehensive Code Review

### Process

5-phase review with 10+ specialized subagent reviewers:

| Phase | Reviewers | Findings |
|-------|-----------|----------|
| 1. Code Quality + Architecture | code-reviewer + architect-review | 3 high, 3 medium |
| 2. Security + Performance | security-auditor + performance | 2 medium security, 1 medium perf |
| 3. Testing + Documentation | test-engineer + docs-architect | 2 high testing, 1 high docs |
| 4. Best Practices | python-expert | 3 medium |
| 5. Consolidated Report | — | 0 critical, 3 high, 5 medium, 5 low |

### Key Findings and Resolutions

**False alarm — "unrelated memory changes":**
Both Phase 1 reviewers flagged "out-of-scope memory module changes" as critical
scope violations. Investigation revealed this was a **merge gap**: the branch
was forked from main before PR #162 (`perf/memory-perf-optimizations`) merged.
The diff showed the perf changes as "missing" from our branch. A rebase resolved
it. No code on the branch touched memory modules.

**Security — `"401"` false-positive (fixed during review):**
`"401" in error_str` could misclassify a retryable rate limit error as a terminal
`auth_error`. Fixed by replacing string matching with `isinstance` checks against
litellm exception types. Regression test added: `test_chat_rate_limit_with_401_substring_not_misclassified`.

**Architecture — dumb loop principle (fixed during design):**
Initial design proposed inspecting error content in the turn runner. Revised to
use distinct `finish_reason` values set by the provider, with mechanical branches
in the turn runner following the existing `content_filter`/`length` pattern.

**Performance — backward scan unbounded (P3, documented):**
The boundary-aware slicing scans backward from `target_start` to index 0.
Theoretical O(M) for pathological sessions, but practical impact is negligible
due to consolidation keeping unconsolidated counts low. Documented as P3 backlog item.

**Documentation — spec deviations (fixed):**
Implementation diverged from spec (isinstance vs string parsing, rate_limit
finish_reason omitted). `## Deviations` section added per change protocol.

### Review Output Files

All stored in `.full-review/` directory on the worktree (cleaned up after merge):
- `00-scope.md` — review target and file list
- `01-quality-architecture.md` — code quality + architecture findings
- `02-security-performance.md` — security + performance findings
- `03-testing-documentation.md` — testing + documentation findings
- `04-best-practices.md` — Python idioms + framework patterns
- `05-final-report.md` — consolidated findings with priority rankings

---

## 5. Test Coverage

### Tests Added (total: ~25 new tests)

**`tests/test_session_manager.py`** (6 new):
- `test_get_history_boundary_slicing_finds_user` — backward scan to user
- `test_get_history_boundary_slicing_finds_standalone_assistant` — backward scan to standalone assistant
- `test_get_history_boundary_no_clean_boundary_returns_empty` — degenerate case
- `test_get_history_boundary_forward_scan_finds_user` — forward scan fallback
- `test_get_history_boundary_preserves_complete_cycles` — 15-cycle stress test
- `test_get_history_empty_unconsolidated` — empty session guard
- `test_get_history_id_remapping_preserves_pairing` — _clamp_tool_id consistency

**`tests/test_litellm_provider.py`** (7 new):
- `test_sanitize_strips_orphaned_tool_result` — reverse repair
- `test_sanitize_keeps_paired_tool_result` — valid pairs preserved
- `test_sanitize_bidirectional_repair` — both directions combined
- `test_chat_invalid_request_sets_finish_reason` — BadRequestError classification
- `test_chat_auth_error_sets_finish_reason` — AuthenticationError classification
- `test_chat_generic_error_keeps_finish_reason_error` — fallback behavior
- `test_chat_rate_limit_with_401_substring_not_misclassified` — security regression
- `test_stream_chat_invalid_request_sets_finish_reason` — streaming path

**`tests/test_compression_coherence.py`** (4 new):
- `test_tail_result_with_call_in_tail_preserved`
- `test_tail_result_with_call_in_middle_preserved`
- `test_tail_result_with_call_nowhere_stripped`
- `test_tail_non_tool_messages_untouched`

**`tests/test_turn_runner.py`** (3 new):
- `test_invalid_request_fails_fast` — no retries, immediate break
- `test_auth_error_fails_fast` — no retries, immediate break
- `test_generic_error_still_retries` — existing behavior preserved

**`tests/contract/test_message_pairing_contracts.py`** (4 new):
- `test_large_session_small_window` — exact incident scenario (39 msgs, max=25)
- `test_all_tool_messages_no_user` — no user messages at all
- `test_multi_tool_batch_split` — multi-tool batch across boundary
- `test_full_pipeline_with_compression` — session + compression + sanitize

### Validation

- `make check`: All pass (lint, typecheck, import-check, structure-check)
- `make pre-push`: All pass (105 tests, coverage gate, merge-readiness)
- Contract tests: 4/4 pass including exact incident reproduction

---

## 6. Remaining Items

### P3 (documented in `.full-review/05-final-report.md`)

| Item | Effort | Description |
|------|--------|-------------|
| `_strip_tail_orphans` docstring | Trivial | "tool_calls was" → "tool_call was" |
| Backward scan cap | Small | Cap `_find_clean_boundary` backward scan at `max_messages` iterations |
| `_classify_llm_error` Literal return | Small | Future `finish_reason` type unification |

### Unresolved: FailureEscalation Guardrail Stub

The `FailureEscalation` guardrail (`turn_guardrails.py:227-244`) is a stub
that always returns `None`. The real failure escalation logic (60 LOC) lives
inline in `turn_runner._execute_tool_batch()` (lines 435-494).

**Created:** Commit `b85b6f7` (2026-03-27) with note "stub for Phase 3
ToolCallTracker integration." Phase 3 never happened.

**Impact:**
- Violates "dumb loop" principle — domain logic in the turn runner
- `turn_runner.py` at 691 LOC (over 500 limit) partly due to this
- Inline failure messages invisible to Langfuse guardrail observability
- Wrong extension point for future failure pattern changes

**Obstacles to migration:**
1. Per-tool-call vs per-batch granularity
2. State mutation (tracker, disabled_tools, nudged_for_final)
3. Message timing (deferred_messages for API ordering)
4. Success tracking scope (not just failures)

**Full analysis saved to:** `memory/project_failure_escalation_stub.md`

---

## 7. Artifacts

### Documents Created

| Document | Path |
|----------|------|
| Design spec | `docs/superpowers/specs/2026-04-07-bidirectional-orphan-repair-design.md` |
| Implementation plan (orphan repair) | `docs/superpowers/plans/2026-04-07-bidirectional-orphan-repair.md` |
| Implementation plan (P2 follow-ups) | `docs/superpowers/plans/2026-04-07-p2-review-followups.md` |
| This report | `docs/superpowers/reports/2026-04-07-orphan-repair-session-report.md` |

### Files Modified (merged to main)

| File | Change |
|------|--------|
| `nanobot/providers/sanitize.py` | **New.** Extracted + bidirectional orphan repair (73 LOC) |
| `nanobot/providers/litellm_provider.py` | Extraction + `_classify_llm_error()` with isinstance |
| `nanobot/session/manager.py` | Boundary-aware slicing via `_find_clean_boundary()` |
| `nanobot/context/compression.py` | `_strip_tail_orphans()` after `_paired_drop_tools()` |
| `nanobot/agent/turn_runner.py` | Fail-fast branches for `invalid_request` / `auth_error` |
| `.claude/rules/cognitive-architecture.md` | Non-retryable error handling documentation |
| `tests/test_session_manager.py` | 7 new boundary slicing + pairing tests |
| `tests/test_litellm_provider.py` | 8 new sanitize + error classification tests |
| `tests/test_compression_coherence.py` | 4 new tail orphan tests |
| `tests/test_turn_runner.py` | 3 new error classification tests |
| `tests/contract/test_message_pairing_contracts.py` | **New.** 4 pipeline invariant tests |

### Memory Entries

| Entry | Path |
|-------|------|
| FailureEscalation stub analysis | `memory/project_failure_escalation_stub.md` |

---

## 8. Methodology Notes

### What Worked Well

- **Langfuse + session file + application log triangulation** — the three data
  sources together pinpointed the exact message (msg 15) and the exact API error
  within minutes.
- **Architectural review before implementation** — caught the `usage` dict type
  violation and the dumb-loop principle violation before code was written,
  avoiding a rework cycle.
- **Comprehensive 5-phase review** — the security reviewer caught the `"401"`
  false-positive that would have caused a functional regression in production
  (retryable rate limits terminated as auth errors).
- **Subagent-driven development with parallel dispatch** — Tasks 4+5 and their
  reviews ran in parallel since they touched different files.
- **Contract tests reproducing the exact incident** — the test
  `test_large_session_small_window` uses 33 messages with max_messages=25,
  directly modeling the incident scenario.

### What Could Be Improved

- **Background agents sometimes failed to commit** — Tasks 4 and 5 ran in
  parallel but one agent's commit picked up the other's staged changes,
  resulting in a wrong commit message that needed manual correction.
- **Pre-commit hooks are slow** — mypy takes 30-60 seconds per commit,
  making the TDD cycle slower than necessary. A mypy daemon would help.
- **The comprehensive review's "unrelated changes" false alarm** wasted
  investigation time. A future improvement: have reviewers check
  `git log branch..main` to detect merge gaps before flagging scope violations.
