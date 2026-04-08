# FailureEscalation Guardrail Stub — Complete Analysis

> Date: 2026-04-07
> Status: Resolved
> Created: Commit `b85b6f7` (2026-03-27)
> Related: PR #163 (orphan repair), cognitive-architecture.md

---

## 1. Executive Summary

The `FailureEscalation` guardrail is a stub that always returns `None`. It was
created as a placeholder during the cognitive architecture redesign (Phase 2)
with the intention of being implemented in Phase 3. Phase 3 never happened.
Instead, the `ToolCallTracker` was wired directly into the turn runner's
`_execute_tool_batch()` method, where 60 lines of behavioral logic now live
inline — violating the project's "dumb loop" principle.

The system works correctly. Tool failures are tracked, tools get disabled, the
user gets informed. The functional behavior matches the architecture's intent.
What's wrong is the **placement**: domain logic in the loop instead of the
guardrail extension point.

---

## 2. What the Architecture Prescribes

The cognitive architecture document (`cognitive-architecture.md:581-627`)
designs FailureEscalation as the **highest-priority guardrail** (Priority 1
in the GuardrailChain). The design specifies:

```python
class FailureEscalation:
    """Fires when a specific tool has failed enough times to be disabled."""

    name = "failure_escalation"

    def check(self, state, latest_results):
        failed = [r for r in latest_results if not r.success]
        if not failed:
            return None

        messages = []
        for result in failed:
            count, fc = state.tracker.record_failure(
                result.tool_name, result.arguments, result
            )
            if count >= ToolCallTracker.REMOVE_THRESHOLD or fc.is_permanent:
                state.disabled_tools.add(result.tool_name)
                messages.append(
                    f"`{result.tool_name}` disabled ({fc.value}). "
                    "Use a different tool."
                )
            elif count >= ToolCallTracker.WARN_THRESHOLD:
                messages.append(
                    f"`{result.tool_name}` has failed {count} times. "
                    "Try different arguments or a different tool."
                )

        if messages:
            return Intervention(
                source=self.name,
                message="\n".join(messages),
                severity="directive",
                strategy_tag=None,
            )
        return None
```

The doc explicitly acknowledges this guardrail is special: *"This guardrail is
the one exception to the 'pure function' rule — it calls
`state.tracker.record_failure()` which mutates the tracker. This is acceptable
because failure tracking is a bookkeeping concern, not a reasoning concern."*

The guardrail registration order in the factory should be:

```python
guardrails = GuardrailChain([
    FailureEscalation(),          # Priority 1: disable broken tools
    NoProgressBudget(),           # Priority 2: stop after too many failures
    RepeatedStrategyDetection(),  # Priority 3: break strategy loops
    EmptyResultRecovery(),        # Priority 4: suggest alternatives
    SkillTunnelVision(),          # Priority 5: fall back to base tools
])
```

---

## 3. What Actually Exists

### 3.1 The Stub (`turn_guardrails.py:227-244`)

```python
class FailureEscalation:
    """Escalates repeated failures into stronger interventions.

    # Full implementation in Phase 3 when ToolCallTracker is wired
    """

    @property
    def name(self) -> str:
        return "failure_escalation"

    def check(
        self,
        all_attempts: list[ToolAttempt],
        latest_results: list[ToolAttempt],
        *,
        iteration: int = 0,
    ) -> Intervention | None:
        return None
```

This is registered in the guardrail chain at Priority 1 (`agent_factory.py:326`):

```python
guardrails = GuardrailChain(
    [
        FailureEscalation(),      # ← Always returns None
        NoProgressBudget(),
        RepeatedStrategyDetection(),
        EmptyResultRecovery(),
        SkillTunnelVision(),
    ]
)
```

It runs on every tool batch and does absolutely nothing. It occupies the highest
priority slot, meaning if it were implemented, it would fire before any other
guardrail. Currently it just wastes a function call.

### 3.2 The Real Logic (`turn_runner.py:435-494`)

The actual failure escalation behavior lives inline in `_execute_tool_batch()`,
a method on `TurnRunner`. This is 60 lines of behavioral logic:

#### Per-tool failure tracking (lines 436-462):

```python
# Failure / success tracking
if not result.success:
    count, fc = state.tracker.record_failure(tc.name, tc.arguments, result)
    if count >= ToolCallTracker.REMOVE_THRESHOLD or fc.is_permanent:
        to_remove.append(tc.name)
        reason = (
            f"permanently unavailable ({fc.value})"
            if fc.is_permanent
            else f"failed {count} times with identical arguments"
        )
        deferred_messages.append(
            {
                "role": "system",
                "content": (
                    f"TOOL REMOVED: `{tc.name}` is {reason} and has been "
                    "disabled. Use a different approach."
                ),
            }
        )
    elif count >= ToolCallTracker.WARN_THRESHOLD:
        deferred_messages.append(
            {
                "role": "system",
                "content": (
                    f"STOP: `{tc.name}` has failed {count} times with the "
                    "same arguments and error. Do NOT call it again with the "
                    "same arguments. Use a different approach or provide your "
                    "best answer."
                ),
            }
        )
```

This handles:
- `WARN_THRESHOLD` (2 identical failures) → warning system message
- `REMOVE_THRESHOLD` (3 identical failures) → tool disabled + removal message
- `fc.is_permanent` (permanent failures like missing API key) → immediate disable

#### Per-tool success tracking (lines 463-475):

```python
else:
    sc = state.tracker.record_success(tc.name, tc.arguments)
    if sc >= ToolCallTracker.REPEAT_SUCCESS_THRESHOLD:
        to_remove.append(tc.name)
        deferred_messages.append(
            {
                "role": "system",
                "content": (
                    f"TOOL REMOVED: `{tc.name}` has been called {sc} times "
                    "with identical arguments and is not making progress. It "
                    "has been disabled. Use a different approach or provide "
                    "your best answer."
                ),
            }
        )
```

This handles:
- `REPEAT_SUCCESS_THRESHOLD` (3 identical successes) → tool disabled. This
  catches loops where a side-effect tool (e.g., `message`) succeeds with
  identical arguments every iteration without advancing the agent's goal.

#### Tool disabling (lines 479-481):

```python
state.messages.extend(deferred_messages)
state.disabled_tools.update(to_remove)
```

The `deferred_messages` are appended AFTER all tool results to maintain
contiguous tool-result message ordering required by OpenAI/Anthropic APIs.
Then `disabled_tools` is updated, which causes the next iteration to filter
those tools out of the tool definitions sent to the LLM.

#### Global failure budget (lines 483-494):

```python
if state.tracker.budget_exhausted:
    state.messages.append(
        {
            "role": "system",
            "content": (
                f"You have {state.tracker.total_failures} failed tool calls "
                "this turn. Stop calling tools and produce your final answer "
                "NOW with whatever information you have."
            ),
        }
    )
    state.nudged_for_final = True
```

This is the nuclear option: when total failures exceed `GLOBAL_BUDGET` (8),
force the agent to stop calling tools entirely.

### 3.3 The Guardrail Checkpoint (`turn_runner.py:504-531`)

After all the inline logic runs, the guardrail chain checkpoint executes:

```python
# Guardrail checkpoint
intervention = self._guardrails.check(
    state.tool_results_log, latest_attempts, iteration=state.iteration
)
if intervention is not None:
    logger.info(
        "Guardrail '{}' fired (severity={}): {}",
        intervention.source,
        intervention.severity,
        intervention.message[:120],
    )
    state.messages.append({"role": "system", "content": intervention.message})
    state.guardrail_activations.append(
        {
            "source": intervention.source,
            "severity": intervention.severity,
            "iteration": state.iteration,
            "message": intervention.message,
            "strategy_tag": intervention.strategy_tag,
            "failed_tool": _failed.tool_name if _failed else "unknown",
            "failed_args": _failed.arguments if _failed else {},
        }
    )
```

This is where guardrail interventions are logged and tracked for observability
(Langfuse) and procedural memory (strategy extraction). The inline failure
messages from `_execute_tool_batch()` **bypass this entirely** — they are
appended directly to `state.messages` without going through the guardrail
activation tracking.

---

## 4. The ToolCallTracker (`failure.py`)

The `ToolCallTracker` class (`nanobot/agent/failure.py:60-207`) is the
stateful component that tracks failures and successes:

```python
class ToolCallTracker:
    WARN_THRESHOLD: ClassVar[int] = 2      # 2nd identical failure → warning
    REMOVE_THRESHOLD: ClassVar[int] = 3    # 3rd identical failure → disable
    REPEAT_SUCCESS_THRESHOLD: ClassVar[int] = 3  # 3rd identical success → disable
    GLOBAL_BUDGET: ClassVar[int] = 8       # 8 total failures → force answer

    def record_failure(self, name, args, result) -> tuple[int, FailureClass]:
        """Record failure; returns (count, classification)."""

    def record_success(self, name, args) -> int:
        """Record success; returns repeated-success count."""

    @property
    def budget_exhausted(self) -> bool:
        """True when total failures >= GLOBAL_BUDGET."""

    @property
    def permanent_failures(self) -> frozenset[str]:
        """Tool names permanently removed this turn."""
```

The `FailureClass` enum classifies failures:
- `PERMANENT_CONFIG` — missing API key, binary not installed
- `PERMANENT_AUTH` — invalid credentials
- `TRANSIENT_TIMEOUT` — network timeout, rate limit
- `TRANSIENT_ERROR` — server 500, temporary failure
- `LOGICAL_ERROR` — wrong arguments, bad input
- `UNKNOWN` — fall-through

Classification uses a priority cascade: explicit `error_type` metadata →
keyword scan of error message → `UNKNOWN` fallback.

A `ToolCallTracker` instance is scoped to a single turn (created fresh at
the start of each `_run_agent_loop` invocation). It is stored in
`TurnState.tracker`.

---

## 5. Why This Matters

### 5.1 Violates the "Dumb Loop" Principle

The cognitive architecture (Pattern 1) states:

> *"The cognitive loop is a mechanical executor — it calls the LLM, runs tools,
> and checks guardrails. It has no domain knowledge, no task understanding, no
> strategy."*

> *"The loop must never contain: task-type detection, domain-specific logic,
> model-specific behavior, strategy selection."*

The inline failure escalation logic makes **policy decisions**:
- Whether a tool should be disabled (count >= threshold)
- What message to show the user (warning vs removal vs budget exhaustion)
- Whether to force a final answer (budget exhaustion)

These are behavioral decisions that belong in the guardrail extension point,
not in the mechanical loop.

The prohibited patterns doc reinforces: *"Domain logic in the loop — the loop
is a dumb tool-use driver. Behavioral fixes go through extension points
(guardrails, context contributors, prompt templates), not by adding
conditionals to TurnRunner."*

### 5.2 Contributes to File Size Violation

`turn_runner.py` is 691 LOC, well over the 500 LOC hard limit. The inline
failure escalation logic accounts for ~60 of those lines. Moving it to the
guardrail would bring the file to ~630 LOC — still over the limit but a
meaningful reduction.

### 5.3 Invisible to Observability

The guardrail activation tracking (lines 508-531) only logs interventions
returned by the `GuardrailChain`. The inline failure messages are appended
directly to `state.messages` and are **invisible** to:

- **Langfuse guardrail metrics** — "which guardrails fire most often?" misses
  all failure escalation events
- **Strategy extraction** — the `StrategyExtractor` looks at
  `state.guardrail_activations` to learn from recoveries. Failure escalation
  recoveries (tool disabled → agent switches to alternative → success) are
  never captured as strategies.
- **The `guardrail_activations` field in `TurnResult`** — downstream consumers
  that analyze guardrail behavior don't see failure escalation events.

### 5.4 Wrong Extension Point for Future Changes

The guardrail plugin pattern exists so behavioral changes are additive:

> *"When a new failure pattern is discovered: 1. Create a new class implementing
> Guardrail protocol. 2. Write unit tests. 3. Add to registration list. 4. No
> changes to TurnRunner."*

With the inline approach, adding a new failure pattern means modifying
`_execute_tool_batch()` — the stable core that should rarely change. The
loop grows with each new pattern instead of the guardrail list growing.

### 5.5 Dead Code Confusion

The stub runs at Priority 1 in every guardrail check and does nothing. Future
sessions encountering it may:
- Waste time trying to understand why the guardrail doesn't fire
- Assume failure escalation is not implemented (missing the inline logic)
- Try to "fix" the stub without realizing the logic exists elsewhere
- Add duplicate logic in the guardrail without removing the inline version

---

## 6. How It Ended Up This Way

The commit history tells the story:

**Commit `b85b6f7` (2026-03-27):**
```
feat(agent): add guardrail layer with 5 initial guardrails

Five guardrails implemented:
- EmptyResultRecovery: escalates hint->directive on repeated empty results
- RepeatedStrategyDetection: fires on 3+ identical tool+args combinations
- SkillTunnelVision: detects exec-only loops with no useful output
- NoProgressBudget: stops tool calls after 4+ iterations with no data
- FailureEscalation: stub for Phase 3 ToolCallTracker integration
```

The cognitive architecture redesign was done in phases:
- **Phase 1-2**: Built the guardrail infrastructure (GuardrailChain, Intervention,
  4 working guardrails + 1 stub)
- **Phase 3 (never happened)**: Migrate the existing ToolCallTracker inline
  logic into the FailureEscalation guardrail

No subsequent commit has touched the stub. The inline logic in
`_execute_tool_batch()` has remained unchanged since before the guardrail
system was introduced.

---

## 7. Obstacles to Migration

### Obstacle 1: Per-Tool-Call vs Per-Batch Granularity

The guardrail chain runs **once per batch** — after all tools in a batch
execute (`turn_runner.py:504`). But the inline logic runs **per tool call**
inside the results loop (lines 436-475). Each individual tool result is
evaluated against the tracker immediately.

**Resolution:** The guardrail can loop over `latest_results` internally.
This is what the cognitive architecture doc's design shows — the guardrail
iterates `for result in failed`. The other guardrails already do similar
iteration (e.g., `RepeatedStrategyDetection` loops over `latest_results`).

### Obstacle 2: State Mutation

Guardrails are supposed to be pure functions (Pattern 2: *"no side effects,
no state mutation, no LLM calls, no I/O"*). But FailureEscalation must:

- Call `state.tracker.record_failure()` — mutates the tracker's internal counts
- Call `state.tracker.record_success()` — mutates the success streak counts
- Add to `state.disabled_tools` — mutates the disabled set
- Set `state.nudged_for_final = True` — mutates state flag

**Resolution:** The doc explicitly acknowledges this exception. The mutation
is bookkeeping (counting failures), not reasoning (deciding strategy). The
other guardrails don't need to mutate state, but this one does by design.

### Obstacle 3: Message Timing and Multiplicity

The inline code uses `deferred_messages` — a list of system messages appended
**after all tool results** to maintain contiguous tool-result ordering
(required by OpenAI/Anthropic APIs). The messages are batched:

```python
# Line 479: After processing ALL tool results in the batch
state.messages.extend(deferred_messages)
```

The guardrail chain runs at line 504, which is **after** `deferred_messages`
are already appended. So the timing is actually correct — a guardrail
Intervention at this point would be appended after tool results.

However, the inline code generates **multiple** messages (one per failed
tool), while a guardrail `Intervention` is a **single** message. The
guardrail could:

**(a)** Join all failure messages into one Intervention:
```python
return Intervention(
    source=self.name,
    message="\n".join(messages),
    severity="directive",
)
```

**(b)** Or the `Intervention` type could be extended to support a list of
messages. But this changes the guardrail protocol for all guardrails, which
is over-engineering for one case.

**Resolution:** Option (a) — join messages. This is what the cognitive
architecture doc's design shows.

### Obstacle 4: Success Tracking Scope

The inline code tracks both **failures** AND **repeated successes** (lines
463-475). A tool that succeeds with identical arguments 3 times gets disabled.
This prevents loops where a side-effect tool (e.g., `message`) succeeds
every iteration without advancing the goal.

The cognitive architecture doc's FailureEscalation design only covers
**failures**. The repeated-success detection is not mentioned.

**Resolution options:**
- **(a)** Include success tracking in FailureEscalation (rename to
  `FailureAndLoopEscalation` or keep the name since "escalation" covers
  both failure and loop detection).
- **(b)** Create a separate `RepeatedSuccessDetection` guardrail. But this
  is already partially covered by `RepeatedStrategyDetection` (which checks
  for identical tool+args). The difference: `RepeatedStrategyDetection`
  fires on the 3rd **attempt** (regardless of success/failure), while the
  inline success tracking fires on the 3rd **success** specifically.

**Recommendation:** Option (a). The success tracking is part of the same
"detect and break loops" concern. Adding a separate guardrail for 12 lines
of logic is over-engineering.

### Obstacle 5: Global Budget as Separate Concern

The global budget check (lines 483-494) is conceptually different from
per-tool failure tracking. It's a **resource limit** ("you've failed too
many times total, stop everything") rather than a per-tool escalation.

**Resolution options:**
- **(a)** Include in FailureEscalation — the guardrail checks both per-tool
  thresholds and the global budget.
- **(b)** Create a separate `GlobalFailureBudget` guardrail at Priority 0
  (even higher than FailureEscalation).
- **(c)** Leave the global budget in the turn runner — it's a resource limit
  like the wall-time check, not a behavioral guardrail.

**Recommendation:** Option (c). The global budget is analogous to the
wall-time limit (line 201) and max-iterations limit (line 197) — both are
resource limits that live in the loop. The guardrail system is for behavioral
pattern detection, not resource accounting.

---

## 8. Proposed Fix Design

### What Moves to the Guardrail

The `FailureEscalation.check()` method handles:
1. Per-tool failure tracking → warn at 2, disable at 3, immediate disable for permanent
2. Per-tool success tracking → disable at 3 identical successes
3. Tool disabling → populate `state.disabled_tools`

### What Stays in the Turn Runner

1. Global failure budget → resource limit, analogous to wall-time/max-iterations
2. `deferred_messages` timing → the guardrail runs after tool results are appended
3. `to_remove` → replaced by direct `state.disabled_tools` mutation in guardrail

### Implementation Sketch

```python
class FailureEscalation:
    """Disables tools after repeated failures or identical successes."""

    @property
    def name(self) -> str:
        return "failure_escalation"

    def check(
        self,
        all_attempts: list[ToolAttempt],
        latest_results: list[ToolAttempt],
        *,
        iteration: int = 0,
        tracker: ToolCallTracker | None = None,
        disabled_tools: set[str] | None = None,
    ) -> Intervention | None:
        if not tracker or disabled_tools is None:
            return None

        messages: list[str] = []
        for attempt in latest_results:
            if not attempt.success:
                # Need the original ToolResult for classification —
                # this is a design gap since ToolAttempt doesn't carry it.
                # Options: extend ToolAttempt, or pass results separately.
                count, fc = tracker.record_failure(
                    attempt.tool_name, attempt.arguments, result=None
                )
                if count >= ToolCallTracker.REMOVE_THRESHOLD or fc.is_permanent:
                    disabled_tools.add(attempt.tool_name)
                    reason = (
                        f"permanently unavailable ({fc.value})"
                        if fc.is_permanent
                        else f"failed {count} times with identical arguments"
                    )
                    messages.append(
                        f"TOOL REMOVED: `{attempt.tool_name}` is {reason}. "
                        "Use a different approach."
                    )
                elif count >= ToolCallTracker.WARN_THRESHOLD:
                    messages.append(
                        f"STOP: `{attempt.tool_name}` has failed {count} "
                        "times. Use a different approach."
                    )
            else:
                sc = tracker.record_success(attempt.tool_name, attempt.arguments)
                if sc >= ToolCallTracker.REPEAT_SUCCESS_THRESHOLD:
                    disabled_tools.add(attempt.tool_name)
                    messages.append(
                        f"TOOL REMOVED: `{attempt.tool_name}` called {sc} "
                        "times with identical arguments. Use a different approach."
                    )

        if messages:
            return Intervention(
                source=self.name,
                message="\n".join(messages),
                severity="directive",
                strategy_tag=None,
            )
        return None
```

### Design Gap: ToolResult Access

The current `ToolAttempt` dataclass doesn't carry the original `ToolResult`
object, which `tracker.record_failure()` needs for `FailureClass`
classification (it reads `result.metadata["error_type"]` and `result.error`):

```python
@dataclass(slots=True, frozen=True)
class ToolAttempt:
    tool_name: str
    arguments: dict
    success: bool
    output_empty: bool
    output_snippet: str
    iteration: int
    # Missing: no reference to ToolResult
```

Options to resolve:
- **(a)** Add `error_type: str | None = None` to `ToolAttempt` — carry just
  the classification, not the full result.
- **(b)** Add `tool_result: ToolResult | None = None` to `ToolAttempt` — but
  this creates a dependency from `agent/turn_types.py` to `tools/base.py`
  (which is already allowed per import rules).
- **(c)** Pass the `ToolResult` list alongside `latest_results` to the
  guardrail chain — but this changes the `GuardrailChain.check()` signature.

**Recommendation:** Option (a). Add `error_snippet: str = ""` and
`error_type: str = "unknown"` to `ToolAttempt`. The turn runner already has
the `ToolResult` when building `ToolAttempt` and can extract these fields.

### Changes Required

| File | Change | LOC |
|------|--------|-----|
| `nanobot/agent/turn_types.py` | Add `error_type` to `ToolAttempt` | +2 |
| `nanobot/agent/turn_guardrails.py` | Implement `FailureEscalation.check()` | +45, -5 |
| `nanobot/agent/turn_runner.py` | Remove inline logic, pass tracker/disabled_tools to guardrail | -55, +5 |
| `tests/test_guardrails.py` | Add FailureEscalation unit tests | +80 |
| `tests/test_turn_runner.py` | Update to verify guardrail-based escalation | +20, -10 |

**Estimated effort:** Medium (2-3 hours including tests and review).

---

## 9. Verification Checklist

Before declaring the migration complete:

- [ ] FailureEscalation.check() returns Intervention on 2nd failure (warn)
- [ ] FailureEscalation.check() returns Intervention on 3rd failure (disable)
- [ ] FailureEscalation.check() handles permanent failures (immediate disable)
- [ ] FailureEscalation.check() handles repeated successes (3x disable)
- [ ] Disabled tools are excluded from next LLM call's tool definitions
- [ ] Failure messages appear in `state.guardrail_activations`
- [ ] Failure messages are visible in Langfuse spans
- [ ] Global budget check remains in turn runner (not moved to guardrail)
- [ ] `turn_runner.py` LOC reduced by ~55 lines
- [ ] All existing `ToolCallTracker` tests still pass
- [ ] All existing turn runner tests still pass
- [ ] No behavior change from the user's perspective
