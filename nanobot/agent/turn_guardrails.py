"""Guardrail layer for the Plan-Act-Observe-Reflect loop.

Each guardrail inspects tool-call history and returns an ``Intervention``
(a system-message injection) when it detects a problematic pattern.
``GuardrailChain`` runs guardrails in priority order; first intervention wins.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from nanobot.agent.turn_types import ToolAttempt
from nanobot.tools.base import ToolResult


def _make_classify_result(attempt: ToolAttempt) -> ToolResult:
    """Build a minimal ToolResult for failure classification from ToolAttempt fields."""
    return ToolResult(
        output=attempt.error_snippet,
        success=False,
        error=attempt.error_snippet,
        metadata={"error_type": attempt.error_type},
    )


def _canonical_args(arguments: dict[str, Any]) -> str:
    """Canonical string representation of tool arguments for dedup.

    Uses ``json.dumps(sort_keys=True)`` so dict key order is deterministic.
    Values are never compared by ``sorted()`` — avoids ``TypeError`` on
    mixed types (str vs None vs int) that real tool arguments contain.
    """
    return json.dumps(arguments, sort_keys=True)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class Intervention:
    """Instruction injected into the message stream by a guardrail."""

    source: str  # guardrail name (for observability)
    message: str  # system message content to inject
    severity: str  # "hint" | "directive" | "override"
    strategy_tag: str | None = None  # for procedural memory extraction


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------


class GuardrailChain:
    """Runs guardrails in order; first intervention wins."""

    def __init__(self, guardrails: list[Any]) -> None:
        self._guardrails = guardrails

    def check(
        self,
        all_attempts: list[ToolAttempt],
        latest_results: list[ToolAttempt],
        *,
        iteration: int = 0,
        **kwargs: Any,
    ) -> Intervention | None:
        for guardrail in self._guardrails:
            result: Intervention | None = guardrail.check(
                all_attempts, latest_results, iteration=iteration, **kwargs
            )
            if result is not None:
                return result
        return None


# ---------------------------------------------------------------------------
# Guardrail implementations
# ---------------------------------------------------------------------------


class EmptyResultRecovery:
    """Fires when a tool returns success but no meaningful data."""

    @property
    def name(self) -> str:
        return "empty_result_recovery"

    def check(
        self,
        all_attempts: list[ToolAttempt],
        latest_results: list[ToolAttempt],
        *,
        iteration: int = 0,
        **kwargs: Any,
    ) -> Intervention | None:
        empty_latest = [a for a in latest_results if a.success and a.output_empty]
        if not empty_latest:
            return None

        tool = empty_latest[0].tool_name

        # Count prior empties for the same tool across the full history
        prior_empties = sum(
            1
            for a in all_attempts
            if a.tool_name == tool and a.success and a.output_empty and a is not empty_latest[0]
        )

        if prior_empties >= 1:
            return Intervention(
                source=self.name,
                message=(
                    f"Tool '{tool}' returned empty results twice. "
                    "STOP using it this way. Try a structural approach "
                    "such as list_dir to discover available paths."
                ),
                severity="directive",
                strategy_tag=f"empty_recovery:{tool}",
            )

        return Intervention(
            source=self.name,
            message=(
                f"Tool '{tool}' returned success but no data. "
                "Consider trying an alternative approach or different arguments."
            ),
            severity="hint",
            strategy_tag=f"empty_recovery:{tool}",
        )


class RepeatedStrategyDetection:
    """Fires when the same tool+arguments combination appears 3+ times."""

    @property
    def name(self) -> str:
        return "repeated_strategy_detection"

    def check(
        self,
        all_attempts: list[ToolAttempt],
        latest_results: list[ToolAttempt],
        *,
        iteration: int = 0,
        **kwargs: Any,
    ) -> Intervention | None:
        counts: dict[tuple[str, str], int] = {}
        for a in all_attempts:
            key = (a.tool_name, _canonical_args(a.arguments))
            counts[key] = counts.get(key, 0) + 1

        for (tool_name, _), count in counts.items():
            if count >= 3:
                return Intervention(
                    source=self.name,
                    message=(
                        f"Tool '{tool_name}' called {count} times with identical "
                        "arguments. You MUST try a fundamentally different strategy."
                    ),
                    severity="override",
                    strategy_tag="repeated_strategy",
                )
        return None


class SkillTunnelVision:
    """Fires when the agent is stuck using only exec with no useful results."""

    @property
    def name(self) -> str:
        return "skill_tunnel_vision"

    def check(
        self,
        all_attempts: list[ToolAttempt],
        latest_results: list[ToolAttempt],
        *,
        iteration: int = 0,
        **kwargs: Any,
    ) -> Intervention | None:
        if iteration < 3:
            return None

        recent = all_attempts[-6:]
        if len(recent) < 1:
            return None

        all_exec = all(a.tool_name == "exec" for a in recent)
        any_data = any(not a.output_empty for a in recent)

        if all_exec and not any_data:
            return Intervention(
                source=self.name,
                message=(
                    "You have been using only 'exec' with no useful results. "
                    "Try base tools like list_dir and read_file to gather "
                    "information directly."
                ),
                severity="directive",
                strategy_tag="skill_tunnel_vision",
            )
        return None


class NoProgressBudget:
    """Fires when no useful data has been obtained after many iterations."""

    @property
    def name(self) -> str:
        return "no_progress_budget"

    def check(
        self,
        all_attempts: list[ToolAttempt],
        latest_results: list[ToolAttempt],
        *,
        iteration: int = 0,
        **kwargs: Any,
    ) -> Intervention | None:
        if iteration < 4:
            return None

        has_useful = any(a.success and not a.output_empty for a in all_attempts)
        if has_useful:
            return None

        return Intervention(
            source=self.name,
            message=(
                "No tool call has returned useful data after multiple iterations. "
                "Stop calling tools and explain what you tried and why it failed."
            ),
            severity="override",
            strategy_tag="no_progress_budget",
        )


class FailureEscalation:
    """Disables tools after repeated failures or identical successes.

    This guardrail is the one exception to the pure-function rule — it calls
    ``tracker.record_failure()`` and ``tracker.record_success()`` which mutate
    the tracker. This is acceptable because failure tracking is a bookkeeping
    concern, not a reasoning concern.

    Requires ``tracker`` (ToolCallTracker) and ``disabled_tools`` (set[str])
    passed as kwargs from the turn runner via GuardrailChain.
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
        **kwargs: Any,
    ) -> Intervention | None:
        tracker = kwargs.get("tracker")
        disabled_tools = kwargs.get("disabled_tools")
        if tracker is None or disabled_tools is None:
            return None

        messages: list[str] = []
        for attempt in latest_results:
            if not attempt.success:
                result = _make_classify_result(attempt)
                count, fc = tracker.record_failure(attempt.tool_name, attempt.arguments, result)
                if count >= tracker.REMOVE_THRESHOLD or fc.is_permanent:
                    disabled_tools.add(attempt.tool_name)
                    reason = (
                        f"permanently unavailable ({fc.value})"
                        if fc.is_permanent
                        else f"failed {count} times with identical arguments"
                    )
                    messages.append(
                        f"TOOL REMOVED: `{attempt.tool_name}` is {reason} "
                        "and has been disabled. Use a different approach."
                    )
                elif count >= tracker.WARN_THRESHOLD:
                    messages.append(
                        f"STOP: `{attempt.tool_name}` has failed {count} times "
                        "with the same arguments and error. Do NOT call it again "
                        "with the same arguments. Use a different approach or "
                        "provide your best answer."
                    )
            else:
                sc = tracker.record_success(attempt.tool_name, attempt.arguments)
                if sc >= tracker.REPEAT_SUCCESS_THRESHOLD:
                    disabled_tools.add(attempt.tool_name)
                    messages.append(
                        f"TOOL REMOVED: `{attempt.tool_name}` has been called "
                        f"{sc} times with identical arguments and is not making "
                        "progress. It has been disabled. Use a different approach "
                        "or provide your best answer."
                    )

        if messages:
            return Intervention(
                source=self.name,
                message="\n".join(messages),
                severity="directive",
                strategy_tag=None,
            )
        return None
