"""Tests for the guardrail layer (turn_guardrails.py)."""

from __future__ import annotations

import pytest

from nanobot.agent.failure import ToolCallTracker
from nanobot.agent.turn_types import ToolAttempt


def _attempt(
    tool: str = "exec",
    args: dict | None = None,
    success: bool = True,
    empty: bool = False,
    snippet: str = "data",
    iteration: int = 1,
    error_type: str = "unknown",
    error_snippet: str = "",
) -> ToolAttempt:
    return ToolAttempt(
        tool_name=tool,
        arguments=args or {},
        success=success,
        output_empty=empty,
        output_snippet=snippet,
        iteration=iteration,
        error_type=error_type,
        error_snippet=error_snippet,
    )


# ---------------------------------------------------------------------------
# Intervention
# ---------------------------------------------------------------------------


class TestIntervention:
    def test_creation(self) -> None:
        from nanobot.agent.turn_guardrails import Intervention

        iv = Intervention(source="test", message="hello", severity="hint", strategy_tag="tag1")
        assert iv.source == "test"
        assert iv.message == "hello"
        assert iv.severity == "hint"
        assert iv.strategy_tag == "tag1"

    def test_frozen(self) -> None:
        from nanobot.agent.turn_guardrails import Intervention

        iv = Intervention(source="test", message="hello", severity="hint")
        with pytest.raises(AttributeError):
            iv.source = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# GuardrailChain
# ---------------------------------------------------------------------------


class TestGuardrailChain:
    def test_returns_none_when_empty(self) -> None:
        from nanobot.agent.turn_guardrails import GuardrailChain

        chain = GuardrailChain([])
        assert chain.check([], []) is None

    def test_first_intervention_wins(self) -> None:
        from nanobot.agent.turn_guardrails import (
            GuardrailChain,
            Intervention,
        )

        class AlwaysFires:
            name = "always"

            def check(self, all_attempts, latest_results, *, iteration=0) -> Intervention:
                return Intervention(source=self.name, message="fired", severity="hint")

        class NeverReached:
            name = "never"

            def check(self, all_attempts, latest_results, *, iteration=0) -> Intervention:
                return Intervention(source=self.name, message="should not appear", severity="hint")

        chain = GuardrailChain([AlwaysFires(), NeverReached()])
        result = chain.check([], [])
        assert result is not None
        assert result.source == "always"

    def test_skips_non_firing(self) -> None:
        from nanobot.agent.turn_guardrails import (
            GuardrailChain,
            Intervention,
        )

        class NoFire:
            name = "nope"

            def check(self, all_attempts, latest_results, *, iteration=0) -> None:
                return None

        class Fires:
            name = "yes"

            def check(self, all_attempts, latest_results, *, iteration=0) -> Intervention:
                return Intervention(source=self.name, message="got it", severity="directive")

        chain = GuardrailChain([NoFire(), Fires()])
        result = chain.check([], [])
        assert result is not None
        assert result.source == "yes"


# ---------------------------------------------------------------------------
# EmptyResultRecovery
# ---------------------------------------------------------------------------


class TestEmptyResultRecovery:
    def test_no_fire_on_success_with_data(self) -> None:
        from nanobot.agent.turn_guardrails import EmptyResultRecovery

        g = EmptyResultRecovery()
        latest = [_attempt(success=True, empty=False)]
        assert g.check(latest, latest) is None

    def test_no_fire_on_failure(self) -> None:
        from nanobot.agent.turn_guardrails import EmptyResultRecovery

        g = EmptyResultRecovery()
        latest = [_attempt(success=False, empty=True)]
        assert g.check(latest, latest) is None

    def test_hint_on_first_empty(self) -> None:
        from nanobot.agent.turn_guardrails import EmptyResultRecovery

        g = EmptyResultRecovery()
        latest = [_attempt(tool="exec", success=True, empty=True)]
        result = g.check(latest, latest)
        assert result is not None
        assert result.severity == "hint"

    def test_directive_on_second_empty_same_tool(self) -> None:
        from nanobot.agent.turn_guardrails import EmptyResultRecovery

        g = EmptyResultRecovery()
        first = _attempt(tool="exec", success=True, empty=True, iteration=1)
        second = _attempt(tool="exec", success=True, empty=True, iteration=2)
        all_attempts = [first, second]
        result = g.check(all_attempts, [second])
        assert result is not None
        assert result.severity == "directive"

    def test_strategy_tag_present(self) -> None:
        from nanobot.agent.turn_guardrails import EmptyResultRecovery

        g = EmptyResultRecovery()
        latest = [_attempt(tool="exec", success=True, empty=True)]
        result = g.check(latest, latest)
        assert result is not None
        assert result.strategy_tag is not None


# ---------------------------------------------------------------------------
# RepeatedStrategyDetection
# ---------------------------------------------------------------------------


class TestRepeatedStrategyDetection:
    def test_no_fire_on_first(self) -> None:
        from nanobot.agent.turn_guardrails import RepeatedStrategyDetection

        g = RepeatedStrategyDetection()
        a = _attempt(tool="exec", args={"cmd": "ls"})
        assert g.check([a], [a]) is None

    def test_no_fire_different_args(self) -> None:
        from nanobot.agent.turn_guardrails import RepeatedStrategyDetection

        g = RepeatedStrategyDetection()
        attempts = [
            _attempt(tool="exec", args={"cmd": "ls"}),
            _attempt(tool="exec", args={"cmd": "cat foo"}),
            _attempt(tool="exec", args={"cmd": "pwd"}),
        ]
        assert g.check(attempts, [attempts[-1]]) is None

    def test_fires_on_third_similar(self) -> None:
        from nanobot.agent.turn_guardrails import RepeatedStrategyDetection

        g = RepeatedStrategyDetection()
        attempts = [
            _attempt(tool="exec", args={"cmd": "ls"}),
            _attempt(tool="exec", args={"cmd": "ls"}),
            _attempt(tool="exec", args={"cmd": "ls"}),
        ]
        result = g.check(attempts, [attempts[-1]])
        assert result is not None
        assert result.severity == "override"
        assert result.strategy_tag == "repeated_strategy"

    def test_mixed_type_args_no_crash(self) -> None:
        """Real tool arguments have mixed types (str, int, None, list, dict)."""
        from nanobot.agent.turn_guardrails import RepeatedStrategyDetection

        g = RepeatedStrategyDetection()
        real_args = {
            "command": 'obsidian search query="DS10540"',
            "working_dir": None,
            "timeout": 60,
        }
        attempts = [
            _attempt(tool="exec", args=real_args),
            _attempt(tool="exec", args=real_args),
            _attempt(tool="exec", args=real_args),
        ]
        result = g.check(attempts, [attempts[-1]])
        assert result is not None
        assert result.severity == "override"

    def test_nested_dict_args_no_crash(self) -> None:
        """Tool arguments may contain nested dicts and lists."""
        from nanobot.agent.turn_guardrails import RepeatedStrategyDetection

        g = RepeatedStrategyDetection()
        nested_args = {
            "options": {"recursive": True, "depth": 3},
            "tags": ["urgent", "bug"],
            "path": "/foo",
        }
        attempts = [
            _attempt(tool="exec", args=nested_args),
            _attempt(tool="exec", args=nested_args),
            _attempt(tool="exec", args=nested_args),
        ]
        result = g.check(attempts, [attempts[-1]])
        assert result is not None

    def test_empty_args(self) -> None:
        from nanobot.agent.turn_guardrails import RepeatedStrategyDetection

        g = RepeatedStrategyDetection()
        attempts = [
            _attempt(tool="list_dir", args={}),
            _attempt(tool="list_dir", args={}),
            _attempt(tool="list_dir", args={}),
        ]
        result = g.check(attempts, [attempts[-1]])
        assert result is not None


# ---------------------------------------------------------------------------
# _canonical_args
# ---------------------------------------------------------------------------


class TestCanonicalArgs:
    def test_deterministic_key_order(self) -> None:
        from nanobot.agent.turn_guardrails import _canonical_args

        a = {"z": 1, "a": 2}
        b = {"a": 2, "z": 1}
        assert _canonical_args(a) == _canonical_args(b)

    def test_mixed_types(self) -> None:
        from nanobot.agent.turn_guardrails import _canonical_args

        args = {"command": "obsidian search", "working_dir": None, "timeout": 60}
        result = _canonical_args(args)
        assert isinstance(result, str)
        assert "null" in result  # None -> null in JSON

    def test_nested_structures(self) -> None:
        from nanobot.agent.turn_guardrails import _canonical_args

        args = {"options": {"recursive": True}, "tags": ["a", "b"]}
        result = _canonical_args(args)
        assert "recursive" in result
        assert '["a", "b"]' in result

    def test_empty_dict(self) -> None:
        from nanobot.agent.turn_guardrails import _canonical_args

        assert _canonical_args({}) == "{}"


# ---------------------------------------------------------------------------
# SkillTunnelVision
# ---------------------------------------------------------------------------


class TestSkillTunnelVision:
    def test_no_fire_before_iteration_3(self) -> None:
        from nanobot.agent.turn_guardrails import SkillTunnelVision

        g = SkillTunnelVision()
        attempts = [_attempt(tool="exec", empty=True)] * 6
        assert g.check(attempts, [attempts[-1]], iteration=2) is None

    def test_fires_all_exec_no_data(self) -> None:
        from nanobot.agent.turn_guardrails import SkillTunnelVision

        g = SkillTunnelVision()
        attempts = [_attempt(tool="exec", empty=True) for _ in range(6)]
        result = g.check(attempts, [attempts[-1]], iteration=3)
        assert result is not None
        assert result.severity == "directive"
        assert result.strategy_tag == "skill_tunnel_vision"

    def test_no_fire_when_data_returned(self) -> None:
        from nanobot.agent.turn_guardrails import SkillTunnelVision

        g = SkillTunnelVision()
        attempts = [_attempt(tool="exec", empty=True) for _ in range(5)]
        attempts.append(_attempt(tool="exec", empty=False))
        assert g.check(attempts, [attempts[-1]], iteration=3) is None

    def test_no_fire_mixed_tools(self) -> None:
        from nanobot.agent.turn_guardrails import SkillTunnelVision

        g = SkillTunnelVision()
        attempts = [_attempt(tool="exec", empty=True) for _ in range(5)]
        attempts.append(_attempt(tool="read_file", empty=True))
        assert g.check(attempts, [attempts[-1]], iteration=3) is None


# ---------------------------------------------------------------------------
# NoProgressBudget
# ---------------------------------------------------------------------------


class TestNoProgressBudget:
    def test_no_fire_before_4(self) -> None:
        from nanobot.agent.turn_guardrails import NoProgressBudget

        g = NoProgressBudget()
        attempts = [_attempt(success=True, empty=True)] * 4
        assert g.check(attempts, [attempts[-1]], iteration=3) is None

    def test_fires_no_useful_data(self) -> None:
        from nanobot.agent.turn_guardrails import NoProgressBudget

        g = NoProgressBudget()
        attempts = [_attempt(success=True, empty=True)] * 5
        result = g.check(attempts, [attempts[-1]], iteration=4)
        assert result is not None
        assert result.severity == "override"
        assert result.strategy_tag == "no_progress_budget"

    def test_no_fire_some_data(self) -> None:
        from nanobot.agent.turn_guardrails import NoProgressBudget

        g = NoProgressBudget()
        attempts = [_attempt(success=True, empty=True)] * 4
        attempts.append(_attempt(success=True, empty=False))
        assert g.check(attempts, [attempts[-1]], iteration=4) is None


# ---------------------------------------------------------------------------
# ToolAttempt error fields
# ---------------------------------------------------------------------------


class TestToolAttemptErrorFields:
    def test_error_fields_default(self) -> None:
        """New error fields have safe defaults for backward compatibility."""
        a = _attempt()
        assert a.error_type == "unknown"
        assert a.error_snippet == ""

    def test_error_fields_populated(self) -> None:
        """Error fields can be set on failed attempts."""
        a = ToolAttempt(
            tool_name="exec",
            arguments={"cmd": "ls"},
            success=False,
            output_empty=False,
            output_snippet="Error: not found",
            iteration=1,
            error_type="not_found",
            error_snippet="Error: not found",
        )
        assert a.error_type == "not_found"
        assert a.error_snippet == "Error: not found"


# ---------------------------------------------------------------------------
# GuardrailChain kwargs forwarding
# ---------------------------------------------------------------------------


class TestGuardrailChainKwargs:
    def test_extra_kwargs_forwarded(self) -> None:
        """GuardrailChain forwards extra kwargs to guardrails."""
        from nanobot.agent.turn_guardrails import GuardrailChain, Intervention

        class KwargCapture:
            name = "capture"

            def check(self, all_attempts, latest_results, *, iteration=0, **kwargs):
                if "tracker" in kwargs:
                    return Intervention(
                        source=self.name,
                        message=f"got tracker={type(kwargs['tracker']).__name__}",
                        severity="hint",
                    )
                return None

        chain = GuardrailChain([KwargCapture()])
        result = chain.check([], [], tracker="fake_tracker")
        assert result is not None
        assert "got tracker=str" in result.message


# ---------------------------------------------------------------------------
# FailureEscalation
# ---------------------------------------------------------------------------


class TestFailureEscalation:
    """Tests for the FailureEscalation guardrail."""

    def _check(
        self,
        all_attempts: list[ToolAttempt],
        latest: list[ToolAttempt],
        tracker: ToolCallTracker | None = None,
        disabled_tools: set[str] | None = None,
    ):
        from nanobot.agent.turn_guardrails import FailureEscalation

        g = FailureEscalation()
        return g.check(
            all_attempts,
            latest,
            tracker=tracker or ToolCallTracker(),
            disabled_tools=disabled_tools if disabled_tools is not None else set(),
        )

    def test_no_fire_on_success(self) -> None:
        """Successful tool calls produce no intervention."""
        latest = [_attempt(success=True)]
        assert self._check(latest, latest) is None

    def test_no_fire_on_first_failure(self) -> None:
        """First failure just records — no intervention yet."""
        latest = [_attempt(success=False, error_type="unknown", error_snippet="some error")]
        assert self._check(latest, latest) is None

    def test_warn_on_second_identical_failure(self) -> None:
        """WARN_THRESHOLD (2) identical failures produce a warning."""
        tracker = ToolCallTracker()
        args = {"cmd": "ls /nonexistent"}
        tracker.record_failure("exec", args)
        latest = [_attempt(tool="exec", args=args, success=False)]
        result = self._check(latest, latest, tracker=tracker)
        assert result is not None
        assert result.severity == "directive"
        assert "exec" in result.message
        assert "failed" in result.message.lower()

    def test_disable_on_third_identical_failure(self) -> None:
        """REMOVE_THRESHOLD (3) identical failures disable the tool."""
        tracker = ToolCallTracker()
        disabled: set[str] = set()
        args = {"cmd": "ls /nonexistent"}
        tracker.record_failure("exec", args)
        tracker.record_failure("exec", args)
        latest = [_attempt(tool="exec", args=args, success=False)]
        result = self._check(latest, latest, tracker=tracker, disabled_tools=disabled)
        assert result is not None
        assert "TOOL REMOVED" in result.message
        assert "exec" in disabled

    def test_permanent_failure_disables_immediately(self) -> None:
        """Permanent failures (missing API key) disable on first occurrence."""
        tracker = ToolCallTracker()
        disabled: set[str] = set()
        latest = [
            _attempt(
                tool="web_search",
                args={"query": "test"},
                success=False,
                error_type="not_found",
                error_snippet="web_search is not configured",
            )
        ]
        result = self._check(latest, latest, tracker=tracker, disabled_tools=disabled)
        assert result is not None
        assert "TOOL REMOVED" in result.message
        assert "permanently unavailable" in result.message
        assert "web_search" in disabled

    def test_repeated_success_disables_tool(self) -> None:
        """REPEAT_SUCCESS_THRESHOLD (3) identical successes disable the tool."""
        tracker = ToolCallTracker()
        disabled: set[str] = set()
        args = {"content": "hello"}
        tracker.record_success("message", args)
        tracker.record_success("message", args)
        latest = [_attempt(tool="message", args=args, success=True)]
        result = self._check(latest, latest, tracker=tracker, disabled_tools=disabled)
        assert result is not None
        assert "TOOL REMOVED" in result.message
        assert "message" in disabled

    def test_no_fire_without_tracker(self) -> None:
        """Without tracker kwargs, guardrail returns None safely."""
        from nanobot.agent.turn_guardrails import FailureEscalation

        g = FailureEscalation()
        latest = [_attempt(success=False)]
        assert g.check(latest, latest) is None

    def test_multiple_failures_joined(self) -> None:
        """Multiple tool failures in one batch produce a single joined intervention."""
        tracker = ToolCallTracker()
        disabled: set[str] = set()
        args_a = {"cmd": "foo"}
        args_b = {"path": "/bad"}
        tracker.record_failure("exec", args_a)
        tracker.record_failure("read_file", args_b)
        latest = [
            _attempt(tool="exec", args=args_a, success=False),
            _attempt(tool="read_file", args=args_b, success=False),
        ]
        result = self._check(latest, latest, tracker=tracker, disabled_tools=disabled)
        assert result is not None
        assert "exec" in result.message
        assert "read_file" in result.message

    def test_mixed_type_args(self) -> None:
        """Real tool arguments with mixed types (str, int, None) work correctly."""
        tracker = ToolCallTracker()
        real_args = {
            "command": 'obsidian search query="DS10540"',
            "working_dir": None,
            "timeout": 60,
        }
        tracker.record_failure("exec", real_args)
        latest = [_attempt(tool="exec", args=real_args, success=False)]
        result = self._check(latest, latest, tracker=tracker)
        assert result is not None
        assert "exec" in result.message
