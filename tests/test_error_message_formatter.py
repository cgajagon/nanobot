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

    def test_content_filter_no_tools_returns_specific_message(self):
        """content_filter error type returns the content-filter message."""
        state = _make_state(
            messages=[
                {"role": "system", "content": "prompt"},
                {"role": "user", "content": "hello"},
            ],
        )
        msg = _build_error_with_progress(state, error_type="content_filter")
        assert "content filter" in msg.lower()
        assert "rephrasing" in msg.lower()
        assert "Progress" not in msg

    def test_length_no_tools_returns_specific_message(self):
        """length error type returns the truncation message."""
        state = _make_state(
            messages=[
                {"role": "system", "content": "prompt"},
                {"role": "user", "content": "hello"},
            ],
        )
        msg = _build_error_with_progress(state, error_type="length")
        assert "too long" in msg.lower()
        assert "more specific" in msg.lower()
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
        assert "- exec" in msg
        assert "- read_file" in msg
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
        assert "x" * 201 not in msg
        assert "x" * 200 in msg
