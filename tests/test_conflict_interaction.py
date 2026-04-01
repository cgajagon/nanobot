"""Tests for conflict_interaction: user action parsing, relevance gating, conflict resolution."""

from __future__ import annotations

from typing import Any

import pytest

from nanobot.memory.persistence.conflict_types import ConflictRecord
from nanobot.memory.write.conflict_interaction import (
    ask_user_for_conflict,
    conflict_relevant_to,
    get_next_user_conflict,
    handle_user_conflict_reply,
    parse_conflict_user_action,
)

# ---------------------------------------------------------------------------
# parse_conflict_user_action
# ---------------------------------------------------------------------------


class TestParseConflictUserAction:
    """Exhaustive tests for the user-facing action parser."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("keep 1", "keep_old"),
            ("1", "keep_old"),
            ("old", "keep_old"),
            ("keep old", "keep_old"),
            ("keep_old", "keep_old"),
        ],
        ids=["keep-1", "bare-1", "old", "keep-old-space", "keep_old-underscore"],
    )
    def test_keep_old_markers(self, text: str, expected: str) -> None:
        assert parse_conflict_user_action(text) == expected

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("keep 2", "keep_new"),
            ("2", "keep_new"),
            ("new", "keep_new"),
            ("keep new", "keep_new"),
            ("keep_new", "keep_new"),
        ],
        ids=["keep-2", "bare-2", "new", "keep-new-space", "keep_new-underscore"],
    )
    def test_keep_new_markers(self, text: str, expected: str) -> None:
        assert parse_conflict_user_action(text) == expected

    @pytest.mark.parametrize(
        "text",
        ["neither", "dismiss", "none", "skip"],
    )
    def test_dismiss_markers(self, text: str) -> None:
        assert parse_conflict_user_action(text) == "dismiss"

    @pytest.mark.parametrize(
        "text",
        ["merge", "combine"],
    )
    def test_merge_markers(self, text: str) -> None:
        assert parse_conflict_user_action(text) == "merge"

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("KEEP 1", "keep_old"),
            ("Keep 2", "keep_new"),
            ("MERGE", "merge"),
            ("Neither", "dismiss"),
        ],
        ids=["upper-keep-1", "title-keep-2", "upper-merge", "title-neither"],
    )
    def test_case_insensitive(self, text: str, expected: str) -> None:
        assert parse_conflict_user_action(text) == expected

    def test_empty_string(self) -> None:
        assert parse_conflict_user_action("") is None

    def test_none_input(self) -> None:
        assert parse_conflict_user_action(None) is None  # type: ignore[arg-type]

    def test_unrecognized_text(self) -> None:
        assert parse_conflict_user_action("I want to keep both") is None

    def test_whitespace_stripped(self) -> None:
        assert parse_conflict_user_action(" 1 ") == "keep_old"


# ---------------------------------------------------------------------------
# conflict_relevant_to
# ---------------------------------------------------------------------------


class TestConflictRelevantTo:
    """Test relevance gating between conflict and user message."""

    def _make_conflict(
        self, old: str = "prefers dark mode", new: str = "prefers light mode"
    ) -> ConflictRecord:
        return ConflictRecord(
            timestamp="2026-01-01T00:00:00",
            field="preferences",
            old=old,
            new=new,
        )

    def test_empty_message_returns_true(self) -> None:
        conflict = self._make_conflict()
        assert conflict_relevant_to(conflict, "") is True

    def test_high_overlap_returns_true(self) -> None:
        conflict = self._make_conflict()
        # "prefers" and "dark" and "mode" overlap with old
        assert conflict_relevant_to(conflict, "I prefer dark mode") is True

    def test_low_overlap_returns_false(self) -> None:
        conflict = self._make_conflict()
        assert conflict_relevant_to(conflict, "unrelated topic about databases") is False

    def test_empty_conflict_tokens_returns_true(self) -> None:
        conflict = self._make_conflict(old="", new="")
        assert conflict_relevant_to(conflict, "anything") is True

    def test_realistic_conflict(self) -> None:
        conflict = self._make_conflict(
            old="user prefers dark mode",
            new="user prefers light mode",
        )
        # Message about mode preference should be relevant
        assert conflict_relevant_to(conflict, "what mode does the user prefer") is True


# ---------------------------------------------------------------------------
# Mock ConflictManager for integration-like tests
# ---------------------------------------------------------------------------


class MockProfileMgr:
    def __init__(self, profile: dict[str, Any] | None = None) -> None:
        self._profile = profile or {}

    def read_profile(self) -> dict[str, Any]:
        return self._profile

    def write_profile(self, profile: dict[str, Any]) -> None:
        self._profile = profile


class MockConflictManager:
    def __init__(
        self,
        conflicts: list[ConflictRecord] | None = None,
        profile: dict[str, Any] | None = None,
        resolve_result: dict[str, Any] | None = None,
    ) -> None:
        self.profile_mgr = MockProfileMgr(profile or {})
        self._conflicts = conflicts or []
        self._resolve_result = resolve_result or {"ok": True, "db_operation": "update"}

    def list_conflicts(self, *, include_closed: bool = False) -> list[ConflictRecord]:
        return self._conflicts

    def resolve_conflict_details(self, index: int, action: str) -> dict[str, Any]:
        return self._resolve_result


# ---------------------------------------------------------------------------
# get_next_user_conflict
# ---------------------------------------------------------------------------


class TestGetNextUserConflict:
    """Test retrieval of the most-recently-asked conflict."""

    def test_no_conflicts(self) -> None:
        mgr = MockConflictManager(conflicts=[])
        assert get_next_user_conflict(mgr) is None

    def test_conflicts_without_asked_at(self) -> None:
        conflict = ConflictRecord(
            timestamp="2026-01-01T00:00:00",
            field="preferences",
            old="dark",
            new="light",
            asked_at="",  # not asked yet
        )
        mgr = MockConflictManager(conflicts=[conflict])
        assert get_next_user_conflict(mgr) is None

    def test_conflict_with_asked_at(self) -> None:
        conflict = ConflictRecord(
            timestamp="2026-01-01T00:00:00",
            field="preferences",
            old="dark",
            new="light",
            asked_at="2026-01-02T00:00:00",
            index=0,
        )
        mgr = MockConflictManager(conflicts=[conflict])
        result = get_next_user_conflict(mgr)
        assert result is not None
        assert result.old == "dark"


# ---------------------------------------------------------------------------
# ask_user_for_conflict
# ---------------------------------------------------------------------------


class TestAskUserForConflict:
    """Test conflict prompt generation."""

    def test_no_conflicts_in_profile(self) -> None:
        mgr = MockConflictManager(profile={"conflicts": []})
        assert ask_user_for_conflict(mgr) is None

    def test_conflict_with_needs_user_status(self) -> None:
        conflict_dict = ConflictRecord(
            timestamp="2026-01-01T00:00:00",
            field="preferences",
            old="dark mode",
            new="light mode",
            status="needs_user",
            old_last_seen_at="2026-01-01",
            new_last_seen_at="2026-01-02",
        ).to_dict()
        mgr = MockConflictManager(profile={"conflicts": [conflict_dict]})
        result = ask_user_for_conflict(mgr)
        assert result is not None
        assert "dark mode" in result
        assert "light mode" in result
        assert "keep 1" in result

    def test_relevance_gating_filters_irrelevant(self) -> None:
        conflict_dict = ConflictRecord(
            timestamp="2026-01-01T00:00:00",
            field="preferences",
            old="dark mode",
            new="light mode",
            status="needs_user",
        ).to_dict()
        mgr = MockConflictManager(profile={"conflicts": [conflict_dict]})
        # Message about databases should not surface a mode conflict
        result = ask_user_for_conflict(mgr, user_message="tell me about databases")
        assert result is None

    def test_non_needs_user_status_skipped(self) -> None:
        conflict_dict = ConflictRecord(
            timestamp="2026-01-01T00:00:00",
            field="preferences",
            old="dark",
            new="light",
            status="resolved",
        ).to_dict()
        mgr = MockConflictManager(profile={"conflicts": [conflict_dict]})
        assert ask_user_for_conflict(mgr) is None


# ---------------------------------------------------------------------------
# handle_user_conflict_reply
# ---------------------------------------------------------------------------


class TestHandleUserConflictReply:
    """Test end-to-end conflict reply handling."""

    def test_valid_action_with_conflict(self) -> None:
        conflict = ConflictRecord(
            timestamp="2026-01-01T00:00:00",
            field="preferences",
            old="dark",
            new="light",
            asked_at="2026-01-02T00:00:00",
            index=0,
        )
        mgr = MockConflictManager(conflicts=[conflict])
        result = handle_user_conflict_reply(mgr, "keep 1")
        assert result["handled"] is True
        assert result["ok"] is True

    def test_no_action_parsed(self) -> None:
        mgr = MockConflictManager()
        result = handle_user_conflict_reply(mgr, "I want something else")
        assert result["handled"] is False

    def test_no_conflict_available(self) -> None:
        mgr = MockConflictManager(conflicts=[])
        result = handle_user_conflict_reply(mgr, "keep 1")
        assert result["handled"] is False

    def test_merge_resolves_as_keep_new(self) -> None:
        conflict = ConflictRecord(
            timestamp="2026-01-01T00:00:00",
            field="preferences",
            old="dark",
            new="light",
            asked_at="2026-01-02T00:00:00",
            index=0,
        )
        mgr = MockConflictManager(conflicts=[conflict])
        result = handle_user_conflict_reply(mgr, "merge")
        assert result["handled"] is True
        assert result["ok"] is True
        assert "keep_new" in result["message"]

    def test_resolve_failure(self) -> None:
        conflict = ConflictRecord(
            timestamp="2026-01-01T00:00:00",
            field="preferences",
            old="dark",
            new="light",
            asked_at="2026-01-02T00:00:00",
            index=0,
        )
        mgr = MockConflictManager(
            conflicts=[conflict],
            resolve_result={"ok": False},
        )
        result = handle_user_conflict_reply(mgr, "keep 1")
        assert result["handled"] is True
        assert result["ok"] is False
