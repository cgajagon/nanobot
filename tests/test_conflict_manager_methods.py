"""Tests for methods moved from ProfileStore to ConflictManager."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nanobot.memory.write.conflicts import ConflictManager


def _make_mgr(profile_store=None):
    ps = profile_store or MagicMock()
    return ConflictManager(
        ps,
    )


class TestConflictPair:
    def test_identical_values_return_false(self):
        mgr = _make_mgr()
        assert mgr._conflict_pair("coffee", "coffee") is False

    def test_different_values_no_negation_return_false(self):
        # No negation markers — not a conflict pair
        mgr = _make_mgr()
        assert mgr._conflict_pair("coffee", "tea") is False

    def test_similar_values_return_false(self):
        # Near-identical after normalisation should not be a conflict
        mgr = _make_mgr()
        assert mgr._conflict_pair("I like coffee", "I like coffee") is False

    def test_negation_with_overlap_returns_true(self):
        # One side has "not", other doesn't, with enough token overlap
        mgr = _make_mgr()
        assert mgr._conflict_pair("I like coffee", "I do not like coffee") is True

    def test_empty_values_return_false(self):
        mgr = _make_mgr()
        assert mgr._conflict_pair("", "tea") is False
        assert mgr._conflict_pair("coffee", "") is False


class TestHasOpenConflict:
    def test_returns_false_when_no_conflicts(self):
        mgr = _make_mgr()
        profile = {"conflicts": []}
        assert mgr.has_open_conflict(profile, "preferences") is False

    def test_returns_true_when_open_conflict_exists(self):
        mgr = _make_mgr()
        profile = {
            "conflicts": [{"field": "preferences", "status": "open", "old": "coffee", "new": "tea"}]
        }
        assert mgr.has_open_conflict(profile, "preferences") is True

    def test_returns_false_when_conflict_resolved(self):
        mgr = _make_mgr()
        profile = {
            "conflicts": [
                {"field": "preferences", "status": "resolved", "old": "coffee", "new": "tea"}
            ]
        }
        assert mgr.has_open_conflict(profile, "preferences") is False

    def test_returns_false_for_different_field(self):
        mgr = _make_mgr()
        profile = {
            "conflicts": [{"field": "stable_facts", "status": "open", "old": "a", "new": "b"}]
        }
        assert mgr.has_open_conflict(profile, "preferences") is False

    def test_returns_true_for_needs_user_status(self):
        mgr = _make_mgr()
        profile = {
            "conflicts": [{"field": "preferences", "status": "needs_user", "old": "x", "new": "y"}]
        }
        assert mgr.has_open_conflict(profile, "preferences") is True

    def test_returns_false_when_no_conflicts_key(self):
        mgr = _make_mgr()
        profile = {}
        assert mgr.has_open_conflict(profile, "preferences") is False


def _make_real_store(tmp_path):
    """Build a minimal ProfileStore for delegation tests."""
    from nanobot.memory.persistence.profile_io import ProfileStore

    store = ProfileStore()
    return store


class TestApplyProfileUpdates:
    def _make_mgr_with_store(self, tmp_path):
        from nanobot.memory.write.conflicts import ConflictManager

        store = _make_real_store(tmp_path)
        mgr = ConflictManager(
            store,
        )
        store._conflict_mgr = mgr
        return mgr, store

    def test_returns_tuple_of_three_ints(self, tmp_path):
        mgr, _ = self._make_mgr_with_store(tmp_path)
        profile = {
            "preferences": [],
            "stable_facts": [],
            "active_projects": [],
            "relationships": [],
            "constraints": [],
            "conflicts": [],
            "meta": {
                "preferences": {},
                "stable_facts": {},
                "active_projects": {},
                "relationships": {},
                "constraints": {},
            },
        }
        result = mgr._apply_profile_updates(
            profile,
            {"preferences": ["likes hiking"]},
            enable_contradiction_check=False,
        )
        assert isinstance(result, tuple) and len(result) == 3
        added, conflicts, touched = result
        assert isinstance(added, int)
        assert isinstance(conflicts, int)
        assert isinstance(touched, int)

    def test_adds_new_preference(self, tmp_path):
        mgr, _ = self._make_mgr_with_store(tmp_path)
        profile = {
            "preferences": [],
            "stable_facts": [],
            "active_projects": [],
            "relationships": [],
            "constraints": [],
            "conflicts": [],
            "meta": {
                "preferences": {},
                "stable_facts": {},
                "active_projects": {},
                "relationships": {},
                "constraints": {},
            },
        }
        added, conflicts, touched = mgr._apply_profile_updates(
            profile,
            {"preferences": ["likes coffee"]},
            enable_contradiction_check=False,
        )
        assert added >= 1
        assert "likes coffee" in profile["preferences"]

    def test_no_duplicate_added(self, tmp_path):
        mgr, _ = self._make_mgr_with_store(tmp_path)
        profile = {
            "preferences": ["likes coffee"],
            "stable_facts": [],
            "active_projects": [],
            "relationships": [],
            "constraints": [],
            "conflicts": [],
            "meta": {
                "preferences": {},
                "stable_facts": {},
                "active_projects": {},
                "relationships": {},
                "constraints": {},
            },
        }
        added, conflicts, touched = mgr._apply_profile_updates(
            profile,
            {"preferences": ["likes coffee"]},
            enable_contradiction_check=False,
        )
        assert added == 0
        assert profile["preferences"].count("likes coffee") == 1

    def test_unknown_key_is_skipped(self, tmp_path):
        mgr, _ = self._make_mgr_with_store(tmp_path)
        profile = {
            "preferences": [],
            "stable_facts": [],
            "active_projects": [],
            "relationships": [],
            "constraints": [],
            "conflicts": [],
            "meta": {
                "preferences": {},
                "stable_facts": {},
                "active_projects": {},
                "relationships": {},
                "constraints": {},
            },
        }
        # "hobbies" is not a valid PROFILE_KEYS key — should be ignored
        added, conflicts, touched = mgr._apply_profile_updates(
            profile,
            {"hobbies": ["hiking"]},
            enable_contradiction_check=False,
        )
        assert added == 0


def _empty_profile() -> dict:
    """Return a minimal empty profile for testing."""
    return {
        "preferences": [],
        "stable_facts": [],
        "active_projects": [],
        "relationships": [],
        "constraints": [],
        "conflicts": [],
        "meta": {
            "preferences": {},
            "stable_facts": {},
            "active_projects": {},
            "relationships": {},
            "constraints": {},
        },
    }


class TestProfileEchoGuard:
    """Profile belief confidence should not be bumped by assistant echo."""

    def _make_mgr_with_store(self):
        from nanobot.memory.persistence.profile_io import ProfileStore
        from nanobot.memory.write.conflicts import ConflictManager

        store = ProfileStore()
        mgr = ConflictManager(store)
        store._conflict_mgr = mgr
        return mgr, store

    def test_existing_belief_not_bumped_for_assistant_echo(self) -> None:
        """source_role='assistant' should skip confidence bump for existing beliefs."""
        mgr, store = self._make_mgr_with_store()
        profile = _empty_profile()
        profile["preferences"] = ["likes coffee"]
        # Seed the meta entry with known confidence.
        entry = store._meta_entry(profile, "preferences", "likes coffee")
        entry["confidence"] = 0.70

        mgr._apply_profile_updates(
            profile,
            {"preferences": ["likes coffee"]},
            enable_contradiction_check=False,
            source_role="assistant",
        )

        updated = store._meta_entry(profile, "preferences", "likes coffee")
        # Confidence should remain at 0.70 (no +0.03 bump).
        assert updated["confidence"] == 0.70

    def test_existing_belief_bumped_for_genuine_source(self) -> None:
        """source_role='consolidation' should bump confidence normally."""
        mgr, store = self._make_mgr_with_store()
        profile = _empty_profile()
        profile["preferences"] = ["likes coffee"]
        entry = store._meta_entry(profile, "preferences", "likes coffee")
        entry["confidence"] = 0.70

        mgr._apply_profile_updates(
            profile,
            {"preferences": ["likes coffee"]},
            enable_contradiction_check=False,
            source_role="consolidation",
        )

        updated = store._meta_entry(profile, "preferences", "likes coffee")
        assert updated["confidence"] == pytest.approx(0.73)

    def test_new_belief_not_boosted_for_assistant_echo(self) -> None:
        """source_role='assistant' should skip the +0.1 boost for new beliefs."""
        mgr, store = self._make_mgr_with_store()
        profile = _empty_profile()

        mgr._apply_profile_updates(
            profile,
            {"preferences": ["likes hiking"]},
            enable_contradiction_check=False,
            source_role="assistant",
        )

        entry = store._meta_entry(profile, "preferences", "likes hiking")
        # Created at 0.65, no +0.1 boost for echo -> stays 0.65.
        assert entry["confidence"] == 0.65

    def test_new_belief_boosted_for_genuine_source(self) -> None:
        """source_role='consolidation' should apply the +0.1 boost for new beliefs."""
        mgr, store = self._make_mgr_with_store()
        profile = _empty_profile()

        mgr._apply_profile_updates(
            profile,
            {"preferences": ["likes hiking"]},
            enable_contradiction_check=False,
            source_role="consolidation",
        )

        entry = store._meta_entry(profile, "preferences", "likes hiking")
        assert entry["confidence"] == pytest.approx(0.75)

    def test_default_source_role_is_genuine(self) -> None:
        """Empty source_role defaults to genuine (backward compat)."""
        mgr, store = self._make_mgr_with_store()
        profile = _empty_profile()
        profile["preferences"] = ["likes coffee"]
        entry = store._meta_entry(profile, "preferences", "likes coffee")
        entry["confidence"] = 0.70

        # No source_role argument — should default to genuine behavior.
        mgr._apply_profile_updates(
            profile,
            {"preferences": ["likes coffee"]},
            enable_contradiction_check=False,
        )

        updated = store._meta_entry(profile, "preferences", "likes coffee")
        assert updated["confidence"] == pytest.approx(0.73)

    def test_last_seen_still_updated_for_echo(self) -> None:
        """Even for echoes, last_seen_at should be updated."""
        mgr, store = self._make_mgr_with_store()
        profile = _empty_profile()
        profile["preferences"] = ["likes coffee"]
        store._meta_entry(profile, "preferences", "likes coffee")

        mgr._apply_profile_updates(
            profile,
            {"preferences": ["likes coffee"]},
            enable_contradiction_check=False,
            source_role="assistant",
        )

        updated = store._meta_entry(profile, "preferences", "likes coffee")
        # last_seen_at should be updated (or at least present).
        assert updated.get("last_seen_at")
        # Confidence should NOT have changed.
        assert updated["confidence"] == 0.65
