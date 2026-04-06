"""Tests for nanobot.memory.write.dedup — EventDeduplicator."""

from __future__ import annotations

import sqlite3

from nanobot.memory.constants import ALIAS_REGISTRY_DDL
from nanobot.memory.db.alias_store import AliasRegistry, AliasStore
from nanobot.memory.write.classification import EventClassifier
from nanobot.memory.write.coercion import EventCoercer
from nanobot.memory.write.dedup import EventDeduplicator


def _make_dedup(
    *,
    conflict_pair_fn: object = None,
    embedder: object = None,
    alias_registry: object = None,
) -> EventDeduplicator:
    classifier = EventClassifier()
    coercer = EventCoercer(classifier)
    return EventDeduplicator(
        coercer=coercer,
        conflict_pair_fn=conflict_pair_fn,
        embedder=embedder,
        alias_registry=alias_registry,
    )


def _make_registry(aliases: dict[str, str] | None = None) -> AliasRegistry:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(ALIAS_REGISTRY_DDL)
    store = AliasStore(conn)
    if aliases:
        for alias, canonical in aliases.items():
            store.register(alias, canonical, confidence=1.0, source="config")
    registry = AliasRegistry(store)
    registry.load()
    return registry


class TestEventSimilarity:
    def test_identical_events(self) -> None:
        d = _make_dedup()
        a = {"type": "fact", "summary": "User likes Python", "entities": ["Python"]}
        lexical, semantic = d.event_similarity(a, a)
        assert lexical == 1.0
        assert semantic == 1.0

    def test_different_events(self) -> None:
        d = _make_dedup()
        a = {"type": "fact", "summary": "User likes Python", "entities": ["Python"]}
        b = {"type": "fact", "summary": "Weather is sunny in Tokyo", "entities": ["Tokyo"]}
        lexical, _ = d.event_similarity(a, b)
        assert lexical < 0.3


class TestFindSemanticDuplicate:
    def test_exact_match(self) -> None:
        d = _make_dedup()
        existing = [
            {"type": "fact", "summary": "User likes Python programming", "entities": ["Python"]}
        ]
        candidate = {
            "type": "fact",
            "summary": "User likes Python programming",
            "entities": ["Python"],
        }
        idx, score = d.find_semantic_duplicate(candidate, existing)
        assert idx == 0
        assert score > 0.5

    def test_different_type_no_match(self) -> None:
        d = _make_dedup()
        existing = [{"type": "fact", "summary": "User likes Python", "entities": ["Python"]}]
        candidate = {"type": "task", "summary": "User likes Python", "entities": ["Python"]}
        idx, _ = d.find_semantic_duplicate(candidate, existing)
        assert idx is None

    def test_no_match(self) -> None:
        d = _make_dedup()
        existing = [{"type": "fact", "summary": "User likes Python", "entities": ["Python"]}]
        candidate = {
            "type": "fact",
            "summary": "Weather in Tokyo is sunny",
            "entities": ["Tokyo"],
        }
        idx, _ = d.find_semantic_duplicate(candidate, existing)
        assert idx is None

    def test_entity_name_mismatch_same_type_matches(self) -> None:
        """'User's project is DS10540' vs 'Carlos's project is DS10540' should merge."""
        d = _make_dedup()
        existing = [{"type": "fact", "summary": "User's primary project is DS10540"}]
        candidate = {"type": "fact", "summary": "Carlos's primary project is DS10540"}
        idx, score = d.find_semantic_duplicate(candidate, existing)
        assert idx == 0

    def test_different_facts_same_type_not_merged(self) -> None:
        """Similar structure but different content should not merge."""
        d = _make_dedup()
        existing = [{"type": "fact", "summary": "User prefers Python for data science"}]
        candidate = {"type": "fact", "summary": "User prefers Java for web development"}
        idx, _ = d.find_semantic_duplicate(candidate, existing)
        assert idx is None

    def test_boundary_just_below_threshold_not_merged(self) -> None:
        """Events with Jaccard just below 0.70 should NOT merge (0.556 here)."""
        d = _make_dedup()
        existing = [{"type": "fact", "summary": "User prefers Python for backend development"}]
        candidate = {"type": "fact", "summary": "Carlos prefers Python for backend services"}
        idx, _ = d.find_semantic_duplicate(candidate, existing)
        assert idx is None


class TestFindSemanticSupersession:
    def test_negation_detected(self) -> None:
        d = _make_dedup()
        existing = [
            {
                "type": "preference",
                "summary": "User likes dark mode",
                "entities": ["dark mode"],
                "memory_type": "semantic",
            }
        ]
        candidate = {
            "type": "preference",
            "summary": "User does not like dark mode",
            "entities": ["dark mode"],
            "memory_type": "semantic",
        }
        idx = d.find_semantic_supersession(candidate, existing)
        assert idx == 0

    def test_episodic_not_superseded(self) -> None:
        d = _make_dedup()
        existing = [{"type": "task", "summary": "Deploy v2", "memory_type": "episodic"}]
        candidate = {"type": "task", "summary": "Deploy v2 canceled", "memory_type": "episodic"}
        idx = d.find_semantic_supersession(candidate, existing)
        assert idx is None


class TestMergeEvents:
    def test_entities_unioned(self) -> None:
        d = _make_dedup()
        base = {
            "id": "ev1",
            "type": "fact",
            "summary": "User likes Python",
            "entities": ["Python"],
            "confidence": 0.7,
            "salience": 0.6,
            "source": "chat",
        }
        incoming = {
            "id": "ev2",
            "type": "fact",
            "summary": "User likes Python",
            "entities": ["Python", "coding"],
            "confidence": 0.8,
            "salience": 0.7,
            "source": "chat",
        }
        merged = d.merge_events(base, incoming, similarity=0.9)
        assert "Python" in merged["entities"]
        assert "coding" in merged["entities"]

    def test_confidence_averaged(self) -> None:
        d = _make_dedup()
        base = {
            "id": "ev1",
            "type": "fact",
            "summary": "A fact",
            "entities": [],
            "confidence": 0.6,
            "salience": 0.5,
            "source": "chat",
        }
        incoming = {
            "id": "ev2",
            "type": "fact",
            "summary": "A fact",
            "entities": [],
            "confidence": 0.8,
            "salience": 0.6,
            "source": "chat",
        }
        merged = d.merge_events(base, incoming, similarity=0.85)
        assert 0.72 < merged["confidence"] < 0.74


class TestEntityAliasNormalization:
    def test_user_alias_normalized_in_similarity(self) -> None:
        """'User likes Python' vs 'Carlos likes Python' should have high similarity with aliases."""
        registry = _make_registry({"user": "_user_", "carlos": "_user_"})
        d = _make_dedup(alias_registry=registry)
        a = {"type": "fact", "summary": "User likes Python"}
        b = {"type": "fact", "summary": "Carlos likes Python"}
        lexical, _ = d.event_similarity(a, b)
        assert lexical >= 0.84  # after normalization, near-identical

    def test_no_aliases_lower_similarity(self) -> None:
        """Without aliases, 'User likes Python' vs 'Carlos likes Python' has lower similarity."""
        d = _make_dedup()
        a = {"type": "fact", "summary": "User likes Python"}
        b = {"type": "fact", "summary": "Carlos likes Python"}
        lexical, _ = d.event_similarity(a, b)
        assert lexical < 0.84  # entity name difference lowers similarity

    def test_alias_normalization_enables_dedup(self) -> None:
        """With aliases, entity name mismatch events are detected as duplicates."""
        registry = _make_registry({"user": "_user_", "carlos": "_user_"})
        d = _make_dedup(alias_registry=registry)
        existing = [{"type": "fact", "summary": "User uses Obsidian for knowledge management"}]
        candidate = {"type": "fact", "summary": "Carlos uses Obsidian for knowledge management"}
        idx, score = d.find_semantic_duplicate(candidate, existing)
        assert idx == 0


class TestEntityNormalizationInSimilarity:
    def test_possessive_entities_match_in_similarity(self) -> None:
        """'User's project' and 'User project' should have high entity overlap."""
        d = _make_dedup()
        a = {"type": "fact", "summary": "Working on the project", "entities": ["User's"]}
        b = {"type": "fact", "summary": "Working on the project", "entities": ["User"]}
        # After normalization, both entity tokens become "user"
        lexical, _ = d.event_similarity(a, b)
        assert lexical >= 0.9

    def test_possessive_entities_match_in_dedup(self) -> None:
        """Entity overlap in find_semantic_duplicate should normalize possessives.

        Summaries differ enough that lexical similarity alone (~0.27) won't trigger
        any merge threshold. Entity overlap is the deciding factor:
        - With _norm_text: "user's" != "user" -> overlap=0 -> no merge
        - With normalize_entity_name: both -> "user" -> overlap=1.0 -> merges
          (entity_overlap >= 0.30 AND lexical >= 0.25 AND same type)
        """
        d = _make_dedup()
        existing = [
            {
                "type": "fact",
                "summary": "Prefers dark mode for coding",
                "entities": ["User's"],
            }
        ]
        candidate = {
            "type": "fact",
            "summary": "Likes dark theme when programming",
            "entities": ["User"],
        }
        idx, score = d.find_semantic_duplicate(candidate, existing)
        assert idx == 0


class TestEmbeddingSemanticSimilarity:
    def test_semantic_differs_from_lexical_with_embedder(self) -> None:
        """With an embedder, semantic should use cosine, not equal lexical."""
        from nanobot.memory.embedder import HashEmbedder

        embedder = HashEmbedder(dims=384)
        d = _make_dedup(embedder=embedder)
        a = {"type": "fact", "summary": "User enjoys programming in Python"}
        b = {"type": "fact", "summary": "Carlos likes coding with Python language"}
        lexical, semantic = d.event_similarity(a, b)
        assert semantic != lexical

    def test_semantic_equals_lexical_without_embedder(self) -> None:
        """Without embedder, semantic falls back to lexical."""
        d = _make_dedup()
        a = {"type": "fact", "summary": "User likes Python"}
        b = {"type": "fact", "summary": "User likes Python and Java"}
        lexical, semantic = d.event_similarity(a, b)
        assert semantic == lexical


class TestMergeSourceSpan:
    def test_basic_merge(self) -> None:
        assert EventDeduplicator.merge_source_span([5, 10], [3, 12]) == [3, 12]

    def test_invalid_base(self) -> None:
        assert EventDeduplicator.merge_source_span("bad", [3, 12]) == [0, 12]

    def test_invalid_incoming(self) -> None:
        assert EventDeduplicator.merge_source_span([5, 10], None) == [5, 10]


class TestAliasRegistryInSimilarity:
    def test_registry_resolves_aliases_in_similarity(self) -> None:
        registry = _make_registry({"carlos": "_user_", "user": "_user_"})
        d = _make_dedup(alias_registry=registry)
        a = {"type": "fact", "summary": "Carlos likes Python", "entities": ["Carlos"]}
        b = {"type": "fact", "summary": "User likes Python", "entities": ["User"]}
        lexical, _ = d.event_similarity(a, b)
        # With alias resolution, "carlos" and "user" both become "_user_"
        assert lexical >= 0.9


class TestSourceRolePreservation:
    def test_coercion_preserves_source_role(self) -> None:
        classifier = EventClassifier()
        coercer = EventCoercer(classifier)
        raw = {
            "type": "fact",
            "summary": "User uses Python",
            "source_role": "user",
        }
        event = coercer.coerce_event(raw, source_span=[0, 1])
        assert event is not None
        assert event.source_role == "user"

    def test_coercion_defaults_source_role_empty(self) -> None:
        classifier = EventClassifier()
        coercer = EventCoercer(classifier)
        raw = {"type": "fact", "summary": "User uses Python"}
        event = coercer.coerce_event(raw, source_span=[0, 1])
        assert event is not None
        assert event.source_role == ""

    def test_from_dict_preserves_source_role(self) -> None:
        """MemoryEvent.from_dict preserves source_role (micro-extraction path)."""
        from nanobot.memory.event import MemoryEvent

        event = MemoryEvent.from_dict(
            {
                "type": "fact",
                "summary": "User uses Python",
                "source_role": "user",
            }
        )
        assert event.source_role == "user"

    def test_from_dict_defaults_source_role_empty(self) -> None:
        """MemoryEvent.from_dict defaults source_role to empty string."""
        from nanobot.memory.event import MemoryEvent

        event = MemoryEvent.from_dict(
            {
                "type": "fact",
                "summary": "User uses Python",
            }
        )
        assert event.source_role == ""

    def test_coercion_rejects_invalid_source_role(self) -> None:
        classifier = EventClassifier()
        coercer = EventCoercer(classifier)
        raw = {
            "type": "fact",
            "summary": "User uses Python",
            "source_role": "bogus_value",
        }
        event = coercer.coerce_event(raw, source_span=[0, 1])
        assert event is not None
        assert event.source_role == ""


class TestLastConfirmedInMerge:
    def test_merge_bumps_last_confirmed_for_user_source(self) -> None:
        d = _make_dedup()
        base = {
            "type": "fact",
            "summary": "User uses Python",
            "entities": ["Python"],
            "last_confirmed": "2025-01-01T00:00:00Z",
        }
        incoming = {
            "type": "fact",
            "summary": "User uses Python",
            "entities": ["Python"],
            "source_role": "user",
        }
        merged = d.merge_events(base, incoming, similarity=0.9)
        assert merged["last_confirmed"] > "2025-01-01T00:00:00Z"

    def test_merge_skips_last_confirmed_for_assistant_echo(self) -> None:
        d = _make_dedup()
        base = {
            "type": "fact",
            "summary": "User uses Python",
            "entities": ["Python"],
            "last_confirmed": "2025-06-01T00:00:00Z",
        }
        incoming = {
            "type": "fact",
            "summary": "User uses Python",
            "entities": ["Python"],
            "source_role": "assistant",
        }
        merged = d.merge_events(base, incoming, similarity=0.9)
        assert merged["last_confirmed"] == "2025-06-01T00:00:00Z"

    def test_merge_bumps_last_confirmed_for_consolidation(self) -> None:
        d = _make_dedup()
        base = {
            "type": "fact",
            "summary": "User uses Python",
            "entities": ["Python"],
            "last_confirmed": "2025-01-01T00:00:00Z",
        }
        incoming = {
            "type": "fact",
            "summary": "User uses Python",
            "entities": ["Python"],
            "source_role": "consolidation",
        }
        merged = d.merge_events(base, incoming, similarity=0.9)
        assert merged["last_confirmed"] > "2025-01-01T00:00:00Z"

    def test_merge_bumps_last_confirmed_for_tool_source(self) -> None:
        d = _make_dedup()
        base = {
            "type": "fact",
            "summary": "User uses Python",
            "entities": ["Python"],
            "last_confirmed": "2025-01-01T00:00:00Z",
        }
        incoming = {
            "type": "fact",
            "summary": "User uses Python",
            "entities": ["Python"],
            "source_role": "tool",
        }
        merged = d.merge_events(base, incoming, similarity=0.9)
        assert merged["last_confirmed"] > "2025-01-01T00:00:00Z"

    def test_merge_default_empty_source_role_is_genuine(self) -> None:
        d = _make_dedup()
        base = {
            "type": "fact",
            "summary": "User uses Python",
            "entities": ["Python"],
            "last_confirmed": "2025-01-01T00:00:00Z",
        }
        incoming = {
            "type": "fact",
            "summary": "User uses Python",
            "entities": ["Python"],
            "source_role": "",
        }
        merged = d.merge_events(base, incoming, similarity=0.9)
        assert merged["last_confirmed"] > "2025-01-01T00:00:00Z"
