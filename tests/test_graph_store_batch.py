"""Tests for batch entity lookup in GraphStore and KnowledgeGraph."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from nanobot.memory.db import MemoryDatabase


class TestGraphStoreBatch:
    """GraphStore.get_entities_batch returns correct results."""

    def test_returns_existing_entities(self, tmp_path: Path) -> None:
        db = MemoryDatabase(tmp_path / "test.db", dims=384)
        gs = db.graph_store
        gs.upsert_entity("alice", type="PERSON", first_seen="2025-01-01", last_seen="2025-06-01")
        gs.upsert_entity("bob", type="PERSON", first_seen="2025-01-01", last_seen="2025-03-01")

        result = gs.get_entities_batch({"alice", "bob"})
        assert "alice" in result
        assert "bob" in result
        assert result["alice"]["type"] == "PERSON"
        assert result["alice"]["last_seen"] == "2025-06-01"

    def test_missing_entities_omitted(self, tmp_path: Path) -> None:
        db = MemoryDatabase(tmp_path / "test.db", dims=384)
        gs = db.graph_store
        gs.upsert_entity("alice", type="PERSON", first_seen="2025-01-01", last_seen="2025-06-01")

        result = gs.get_entities_batch({"alice", "nonexistent"})
        assert "alice" in result
        assert "nonexistent" not in result

    def test_empty_input_returns_empty(self, tmp_path: Path) -> None:
        db = MemoryDatabase(tmp_path / "test.db", dims=384)
        gs = db.graph_store
        result = gs.get_entities_batch(set())
        assert result == {}

    def test_single_entity(self, tmp_path: Path) -> None:
        db = MemoryDatabase(tmp_path / "test.db", dims=384)
        gs = db.graph_store
        gs.upsert_entity("charlie", type="PERSON", first_seen="2025-02-01", last_seen="2025-04-01")

        result = gs.get_entities_batch({"charlie"})
        assert len(result) == 1
        assert result["charlie"]["last_seen"] == "2025-04-01"


class TestKnowledgeGraphBatch:
    """KnowledgeGraph.get_entities_batch applies normalization."""

    def test_normalizes_names(self, tmp_path: Path) -> None:
        from nanobot.memory.db.alias_store import AliasRegistry
        from nanobot.memory.graph.graph import KnowledgeGraph

        db = MemoryDatabase(tmp_path / "test.db", dims=384)
        gs = db.graph_store
        gs.upsert_entity("alice", type="PERSON", first_seen="2025-01-01", last_seen="2025-06-01")

        alias_registry = AliasRegistry(db.connection)
        kg = KnowledgeGraph(db=gs, alias_registry=alias_registry)

        result = kg.get_entities_batch({"Alice", "ALICE"})
        found_names = set(result.keys())
        assert found_names & {"Alice", "ALICE"}, f"Expected at least one match, got {found_names}"

    def test_disabled_graph_returns_empty(self) -> None:
        from nanobot.memory.graph.graph import KnowledgeGraph

        kg = KnowledgeGraph(db=None, alias_registry=None)
        assert not kg.enabled
        result = kg.get_entities_batch({"alice"})
        assert result == {}


class TestGraphAugmenterUsesBatch:
    """GraphAugmenter uses batch lookup instead of per-entity loop."""

    def test_collect_uses_batch(self) -> None:
        from nanobot.memory.read.graph_augmentation import GraphAugmenter

        graph = MagicMock()
        graph.enabled = True
        graph.get_related_entity_names_sync = MagicMock(return_value={"alice", "bob"})
        graph.get_entities_batch = MagicMock(
            return_value={
                "alice": {"last_seen": "2025-06-01"},
                "bob": {"last_seen": "2025-03-01"},
            }
        )

        extractor = MagicMock()
        graph_aug = GraphAugmenter(
            graph=graph,
            extractor=extractor,
            read_events_fn=lambda **kw: [],
        )

        with patch(
            "nanobot.memory.read.graph_augmentation.extract_entities",
            return_value=["Web"],
        ):
            result = graph_aug.collect_graph_entity_names("web framework", [])

        graph.get_entities_batch.assert_called_once()
        graph.get_entity_row.assert_not_called()
        assert "alice" in result
        assert "bob" in result
