"""Contract tests for the unified alias registry wiring."""

from __future__ import annotations

from pathlib import Path

from nanobot.config.memory import MemoryConfig
from nanobot.memory.store import MemoryStore


def _make_store(tmp_path: Path, **kwargs) -> MemoryStore:
    defaults = {"embedding_provider": "hash", "memory_config": MemoryConfig(graph_enabled=False)}
    defaults.update(kwargs)
    return MemoryStore(tmp_path, **defaults)


class TestAliasRegistryWiring:
    def test_alias_store_accessible(self, tmp_path: Path) -> None:
        """MemoryDatabase exposes alias_store property."""
        store = _make_store(tmp_path)
        assert store.db.alias_store is not None

    def test_registry_seeded_from_config(self, tmp_path: Path) -> None:
        """user_aliases from config are seeded into the alias registry."""
        config = MemoryConfig(graph_enabled=False, user_aliases=["carlos", "the user"])
        store = MemoryStore(tmp_path, embedding_provider="hash", memory_config=config)
        registry = store.alias_registry
        assert registry.resolve("carlos") == "_user_"
        assert registry.resolve("the user") == "_user_"

    def test_registry_seeded_from_linker(self, tmp_path: Path) -> None:
        """Static entity_linker aliases are seeded into the registry."""
        store = _make_store(tmp_path)
        registry = store.alias_registry
        assert registry.resolve("pg") == "postgresql"

    def test_dedup_uses_registry(self, tmp_path: Path) -> None:
        """EventDeduplicator receives the alias registry."""
        config = MemoryConfig(graph_enabled=False, user_aliases=["carlos"])
        store = MemoryStore(tmp_path, embedding_provider="hash", memory_config=config)
        assert store._dedup._alias_registry is not None

    def test_graph_receives_registry(self, tmp_path: Path) -> None:
        """KnowledgeGraph receives the alias registry when graph is enabled."""
        config = MemoryConfig(graph_enabled=True)
        store = MemoryStore(tmp_path, embedding_provider="hash", memory_config=config)
        assert store.graph._alias_registry is not None

    def test_graph_disabled_no_registry(self, tmp_path: Path) -> None:
        """Disabled KnowledgeGraph has no alias registry."""
        store = _make_store(tmp_path)
        assert store.graph._alias_registry is None
