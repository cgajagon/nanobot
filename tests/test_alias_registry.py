"""Tests for AliasStore and AliasRegistry."""

from __future__ import annotations

import sqlite3

from nanobot.memory.constants import ALIAS_REGISTRY_DDL
from nanobot.memory.db.alias_store import AliasRegistry, AliasStore


def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(ALIAS_REGISTRY_DDL)
    return conn


class TestAliasStore:
    def test_register_and_load_all(self) -> None:
        store = AliasStore(_make_conn())
        store.register("carlos", "user", confidence=1.0, source="config")
        store.register("pg", "postgresql", confidence=0.9, source="linker")
        result = store.load_all()
        assert result == {"carlos": "user", "pg": "postgresql"}

    def test_higher_confidence_wins(self) -> None:
        store = AliasStore(_make_conn())
        store.register("carlos", "person_a", confidence=0.5, source="graph")
        store.register("carlos", "user", confidence=1.0, source="config")
        assert store.get_canonical("carlos") == "user"

    def test_lower_confidence_does_not_overwrite(self) -> None:
        store = AliasStore(_make_conn())
        store.register("carlos", "user", confidence=1.0, source="config")
        store.register("carlos", "person_a", confidence=0.5, source="graph")
        assert store.get_canonical("carlos") == "user"

    def test_get_canonical_missing(self) -> None:
        store = AliasStore(_make_conn())
        assert store.get_canonical("unknown") is None

    def test_remove_by_canonical(self) -> None:
        store = AliasStore(_make_conn())
        store.register("carlos", "user", confidence=1.0, source="config")
        store.register("the_user", "user", confidence=0.9, source="config")
        store.register("pg", "postgresql", confidence=0.9, source="linker")
        store.remove_by_canonical("user")
        result = store.load_all()
        assert result == {"pg": "postgresql"}


class TestAliasRegistry:
    def test_resolve_known_alias(self) -> None:
        store = AliasStore(_make_conn())
        store.register("carlos", "user", confidence=1.0, source="config")
        registry = AliasRegistry(store)
        registry.load()
        assert registry.resolve("Carlos") == "user"

    def test_resolve_unknown_passes_through(self) -> None:
        store = AliasStore(_make_conn())
        registry = AliasRegistry(store)
        registry.load()
        assert registry.resolve("SomeEntity") == "someentity"

    def test_resolve_possessive(self) -> None:
        store = AliasStore(_make_conn())
        store.register("carlos", "user", confidence=1.0, source="config")
        registry = AliasRegistry(store)
        registry.load()
        assert registry.resolve("Carlos's") == "user"

    def test_register_updates_cache(self) -> None:
        store = AliasStore(_make_conn())
        registry = AliasRegistry(store)
        registry.load()
        assert registry.resolve("carlos") == "carlos"  # not registered yet
        registry.register("carlos", "user", confidence=1.0, source="config")
        assert registry.resolve("carlos") == "user"  # now registered

    def test_load_populates_from_store(self) -> None:
        store = AliasStore(_make_conn())
        store.register("pg", "postgresql", confidence=0.9, source="linker")
        registry = AliasRegistry(store)
        # Before load, cache is empty
        assert registry.resolve("pg") == "pg"
        registry.load()
        assert registry.resolve("pg") == "postgresql"
