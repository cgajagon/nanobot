"""Alias registry -- SQLite storage and in-memory cache for entity aliases.

``AliasStore`` owns the ``alias_registry`` table CRUD.
``AliasRegistry`` provides O(1) in-memory resolution, loaded from the store.
"""

from __future__ import annotations

import sqlite3

from nanobot.memory._text import normalize_entity_name

__all__ = ["AliasRegistry", "AliasStore"]


class AliasStore:
    """SQLite CRUD for the alias_registry table."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def load_all(self) -> dict[str, str]:
        """Load all alias -> canonical mappings."""
        rows = self._conn.execute("SELECT alias, canonical FROM alias_registry").fetchall()
        return {row["alias"]: row["canonical"] for row in rows}

    def register(
        self,
        alias: str,
        canonical: str,
        *,
        confidence: float = 0.8,
        source: str = "config",
    ) -> None:
        """Upsert an alias. Higher confidence wins on conflict."""
        with self._conn:
            self._conn.execute(
                """INSERT INTO alias_registry (alias, canonical, confidence, source)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(alias) DO UPDATE SET
                       canonical = CASE
                           WHEN excluded.confidence > alias_registry.confidence
                           THEN excluded.canonical
                           ELSE alias_registry.canonical
                       END,
                       confidence = MAX(excluded.confidence, alias_registry.confidence),
                       source = CASE
                           WHEN excluded.confidence > alias_registry.confidence
                           THEN excluded.source
                           ELSE alias_registry.source
                       END""",
                (alias, canonical, confidence, source),
            )

    def get_canonical(self, alias: str) -> str | None:
        """Look up a single alias. Returns None if not found."""
        row = self._conn.execute(
            "SELECT canonical FROM alias_registry WHERE alias = ?", (alias,)
        ).fetchone()
        return row["canonical"] if row else None

    def remove_by_canonical(self, canonical: str) -> None:
        """Remove all aliases pointing to a canonical name."""
        with self._conn:
            self._conn.execute("DELETE FROM alias_registry WHERE canonical = ?", (canonical,))


class AliasRegistry:
    """In-memory alias cache backed by AliasStore.

    Call ``load()`` after construction to populate the cache from SQLite.
    Use ``resolve(name)`` for O(1) alias resolution.
    """

    def __init__(self, store: AliasStore) -> None:
        self._store = store
        self._cache: dict[str, str] = {}

    def load(self) -> None:
        """Populate the in-memory cache from the SQLite store."""
        self._cache = self._store.load_all()

    def resolve(self, name: str) -> str:
        """Normalize name and resolve through alias cache.

        Returns the canonical name if an alias exists, otherwise returns
        the normalized form of the input.
        """
        normalized = normalize_entity_name(name)
        return self._cache.get(normalized, normalized)

    def register(
        self,
        alias: str,
        canonical: str,
        *,
        confidence: float = 0.8,
        source: str = "graph",
    ) -> None:
        """Register an alias in both the store and the cache."""
        normalized_alias = normalize_entity_name(alias)
        normalized_canonical = normalize_entity_name(canonical)
        if not normalized_alias or not normalized_canonical:
            return
        # Only update if new confidence is higher than cached
        existing = self._cache.get(normalized_alias)
        if existing == normalized_canonical:
            return  # already mapped correctly
        self._store.register(
            normalized_alias, normalized_canonical, confidence=confidence, source=source
        )
        # Refresh the specific entry from store (respects confidence logic)
        stored = self._store.get_canonical(normalized_alias)
        if stored:
            self._cache[normalized_alias] = stored
