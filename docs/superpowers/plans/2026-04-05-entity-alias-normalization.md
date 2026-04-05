# Entity Alias Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify entity alias handling across the memory subsystem with enhanced string normalization (Phase 1) and a unified alias registry backed by SQLite (Phase 2).

**Architecture:** Phase 1 adds `normalize_entity_name()` to `_text.py` and wires it into all entity-handling code paths (graph, dedup, entity linker). Phase 2 adds `AliasStore` + `AliasRegistry` in `memory/db/alias_store.py`, a new `alias_registry` SQLite table, and replaces the `user_aliases` frozenset in `EventDeduplicator` with the registry. Both phases are additive — no schema migrations, no boundary crossings.

**Tech Stack:** Python 3.10+, SQLite, pytest, ruff, mypy

**Spec:** `docs/superpowers/specs/2026-04-05-entity-alias-normalization-design.md`

**Branch:** `feat/entity-alias-normalization`

**Important conventions to follow:**
- Every module starts with `from __future__ import annotations`
- Type hints on all function signatures
- `__all__` in every `__init__.py`
- `make lint && make typecheck` after every edit
- `make check` before every commit
- Code review before every commit (dispatch code-reviewer subagent)

---

## File Map

### New files
| File | Responsibility |
|------|---------------|
| `nanobot/memory/db/alias_store.py` | `AliasStore` (SQLite CRUD) + `AliasRegistry` (in-memory cache) |
| `tests/test_normalize_entity_name.py` | Unit tests for `normalize_entity_name()` |
| `tests/test_alias_registry.py` | Unit tests for `AliasStore` + `AliasRegistry` |
| `tests/contract/test_alias_contracts.py` | Contract tests for alias wiring |

### Modified files
| File | Change |
|------|--------|
| `nanobot/memory/_text.py` | Add `normalize_entity_name()` |
| `nanobot/memory/constants.py` | Add `ALIAS_REGISTRY_DDL` constant |
| `nanobot/memory/db/connection.py` | Add alias table DDL + `alias_store` lazy property |
| `nanobot/memory/db/__init__.py` | Export `AliasStore` |
| `nanobot/memory/graph/__init__.py` | `_norm()` delegates to `normalize_entity_name()` |
| `nanobot/memory/graph/ontology_types.py` | `Entity.canonical_name` uses `normalize_entity_name()` |
| `nanobot/memory/graph/entity_linker.py` | `resolve_alias()` accepts optional registry |
| `nanobot/memory/graph/graph.py` | `upsert_entity()` registers aliases in registry |
| `nanobot/memory/write/dedup.py` | Replace `user_aliases` with `AliasRegistry` |
| `nanobot/memory/store.py` | Create and seed `AliasRegistry`, pass to consumers |
| `tests/test_event_deduplicator.py` | Update `_make_dedup` for new param |
| `tests/test_ingester.py` | Update `EventDeduplicator()` calls |
| `tests/contract/test_memory_wiring.py` | Add alias_store assertion |
| `tests/contract/test_memory_constants.py` | Add `ALIAS_REGISTRY_DDL` assertion |

---

## Task 1: Add `normalize_entity_name()` to `_text.py`

**Files:**
- Modify: `nanobot/memory/_text.py`
- Create: `tests/test_normalize_entity_name.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_normalize_entity_name.py`:

```python
"""Tests for normalize_entity_name in memory._text."""

from __future__ import annotations

import pytest

from nanobot.memory._text import normalize_entity_name


class TestNormalizeEntityName:
    """normalize_entity_name strips possessives, titles, punctuation, normalizes."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            # Basic lowercasing and whitespace
            ("Carlos", "carlos"),
            ("  Carlos  ", "carlos"),
            ("Carlos Gajardo", "carlos_gajardo"),
            # Possessives (straight and smart quotes)
            ("User's", "user"),
            ("Carlos's", "carlos"),
            ("User\u2019s", "user"),  # smart quote '
            ("User\u2018s", "user"),  # smart quote '
            # Titles stripped at start
            ("Dr. Smith", "smith"),
            ("Mr. Jones", "jones"),
            ("Mrs. Williams", "williams"),
            ("Ms. Davis", "davis"),
            ("Prof. Lee", "lee"),
            # Titles NOT stripped mid-string
            ("Visit Dr. Smith", "visit_dr_smith"),
            # Punctuation stripped (except hyphens/underscores)
            ("O'Brien", "obrien"),
            ("vue-router", "vue-router"),
            ("my_project", "my_project"),
            ("hello.world", "helloworld"),
            # Unicode NFKC
            ("\ufb01nance", "finance"),  # fi ligature
            ("caf\u00e9", "caf\u00e9"),  # accented char preserved after NFKC
            # Empty and whitespace
            ("", ""),
            ("   ", ""),
            # Multiple spaces become single underscore
            ("New   York   City", "new_york_city"),
        ],
    )
    def test_normalization(self, raw: str, expected: str) -> None:
        assert normalize_entity_name(raw) == expected
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_normalize_entity_name.py -v`
Expected: FAIL with `ImportError: cannot import name 'normalize_entity_name'`

- [ ] **Step 3: Implement `normalize_entity_name()`**

Add to `nanobot/memory/_text.py` after the `_norm_text` function:

```python
# Regex for possessive suffixes (straight and smart quotes)
_POSSESSIVE_RE = re.compile(r"['\u2018\u2019]s$", re.IGNORECASE)

# Titles stripped only at start of name
_TITLE_RE = re.compile(r"^(?:dr|mr|mrs|ms|prof)\.?\s+", re.IGNORECASE)

# Punctuation to strip (keep hyphens, underscores, alphanumeric, spaces)
_ENTITY_PUNCT_RE = re.compile(r"[^\w\s-]", re.UNICODE)


def normalize_entity_name(name: str) -> str:
    """Normalize an entity name to its canonical form.

    Pipeline: NFKC -> strip -> strip possessives -> strip titles ->
    strip punctuation (preserve hyphens/underscores) -> lowercase ->
    collapse whitespace -> spaces to underscores.
    """
    import unicodedata

    if not name or not name.strip():
        return ""
    text = unicodedata.normalize("NFKC", name)
    text = text.strip()
    text = _POSSESSIVE_RE.sub("", text)
    text = _TITLE_RE.sub("", text)
    text = _ENTITY_PUNCT_RE.sub("", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace(" ", "_")
    return text
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_normalize_entity_name.py -v`
Expected: all PASS

- [ ] **Step 5: Run lint and typecheck**

Run: `make lint && make typecheck`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add nanobot/memory/_text.py tests/test_normalize_entity_name.py
git commit -m "feat(memory): add normalize_entity_name for entity string normalization"
```

---

## Task 2: Wire `normalize_entity_name` into graph and entity linker

**Files:**
- Modify: `nanobot/memory/graph/__init__.py`
- Modify: `nanobot/memory/graph/ontology_types.py`
- Modify: `nanobot/memory/graph/entity_linker.py`

- [ ] **Step 1: Write a test verifying Entity.canonical_name uses new normalization**

Add to `tests/test_normalize_entity_name.py`:

```python
from nanobot.memory.graph.ontology_types import Entity


class TestEntityCanonicalName:
    """Entity.canonical_name uses normalize_entity_name."""

    def test_possessive_stripped(self) -> None:
        e = Entity(name="User's")
        assert e.canonical_name == "user"

    def test_title_stripped(self) -> None:
        e = Entity(name="Dr. Smith")
        assert e.canonical_name == "smith"

    def test_basic(self) -> None:
        e = Entity(name="Carlos Gajardo")
        assert e.canonical_name == "carlos_gajardo"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_normalize_entity_name.py::TestEntityCanonicalName -v`
Expected: FAIL — `Entity(name="User's").canonical_name` returns `"user's"` not `"user"`

- [ ] **Step 3: Update `Entity.canonical_name`**

In `nanobot/memory/graph/ontology_types.py`, add import and change the property:

Add at top (after other imports):
```python
from nanobot.memory._text import normalize_entity_name
```

Replace the `canonical_name` property:
```python
    @property
    def canonical_name(self) -> str:
        """Normalised lowercase name used as the graph key."""
        return normalize_entity_name(self.name)
```

- [ ] **Step 4: Update `_norm()` in `graph/__init__.py`**

Replace the existing `_norm` function:

```python
from nanobot.memory._text import normalize_entity_name


def _norm(name: str) -> str:
    """Canonical name normalisation: delegates to normalize_entity_name.

    Package-private helper used by graph.py and graph_traversal.py.
    """
    return normalize_entity_name(name)
```

- [ ] **Step 5: Update `entity_linker.py` to use normalized lookup key**

In `nanobot/memory/graph/entity_linker.py`, update the `resolve_alias` function:

```python
from nanobot.memory._text import normalize_entity_name

_ALIAS_MAP: dict[str, str] = {
    # Databases
    "pg": "postgresql",
    "postgres": "postgresql",
    "mongo": "mongodb",
    "es": "elasticsearch",
    # Technologies
    "k8s": "kubernetes",
    "kube": "kubernetes",
    "tf": "terraform",
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "python3": "python",
    "gh": "github",
    "gh actions": "github actions",
    # Environments
    "prod": "production",
    "dev": "development",
    "preprod": "pre-production",
    "pre-prod": "pre-production",
}


def resolve_alias(name: str) -> str:
    """Map known shorthand/alias to its canonical entity name.

    Returns the original name (stripped) if no alias is registered.
    """
    key = normalize_entity_name(name)
    return _ALIAS_MAP.get(key, name.strip())
```

- [ ] **Step 6: Run all tests**

Run: `python -m pytest tests/test_normalize_entity_name.py -v && make lint && make typecheck`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add nanobot/memory/graph/__init__.py nanobot/memory/graph/ontology_types.py nanobot/memory/graph/entity_linker.py tests/test_normalize_entity_name.py
git commit -m "feat(memory): wire normalize_entity_name into graph and entity linker"
```

---

## Task 3: Wire `normalize_entity_name` into dedup entity tokens

**Files:**
- Modify: `nanobot/memory/write/dedup.py`
- Modify: `tests/test_event_deduplicator.py`

- [ ] **Step 1: Write a test verifying possessive normalization in dedup**

Add to `tests/test_event_deduplicator.py`:

```python
class TestEntityNormalizationInSimilarity:
    def test_possessive_entities_match(self) -> None:
        """'User's project' and 'User project' should have high entity overlap."""
        d = _make_dedup()
        a = {"type": "fact", "summary": "Working on the project", "entities": ["User's"]}
        b = {"type": "fact", "summary": "Working on the project", "entities": ["User"]}
        # After normalization, both entity tokens become "user"
        lexical, _ = d.event_similarity(a, b)
        assert lexical >= 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_event_deduplicator.py::TestEntityNormalizationInSimilarity -v`
Expected: FAIL — entity tokens "user's" and "user" don't match without normalization

- [ ] **Step 3: Update `event_similarity()` to normalize entity tokens**

In `nanobot/memory/write/dedup.py`, add import:

```python
from .._text import normalize_entity_name
```

(Add `normalize_entity_name` to the existing import line from `._text`.)

Then in `event_similarity()`, update the `_event_text` inner function:

```python
        def _event_text(event: dict[str, Any]) -> str:
            summary = str(event.get("summary", ""))
            raw_entities = _to_str_list(event.get("entities"))
            entities = " ".join(normalize_entity_name(e) for e in raw_entities)
            event_type = str(event.get("type", "fact"))
            return f"{event_type}. {summary}. {entities}".strip()
```

Also update `find_semantic_duplicate()` entity overlap computation to normalize:

```python
            candidate_entities = {normalize_entity_name(x) for x in _to_str_list(candidate.get("entities"))}
            existing_entities = {normalize_entity_name(x) for x in _to_str_list(existing.get("entities"))}
```

(Replace `_norm_text(x)` with `normalize_entity_name(x)` in both lines.)

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_event_deduplicator.py -v && make lint && make typecheck`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add nanobot/memory/write/dedup.py tests/test_event_deduplicator.py
git commit -m "feat(memory): normalize entity tokens in dedup similarity computation"
```

---

## Task 4: Add `ALIAS_REGISTRY_DDL` constant and schema

**Files:**
- Modify: `nanobot/memory/constants.py`
- Modify: `nanobot/memory/db/connection.py`
- Modify: `tests/contract/test_memory_constants.py`
- Modify: `tests/contract/test_memory_wiring.py`

- [ ] **Step 1: Write the failing constant test**

Add to `tests/contract/test_memory_constants.py`:

```python
class TestAliasRegistryDDL:
    def test_alias_registry_ddl_importable(self) -> None:
        from nanobot.memory.constants import ALIAS_REGISTRY_DDL

        assert "CREATE TABLE" in ALIAS_REGISTRY_DDL
        assert "alias_registry" in ALIAS_REGISTRY_DDL
```

- [ ] **Step 2: Write the failing wiring test**

Add to `tests/contract/test_memory_wiring.py`:

```python
def test_alias_registry_table_exists(tmp_path):
    """alias_registry table is created by MemoryDatabase schema init."""
    store = _make_store(tmp_path)
    cursor = store.db.connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='alias_registry'"
    )
    assert cursor.fetchone() is not None
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/contract/test_memory_constants.py::TestAliasRegistryDDL tests/contract/test_memory_wiring.py::test_alias_registry_table_exists -v`
Expected: FAIL

- [ ] **Step 4: Add `ALIAS_REGISTRY_DDL` to constants.py**

Add after the `STRATEGIES_DDL` block in `nanobot/memory/constants.py`:

```python
# Schema DDL for the alias registry table — single source of truth.
# Used by MemoryDatabase._init_schema() and imported by test fixtures.
ALIAS_REGISTRY_DDL = """
CREATE TABLE IF NOT EXISTS alias_registry (
    alias      TEXT PRIMARY KEY,
    canonical  TEXT NOT NULL,
    confidence REAL DEFAULT 0.8,
    source     TEXT DEFAULT 'config'
);
CREATE INDEX IF NOT EXISTS idx_alias_canonical ON alias_registry(canonical);
"""
```

- [ ] **Step 5: Add DDL to `connection.py` schema init**

In `nanobot/memory/db/connection.py`, import the DDL:

Add `ALIAS_REGISTRY_DDL` to the import from `..constants` (alongside `STRATEGIES_DDL`).

Then in `_init_schema()`, add `{ALIAS_REGISTRY_DDL}` inside the `executescript` call, after `{STRATEGIES_DDL}`.

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/contract/test_memory_constants.py::TestAliasRegistryDDL tests/contract/test_memory_wiring.py::test_alias_registry_table_exists -v && make lint && make typecheck`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add nanobot/memory/constants.py nanobot/memory/db/connection.py tests/contract/test_memory_constants.py tests/contract/test_memory_wiring.py
git commit -m "feat(memory): add alias_registry DDL constant and schema"
```

---

## Task 5: Create `AliasStore` and `AliasRegistry`

**Files:**
- Create: `nanobot/memory/db/alias_store.py`
- Modify: `nanobot/memory/db/__init__.py`
- Modify: `nanobot/memory/db/connection.py`
- Create: `tests/test_alias_registry.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_alias_registry.py`:

```python
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

    def test_register_batch(self) -> None:
        store = AliasStore(_make_conn())
        entries = [
            ("carlos", "user", 1.0, "config"),
            ("the_user", "user", 0.9, "config"),
        ]
        store.register_batch(entries)
        assert store.load_all() == {"carlos": "user", "the_user": "user"}

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_alias_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nanobot.memory.db.alias_store'`

- [ ] **Step 3: Implement `AliasStore` and `AliasRegistry`**

Create `nanobot/memory/db/alias_store.py`:

```python
"""Alias registry — SQLite storage and in-memory cache for entity aliases.

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

    def register_batch(self, entries: list[tuple[str, str, float, str]]) -> None:
        """Upsert multiple aliases. Each entry: (alias, canonical, confidence, source)."""
        for alias, canonical, confidence, source in entries:
            self.register(alias, canonical, confidence=confidence, source=source)

    def get_canonical(self, alias: str) -> str | None:
        """Look up a single alias. Returns None if not found."""
        row = self._conn.execute(
            "SELECT canonical FROM alias_registry WHERE alias = ?", (alias,)
        ).fetchone()
        return row["canonical"] if row else None

    def remove_by_canonical(self, canonical: str) -> None:
        """Remove all aliases pointing to a canonical name."""
        with self._conn:
            self._conn.execute(
                "DELETE FROM alias_registry WHERE canonical = ?", (canonical,)
            )


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
```

- [ ] **Step 4: Update `memory/db/__init__.py`**

```python
"""Database layer -- focused repository classes sharing one SQLite connection."""

from __future__ import annotations

from .alias_store import AliasStore
from .connection import MemoryDatabase
from .event_store import EventStore
from .graph_store import GraphStore

__all__ = ["AliasStore", "EventStore", "GraphStore", "MemoryDatabase"]
```

- [ ] **Step 5: Add `alias_store` lazy property to `MemoryDatabase`**

In `nanobot/memory/db/connection.py`, add to `__init__` after the `_graph_store` line:

```python
        self._alias_store: AliasStore | None = None
```

Add the lazy property after the `graph_store` property:

```python
    @property
    def alias_store(self) -> AliasStore:
        """Focused repository for entity alias CRUD."""
        if self._alias_store is None:
            from .alias_store import AliasStore

            self._alias_store = AliasStore(self._conn)
        return self._alias_store
```

Also add to the TYPE_CHECKING block if there is one, or add the forward reference in the init annotation.

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_alias_registry.py -v && make lint && make typecheck`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add nanobot/memory/db/alias_store.py nanobot/memory/db/__init__.py nanobot/memory/db/connection.py tests/test_alias_registry.py
git commit -m "feat(memory): add AliasStore and AliasRegistry for unified alias management"
```

---

## Task 6: Replace `user_aliases` frozenset with `AliasRegistry` in dedup

**Files:**
- Modify: `nanobot/memory/write/dedup.py`
- Modify: `tests/test_event_deduplicator.py`
- Modify: `tests/test_ingester.py`

- [ ] **Step 1: Write a test using `AliasRegistry` in dedup**

Add to `tests/test_event_deduplicator.py`:

```python
import sqlite3

from nanobot.memory.constants import ALIAS_REGISTRY_DDL
from nanobot.memory.db.alias_store import AliasRegistry, AliasStore


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


class TestAliasRegistryInSimilarity:
    def test_registry_resolves_aliases_in_similarity(self) -> None:
        registry = _make_registry({"carlos": "_user_", "user": "_user_"})
        d = _make_dedup(alias_registry=registry)
        a = {"type": "fact", "summary": "Carlos likes Python", "entities": ["Carlos"]}
        b = {"type": "fact", "summary": "User likes Python", "entities": ["User"]}
        lexical, _ = d.event_similarity(a, b)
        # With alias resolution, "carlos" and "user" both become "_user_"
        assert lexical >= 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_event_deduplicator.py::TestAliasRegistryInSimilarity -v`
Expected: FAIL — `_make_dedup` doesn't accept `alias_registry`

- [ ] **Step 3: Update `EventDeduplicator` to accept `AliasRegistry`**

In `nanobot/memory/write/dedup.py`, update the imports and `__init__`:

Add to TYPE_CHECKING block:
```python
    from ..db.alias_store import AliasRegistry
```

Update `__init__` signature:

```python
    def __init__(
        self,
        coercer: EventCoercer,
        conflict_pair_fn: Callable[[str, str], bool] | None = None,
        user_aliases: frozenset[str] | None = None,
        embedder: Embedder | None = None,
        alias_registry: AliasRegistry | None = None,
    ) -> None:
        self._coercer = coercer
        self._conflict_pair_fn = conflict_pair_fn
        self._user_aliases = user_aliases or frozenset()
        self._embedder = embedder
        self._alias_registry = alias_registry
```

Update `event_similarity()` — replace the alias normalization block:

```python
        # Normalize user aliases to canonical tokens
        if self._alias_registry:
            left_tokens = {self._alias_registry.resolve(t) for t in left_tokens}
            right_tokens = {self._alias_registry.resolve(t) for t in right_tokens}
        elif self._user_aliases:
            canonical = "_user_"
            left_tokens = {canonical if t in self._user_aliases else t for t in left_tokens}
            right_tokens = {canonical if t in self._user_aliases else t for t in right_tokens}
```

- [ ] **Step 4: Update `_make_dedup` helper in test file**

```python
def _make_dedup(
    *,
    conflict_pair_fn: object = None,
    user_aliases: frozenset[str] | None = None,
    embedder: object = None,
    alias_registry: object = None,
) -> EventDeduplicator:
    classifier = EventClassifier()
    coercer = EventCoercer(classifier)
    return EventDeduplicator(
        coercer=coercer,
        conflict_pair_fn=conflict_pair_fn,
        user_aliases=user_aliases,
        embedder=embedder,
        alias_registry=alias_registry,
    )
```

- [ ] **Step 5: Run all dedup tests**

Run: `python -m pytest tests/test_event_deduplicator.py -v && make lint && make typecheck`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add nanobot/memory/write/dedup.py tests/test_event_deduplicator.py
git commit -m "feat(memory): EventDeduplicator accepts AliasRegistry for alias resolution"
```

---

## Task 7: Seed `AliasRegistry` in `MemoryStore` and wire to consumers

**Files:**
- Modify: `nanobot/memory/store.py`
- Modify: `nanobot/memory/graph/graph.py`
- Modify: `nanobot/memory/graph/entity_linker.py`
- Create: `tests/contract/test_alias_contracts.py`

- [ ] **Step 1: Write contract tests**

Create `tests/contract/test_alias_contracts.py`:

```python
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
        # Both aliases should resolve to _user_ (the canonical token for user aliases)
        registry = store.alias_registry
        assert registry.resolve("carlos") == "_user_"
        assert registry.resolve("the user") == "_user_"

    def test_registry_seeded_from_linker(self, tmp_path: Path) -> None:
        """Static entity_linker aliases are seeded into the registry."""
        store = _make_store(tmp_path)
        registry = store.alias_registry
        # entity_linker maps "pg" -> "postgresql"
        assert registry.resolve("pg") == "postgresql"

    def test_dedup_uses_registry(self, tmp_path: Path) -> None:
        """EventDeduplicator receives the alias registry."""
        config = MemoryConfig(graph_enabled=False, user_aliases=["carlos"])
        store = MemoryStore(tmp_path, embedding_provider="hash", memory_config=config)
        assert store._dedup._alias_registry is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/contract/test_alias_contracts.py -v`
Expected: FAIL — `store.alias_registry` doesn't exist

- [ ] **Step 3: Wire `AliasRegistry` in `MemoryStore.__init__`**

In `nanobot/memory/store.py`, add imports:

```python
from .db.alias_store import AliasRegistry
from .graph.entity_linker import _ALIAS_MAP
```

In `__init__`, after the graph setup and before the `EventDeduplicator` construction (around line 169), replace the alias/dedup block:

```python
        # Unified alias registry — single source of truth for entity aliases.
        self.alias_registry = AliasRegistry(self.db.alias_store)
        self._seed_alias_registry()
        self.alias_registry.load()

        # EventDeduplicator + EventIngester: own the full event write path.
        self._dedup = EventDeduplicator(
            coercer=self._coercer,
            conflict_pair_fn=self.profile_mgr._conflict_pair,
            alias_registry=self.alias_registry,
            embedder=self._embedder,
        )
```

Add the seeding method:

```python
    def _seed_alias_registry(self) -> None:
        """Seed the alias registry from config, entity_linker, and graph aliases."""
        store = self.db.alias_store

        # Source 1: config user_aliases (highest confidence)
        for alias in self._memory_config.user_aliases:
            store.register(
                normalize_entity_name(alias), "_user_", confidence=1.0, source="config"
            )

        # Source 2: static entity_linker map
        for alias, canonical in _ALIAS_MAP.items():
            store.register(
                normalize_entity_name(alias),
                normalize_entity_name(canonical),
                confidence=0.9,
                source="linker",
            )

        # Source 3: existing graph entity aliases (if graph enabled)
        if self.graph.enabled:
            rows = self.db.graph_store.search_entities("", limit=1000)
            for row in rows:
                canonical = str(row.get("name", ""))
                aliases_text = str(row.get("aliases", ""))
                if aliases_text:
                    for alias in aliases_text.split(","):
                        alias = alias.strip()
                        if alias:
                            store.register(
                                normalize_entity_name(alias),
                                normalize_entity_name(canonical),
                                confidence=0.8,
                                source="graph",
                            )
```

Add the `normalize_entity_name` import:

```python
from ._text import _norm_text, _to_str_list, _utc_now_iso, normalize_entity_name
```

- [ ] **Step 4: Update `KnowledgeGraph.upsert_entity` to register aliases**

In `nanobot/memory/graph/graph.py`, add an `_alias_registry` parameter to `__init__`:

```python
    def __init__(
        self,
        db: GraphStore | None = None,
        alias_registry: AliasRegistry | None = None,
    ) -> None:
        self._db = db
        self._alias_registry = alias_registry
        self.enabled: bool = db is not None
        self.error: str | None = None
```

Add to the TYPE_CHECKING block:

```python
    from ..db.alias_store import AliasRegistry
```

At the end of `upsert_entity()` (after the `self._db.upsert_entity()` call), add:

```python
        # Register aliases in the unified registry
        if self._alias_registry:
            for alias in entity.aliases:
                if alias.strip():
                    self._alias_registry.register(
                        alias, entity.name, confidence=0.8, source="graph"
                    )
```

Then in `store.py`, update the graph construction to pass the registry:

```python
        if graph_enabled:
            self.graph = KnowledgeGraph(db=self.db.graph_store, alias_registry=self.alias_registry)
        else:
            self.graph = KnowledgeGraph()
```

Note: this requires moving the alias registry creation BEFORE the graph construction. Reorder the `__init__` so:
1. Embedder
2. MemoryDatabase
3. ProfileStore, classifiers, coercer, extractor
4. Reranker
5. **AliasRegistry** (new — moved before graph)
6. KnowledgeGraph (now receives alias_registry)
7. EventDeduplicator (receives alias_registry)
8. EventIngester
9. ...rest

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/contract/test_alias_contracts.py -v && make lint && make typecheck`
Expected: all PASS

- [ ] **Step 6: Run full test suite**

Run: `python -m pytest tests/ --ignore=tests/integration -x -v`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add nanobot/memory/store.py nanobot/memory/graph/graph.py tests/contract/test_alias_contracts.py
git commit -m "feat(memory): wire AliasRegistry into MemoryStore, seed from config/linker/graph"
```

---

## Task 8: Update existing tests and remove stale `user_aliases` param

**Files:**
- Modify: `tests/test_ingester.py`
- Modify: `tests/contract/test_memory_wiring.py`

- [ ] **Step 1: Grep for all `user_aliases` references in tests**

Run: `grep -rn "user_aliases" tests/ --include="*.py"`

Update each call site:
- `tests/test_ingester.py`: `EventDeduplicator(coercer=coercer)` calls don't pass `user_aliases`, so they should be fine. Verify.
- `tests/test_event_deduplicator.py`: `_make_dedup` already updated in Task 6.

- [ ] **Step 2: Add alias_store wiring test**

In `tests/contract/test_memory_wiring.py`, add:

```python
def test_alias_store_property(tmp_path):
    """MemoryDatabase exposes alias_store lazy property."""
    store = _make_store(tmp_path)
    alias_store = store.db.alias_store
    assert alias_store is not None
    # Should be the same instance on second access
    assert store.db.alias_store is alias_store
```

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/ --ignore=tests/integration -x && make check`
Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_ingester.py tests/contract/test_memory_wiring.py
git commit -m "test(memory): update tests for alias registry wiring"
```

---

## Task 9: Final validation

- [ ] **Step 1: Run `make check`**

Run: `make check`
Expected: all PASS (lint + typecheck + import-check + structure-check + prompt-check + phase-todo-check + doc-check)

- [ ] **Step 2: Run full unit tests with coverage**

Run: `make test-cov`
Expected: PASS with >= 85% coverage

- [ ] **Step 3: Update living docs if needed**

Check if `docs/memory-system-reference.md` or `.claude/rules/memory-architecture.md` reference the old `user_aliases` pattern or need to mention the alias registry. Update if so.

- [ ] **Step 4: Final commit (docs only, if needed)**

```bash
git add -A
git commit -m "docs: update memory architecture for unified alias registry"
```
