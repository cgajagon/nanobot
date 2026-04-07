# Memory Performance Optimizations Round 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce redundant DB queries in graph augmentation (15-25 → 1), cache event reads, and harden SQLite reliability with busy_timeout.

**Architecture:** Three independent optimizations, each touching 1-3 files within the `memory/` package. No cross-component changes, no import boundary violations. TDD for new methods; existing tests cover behavioral equivalence for internal refactors.

**Tech Stack:** Python 3.12, asyncio, SQLite, pytest-asyncio

**Spec:** `docs/superpowers/specs/2026-04-07-memory-perf-round2-design.md`

---

## File Map

| File | Action | Optimization |
|------|--------|-------------|
| `nanobot/memory/read/graph_augmentation.py` | Modify: add `_events_cache`, `_cached_events()`, update callers | Opt 1 + Opt 2 |
| `nanobot/memory/read/retriever.py` | Modify: line 205 uses cached events path | Opt 1 |
| `nanobot/memory/db/graph_store.py` | Modify: add `get_entities_batch()` | Opt 2 |
| `nanobot/memory/graph/graph.py` | Modify: add `get_entities_batch()` wrapper | Opt 2 |
| `nanobot/memory/db/connection.py` | Modify: add PRAGMA busy_timeout, update comment | Opt 3 |
| `tests/test_retriever.py` | Modify: add events cache test | Opt 1 |
| `tests/test_graph_store_batch.py` | Create: batch entity lookup tests | Opt 2 |
| `tests/contract/test_memory_wiring.py` | Modify: add busy_timeout check | Opt 3 |

---

## Task 1: SQLite busy_timeout (Opt 3)

**Files:**
- Modify: `nanobot/memory/db/connection.py:61-65`
- Modify: `tests/contract/test_memory_wiring.py`

This is the simplest change — start here to build momentum.

### Step 1.1: Write the failing test

- [ ] Add this test to the end of `tests/contract/test_memory_wiring.py`:

```python
class TestBusyTimeout:
    """SQLite connection has busy_timeout set for concurrent access safety."""

    def test_busy_timeout_set(self, tmp_path: Path) -> None:
        """Connection should have busy_timeout > 0 for WAL concurrent writes."""
        db = MemoryDatabase(tmp_path / "test.db", dims=384)
        result = db.connection.execute("PRAGMA busy_timeout").fetchone()
        assert result is not None
        timeout = result[0] if isinstance(result, tuple) else result["busy_timeout"]
        assert timeout >= 5000, f"Expected busy_timeout >= 5000, got {timeout}"
```

Note: check the existing imports at the top of `test_memory_wiring.py` — `MemoryDatabase` and `Path` should already be imported. If not, add:
```python
from pathlib import Path
from nanobot.memory.db import MemoryDatabase
```

### Step 1.2: Run test to verify it fails

- [ ] Run:

```bash
python -m pytest tests/contract/test_memory_wiring.py::TestBusyTimeout -v
```

Expected: FAIL — busy_timeout is currently 0.

### Step 1.3: Implement busy_timeout

- [ ] In `nanobot/memory/db/connection.py`, replace lines 61-65:

**Replace this:**

```python
        # check_same_thread=False: safe because WAL mode allows concurrent
        # readers, and asyncio.to_thread() only dispatches read-only methods.
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
```

**With this:**

```python
        # check_same_thread=False: safe because WAL mode allows concurrent
        # readers, and busy_timeout handles the rare case where asyncio.to_thread
        # dispatches overlapping writes (micro-extraction + consolidation).
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout = 5000")
```

### Step 1.4: Run tests to verify they pass

- [ ] Run:

```bash
python -m pytest tests/contract/test_memory_wiring.py -v
```

Expected: ALL tests PASS including the new one.

### Step 1.5: Lint + typecheck

- [ ] Run:

```bash
make lint && make typecheck
```

### Step 1.6: Commit

- [ ] Run:

```bash
git add nanobot/memory/db/connection.py tests/contract/test_memory_wiring.py
git commit -m "fix(memory): set SQLite busy_timeout for concurrent access safety

Add PRAGMA busy_timeout=5000 after WAL mode setup. Prevents
intermittent SQLITE_BUSY errors when micro-extraction writes
overlap with retrieval reads or consolidation writes.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Batch Graph Entity Row Lookups (Opt 2)

**Files:**
- Modify: `nanobot/memory/db/graph_store.py`
- Modify: `nanobot/memory/graph/graph.py`
- Modify: `nanobot/memory/read/graph_augmentation.py:88-91`
- Create: `tests/test_graph_store_batch.py`

### Step 2.1: Write failing tests for the batch method

- [ ] Create `tests/test_graph_store_batch.py`:

```python
"""Tests for batch entity lookup in GraphStore and KnowledgeGraph."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from nanobot.memory.db import MemoryDatabase
from nanobot.memory.db.graph_store import GraphStore


class TestGraphStoreBatch:
    """GraphStore.get_entities_batch returns correct results."""

    def test_returns_existing_entities(self, tmp_path: Path) -> None:
        db = MemoryDatabase(tmp_path / "test.db", dims=384)
        gs = db.graph_store
        # Insert test entities
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
        """Names are normalized before DB lookup, results mapped back to originals."""
        from nanobot.memory.db.alias_store import AliasRegistry
        from nanobot.memory.graph.graph import KnowledgeGraph

        db = MemoryDatabase(tmp_path / "test.db", dims=384)
        gs = db.graph_store
        # Store with canonical name (normalize_entity_name lowercases + strips)
        gs.upsert_entity("alice", type="PERSON", first_seen="2025-01-01", last_seen="2025-06-01")

        alias_registry = AliasRegistry(db.connection)
        kg = KnowledgeGraph(db=gs, alias_registry=alias_registry, enabled=True)

        # Query with un-normalized name — should still find "alice"
        result = kg.get_entities_batch({"Alice", "ALICE"})
        # Both map to canonical "alice", so at least one should be in result
        found_names = set(result.keys())
        assert found_names & {"Alice", "ALICE"}, f"Expected at least one match, got {found_names}"

    def test_disabled_graph_returns_empty(self) -> None:
        from nanobot.memory.graph.graph import KnowledgeGraph

        kg = KnowledgeGraph(db=None, alias_registry=None, enabled=False)
        result = kg.get_entities_batch({"alice"})
        assert result == {}


class TestGraphAugmenterUsesBatch:
    """GraphAugmenter uses batch lookup instead of per-entity loop."""

    def test_collect_uses_batch(self) -> None:
        """After batch optimization, get_entity_row should NOT be called per entity."""
        from unittest.mock import patch

        from nanobot.memory.read.graph_augmentation import GraphAugmenter, extract_entities

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

        # get_entities_batch should be called instead of per-entity get_entity_row
        graph.get_entities_batch.assert_called_once()
        graph.get_entity_row.assert_not_called()
        assert "alice" in result
        assert "bob" in result
```

### Step 2.2: Run tests to verify they fail

- [ ] Run:

```bash
python -m pytest tests/test_graph_store_batch.py -v
```

Expected: `TestGraphStoreBatch` tests fail (method doesn't exist), `TestKnowledgeGraphBatch` tests fail, `TestGraphAugmenterUsesBatch` fails.

### Step 2.3: Implement GraphStore.get_entities_batch

- [ ] In `nanobot/memory/db/graph_store.py`, add this method after `get_entity()` (after line 57):

```python
    def get_entities_batch(self, names: set[str]) -> dict[str, dict[str, Any]]:
        """Fetch multiple entities by name in one query.

        Returns a dict mapping entity name to row dict. Missing entities
        are omitted from the result (callers should handle absent keys).
        """
        if not names:
            return {}
        name_list = list(names)
        placeholders = ",".join("?" for _ in name_list)
        rows = self._conn.execute(
            f"SELECT * FROM entities WHERE name IN ({placeholders})",  # noqa: S608
            name_list,
        ).fetchall()
        return {str(row["name"]): dict(row) for row in rows}
```

Note: `# noqa: S608` suppresses the ruff "possible SQL injection" warning — the placeholders are parameterized, the f-string only builds the placeholder count.

### Step 2.4: Implement KnowledgeGraph.get_entities_batch

- [ ] In `nanobot/memory/graph/graph.py`, add this method after `get_entity_row()` (after line 412):

```python
    def get_entities_batch(self, names: set[str]) -> dict[str, dict[str, Any]]:
        """Batch entity lookup with normalization.

        Normalizes each name via ``_norm()`` before the DB query, then maps
        results back to the original (un-normalized) names provided by the caller.
        """
        if not self.enabled or not names or self._db is None:
            return {}
        norm_to_orig: dict[str, str] = {}
        for name in names:
            normalized = _norm(name)
            if normalized:
                norm_to_orig[normalized] = name
        if not norm_to_orig:
            return {}
        raw = self._db.get_entities_batch(set(norm_to_orig.keys()))
        result: dict[str, dict[str, Any]] = {}
        for norm_name, orig_name in norm_to_orig.items():
            if norm_name in raw:
                result[orig_name] = raw[norm_name]
        return result
```

### Step 2.5: Update GraphAugmenter to use batch lookup

- [ ] In `nanobot/memory/read/graph_augmentation.py`, replace lines 87-91:

**Replace this:**

```python
        # Look up last_seen for each entity from the graph
        result: dict[str, str] = {}
        for name in all_names:
            row = self._graph.get_entity_row(name)
            result[name] = str(row.get("last_seen", "")) if row else ""
```

**With this:**

```python
        # Look up last_seen for all entities in one batch query
        result: dict[str, str] = {}
        entity_rows = self._graph.get_entities_batch(all_names)
        for name in all_names:
            row = entity_rows.get(name)
            result[name] = str(row.get("last_seen", "")) if row else ""
```

### Step 2.6: Run all tests

- [ ] Run:

```bash
python -m pytest tests/test_graph_store_batch.py tests/test_retriever.py -v
```

Expected: ALL tests PASS.

### Step 2.7: Lint + typecheck

- [ ] Run:

```bash
make lint && make typecheck
```

### Step 2.8: Commit

- [ ] Run:

```bash
git add nanobot/memory/db/graph_store.py nanobot/memory/graph/graph.py nanobot/memory/read/graph_augmentation.py tests/test_graph_store_batch.py
git commit -m "perf(memory): batch graph entity lookups with WHERE IN query

Replace per-entity get_entity_row() loop (15-25 SQL queries) with
single get_entities_batch() call using WHERE name IN (...). Names
are normalized via _norm() before lookup and mapped back to originals.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Cache Event Reads in Graph Augmentation (Opt 1)

**Files:**
- Modify: `nanobot/memory/read/graph_augmentation.py:24-42, 150`
- Modify: `nanobot/memory/read/retriever.py:204-205`
- Modify: `tests/test_retriever.py`

### Step 3.1: Write the failing test

- [ ] Add this test class to `tests/test_retriever.py`, after the existing `TestGraphEntityCache` class:

```python
class TestEventsCache:
    """Events are read once and cached across collect + build calls."""

    def test_read_events_called_once_across_methods(self) -> None:
        """_read_events_fn should be called once, not twice."""
        call_count = 0
        events_data = [
            {
                "entities": ["Alice"],
                "triples": [
                    {"subject": "Alice", "predicate": "knows", "object": "Bob"},
                ],
            }
        ]

        def counting_read(**kwargs: Any) -> list[dict[str, Any]]:
            nonlocal call_count
            call_count += 1
            return events_data

        from nanobot.memory.read.graph_augmentation import GraphAugmenter

        graph = MagicMock()
        graph.enabled = True
        graph.get_related_entity_names_sync = MagicMock(return_value=set())
        graph.get_entities_batch = MagicMock(return_value={})
        graph.get_triples_for_entities_sync = MagicMock(return_value=[])
        graph.get_entity_row = MagicMock(return_value={"last_seen": ""})

        extractor = MagicMock()
        extractor._extract_entities = MagicMock(return_value=[])

        graph_aug = GraphAugmenter(
            graph=graph,
            extractor=extractor,
            read_events_fn=counting_read,
        )

        with patch(
            "nanobot.memory.read.graph_augmentation.extract_entities",
            return_value=["Alice"],
        ):
            # First call — populates cache
            graph_aug.collect_graph_entity_names("Alice query", graph_aug.read_events())
            # Second call — should reuse cache
            graph_aug.build_graph_context_lines("Alice query", [], max_tokens=100)

        assert call_count == 1, f"Expected 1 call to read_events, got {call_count}"

    def test_cache_cleared_on_reset(self) -> None:
        """After reset_cache(), events are re-fetched."""
        call_count = 0

        def counting_read(**kwargs: Any) -> list[dict[str, Any]]:
            nonlocal call_count
            call_count += 1
            return []

        from nanobot.memory.read.graph_augmentation import GraphAugmenter

        graph = MagicMock()
        graph.enabled = False

        graph_aug = GraphAugmenter(
            graph=graph,
            extractor=MagicMock(),
            read_events_fn=counting_read,
        )

        graph_aug.read_events()  # call 1
        graph_aug.reset_cache()
        graph_aug.read_events()  # call 2 (cache was cleared)
        assert call_count == 2
```

Make sure these imports are at the top of the file (they should already be there):
```python
from unittest.mock import MagicMock, patch
from typing import Any
```

### Step 3.2: Run test to verify it fails

- [ ] Run:

```bash
python -m pytest tests/test_retriever.py::TestEventsCache -v
```

Expected: `test_read_events_called_once_across_methods` FAILS (currently reads twice).

### Step 3.3: Implement the events cache

- [ ] In `nanobot/memory/read/graph_augmentation.py`, modify `__init__` to add cache. Replace lines 31-42:

**Replace this:**

```python
        self._graph = graph
        self._extractor = extractor
        self._read_events_fn = read_events_fn
        self._graph_cache: dict[frozenset[str], dict[str, str]] = {}

    def read_events(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Public accessor for the read-events callable."""
        return self._read_events_fn(**kwargs)

    def reset_cache(self) -> None:
        """Clear the per-request graph entity cache."""
        self._graph_cache = {}
```

**With this:**

```python
        self._graph = graph
        self._extractor = extractor
        self._read_events_fn = read_events_fn
        self._graph_cache: dict[frozenset[str], dict[str, str]] = {}
        self._events_cache: list[dict[str, Any]] | None = None

    def read_events(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Public accessor — returns cached events (limit=200 superset)."""
        if self._events_cache is None:
            self._events_cache = self._read_events_fn(limit=200)
        return self._events_cache

    def reset_cache(self) -> None:
        """Clear all per-request caches (graph entities + events)."""
        self._graph_cache = {}
        self._events_cache = None
```

### Step 3.4: Update build_graph_context_lines to use cached events

- [ ] In `nanobot/memory/read/graph_augmentation.py`, find `build_graph_context_lines()` around line 150 where it calls `self._read_events_fn(limit=200)`. Replace that line:

**Replace this:**

```python
        events = self._read_events_fn(limit=200)
```

**With this:**

```python
        events = self.read_events()
```

### Step 3.5: Update retriever to use cached events path

- [ ] In `nanobot/memory/read/retriever.py`, find lines 204-206:

**Replace this:**

```python
        graph_entities = self._graph_aug.collect_graph_entity_names(
            query, self._graph_aug.read_events()
        )
```

**With this:**

```python
        graph_entities = self._graph_aug.collect_graph_entity_names(
            query, self._graph_aug.read_events()
        )
```

Actually, this line already calls `self._graph_aug.read_events()` which now uses the cache. **No change needed in retriever.py** — the cache is transparent.

### Step 3.6: Run all tests

- [ ] Run:

```bash
python -m pytest tests/test_retriever.py -v
```

Expected: ALL tests PASS including the new ones and all existing graph cache tests.

### Step 3.7: Lint + typecheck

- [ ] Run:

```bash
make lint && make typecheck
```

### Step 3.8: Commit

- [ ] Run:

```bash
git add nanobot/memory/read/graph_augmentation.py tests/test_retriever.py
git commit -m "perf(memory): cache event reads in graph augmentation

Add _events_cache to GraphAugmenter that stores events (limit=200)
on first read_events() call. Both collect_graph_entity_names() and
build_graph_context_lines() reuse the cached result, eliminating
one full events table scan per request.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Update Documentation + Final Validation

**Files:**
- Modify: `.claude/rules/memory-architecture.md`

### Step 4.1: Update memory-architecture.md

- [ ] In `.claude/rules/memory-architecture.md`, find the Knowledge Graph section (Section 6) and add `get_entities_batch` to the method table. Find the table that lists graph methods like `get_entity_row`, and add a row:

```
| `get_entities_batch` | 413+ | Batch entity lookup with IN clause | 1 SQL query for N entities |
```

- [ ] In the Read Pipeline section (Section 5), find the "Stage 6: SCORE" or graph-related description and add a note about the events cache:

After the existing graph augmentation mention, add:
```
GraphAugmenter caches events on first ``read_events()`` call per request
(cleared in ``reset_cache()``).  Both ``collect_graph_entity_names`` and
``build_graph_context_lines`` reuse the cached superset (limit=200).
```

### Step 4.2: Run make check

- [ ] Run:

```bash
make check
```

Expected: ALL checks pass.

### Step 4.3: Run targeted tests

- [ ] Run:

```bash
python -m pytest tests/test_retriever.py tests/test_graph_store_batch.py tests/contract/test_memory_wiring.py tests/contract/test_memory_contracts.py -v
```

Expected: ALL tests pass.

### Step 4.4: Commit docs

- [ ] Run:

```bash
git add .claude/rules/memory-architecture.md
git commit -m "docs(memory): document batch entity lookups and events cache

Update memory-architecture.md with get_entities_batch method in
Knowledge Graph section and events cache in Read Pipeline section.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

### Step 4.5: Run make pre-push

- [ ] Run:

```bash
make pre-push
```

Expected: Full CI suite passes.
