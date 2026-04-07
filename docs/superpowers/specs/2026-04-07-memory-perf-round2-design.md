# Memory Performance Optimizations Round 2 Design

> Date: 2026-04-07
> Status: Draft
> Scope: 3 independent optimizations to the memory subsystem (graph augmentation, entity lookups, SQLite reliability)

## Problem Statement

Profiling and deep code review of the memory subsystem identified additional
optimization opportunities after the first round (PR #162). This round focuses
on reducing redundant DB queries in graph augmentation, batching per-entity SQL
lookups, and hardening SQLite connection reliability.

## Items Dropped After Review

Two originally-proposed optimizations were rejected during pre-implementation
architecture and correctness review:

- **Numpy cosine similarity (dropped):** Dual-path code (`try: import numpy`
  with pure-Python fallback) is a prohibited pattern per `prohibited-patterns.md`.
  numpy is only a transitive dependency (via onnxruntime optional extra). Performance
  gain is negligible for single-pair dot products — the inner loop at `dedup.py:81-85`
  processes one vector pair per call, not batches.

- **Filter superseded events in FTS5 SQL (dropped):** The proposed filter targeted
  `search_fts_with_vectors()`, which is exclusively called by the write path
  (`ingester.py:219`). The dedup pipeline MUST see superseded events to: (1) detect
  duplicates of already-superseded events, and (2) maintain supersession chains via
  `supersedes_event_id`. Filtering them in SQL would cause duplicate re-insertion.
  The read path already handles superseded events correctly via a -0.2 score penalty
  in `scoring.py:337`.

- **LRU cache for query embeddings (dropped):** The first PR already parallelized
  embedding with FTS5, hiding the latency. Cost savings are negligible ($0.000002/call).

## Optimization 1: Cache Event Reads in Graph Augmentation

**Files:** `nanobot/memory/read/graph_augmentation.py`

### Current Problem

Two methods in `GraphAugmenter` independently call `self._read_events_fn()`:

- `collect_graph_entity_names()` — called from `retriever.py:204` during scoring
- `build_graph_context_lines()` at line 150 — called from `ContextAssembler.build()`

Each call reads up to 200 events from the database. Both happen within the same
request cycle (system prompt assembly), producing identical results. This wastes
one full events table scan (~5-15ms) per request.

### Design

Add an `_events_cache` to `GraphAugmenter` that stores the superset of events
(limit=200) on first access. Both callers share the cached result.

```python
# In __init__:
self._events_cache: list[dict[str, Any]] | None = None

# In reset_cache() (already clears _graph_cache):
self._events_cache = None

# Cached accessor used by both methods:
def _cached_events(self) -> list[dict[str, Any]]:
    if self._events_cache is None:
        self._events_cache = self._read_events_fn(limit=200)
    return self._events_cache
```

Both `collect_graph_entity_names()` and `build_graph_context_lines()` use
`self._cached_events()` instead of calling `self._read_events_fn()` directly.

The retriever call at `retriever.py:204` changes from
`self._graph_aug.read_events()` to reusing the same cached path — either by
calling `collect_graph_entity_names(query, self._graph_aug._cached_events())`
or by having `read_events()` delegate to `_cached_events()`.

### Key Design Decisions

- **Always fetch limit=200** — the maximum needed by any caller. Both callers
  operate on subsets of this superset. `build_graph_context_lines` needs 200,
  `collect_graph_entity_names` works with any subset.
- **Cache cleared in `reset_cache()`** — already called at the top of every
  `retrieve()` call (`retriever.py:101`). The `ContextAssembler.build()` path
  runs after `retrieve()` within the same request, so it sees the same data.
- **Staleness is not a concern** — events are only written by micro-extraction
  (async background task) and consolidation (periodic). Neither runs during
  the synchronous retrieval+assembly pipeline.

### Savings

Eliminates 1 full events table scan (~5-15ms) per request.

### Test Plan

- Existing `TestGraphEntityCache` tests pass unchanged (they already test
  cache reset behavior).
- New test: verify `_read_events_fn` is called once when both
  `collect_graph_entity_names()` and `build_graph_context_lines()` are called
  in sequence within the same request scope.

## Optimization 2: Batch Graph Entity Row Lookups

**Files:** `nanobot/memory/db/graph_store.py`, `nanobot/memory/graph/graph.py`,
`nanobot/memory/read/graph_augmentation.py`

### Current Problem

`graph_augmentation.py:89-91` loops through all entity names calling
`graph.get_entity_row(name)` individually. Each call executes a separate
`SELECT * FROM entities WHERE name = ?` query. For 15-25 related entities,
this is 15-25 separate SQL round-trips.

### Design

Add a batch method at each layer:

**GraphStore** (`graph_store.py`):
```python
def get_entities_batch(self, names: set[str]) -> dict[str, dict[str, Any]]:
    """Fetch multiple entities in one query."""
    if not names:
        return {}
    name_list = list(names)
    placeholders = ",".join("?" for _ in name_list)
    rows = self._conn.execute(
        f"SELECT * FROM entities WHERE name IN ({placeholders})",
        name_list,
    ).fetchall()
    return {str(row["name"]): dict(row) for row in rows}
```

**KnowledgeGraph** (`graph.py`):
```python
def get_entities_batch(self, names: set[str]) -> dict[str, dict[str, Any]]:
    """Batch entity lookup with normalization."""
    if not self.enabled or not names:
        return {}
    # Build normalized→original mapping
    norm_to_orig: dict[str, str] = {}
    for name in names:
        normalized = _norm(name)
        if normalized:
            norm_to_orig[normalized] = name
    raw = self._db.get_entities_batch(set(norm_to_orig.keys()))
    # Map results back to original (un-normalized) names
    result: dict[str, dict[str, Any]] = {}
    for norm_name, orig_name in norm_to_orig.items():
        if norm_name in raw:
            result[orig_name] = raw[norm_name]
    return result
```

**GraphAugmenter** (`graph_augmentation.py:88-91`):
```python
# Before (per-entity loop):
for name in all_names:
    row = self._graph.get_entity_row(name)
    result[name] = str(row.get("last_seen", "")) if row else ""

# After (single batch call):
entity_rows = self._graph.get_entities_batch(set(all_names))
for name in all_names:
    row = entity_rows.get(name)
    result[name] = str(row.get("last_seen", "")) if row else ""
```

### Key Design Decisions

- **`_norm()` applied in KnowledgeGraph layer** — matches how `get_entity_row()`
  works. Entity names in the DB are stored as canonical (normalized) names.
  The augmenter passes un-normalized names (lowercased from event triples),
  so normalization must happen before the SQL query.
- **Map back to original names** — the caller uses original names as dict keys.
  The `norm_to_orig` mapping ensures results are keyed by the name the caller
  provided, not the normalized form.
- **Missing entities return no entry** — the caller already checks `if row else ""`
  so entities not in the batch result simply get an empty string. No special
  handling needed.
- **Internal to memory package** — `GraphStore` and `KnowledgeGraph` are internal
  classes. Adding a method does not violate the facade pattern (Pattern 6).
  External code still only uses `MemoryStore`.

### Savings

15-25 SQL queries → 1 query per `collect_graph_entity_names()` call.

### Test Plan

- New test: verify `get_entities_batch()` returns correct results for a mix
  of existing and non-existing entities.
- New test: verify normalization is applied (e.g., "Alice Smith" maps to
  canonical "alice_smith" and returns the correct row).
- Existing graph augmentation tests pass unchanged.

## Optimization 3: SQLite busy_timeout

**File:** `nanobot/memory/db/connection.py`

### Current Problem

The SQLite connection is created with only `PRAGMA journal_mode=WAL`. No
`busy_timeout` is set, defaulting to 0ms — any lock contention immediately
raises `SQLITE_BUSY`. Under concurrent load (micro-extraction write overlapping
with retrieval reads or consolidation writes), this could silently degrade
retrieval quality when the `OperationalError` is caught by crash barriers.

### Design

Add one PRAGMA after WAL mode setup:

```python
# After line 65 in connection.py:
self._conn.execute("PRAGMA busy_timeout = 5000")
```

Also update the `check_same_thread` comment at line 63 to accurately reflect
that write operations also go through `asyncio.to_thread`:

```python
# check_same_thread=False: safe because WAL mode allows concurrent readers,
# and busy_timeout handles the rare case where asyncio.to_thread dispatches
# overlapping writes (micro-extraction + consolidation).
```

### Key Design Decisions

- **5000ms (5 seconds)** — standard SQLite recommendation for WAL-mode
  connections. Long enough to handle transient contention, short enough to
  surface real deadlocks.
- **Not configurable** — this is a safety net, not a tuning knob. No user
  would ever change this value. Adding a config field would be speculative
  abstraction.
- **WAL checkpoint unaffected** — `busy_timeout` does not change WAL
  checkpoint behavior. Checkpoints are automatic and fast.

### Savings

Eliminates intermittent `SQLITE_BUSY` errors under concurrent load. Pure
reliability improvement with no performance cost.

### Test Plan

- New test: verify the connection has `busy_timeout` set by querying
  `PRAGMA busy_timeout` after construction.
- Existing connection/wiring tests pass unchanged.

## Cross-Cutting Concerns

### Independence

All three optimizations are independent. Each can be implemented, tested,
and reverted without affecting the others.

### Observability

- Opt 1: Event cache hits reduce `read_events` calls — observable in debug
  logs (fewer "memory_append" entries during retrieval).
- Opt 2: Batch query replaces per-entity loops — no external observability
  change, but internal timing improves.
- Opt 3: `busy_timeout` prevents silent failures — reduces `OperationalError`
  occurrences in logs.

### Documentation

Update `.claude/rules/memory-architecture.md` to document:
- The events cache in GraphAugmenter (Section 5: Read Pipeline)
- The batch entity lookup method (Section 6: Knowledge Graph)
- The `busy_timeout` PRAGMA (Section 3: Storage Layer, SQLite Schema section)
