# Entity Alias Normalization Design

> Date: 2026-04-05
> Status: Approved
> Research: `docs/superpowers/reports/2026-04-05-entity-alias-temporal-confirmation-architecture-review.md`

---

## Problem

Two independent alias systems that never interact:

1. **Knowledge graph aliases** (`entities.aliases` column) — populated during triple
   ingestion, searched via O(n) scan in `find_entity()`.
2. **Dedup user aliases** (`user_aliases: frozenset` in `EventDeduplicator`) — loaded
   from config, used only for token normalization during Jaccard computation.

Neither handles possessives ("User's" creates entity "user's"), titles, or Unicode
variants. Two events referencing "Carlos" and "User" create separate graph nodes
despite being the same person.

## Solution

Two-phase approach: enhanced string normalization (Phase 1), then unified alias
registry (Phase 2).

### Phase 1: Enhanced String Normalization

New function `normalize_entity_name()` in `memory/_text.py`.

**Pipeline:**
1. Unicode NFKC normalization
2. Strip whitespace
3. Strip possessive suffixes: `'s`, `'s`, `'s` (including smart quotes)
4. Strip common titles at start: `dr.`, `mr.`, `mrs.`, `ms.`, `prof.`
5. Strip punctuation (preserve hyphens and underscores)
6. Lowercase
7. Collapse whitespace, replace spaces with underscores

**Callers:**

| Caller | Before | After |
|--------|--------|-------|
| `graph/__init__.py` `_norm()` | `strip().lower().replace(" ", "_")` | Delegates to `normalize_entity_name()` |
| `Entity.canonical_name` | Same 3-step | Uses `normalize_entity_name()` |
| `entity_linker.py` `resolve_alias()` | `name.strip().lower()` | Uses `normalize_entity_name()` |
| `dedup.py` `event_similarity()` | `_tokenize()` on entity text | Applies `normalize_entity_name()` to entity tokens before Jaccard |

### Phase 2: Unified Alias Registry

#### New SQLite table

```sql
CREATE TABLE IF NOT EXISTS alias_registry (
    alias     TEXT PRIMARY KEY,
    canonical TEXT NOT NULL,
    confidence REAL DEFAULT 0.8,
    source    TEXT DEFAULT 'config'
);
CREATE INDEX IF NOT EXISTS idx_alias_canonical ON alias_registry(canonical);
```

Added to `MemoryDatabase._init_schema()`.

#### New file: `memory/db/alias_store.py`

Two classes in one file (~80 LOC total):

**`AliasStore`** — SQLite CRUD on `alias_registry` table:
- `load_all() -> dict[str, str]` — bulk load alias to canonical mapping
- `register(alias, canonical, confidence, source)` — upsert single alias
- `register_batch(entries)` — upsert multiple
- `get_canonical(alias) -> str | None` — single lookup
- `remove_by_canonical(canonical)` — cleanup

**`AliasRegistry`** — in-memory cache wrapping `AliasStore`:
- `load()` — populate cache from store
- `resolve(name) -> str` — normalize + lookup, returns canonical or passthrough
- `register(alias, canonical, ...)` — write to store + update cache

#### Seeding at startup

`MemoryStore.__init__` seeds from three sources (highest confidence wins on conflict):

1. **Config `user_aliases`** — confidence 1.0, source `config`
2. **Static `entity_linker.py` `_ALIAS_MAP`** — confidence 0.9, source `linker`
3. **Graph entity aliases** — confidence 0.8, source `graph`

#### Wiring changes

| Component | Before | After |
|-----------|--------|-------|
| `MemoryDatabase` | `event_store`, `graph_store` properties | Also `alias_store` (lazy, same pattern) |
| `MemoryStore.__init__` | `EventDeduplicator(user_aliases=frozenset)` | Creates `AliasRegistry`, seeds, passes to dedup + graph |
| `EventDeduplicator` | `user_aliases: frozenset[str]` param | `alias_registry: AliasRegistry` param |
| `EventDeduplicator.event_similarity()` | Replaces tokens in `_user_aliases` with `_user_` | Calls `registry.resolve(token)` per entity token |
| `KnowledgeGraph.upsert_entity()` | Writes aliases to `entities.aliases` only | Also calls `registry.register()` for each alias |
| `entity_linker.py` `resolve_alias()` | Checks static `_ALIAS_MAP` | Accepts optional `AliasRegistry` param; checks registry first, falls back to `_ALIAS_MAP` |

#### Data flow

```
Startup:
  config user_aliases ---+
  entity_linker map -----+---> AliasStore (SQLite) ---> AliasRegistry (dict)
  graph entity aliases --+

Ingestion:
  "Carlos's project"
    -> normalize_entity_name -> "carlos"
    -> AliasRegistry.resolve("carlos") -> "_user_" (if configured)
    -> dedup uses "_user_" for Jaccard

Graph upsert:
  KnowledgeGraph.upsert_entity(Entity(name="Carlos"))
    -> writes to entities table (existing)
    -> registers "carlos" in AliasRegistry (new)
```

## Files

### Created

| File | LOC | Purpose |
|------|-----|---------|
| `memory/db/alias_store.py` | ~80 | `AliasStore` + `AliasRegistry` |

### Modified

| File | Change |
|------|--------|
| `memory/_text.py` | Add `normalize_entity_name()` (~20 LOC) |
| `memory/db/connection.py` | Add `alias_registry` table DDL + `alias_store` lazy property |
| `memory/db/__init__.py` | Export `AliasStore` |
| `memory/store.py` | Create `AliasRegistry`, seed, pass to dedup + graph |
| `memory/graph/__init__.py` | `_norm()` delegates to `normalize_entity_name()` |
| `memory/graph/ontology_types.py` | `Entity.canonical_name` uses `normalize_entity_name()` |
| `memory/graph/graph.py` | `upsert_entity()` registers aliases in registry |
| `memory/graph/entity_linker.py` | `resolve_alias()` checks registry first |
| `memory/write/dedup.py` | Replace `user_aliases: frozenset` with `AliasRegistry` |

### Not modified

- `agent/`, `context/`, `tools/`, `channels/` — no boundary crossings
- `memory/read/` — events normalized at write time
- `memory/write/ingester.py` — delegates to dedup
- `config/memory.py` — `user_aliases` field unchanged

### Package growth

- `memory/db/`: 4 files -> 5 (under 15 limit)
- `memory/db/__init__.py` exports: 3 -> 4 (under 12 limit)

## Testing

### New test files

| File | Cases |
|------|-------|
| `tests/test_normalize_entity_name.py` | Possessives, titles, Unicode NFKC, hyphens preserved, empty/whitespace, smart quotes |
| `tests/test_alias_store.py` | CRUD, confidence conflict resolution, `load_all()`, `remove_by_canonical()` |
| `tests/test_alias_registry.py` | `resolve()` canonical, unknown passthrough, `register()` updates cache, multi-source seeding priority |

### New contract tests

| File | Cases |
|------|-------|
| `tests/contract/test_alias_contracts.py` | Registry seeded at startup, graph upsert populates registry, dedup uses registry, alias DDL present |

### Updated existing tests

| File | Change |
|------|--------|
| `tests/test_dedup.py` | `EventDeduplicator` receives `AliasRegistry` instead of `frozenset` |
| `tests/contract/test_memory_wiring.py` | Assert `alias_store` accessible on `MemoryDatabase` |
| `tests/contract/test_memory_constants.py` | Add `ALIAS_REGISTRY_DDL` constant check |

## Migration

No data migration. `CREATE TABLE IF NOT EXISTS` creates the table on existing
databases. Graph aliases are seeded into the registry at startup automatically.

## Phase 3 readiness

The in-memory `AliasRegistry` dict enables Phase 3 (fuzzy matching) by providing
O(1) iteration over all known canonical names for Jaccard comparison at entity
write time. No architectural changes needed — fuzzy matching adds logic to the
`resolve()` lookup path.

## Risks

| Risk | Mitigation |
|------|-----------|
| Aggressive normalization loses meaning (e.g., "O'Brien" -> "obrien") | Preserve original in `entities.aliases`; test with real names |
| Alias conflicts (A->B and A->C) | Highest confidence wins; config source = 1.0 always wins |
| Registry becomes stale if graph writes bypass it | All graph writes go through `KnowledgeGraph.upsert_entity()` which updates registry |

## Open decisions

1. **Entity linker static map** — keep `_ALIAS_MAP` as seed data or migrate entirely
   into config? Keeping it as code is simpler and avoids config bloat. Recommend: keep
   as seed source.
2. **Alias confidence evolution** — should alias confidence decay if never re-observed?
   Not for Phase 1-2. Aliases are permanent once established. Revisit if false positives
   become an issue.
