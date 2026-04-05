# Architecture Review: Entity Alias Normalization & Temporal Confirmation

> Date: 2026-04-05
> Status: Complete
> Sources: 3 deep-dive research agents (entity patterns, temporal patterns, code audit)

---

## 1. Executive Summary

This review evaluates two features for the nanobot memory subsystem: **entity alias
normalization** (ensuring "User", "Carlos", "the user" resolve to the same entity) and
**temporal confirmation** (tracking when facts were last observed to still be true,
with confidence that reflects freshness rather than echo count).

### Entity Alias Normalization

**Current state:** Two independent alias systems that never interact. The dedup layer
has a `user_aliases` frozenset (PR #148) for token normalization during Jaccard
computation. The knowledge graph has a separate `entities.aliases` column populated
during triple ingestion. Neither system handles possessives, pronouns, or dynamic
alias learning.

**Recommendation:** A **three-layer approach** — enhanced string normalization (immediate),
unified alias registry (near-term), and optional fuzzy matching (future). Do NOT build
a full entity resolution pipeline (Graphiti-style) — it's overengineered for a
single-user personal agent.

### Temporal Confirmation

**Current state:** Events have `timestamp` and `created_at`. Profile beliefs have
`last_seen_at`. Dedup merge takes the newer timestamp and discards the older. `ttl_days`
exists in schema but is never enforced. No `last_confirmed` field. No time-based
confidence decay. No distinction between agent echo and genuine reconfirmation.

**Recommendation:** A **two-phase approach** — add `last_confirmed` field with
source-aware reconfirmation (immediate), then operationalize `ttl_days` enforcement
with stability-aware retrieval decay (near-term). Do NOT implement full bitemporal
tracking — it's unnecessary for a single-user agent with SQLite.

### Build order: Entity aliases first, then temporal confirmation.

Entity aliases are a prerequisite — without correct entity resolution, temporal
confirmation would track freshness of duplicate entities rather than canonical facts.

---

## 2. Research Findings: Entity Alias Normalization

### Industry Patterns

| System | Approach | Strengths | Weaknesses |
|--------|----------|-----------|------------|
| **Graphiti/Zep** | 3-tier cascade: exact → fuzzy (MinHash/LSH) → LLM | Most thorough; entropy gating prevents short-name false positives | Complex (300+ LOC for resolution alone) |
| **mem0** | Embedding similarity only (cosine ≥ 0.7) | Simple; no alias table needed | "User" and "Carlos" won't merge; no string normalization |
| **Letta/MemGPT** | No entity resolution; agent manages structured text blocks | Sidesteps the problem entirely | Only works for bounded core memory, not event stores |
| **cognee** | Post-extraction embedding cache (cosine ≥ 0.8) | Growing canonical vocabulary during ingestion | POC only; not in production pipeline |
| **Wikidata** | Canonical Q-IDs with human-curated alias lists per language | Gold standard for precision | Requires manual curation at scale |

### Key Design Insights

**1. Deterministic first, probabilistic only for ambiguity.** Graphiti's cascade is the
industry best practice: exact match (free) → fuzzy match (cheap) → LLM (expensive,
only for what remains). This mirrors the "cheap filters first" pattern in database
query optimization.

**2. Possessive and title stripping is universally needed.** Every system that handles
entity names normalizes possessives ("User's" → "User") and titles ("Dr. Smith" →
"Smith"). Nanobot's current `_norm()` misses these.

**3. Self-reference mapping is the highest-value, lowest-cost alias.** mem0's approach
of mapping "I", "me", "my" → user_id in the extraction prompt handles the most common
alias case with zero additional infrastructure.

**4. Dynamic alias accumulation beats static tables.** Graphiti learns aliases during
entity resolution and promotes them to the canonical node. cognee's embedding cache
grows during ingestion. Static tables (Nanobot's current `entity_linker.py`) miss
any alias not pre-registered.

**5. Entity linking models (BLINK, GENRE, ReFinED) are overkill.** These resolve
mentions against a KB of millions of entities. A personal agent has hundreds at most.
The model download cost alone exceeds the benefit.

### Minimum Viable Entity Normalization

For a single-user personal agent, 80% of the value comes from:
1. Enhanced string normalization (possessives, titles, Unicode)
2. Self-reference mapping in extraction prompts
3. A unified alias registry (graph aliases + dedup aliases merged)
4. Alias accumulation when entities co-occur in context

---

## 3. Research Findings: Temporal Confirmation Patterns

### Industry Patterns

| System | Temporal Model | Freshness | Contradiction | Echo Handling |
|--------|---------------|-----------|---------------|---------------|
| **Graphiti/Zep** | Full bitemporal (`valid_at`/`invalid_at`/`created_at`/`expired_at`) | Temporal queries filter by validity range | Old edge gets `invalid_at` + `expired_at` | Not addressed explicitly |
| **Generative Agents** | Last-access recency decay (`0.99^hours_since_access`) | Aggressive decay; memories fade in ~1 week | Not implemented | Implicit: accessed memories stay fresh |
| **mem0** | `created_at`/`updated_at` only; LLM overwrites on contradiction | No decay; all memories equal regardless of age | LLM decides UPDATE/DELETE/NONE | Not addressed |
| **NELL** | Confidence accumulation from independent sources | No decay; old beliefs fossilize | Known weakness — high-confidence old beliefs resist update | Not addressed (caused fossilization) |
| **ACT-R** | Base-level activation: `ln(Σ t_j^(-d))` over all retrievals | Mathematical decay model; frequently-accessed items stay active | Not modeled | Implicit through access-based formula |

### Key Design Insights

**1. `last_confirmed` is distinct from `updated_at`.** `updated_at` changes when content
changes. `last_confirmed` changes when the fact is re-observed without content change.
A fact created 6 months ago but confirmed yesterday is clearly still relevant — but
without `last_confirmed`, it looks 6 months stale.

**2. Confidence should NOT decay uniformly with time.** NELL's experience shows that
pure confidence accumulation causes fossilization. But Generative Agents' aggressive
decay (half-life ~1 week) makes stable facts disappear. The solution is
**stability-aware decay**: high-stability facts (preferences, identity) decay slowly
or not at all; low-stability facts (task status, project state) decay quickly.

**3. Source tracking prevents the echo problem.** When the agent restates a fact from
memory, that restatement is NOT genuine reconfirmation. The observation source must be
tracked: `user_message` and `tool_result` are genuine; `agent_output` is echo. Only
genuine sources should bump `last_confirmed` and confidence.

**4. TTL enforcement is the lowest-cost temporal improvement.** Nanobot already has
`ttl_days` in the schema — it just needs a check at retrieval time. This immediately
handles task-related facts that have a natural expiry.

**5. Full bitemporal tracking is unnecessary for single-user agents.** Graphiti needs
it because they serve multi-user, multi-source scenarios where facts arrive out of
order. A personal agent's facts arrive in conversation order — transaction time ≈ valid
time. Simple `created_at` + `last_confirmed` + `superseded_at` covers 95% of temporal
needs.

**6. The distinction between "user confirms" and "agent restates" is critical.** Without
it, the echo problem inflates confidence of frequently-mentioned facts to near 1.0,
making even wrong facts unfalsifiable. The fix is structural (source field on
observations), not heuristic (prompt instructions).

### The `last_confirmed` Pattern

The most impactful temporal improvement. Implementation approaches:

| Approach | Schema Change | Code Change | Value |
|----------|--------------|-------------|-------|
| **Dedicated field** (`last_confirmed` on events) | 1 column | ~30 LOC in ingester | High — direct freshness signal |
| **Observation log** (separate table) | 1 table | ~80 LOC | Medium — full history but overkill |
| **Confidence bump only** (no separate field) | None | ~10 LOC | Low — conflates content update with reconfirmation |

**Recommended:** Dedicated field. The observation log is overkill for a single-user
agent, and confidence-bump-only loses the temporal signal that retrieval scoring needs.

---

## 4. Current Codebase Analysis

### Entity Handling: Two Independent Systems

**System A — Knowledge Graph Aliases (`memory/graph/`)**:
- `entities` table: `name TEXT PRIMARY KEY, aliases TEXT DEFAULT ''`
- Aliases stored as comma-separated string in a single column
- Populated during triple ingestion via `KnowledgeGraph.upsert_entity()`
- Searched in `find_entity()` by scanning ALL entities' alias strings — O(n)
- Entity canonicalization: `name.strip().lower().replace(" ", "_")`

**System B — Dedup User Aliases (`memory/write/dedup.py`)**:
- `user_aliases: frozenset[str]` from `MemoryConfig.user_aliases`
- Used only in `event_similarity()` to replace alias tokens with `_user_`
- Never touches the knowledge graph
- Never persisted — exists only in memory during dedup computation

**They never interact.** Graph aliases are never populated from dedup aliases.
Dedup aliases are never consulted during graph operations.

### Entity Flow Through Ingestion

```
MemoryEvent.entities: ["Carlos"]    # bare string, no type, no canonical ID
        |
        v
EventIngester.append_events()       # entities field used ONLY for Jaccard dedup
        |                            # (token overlap computation in dedup.py)
        v
EventIngester.ingest_graph_triples() # uses event.triples, NOT event.entities
        |
        v
entity_classifier.classify_entity_type("Carlos")  # infers type (PERSON)
        |
        v
KnowledgeGraph.upsert_entity(Entity(name="carlos", type="person"))
        |
        v
entities table: name="carlos", type="person", aliases=""
```

**Critical gap:** `MemoryEvent.entities` (the bare string list) is never linked to the
graph. Two events with `entities: ["Carlos"]` and `entities: ["User"]` create two
separate graph nodes. The dedup `user_aliases` normalizes tokens for similarity scoring
but does not canonicalize the entities themselves.

### Temporal Handling: Schema-Rich, Implementation-Poor

**Fields that exist and ARE used:**
- `events.timestamp` — event occurrence time
- `events.created_at` — DB insertion time
- `events.status` — `active` | `superseded`
- `profile.meta[key].last_seen_at` — updated on belief confidence touch
- `profile.meta[key].created_at` — set once at belief creation
- `dedup: last_merged_at` — set during event merge
- `supersedes_event_id` / `superseded_at` — supersession linkage

**Fields that exist but are NEVER used:**
- `events.metadata.ttl_days` — parsed and stored, never enforced at retrieval
- `entities.first_seen` / `entities.last_seen` — set on upsert, never queried
- `edges.timestamp` — set on creation, never updated on re-observation

**Fields that DON'T exist but are needed:**
- `last_confirmed` — when a fact was last observed from a genuine source
- `confirmation_count` — how many times observed (distinct from `merged_event_count`)
- `observation_source` — whether the observation was genuine or echo

### Confidence Lifecycle

Profile beliefs use delta-based confidence:
- New belief: 0.65 baseline
- Normal re-observation: +0.03
- Conflict (old value): -0.12
- Conflict (new value): -0.20
- Resolved winner: +0.08

**No time-based decay.** A belief from a year ago with no re-observations has the same
confidence as one from yesterday. The `stability` field (high/medium/low) from
`EventClassifier` exists on events but is not used to modulate decay.

### Retrieval Recency Scoring

The `RetrievalScorer` applies recency as a retrieval boost (not confidence decay).
The decay formula uses `recency_half_life_days` configuration. This is architecturally
correct — decay affects retrieval priority, not stored confidence.

---

## 5. Current Gaps / Weaknesses

### Entity Gaps (Ordered by Impact)

| # | Gap | Impact | Current Workaround |
|---|-----|--------|--------------------|
| 1 | No possessive stripping | "User's project" creates entity "user's", not "user" | None |
| 2 | Two independent alias systems | Graph aliases and dedup aliases never interact | None |
| 3 | No dynamic alias learning | New entity names require manual config | Static `entity_linker.py` table |
| 4 | `entities` list never linked to graph | Event entities and graph entities are disjoint | Triples drive graph; entities list is dedup-only |
| 5 | No pronoun/reference resolution | "the user", "he", "my boss" create separate entities | Extraction prompt partially handles "I"/"me" |
| 6 | Alias search is O(n) | `find_entity()` scans all entities' alias strings | Acceptable at current scale (<100 entities) |
| 7 | Entity type confidence not stored | Classification picks one type; alternatives lost | Single-type model works at current scale |

### Temporal Gaps (Ordered by Impact)

| # | Gap | Impact | Current Workaround |
|---|-----|--------|--------------------|
| 1 | No `last_confirmed` field | Cannot distinguish "old and stale" from "old but recently confirmed" | Merge takes newer timestamp (lossy) |
| 2 | TTL never enforced | Expired events (e.g., "deploy by Friday") persist indefinitely | Manual consolidation may clean up |
| 3 | No echo vs genuine distinction | Agent restatements inflate confidence | PR #144 prompt partially mitigates |
| 4 | No time-based confidence decay | Year-old unconfirmed beliefs have same weight as fresh ones | Retrieval recency scoring partially compensates |
| 5 | Merge discards older timestamp | No `first_occurrence` tracking | `created_at` partially captures this |
| 6 | Graph edge timestamps never updated | Relationships appear stale even if re-observed | None |
| 7 | No observation history | Cannot audit "how many times was this fact seen?" | `merged_event_count` tracks merge count only |

---

## 6. Solution Options for Entity Alias Normalization

### Option A: Enhanced String Normalization (Recommended — Phase 1)

**Core idea:** Improve the `_norm()` function and entity processing to handle
possessives, titles, and Unicode consistently across both alias systems.

**Changes:**
- `memory/_text.py`: Add `_normalize_entity_name()` that strips possessives
  (`'s$`), titles (Dr., Mr., Mrs.), punctuation, and applies Unicode NFKC
- `memory/graph/entity_linker.py`: Use `_normalize_entity_name()` instead
  of simple `_norm()`
- `memory/write/dedup.py`: Apply same normalization to entity tokens before
  Jaccard computation

**Advantages:**
- Zero new infrastructure; enhances existing functions
- Fixes the "User's" vs "User" problem structurally
- Consistent normalization across both alias systems
- ~20 LOC change, independently testable

**Disadvantages:**
- Doesn't solve "Carlos" vs "the user" (different strings, not just formatting)
- Static rules; doesn't learn from context

**Complexity:** LOW
**Migration:** None — backward compatible
**Risk:** Aggressive stripping could lose meaningful distinctions (e.g., "O'Brien" → "obrien")

### Option B: Unified Alias Registry (Recommended — Phase 2)

**Core idea:** Merge the two alias systems into a single registry that serves both
the knowledge graph and the dedup pipeline.

**Changes:**
- Add `alias_registry` table: `(alias TEXT PRIMARY KEY, canonical TEXT NOT NULL,
  confidence REAL DEFAULT 0.8, source TEXT DEFAULT 'config')`
- `MemoryStore.__init__` seeds registry from config `user_aliases` + graph entity
  aliases + static `entity_linker.py` entries
- `EventDeduplicator.event_similarity()` uses registry for token normalization
  (replaces the current `_user_aliases` frozenset)
- `KnowledgeGraph.upsert_entity()` registers aliases in the unified registry
- `EventIngester` normalizes `MemoryEvent.entities` through the registry before
  graph ingestion

**Advantages:**
- Single source of truth for all entity aliases
- Graph aliases automatically benefit dedup and vice versa
- Aliases accumulate dynamically as new entities are discovered
- Confidence-weighted: high-confidence aliases auto-resolve, low-confidence flagged

**Disadvantages:**
- New table in SQLite schema
- Migration path for existing graph aliases
- Registry must be loaded into memory for O(1) lookup (memory cost)
- More complex than the current two-system approach

**Complexity:** MEDIUM
**Migration:** Schema migration (new table), data migration (existing aliases)
**Risk:** Registry inconsistency if updated from multiple code paths without locking

### Option C: Fuzzy Matching at Entity Write Time (Future — Phase 3)

**Core idea:** Before creating a new entity in the graph, compute normalized
similarity against existing entities and merge if above threshold.

**Changes:**
- `KnowledgeGraph.upsert_entity()`: Before insert, query existing entities with
  normalized Jaccard ≥ 0.85 (following Graphiti's approach but without MinHash/LSH)
- Entropy gating: Skip fuzzy match for names shorter than 4 characters
- On match: Merge into existing entity, add new name as alias

**Advantages:**
- Catches typos and minor variations automatically
- No LLM cost
- Follows Graphiti's proven pattern (simplified)

**Disadvantages:**
- Jaccard on short entity names is unreliable ("Sam" vs "Pam")
- Adds latency to entity ingestion (~1ms per entity for small stores)
- False positives require human-in-the-loop resolution

**Complexity:** MEDIUM
**When justified:** When entity count exceeds ~50 and false negatives from string
matching become noticeable in retrieval quality

### Option D: LLM-Based Entity Resolution (NOT Recommended)

**Core idea:** Use an LLM call to verify ambiguous entity matches (Graphiti's Tier 3).

**Why not:** Adds an LLM call per ambiguous entity per ingestion. For a personal
agent processing 5-10 events per turn, this could add 5-10 LLM calls to each turn.
The cost/benefit ratio is wrong — the entity space is small enough that deterministic
methods handle 95%+ of cases.

**When justified:** Only if entity count exceeds ~200 and the agent serves multiple
users with overlapping entity names.

---

## 7. Solution Options for Temporal Confirmation

### Option A: `last_confirmed` Field with Source-Aware Reconfirmation (Recommended — Phase 1)

**Core idea:** Add a `last_confirmed` timestamp to events, updated only when a fact is
re-observed from a genuine source (user message or tool result, NOT agent output).

**Changes:**
- `events` table: Add `last_confirmed TEXT` column (nullable, defaults to `created_at`)
- `EventIngester.append_events()`: When dedup detects a near-duplicate (merge path),
  check the source. If genuine (`source != "agent_echo"`), update `last_confirmed`
  and bump confidence. If echo, update `last_merged_at` but NOT `last_confirmed`.
- `MemoryEvent`: Add `last_confirmed: str = ""` field
- `RetrievalScorer`: Use `last_confirmed` (when available) instead of `timestamp`
  for recency scoring — a fact confirmed yesterday scores higher than one created
  6 months ago with no reconfirmation

**Source classification logic:**
```
user_message → genuine (boost last_confirmed + confidence)
tool_result  → genuine (boost last_confirmed + confidence)
agent_output → echo (update last_merged_at only)
consolidation → genuine (consolidation reviews the full conversation)
```

**Advantages:**
- Directly solves "old but recently confirmed" vs "old and stale"
- Prevents echo-driven confidence inflation
- Minimal schema change (1 column)
- Source classification uses existing `event.source` field
- ~40 LOC change in ingester + scorer

**Disadvantages:**
- Source classification is heuristic (micro-extraction sets source from channel +
  tool hints, not explicitly "user" vs "agent")
- Requires distinguishing between micro-extraction from user turns vs agent turns

**Complexity:** LOW-MEDIUM
**Migration:** Add column with default = `created_at` for existing events

### Option B: Operationalize TTL Enforcement (Recommended — Phase 2)

**Core idea:** Enforce `ttl_days` at retrieval time. Events past their TTL are
excluded from results (soft expiry, not deletion).

**Changes:**
- `EventStore.search_fts()` and `EventStore.search_vector()`: Add optional
  `exclude_expired: bool = True` parameter. When true, filter out events where
  `ttl_days` is set and `created_at + ttl_days < now()`
- Alternatively, add the filter in `MemoryRetriever._retrieve_unified()`

**Advantages:**
- Zero schema changes — `ttl_days` already exists and is populated
- ~15 LOC change
- Immediately handles stale task-related facts

**Disadvantages:**
- TTL is set at extraction time — if the LLM doesn't set it, there's no expiry
- Hard cutoff (expired = invisible) vs gradual decay

**Complexity:** LOW
**Migration:** None

### Option C: Stability-Aware Retrieval Decay (Recommended — Phase 3)

**Core idea:** Use the `stability` field (already classified by `EventClassifier`)
to vary the recency decay half-life in retrieval scoring.

**Changes:**
- `RetrievalScorer`: Instead of a single `recency_half_life_days`, use stability-based
  half-lives:
  - `high` stability (preferences, identity facts): 365 days (nearly no decay)
  - `medium` stability (relationships, constraints): 90 days
  - `low` stability (task status, decisions): 14 days

**Advantages:**
- "User's name is Carlos" (high stability) persists; "build is broken" (low stability)
  fades quickly
- Uses existing classified field — no new extraction logic
- ~20 LOC change in scorer

**Disadvantages:**
- Stability classification is heuristic (based on event type and incident markers)
- Miscategorized stability affects retrieval inappropriately
- Half-life values need empirical tuning

**Complexity:** LOW
**Migration:** None

### Option D: Observation Log Table (NOT Recommended Now)

**Core idea:** Separate table tracking every observation of every fact, with source,
timestamp, and context snippet.

**Why defer:** Overkill for a single-user agent. The `last_confirmed` field plus
`merged_event_count` capture the essential signal without the storage and query
overhead of a separate table. Consider if observation-level auditing becomes a
requirement.

### Option E: Full Bitemporal Tracking (NOT Recommended)

**Core idea:** Add `valid_from`/`valid_to` to events for full temporal range queries.

**Why not:** Graphiti needs bitemporal tracking because it serves multi-user,
multi-source scenarios where facts arrive out of order. Nanobot is a single-user
agent where facts arrive in conversation order — transaction time ≈ valid time.
The complexity cost (temporal range queries, overlapping period constraints, cascade
invalidation) vastly exceeds the benefit.

---

## 8. Recommended Architecture

### Entity Alias: Three-Layer Model

```
Layer 1: String Normalization (all code paths)
├── Lowercase, strip, collapse whitespace
├── Strip possessives ('s$)
├── Strip titles (Dr., Mr., etc.)
└── Unicode NFKC

Layer 2: Alias Registry (unified, in SQLite)
├── Seeded from: config user_aliases + entity_linker.py + graph aliases
├── Updated by: graph entity upsert, micro-extraction entity co-occurrence
├── Queried by: dedup similarity, graph lookup, retrieval scoring
└── Confidence-weighted (auto-resolve above 0.8, flag below)

Layer 3: Fuzzy Matching (optional, entity write time)
├── Normalized Jaccard ≥ 0.85 against existing entities
├── Entropy gate for short names
└── Only activated when entity count > ~50
```

### Temporal Confirmation: Layered Model

```
Layer 1: last_confirmed (on events and profile beliefs)
├── Updated on genuine re-observation (source = user/tool)
├── NOT updated on agent echo (source = agent)
├── Used in retrieval recency scoring
└── Migration: default to created_at for existing events

Layer 2: TTL Enforcement (at retrieval time)
├── Check ttl_days on events during search
├── Expired events excluded from results
└── No schema change needed

Layer 3: Stability-Aware Decay (in retrieval scorer)
├── High stability: 365-day half-life
├── Medium stability: 90-day half-life
├── Low stability: 14-day half-life
└── Uses existing stability field from EventClassifier
```

---

## 9. Phased Implementation Strategy

### Phase 1: Foundation (1 PR, ~2 hours)

**Entity:** Enhanced string normalization
- Add `_normalize_entity_name()` to `_text.py`
- Apply in entity_linker, graph upsert, dedup token normalization
- Tests: possessives, titles, Unicode variants

**Temporal:** Add `last_confirmed` field
- Add column to events table (with migration default)
- Add field to MemoryEvent
- Update `merge_events()` to set `last_confirmed` on genuine re-observation
- Update retrieval scoring to prefer `last_confirmed` over `timestamp`

### Phase 2: Unification (1 PR, ~3 hours)

**Entity:** Unified alias registry
- New `alias_registry` table
- Seed from config + entity_linker + graph aliases
- Dedup uses registry instead of raw `user_aliases` frozenset
- Graph upsert registers new aliases in registry

**Temporal:** TTL enforcement + stability-aware decay
- Filter expired events at retrieval time
- Variable half-life based on stability field

### Phase 3: Intelligence (Future, only if needed)

**Entity:** Fuzzy matching at write time
- Jaccard-based entity dedup before graph insert
- Entropy gating for short names

**Temporal:** Observation-aware confidence
- Separate confirmation_count from merged_event_count
- Source diversity tracking (N independent sources)

---

## 10. Risks and Tradeoffs

### Entity Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Aggressive normalization loses meaning | Medium | High | Test with real entity names; preserve original in aliases |
| Unified registry becomes single point of failure | Low | High | Registry is a cache; rebuild from graph + config on startup |
| Alias conflicts (A→B and A→C) | Low | Medium | Confidence-weighted; highest confidence wins |
| O(n) alias lookup at scale | Low (n < 100) | Low | Index if n > 500; fine at current scale |

### Temporal Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Source classification is wrong | Medium | Medium | Conservative: default to "genuine"; only mark as echo when certain |
| TTL too aggressive (useful facts expire) | Low | Medium | TTL only set by LLM extraction; most facts have no TTL |
| Stability-aware decay miscategorizes | Medium | Low | Start with conservative half-lives; tune based on retrieval quality |
| Migration breaks existing events | Low | High | Default last_confirmed to created_at; additive change |

### Overengineering Risks

| Feature | Overengineered If... | Right-Sized If... |
|---------|---------------------|-------------------|
| Fuzzy entity matching | Entity count < 50 | Entity count > 100 with typo-driven false negatives |
| Observation log table | Single-user agent | Multi-user or compliance requirements |
| Full bitemporal tracking | Single-user, conversation-ordered facts | Multi-source, out-of-order facts |
| LLM entity resolution | Entity space < 200 | Multi-user with overlapping entity names |

---

## 11. Open Questions

1. **How should `source` be classified for micro-extraction?** Currently, micro-extraction
   sets `source` from channel + tool hints (e.g., "web,read_file"). This doesn't
   distinguish "extracted from user message" vs "extracted from assistant response."
   The `_extract_and_ingest()` method receives both `user_message` and
   `assistant_message` — should it tag events with which message they came from?

2. **Should the alias registry be in-memory or disk-based?** In-memory gives O(1)
   lookup but must be rebuilt on startup. Disk-based (SQLite table) persists but adds
   I/O per lookup. At current scale (<100 entities), in-memory is fine.

3. **What half-life values are correct for stability-aware decay?** The 365/90/14 day
   split is a hypothesis. Empirical tuning requires observing retrieval quality over
   real conversations — possibly via the existing memory-eval benchmark.

4. **Should `last_confirmed` apply to profile beliefs too?** Profile beliefs already
   have `last_seen_at` which serves a similar purpose. Should we align these, or keep
   them separate?

5. **How should alias confidence evolve?** Config-sourced aliases start at 1.0.
   Graph-discovered aliases start at 0.8. Should confidence decay over time if the
   alias is never re-observed? Or should aliases be permanent once established?

6. **What happens to existing events with no `last_confirmed`?** Migration sets
   `last_confirmed = created_at`. But this means all existing events look equally
   "recently confirmed." Should we instead set `last_confirmed = NULL` and treat
   NULL as "never confirmed — use created_at as fallback"?

---

## 12. Final Recommendation

### Build entity alias normalization first.

Entity aliases are a prerequisite for temporal confirmation to be meaningful. Without
correct entity resolution, `last_confirmed` would track freshness of duplicate entities
rather than canonical facts. Fix the entity foundation, then add temporal intelligence.

### Phase 1 (Immediate): Foundation

**Entity:**
- Enhanced `_normalize_entity_name()` — strip possessives, titles, Unicode
- Apply consistently across `_text.py`, `entity_linker.py`, `dedup.py`, `graph.py`

**Temporal:**
- Add `last_confirmed` column to events table
- Add `last_confirmed` field to MemoryEvent
- Update `merge_events()`: set `last_confirmed` on genuine re-observation
- Update retrieval scoring: use `max(last_confirmed, timestamp)` for recency

**This is a localized refactor, not a deep architectural change.** It touches 4-5 files,
adds 1 column, and enhances existing functions. No new subsystems, no new tables, no
new dependencies.

### Phase 2 (Near-term): Unification + TTL

**Entity:**
- Unified alias registry table (replaces both dedup `user_aliases` and graph aliases)
- Graph upsert populates registry; dedup reads from registry

**Temporal:**
- TTL enforcement at retrieval time
- Stability-aware decay half-lives

**This is a medium refactor.** New table, migration, registry wiring. But no architectural
boundary changes — the registry lives in `memory/db/` or `memory/persistence/`, the
dedup layer reads from it, the graph layer writes to it.

### What should wait:
- Fuzzy matching (Phase 3 — only if entity count > ~50)
- LLM entity resolution (not needed for single-user)
- Observation log table (not needed for single-user)
- Full bitemporal tracking (not needed for conversation-ordered facts)

### Architecture boundaries:
- String normalization: `memory/_text.py` (shared utility)
- Alias registry: `memory/db/` (storage) + `memory/persistence/` (CRUD)
- Dedup reads from registry: `memory/write/dedup.py`
- Graph writes to registry: `memory/graph/graph.py`
- Retrieval uses `last_confirmed`: `memory/read/scoring.py`
- No changes to `agent/`, `context/`, `tools/`, or `channels/`

---

## References

### Production Frameworks
- Graphiti/Zep: github.com/getzep/graphiti — 3-tier entity resolution, bitemporal edges
- mem0: github.com/mem0ai/mem0 — embedding-only entity matching, LLM update decisions
- Letta/MemGPT: github.com/letta-ai/letta — structured text blocks, no entity resolution
- cognee: github.com/topoteretes/cognee — post-extraction embedding canonicalization (POC)
- ReFinED (Amazon Science): github.com/amazon-science/ReFinED — efficient zero-shot entity linking

### Academic and Research
- Park et al., "Generative Agents" (UIST 2023) — recency decay formula
- Anderson & Lebiere, "Atomic Components of Thought" (1998) — ACT-R activation model
- Gärdenfors, "Knowledge in Flux" (1988) — AGM belief revision
- de Kleer, "An Assumption-based TMS" (1986) — truth maintenance
- Kulkarni & Michels, "Temporal features in SQL:2011" — bitemporal standard
- NELL (CMU) — confidence accumulation, fossilization problem
- Halpin et al., "When owl:sameAs isn't the Same" (2010) — alias identity pitfalls

### Architectural Patterns
- Graphiti 3-tier cascade: exact → fuzzy (MinHash/LSH) → LLM
- CQRS/Event Sourcing: write-time reconciliation pattern
- Wikidata: canonical IDs + human-curated alias lists
