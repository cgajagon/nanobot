# Temporal Confirmation Design

> Date: 2026-04-06
> Status: Approved
> Research: `docs/superpowers/reports/2026-04-05-entity-alias-temporal-confirmation-architecture-review.md`
> Prerequisite: Entity alias normalization (PR #150, merged)

---

## Problem

Events have `timestamp` and `created_at` but no way to track when a fact was last
genuinely re-observed. A preference stated 6 months ago and confirmed yesterday looks
identical to one stated 6 months ago and never mentioned again. `ttl_days` exists in
the schema but is never enforced at retrieval time. The recency decay uses a single
half-life regardless of fact stability.

Additionally, agent restatements from memory (echo) can inflate temporal freshness.
When the agent says "I remember you use Python" and micro-extraction picks it up as
a duplicate, the merged event's timestamp gets refreshed — making the fact look recently
confirmed when it was only recently echoed.

## Solution

Three complementary changes, all within the memory subsystem:

1. **`last_confirmed` field** with source-aware echo detection
2. **TTL enforcement** at retrieval time
3. **Stability-aware decay** in retrieval scoring

### Phase 1: `last_confirmed` with Source-Aware Echo Detection

#### Schema change

Add `last_confirmed TEXT` column to the `events` table via ALTER TABLE migration
in `connection.py`. For existing events, the column is NULL — retrieval falls back
to `timestamp`.

Migration pattern (safe for SQLite, metadata-only operation):
```python
def _migrate_schema(self) -> None:
    try:
        self._conn.execute("ALTER TABLE events ADD COLUMN last_confirmed TEXT")
    except sqlite3.OperationalError:
        pass  # column already exists
```

Called after `_init_schema()`.

#### MemoryEvent fields

Add to the `MemoryEvent` Pydantic model:
- `last_confirmed: str = ""` — when the fact was last genuinely observed
- `source_role: str = ""` — origin of the fact: `"user"`, `"assistant"`, `"tool"`,
  `"consolidation"`, or empty (unknown/legacy)

#### Source role classification

**Micro-extraction:** Add `source_role` to `_MICRO_EXTRACT_TOOL` schema:
```python
"source_role": {
    "type": "string",
    "enum": ["user", "assistant"],
    "description": "Whether the fact originated from the user's message or the assistant's response",
},
```

The LLM already sees user and assistant messages as separate roles in the conversation
(system + user + assistant messages). It can determine which message a fact originated
from.

**Consolidation:** Set `source_role="consolidation"` on extracted events (reviews
full conversation, always genuine).

**Default:** Empty `source_role` is treated as genuine (conservative — avoids
penalizing legacy events).

#### Source role to `last_confirmed` mapping

| `source_role` | Genuine? | Bumps `last_confirmed`? |
|---------------|----------|------------------------|
| `"user"` | Yes | Yes |
| `"tool"` | Yes | Yes |
| `"consolidation"` | Yes | Yes |
| `""` (empty/unknown) | Default yes | Yes |
| `"assistant"` | No (echo) | No |

#### Merge behavior

In `merge_events()` in `dedup.py`, when merging a near-duplicate:

```
if source_role != "assistant":
    merged["last_confirmed"] = _utc_now_iso()
else:
    # Echo — preserve existing last_confirmed, don't bump
    pass
```

Normal merge behavior (entity union, confidence averaging, timestamp update) happens
regardless of source role. Only `last_confirmed` is gated.

#### Retrieval scoring

In `scoring.py`, change recency computation to prefer `last_confirmed`:

```python
recency_ts = str(item.get("last_confirmed") or item.get("timestamp", ""))
recency = RetrievalPlanner.recency_signal(recency_ts, half_life_days=half_life)
```

A fact confirmed yesterday scores higher than one created 6 months ago with no
reconfirmation.

#### Echo prevention flow

```
Turn 1: User says "I use Python"
  → micro-extraction: source_role="user"
  → stored, last_confirmed = now()

Turn 5: User asks about weather
  → Agent responds mentioning Python from memory
  → micro-extraction: source_role="assistant" (echo)
  → dedup: matches existing event → merge
  → last_confirmed NOT bumped (source_role="assistant")
  → fact's last_confirmed still reflects Turn 1
```

### Phase 2: TTL Enforcement at Retrieval

#### Current state

`ttl_days` is populated by extraction (EventClassifier defaults task events to
`ttl_days=30`). The snapshot module counts expired events. But retrieval never
checks it — expired events appear in results.

#### Change

Add `_filter_expired()` in `retriever.py`, called after RRF fusion and before scoring:

```python
def _filter_expired(items: list[dict], now: datetime) -> list[dict]:
    filtered = []
    for item in items:
        ttl = item.get("ttl_days")
        if not isinstance(ttl, int) or ttl <= 0:
            filtered.append(item)  # no TTL = never expires
            continue
        ts_str = str(item.get("last_confirmed") or item.get("timestamp", ""))
        ts = _to_datetime(ts_str)
        if ts is None:
            filtered.append(item)  # no timestamp = can't check
            continue
        age_days = (now - ts).total_seconds() / 86400.0
        if age_days <= ttl:
            filtered.append(item)
        # else: expired, skip
    return filtered
```

TTL age uses `last_confirmed` when available — a fact with `ttl_days=30` confirmed
5 days ago should not expire even if originally created 60 days ago.

**Soft expiry:** Events are excluded from retrieval, not deleted. They remain
available for consolidation, auditing, and reactivation if re-observed.

### Phase 3: Stability-Aware Decay

#### Current state

Retrieval scoring uses a single `half_life_days` (default 60) from the retrieval
policy. The `stability` field (high/medium/low) is classified by `EventClassifier`
but only used for a small additive boost (`_STABILITY_BOOST`: +0.03/+0.01/-0.02).

#### Change

Use stability-aware half-lives for the recency decay curve:

| Stability | Half-life | Effect | Examples |
|-----------|-----------|--------|----------|
| `high` | 365 days | Near-zero decay | Preferences, identity facts |
| `medium` | 90 days | Moderate decay | Relationships, constraints |
| `low` | 14 days | Fast decay | Task status, decisions |

New constant in `scoring.py`:
```python
_STABILITY_HALF_LIFE: dict[str, float] = {
    "high": 365.0,
    "medium": 90.0,
    "low": 14.0,
}
```

In `score_items()`:
```python
recency_ts = str(item.get("last_confirmed") or item.get("timestamp", ""))
stability = str(item.get("stability", "medium")).strip().lower()
half_life = _STABILITY_HALF_LIFE.get(stability, 90.0)
recency = RetrievalPlanner.recency_signal(recency_ts, half_life_days=half_life)
```

The existing `_STABILITY_BOOST` additive adjustment stays — it's complementary
(small score bump vs decay curve shape).

The per-intent `half_life_days` from retrieval policy becomes the default when
stability is unknown. Stability-aware half-life only applies when the field is present.

## Files Modified

| File | Change | LOC |
|------|--------|-----|
| `memory/db/connection.py` | ALTER TABLE migration for `last_confirmed` | ~10 |
| `memory/event.py` | Add `last_confirmed` and `source_role` fields | ~3 |
| `memory/write/dedup.py` | Conditionally bump `last_confirmed` in `merge_events()` | ~10 |
| `memory/write/micro_extractor.py` | Add `source_role` to tool schema | ~5 |
| `memory/write/coercion.py` | Preserve `source_role` through pipeline | ~3 |
| `memory/consolidation_pipeline.py` | Set `source_role="consolidation"` | ~3 |
| `memory/read/retriever.py` | Add `_filter_expired()` TTL filter | ~15 |
| `memory/read/scoring.py` | Use `last_confirmed` + stability-aware half-life | ~10 |

## Files NOT Modified

- `agent/`, `context/`, `tools/`, `channels/` — no boundary crossings
- `memory/db/event_store.py` — no query changes
- `memory/graph/` — no temporal changes
- `config/memory.py` — half-lives are constants, not configurable

No new files created. No package growth.

## Testing

### Unit tests

| Test | Verifies |
|------|----------|
| `test_merge_bumps_last_confirmed_for_user_source` | source_role="user" sets last_confirmed |
| `test_merge_skips_last_confirmed_for_assistant_echo` | source_role="assistant" does NOT set last_confirmed |
| `test_merge_bumps_last_confirmed_for_consolidation` | source_role="consolidation" sets last_confirmed |
| `test_merge_default_empty_source_role_is_genuine` | Empty source_role defaults to genuine |
| `test_filter_expired_respects_ttl` | Events past TTL excluded |
| `test_filter_expired_uses_last_confirmed` | TTL age uses last_confirmed when available |
| `test_filter_expired_no_ttl_passes_through` | Events without TTL always included |
| `test_stability_half_life_high` | High stability → 365-day half-life |
| `test_stability_half_life_low` | Low stability → 14-day half-life |
| `test_recency_uses_last_confirmed` | Scorer uses last_confirmed over timestamp |

### Contract tests

| Test | Verifies |
|------|----------|
| `test_last_confirmed_column_exists` | Migration creates the column |
| `test_source_role_in_micro_extract_schema` | Tool schema includes source_role |
| `test_stability_half_life_constants_defined` | _STABILITY_HALF_LIFE has all three keys |

## Risks

| Risk | Mitigation |
|------|-----------|
| LLM misclassifies source_role | Default to "user" (genuine) — conservative. Misclassifying user as assistant only delays last_confirmed, doesn't corrupt data |
| TTL too aggressive (useful facts expire) | TTL only set by LLM extraction; most facts have no TTL. TTL uses last_confirmed so re-observed facts reset the clock |
| Stability half-life values wrong | Start with 365/90/14. These are constants in scoring.py, easy to tune. The research recommends these as starting points |
| Migration fails on existing databases | ALTER TABLE ADD COLUMN is metadata-only in SQLite. Catch OperationalError for idempotency |

## Open decisions

1. **Should half-lives be configurable via MemoryConfig?** Starting as constants.
   Promote to config if tuning is needed after observing retrieval quality.
2. **Should `last_confirmed` apply to profile beliefs too?** Profile beliefs already
   have `last_seen_at` which serves a similar purpose. Keeping them separate for now —
   aligning could be a future iteration.
