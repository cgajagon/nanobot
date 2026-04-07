# Memory Performance Optimizations Design

> Date: 2026-04-07
> Status: Approved
> Scope: 3 independent, low-risk performance optimizations to the memory subsystem

## Problem Statement

Profiling the memory subsystem identified three performance bottlenecks:

1. **Read path latency:** The embedding API call (200-500ms) blocks FTS5 search from
   starting, even though FTS5 doesn't need the embedding vector.
2. **Write path redundancy:** The dedup pipeline in `append_events()` calls
   `_find_dedup_candidates()` twice with identical parameters — once for supersession
   (step 2), once for duplicate detection (step 3).
3. **LLM cost waste:** Micro-extraction calls the LLM for every turn, including trivial
   messages ("ok", "thanks", "yes") that never produce memory events.

## Optimization 1: Parallelize Embedding + FTS5 Search

**File:** `nanobot/memory/read/retriever.py`, method `_retrieve_unified()`

### Current Flow

```
embed(query) → [wait 200-500ms] → gather(search_vector, search_fts) → fuse
```

### New Flow

```
gather(embed(query), search_fts(query)) → search_vector(query_vec) → fuse
```

### Change

Replace the sequential `embed()` + two-way `asyncio.gather()` with a two-way
`gather(embed, search_fts)`, then run `search_vector` with the embedding result:

```python
# Before (lines 155-162):
query_vec = await self._embedder.embed(query)
vec_results, fts_results = await asyncio.gather(
    asyncio.to_thread(self._db.search_vector, query_vec, candidate_k),
    asyncio.to_thread(self._db.search_fts, query, candidate_k),
)

# After:
query_vec, fts_results = await asyncio.gather(
    self._embedder.embed(query),
    asyncio.to_thread(self._db.search_fts, query, candidate_k),
)
vec_results = await asyncio.to_thread(
    self._db.search_vector, query_vec, candidate_k
)
```

### Savings

~200-300ms per query. FTS latency (previously sequential) is hidden behind the
embedding API call. Vector search adds ~5ms sequentially after embed completes.

### Test Plan

- Existing `TestUnifiedRetrievePath` tests pass unchanged (mock embed as async,
  search_fts/search_vector as sync; same assertions).
- New test: verify FTS is called concurrently with embed by mocking embed with
  `asyncio.sleep(0.1)` and asserting FTS completes before embed returns.

## Optimization 2: Cache FTS Dedup Candidates

**File:** `nanobot/memory/write/ingester.py`, method `append_events()`

### Current Flow

For each semantic event:

```
Step 2: fts_candidates = _find_dedup_candidates(candidate)  # FTS query
        → check supersession
Step 3: fts_candidates = _find_dedup_candidates(candidate)  # SAME FTS query
        → check duplicate
```

### New Flow

```
fts_candidates = _find_dedup_candidates(candidate)  # single FTS query
Step 2: check supersession against fts_candidates
Step 3: reuse fts_candidates for duplicate check
```

### Change

Hoist the `_find_dedup_candidates()` call above the supersession check. Use the result
for both steps. Non-semantic events that skip step 2 still call
`_find_dedup_candidates()` in step 3 as before.

```python
fts_candidates: list[dict[str, Any]] = []
fts_vectors: dict[str, list[float]] = {}
is_semantic = memory_type_for_item(candidate) == "semantic"

if is_semantic:
    fts_candidates, fts_vectors = self._find_dedup_candidates(candidate, limit=30)

# Step 2: Supersession (semantic only, reuses fts_candidates)
if is_semantic:
    semantic_candidates = [
        c for c in fts_candidates
        if memory_type_for_item(c) == "semantic"
        and str(c.get("status", "")).lower() != "superseded"
    ]
    # ... supersession logic ...

# Step 3: Duplicate (reuses fts_candidates if already fetched)
if not supersession_found:
    if not fts_candidates:
        fts_candidates, fts_vectors = self._find_dedup_candidates(candidate, limit=30)
    # ... duplicate logic ...
```

### Savings

Halves FTS5 DB queries + unpacking work for every semantic event written.

### Test Plan

- Existing `test_ingester.py` tests pass unchanged (test via public `append_events()`).
- New test: mock `_find_dedup_candidates` and verify `call_count == 1` for a semantic
  event that goes through both supersession and duplicate checks.

## Optimization 3: Trivial-Turn Skip for Micro-Extraction

**File:** `nanobot/memory/write/micro_extractor.py`, method `submit()`

### Change

Add a pre-filter in `submit()` that skips trivial user messages before creating the
async task.

#### Module-level constant

```python
_TRIVIAL_PATTERNS: frozenset[str] = frozenset({
    "ok", "okay", "yes", "no", "yep", "nope", "sure", "thanks",
    "thank you", "ty", "thx", "got it", "sounds good", "perfect",
    "great", "good", "nice", "cool", "right", "agreed", "exactly",
    "correct", "understood", "k", "kk", "yea", "yeah", "nah",
    "fine", "done", "next", "continue", "go ahead", "proceed",
    "lgtm", "👍", "👎", "✅", "❌",
})

_TRIVIAL_MAX_LEN: int = 20
```

#### Pre-filter logic

```python
async def submit(self, user_message, assistant_message, **kwargs):
    if not self._enabled:
        return
    stripped = user_message.strip()
    if len(stripped) <= _TRIVIAL_MAX_LEN and stripped.lower().rstrip("!.,?") in _TRIVIAL_PATTERNS:
        logger.debug("Micro-extraction: skipped trivial turn ({!r})", stripped[:30])
        return
    # ... create_task as before ...
```

### Design Decisions

- `rstrip("!.,?")` handles "Thanks!", "Ok.", "Yes?" etc.
- Length check first (O(1)) before normalization to short-circuit
- Conservative threshold: only skips exact pattern matches at ≤20 chars.
  Multi-word messages beyond the pattern set always pass through (e.g.,
  "no, use Python 3.12" is 21 chars → exceeds threshold → not skipped).
- `frozenset` for O(1) lookup
- Debug log for observability

### Savings

~30-40% fewer micro-extraction LLM calls. Each saved call avoids ~$0.0001 cost +
~500-1000ms background API latency.

### Test Plan

- Trivial messages skipped: "ok", "Thanks!", "yes?", "👍"
- Non-trivial messages pass through: "no, use Python 3.12", "the vault is at C:\..."
- Edge cases: punctuation stripping, mixed case, empty string, just-over-threshold
  length, multi-word patterns ("sounds good", "go ahead")
- Verify no async task is created for skipped turns

## Cross-Cutting Concerns

### Independence

All three optimizations are independent. Each touches exactly one file. They can be
implemented and tested in any order. No cross-component data contracts change.

### Observability

- Optimization 1: Existing Langfuse `retriever_span` captures `duration_ms` — the
  improvement will be visible in trace latency metrics.
- Optimization 2: Existing `bind_trace().debug("memory_append | ...")` log line
  reports timing — improvement visible in debug logs.
- Optimization 3: New `logger.debug` line logs skipped turns. Langfuse micro-extraction
  span count will decrease (fewer LLM calls).

### Rollback

Each optimization is a self-contained code change. Reverting any one has no impact on
the others.
