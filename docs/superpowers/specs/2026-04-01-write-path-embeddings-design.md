# Write-Path Embeddings — Design Spec

> Date: 2026-04-01
> Status: Approved
> Scope: Wire embedder into the memory write path so events are vector-searchable at insert time

## Problem

`EventIngester._write_events()` always passes `embedding=None` to `insert_event()`.
The `events_vec` table is never populated during ingestion. Vector search (sqlite-vec KNN)
only works for events that were retroactively embedded — which currently never happens.
The entire vector retrieval arm of the dual-source RRF pipeline is dead on write.

## Design

### Approach: Embed at the async boundary, pass vectors into the sync ingester

The ingester (`append_events`, `_write_events`) is synchronous. The embedder protocol
is async (`embed_batch`). Rather than making the ingester async (cascading refactor),
compute embeddings at the caller level (which is already async) and pass pre-computed
vectors into the ingester.

### Changes

**1. `EventIngester.append_events()` — accept optional embeddings**

Add parameter `embeddings: dict[str, list[float]] | None = None` — a mapping from
event ID to pre-computed embedding vector. Pass through to `_write_events()`.

**2. `EventIngester._write_events()` — accept and use embeddings**

Add parameter `embeddings: dict[str, list[float]] | None = None`. When writing each
event, look up its ID in the dict and pass the vector to `insert_event()`:

```python
vec = embeddings.get(event["id"]) if embeddings else None
self._db.insert_event(evt_copy, embedding=vec)
```

**3. Callers — compute embeddings before calling `append_events()`**

Three callers, all in async contexts:

- `ConsolidationPipeline._consolidate_single_tool()` — receives embedder via constructor
- `MicroExtractor.submit()` — receives embedder via constructor
- `CorrectionOrchestrator.apply_live_user_correction()` — receives embedder via constructor

Each caller:
```python
embeddings = None
if self._embedder and self._embedder.available:
    summaries = [e.summary for e in events]
    ids = [e.id for e in events if e.id]
    try:
        vectors = await self._embedder.embed_batch(summaries)
        embeddings = dict(zip(ids, vectors))
    except Exception:  # crash-barrier: embedding failure must not block ingestion
        logger.warning("Embedding failed, events will use FTS-only retrieval")
self._ingester.append_events(events, embeddings=embeddings)
```

**4. Wiring — inject embedder into callers**

In `MemoryStore.__init__()`, pass `self._embedder` to `ConsolidationPipeline`,
`MicroExtractor` (already has its own provider — check if embedder is separate),
and `CorrectionOrchestrator`.

### Graceful degradation

If `embed_batch()` fails (network error, model unavailable), the caller catches the
exception and passes `embeddings=None`. The event is written without a vector — FTS
retrieval still works. This matches the existing RRF fusion design: vector_weight=0.7,
fts_weight=0.3, but FTS alone still returns results.

### What does NOT change

- `EventStore.insert_event()` — already handles optional embeddings correctly
- `events_vec` table schema — already created with correct dimensions
- Read path (`MemoryRetriever`) — already does vector search when vectors exist
- `Embedder` protocol — no changes needed
- `append_events()` stays synchronous — async embedding happens at the caller

### Files changed

| File | Change |
|------|--------|
| `nanobot/memory/write/ingester.py` | Add `embeddings` param to `append_events()` and `_write_events()` |
| `nanobot/memory/consolidation_pipeline.py` | Add embedder to constructor, compute embeddings before `append_events()` |
| `nanobot/memory/write/micro_extractor.py` | Add embedder to constructor, compute embeddings before `append_events()` |
| `nanobot/memory/persistence/profile_correction.py` | Add embedder to constructor, compute embeddings before `append_events()` |
| `nanobot/memory/store.py` | Pass `_embedder` to the three callers during construction |
| Tests | Contract test: event written via `append_events` with embeddings is vector-searchable |
