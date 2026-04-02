# Write-Path Embeddings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the embedder into the memory write path so events are vector-searchable immediately after ingestion, not just via FTS.

**Architecture:** Compute embeddings at the async caller level, pass pre-computed vectors into the sync ingester via a new `embeddings` parameter. Three callers (ConsolidationPipeline, MicroExtractor, CorrectionOrchestrator) each get the embedder injected and compute vectors before calling `append_events()`.

**Tech Stack:** Python 3.10+, pytest, sqlite-vec, async embedders (OpenAI/Local/Hash)

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `nanobot/memory/write/ingester.py` | Modify | Add `embeddings` param to `append_events()` and `_write_events()` |
| `nanobot/memory/consolidation_pipeline.py` | Modify | Add `embedder` to constructor, embed before `append_events()` |
| `nanobot/memory/write/micro_extractor.py` | Modify | Add `embedder` to constructor, embed before `append_events()` |
| `nanobot/memory/persistence/profile_correction.py` | Modify | Add `embedder` to constructor, embed before `append_events()` |
| `nanobot/memory/store.py` | Modify | Pass `_embedder` to ConsolidationPipeline and CorrectionOrchestrator |
| `nanobot/agent/agent_factory.py` | Modify | Pass `memory._embedder` to MicroExtractor |
| `tests/contract/test_memory_data_contracts.py` | Modify | Add contract test for write-path embedding |
| `tests/test_ingester.py` | Modify | Add unit test for embeddings parameter |

---

### Task 1: Add `embeddings` parameter to EventIngester

**Files:**
- Modify: `nanobot/memory/write/ingester.py:77,229`
- Test: `tests/test_ingester.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_ingester.py`:

```python
def test_append_events_passes_embeddings_to_write(tmp_path: Path) -> None:
    """Embeddings dict is forwarded to _write_events and used in insert_event."""
    from unittest.mock import MagicMock
    from nanobot.memory.write.ingester import EventIngester
    from nanobot.memory.event import MemoryEvent

    db = MagicMock()
    db.insert_event = MagicMock()
    ingester = EventIngester(coercer=MagicMock(), dedup=MagicMock(), graph=None, db=db)
    # Bypass dedup — make _find_dedup_candidates return no matches
    ingester._find_dedup_candidates = MagicMock(return_value=(None, []))

    event = MemoryEvent(
        id="test-embed-001",
        type="fact",
        summary="User likes Python",
        timestamp="2026-04-01T00:00:00+00:00",
    )
    vec = [0.1] * 384
    embeddings = {"test-embed-001": vec}

    ingester.append_events([event], embeddings=embeddings)

    # Verify insert_event was called with the embedding vector
    calls = db.insert_event.call_args_list
    assert len(calls) >= 1
    _, kwargs = calls[0]
    assert kwargs.get("embedding") == vec
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingester.py::test_append_events_passes_embeddings_to_write -v`
Expected: FAIL — `append_events()` doesn't accept `embeddings` parameter.

- [ ] **Step 3: Add `embeddings` parameter to `append_events()`**

In `nanobot/memory/write/ingester.py`, modify `append_events()` signature (line 77):

FROM:
```python
def append_events(self, events: Sequence[MemoryEvent]) -> int:
```

TO:
```python
def append_events(
    self,
    events: Sequence[MemoryEvent],
    embeddings: dict[str, list[float]] | None = None,
) -> int:
```

Find where `_write_events` is called inside `append_events()` (there should be one or more calls) and pass `embeddings` through. Look for `self._write_events(...)` calls and add `embeddings=embeddings`.

- [ ] **Step 4: Add `embeddings` parameter to `_write_events()`**

Modify `_write_events()` signature (line 229):

FROM:
```python
def _write_events(self, events: list[dict[str, Any]]) -> None:
```

TO:
```python
def _write_events(
    self,
    events: list[dict[str, Any]],
    embeddings: dict[str, list[float]] | None = None,
) -> None:
```

Change line 254:

FROM:
```python
self._db.insert_event(evt_copy, embedding=None)
```

TO:
```python
vec = embeddings.get(evt_copy["id"]) if embeddings else None
self._db.insert_event(evt_copy, embedding=vec)
```

- [ ] **Step 5: Run the test**

Run: `pytest tests/test_ingester.py::test_append_events_passes_embeddings_to_write -v`
Expected: PASS

- [ ] **Step 6: Run `make lint && make typecheck`**

- [ ] **Step 7: Commit**

```
feat(memory): add embeddings parameter to EventIngester write path
```

---

### Task 2: Wire embedder into ConsolidationPipeline

**Files:**
- Modify: `nanobot/memory/consolidation_pipeline.py:48-65,221`
- Modify: `nanobot/memory/store.py:246-253`

- [ ] **Step 1: Add `embedder` to ConsolidationPipeline constructor**

In `nanobot/memory/consolidation_pipeline.py`, add to `__init__` params (after `db`):

```python
embedder: Embedder | None = None,
```

Store as `self._embedder = embedder`.

Add TYPE_CHECKING import at top of file:

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nanobot.memory.embedder import Embedder
```

- [ ] **Step 2: Compute embeddings before `append_events()` call**

Find the `append_events()` call (line ~221). Replace:

```python
events_written = self._ingester.append_events(events)
```

With:

```python
embeddings = await self._compute_embeddings(events)
events_written = self._ingester.append_events(events, embeddings=embeddings)
```

Add the helper method to the class:

```python
async def _compute_embeddings(
    self, events: list[MemoryEvent],
) -> dict[str, list[float]] | None:
    """Compute embedding vectors for events. Returns None on failure."""
    if not self._embedder or not self._embedder.available or not events:
        return None
    try:
        summaries = [e.summary for e in events]
        ids = [e.id for e in events if e.id]
        if not ids:
            return None
        vectors = await self._embedder.embed_batch(summaries)
        return dict(zip(ids, vectors))
    except Exception:  # crash-barrier: embedding failure must not block ingestion
        logger.warning("Embedding failed during consolidation, using FTS-only")
        return None
```

- [ ] **Step 3: Wire embedder in `store.py`**

In `nanobot/memory/store.py`, update the ConsolidationPipeline instantiation (lines 246-253). Add `embedder=self._embedder`:

```python
self._consolidation = ConsolidationPipeline(
    extractor=self.extractor,
    ingester=self.ingester,
    profile_mgr=self.profile_mgr,
    conflict_mgr=self.conflict_mgr,
    snapshot=self.snapshot,
    db=self.db,
    embedder=self._embedder,
)
```

- [ ] **Step 4: Run `make lint && make typecheck`**

- [ ] **Step 5: Commit**

```
feat(memory): embed events during consolidation write path
```

---

### Task 3: Wire embedder into MicroExtractor

**Files:**
- Modify: `nanobot/memory/write/micro_extractor.py:94-106,171`
- Modify: `nanobot/agent/agent_factory.py:346-356`

- [ ] **Step 1: Add `embedder` to MicroExtractor constructor**

In `nanobot/memory/write/micro_extractor.py`, add to `__init__` params:

```python
embedder: Embedder | None = None,
```

Store as `self._embedder = embedder`.

Add TYPE_CHECKING import:

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nanobot.memory.embedder import Embedder
```

- [ ] **Step 2: Compute embeddings before `append_events()` call**

Find the `append_events()` call (line ~171). Replace:

```python
self._ingester.append_events(events)
```

With:

```python
embeddings = await self._compute_embeddings(events)
self._ingester.append_events(events, embeddings=embeddings)
```

Add the same `_compute_embeddings` helper as in Task 2 (same pattern, different log message: "Embedding failed during micro-extraction").

- [ ] **Step 3: Wire embedder in `agent_factory.py`**

In `nanobot/agent/agent_factory.py`, update MicroExtractor instantiation (lines 346-356). Add `embedder=memory._embedder`:

```python
_micro_extractor = _MicroExtractor(
    provider=provider,
    ingester=memory.ingester,
    model=config.memory.micro_extraction_model or "gpt-4o-mini",
    enabled=True,
    embedder=memory._embedder,
)
```

- [ ] **Step 4: Run `make lint && make typecheck`**

- [ ] **Step 5: Commit**

```
feat(memory): embed events during micro-extraction write path
```

---

### Task 4: Wire embedder into CorrectionOrchestrator

**Files:**
- Modify: `nanobot/memory/persistence/profile_correction.py:31-46,197`
- Modify: `nanobot/memory/store.py:235-242`

- [ ] **Step 1: Add `embedder` to CorrectionOrchestrator constructor**

In `nanobot/memory/persistence/profile_correction.py`, add to `__init__` params:

```python
embedder: Embedder | None = None,
```

Store as `self._embedder = embedder`.

Add TYPE_CHECKING import:

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nanobot.memory.embedder import Embedder
```

- [ ] **Step 2: Compute embeddings before `append_events()` call**

Find the `append_events()` call (line ~197). Replace:

```python
events_written = self._ingester.append_events(events)
```

With:

```python
embeddings = await self._compute_embeddings(events)
events_written = self._ingester.append_events(events, embeddings=embeddings)
```

Add the same helper. But wait — check if `apply_live_user_correction` is async. If it's sync, we need a different approach. Read the method signature first. If sync, use `asyncio.get_event_loop().run_until_complete()` — NO, that blocks. If the method is sync, skip embedding here (pass None) rather than blocking. Check and decide.

- [ ] **Step 3: Wire embedder in `store.py`**

Update CorrectionOrchestrator instantiation (lines 235-242). Add `embedder=self._embedder`.

- [ ] **Step 4: Run `make lint && make typecheck`**

- [ ] **Step 5: Commit**

```
feat(memory): embed events during profile correction write path
```

---

### Task 5: Contract test — vector-searchable after write

**Files:**
- Test: `tests/contract/test_memory_data_contracts.py`

- [ ] **Step 1: Write contract test**

Add a test that verifies the full round-trip: write an event with embedding → vector search finds it.

```python
async def test_event_with_embedding_is_vector_searchable(tmp_path: Path) -> None:
    """Events written with embeddings should be retrievable via vector search."""
    from nanobot.memory.db.event_store import EventStore
    from nanobot.memory.embedder import HashEmbedder

    embedder = HashEmbedder(dims=384)
    db = EventStore(tmp_path / "test.db", dims=384)

    # Write event with embedding
    text = "User prefers dark mode for all applications"
    vec = await embedder.embed(text)
    db.insert_event(
        {"id": "vec-test-001", "type": "preference", "summary": text,
         "timestamp": "2026-04-01T00:00:00+00:00", "status": "active",
         "metadata": None, "created_at": "2026-04-01T00:00:00+00:00"},
        embedding=vec,
    )

    # Vector search should find it
    query_vec = await embedder.embed("dark mode preference")
    results = db.search_vector(query_vec, k=5)
    assert len(results) >= 1
    assert any(r["id"] == "vec-test-001" for r in results)
```

- [ ] **Step 2: Run test**

Run: `pytest tests/contract/test_memory_data_contracts.py::test_event_with_embedding_is_vector_searchable -v`
Expected: PASS (this tests EventStore directly, which already supports embeddings)

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ --ignore=tests/integration -q`
Expected: All pass, no regressions.

- [ ] **Step 4: Run `make check`**

- [ ] **Step 5: Commit**

```
test(memory): add contract test for write-path embedding round-trip
```

---

### Task 6: Final verification

- [ ] **Step 1: Run `make check`**

- [ ] **Step 2: Run full test suite**

- [ ] **Step 3: Verify the complete write→read path conceptually**

Trace the flow: ConsolidationPipeline calls `embed_batch(summaries)` → gets vectors → passes to `append_events(events, embeddings={id: vec})` → `_write_events()` looks up each event's vector → `insert_event(event, embedding=vec)` → SQLite `events_vec` table populated → `MemoryRetriever._retrieve_unified()` calls `search_vector(query_vec)` → finds the event.
