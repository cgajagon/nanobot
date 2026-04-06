# Dedup Embedding Data Flow Fix

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the 773-second `append_events` bottleneck by making embeddings flow as data through the dedup pipeline instead of being recomputed per candidate pair, and prevent event-loop blocking in async callers.

**Architecture:** The dedup pipeline currently calls `_sync_embed()` (OpenAI API) for every candidate pair comparison. The fix routes pre-computed embeddings (from callers and from `events_vec` storage) through the comparison functions, removes the embedder dependency from `EventDeduplicator`, and wraps sync `append_events` calls in `asyncio.to_thread()`.

**Tech Stack:** Python 3.10+, sqlite3, sqlite-vec, pytest, asyncio

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `nanobot/memory/db/event_store.py` | Modify | Add `search_fts_with_vectors()` method |
| `nanobot/memory/write/dedup.py` | Modify | Accept pre-computed vectors in similarity/dedup/supersession methods; remove `embedder` dependency and `_sync_embed` |
| `nanobot/memory/write/ingester.py` | Modify | Wire embedding flow: fetch stored vectors from FTS, pass caller vectors + stored vectors to dedup |
| `nanobot/memory/store.py` | Modify | Remove `embedder=` arg from `EventDeduplicator` construction |
| `nanobot/memory/write/micro_extractor.py` | Modify | Wrap `append_events` in `asyncio.to_thread()` |
| `nanobot/memory/consolidation_pipeline.py` | Modify | Wrap `append_events` in `asyncio.to_thread()` |
| `tests/test_event_deduplicator.py` | Modify | Update tests for new signatures, remove embedder tests, add vector-injection tests |
| `tests/test_ingester.py` | Modify | Update `_find_dedup_candidates` mock expectations, add embedding flow test |
| `frontend/src/lib/thread-list-adapter.ts` | Modify | Revert debug console.log statements |

---

### Task 1: Add `search_fts_with_vectors()` to EventStore

**Files:**
- Modify: `nanobot/memory/db/event_store.py:129-157`
- Test: `tests/test_ingester.py` (new test class)

- [ ] **Step 1: Write the failing test**

In `tests/test_ingester.py`, add a new test class at the end of the file:

```python
class TestSearchFtsWithVectors:
    """Tests for EventStore.search_fts_with_vectors."""

    def test_returns_events_and_vectors(self, tmp_path: Path) -> None:
        import struct

        db = MemoryDatabase(tmp_path / "memory.db", dims=4)
        # Insert event with embedding
        db.event_store.insert_event(
            {
                "id": "vec-001",
                "type": "fact",
                "summary": "User likes Python programming",
                "timestamp": "2026-01-01T00:00:00Z",
            },
            embedding=[0.1, 0.2, 0.3, 0.4],
        )
        # Insert event without embedding
        db.event_store.insert_event(
            {
                "id": "vec-002",
                "type": "fact",
                "summary": "User likes Python scripting",
                "timestamp": "2026-01-01T00:00:00Z",
            },
        )

        events, vectors = db.event_store.search_fts_with_vectors("Python", k=10)
        assert len(events) == 2
        assert "vec-001" in vectors
        assert "vec-002" not in vectors
        vec = vectors["vec-001"]
        assert len(vec) == 4
        assert abs(vec[0] - 0.1) < 0.01
        db.close()

    def test_returns_empty_on_no_match(self, tmp_path: Path) -> None:
        db = MemoryDatabase(tmp_path / "memory.db", dims=4)
        db.event_store.insert_event(
            {
                "id": "vec-003",
                "type": "fact",
                "summary": "User likes Python",
                "timestamp": "2026-01-01T00:00:00Z",
            },
        )
        events, vectors = db.event_store.search_fts_with_vectors("nonexistent", k=10)
        assert events == []
        assert vectors == {}
        db.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ingester.py::TestSearchFtsWithVectors -v`
Expected: FAIL with `AttributeError: 'EventStore' object has no attribute 'search_fts_with_vectors'`

- [ ] **Step 3: Implement `search_fts_with_vectors`**

In `nanobot/memory/db/event_store.py`, add this method after `search_fts` (after line 157):

```python
def search_fts_with_vectors(
    self, query_text: str, k: int = 10
) -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
    """FTS5 search returning events and their stored embedding vectors.

    Returns a tuple of (events, vectors_dict) where vectors_dict maps
    event ID -> float vector for events that have stored embeddings.
    Events without embeddings are still returned but absent from the dict.
    """
    import struct

    terms = re.findall(r"\w+", query_text)
    if not terms:
        return [], {}
    safe_query = " OR ".join(f"{t}*" for t in terms)
    try:
        rows = self._conn.execute(
            """SELECT e.*, v.embedding
               FROM events_fts fts
               JOIN events e ON e.rowid = fts.rowid
               LEFT JOIN events_vec v ON e.rowid = v.id
               WHERE events_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (safe_query, k),
        ).fetchall()
    except sqlite3.OperationalError:
        return [], {}

    events: list[dict[str, Any]] = []
    vectors: dict[str, list[float]] = {}
    for row in rows:
        event = dict(row)
        raw_vec = event.pop("embedding", None)
        events.append(event)
        if raw_vec is not None:
            n_floats = len(raw_vec) // 4
            vectors[event["id"]] = list(struct.unpack(f"{n_floats}f", raw_vec))
    return events, vectors
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ingester.py::TestSearchFtsWithVectors -v`
Expected: PASS

- [ ] **Step 5: Run lint and typecheck**

Run: `make lint && make typecheck`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add nanobot/memory/db/event_store.py tests/test_ingester.py
git commit -m "feat(memory): add search_fts_with_vectors for embedding-aware FTS"
```

---

### Task 2: Refactor `EventDeduplicator` to accept pre-computed vectors

**Files:**
- Modify: `nanobot/memory/write/dedup.py`
- Modify: `tests/test_event_deduplicator.py`

- [ ] **Step 1: Write the failing test for vector injection**

Add to `tests/test_event_deduplicator.py`:

```python
class TestVectorInjection:
    def test_event_similarity_uses_provided_vectors(self) -> None:
        """When both vectors are provided, semantic should be cosine, not lexical."""
        d = _make_dedup()
        a = {"type": "fact", "summary": "User likes Python"}
        b = {"type": "fact", "summary": "Completely different topic about weather"}
        # Provide identical vectors -> cosine = 1.0
        vec = [1.0, 0.0, 0.0]
        lexical, semantic = d.event_similarity(a, b, left_vec=vec, right_vec=vec)
        assert lexical < 0.3  # texts are different
        assert semantic == 1.0  # vectors are identical

    def test_event_similarity_partial_vectors_falls_back(self) -> None:
        """When only one vector provided, semantic = lexical (no embedding call)."""
        d = _make_dedup()
        a = {"type": "fact", "summary": "User likes Python"}
        b = {"type": "fact", "summary": "User likes Python"}
        lexical, semantic = d.event_similarity(a, b, left_vec=[1.0, 0.0])
        assert semantic == lexical

    def test_find_duplicate_with_vectors(self) -> None:
        """find_semantic_duplicate should thread vectors to event_similarity."""
        d = _make_dedup()
        existing = [
            {"id": "e1", "type": "fact", "summary": "User likes Python", "entities": ["Python"]},
        ]
        candidate = {
            "id": "c1",
            "type": "fact",
            "summary": "User likes Python",
            "entities": ["Python"],
        }
        # Provide vectors that are identical -> high semantic score
        candidate_vec = [1.0, 0.0, 0.0]
        existing_vecs = {"e1": [1.0, 0.0, 0.0]}
        idx, score = d.find_semantic_duplicate(
            candidate, existing, candidate_vec=candidate_vec, existing_vecs=existing_vecs
        )
        assert idx == 0

    def test_find_supersession_with_vectors(self) -> None:
        """find_semantic_supersession should thread vectors to event_similarity."""
        d = _make_dedup()
        existing = [
            {
                "id": "e1",
                "type": "preference",
                "summary": "User likes dark mode",
                "entities": ["dark mode"],
                "memory_type": "semantic",
            }
        ]
        candidate = {
            "id": "c1",
            "type": "preference",
            "summary": "User does not like dark mode",
            "entities": ["dark mode"],
            "memory_type": "semantic",
        }
        candidate_vec = [1.0, 0.0, 0.0]
        existing_vecs = {"e1": [0.9, 0.1, 0.0]}
        idx = d.find_semantic_supersession(
            candidate, existing, candidate_vec=candidate_vec, existing_vecs=existing_vecs
        )
        assert idx == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_event_deduplicator.py::TestVectorInjection -v`
Expected: FAIL with `TypeError: event_similarity() got an unexpected keyword argument 'left_vec'`

- [ ] **Step 3: Refactor `event_similarity` to accept vectors**

In `nanobot/memory/write/dedup.py`, replace the `event_similarity` method (lines 66-102):

```python
def event_similarity(
    self,
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    left_vec: list[float] | None = None,
    right_vec: list[float] | None = None,
) -> tuple[float, float]:
    """Compute similarity between two events (lexical, semantic).

    When both ``left_vec`` and ``right_vec`` are provided, semantic similarity
    is computed as cosine distance between the vectors.  Otherwise semantic
    falls back to lexical (no embedding calls — dedup does not own an embedder).
    """

    def _event_text(event: dict[str, Any]) -> str:
        summary = str(event.get("summary", ""))
        raw_entities = _to_str_list(event.get("entities"))
        entities = " ".join(normalize_entity_name(e) for e in raw_entities)
        event_type = str(event.get("type", "fact"))
        return f"{event_type}. {summary}. {entities}".strip()

    left_text = _event_text(left)
    right_text = _event_text(right)

    left_tokens = _tokenize(left_text)
    right_tokens = _tokenize(right_text)

    # Normalize aliases to canonical tokens via registry
    if self._alias_registry:
        left_tokens = {self._alias_registry.resolve(t) for t in left_tokens}
        right_tokens = {self._alias_registry.resolve(t) for t in right_tokens}

    overlap = left_tokens & right_tokens
    union = left_tokens | right_tokens
    lexical = (len(overlap) / len(union)) if union else 0.0

    # Use pre-computed vectors when available; otherwise semantic = lexical
    semantic = lexical
    if left_vec is not None and right_vec is not None:
        dot = sum(a * b for a, b in zip(left_vec, right_vec))
        norm_l = sum(a * a for a in left_vec) ** 0.5
        norm_r = sum(b * b for b in right_vec) ** 0.5
        if norm_l > 0 and norm_r > 0:
            semantic = dot / (norm_l * norm_r)

    return lexical, semantic
```

- [ ] **Step 4: Refactor `find_semantic_duplicate` to accept and thread vectors**

Replace `find_semantic_duplicate` (lines 104-149):

```python
def find_semantic_duplicate(
    self,
    candidate: dict[str, Any],
    existing_events: list[dict[str, Any]],
    *,
    candidate_vec: list[float] | None = None,
    existing_vecs: dict[str, list[float]] | None = None,
) -> tuple[int | None, float]:
    """Find an existing event that is a semantic duplicate of *candidate*."""
    best_idx: int | None = None
    best_score = 0.0
    candidate_type = str(candidate.get("type", ""))

    for idx, existing in enumerate(existing_events):
        if str(existing.get("type", "")) != candidate_type:
            continue
        existing_id = str(existing.get("id", ""))
        existing_vec = existing_vecs.get(existing_id) if existing_vecs else None
        lexical, semantic = self.event_similarity(
            candidate, existing, left_vec=candidate_vec, right_vec=existing_vec
        )
        candidate_entities = {
            normalize_entity_name(x) for x in _to_str_list(candidate.get("entities"))
        }
        existing_entities = {
            normalize_entity_name(x) for x in _to_str_list(existing.get("entities"))
        }
        entity_overlap = 0.0
        if candidate_entities and existing_entities:
            entity_overlap = len(candidate_entities & existing_entities) / max(
                len(candidate_entities | existing_entities), 1
            )

        score = 0.4 * semantic + 0.45 * lexical + 0.15 * entity_overlap
        is_duplicate = (
            lexical >= 0.84
            or semantic >= 0.94
            or (lexical >= 0.6 and semantic >= 0.86)
            or (entity_overlap >= 0.33 and (lexical >= 0.42 or semantic >= 0.52))
            or (
                entity_overlap >= 0.30
                and lexical >= 0.25
                and candidate_type == str(existing.get("type", ""))
            )
            or (lexical >= 0.70 and candidate_type == str(existing.get("type", "")))
        )
        if not is_duplicate:
            continue
        if score > best_score:
            best_score = score
            best_idx = idx

    return best_idx, best_score
```

- [ ] **Step 5: Refactor `find_semantic_supersession` to accept and thread vectors**

Replace `find_semantic_supersession` (lines 151-204):

```python
def find_semantic_supersession(
    self,
    candidate: dict[str, Any],
    existing_events: list[dict[str, Any]],
    *,
    candidate_vec: list[float] | None = None,
    existing_vecs: dict[str, list[float]] | None = None,
) -> int | None:
    """Find an existing event that the *candidate* supersedes (contradicts)."""
    if memory_type_for_item(candidate) != "semantic":
        return None
    candidate_summary = str(candidate.get("summary", "")).strip()
    candidate_type = str(candidate.get("type", ""))
    if not candidate_summary:
        return None

    for idx, existing in enumerate(existing_events):
        if memory_type_for_item(existing) != "semantic":
            continue
        if str(existing.get("type", "")) != candidate_type:
            continue
        if str(existing.get("status", "")).strip().lower() == "superseded":
            continue

        existing_summary = str(existing.get("summary", "")).strip()
        if not existing_summary:
            continue
        has_conflict = (
            self._conflict_pair_fn(existing_summary, candidate_summary)
            if self._conflict_pair_fn
            else False
        )
        if not has_conflict:
            existing_norm = _norm_text(existing_summary)
            candidate_norm = _norm_text(candidate_summary)
            existing_not = " not " in f" {existing_norm} " or "n't" in existing_norm
            candidate_not = " not " in f" {candidate_norm} " or "n't" in candidate_norm
            if existing_not != candidate_not:
                stop = {"do", "does", "did"}
                left_tokens = {
                    t for t in _tokenize(existing_norm.replace("not", "")) if t not in stop
                }
                right_tokens = {
                    t for t in _tokenize(candidate_norm.replace("not", "")) if t not in stop
                }
                if left_tokens and right_tokens:
                    overlap = len(left_tokens & right_tokens) / max(
                        len(left_tokens | right_tokens), 1
                    )
                    has_conflict = overlap >= 0.45
        if not has_conflict:
            continue

        existing_id = str(existing.get("id", ""))
        existing_vec = existing_vecs.get(existing_id) if existing_vecs else None
        lexical, semantic = self.event_similarity(
            candidate, existing, left_vec=candidate_vec, right_vec=existing_vec
        )
        if lexical >= 0.35 or semantic >= 0.35:
            return idx
    return None
```

- [ ] **Step 6: Remove `_sync_embed` method and `embedder` from constructor**

In `nanobot/memory/write/dedup.py`:

Remove the `embedder` parameter from `__init__` and the `_sync_embed` method entirely.  Remove the `self._embedder` attribute. Remove the `Embedder` import from `TYPE_CHECKING`.

New `__init__`:

```python
def __init__(
    self,
    coercer: EventCoercer,
    conflict_pair_fn: Callable[[str, str], bool] | None = None,
    alias_registry: AliasRegistry | None = None,
) -> None:
    self._coercer = coercer
    self._conflict_pair_fn = conflict_pair_fn
    self._alias_registry = alias_registry
```

Remove the `TYPE_CHECKING` import for `Embedder`:

```python
if TYPE_CHECKING:
    from ..db.alias_store import AliasRegistry
    from .coercion import EventCoercer
```

- [ ] **Step 7: Update existing tests for removed embedder**

In `tests/test_event_deduplicator.py`:

Update `_make_dedup` — remove the `embedder` parameter:

```python
def _make_dedup(
    *,
    conflict_pair_fn: object = None,
    alias_registry: object = None,
) -> EventDeduplicator:
    classifier = EventClassifier()
    coercer = EventCoercer(classifier)
    return EventDeduplicator(
        coercer=coercer,
        conflict_pair_fn=conflict_pair_fn,
        alias_registry=alias_registry,
    )
```

Replace `TestEmbeddingSemanticSimilarity` class entirely:

```python
class TestEmbeddingSemanticSimilarity:
    def test_semantic_uses_vectors_when_provided(self) -> None:
        """With vectors, semantic should use cosine, not equal lexical."""
        d = _make_dedup()
        a = {"type": "fact", "summary": "User enjoys programming in Python"}
        b = {"type": "fact", "summary": "Carlos likes coding with Python language"}
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.9, 0.1, 0.0]
        lexical, semantic = d.event_similarity(a, b, left_vec=vec_a, right_vec=vec_b)
        assert semantic != lexical

    def test_semantic_equals_lexical_without_vectors(self) -> None:
        """Without vectors, semantic falls back to lexical."""
        d = _make_dedup()
        a = {"type": "fact", "summary": "User likes Python"}
        b = {"type": "fact", "summary": "User likes Python and Java"}
        lexical, semantic = d.event_similarity(a, b)
        assert semantic == lexical
```

- [ ] **Step 8: Run all dedup tests**

Run: `python -m pytest tests/test_event_deduplicator.py -v`
Expected: ALL PASS

- [ ] **Step 9: Run lint and typecheck**

Run: `make lint && make typecheck`
Expected: PASS

- [ ] **Step 10: Commit**

```bash
git add nanobot/memory/write/dedup.py tests/test_event_deduplicator.py
git commit -m "refactor(memory): remove embedder from dedup, accept pre-computed vectors"
```

---

### Task 3: Wire embedding flow through `EventIngester`

**Files:**
- Modify: `nanobot/memory/write/ingester.py`
- Modify: `tests/test_ingester.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ingester.py`:

```python
class TestDedupEmbeddingFlow:
    """Test that append_events threads embeddings to dedup methods."""

    def test_caller_embeddings_passed_to_dedup(self, tmp_path: Path) -> None:
        """Pre-computed embeddings from caller flow to find_semantic_duplicate."""
        import struct

        db = MemoryDatabase(tmp_path / "memory.db", dims=4)
        # Seed an existing event WITH a stored embedding
        db.event_store.insert_event(
            {
                "id": "existing-001",
                "type": "fact",
                "summary": "User likes Python programming language",
                "timestamp": "2026-01-01T00:00:00Z",
            },
            embedding=[0.5, 0.5, 0.5, 0.5],
        )

        graph = MagicMock()
        graph.enabled = False
        classifier = EventClassifier()
        coercer = EventCoercer(classifier)
        dedup = EventDeduplicator(coercer=coercer)
        ing = EventIngester(coercer=coercer, dedup=dedup, graph=graph, db=db.event_store)

        new_event = MemoryEvent.from_dict(
            {
                "id": "new-001",
                "type": "fact",
                "summary": "User likes Python programming language very much",
                "timestamp": "2026-01-02T00:00:00Z",
            }
        )
        caller_vec = [0.5, 0.5, 0.5, 0.5]
        embeddings = {"new-001": caller_vec}

        # Should work without hanging (no OpenAI calls)
        ing.append_events([new_event], embeddings=embeddings)

        # Event should have been merged (high lexical + stored vectors available)
        events = db.event_store.read_events(limit=100)
        assert len(events) == 1  # merged, not 2
        db.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ingester.py::TestDedupEmbeddingFlow -v`
Expected: FAIL (because `_find_dedup_candidates` doesn't return vectors yet, and dedup methods don't receive them)

- [ ] **Step 3: Refactor `_find_dedup_candidates` to return vectors**

In `nanobot/memory/write/ingester.py`, replace `_find_dedup_candidates` (lines 188-211):

```python
def _find_dedup_candidates(
    self, candidate: dict[str, Any], limit: int = 30
) -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
    """Use FTS5 to find events with overlapping tokens, with stored embeddings.

    Returns (candidates, vectors_dict) where vectors_dict maps event ID to
    stored embedding vector for candidates that have one.
    """
    if self._db is None:
        return [], {}
    summary = str(candidate.get("summary", ""))
    event_type = str(candidate.get("type", ""))
    query_text = f"{event_type} {summary}".strip()
    if not query_text:
        return [], {}
    fts_results, all_vectors = self._db.search_fts_with_vectors(
        query_text, k=limit * 2
    )
    candidates: list[dict[str, Any]] = []
    candidate_vectors: dict[str, list[float]] = {}
    for event in fts_results:
        unpacked = self._unpack_event(event)
        if str(unpacked.get("type", "")) == event_type:
            provenance = self._coercer.ensure_event_provenance(unpacked)
            candidates.append(provenance)
            event_id = str(provenance.get("id", ""))
            if event_id in all_vectors:
                candidate_vectors[event_id] = all_vectors[event_id]
        if len(candidates) >= limit:
            break
    return candidates, candidate_vectors
```

- [ ] **Step 4: Update `append_events` to wire vectors through**

In `nanobot/memory/write/ingester.py`, replace the per-event loop body in `append_events` (lines 100-170):

```python
for raw in raw_events:
    # Generate ID if missing (same logic as before)
    event_id = raw.get("id")
    if not event_id:
        summary = str(raw.get("summary", "")).strip()
        if not summary:
            continue
        event_type = str(raw.get("type", "fact"))
        ts = str(raw.get("timestamp", _utc_now_iso()))
        event_id = self._coercer.build_event_id(event_type, summary, ts)
        raw = {**raw, "id": event_id, "timestamp": ts}
    candidate = self._coercer.ensure_event_provenance(raw)

    # Caller-provided embedding for this event (micro-extraction / consolidation)
    candidate_vec = embeddings.get(event_id) if embeddings else None

    # Step 1: Exact ID dedup -- O(1) PK lookup
    existing_row = self._db.get_event_by_id(event_id)
    if existing_row is not None:
        existing = self._unpack_event(existing_row)
        existing = self._coercer.ensure_event_provenance(existing)
        merged_event = self._dedup.merge_events(existing, candidate, similarity=1.0)
        self._write_events([merged_event], embeddings=embeddings)
        merged += 1
        continue

    # Step 2: Supersession -- FTS5 pre-filter then semantic check
    supersession_found = False
    if memory_type_for_item(candidate) == "semantic":
        fts_candidates, fts_vectors = self._find_dedup_candidates(candidate, limit=30)
        semantic_candidates = [
            c
            for c in fts_candidates
            if memory_type_for_item(c) == "semantic"
            and str(c.get("status", "")).lower() != "superseded"
        ]
        if semantic_candidates:
            superseded_idx = self._dedup.find_semantic_supersession(
                candidate,
                semantic_candidates,
                candidate_vec=candidate_vec,
                existing_vecs=fts_vectors,
            )
            if superseded_idx is not None:
                now_iso = _utc_now_iso()
                sup_event = dict(semantic_candidates[superseded_idx])
                sup_id = str(sup_event.get("id", ""))
                sup_event["status"] = "superseded"
                sup_event["superseded_at"] = now_iso
                if event_id:
                    sup_event["superseded_by_event_id"] = event_id
                if sup_id:
                    candidate["supersedes_event_id"] = sup_id
                candidate["supersedes_at"] = now_iso
                self._write_events([sup_event, candidate], embeddings=embeddings)
                written += 1
                superseded += 1
                supersession_found = True

    # Step 3: Semantic duplicate -- FTS5 pre-filter + Jaccard
    if not supersession_found:
        fts_candidates, fts_vectors = self._find_dedup_candidates(candidate, limit=30)
        if fts_candidates:
            dup_idx, dup_score = self._dedup.find_semantic_duplicate(
                candidate,
                fts_candidates,
                candidate_vec=candidate_vec,
                existing_vecs=fts_vectors,
            )
            if dup_idx is not None:
                merged_event = self._dedup.merge_events(
                    fts_candidates[dup_idx], candidate, similarity=dup_score
                )
                self._write_events([merged_event], embeddings=embeddings)
                merged += 1
                continue

        # Step 4: New event -- no match in any step
        self._write_events([candidate], embeddings=embeddings)
        written += 1
```

- [ ] **Step 5: Update mocked `_find_dedup_candidates` in existing tests**

In `tests/test_ingester.py`, update `TestAppendEventsEmbeddings`:

Line 342: `ingester._find_dedup_candidates = MagicMock(return_value=([], {}))` (was `return_value=[]`)

Line 372 (second test in that class): same change: `ingester._find_dedup_candidates = MagicMock(return_value=([], {}))`

- [ ] **Step 6: Run all ingester tests**

Run: `python -m pytest tests/test_ingester.py -v`
Expected: ALL PASS

- [ ] **Step 7: Run lint and typecheck**

Run: `make lint && make typecheck`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add nanobot/memory/write/ingester.py tests/test_ingester.py
git commit -m "feat(memory): wire pre-computed embeddings through dedup pipeline"
```

---

### Task 4: Remove `embedder` from `EventDeduplicator` construction in `store.py`

**Files:**
- Modify: `nanobot/memory/store.py:177-182`

- [ ] **Step 1: Update the construction**

In `nanobot/memory/store.py`, change lines 177-182 from:

```python
self._dedup = EventDeduplicator(
    coercer=self._coercer,
    conflict_pair_fn=self.profile_mgr._conflict_pair,
    alias_registry=self.alias_registry,
    embedder=self._embedder,
)
```

to:

```python
self._dedup = EventDeduplicator(
    coercer=self._coercer,
    conflict_pair_fn=self.profile_mgr._conflict_pair,
    alias_registry=self.alias_registry,
)
```

- [ ] **Step 2: Run contract tests**

Run: `python -m pytest tests/contract/ -v`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/ --ignore=tests/integration -x -q`
Expected: ALL PASS

- [ ] **Step 4: Run lint and typecheck**

Run: `make lint && make typecheck`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add nanobot/memory/store.py
git commit -m "refactor(memory): remove embedder dependency from EventDeduplicator"
```

---

### Task 5: Wrap `append_events` in `asyncio.to_thread()` for async callers

**Files:**
- Modify: `nanobot/memory/write/micro_extractor.py:193`
- Modify: `nanobot/memory/consolidation_pipeline.py:250`

- [ ] **Step 1: Fix micro_extractor.py**

In `nanobot/memory/write/micro_extractor.py`, change line 193 from:

```python
self._ingester.append_events(events, embeddings=embeddings)
```

to:

```python
await asyncio.to_thread(self._ingester.append_events, events, embeddings)
```

Note: `asyncio` is already imported at line 15.

- [ ] **Step 2: Fix consolidation_pipeline.py**

In `nanobot/memory/consolidation_pipeline.py`, change line 250 from:

```python
events_written = self._ingester.append_events(events, embeddings=embeddings)
```

to:

```python
events_written = await asyncio.to_thread(
    self._ingester.append_events, events, embeddings
)
```

Verify that `asyncio` is already imported at the top of the file (it should be, since the file uses `async def`).

- [ ] **Step 3: Run lint and typecheck**

Run: `make lint && make typecheck`
Expected: PASS

- [ ] **Step 4: Run full test suite**

Run: `python -m pytest tests/ --ignore=tests/integration -x -q`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add nanobot/memory/write/micro_extractor.py nanobot/memory/consolidation_pipeline.py
git commit -m "fix(memory): run append_events in thread to avoid blocking event loop"
```

---

### Task 6: Revert debug logging in frontend and update docs

**Files:**
- Modify: `frontend/src/lib/thread-list-adapter.ts`
- Modify: `nanobot/memory/write/dedup.py` (docstring only)

- [ ] **Step 1: Revert thread-list-adapter.ts debug logging**

Revert the `list()` method to its original form (remove try/catch wrapper and console.log statements):

```typescript
async list() {
    const response = await fetch("/api/threads");
    if (!response.ok) {
      return { threads: [] };
    }
    const data: ServerThreadListResponse = await response.json();
    return {
      threads: data.threads.map((t) => ({
        remoteId: t.threadId,
        status: "regular" as const,
        title: t.title === "New Chat" ? undefined : t.title,
      })),
    };
  },
```

- [ ] **Step 2: Run make check**

Run: `make check`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add frontend/src/lib/thread-list-adapter.ts
git commit -m "chore: revert debug logging in thread-list-adapter"
```

---

### Task 7: Update architecture documentation

**Files:**
- Modify: `.claude/rules/memory-architecture.md`

- [ ] **Step 1: Update Known Technical Debt section**

In `.claude/rules/memory-architecture.md`, under "## 12. Known Technical Debt", add to the "### Design Issues" resolved table:

```markdown
| `_sync_embed` in dedup | **Removed** — dedup no longer owns an embedder; uses pre-computed vectors from callers and `events_vec` storage |
| sync `append_events` in async callers | **Fixed** — micro-extractor and consolidation pipeline now use `asyncio.to_thread()` |
```

- [ ] **Step 2: Update the Write Pipeline diagram**

In section "## 4. Write Pipeline", update the "Micro-Extraction Differences" table to note that embeddings now flow through dedup:

Add a row: `| Dedup embedding | Calls `_sync_embed()` per candidate | Uses stored vectors from `events_vec` + caller pre-computed vectors |`

- [ ] **Step 3: Run doc-check**

Run: `make doc-check`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add .claude/rules/memory-architecture.md
git commit -m "docs: update memory architecture for dedup embedding flow fix"
```

---

### Task 8: Full validation

- [ ] **Step 1: Run make check**

Run: `make check`
Expected: PASS (lint + typecheck + import-check + structure-check + prompt-check + phase-todo-check + doc-check)

- [ ] **Step 2: Run full test suite with coverage**

Run: `make test-cov`
Expected: PASS with >= 85% coverage

- [ ] **Step 3: Verify no regressions in existing dedup behavior**

Run: `python -m pytest tests/test_event_deduplicator.py tests/test_ingester.py -v`
Expected: ALL PASS with same test count as before (no tests removed, only added/updated)
