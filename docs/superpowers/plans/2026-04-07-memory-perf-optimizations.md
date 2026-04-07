# Memory Performance Optimizations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce memory read latency (~200-300ms), halve write-path DB queries, and cut ~30-40% of micro-extraction LLM calls.

**Architecture:** Three independent optimizations, each touching exactly one file. No cross-component changes, no new files, no import boundary changes. TDD where new behavior is added (Opt 1 degradation, Opt 3 skip logic); existing tests cover Opt 2.

**Tech Stack:** Python 3.12, asyncio, SQLite (FTS5 + sqlite-vec), pytest-asyncio

**Spec:** `docs/superpowers/specs/2026-04-07-memory-perf-optimizations-design.md`

---

## File Map

| File | Action | Optimization |
|------|--------|-------------|
| `nanobot/memory/read/retriever.py` | Modify lines 155-162 | Opt 1: Parallelize embed + FTS |
| `nanobot/memory/write/ingester.py` | Modify lines 100-178 | Opt 2: Cache FTS dedup candidates |
| `nanobot/memory/write/micro_extractor.py` | Modify lines 124-146, add constants | Opt 3: Trivial-turn skip |
| `tests/test_retriever.py` | Add test class | Opt 1: Embed failure degradation |
| `tests/test_micro_extraction_skip.py` | Create | Opt 3: Trivial-turn skip tests |

---

## Task 1: Parallelize Embedding + FTS5 Search (Opt 1)

**Files:**
- Modify: `nanobot/memory/read/retriever.py:133-167`
- Test: `tests/test_retriever.py` (add class)

### Step 1.1: Write the failing test — embed failure degrades to FTS-only

- [ ] Add this test class to the end of `tests/test_retriever.py`:

```python
class TestEmbedFailureDegradation:
    """When embed() raises, retrieval falls back to FTS-only results."""

    async def test_embed_failure_returns_fts_results(self) -> None:
        """Embed raises → FTS results still returned, search_vector not called."""
        retriever = _make_retriever()

        mock_db = MagicMock()
        mock_db.search_fts = MagicMock(
            return_value=[
                {
                    "id": "f1",
                    "type": "fact",
                    "summary": "fts hit survives embed failure",
                    "timestamp": "2025-01-01T00:00:00Z",
                    "entities": [],
                    "status": "active",
                },
            ]
        )
        mock_db.search_vector = MagicMock(return_value=[])
        mock_db.read_events = MagicMock(return_value=[])

        mock_embedder = MagicMock()
        mock_embedder.vector_quality = 0.7

        async def _failing_embed(text: str) -> list[float]:
            raise RuntimeError("API key expired")

        mock_embedder.embed = _failing_embed

        retriever._db = mock_db
        retriever._embedder = mock_embedder

        results = await retriever.retrieve("test query", top_k=5)
        # FTS results should survive embed failure
        assert len(results) >= 1
        assert results[0].summary == "fts hit survives embed failure"
        # search_vector should NOT be called (no vector available)
        mock_db.search_vector.assert_not_called()

    async def test_embed_success_still_calls_vector_search(self) -> None:
        """Normal path: embed succeeds → both FTS and vector search run."""
        retriever = _make_retriever()

        mock_db = MagicMock()
        mock_db.search_vector = MagicMock(
            return_value=[
                {
                    "id": "v1",
                    "type": "fact",
                    "summary": "vector hit",
                    "timestamp": "2025-01-01T00:00:00Z",
                    "entities": [],
                    "status": "active",
                },
            ]
        )
        mock_db.search_fts = MagicMock(
            return_value=[
                {
                    "id": "f1",
                    "type": "fact",
                    "summary": "fts hit",
                    "timestamp": "2025-01-01T00:00:00Z",
                    "entities": [],
                    "status": "active",
                },
            ]
        )

        mock_embedder = MagicMock()
        mock_embedder.vector_quality = 0.7

        async def _good_embed(text: str) -> list[float]:
            return [0.1, 0.2, 0.3]

        mock_embedder.embed = _good_embed

        retriever._db = mock_db
        retriever._embedder = mock_embedder

        results = await retriever.retrieve("test query", top_k=5)
        mock_db.search_vector.assert_called_once()
        mock_db.search_fts.assert_called_once()
        assert len(results) >= 1
```

### Step 1.2: Run test to verify it fails

- [ ] Run:

```bash
python -m pytest tests/test_retriever.py::TestEmbedFailureDegradation -v
```

Expected: `test_embed_failure_returns_fts_results` FAILS (embed exception propagates uncaught, retrieval raises instead of returning FTS results).
`test_embed_success_still_calls_vector_search` should PASS (existing behavior).

### Step 1.3: Implement the parallel embed + FTS with graceful degradation

- [ ] In `nanobot/memory/read/retriever.py`, replace lines 155-162 in `_retrieve_unified()`:

**Replace this:**

```python
        # 1. Embed query
        query_vec = await self._embedder.embed(query)

        # 2. Dual source — DB methods are synchronous; run concurrently via to_thread
        vec_results, fts_results = await asyncio.gather(
            asyncio.to_thread(self._db.search_vector, query_vec, candidate_k),
            asyncio.to_thread(self._db.search_fts, query, candidate_k),
        )
```

**With this:**

```python
        # 1. Embed + FTS concurrently (FTS does not need the vector)
        async def _safe_embed() -> list[float] | None:
            try:
                return await self._embedder.embed(query)
            except Exception:  # crash-barrier: degrade to FTS-only on embed failure
                bind_trace().warning("Embedding failed, falling back to FTS-only retrieval")
                return None

        query_vec, fts_results = await asyncio.gather(
            _safe_embed(),
            asyncio.to_thread(self._db.search_fts, query, candidate_k),
        )

        # 2. Vector search only if embedding succeeded
        if query_vec is not None:
            vec_results = await asyncio.to_thread(
                self._db.search_vector, query_vec, candidate_k
            )
        else:
            vec_results = []
```

### Step 1.4: Run tests to verify they pass

- [ ] Run:

```bash
python -m pytest tests/test_retriever.py -v
```

Expected: ALL tests PASS, including the two new ones and all existing ones.

### Step 1.5: Run lint + typecheck

- [ ] Run:

```bash
make lint && make typecheck
```

Expected: No errors.

### Step 1.6: Commit

- [ ] Run:

```bash
git add nanobot/memory/read/retriever.py tests/test_retriever.py
git commit -m "perf(memory): parallelize embedding + FTS5 search in retriever

Embed API call (200-500ms) and FTS5 search now run concurrently via
asyncio.gather. Vector search runs sequentially after embed completes.
Embed failure degrades gracefully to FTS-only results instead of
failing the entire retrieval.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Cache FTS Dedup Candidates (Opt 2)

**Files:**
- Modify: `nanobot/memory/write/ingester.py:100-178`

### Step 2.1: Verify existing tests pass before modifying

- [ ] Run:

```bash
python -m pytest tests/test_ingester.py tests/test_event_deduplicator.py -v
```

Expected: ALL tests PASS. This is the baseline.

### Step 2.2: Implement the FTS candidate caching

- [ ] In `nanobot/memory/write/ingester.py`, replace the per-event loop body (lines 100-178) inside `append_events()`.

**Replace this (lines 100-178):**

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

            # Step 1: Exact ID dedup — O(1) PK lookup
            existing_row = self._db.get_event_by_id(event_id)
            if existing_row is not None:
                existing = self._unpack_event(existing_row)
                existing = self._coercer.ensure_event_provenance(existing)
                merged_event = self._dedup.merge_events(existing, candidate, similarity=1.0)
                self._write_events([merged_event], embeddings=embeddings)
                merged += 1
                continue

            # Step 2: Supersession — FTS5 pre-filter then existing logic
            candidate_vec = embeddings.get(event_id) if embeddings else None
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

            # Step 3: Semantic duplicate — FTS5 pre-filter + Jaccard
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

                # Step 4: New event — no match in any step
                self._write_events([candidate], embeddings=embeddings)
                written += 1
```

**With this:**

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

            # Step 1: Exact ID dedup — O(1) PK lookup
            existing_row = self._db.get_event_by_id(event_id)
            if existing_row is not None:
                existing = self._unpack_event(existing_row)
                existing = self._coercer.ensure_event_provenance(existing)
                merged_event = self._dedup.merge_events(existing, candidate, similarity=1.0)
                self._write_events([merged_event], embeddings=embeddings)
                merged += 1
                continue

            # Pre-fetch FTS candidates once — reused for both supersession and dedup.
            # memory_type_for_item is a pure function on the candidate dict, so
            # is_semantic is stable across steps 2 and 3.
            candidate_vec = embeddings.get(event_id) if embeddings else None
            is_semantic = memory_type_for_item(candidate) == "semantic"
            fts_candidates: list[dict[str, Any]] = []
            fts_vectors: dict[str, list[float]] = {}
            if is_semantic:
                fts_candidates, fts_vectors = self._find_dedup_candidates(
                    candidate, limit=30
                )

            # Step 2: Supersession — semantic events only
            supersession_found = False
            if is_semantic:
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

            # Step 3: Semantic duplicate — reuses FTS candidates from step 2
            if not supersession_found:
                if not fts_candidates:
                    fts_candidates, fts_vectors = self._find_dedup_candidates(
                        candidate, limit=30
                    )
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

                # Step 4: New event — no match in any step
                self._write_events([candidate], embeddings=embeddings)
                written += 1
```

### Step 2.3: Run tests to verify behavioral equivalence

- [ ] Run:

```bash
python -m pytest tests/test_ingester.py tests/test_event_deduplicator.py tests/contract/test_memory_contracts.py -v
```

Expected: ALL tests PASS. No behavioral change — only the internal number of FTS queries changed.

### Step 2.4: Run lint + typecheck

- [ ] Run:

```bash
make lint && make typecheck
```

Expected: No errors.

### Step 2.5: Commit

- [ ] Run:

```bash
git add nanobot/memory/write/ingester.py
git commit -m "perf(memory): cache FTS dedup candidates across supersession and duplicate checks

Hoist _find_dedup_candidates() above the supersession check so both
step 2 (supersession) and step 3 (duplicate) reuse the same FTS
results. Halves DB queries per semantic event in the write path.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Trivial-Turn Skip for Micro-Extraction (Opt 3)

**Files:**
- Modify: `nanobot/memory/write/micro_extractor.py:31-146`
- Create: `tests/test_micro_extraction_skip.py`

### Step 3.1: Write the failing tests

- [ ] Create `tests/test_micro_extraction_skip.py`:

```python
"""Tests for micro-extraction trivial-turn skip pre-filter."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.memory.write.micro_extractor import (
    MicroExtractor,
    _TRIVIAL_ASSISTANT_MAX_LEN,
    _TRIVIAL_MAX_LEN,
    _TRIVIAL_PATTERNS,
)


def _make_extractor(*, enabled: bool = True) -> tuple[MicroExtractor, MagicMock]:
    """Build a MicroExtractor with mocked provider and ingester."""
    provider = MagicMock()
    ingester = MagicMock()
    extractor = MicroExtractor(
        provider=provider,
        ingester=ingester,
        model="gpt-4o-mini",
        enabled=enabled,
    )
    return extractor, provider


class TestTrivialTurnSkip:
    """Trivial user messages skip the LLM call entirely."""

    async def test_ok_skipped(self) -> None:
        ext, provider = _make_extractor()
        await ext.submit("ok", "Got it.", channel="cli")
        # No async task should be pending — LLM never called
        assert len(ext._pending_tasks) == 0

    async def test_thanks_with_punctuation_skipped(self) -> None:
        ext, provider = _make_extractor()
        await ext.submit("Thanks!", "You're welcome.", channel="cli")
        assert len(ext._pending_tasks) == 0

    async def test_yes_question_mark_skipped(self) -> None:
        ext, provider = _make_extractor()
        await ext.submit("yes?", "Proceeding.", channel="cli")
        assert len(ext._pending_tasks) == 0

    async def test_emoji_skipped(self) -> None:
        ext, provider = _make_extractor()
        await ext.submit("👍", "Great!", channel="cli")
        assert len(ext._pending_tasks) == 0

    async def test_mixed_case_skipped(self) -> None:
        ext, provider = _make_extractor()
        await ext.submit("OKAY", "Fine.", channel="cli")
        assert len(ext._pending_tasks) == 0

    async def test_multiword_trivial_skipped(self) -> None:
        ext, provider = _make_extractor()
        await ext.submit("sounds good", "Thanks!", channel="cli")
        assert len(ext._pending_tasks) == 0

    async def test_go_ahead_skipped(self) -> None:
        ext, provider = _make_extractor()
        await ext.submit("go ahead", "Starting now.", channel="cli")
        assert len(ext._pending_tasks) == 0


class TestNonTrivialPassThrough:
    """Non-trivial messages must always pass through to LLM extraction."""

    async def test_meaningful_short_message_passes(self) -> None:
        ext, provider = _make_extractor()
        # "no, use Python 3.12" is 20 chars, within length threshold,
        # but NOT in the pattern set — must pass through
        await ext.submit("no, use Python 3.12", "Updating to 3.12.", channel="cli")
        # Give the background task a moment to be created
        await asyncio.sleep(0.01)
        assert len(ext._pending_tasks) >= 1

    async def test_long_message_passes(self) -> None:
        ext, provider = _make_extractor()
        await ext.submit(
            "The vault is at C:\\Users\\me\\Documents\\PM",
            "Noted.",
            channel="cli",
        )
        await asyncio.sleep(0.01)
        assert len(ext._pending_tasks) >= 1

    async def test_trivial_user_long_assistant_passes(self) -> None:
        """Trivial user msg + long assistant response → must NOT skip."""
        ext, provider = _make_extractor()
        long_assistant = "Actually, I realize DS10540 is in a different folder. " * 5
        assert len(long_assistant.strip()) > _TRIVIAL_ASSISTANT_MAX_LEN
        await ext.submit("ok", long_assistant, channel="cli")
        await asyncio.sleep(0.01)
        assert len(ext._pending_tasks) >= 1


class TestEmptyMessageSkip:
    """Empty or whitespace-only messages are skipped immediately."""

    async def test_empty_string_skipped(self) -> None:
        ext, provider = _make_extractor()
        await ext.submit("", "Hello.", channel="cli")
        assert len(ext._pending_tasks) == 0

    async def test_whitespace_only_skipped(self) -> None:
        ext, provider = _make_extractor()
        await ext.submit("   ", "Response.", channel="cli")
        assert len(ext._pending_tasks) == 0


class TestDisabledExtractor:
    """When disabled, submit() is a no-op regardless of message content."""

    async def test_disabled_skips_everything(self) -> None:
        ext, provider = _make_extractor(enabled=False)
        await ext.submit("important fact about the user", "Noted.", channel="cli")
        assert len(ext._pending_tasks) == 0


class TestTrivialConstants:
    """Verify the constants are well-formed."""

    def test_patterns_is_frozenset(self) -> None:
        assert isinstance(_TRIVIAL_PATTERNS, frozenset)

    def test_all_patterns_are_lowercase(self) -> None:
        for p in _TRIVIAL_PATTERNS:
            assert p == p.lower() or not p.isascii(), f"Pattern {p!r} is not lowercase"

    def test_max_len_is_positive(self) -> None:
        assert _TRIVIAL_MAX_LEN > 0

    def test_assistant_max_len_is_positive(self) -> None:
        assert _TRIVIAL_ASSISTANT_MAX_LEN > 0

    def test_all_patterns_within_max_len(self) -> None:
        """Every pattern in the set should be within the max length threshold."""
        for p in _TRIVIAL_PATTERNS:
            assert len(p) <= _TRIVIAL_MAX_LEN, (
                f"Pattern {p!r} ({len(p)} chars) exceeds _TRIVIAL_MAX_LEN ({_TRIVIAL_MAX_LEN})"
            )
```

### Step 3.2: Run tests to verify they fail

- [ ] Run:

```bash
python -m pytest tests/test_micro_extraction_skip.py -v
```

Expected: Import errors for `_TRIVIAL_PATTERNS`, `_TRIVIAL_MAX_LEN`, `_TRIVIAL_ASSISTANT_MAX_LEN` (they don't exist yet). All tests FAIL.

### Step 3.3: Implement the trivial-turn skip

- [ ] In `nanobot/memory/write/micro_extractor.py`, add the constants after the existing `_MICRO_EXTRACT_TOOL` definition (after line 85):

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
_TRIVIAL_ASSISTANT_MAX_LEN: int = 100
```

- [ ] Modify the `submit()` method. Replace lines 124-146:

**Replace this:**

```python
    async def submit(
        self,
        user_message: str,
        assistant_message: str,
        *,
        channel: str = "",
        tool_hints: list[str] | None = None,
        turn_timestamp: str = "",
    ) -> None:
        """Submit a turn for background extraction. Returns immediately."""
        if not self._enabled:
            return
        task = asyncio.create_task(
```

**With this:**

```python
    async def submit(
        self,
        user_message: str,
        assistant_message: str,
        *,
        channel: str = "",
        tool_hints: list[str] | None = None,
        turn_timestamp: str = "",
    ) -> None:
        """Submit a turn for background extraction. Returns immediately."""
        if not self._enabled:
            return
        # Pre-filter: skip trivial turns that never produce memory events
        stripped = user_message.strip()
        if not stripped:
            logger.debug("Micro-extraction: skipped empty turn")
            return
        if (
            len(stripped) <= _TRIVIAL_MAX_LEN
            and stripped.lower().rstrip("!.,?") in _TRIVIAL_PATTERNS
            and len(assistant_message.strip()) <= _TRIVIAL_ASSISTANT_MAX_LEN
        ):
            logger.debug("Micro-extraction: skipped trivial turn ({!r})", stripped[:30])
            return
        task = asyncio.create_task(
```

### Step 3.4: Run tests to verify they pass

- [ ] Run:

```bash
python -m pytest tests/test_micro_extraction_skip.py -v
```

Expected: ALL tests PASS.

### Step 3.5: Run full test suite to verify no regressions

- [ ] Run:

```bash
python -m pytest tests/test_micro_extraction_skip.py tests/integration/test_micro_extraction.py -v
```

Expected: ALL tests PASS (both new and existing).

### Step 3.6: Run lint + typecheck

- [ ] Run:

```bash
make lint && make typecheck
```

Expected: No errors.

### Step 3.7: Commit

- [ ] Run:

```bash
git add nanobot/memory/write/micro_extractor.py tests/test_micro_extraction_skip.py
git commit -m "perf(memory): skip micro-extraction for trivial turns

Add pre-filter in MicroExtractor.submit() that skips LLM calls for
trivial user messages (ok, thanks, yes, etc.) when the assistant
response is also short. Saves ~30-40% of micro-extraction LLM calls.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Final Validation

### Step 4.1: Run make check

- [ ] Run:

```bash
make check
```

Expected: ALL checks pass (lint + typecheck + import-check + structure-check + prompt-check + phase-todo-check + doc-check).

### Step 4.2: Run make test

- [ ] Run:

```bash
make test
```

Expected: ALL unit tests pass. No regressions.

### Step 4.3: Run make pre-push (if pushing)

- [ ] Run:

```bash
make pre-push
```

Expected: Full CI suite passes including coverage gate.
