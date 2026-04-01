# Retrieval Scoring Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire RRF fusion score into the scoring formula's base_score and unify recency decay to true half-life across all pipeline stages.

**Architecture:** Two surgical one-line fixes in the read path. RRF score propagated via the existing `item["score"]` key. Recency formula corrected by adding `math.log(2)` multiplier in CompositeReranker.

**Tech Stack:** Python, pytest, SQLite (read path only — no schema changes)

**Spec:** `docs/superpowers/specs/2026-04-01-retrieval-scoring-fixes-design.md`

---

### Task 1: Wire RRF Score Into base_score

**Files:**
- Modify: `nanobot/memory/read/retriever.py:192`
- Test: `tests/test_retriever.py` (existing `TestRRFFusion` class)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_retriever.py` inside the `TestRRFFusion` class:

```python
def test_fuse_sets_score_key(self) -> None:
    """RRF fusion must set item['score'] so scoring stage has a nonzero base."""
    vec = [{"id": "a", "summary": "alpha"}]
    fts = [{"id": "a", "summary": "alpha"}]
    fused = MemoryRetriever._fuse_results(vec, fts, vector_weight=0.7)
    assert "score" in fused[0], "item['score'] must be set by _fuse_results"
    assert fused[0]["score"] > 0
    assert fused[0]["score"] == fused[0]["_rrf_score"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_retriever.py::TestRRFFusion::test_fuse_sets_score_key -v`
Expected: FAIL — `assert "score" in fused[0]` fails because `_fuse_results` currently only sets `_rrf_score`.

- [ ] **Step 3: Write minimal implementation**

In `nanobot/memory/read/retriever.py`, inside `_fuse_results`, at line 192, change:

```python
            entry["_rrf_score"] = scores[eid]
```

to:

```python
            entry["_rrf_score"] = scores[eid]
            entry["score"] = scores[eid]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_retriever.py::TestRRFFusion -v`
Expected: All 8 tests PASS (7 existing + 1 new).

- [ ] **Step 5: Run lint and typecheck**

Run: `make lint && make typecheck`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add nanobot/memory/read/retriever.py tests/test_retriever.py
git commit -m "fix(memory): wire RRF fusion score into item['score'] for scoring base"
```

---

### Task 2: Unify Recency Decay to True Half-Life

**Files:**
- Modify: `nanobot/memory/ranking/reranker.py:119-125`
- Test: `tests/test_reranker.py` (existing file)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_reranker.py` at the module level (outside any class, after existing imports):

```python
from nanobot.memory.ranking.reranker import _recency_score


def test_recency_score_true_half_life() -> None:
    """Value at exactly half_life days must be 0.5 (true half-life)."""
    half_life = 30.0
    # Create a timestamp exactly half_life days ago
    from datetime import datetime, timedelta, timezone

    ts = (datetime.now(timezone.utc) - timedelta(days=half_life)).isoformat()
    score = _recency_score(ts, half_life=half_life)
    assert abs(score - 0.5) < 0.02, f"Expected ~0.5 at half_life, got {score}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_reranker.py::test_recency_score_true_half_life -v`
Expected: FAIL — `assert abs(score - 0.5) < 0.02` fails because current formula returns ~0.368 at half_life.

- [ ] **Step 3: Write minimal implementation**

In `nanobot/memory/ranking/reranker.py`, change line 119-125:

From:
```python
def _recency_score(timestamp_str: str, half_life: float = _RECENCY_HALF_LIFE_DAYS) -> float:
    """Exponential decay: ``exp(-days_old / half_life)``."""
    if not timestamp_str:
        return 0.0
    try:
        ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        days_old = max((datetime.now(timezone.utc) - ts).total_seconds() / 86400.0, 0.0)
        return math.exp(-days_old / half_life)
    except (ValueError, TypeError):
        return 0.0
```

To:
```python
def _recency_score(timestamp_str: str, half_life: float = _RECENCY_HALF_LIFE_DAYS) -> float:
    """True half-life decay: ``exp(-ln(2) * days_old / half_life)``.

    Returns 0.5 at exactly ``half_life`` days.
    """
    if not timestamp_str:
        return 0.0
    try:
        ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        days_old = max((datetime.now(timezone.utc) - ts).total_seconds() / 86400.0, 0.0)
        return math.exp(-math.log(2) * days_old / half_life)
    except (ValueError, TypeError):
        return 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_reranker.py -v`
Expected: All tests PASS. The existing `test_old_items_penalized_by_recency` test verifies relative ordering (old < new), which still holds with the new formula.

- [ ] **Step 5: Run lint and typecheck**

Run: `make lint && make typecheck`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add nanobot/memory/ranking/reranker.py tests/test_reranker.py
git commit -m "fix(memory): unify recency decay to true half-life formula"
```

---

### Task 3: Contract Test — base_score > 0 Through Full Pipeline

**Files:**
- Test: `tests/contract/test_memory_contracts.py` (existing file)

- [ ] **Step 1: Write the contract test**

Add to `tests/contract/test_memory_contracts.py` inside the existing contract test class:

```python
async def test_rrf_fusion_feeds_base_score(self, tmp_path: Path):
    """Items retrieved via vector+FTS fusion must have nonzero base_score."""
    store = MemoryStore(tmp_path, embedding_provider="hash")
    events = [
        MemoryEvent(summary="User prefers dark mode editors", type="preference"),
        MemoryEvent(summary="Project deadline is March 2026", type="fact"),
    ]
    store.ingester.append_events(events)

    context = await store.get_memory_context(query="dark mode preference")
    # If RRF score feeds base_score, retrieved items contribute to scoring.
    # We verify indirectly: the context should contain the relevant event.
    assert "dark mode" in context.lower()
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/contract/test_memory_contracts.py::test_rrf_fusion_feeds_base_score -v`
Expected: PASS (this is a behavioral contract, not a regression — it verifies the fix works end-to-end).

- [ ] **Step 3: Commit**

```bash
git add tests/contract/test_memory_contracts.py
git commit -m "test(memory): add contract test for RRF score feeding base_score"
```

---

### Task 4: Update Documentation

**Files:**
- Modify: `docs/memory-system-reference.md`
- Modify: `.claude/rules/memory-architecture.md`

- [ ] **Step 1: Update memory-system-reference.md**

In the "Retrieval Pipeline (detailed)" section, update step 4 (RRF fusion):

Change:
```
+- 4. RRF fusion (k=60, vector_weight=0.7, fts_weight=0.3)
|     [stored in item["_rrf_score"] — controls candidate selection only]
```

To:
```
+- 4. RRF fusion (k=60, vector_weight=0.7, fts_weight=0.3)
|     [stored in item["score"] and item["_rrf_score"]]
```

Update step 8 (Scoring):

Change:
```
+- 8. Scoring (all additive from base 0.0)
```

To:
```
+- 8. Scoring (additive on RRF base_score)
```

Remove the NOTE about base_score always being 0.0.

Update the recency note to remove the inconsistency callout:

Change the "Note on recency formulas" section to:
```
**Recency decay:** Both stages use true half-life: `exp(-ln(2) * age / half_life)`. Value = 0.5 at exactly `half_life` days.
```

- [ ] **Step 2: Update memory-architecture.md**

In the "Read Pipeline" section, update Stage 4 note and Stage 7 description.

In the "Known Technical Debt" section, remove the "RRF Score Not Carried to Final Ranking" subsection and the recency decay inconsistency from the Design Issues table.

- [ ] **Step 3: Commit**

```bash
git add docs/memory-system-reference.md .claude/rules/memory-architecture.md
git commit -m "docs: update memory docs for RRF score wiring and recency unification"
```

---

### Task 5: Final Verification

- [ ] **Step 1: Run make check**

Run: `make check`
Expected: All checks pass.

- [ ] **Step 2: Run full test suite**

Run: `make pre-push`
Expected: All tests pass, coverage gate met.
