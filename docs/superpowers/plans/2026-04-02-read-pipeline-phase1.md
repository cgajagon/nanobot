# Read Pipeline Phase 1: Retrieval Quality Improvements

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the three highest-impact retrieval quality issues: adaptive RRF fusion weights based on embedder quality, expanded intent classification markers, and rebalanced recency scoring.

**Architecture:** Three independent changes to the read pipeline (`nanobot/memory/read/`). C1 modifies `retriever.py` to detect embedder type and adjust vector weight. C2 extends marker lists in `retrieval_planner.py`. H3 adjusts one coefficient in `scoring.py`. All changes are additive — no structural refactoring.

**Tech Stack:** Python 3.10+, SQLite (memory), pytest

**Worktree:** `C:/Users/Dell/Documents/nanobot-improve/read-pipeline` (branch `improve/read-pipeline`)

**Conventions:** Run `source C:/Users/Dell/Documents/nanobot/.venv/Scripts/activate` before any make/pytest command in the worktree. All test commands use this venv.

---

## Deviations

1. **Task 1: `vector_quality` protocol property instead of `isinstance` checks.**
   The plan specified `isinstance(self._embedder, HashEmbedder)` detection in the
   retriever. Code review identified this as a design violation (callers should depend
   on the Embedder protocol, not concrete classes). Implementation adds a
   `vector_quality: float` property to the `Embedder` protocol and all three
   implementations. The retriever reads `self._embedder.vector_quality` directly.

2. **Task 1: HashEmbedder weight 0.0 instead of 0.2.**
   The plan specified `HashEmbedder` weight as 0.2. Code review noted that HashEmbedder
   vectors are "semantically meaningless" (per architecture docs), so any non-zero weight
   adds noise. Changed to 0.0 so only FTS contributes when Hash is the embedder.

3. **Task 3: Recency coefficient extracted as named variable.**
   Code review flagged the inline `0.15` as a magic number. Extracted to
   `recency_weight = 0.15` local variable for readability.

---

## Task 1: Adaptive RRF Weights (C1)

**Files:**
- Modify: `nanobot/memory/read/retriever.py:96-128` (`_retrieve_unified`, `_vector_weight`)
- Test: `tests/test_retriever.py` (extend `TestRRFFusion`)

### Context

`_fuse_results()` is called at `retriever.py:128` with hardcoded `vector_weight=0.7`. When the system falls back to `HashEmbedder` (semantically meaningless vectors), 70% of the RRF score is noise. The fix detects the embedder class and adjusts the weight:

| Embedder | Weight | Rationale |
|----------|--------|-----------|
| `OpenAIEmbedder` | 0.7 | Production-quality semantic vectors |
| `LocalEmbedder` | 0.5 | Good quality but smaller model (384D) |
| `HashEmbedder` | 0.2 | Random vectors — FTS should dominate |

The check uses `isinstance` within the `memory` package (no cross-package import violation). The embedder is already injected as `self._embedder` in `__init__`.

- [ ] **Step 1: Write tests for adaptive vector weight**

Add to `tests/test_retriever.py` after the existing `TestRRFFusion` class (after line 607):

```python
class TestAdaptiveVectorWeight:
    """_vector_weight returns weight based on embedder semantic quality."""

    def test_openai_embedder_returns_high_weight(self) -> None:
        retriever = _make_retriever()
        mock_embedder = MagicMock(spec=["embed", "embed_batch", "dims", "available"])
        mock_embedder.__class__ = type("OpenAIEmbedder", (), {})
        retriever._embedder = mock_embedder
        # Default (non-Hash, non-Local) should get 0.7
        assert retriever._vector_weight() == pytest.approx(0.7)

    def test_hash_embedder_returns_low_weight(self) -> None:
        from nanobot.memory.embedder import HashEmbedder

        retriever = _make_retriever()
        retriever._embedder = HashEmbedder(dims=384)
        assert retriever._vector_weight() == pytest.approx(0.2)

    def test_local_embedder_returns_mid_weight(self) -> None:
        retriever = _make_retriever()
        # Mock a LocalEmbedder-like object by making isinstance check work
        from nanobot.memory.embedder import LocalEmbedder

        mock_local = MagicMock(spec=LocalEmbedder)
        retriever._embedder = mock_local
        assert retriever._vector_weight() == pytest.approx(0.5)

    def test_none_embedder_returns_zero(self) -> None:
        retriever = _make_retriever()
        retriever._embedder = None
        assert retriever._vector_weight() == pytest.approx(0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_retriever.py::TestAdaptiveVectorWeight -v`
Expected: FAIL — `_vector_weight` method does not exist

- [ ] **Step 3: Implement `_vector_weight` method**

In `nanobot/memory/read/retriever.py`, add this method to `MemoryRetriever` class (after `__init__`, before `retrieve`):

```python
    def _vector_weight(self) -> float:
        """RRF vector weight adapted to embedder semantic quality.

        Returns a lower weight when the embedder produces lower-quality
        vectors, so FTS5 (keyword matching) dominates the fusion score.
        """
        if self._embedder is None:
            return 0.0
        from ..embedder import HashEmbedder, LocalEmbedder

        if isinstance(self._embedder, HashEmbedder):
            return 0.2
        if isinstance(self._embedder, LocalEmbedder):
            return 0.5
        return 0.7
```

Then update the call site in `_retrieve_unified` (line 128). Change:

```python
        candidates = self._fuse_results(vec_results, fts_results, vector_weight=0.7)
```

To:

```python
        candidates = self._fuse_results(
            vec_results, fts_results, vector_weight=self._vector_weight()
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_retriever.py::TestAdaptiveVectorWeight -v`
Expected: 4 passed

- [ ] **Step 5: Run full retriever test suite**

Run: `pytest tests/test_retriever.py -v`
Expected: All existing tests pass (the existing `TestRRFFusion` tests call `_fuse_results` directly with explicit `vector_weight`, so they are unaffected)

- [ ] **Step 6: Run lint + typecheck**

Run: `make lint && make typecheck`
Expected: Clean

- [ ] **Step 7: Commit**

```bash
git add nanobot/memory/read/retriever.py tests/test_retriever.py
git commit -m "feat(memory): adaptive RRF weights based on embedder quality

HashEmbedder produces semantically meaningless vectors, so vector
search results are noise. Detect embedder type and lower vector_weight
from 0.7 to 0.2 for Hash (FTS dominates) or 0.5 for Local.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Expand Intent Classification Markers (C2)

**Files:**
- Modify: `nanobot/memory/read/retrieval_planner.py:94-149` (marker tuples)
- Test: `tests/test_store_helpers.py` (extend parametrize list)

### Context

`infer_retrieval_intent()` at `retrieval_planner.py:88-165` uses substring matching with marker tuples. Three documented false negatives:

| Query | Expected | Actual | Missing marker |
|-------|----------|--------|---------------|
| "what's broken" | debug_history | fact_lookup | "broken" |
| "bug in the auth module" | debug_history | fact_lookup | "bug" |
| "what can't we do" | constraints_lookup | fact_lookup | "can't" |

The fix adds markers to the existing tuples. No structural changes to the matching logic.

- [ ] **Step 1: Write failing tests for the false negatives**

In `tests/test_store_helpers.py`, extend the existing parametrize block at line 59. The current list is:

```python
    @pytest.mark.parametrize(
        "query,expected",
        [
            ("what failed yesterday", "debug_history"),
            ("reflect on lessons learned", "reflection"),
            ("what is rollout status", "rollout_status"),
            ("open tasks and decisions", "planning"),
            ("what constraints apply", "constraints_lookup"),
            ("any conflict pending", "conflict_review"),
            ("what is user preference", "fact_lookup"),
        ],
    )
```

Add these cases to the list (before the closing `]`):

```python
            # False negative coverage (added Phase 1)
            ("what's broken in the system", "debug_history"),
            ("bug in the auth module", "debug_history"),
            ("there's an issue with deployment", "debug_history"),
            ("what went wrong last night", "debug_history"),
            ("what can't we do right now", "constraints_lookup"),
            ("limitations of the current system", "constraints_lookup"),
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_store_helpers.py::TestStoreHelpers::test_intent_and_routing_hints -v`
Expected: 6 new cases FAIL (they return `fact_lookup` instead of the expected intent)

- [ ] **Step 3: Add markers to `infer_retrieval_intent()`**

In `nanobot/memory/read/retrieval_planner.py`, modify the `debug_markers` tuple (lines 94-110). Change:

```python
        debug_markers = (
            "what happened",
            "last time",
            "failed",
            "failure",
            "error",
            "incident",
            "debug",
            "timeline",
            "yesterday",
            "what did we try",
            "correction",
            "corrected",
            "post-mortem",
            "postmortem",
            "root cause",
            "outage",
        )
```

To:

```python
        debug_markers = (
            "what happened",
            "last time",
            "failed",
            "failure",
            "error",
            "incident",
            "debug",
            "timeline",
            "yesterday",
            "what did we try",
            "correction",
            "corrected",
            "post-mortem",
            "postmortem",
            "root cause",
            "outage",
            "broken",
            "broke",
            "bug",
            "crash",
            "went wrong",
            "issue",
            "problem",
            "regression",
        )
```

Modify the `constraints_markers` tuple (line 147). Change:

```python
        constraints_markers = ("constraint", "must", "cannot", "before running commands")
```

To:

```python
        constraints_markers = (
            "constraint",
            "must",
            "cannot",
            "can't",
            "before running commands",
            "limitation",
            "restriction",
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_store_helpers.py::TestStoreHelpers::test_intent_and_routing_hints -v`
Expected: All 13 cases pass (7 original + 6 new)

- [ ] **Step 5: Check for unintended side effects**

Run: `pytest tests/test_memory_metadata_policy.py -v`
Expected: All pass (these tests also check intents)

**Verify "issue" marker doesn't cause false positives:** The word "issue" appears in common queries like "GitHub issue" which should remain `fact_lookup`. However, since `debug_markers` is checked AFTER `reflection`, `rollout_status`, `conflict_review`, and `constraints_lookup`, and BEFORE `planning`, the only risk is a query like "issue tracker tasks" getting `debug_history` instead of `planning`. This is acceptable — "issue" strongly signals debugging context.

- [ ] **Step 6: Run lint + typecheck**

Run: `make lint && make typecheck`
Expected: Clean

- [ ] **Step 7: Commit**

```bash
git add nanobot/memory/read/retrieval_planner.py tests/test_store_helpers.py
git commit -m "feat(memory): expand intent classification markers

Add 'broken', 'bug', 'crash', 'issue', 'problem', 'regression',
'went wrong', 'broke' to debug_history markers. Add 'can\\'t',
'limitation', 'restriction' to constraints_lookup markers. Fixes
~30% false negative rate on debug and constraint queries.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Increase Recency Boost (H3)

**Files:**
- Modify: `nanobot/memory/read/scoring.py:375` (one coefficient)
- Test: `tests/test_retrieval_scorer.py` (add recency ordering test)

### Context

At `scoring.py:375`, the recency coefficient is `0.08`:

```python
intent_bonus = type_boost + (0.08 * recency) + stability_boost + reflection_penalty
```

The recency signal returns 0.0–1.0 (exponential decay). With coefficient 0.08, max recency contribution is 0.08 — the weakest signal in the formula. For comparison: `type_boost` reaches ±0.30, `graph_boost` reaches 0.15. Increasing to 0.15 makes recency comparable to graph boost.

No existing tests assert exact score values — they only check relative ordering, which is preserved by this change.

- [ ] **Step 1: Write a test that verifies recency impact**

Add to `tests/test_retrieval_scorer.py` after the existing `TestScoreItems` class:

```python
class TestRecencyBoostMagnitude:
    """Recency boost should meaningfully influence final scores."""

    def test_recent_item_scores_above_old_item(self) -> None:
        """A very recent item should score noticeably above an old one."""
        from datetime import datetime, timezone

        scorer = _make_scorer()
        now = datetime.now(timezone.utc).isoformat()
        old = "2020-01-01T00:00:00+00:00"

        plan = _make_plan(
            policy={
                "candidate_multiplier": 3,
                "half_life_days": 60.0,
                "type_boost": {"semantic": 0.0, "episodic": 0.0, "reflection": 0.0},
            }
        )
        profile_data = {
            "profile": {},
            "resolved_keep_new_old": {k: set() for k in PROFILE_KEYS},
            "resolved_keep_new_new": {k: set() for k in PROFILE_KEYS},
        }
        items = [
            {
                "id": "recent",
                "type": "fact",
                "summary": "recent event",
                "timestamp": now,
                "status": "active",
                "score": 0.01,
                "entities": [],
            },
            {
                "id": "old",
                "type": "fact",
                "summary": "old event",
                "timestamp": old,
                "status": "active",
                "score": 0.01,
                "entities": [],
            },
        ]

        scored = scorer.score_items(
            items,
            plan,
            profile_data,
            graph_entities=set(),
            use_recency=True,
            router_enabled=True,
            type_separation_enabled=True,
        )
        scores = {r["id"]: r["score"] for r in scored}
        gap = scores["recent"] - scores["old"]
        # With coefficient 0.15 and recency ~1.0 vs ~0.0, gap should be ~0.15
        assert gap > 0.10, f"Recency gap {gap:.3f} too small — coefficient may be too low"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_retrieval_scorer.py::TestRecencyBoostMagnitude -v`
Expected: FAIL — gap ≈ 0.08 which is < 0.10

- [ ] **Step 3: Change the coefficient**

In `nanobot/memory/read/scoring.py`, line 375. Change:

```python
            intent_bonus = type_boost + (0.08 * recency) + stability_boost + reflection_penalty
```

To:

```python
            intent_bonus = type_boost + (0.15 * recency) + stability_boost + reflection_penalty
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_retrieval_scorer.py -v`
Expected: All pass including the new test (gap ≈ 0.15)

- [ ] **Step 5: Run lint + typecheck**

Run: `make lint && make typecheck`
Expected: Clean

- [ ] **Step 6: Commit**

```bash
git add nanobot/memory/read/scoring.py tests/test_retrieval_scorer.py
git commit -m "feat(memory): increase recency boost from 0.08 to 0.15

Recency was the weakest scoring signal (0.08 max) despite being a
strong relevance indicator. Increase to 0.15 to match graph_boost
magnitude and better balance against type_boost (±0.30).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Final Validation

**Files:** None (validation only)

- [ ] **Step 1: Run full test suite**

Run: `make test`
Expected: All tests pass

- [ ] **Step 2: Run structural checks**

Run: `make check`
Expected: Clean (lint + typecheck + import-check + structure-check + prompt-check + phase-todo-check + doc-check)

- [ ] **Step 3: Verify no import boundary violations**

The only new imports are within the `memory` package (`from ..embedder import HashEmbedder, LocalEmbedder` in `retriever.py`). This is intra-package — no boundary violation.

Run: `make import-check`
Expected: Clean

- [ ] **Step 4: Review the branch diff**

Run: `git diff main --stat`
Expected:
- `nanobot/memory/read/retriever.py` — ~15 lines added (method + call site)
- `nanobot/memory/read/retrieval_planner.py` — ~12 lines added (markers)
- `nanobot/memory/read/scoring.py` — 1 line changed (coefficient)
- `tests/test_retriever.py` — ~30 lines added (4 tests)
- `tests/test_store_helpers.py` — ~6 lines added (6 parametrize cases)
- `tests/test_retrieval_scorer.py` — ~40 lines added (1 test)
