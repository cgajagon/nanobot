# Retrieval Scoring Fixes — Design Spec

> Date: 2026-04-01
> Status: Approved
> Scope: Two surgical fixes to the memory retrieval scoring pipeline

## Problem

The retrieval pipeline has two scoring issues identified during the 2026-04-01 memory system code review:

1. **RRF score not carried to final ranking.** `_fuse_results()` stores the RRF score in `item["_rrf_score"]` but never sets `item["score"]`. Downstream, `score_items()` reads `item.get("score", 0.0)` as `base_score`, which is always 0.0. RRF controls candidate selection but contributes nothing to final ranking — the entire ranking is rebuilt from scratch using weaker heuristic signals.

2. **Inconsistent recency decay formulas.** Two different formulas are used in the same pipeline:
   - `RetrievalPlanner.recency_signal`: `exp(-ln(2) * age / half_life)` — true half-life (value = 0.50 at `half_life` days)
   - `CompositeReranker._recency_score`: `exp(-age / half_life)` — simple exponential (value ≈ 0.37 at `half_life` days)
   Both use "half_life" as the parameter name but produce different curves.

## Fix 1: Wire RRF Score Into base_score

### Change

In `retriever.py` `_fuse_results()`, after computing `scores[eid]` and storing it in `item["_rrf_score"]`, also set `item["score"] = scores[eid]`.

This makes the fused score visible to:
- `scoring.py` line 280: `base_score = float(item.get("score", 0.0))` — now reads the RRF score
- `CompositeReranker` via `item["retrieval_reason"]["score"]` — now has a real BM25-proxy signal

### Fallback path

Items from the `read_events` fallback (when both vector and FTS return empty) get no RRF score. Their `base_score` stays 0.0. This is correct — fallback items have no relevance signal.

### File

`nanobot/memory/read/retriever.py` — 1 line added in `_fuse_results`, at the same point where `_rrf_score` is set.

## Fix 2: Unify Recency Decay to True Half-Life

### Change

In `reranker.py`, change `_recency_score` from:
```python
return math.exp(-days_old / half_life)
```
to:
```python
return math.exp(-math.log(2) * days_old / half_life)
```

The `_RECENCY_HALF_LIFE_DAYS = 30` constant stays the same. Items 30 days old now score 0.50 (true half-life) instead of 0.37 (simple exponential).

### File

`nanobot/memory/ranking/reranker.py` — 1 line changed in `_recency_score`. `math` is already imported.

## Tests

1. **RRF score propagation** — test in `test_retriever.py`: items returned from `_fuse_results` have a `"score"` key set to a positive value.
2. **Recency true half-life** — test in `test_reranker.py`: `_recency_score` returns 0.5 (±0.01) when `days_old == half_life`.
3. **Contract test** — `base_score > 0` for items that went through RRF fusion in the full retrieval pipeline.

## Risk

Low. Both are isolated changes in the read path. No write path, schema, or API impact. The scoring formula structure is unchanged — only the inputs to it become more meaningful.

## Non-Goals

- Changing the RRF constants (k=60, vector_weight=0.7, fts_weight=0.3)
- Changing the additive scoring formula weights
- Changing the reranker alpha blending
- Removing the `_rrf_score` key (kept for observability)
