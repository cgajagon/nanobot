# Recency Scoring Analysis — Nanobot Memory Retrieval

**Date:** 2026-04-01  
**Team Lead Request:** Thorough analysis of recency coefficient (0.08) in memory retrieval scoring for implementation planning.

---

## 1. Exact Location of Recency Coefficient

### Primary Location: `nanobot/memory/read/scoring.py:375`

```python
intent_bonus = type_boost + (0.08 * recency) + stability_boost + reflection_penalty
item["score"] = base_score + adjustment + intent_bonus + g_boost
```

**Line 375** — the coefficient **0.08** is hardcoded inline. It is **not a constant** and **not configurable** via `MemoryConfig`.

---

## 2. Full Scoring Formula (Complete Decomposition)

### Formula Components (All Additive)

```
final_score = base_score + adjustment + intent_bonus + g_boost
```

### Breaking Down Each Component:

#### A. `base_score` — RRF Fusion Score
- **Source:** Reciprocal Rank Fusion of vector + FTS5 results
- **Computed in:** `MemoryRetriever._retrieve_unified()` 
- **RRF formula:** `score = 0.7 * vector_rrf + 0.3 * fts_rrf` where `rrf = 1 / (60 + rank)`
- **Typical range:** **0.0 to ~0.02** (RRF is normalized and compressed)
  - Max theoretical: `1 / 60 ≈ 0.0167` for top result
  - Most scores cluster between 0.001–0.01

#### B. `adjustment` — Profile-Aware Adjustments (Conditional)
- **Only applied when:** `use_recency=True` AND event type in `_FIELD_BY_EVENT_TYPE`
- **Computed in:** `score_items()` lines 290–338
- **Components:**
  - **resolved_keep_new_old penalty:** -0.18 (demote old values after conflict resolution)
  - **resolved_keep_new_new boost:** +0.12 (boost new values)
  - **stale_profile_penalty:** -0.08 (demote stale profile entries)
  - **conflicted_profile_penalty:** -0.05 (demote ongoing conflicts)
  - **semantic_superseded_penalty:** -0.20 (demote superseded events)
- **Typical range:** **-0.20 to +0.12** (single penalty/boost applied per item)

#### C. `intent_bonus` — Intent-Driven Scoring
- **Formula:** `type_boost + (0.08 * recency) + stability_boost + reflection_penalty`

##### C1. `type_boost` — Memory Type Preference Per Intent
- **Source:** `RetrievalPlanner.retrieval_policy()` lines 170–228
- **Typical ranges by intent:**
  - `semantic` boost: -0.12 to +0.30 (varies by intent)
  - `episodic` boost: -0.16 to +0.22 (varies by intent)
  - `reflection` boost: -0.20 to +0.20 (varies by intent)
- **Examples:**
  - `fact_lookup`: semantic=+0.18, episodic=-0.05, reflection=-0.12
  - `debug_history`: semantic=-0.04, episodic=+0.22, reflection=-0.10
  - `constraints_lookup`: semantic=+0.24, episodic=-0.10, reflection=-0.14
  - `reflection`: semantic=+0.03, episodic=-0.03, reflection=+0.20
- **Applied conditionally:** Only if `router_enabled=True` and `type_separation_enabled=True`

##### C2. `0.08 * recency` — **THE RECENCY COMPONENT** (HARDCODED)
- **Recency signal:** Exponential decay from `RetrievalPlanner.recency_signal()` lines 318–330
  - Formula: `decay(age_days) = exp(-ln(2) * age_days / half_life_days)`
  - **Typical range:** 0.0 to 1.0
    - At age = 0 days: ~1.0 (maximum)
    - At age = half_life: ~0.5
    - At age = 3 × half_life: ~0.125
    - At age = 10 × half_life: ~0.001
  - `half_life_days` varies by intent (21 to 365 days)
    - `fact_lookup`: 120 days
    - `debug_history`: 21 days
    - `planning`: 45 days
    - `constraints_lookup`: 180 days
    - `rollout_status`: 365 days
- **Contribution:** `0.08 * recency` → **0.0 to 0.08** (8% max boost)
- **Applied conditionally:** Only if `use_recency=True` and `router_enabled=True`

##### C3. `stability_boost` — Confidence in Belief Stability
- **Defined:** Lines 39–43, `_STABILITY_BOOST` dict
- **Values:**
  - `"high"`: +0.03 (very stable facts)
  - `"medium"`: +0.01 (balanced)
  - `"low"`: -0.02 (uncertain/volatile)
- **Applied conditionally:** Only if `router_enabled=True`

##### C4. `reflection_penalty` — Reflection Safety Gate
- **Value:** -0.06 (fixed penalty)
- **Applied:** Only if `use_recency=True` AND `memory_type=="reflection"`
- **Purpose:** Lower confidence in reflection memories (require evidence)
- **Applied conditionally:** Only if `router_enabled=True`

#### D. `g_boost` — Graph Entity Matching Bonus
- **Value:** 0.15 (fixed) or 0.0
- **Applied:** If `graph_entities` (query entity set) is not empty AND item shares entities with query
- **Typical range:** **0.0 or 0.15**

---

## 3. Typical Magnitude Ranges Summary

| Component | Min | Max | Typical | Notes |
|-----------|-----|-----|---------|-------|
| **base_score** (RRF) | 0.0 | ~0.02 | 0.005–0.015 | Compressed by RRF |
| **adjustment** | -0.20 | +0.12 | 0 (no change) | Only one applied per item |
| **type_boost** | -0.16 | +0.30 | 0.05–0.10 | Intent-dependent |
| **recency** (0.08×signal) | 0.0 | **0.08** | 0.02–0.05 | **HARDCODED COEFFICIENT** |
| **stability_boost** | -0.02 | +0.03 | 0.01 | Most events "medium" |
| **reflection_penalty** | -0.06 | 0 | 0 | Only on reflection events |
| **g_boost** | 0.0 | 0.15 | 0 (rare) | Only if graph match |
| **TOTAL intent_bonus** | ~-0.17 | ~0.50 | 0.05–0.15 | Sum of C1–C4 |
| **FINAL SCORE** | 0.0 | ~0.67 | 0.10–0.25 | Sum of all components |

---

## 4. All Tests Covering Scoring Formula

### Unit Tests

#### `tests/test_retrieval_scorer.py` (228 lines)

1. **`TestLoadProfileScoringData::test_empty_profile`** (line 64–69)
   - Verifies conflict extraction from profile is empty on blank profile
   - No scoring assertion

2. **`TestLoadProfileScoringData::test_resolved_keep_new_extracted`** (line 71–86)
   - Verifies conflict records are extracted correctly
   - No scoring assertion

3. **`TestFilterItems::test_general_passes_all`** (line 92–100)
   - Verifies filtering allows all items on "general" intent
   - No scoring assertion

4. **`TestFilterItems::test_focus_task_decision`** (line 102–119)
   - Verifies routing hint filtering works (task vs. fact)
   - No scoring assertion

5. **`TestScoreItems::test_type_boost_increases_score`** (line 125–173) ⭐
   - **DIRECTLY TESTS SCORING FORMULA**
   - Creates two items: semantic (type_boost=0.1) vs episodic (no boost)
   - Asserts `semantic_score > episodic_score`
   - Does NOT check exact score values — only relative ordering
   - Does NOT test recency coefficient
   - **Configurable:** `use_recency=True`, `router_enabled=True`, `type_separation_enabled=True`

6. **`TestScoreItems::test_graph_entity_boost`** (line 175–204) ⭐
   - **DIRECTLY TESTS SCORING FORMULA**
   - Creates item with entity "alice" in graph_entities
   - Asserts `score > 0.5` (baseline)
   - Does NOT check exact g_boost value (0.15) — only that boost occurs
   - Does NOT test recency coefficient

7. **`TestRerankItems::test_enabled_calls_reranker`** (line 210–214)
   - Verifies reranker is called when enabled
   - No scoring assertion

8. **`TestRerankItems::test_disabled_passthrough`** (line 216–221)
   - Verifies reranker is skipped when disabled
   - No scoring assertion

9. **`TestRerankItems::test_empty_items_passthrough`** (line 223–227)
   - Verifies empty list passthrough
   - No scoring assertion

#### `tests/test_retriever.py` (300+ lines)

Mocks RetrievalScorer but does not inspect score values directly.

#### `tests/test_coverage_push_wave6.py` (400+ lines)

Recency-related test:
- **Line 352–386:** `test_store_query_hint_status_and_recency_helpers()`
  - Tests `RetrievalPlanner.recency_signal()` directly
  - Verifies empty timestamp returns 0.0
  - Verifies zero half_life_days returns 0.0
  - Does NOT test coefficient (0.08) — only the exponential decay function
  - Does NOT test score impact in full formula

### Contract Tests

#### `tests/contract/test_memory_contracts.py` (100+ lines)

- No tests of scoring formula
- Tests append/retrieve roundtrip behavior
- Does not inspect score values

#### `tests/contract/test_typed_boundaries.py`

- No scoring tests

### Integration Tests (If Any)

No integration tests found that validate score values or ordering.

### Memory Eval Cases

**`case/memory_eval_cases.json`** (41 test queries)

- **ALL 41 CASES** use advisory (non-gating) benchmark
- **Assertions:** `expected_any`, `expected_topics`, `expected_memory_types`, `expected_status_any`, `required_min_hits`
- **NO ASSERTIONS on score values or ordering** — only on content presence
- Example:
  ```json
  {
    "query": "What are the user's response style preferences?",
    "expected_any": ["prefer","concise","bullet"],
    "expected_any_mode": "normalized",
    "expected_topics": ["user_preference"],
    "expected_memory_types": ["semantic"],
    "top_k": 6
  }
  ```
- **Impact of recency change:** NONE — eval cases do not depend on score values

---

## 5. Impact Assessment: Changing 0.08 to 0.15

### Will Break Existing Tests?

**Summary: NO** ❌ No test will break if recency changes from 0.08 to 0.15.

#### Why:

1. **`test_type_boost_increases_score`** — Compares relative scores only. Changing recency coefficient will not flip the ordering (semantic vs episodic still differ by type_boost).

2. **`test_graph_entity_boost`** — Only checks `score > 0.5` (baseline). Changing recency increases absolute score, still satisfies check.

3. **`recency_signal` tests** — Test the decay function `exp(-ln(2) * age / half_life)`, not the coefficient.

4. **`memory_eval_cases.json`** — No score assertions; only presence of expected content.

### Will Change Behavior?

**Summary: YES** ✅ Recency will be weighted **87.5% higher** (0.15 vs 0.08).

#### Quantified Impact:

- **Old:** `0.08 * recency` contributes max +0.08 to final score
- **New:** `0.15 * recency` contributes max +0.15 to final score
- **Multiplier:** 0.15 / 0.08 = **1.875x**

#### Ranking Impact:

For two items with identical `base_score`, `adjustment`, `type_boost`, `stability_boost`, `g_boost`:

**Old formula (0.08):**
- Recent (age 0d): +0.08
- Old (age 180d, half_life=120): +0.08 × 0.355 ≈ +0.028

**New formula (0.15):**
- Recent (age 0d): +0.15
- Old (age 180d, half_life=120): +0.15 × 0.355 ≈ +0.053

**Ranking effect:** Newer items will rank higher in queries where recency is enabled (`use_recency=True`).

---

## 6. Configuration Status

### Is 0.08 Configurable?

**NO** ❌

- Hardcoded at line 375 of `nanobot/memory/read/scoring.py`
- **Not** in `MemoryConfig` (checked `nanobot/config/memory.py`)
- **Not** in `RetrievalPlan` policy dict
- **Not** a constant (e.g., `RECENCY_WEIGHT = 0.08`)

### What IS Configurable?

1. **`use_recency`** — Boolean flag (passed to `score_items()`)
   - If `False`, recency = 0.0 (coefficient multiplied by 0, result is 0)
   - This disables recency entirely

2. **`half_life_days`** — Per-intent setting in `RetrievalPlanner.retrieval_policy()`
   - Lines 173–228 define half_life for each intent
   - `half_life_days` is in `RetrievalPlan.policy` dict
   - Controls **shape** of decay, not **weight**
   - Example: `fact_lookup` uses 120 days; `debug_history` uses 21 days

3. **`router_enabled`** and `type_separation_enabled`** — Enable/disable scoring features
   - When `router_enabled=False`, all intent-based scoring (type_boost, recency, stability) is zeroed

---

## 7. Summary Table: All Scoring Knobs

| Knob | Type | Typical Range | Configurable? | Location |
|------|------|---------------|---------------|----------|
| Base score (RRF vector/FTS weights) | 0.7 / 0.3 | Fixed | No | `retriever.py` |
| `type_boost` per intent | -0.16 to +0.30 | Per intent | No (static) | `retrieval_planner.py` |
| **Recency coefficient** | **0.08** | **Hardcoded** | **No** | **`scoring.py:375`** |
| `half_life_days` per intent | 21 to 365 | Per intent | No (static) | `retrieval_planner.py` |
| `stability_boost` | -0.02 to +0.03 | Fixed dict | No | `scoring.py:39–43` |
| Graph entity boost | 0.15 | Fixed | No | `scoring.py:266` |
| Profile penalty/boost | -0.20 to +0.12 | Fixed | No | `scoring.py` |
| Reflection penalty | -0.06 | Fixed | No | `scoring.py:348` |
| **`use_recency` flag** | Boolean | True/False | **YES** | `score_items()` param |
| **`router_enabled` flag** | Boolean | True/False | **YES** | `score_items()` param |
| **`type_separation_enabled` flag** | Boolean | True/False | **YES** | `score_items()` param |

---

## 8. Recommendations for Implementation

### If Task #8 (H3: Increase recency boost from 0.08 to 0.15) Proceeds:

1. **Change location:** Line 375 of `nanobot/memory/read/scoring.py`
   - `(0.08 * recency)` → `(0.15 * recency)`

2. **Consider extracting to constant:**
   ```python
   _RECENCY_COEFFICIENT: float = 0.15  # Moved from hardcoded 0.08
   # ... later in score_items():
   intent_bonus = type_boost + (_RECENCY_COEFFICIENT * recency) + stability_boost + reflection_penalty
   ```

3. **No test changes required:**
   - Existing unit tests will pass without modification
   - eval cases unaffected

4. **Consider feature-gating (optional):**
   - If rollout needed, add `MemoryConfig.recency_coefficient: float = 0.15`
   - Update `scoring.py` to read from config
   - Allows A/B testing

5. **Update documentation:**
   - `cognitive-architecture.md` section on "Recency Signal" (if present)
   - Add comment at line 375 explaining choice

6. **Verify with integration test (recommended but not blocking):**
   ```python
   def test_recency_coefficient_changes_ranking():
       # Recent item (age 0d) vs old item (age 180d, half_life=120)
       # With coefficient=0.15, recent should rank higher
       # Coefficient is read from line 375
   ```

---

## 9. Code References (Exact Line Numbers)

| Concern | File | Lines | Note |
|---------|------|-------|------|
| **Recency coefficient (0.08)** | `nanobot/memory/read/scoring.py` | **375** | **HARDCODED** |
| Recency signal function | `nanobot/memory/read/retrieval_planner.py` | 318–330 | Exponential decay |
| Half-life per intent | `nanobot/memory/read/retrieval_planner.py` | 172–228 | Policy dict |
| Type boost per intent | `nanobot/memory/read/retrieval_planner.py` | 172–228 | Policy dict |
| Stability boost dict | `nanobot/memory/read/scoring.py` | 39–43 | `_STABILITY_BOOST` |
| Profile adjustments | `nanobot/memory/read/scoring.py` | 290–338 | Penalties/boosts |
| Graph boost | `nanobot/memory/read/scoring.py` | 266 | Value: 0.15 |
| Reflection penalty | `nanobot/memory/read/scoring.py` | 348 | Value: -0.06 |
| **Test: type_boost** | `tests/test_retrieval_scorer.py` | 125–173 | Relative ordering only |
| **Test: graph_boost** | `tests/test_retrieval_scorer.py` | 175–204 | Presence only |
| **Test: recency_signal** | `tests/test_coverage_push_wave6.py` | 352–386 | Function only, not coefficient |
| Eval cases | `case/memory_eval_cases.json` | All | No score assertions |

---

## Conclusion

The recency coefficient (0.08) is:
- **Hardcoded** at `scoring.py:375`
- **Not configurable** via `MemoryConfig` or `RetrievalPlan`
- **Not tested directly** in unit or contract tests
- **Not asserted in eval cases** (which use advisory benchmarks)

Changing it to 0.15 will:
- ✅ Pass all existing tests (no assertions on absolute scores)
- ✅ Change ranking behavior (newer items ranked higher, 1.875x multiplier)
- ❌ Require no test updates
- ⚠️ Should be documented and ideally feature-gated for rollout

