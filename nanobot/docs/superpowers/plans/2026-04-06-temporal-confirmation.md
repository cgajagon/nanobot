# Temporal Confirmation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add temporal intelligence to the memory subsystem — `last_confirmed` with echo detection, TTL enforcement at retrieval, and stability-aware decay.

**Architecture:** Three independent changes to the memory subsystem: (1) schema migration + merge logic for `last_confirmed`, (2) TTL filter in retrieval pipeline, (3) stability-aware half-life in scoring. All changes stay within `memory/` — no boundary crossings.

**Tech Stack:** Python 3.10+, SQLite, pytest, ruff, mypy

**Spec:** `docs/superpowers/specs/2026-04-06-temporal-confirmation-design.md`

**Branch:** `feat/temporal-confirmation`

**Important conventions to follow:**
- Every module starts with `from __future__ import annotations`
- Type hints on all function signatures
- `make lint && make typecheck` after every edit
- `make check` before every commit
- Code review before every commit (dispatch code-reviewer subagent)

---

## File Map

### Modified files
| File | Change |
|------|--------|
| `nanobot/memory/db/connection.py` | ALTER TABLE migration for `last_confirmed` column |
| `nanobot/memory/event.py` | Add `last_confirmed` and `source_role` fields to `MemoryEvent` |
| `nanobot/memory/write/micro_extractor.py` | Add `source_role` to `_MICRO_EXTRACT_TOOL` schema |
| `nanobot/memory/write/coercion.py` | Preserve `source_role` through coercion pipeline |
| `nanobot/memory/write/dedup.py` | Conditionally bump `last_confirmed` in `merge_events()` |
| `nanobot/memory/consolidation_pipeline.py` | Set `source_role="consolidation"` on extracted events |
| `nanobot/memory/read/retriever.py` | Add TTL expiry filter after fusion |
| `nanobot/memory/read/scoring.py` | Use `last_confirmed` for recency + stability-aware half-life |
| `tests/test_event_deduplicator.py` | Add merge `last_confirmed` tests |
| `tests/test_retrieval_scorer.py` | Add stability half-life and last_confirmed tests |
| `tests/contract/test_memory_wiring.py` | Add `last_confirmed` column migration test |

### New test files
| File | Purpose |
|------|---------|
| `tests/test_ttl_filter.py` | Unit tests for TTL expiry filter |

---

## Task 1: Add `last_confirmed` and `source_role` fields to MemoryEvent

**Files:**
- Modify: `nanobot/memory/event.py`
- Modify: `nanobot/memory/db/connection.py`
- Modify: `tests/contract/test_memory_wiring.py`

- [ ] **Step 1: Write the failing contract test**

Add to `tests/contract/test_memory_wiring.py`:

```python
def test_last_confirmed_column_exists(tmp_path):
    """last_confirmed column is created by schema migration."""
    store = _make_store(tmp_path)
    cursor = store.db.connection.execute("PRAGMA table_info(events)")
    columns = {row["name"] for row in cursor.fetchall()}
    assert "last_confirmed" in columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/contract/test_memory_wiring.py::test_last_confirmed_column_exists -v`
Expected: FAIL — column does not exist

- [ ] **Step 3: Add `last_confirmed` and `source_role` to MemoryEvent**

In `nanobot/memory/event.py`, add two fields to the `MemoryEvent` class after the `supersedes_at` field (around line 140):

```python
    # Temporal confirmation
    last_confirmed: str = ""
    source_role: str = ""  # "user" | "assistant" | "tool" | "consolidation" | ""
```

- [ ] **Step 4: Add schema migration to `connection.py`**

In `nanobot/memory/db/connection.py`, add a `_migrate_schema` method after `_init_schema`:

```python
    def _migrate_schema(self) -> None:
        """Apply incremental schema migrations for columns added after initial release."""
        try:
            self._conn.execute("ALTER TABLE events ADD COLUMN last_confirmed TEXT")
        except sqlite3.OperationalError:
            pass  # column already exists
```

Call it from `__init__`, right after `self._init_schema()`:

```python
        self._init_schema()
        self._migrate_schema()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/contract/test_memory_wiring.py::test_last_confirmed_column_exists -v`
Expected: PASS

- [ ] **Step 6: Run lint and typecheck**

Run: `make lint && make typecheck`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add nanobot/memory/event.py nanobot/memory/db/connection.py tests/contract/test_memory_wiring.py
git commit -m "feat(memory): add last_confirmed column and source_role field to MemoryEvent"
```

---

## Task 2: Add `source_role` to micro-extraction tool schema

**Files:**
- Modify: `nanobot/memory/write/micro_extractor.py`

- [ ] **Step 1: Write the failing contract test**

Add to `tests/contract/test_memory_wiring.py`:

```python
def test_source_role_in_micro_extract_schema():
    """_MICRO_EXTRACT_TOOL schema includes source_role field."""
    from nanobot.memory.write.micro_extractor import _MICRO_EXTRACT_TOOL

    props = _MICRO_EXTRACT_TOOL[0]["function"]["parameters"]["properties"]["events"]["items"]["properties"]
    assert "source_role" in props
    assert props["source_role"]["type"] == "string"
    assert "user" in props["source_role"]["enum"]
    assert "assistant" in props["source_role"]["enum"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/contract/test_memory_wiring.py::test_source_role_in_micro_extract_schema -v`
Expected: FAIL — `source_role` not in properties

- [ ] **Step 3: Add `source_role` to the schema**

In `nanobot/memory/write/micro_extractor.py`, add to the `_MICRO_EXTRACT_TOOL` event item properties (after the `"confidence"` property, around line 63):

```python
                                "source_role": {
                                    "type": "string",
                                    "enum": ["user", "assistant"],
                                    "description": (
                                        "Whether the fact originated from the user's "
                                        "message or the assistant's response"
                                    ),
                                },
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/contract/test_memory_wiring.py::test_source_role_in_micro_extract_schema -v`
Expected: PASS

- [ ] **Step 5: Run lint and typecheck**

Run: `make lint && make typecheck`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add nanobot/memory/write/micro_extractor.py tests/contract/test_memory_wiring.py
git commit -m "feat(memory): add source_role to micro-extraction tool schema"
```

---

## Task 3: Preserve `source_role` through coercion and consolidation

**Files:**
- Modify: `nanobot/memory/write/coercion.py`
- Modify: `nanobot/memory/consolidation_pipeline.py`

- [ ] **Step 1: Write tests**

Add to `tests/test_event_deduplicator.py` (or a relevant test file that tests coercion):

```python
class TestSourceRolePreservation:
    def test_coercion_preserves_source_role(self) -> None:
        from nanobot.memory.write.classification import EventClassifier
        from nanobot.memory.write.coercion import EventCoercer

        classifier = EventClassifier()
        coercer = EventCoercer(classifier)
        raw = {
            "type": "fact",
            "summary": "User uses Python",
            "source_role": "user",
        }
        event = coercer.coerce_event(raw)
        assert event is not None
        assert event.source_role == "user"

    def test_coercion_defaults_source_role_empty(self) -> None:
        from nanobot.memory.write.classification import EventClassifier
        from nanobot.memory.write.coercion import EventCoercer

        classifier = EventClassifier()
        coercer = EventCoercer(classifier)
        raw = {"type": "fact", "summary": "User uses Python"}
        event = coercer.coerce_event(raw)
        assert event is not None
        assert event.source_role == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_event_deduplicator.py::TestSourceRolePreservation -v`
Expected: FAIL — `source_role` not passed through coercion

- [ ] **Step 3: Preserve `source_role` in coercion**

In `nanobot/memory/write/coercion.py`, in the `coerce_event` method, extract `source_role` from raw input and include it in the dict passed to `MemoryEvent.from_dict()`.

Add after the `source` line (around line 79):

```python
        source_role = str(raw.get("source_role", "")).strip().lower()
        if source_role not in {"user", "assistant", "tool", "consolidation"}:
            source_role = ""
```

Then include it in the return dict (around line 128, in the dict passed to `MemoryEvent.from_dict()`):

```python
                "source_role": source_role,
```

- [ ] **Step 4: Set `source_role="consolidation"` in consolidation pipeline**

In `nanobot/memory/consolidation_pipeline.py`, in `_consolidate_single_tool()`, after events are coerced (around line 227, after the `if event:` block):

```python
                if event:
                    event.source_role = "consolidation"
                    events.append(event)
```

Also tag the heuristic fallback events (around line 241-243):

```python
            events, profile_updates = self._extractor.heuristic_extract_events(
                old_messages, source_start=source_start
            )
            for event in events:
                event.source_role = "consolidation"
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_event_deduplicator.py::TestSourceRolePreservation -v && make lint && make typecheck`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add nanobot/memory/write/coercion.py nanobot/memory/consolidation_pipeline.py tests/test_event_deduplicator.py
git commit -m "feat(memory): preserve source_role through coercion, tag consolidation events"
```

---

## Task 4: Conditionally bump `last_confirmed` in merge

**Files:**
- Modify: `nanobot/memory/write/dedup.py`
- Modify: `tests/test_event_deduplicator.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_event_deduplicator.py`:

```python
class TestLastConfirmedInMerge:
    def test_merge_bumps_last_confirmed_for_user_source(self) -> None:
        d = _make_dedup()
        base = {
            "type": "fact",
            "summary": "User uses Python",
            "entities": ["Python"],
            "last_confirmed": "2025-01-01T00:00:00Z",
        }
        incoming = {
            "type": "fact",
            "summary": "User uses Python",
            "entities": ["Python"],
            "source_role": "user",
        }
        merged = d.merge_events(base, incoming, similarity=0.9)
        # last_confirmed should be bumped (recent timestamp, not the old one)
        assert merged["last_confirmed"] > "2025-01-01T00:00:00Z"

    def test_merge_skips_last_confirmed_for_assistant_echo(self) -> None:
        d = _make_dedup()
        base = {
            "type": "fact",
            "summary": "User uses Python",
            "entities": ["Python"],
            "last_confirmed": "2025-06-01T00:00:00Z",
        }
        incoming = {
            "type": "fact",
            "summary": "User uses Python",
            "entities": ["Python"],
            "source_role": "assistant",
        }
        merged = d.merge_events(base, incoming, similarity=0.9)
        # last_confirmed should NOT be bumped — echo
        assert merged["last_confirmed"] == "2025-06-01T00:00:00Z"

    def test_merge_bumps_last_confirmed_for_consolidation(self) -> None:
        d = _make_dedup()
        base = {
            "type": "fact",
            "summary": "User uses Python",
            "entities": ["Python"],
            "last_confirmed": "2025-01-01T00:00:00Z",
        }
        incoming = {
            "type": "fact",
            "summary": "User uses Python",
            "entities": ["Python"],
            "source_role": "consolidation",
        }
        merged = d.merge_events(base, incoming, similarity=0.9)
        assert merged["last_confirmed"] > "2025-01-01T00:00:00Z"

    def test_merge_default_empty_source_role_is_genuine(self) -> None:
        d = _make_dedup()
        base = {
            "type": "fact",
            "summary": "User uses Python",
            "entities": ["Python"],
            "last_confirmed": "2025-01-01T00:00:00Z",
        }
        incoming = {
            "type": "fact",
            "summary": "User uses Python",
            "entities": ["Python"],
            "source_role": "",
        }
        merged = d.merge_events(base, incoming, similarity=0.9)
        # Empty source_role defaults to genuine
        assert merged["last_confirmed"] > "2025-01-01T00:00:00Z"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_event_deduplicator.py::TestLastConfirmedInMerge -v`
Expected: FAIL — `last_confirmed` not set in merge output

- [ ] **Step 3: Update `merge_events()` to handle `last_confirmed`**

In `nanobot/memory/write/dedup.py`, in `merge_events()`, add after the `merged["last_merged_at"]` line (around line 252):

```python
        # Temporal confirmation: bump last_confirmed only for genuine sources
        incoming_source_role = str(candidate.get("source_role", "")).strip().lower()
        if incoming_source_role == "assistant":
            # Echo — preserve existing last_confirmed, don't bump
            merged["last_confirmed"] = str(
                canonical.get("last_confirmed") or canonical.get("timestamp", "")
            )
        else:
            # Genuine re-observation (user, tool, consolidation, or unknown)
            merged["last_confirmed"] = _utc_now_iso()
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_event_deduplicator.py::TestLastConfirmedInMerge -v && make lint && make typecheck`
Expected: PASS

- [ ] **Step 5: Run full dedup test suite**

Run: `python -m pytest tests/test_event_deduplicator.py -v`
Expected: all PASS (no regressions)

- [ ] **Step 6: Commit**

```bash
git add nanobot/memory/write/dedup.py tests/test_event_deduplicator.py
git commit -m "feat(memory): conditionally bump last_confirmed based on source_role in merge"
```

---

## Task 5: Add TTL expiry filter to retrieval

**Files:**
- Modify: `nanobot/memory/read/retriever.py`
- Create: `tests/test_ttl_filter.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ttl_filter.py`:

```python
"""Tests for TTL expiry filter in retrieval."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from nanobot.memory.read.retriever import filter_expired


class TestFilterExpired:
    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def test_no_ttl_passes_through(self) -> None:
        items = [{"id": "1", "summary": "No TTL", "timestamp": "2020-01-01T00:00:00Z"}]
        result = filter_expired(items, self._now())
        assert len(result) == 1

    def test_expired_event_excluded(self) -> None:
        old_ts = (self._now() - timedelta(days=60)).isoformat()
        items = [{"id": "1", "summary": "Old task", "timestamp": old_ts, "ttl_days": 30}]
        result = filter_expired(items, self._now())
        assert len(result) == 0

    def test_fresh_event_with_ttl_included(self) -> None:
        recent_ts = (self._now() - timedelta(days=5)).isoformat()
        items = [{"id": "1", "summary": "Fresh task", "timestamp": recent_ts, "ttl_days": 30}]
        result = filter_expired(items, self._now())
        assert len(result) == 1

    def test_ttl_uses_last_confirmed_when_available(self) -> None:
        old_ts = (self._now() - timedelta(days=60)).isoformat()
        recent_confirmed = (self._now() - timedelta(days=5)).isoformat()
        items = [
            {
                "id": "1",
                "summary": "Old but confirmed",
                "timestamp": old_ts,
                "last_confirmed": recent_confirmed,
                "ttl_days": 30,
            }
        ]
        result = filter_expired(items, self._now())
        # Should be included: last_confirmed is 5 days ago, TTL is 30 days
        assert len(result) == 1

    def test_invalid_ttl_ignored(self) -> None:
        old_ts = (self._now() - timedelta(days=60)).isoformat()
        items = [
            {"id": "1", "summary": "Bad TTL", "timestamp": old_ts, "ttl_days": -5},
            {"id": "2", "summary": "Zero TTL", "timestamp": old_ts, "ttl_days": 0},
            {"id": "3", "summary": "String TTL", "timestamp": old_ts, "ttl_days": "thirty"},
        ]
        result = filter_expired(items, self._now())
        # All pass through — invalid TTL treated as no TTL
        assert len(result) == 3

    def test_no_timestamp_passes_through(self) -> None:
        items = [{"id": "1", "summary": "No ts", "ttl_days": 30}]
        result = filter_expired(items, self._now())
        assert len(result) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ttl_filter.py -v`
Expected: FAIL — `cannot import name 'filter_expired'`

- [ ] **Step 3: Implement `filter_expired` in retriever**

In `nanobot/memory/read/retriever.py`, add the function (as a module-level function, not a method — it's a pure filter):

```python
from datetime import datetime

from .._text import _to_datetime


def filter_expired(items: list[dict[str, Any]], now: datetime) -> list[dict[str, Any]]:
    """Exclude events whose TTL has expired.

    Uses ``last_confirmed`` (if available) or ``timestamp`` for age calculation.
    Events without ``ttl_days`` or with invalid TTL always pass through.
    """
    filtered: list[dict[str, Any]] = []
    for item in items:
        ttl = item.get("ttl_days")
        if not isinstance(ttl, int) or ttl <= 0:
            filtered.append(item)
            continue
        ts_str = str(item.get("last_confirmed") or item.get("timestamp", ""))
        ts = _to_datetime(ts_str)
        if ts is None:
            filtered.append(item)
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=now.tzinfo)
        age_days = (now - ts).total_seconds() / 86400.0
        if age_days <= ttl:
            filtered.append(item)
    return filtered
```

Add the necessary imports at the top of `retriever.py` if not already present (`datetime` from `datetime`, `_to_datetime` from `.._text`).

- [ ] **Step 4: Wire the filter into the retrieval pipeline**

In `nanobot/memory/read/retriever.py`, in `_retrieve_unified()`, add the TTL filter after step 5 (filter) and before step 6 (score). Find the comment `# 5. Filter` and add after it:

```python
        # 5b. TTL expiry filter
        from datetime import timezone
        filtered = filter_expired(filtered, datetime.now(timezone.utc))
```

(Or import `timezone` at the top of the file alongside existing datetime imports.)

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_ttl_filter.py -v && make lint && make typecheck`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add nanobot/memory/read/retriever.py tests/test_ttl_filter.py
git commit -m "feat(memory): add TTL expiry filter to retrieval pipeline"
```

---

## Task 6: Stability-aware decay and `last_confirmed` in scoring

**Files:**
- Modify: `nanobot/memory/read/scoring.py`
- Modify: `tests/test_retrieval_scorer.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_retrieval_scorer.py`:

```python
class TestStabilityAwareDecay:
    def test_high_stability_slower_decay(self) -> None:
        """High stability fact should decay slower than medium."""
        scorer = _make_scorer()
        plan = _make_plan(policy={"half_life_days": 60.0, "type_boost": {}})
        profile_data = {
            "profile": {},
            "resolved_keep_new_old": {k: set() for k in PROFILE_KEYS},
            "resolved_keep_new_new": {k: set() for k in PROFILE_KEYS},
        }
        old_ts = "2025-01-01T00:00:00Z"
        items_high = [
            {
                "id": "h1",
                "type": "preference",
                "summary": "Prefers dark mode",
                "memory_type": "semantic",
                "timestamp": old_ts,
                "score": 0.5,
                "stability": "high",
                "entities": [],
            }
        ]
        items_low = [
            {
                "id": "l1",
                "type": "task",
                "summary": "Deploy by Friday",
                "memory_type": "episodic",
                "timestamp": old_ts,
                "score": 0.5,
                "stability": "low",
                "entities": [],
            }
        ]
        scored_high = scorer.score_items(
            items_high, plan, profile_data, set(),
            use_recency=True, router_enabled=True, type_separation_enabled=False,
        )
        scored_low = scorer.score_items(
            items_low, plan, profile_data, set(),
            use_recency=True, router_enabled=True, type_separation_enabled=False,
        )
        # High stability should have higher final score due to slower decay
        assert scored_high[0]["score"] > scored_low[0]["score"]

    def test_recency_uses_last_confirmed_over_timestamp(self) -> None:
        """last_confirmed should take precedence over timestamp for recency."""
        scorer = _make_scorer()
        plan = _make_plan(policy={"half_life_days": 60.0, "type_boost": {}})
        profile_data = {
            "profile": {},
            "resolved_keep_new_old": {k: set() for k in PROFILE_KEYS},
            "resolved_keep_new_new": {k: set() for k in PROFILE_KEYS},
        }
        from datetime import datetime, timedelta, timezone
        recent = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        old = "2024-01-01T00:00:00Z"

        items_confirmed = [
            {
                "id": "c1",
                "type": "fact",
                "summary": "Old but confirmed",
                "memory_type": "semantic",
                "timestamp": old,
                "last_confirmed": recent,
                "score": 0.5,
                "stability": "medium",
                "entities": [],
            }
        ]
        items_stale = [
            {
                "id": "s1",
                "type": "fact",
                "summary": "Old and stale",
                "memory_type": "semantic",
                "timestamp": old,
                "score": 0.5,
                "stability": "medium",
                "entities": [],
            }
        ]
        scored_confirmed = scorer.score_items(
            items_confirmed, plan, profile_data, set(),
            use_recency=True, router_enabled=True, type_separation_enabled=False,
        )
        scored_stale = scorer.score_items(
            items_stale, plan, profile_data, set(),
            use_recency=True, router_enabled=True, type_separation_enabled=False,
        )
        # Confirmed recently should score higher
        assert scored_confirmed[0]["score"] > scored_stale[0]["score"]
```

Also add a contract test to `tests/contract/test_memory_wiring.py`:

```python
def test_stability_half_life_constants_defined():
    """_STABILITY_HALF_LIFE has entries for all three stability levels."""
    from nanobot.memory.read.scoring import _STABILITY_HALF_LIFE

    assert "high" in _STABILITY_HALF_LIFE
    assert "medium" in _STABILITY_HALF_LIFE
    assert "low" in _STABILITY_HALF_LIFE
    assert _STABILITY_HALF_LIFE["high"] > _STABILITY_HALF_LIFE["medium"]
    assert _STABILITY_HALF_LIFE["medium"] > _STABILITY_HALF_LIFE["low"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_retrieval_scorer.py::TestStabilityAwareDecay tests/contract/test_memory_wiring.py::test_stability_half_life_constants_defined -v`
Expected: FAIL

- [ ] **Step 3: Add `_STABILITY_HALF_LIFE` constant to scoring.py**

In `nanobot/memory/read/scoring.py`, add near the top with other constants:

```python
_STABILITY_HALF_LIFE: dict[str, float] = {
    "high": 365.0,
    "medium": 90.0,
    "low": 14.0,
}
```

- [ ] **Step 4: Update recency computation in `score_items()`**

In `nanobot/memory/read/scoring.py`, in `score_items()`, find the recency block (around line 350-354):

```python
            if use_recency:
                recency = RetrievalPlanner.recency_signal(
                    str(item.get("timestamp", "")),
                    half_life_days=float(policy.get("half_life_days", 60.0)),
                )
```

Replace with:

```python
            if use_recency:
                recency_ts = str(item.get("last_confirmed") or item.get("timestamp", ""))
                stability = str(item.get("stability", "medium")).strip().lower()
                half_life = _STABILITY_HALF_LIFE.get(stability, 90.0)
                recency = RetrievalPlanner.recency_signal(recency_ts, half_life_days=half_life)
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_retrieval_scorer.py::TestStabilityAwareDecay tests/contract/test_memory_wiring.py::test_stability_half_life_constants_defined -v && make lint && make typecheck`
Expected: PASS

- [ ] **Step 6: Run full test suite**

Run: `python -m pytest tests/ --ignore=tests/integration -x -q`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add nanobot/memory/read/scoring.py tests/test_retrieval_scorer.py tests/contract/test_memory_wiring.py
git commit -m "feat(memory): stability-aware decay half-lives and last_confirmed in scoring"
```

---

## Task 7: Update living docs

**Files:**
- Modify: `.claude/rules/memory-architecture.md`

- [ ] **Step 1: Update temporal handling sections**

In `.claude/rules/memory-architecture.md`, find the "Known Technical Debt" section and add a resolved item for temporal confirmation. Also update the schema section to mention `last_confirmed` column.

Check for any references to "no time-based confidence decay" or "ttl_days... never enforced" and update them.

- [ ] **Step 2: Run `make check`**

Run: `make check`
Expected: PASS (doc-check validates living doc references)

- [ ] **Step 3: Commit**

```bash
git add .claude/rules/memory-architecture.md
git commit -m "docs: update memory-architecture for temporal confirmation"
```

---

## Task 8: Final validation

- [ ] **Step 1: Run `make check`**

Run: `make check`
Expected: all structural checks pass

- [ ] **Step 2: Run full unit tests with coverage**

Run: `make test-cov`
Expected: PASS with >= 85% coverage

- [ ] **Step 3: Verify no regressions in existing scorer/retriever tests**

Run: `python -m pytest tests/test_retrieval_scorer.py tests/test_event_deduplicator.py tests/test_ingester.py -v`
Expected: all PASS
