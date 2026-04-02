# ONNX Reranker Graceful Degradation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make ONNX Runtime a gracefully-degrading optional dependency so that a missing or broken `onnxruntime` installation falls back to `CompositeReranker` instead of crashing 230+ tests.

**Architecture:** Move all heavy imports (`onnxruntime`, `numpy`, `tokenizers`) from module-level into the existing `_ensure_model()` lazy loader, matching the `LocalEmbedder` pattern in `embedder.py`. Make `available` property honest (try to init, return `False` on failure). In `store.py`, check `available` after construction and fall back to `CompositeReranker`. Move `onnxruntime` from required to optional dependency in `pyproject.toml`.

**Tech Stack:** Python 3.10+, pytest, onnxruntime (optional), tokenizers (optional), numpy (optional)

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `nanobot/memory/ranking/onnx_reranker.py` | Modify | Move top-level imports into lazy init, fix `available` property |
| `nanobot/memory/store.py` | Modify | Add `available` check + fallback after ONNX reranker construction |
| `pyproject.toml` | Modify | Move `onnxruntime`, `tokenizers` to optional `[reranker]` extra |
| `tests/test_onnx_reranker.py` | Modify | Guard top-level import, add test for unavailable ONNX |
| `tests/test_memory_helper_wave5.py` | Modify | Guard top-level import of `OnnxCrossEncoderReranker` |

---

### Task 1: Guard ONNX imports in `onnx_reranker.py` — write failing test

**Files:**
- Test: `tests/test_onnx_reranker.py`

- [ ] **Step 1: Add test for `available` returning `False` when ONNX is missing**

Add this test to `tests/test_onnx_reranker.py`:

```python
def test_available_false_when_onnx_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """available must return False when onnxruntime cannot be imported."""
    import nanobot.memory.ranking.onnx_reranker as mod

    monkeypatch.setattr(mod, "_ort", None)
    reranker = OnnxCrossEncoderReranker()
    assert reranker.available is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_onnx_reranker.py::test_available_false_when_onnx_missing -v`
Expected: FAIL — `_ort` attribute does not exist yet, and `available` returns hardcoded `True`.

- [ ] **Step 3: Commit failing test**

```
test(memory): add test for ONNX reranker unavailable fallback
```

---

### Task 2: Move heavy imports into lazy init in `onnx_reranker.py`

**Files:**
- Modify: `nanobot/memory/ranking/onnx_reranker.py`

- [ ] **Step 1: Replace top-level imports with lazy module-level try/except**

Replace lines 14-17:

```python
import numpy as np
import onnxruntime as ort
from loguru import logger
from tokenizers import Tokenizer
```

With:

```python
from loguru import logger

try:
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer

    _ort = ort
except (ImportError, OSError):  # crash-barrier: ONNX/numpy/tokenizers may be absent
    np = None  # type: ignore[assignment]
    _ort = None
    Tokenizer = None  # type: ignore[assignment,misc]
```

This matches the graceful degradation pattern from `embedder.py` — the module is always importable, but heavy deps may be `None`.

- [ ] **Step 2: Fix the `available` property to be honest**

Replace lines 43-46:

```python
@property
def available(self) -> bool:
    """Always *True* — onnxruntime is a mandatory dependency."""
    return True
```

With:

```python
@property
def available(self) -> bool:
    """Whether onnxruntime loaded successfully and model can be initialized."""
    if _ort is None:
        return False
    return self._ensure_model()
```

- [ ] **Step 3: Fix the type annotation on `self._session`**

Replace line 35:

```python
self._session: ort.InferenceSession | None = None
```

With:

```python
self._session: Any = None
```

(Since `ort` may be `None` at import time, `ort.InferenceSession` would fail as a type annotation at runtime. Use `Any` — this is the same pattern `embedder.py` uses at line 84.)

- [ ] **Step 4: Guard `_ensure_model` against missing runtime**

At the top of `_ensure_model()`, add an early return:

```python
def _ensure_model(self) -> bool:
    """Load model and tokenizer, downloading if necessary. Returns *True* on success."""
    if _ort is None:
        return False
    if self._session is not None:
        return True
    # ... rest unchanged
```

- [ ] **Step 5: Guard `rerank` against missing numpy**

At the top of `rerank()`, after the `if not items` check, the existing `if not self._ensure_model(): return items` on line 136 already handles this — when `_ort is None`, `_ensure_model()` returns `False`, and `rerank()` returns items unchanged. No change needed here.

- [ ] **Step 6: Run the failing test from Task 1**

Run: `pytest tests/test_onnx_reranker.py::test_available_false_when_onnx_missing -v`
Expected: PASS

- [ ] **Step 7: Run `make lint && make typecheck`**

Expected: PASS

- [ ] **Step 8: Commit**

```
fix(memory): defer ONNX imports to lazy init for graceful degradation
```

---

### Task 3: Add fallback in `store.py` when ONNX reranker is unavailable

**Files:**
- Modify: `nanobot/memory/store.py:139-150`

- [ ] **Step 1: Write a contract test for the fallback**

Add to `tests/contract/test_memory_wiring.py`:

```python
def test_onnx_unavailable_falls_back_to_composite(tmp_path: Path) -> None:
    """When ONNX reranker reports unavailable, store uses CompositeReranker."""
    import nanobot.memory.ranking.onnx_reranker as onnx_mod
    from nanobot.memory.ranking.reranker import CompositeReranker

    original = onnx_mod._ort
    try:
        onnx_mod._ort = None  # simulate missing onnxruntime
        store = MemoryStore(tmp_path)
        assert isinstance(store._reranker, CompositeReranker)
    finally:
        onnx_mod._ort = original
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/contract/test_memory_wiring.py::test_onnx_unavailable_falls_back_to_composite -v`
Expected: FAIL — `store._reranker` is an `OnnxCrossEncoderReranker` whose `available` returns `False`, but `store.py` doesn't check `available`.

- [ ] **Step 3: Add the fallback check in `store.py`**

Replace lines 139-150:

```python
# Cross-encoder re-ranker (Step 7)
reranker_model = self._memory_config.reranker.model.strip()
reranker_alpha = self._memory_config.reranker.alpha
self._reranker: Reranker
if reranker_model.startswith("onnx:"):
    from .ranking.onnx_reranker import OnnxCrossEncoderReranker

    self._reranker = OnnxCrossEncoderReranker(
        model_name=reranker_model.split(":", 1)[1], alpha=reranker_alpha
    )
else:
    self._reranker = CompositeReranker(alpha=reranker_alpha)
```

With:

```python
# Cross-encoder re-ranker.
reranker_model = self._memory_config.reranker.model.strip()
reranker_alpha = self._memory_config.reranker.alpha
self._reranker: Reranker = CompositeReranker(alpha=reranker_alpha)
if reranker_model.startswith("onnx:"):
    from .ranking.onnx_reranker import OnnxCrossEncoderReranker

    candidate = OnnxCrossEncoderReranker(
        model_name=reranker_model.split(":", 1)[1], alpha=reranker_alpha
    )
    if candidate.available:
        self._reranker = candidate
    else:
        logger.warning("ONNX reranker unavailable, using composite fallback")
```

This starts with a safe default and only upgrades to ONNX if it's actually available.

- [ ] **Step 4: Add `logger` import if not already present**

Check if `store.py` already imports `logger`. If not, add near the top:

```python
from loguru import logger
```

- [ ] **Step 5: Run the contract test**

Run: `pytest tests/contract/test_memory_wiring.py::test_onnx_unavailable_falls_back_to_composite -v`
Expected: PASS

- [ ] **Step 6: Run `make lint && make typecheck`**

Expected: PASS

- [ ] **Step 7: Commit**

```
fix(memory): fall back to CompositeReranker when ONNX unavailable
```

---

### Task 4: Move `onnxruntime` and `tokenizers` to optional dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Move deps from required to optional**

In `pyproject.toml`, remove these two lines from `dependencies` (lines 47-48):

```toml
    "onnxruntime>=1.17.0,<2.0.0",
    "tokenizers>=0.15.0,<1.0.0",
```

Add a new optional group after the existing ones (after line 62):

```toml
reranker = [
    "onnxruntime>=1.17.0,<2.0.0",
    "tokenizers>=0.15.0,<1.0.0",
]
```

- [ ] **Step 2: Update `dev` extra to include reranker**

In the `dev` optional group, add `"nanobot-ai[reranker]"` so dev installs still get ONNX:

Actually — don't. The dev extra installs test/lint tools. ONNX is a runtime optional feature. Developers who want ONNX install with `pip install -e ".[reranker]"`. The whole point is that the system works without it.

- [ ] **Step 3: Verify `numpy` dependency status**

`numpy` is a transitive dependency of `onnxruntime` — when onnxruntime is installed, numpy comes with it. It is NOT used elsewhere in the codebase (confirmed by grep). No separate entry needed.

- [ ] **Step 4: Commit**

```
chore: move onnxruntime and tokenizers to optional [reranker] extra
```

---

### Task 5: Guard test file imports that directly reference ONNX

**Files:**
- Modify: `tests/test_onnx_reranker.py:10`
- Modify: `tests/test_memory_helper_wave5.py:8`

- [ ] **Step 1: Guard the import in `test_onnx_reranker.py`**

Replace line 10:

```python
from nanobot.memory.ranking.onnx_reranker import OnnxCrossEncoderReranker
```

With:

```python
pytest.importorskip("onnxruntime", reason="onnxruntime not installed")

from nanobot.memory.ranking.onnx_reranker import OnnxCrossEncoderReranker
```

This skips the entire test file when onnxruntime is not installed, rather than erroring at collection time.

Wait — this won't work because `OnnxCrossEncoderReranker` is now always importable (the guard is inside the module). The import will succeed. The tests that need a real ONNX runtime should use `pytest.importorskip` to skip when there's no functional ONNX.

Actually, since `onnx_reranker.py` now guards its imports, the `from nanobot.memory.ranking.onnx_reranker import OnnxCrossEncoderReranker` import will ALWAYS succeed. The tests that exercise actual ONNX inference need to skip when `_ort is None`.

Replace line 10:

```python
from nanobot.memory.ranking.onnx_reranker import OnnxCrossEncoderReranker
```

With:

```python
from nanobot.memory.ranking.onnx_reranker import OnnxCrossEncoderReranker, _ort
```

Then add a module-level skip for tests that require a working ONNX runtime. The existing tests mock `_ensure_model` and `_session`, so they don't actually need onnxruntime. Check if `numpy` is used directly — yes, line 8: `import numpy as np`. Guard that:

Replace line 8:

```python
import numpy as np
```

With:

```python
np = pytest.importorskip("numpy", reason="numpy not installed")
```

- [ ] **Step 2: Guard the import in `test_memory_helper_wave5.py`**

Replace line 8:

```python
from nanobot.memory.ranking.onnx_reranker import OnnxCrossEncoderReranker
```

With:

```python
from nanobot.memory.ranking.onnx_reranker import OnnxCrossEncoderReranker
```

This now succeeds (module is importable). No change needed — the module-level guard in `onnx_reranker.py` makes it always importable.

But check: does `test_memory_helper_wave5.py` use `numpy` directly? If so, guard that import with `pytest.importorskip`.

- [ ] **Step 3: Run the previously-failing test files**

Run: `pytest tests/test_onnx_reranker.py tests/test_memory_helper_wave5.py tests/test_reranker.py -v`
Expected: Tests either PASS or SKIP (no collection ERRORs)

- [ ] **Step 4: Run the full non-integration suite**

Run: `pytest tests/ --ignore=tests/integration -q`
Expected: 0 errors, 0 failures from ONNX. Tests that need ONNX either pass (if installed and working) or skip.

- [ ] **Step 5: Run `make lint && make typecheck`**

Expected: PASS

- [ ] **Step 6: Commit**

```
test(memory): guard ONNX test imports for optional dependency
```

---

### Task 6: Final verification

- [ ] **Step 1: Run `make check`**

Expected: All structural checks pass.

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/ --ignore=tests/integration -q`
Expected: 0 errors, 0 collection failures. Tests that need ONNX runtime either pass or skip.

- [ ] **Step 3: Verify graceful degradation message**

Run: `python -c "from nanobot.memory.store import MemoryStore; import tempfile, pathlib; MemoryStore(pathlib.Path(tempfile.mkdtemp()))"`
Expected: If ONNX is broken, you see a warning log: "ONNX reranker unavailable, using composite fallback". No crash.
