# Cron Timer Logging — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add logging to `CronService._arm_timer()` and `_on_timer()` so timer arming, firing, and skipping are visible in the log.

**Architecture:** Add `logger.debug()` calls at three points in `cron/service.py`: when the timer is armed (with delay), when `_on_timer()` fires (with due job count), and when arming is skipped (no jobs or not running). No new files, no API changes.

**Tech Stack:** Python 3.10+, loguru, pytest

**Key conventions:**
- Existing cron logs use format `"Cron: <action>"` with `{}` placeholders (loguru style)
- Timer details are `logger.debug()` (not `info`) — `_arm_timer()` is called 7+ times per session
- loguru does NOT propagate to pytest `caplog` by default — tests MUST use the `propagate_loguru_to_caplog` fixture from `tests/conftest.py`

---

### Task 1: Add Cron Timer Logging

**Files:**
- Modify: `nanobot/cron/service.py:209-244` — add logging to `_arm_timer()` and `_on_timer()`
- Modify: `tests/test_cron_service_extended.py` — add tests verifying log output

- [ ] **Step 1: Write test for timer-armed logging**

In `tests/test_cron_service_extended.py`, add at the end of the file:

```python
@pytest.mark.usefixtures("propagate_loguru_to_caplog")
async def test_arm_timer_logs_next_wake(tmp_path, caplog) -> None:
    """When a job is added and the service is running, _arm_timer logs the delay."""
    import logging

    service = CronService(tmp_path / "cron" / "jobs.json")
    with caplog.at_level(logging.DEBUG, logger="nanobot.cron.service"):
        await service.start()
        service.add_job(
            name="log test",
            schedule=CronSchedule(kind="every", every_ms=300_000),
            message="hello",
        )
        service.stop()

    armed_logs = [r for r in caplog.records if "armed" in r.message.lower()]
    assert len(armed_logs) >= 1, f"Expected 'armed' log, got: {[r.message for r in caplog.records]}"
```

- [ ] **Step 2: Write test for timer-skipped logging**

In `tests/test_cron_service_extended.py`, add:

```python
@pytest.mark.usefixtures("propagate_loguru_to_caplog")
async def test_arm_timer_logs_no_wake(tmp_path, caplog) -> None:
    """When no enabled jobs exist, _arm_timer logs that no timer is needed."""
    import logging

    service = CronService(tmp_path / "cron" / "jobs.json")
    with caplog.at_level(logging.DEBUG, logger="nanobot.cron.service"):
        await service.start()
        service.stop()

    no_wake_logs = [r for r in caplog.records if "no wake" in r.message.lower()]
    assert len(no_wake_logs) >= 1, f"Expected 'no wake' log, got: {[r.message for r in caplog.records]}"
```

- [ ] **Step 3: Write test for on_timer logging**

In `tests/test_cron_service_extended.py`, add:

```python
@pytest.mark.usefixtures("propagate_loguru_to_caplog")
async def test_on_timer_logs_due_jobs(tmp_path, caplog) -> None:
    """When _on_timer fires, it logs how many jobs are due."""
    import logging

    service = CronService(tmp_path / "cron" / "jobs.json")
    await service.start()
    job = service.add_job(
        name="due job",
        schedule=CronSchedule(kind="every", every_ms=1),
        message="hello",
    )
    # Force the job to be due by setting next_run to the past
    job.state.next_run_at_ms = 1

    with caplog.at_level(logging.DEBUG, logger="nanobot.cron.service"):
        await service._on_timer()

    service.stop()

    timer_logs = [r for r in caplog.records if "due" in r.message.lower()]
    assert len(timer_logs) >= 1, f"Expected 'due' log, got: {[r.message for r in caplog.records]}"
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `pytest tests/test_cron_service_extended.py -v -k "log"`
Expected: FAIL — no log messages contain "armed", "no wake", or "due".

- [ ] **Step 5: Add logging to `_arm_timer()`**

In `nanobot/cron/service.py`, replace `_arm_timer()` (lines 209-226) with:

```python
def _arm_timer(self) -> None:
    """Schedule the next timer tick."""
    if self._timer_task:
        self._timer_task.cancel()

    next_wake = self._get_next_wake_ms()
    if not next_wake or not self._running:
        logger.debug("Cron: no wake scheduled (running={}, next_wake={})", self._running, next_wake)
        return

    delay_ms = max(0, next_wake - _now_ms())
    delay_s = delay_ms / 1000
    logger.debug("Cron: timer armed, next wake in {:.1f}s", delay_s)

    async def tick() -> None:
        await asyncio.sleep(delay_s)
        if self._running:
            await self._on_timer()

    self._timer_task = asyncio.create_task(tick())
```

- [ ] **Step 6: Add logging to `_on_timer()`**

In `nanobot/cron/service.py`, replace `_on_timer()` (lines 228-244) with:

```python
async def _on_timer(self) -> None:
    """Handle timer tick - run due jobs."""
    if not self._store:
        return

    now = _now_ms()
    due_jobs = [
        j
        for j in self._store.jobs
        if j.enabled and j.state.next_run_at_ms and now >= j.state.next_run_at_ms
    ]

    logger.debug("Cron: timer fired, {} job(s) due", len(due_jobs))

    for job in due_jobs:
        await self._execute_job(job)

    self._save_store()
    self._arm_timer()
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_cron_service_extended.py -v -k "log"`
Expected: PASS — all three log tests pass.

- [ ] **Step 8: Run full test suite and lint**

Run: `make lint && make typecheck`
Expected: PASS

Run: `make test`
Expected: All tests pass.

- [ ] **Step 9: Commit**

```bash
git add nanobot/cron/service.py tests/test_cron_service_extended.py
git commit -m "fix(cron): add logging to timer arming and firing

_arm_timer() now logs the computed delay when armed and the reason
when skipped (not running or no jobs). _on_timer() logs the count
of due jobs when it fires. This makes debugging missed cron
executions possible without adding debug logging and restarting."
```
