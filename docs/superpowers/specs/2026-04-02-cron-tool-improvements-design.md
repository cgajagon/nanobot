# Cron Tool Improvements Design

> Closes gaps 1-3 from the memory review session's cron analysis.

## Goal

Expose enable/disable, rich job listing, and job update capabilities through the
agent's `cron` tool. All three features have existing service-level support or
straightforward service additions.

## Changes

### Gap 2: Enable/Disable via Tool

**Tool (`cron.py`):** Add `"enable"` and `"disable"` to the `action` enum. Both
require `job_id`. Call `self._cron.enable_job(job_id, enabled=True/False)`.

**Service:** No change needed. `CronService.enable_job()` already exists.

**Tool schema addition:**
```json
"action": {"enum": ["add", "list", "remove", "enable", "disable", "update"]}
```

### Gap 3: Rich List Output

**Tool (`cron.py`):** Replace `_list_jobs` output from:
```
- Daily Report (id: abc123, cron)
```
To:
```
- Daily Report (id: abc123)
  Schedule: 0 9 * * * (America/Vancouver) | Status: enabled
  Last run: 2026-04-02 09:00 (ok) | Next: 2026-04-03 09:00
```

Include: schedule details, enabled state, last run time + status, last error (if any),
next run time. Timestamps formatted as `YYYY-MM-DD HH:MM` UTC.

Also pass `include_disabled=True` so the agent can see paused jobs.

**Service:** No change needed. `CronJob` dataclass already contains all fields.

### Gap 1: Update Action

**Service (`service.py`):** Add `update_job()`:
```python
def update_job(
    self,
    job_id: str,
    *,
    name: str | None = None,
    schedule: CronSchedule | None = None,
    message: str | None = None,
) -> CronJob | None:
```
Finds job by ID. Updates only provided (non-None) fields. If schedule changed,
recomputes `next_run_at_ms`. Persists and re-arms timer. Returns updated job or
None if not found.

**Tool (`cron.py`):** Add `"update"` action. Requires `job_id`. Accepts optional
`message`, `every_seconds`, `cron_expr`, `tz`, `at`. Builds a new `CronSchedule`
only if any schedule param is provided. Passes non-None fields to
`service.update_job()`.

### Skill Doc Update

**`nanobot/skills/cron/SKILL.md`:** Add management section with examples:
```
cron(action="enable", job_id="abc123")
cron(action="disable", job_id="abc123")
cron(action="update", job_id="abc123", cron_expr="0 10 * * *")
cron(action="update", job_id="abc123", message="New prompt for this task")
```

## Files

| File | Change |
|------|--------|
| `nanobot/cron/service.py` | Add `update_job()` method (~25 LOC) |
| `nanobot/tools/builtin/cron.py` | Add enable/disable/update actions, rich list (~60 LOC) |
| `nanobot/skills/cron/SKILL.md` | Document new actions |
| `tests/test_cron_tool.py` | Tests for new actions + rich list format |
| `tests/test_cron_service.py` | Test for `update_job()` |

## Out of Scope

- Delivery target update (Gap 5 — separate concern)
- Error recovery/retry (Gap 6 — separate concern)
- Gateway-only limitation (Gap 4 — architectural)
- CLI commands for update (CLI already has enable; update can be added later)
