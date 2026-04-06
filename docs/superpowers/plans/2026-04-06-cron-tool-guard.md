# Cron Tool Service Guard — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `CronTool` refuse write operations (`add`, `enable`, `update`) when the cron service isn't running, and remove unnecessary `CronService` creation from the CLI agent.

**Architecture:** Two commits: (1) add a service-running guard in `CronTool.execute()` with tests, (2) remove the unused `CronService` from the CLI agent as cleanup. The guard is the primary defense; the CLI cleanup is belt-and-suspenders.

**Tech Stack:** Python 3.10+, pytest, ruff, mypy

---

### Task 1: Add Service-Running Guard to CronTool

**Files:**
- Modify: `nanobot/tools/builtin/cron.py:62-82` — add guard in `execute()`
- Modify: `tests/test_cron_tool.py` — add tests for the guard, add `_running` to `_FakeCron`

- [ ] **Step 1: Write test for guard blocking add when service not running**

In `tests/test_cron_tool.py`, add a new test class after `TestCronToolExecute`:

```python
class TestCronToolServiceGuard:
    """Write operations must fail when the cron service is not running."""

    async def test_add_blocked_when_not_running(self) -> None:
        svc = MagicMock()
        svc._running = False
        tool = CronTool(cron_service=svc)
        tool.set_context(channel="test", chat_id="123")
        result = await tool.execute(action="add", message="hello", every_seconds=60)
        assert not result.success
        assert "not available" in result.output.lower()

    async def test_enable_blocked_when_not_running(self) -> None:
        svc = MagicMock()
        svc._running = False
        tool = CronTool(cron_service=svc)
        result = await tool.execute(action="enable", job_id="j1")
        assert not result.success
        assert "not available" in result.output.lower()

    async def test_update_blocked_when_not_running(self) -> None:
        svc = MagicMock()
        svc._running = False
        tool = CronTool(cron_service=svc)
        result = await tool.execute(action="update", job_id="j1", message="new")
        assert not result.success
        assert "not available" in result.output.lower()

    async def test_list_allowed_when_not_running(self) -> None:
        svc = MagicMock()
        svc._running = False
        svc.list_jobs.return_value = []
        tool = CronTool(cron_service=svc)
        result = await tool.execute(action="list")
        assert result.success

    async def test_remove_allowed_when_not_running(self) -> None:
        svc = MagicMock()
        svc._running = False
        svc.remove_job.return_value = True
        tool = CronTool(cron_service=svc)
        result = await tool.execute(action="remove", job_id="j1")
        assert result.success

    async def test_disable_allowed_when_not_running(self) -> None:
        svc = MagicMock()
        svc._running = False
        svc.enable_job.return_value = MagicMock(id="j1", name="test")
        tool = CronTool(cron_service=svc)
        result = await tool.execute(action="disable", job_id="j1")
        assert result.success

    async def test_add_allowed_when_running(self) -> None:
        svc = MagicMock()
        svc._running = True
        mock_job = MagicMock(id="j1")
        mock_job.name = "test"
        svc.add_job.return_value = mock_job
        tool = CronTool(cron_service=svc)
        tool.set_context(channel="test", chat_id="123")
        result = await tool.execute(action="add", message="hello", every_seconds=60)
        assert result.success
```

- [ ] **Step 2: Add `_running = True` to `_FakeCron`**

In `tests/test_cron_tool.py`, add `_running` to the `_FakeCron` class so existing
integration tests keep passing:

```python
class _FakeCron:
    def __init__(self) -> None:
        self.jobs: dict[str, SimpleNamespace] = {}
        self._running = True
```

- [ ] **Step 3: Run tests to verify new tests fail**

Run: `pytest tests/test_cron_tool.py::TestCronToolServiceGuard -v`
Expected: FAIL — `execute()` doesn't check `_running` yet, so `add` succeeds when it
shouldn't.

- [ ] **Step 4: Add the guard to `CronTool.execute()`**

In `nanobot/tools/builtin/cron.py`, add the guard at the start of `execute()`,
after parsing `action`:

```python
async def execute(self, **kwargs: Any) -> ToolResult:
    action: str = kwargs.pop("action")
    message: str = kwargs.pop("message", "")
    every_seconds: int | None = kwargs.pop("every_seconds", None)
    cron_expr: str | None = kwargs.pop("cron_expr", None)
    tz: str | None = kwargs.pop("tz", None)
    at: str | None = kwargs.pop("at", None)
    job_id: str | None = kwargs.pop("job_id", None)
    if action in ("add", "enable", "update") and not self._cron._running:
        return ToolResult.fail(
            "Cron scheduling is not available in this mode. "
            "Use `nanobot gateway` to run the agent with cron support."
        )
    if action == "add":
        return self._add_job(message, every_seconds, cron_expr, tz, at)
    elif action == "list":
        return self._list_jobs()
    elif action == "remove":
        return self._remove_job(job_id)
    elif action == "enable":
        return self._enable_job(job_id, enabled=True)
    elif action == "disable":
        return self._enable_job(job_id, enabled=False)
    elif action == "update":
        return self._update_job(job_id, message, every_seconds, cron_expr, tz, at)
    return ToolResult.fail(f"Unknown action: {action}")
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_cron_tool.py -v`
Expected: All tests pass (new guard tests + existing tests).

- [ ] **Step 6: Run lint and typecheck**

Run: `make lint && make typecheck`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add nanobot/tools/builtin/cron.py tests/test_cron_tool.py
git commit -m "fix(cron): guard write operations when cron service is not running

CronTool.execute() now returns ToolResult.fail for add, enable, and
update actions when the cron service has not been started. List,
remove, and disable still work regardless. This prevents the agent
from promising scheduled jobs that would never fire in CLI mode."
```

---

### Task 2: Remove Unused CronService from CLI Agent

**Files:**
- Modify: `nanobot/cli/agent.py:165,178-180,196` — remove CronService import, creation, and kwarg

- [ ] **Step 1: Remove CronService from CLI agent**

In `nanobot/cli/agent.py`:

Delete the import (line 165):
```python
from nanobot.cron.service import CronService
```

Delete lines 178-180:
```python
    # Create cron service for tool usage (no callback needed for CLI unless running)
    cron_store_path = get_data_dir() / "cron" / "jobs.json"
    cron = CronService(cron_store_path)
```

Delete `cron_service=cron,` from `build_agent()` call (line 196).

Also remove `get_data_dir` from the import on line 164 if it becomes unused. Check
whether any other code in the function uses `get_data_dir` — if not, remove it.

- [ ] **Step 2: Run lint and typecheck**

Run: `make lint && make typecheck`
Expected: PASS — no references to removed symbols.

- [ ] **Step 3: Run tests**

Run: `make test`
Expected: All tests pass.

- [ ] **Step 4: Verify no stale references**

Run:
```bash
grep -n "CronService\|cron_service\|get_data_dir" nanobot/cli/agent.py
```
Expected: Zero matches (unless `get_data_dir` is used elsewhere in the file).

- [ ] **Step 5: Commit**

```bash
git add nanobot/cli/agent.py
git commit -m "chore(cli): remove unused CronService from CLI agent

The CLI agent is ephemeral — cron jobs would never fire. The CronTool
guard (previous commit) handles this at the tool level, but removing
the unnecessary service creation is cleaner. The cron tool is still
registered (via build_agent with cron_service=None) but write
operations return a clear error."
```

---

### Task 3: Final Verification

- [ ] **Step 1: Run full check suite**

Run: `make check`
Expected: All checks pass.

- [ ] **Step 2: Run full test suite**

Run: `make test`
Expected: All tests pass.

- [ ] **Step 3: Verify guard works end-to-end**

Run:
```bash
grep -n "_running" nanobot/tools/builtin/cron.py tests/test_cron_tool.py
```
Expected: Guard in `cron.py`, `_running = True` in `_FakeCron`, and test assertions.
