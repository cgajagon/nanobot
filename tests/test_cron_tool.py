"""Tests for nanobot.tools.builtin.cron — CronTool."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nanobot.tools.builtin.cron import CronTool


@pytest.fixture()
def cron_tool() -> CronTool:
    svc = MagicMock()
    tool = CronTool(cron_service=svc)
    tool.set_context(channel="test", chat_id="123")
    return tool


class TestCronToolProperties:
    def test_name(self, cron_tool: CronTool):
        assert cron_tool.name == "cron"

    def test_description(self, cron_tool: CronTool):
        assert "schedule" in cron_tool.description.lower()

    def test_parameters(self, cron_tool: CronTool):
        params = cron_tool.parameters
        assert "action" in params["properties"]


class TestCronToolExecute:
    async def test_unknown_action(self, cron_tool: CronTool):
        result = await cron_tool.execute(action="unknown")
        assert not result.success

    async def test_add_missing_message(self, cron_tool: CronTool):
        result = await cron_tool.execute(action="add")
        assert not result.success
        assert "message" in result.output.lower()

    async def test_add_no_context(self):
        svc = MagicMock()
        tool = CronTool(cron_service=svc)
        # No set_context called
        result = await tool.execute(action="add", message="hi")
        assert not result.success
        assert "context" in result.output.lower()

    async def test_add_with_every_seconds(self, cron_tool: CronTool):
        mock_job = MagicMock(name="test", id="j1")
        mock_job.name = "test"
        cron_tool._cron.add_job.return_value = mock_job
        result = await cron_tool.execute(action="add", message="remind me", every_seconds=60)
        assert result.success
        assert "j1" in result.output

    async def test_add_with_cron_expr(self, cron_tool: CronTool):
        mock_job = MagicMock(id="j2")
        mock_job.name = "daily"
        cron_tool._cron.add_job.return_value = mock_job
        result = await cron_tool.execute(action="add", message="daily task", cron_expr="0 9 * * *")
        assert result.success

    async def test_add_with_at(self, cron_tool: CronTool):
        mock_job = MagicMock(id="j3")
        mock_job.name = "once"
        cron_tool._cron.add_job.return_value = mock_job
        result = await cron_tool.execute(action="add", message="one-time", at="2026-12-25T10:00:00")
        assert result.success

    async def test_add_no_schedule_fails(self, cron_tool: CronTool):
        result = await cron_tool.execute(action="add", message="no sched")
        assert not result.success
        assert "required" in result.output.lower()

    async def test_add_tz_without_cron_fails(self, cron_tool: CronTool):
        result = await cron_tool.execute(
            action="add", message="tz test", every_seconds=60, tz="America/Vancouver"
        )
        assert not result.success
        assert "tz" in result.output.lower()

    async def test_add_invalid_tz(self, cron_tool: CronTool):
        result = await cron_tool.execute(
            action="add", message="bad tz", cron_expr="0 9 * * *", tz="Invalid/Zone"
        )
        assert not result.success
        assert "timezone" in result.output.lower()

    async def test_list_empty(self, cron_tool: CronTool):
        cron_tool._cron.list_jobs.return_value = []
        result = await cron_tool.execute(action="list")
        assert result.success
        assert "no" in result.output.lower()

    async def test_list_with_jobs(self, cron_tool: CronTool):
        from nanobot.cron.types import CronJob, CronJobState, CronPayload, CronSchedule

        job = CronJob(
            id="j1",
            name="reminder",
            enabled=True,
            schedule=CronSchedule(kind="every", every_ms=60000),
            payload=CronPayload(message="remind me"),
            state=CronJobState(),
        )
        cron_tool._cron.list_jobs.return_value = [job]
        result = await cron_tool.execute(action="list")
        assert result.success
        assert "reminder" in result.output

    async def test_remove_missing_id(self, cron_tool: CronTool):
        result = await cron_tool.execute(action="remove")
        assert not result.success
        assert "job_id" in result.output.lower()

    async def test_remove_success(self, cron_tool: CronTool):
        cron_tool._cron.remove_job.return_value = True
        result = await cron_tool.execute(action="remove", job_id="j1")
        assert result.success


# ---------------------------------------------------------------------------
# Integration-style tests with a fake in-memory cron service
# ---------------------------------------------------------------------------


class TestCronToolServiceGuard:
    """Guard returns ToolResult.fail for write ops when cron service is not running."""

    async def test_add_blocked_when_not_running(self) -> None:
        svc = MagicMock()
        svc._running = False
        tool = CronTool(cron_service=svc)
        tool.set_context(channel="test", chat_id="123")
        result = await tool.execute(action="add", message="ping", every_seconds=60)
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

    async def test_list_works_when_not_running(self) -> None:
        svc = MagicMock()
        svc._running = False
        svc.list_jobs.return_value = []
        tool = CronTool(cron_service=svc)
        result = await tool.execute(action="list")
        assert result.success

    async def test_remove_works_when_not_running(self) -> None:
        svc = MagicMock()
        svc._running = False
        svc.remove_job.return_value = True
        tool = CronTool(cron_service=svc)
        result = await tool.execute(action="remove", job_id="j1")
        assert result.success

    async def test_disable_works_when_not_running(self) -> None:
        svc = MagicMock()
        svc._running = False
        mock_job = MagicMock(id="j1")
        mock_job.name = "reminder"
        svc.enable_job.return_value = mock_job
        tool = CronTool(cron_service=svc)
        result = await tool.execute(action="disable", job_id="j1")
        assert result.success

    async def test_add_works_when_running(self) -> None:
        svc = MagicMock()
        svc._running = True
        mock_job = MagicMock(id="j1")
        mock_job.name = "test"
        svc.add_job.return_value = mock_job
        tool = CronTool(cron_service=svc)
        tool.set_context(channel="test", chat_id="123")
        result = await tool.execute(action="add", message="ping", every_seconds=60)
        assert result.success


class _FakeCron:
    def __init__(self) -> None:
        self._running = True
        self.jobs: dict[str, SimpleNamespace] = {}

    def add_job(self, **kwargs):
        state = SimpleNamespace(
            last_run_at_ms=None,
            next_run_at_ms=None,
            last_status=None,
            last_error=None,
        )
        job = SimpleNamespace(
            id="job-1",
            name=kwargs["name"],
            schedule=kwargs["schedule"],
            enabled=True,
            state=state,
        )
        self.jobs[job.id] = job
        return job

    def list_jobs(self, include_disabled: bool = False):
        return list(self.jobs.values())

    def remove_job(self, job_id: str) -> bool:
        return self.jobs.pop(job_id, None) is not None


def test_cron_tool_invalid_and_remove_paths() -> None:
    tool = CronTool(_FakeCron())

    bad_msg = tool._add_job(message="", every_seconds=1, cron_expr=None, tz=None, at=None)
    assert not bad_msg.success

    tool.set_context("telegram", "123")
    bad_tz = tool._add_job(message="hello", every_seconds=None, cron_expr=None, tz="UTC", at=None)
    assert not bad_tz.success

    missing_schedule = tool._add_job(
        message="hello", every_seconds=None, cron_expr=None, tz=None, at=None
    )
    assert not missing_schedule.success

    assert not tool._remove_job(None).success
    assert not tool._remove_job("missing").success


def test_cron_tool_add_list_remove_success() -> None:
    tool = CronTool(_FakeCron())
    tool.set_context("telegram", "123")

    created = tool._add_job("hello", every_seconds=10, cron_expr=None, tz=None, at=None)
    assert created.success

    listed = tool._list_jobs()
    assert listed.success
    assert "Scheduled jobs" in listed.output

    removed = tool._remove_job("job-1")
    assert removed.success


class TestCronToolEnableDisable:
    async def test_enable_success(self, cron_tool: CronTool):
        mock_job = MagicMock(id="j1")
        mock_job.name = "reminder"
        cron_tool._cron.enable_job.return_value = mock_job
        result = await cron_tool.execute(action="enable", job_id="j1")
        assert result.success
        assert "enabled" in result.output
        cron_tool._cron.enable_job.assert_called_once_with("j1", enabled=True)

    async def test_disable_success(self, cron_tool: CronTool):
        mock_job = MagicMock(id="j1")
        mock_job.name = "reminder"
        cron_tool._cron.enable_job.return_value = mock_job
        result = await cron_tool.execute(action="disable", job_id="j1")
        assert result.success
        assert "disabled" in result.output
        cron_tool._cron.enable_job.assert_called_once_with("j1", enabled=False)

    async def test_enable_missing_job_id(self, cron_tool: CronTool):
        result = await cron_tool.execute(action="enable")
        assert not result.success
        assert "job_id" in result.output.lower()

    async def test_enable_job_not_found(self, cron_tool: CronTool):
        cron_tool._cron.enable_job.return_value = None
        result = await cron_tool.execute(action="enable", job_id="missing")
        assert not result.success
        assert "not found" in result.output.lower()


class TestCronToolRichList:
    async def test_list_shows_schedule_details(self, cron_tool: CronTool) -> None:
        from nanobot.cron.types import CronJob, CronJobState, CronPayload, CronSchedule

        job = CronJob(
            id="j1",
            name="Daily Report",
            enabled=True,
            schedule=CronSchedule(kind="cron", expr="0 9 * * *", tz="America/Vancouver"),
            payload=CronPayload(message="check emails"),
            state=CronJobState(
                next_run_at_ms=1743984000000,
                last_run_at_ms=1743897600000,
                last_status="ok",
            ),
        )
        cron_tool._cron.list_jobs.return_value = [job]
        result = await cron_tool.execute(action="list")
        assert result.success
        assert "Daily Report" in result.output
        assert "j1" in result.output
        assert "0 9 * * *" in result.output
        assert "enabled" in result.output.lower()
        assert "ok" in result.output.lower()

    async def test_list_shows_error_when_present(self, cron_tool: CronTool) -> None:
        from nanobot.cron.types import CronJob, CronJobState, CronPayload, CronSchedule

        job = CronJob(
            id="j2",
            name="Broken Task",
            enabled=True,
            schedule=CronSchedule(kind="every", every_ms=3600000),
            payload=CronPayload(message="failing task"),
            state=CronJobState(
                last_run_at_ms=1743897600000,
                last_status="error",
                last_error="API timeout",
            ),
        )
        cron_tool._cron.list_jobs.return_value = [job]
        result = await cron_tool.execute(action="list")
        assert result.success
        assert "error" in result.output.lower()
        assert "API timeout" in result.output

    async def test_list_shows_disabled_jobs(self, cron_tool: CronTool) -> None:
        from nanobot.cron.types import CronJob, CronJobState, CronPayload, CronSchedule

        job = CronJob(
            id="j3",
            name="Paused Job",
            enabled=False,
            schedule=CronSchedule(kind="every", every_ms=60000),
            payload=CronPayload(message="paused"),
            state=CronJobState(),
        )
        cron_tool._cron.list_jobs.return_value = [job]
        result = await cron_tool.execute(action="list")
        assert result.success
        assert "disabled" in result.output.lower()


class TestCronToolUpdate:
    async def test_update_schedule(self, cron_tool: CronTool) -> None:
        mock_job = MagicMock(id="j1")
        mock_job.name = "updated"
        cron_tool._cron.update_job.return_value = mock_job
        result = await cron_tool.execute(action="update", job_id="j1", every_seconds=120)
        assert result.success
        assert "updated" in result.output.lower()
        call_kwargs = cron_tool._cron.update_job.call_args
        assert call_kwargs[0][0] == "j1"
        assert call_kwargs[1]["schedule"].every_ms == 120000

    async def test_update_message(self, cron_tool: CronTool) -> None:
        mock_job = MagicMock(id="j1")
        mock_job.name = "updated"
        cron_tool._cron.update_job.return_value = mock_job
        result = await cron_tool.execute(action="update", job_id="j1", message="new prompt")
        assert result.success
        call_kwargs = cron_tool._cron.update_job.call_args
        assert call_kwargs[1]["message"] == "new prompt"

    async def test_update_missing_job_id(self, cron_tool: CronTool) -> None:
        result = await cron_tool.execute(action="update")
        assert not result.success
        assert "job_id" in result.output.lower()

    async def test_update_job_not_found(self, cron_tool: CronTool) -> None:
        cron_tool._cron.update_job.return_value = None
        result = await cron_tool.execute(action="update", job_id="missing", message="new")
        assert not result.success
        assert "not found" in result.output.lower()


async def test_cron_tool_execute_dispatch() -> None:
    tool = CronTool(_FakeCron())
    tool.set_context("telegram", "123")

    out = await tool.execute(action="add", message="ping", every_seconds=1)
    assert out.success

    listed = await tool.execute(action="list")
    assert listed.success

    rm = await tool.execute(action="remove", job_id="job-1")
    assert rm.success

    unknown = await tool.execute(action="wat")
    assert not unknown.success
