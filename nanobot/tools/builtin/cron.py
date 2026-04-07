"""Cron tool for scheduling reminders and tasks."""

from __future__ import annotations

from typing import Any, ClassVar

from nanobot.cron.service import CronService
from nanobot.cron.types import CronSchedule
from nanobot.tools.base import Tool, ToolResult


class CronTool(Tool):
    """Tool to schedule reminders and recurring tasks."""

    name = "cron"
    description = "Schedule and manage recurring tasks. Actions: add, list, remove, enable, disable, update, run."
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "list", "remove", "enable", "disable", "update", "run"],
                "description": "Action to perform",
            },
            "message": {
                "type": "string",
                "description": (
                    "The full prompt text the agent will execute when this job fires. "
                    "Include the user's complete instructions verbatim — this is the "
                    "only context the agent receives (for add/update)."
                ),
            },
            "every_seconds": {
                "type": "integer",
                "description": "Interval in seconds (for recurring tasks)",
            },
            "cron_expr": {
                "type": "string",
                "description": "Cron expression like '0 9 * * *' (for scheduled tasks)",
            },
            "tz": {
                "type": "string",
                "description": "IANA timezone for cron expressions (e.g. 'America/Vancouver')",
            },
            "at": {
                "type": "string",
                "description": "ISO datetime for one-time execution (e.g. '2026-02-12T10:30:00')",
            },
            "job_id": {
                "type": "string",
                "description": "Job ID (for remove, enable, disable, update)",
            },
        },
        "required": ["action"],
    }

    def __init__(self, cron_service: CronService):
        self._cron = cron_service
        self._channel = ""
        self._chat_id = ""

    def set_context(
        self, channel: str = "", chat_id: str = "", message_id: str | None = None, **kwargs: Any
    ) -> None:
        """Set the current session context for delivery."""
        self._channel = channel
        self._chat_id = chat_id

    async def execute(self, **kwargs: Any) -> ToolResult:
        action: str = kwargs.pop("action")
        message: str = kwargs.pop("message", "")
        every_seconds: int | None = kwargs.pop("every_seconds", None)
        cron_expr: str | None = kwargs.pop("cron_expr", None)
        tz: str | None = kwargs.pop("tz", None)
        at: str | None = kwargs.pop("at", None)
        job_id: str | None = kwargs.pop("job_id", None)
        if action in ("add", "enable", "update", "run") and not self._cron.is_running:
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
        elif action == "run":
            return await self._run_job(job_id)
        return ToolResult.fail(f"Unknown action: {action}")

    def _add_job(
        self,
        message: str,
        every_seconds: int | None,
        cron_expr: str | None,
        tz: str | None,
        at: str | None,
    ) -> ToolResult:
        if not message:
            return ToolResult.fail("Error: message is required for add")
        if not self._channel or not self._chat_id:
            return ToolResult.fail("Error: no session context (channel/chat_id)")
        if tz and not cron_expr:
            return ToolResult.fail("Error: tz can only be used with cron_expr")
        if tz:
            from zoneinfo import ZoneInfo

            try:
                ZoneInfo(tz)
            except (KeyError, ValueError):
                return ToolResult.fail(f"Error: unknown timezone '{tz}'")

        # Build schedule
        delete_after = False
        if every_seconds:
            schedule = CronSchedule(kind="every", every_ms=every_seconds * 1000)
        elif cron_expr:
            schedule = CronSchedule(kind="cron", expr=cron_expr, tz=tz)
        elif at:
            from datetime import datetime

            dt = datetime.fromisoformat(at)
            at_ms = int(dt.timestamp() * 1000)
            schedule = CronSchedule(kind="at", at_ms=at_ms)
            delete_after = True
        else:
            return ToolResult.fail("Error: either every_seconds, cron_expr, or at is required")

        job = self._cron.add_job(
            name=message[:30],
            schedule=schedule,
            message=message,
            deliver=True,
            channel=self._channel,
            to=self._chat_id,
            delete_after_run=delete_after,
        )
        return ToolResult.ok(f"Created job '{job.name}' (id: {job.id})")

    def _list_jobs(self) -> ToolResult:
        jobs = self._cron.list_jobs(include_disabled=True)
        if not jobs:
            return ToolResult.ok("No scheduled jobs.")
        lines: list[str] = []
        for j in jobs:
            schedule = self._format_schedule(j.schedule)
            status = "enabled" if j.enabled else "disabled"
            last_run = self._format_ts(j.state.last_run_at_ms)
            next_run = self._format_ts(j.state.next_run_at_ms)
            last_status = j.state.last_status or "never run"
            line = (
                f"- {j.name} (id: {j.id})\n"
                f"  Schedule: {schedule} | Status: {status}\n"
                f"  Last run: {last_run} ({last_status}) | Next: {next_run}"
            )
            if j.state.last_error:
                line += f"\n  Error: {j.state.last_error}"
            lines.append(line)
        return ToolResult.ok("Scheduled jobs:\n" + "\n".join(lines))

    @staticmethod
    def _format_ts(ms: int | None) -> str:
        if ms is None:
            return "never"
        from datetime import datetime, timezone

        dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M UTC")

    @staticmethod
    def _format_schedule(schedule: CronSchedule) -> str:
        if schedule.kind == "every":
            seconds = (schedule.every_ms or 0) // 1000
            if seconds >= 3600:
                return f"every {seconds // 3600}h"
            if seconds >= 60:
                return f"every {seconds // 60}m"
            return f"every {seconds}s"
        if schedule.kind == "cron":
            expr = schedule.expr or ""
            return f"{expr} ({schedule.tz})" if schedule.tz else expr
        return "one-time"

    def _enable_job(self, job_id: str | None, *, enabled: bool) -> ToolResult:
        if not job_id:
            return ToolResult.fail("Error: job_id is required for enable/disable")
        job = self._cron.enable_job(job_id, enabled=enabled)
        if job is None:
            return ToolResult.fail(f"Job {job_id} not found")
        state = "enabled" if enabled else "disabled"
        return ToolResult.ok(f"Job '{job.name}' ({job_id}) {state}")

    def _update_job(
        self,
        job_id: str | None,
        message: str,
        every_seconds: int | None,
        cron_expr: str | None,
        tz: str | None,
        at: str | None,
    ) -> ToolResult:
        if not job_id:
            return ToolResult.fail("Error: job_id is required for update")
        if tz and not cron_expr:
            return ToolResult.fail("Error: tz can only be used with cron_expr")
        if tz:
            from zoneinfo import ZoneInfo

            try:
                ZoneInfo(tz)
            except (KeyError, ValueError):
                return ToolResult.fail(f"Error: unknown timezone '{tz}'")

        # Build schedule only if any schedule param is provided
        schedule: CronSchedule | None = None
        if every_seconds:
            schedule = CronSchedule(kind="every", every_ms=every_seconds * 1000)
        elif cron_expr:
            schedule = CronSchedule(kind="cron", expr=cron_expr, tz=tz)
        elif at:
            from datetime import datetime

            dt = datetime.fromisoformat(at)
            at_ms = int(dt.timestamp() * 1000)
            schedule = CronSchedule(kind="at", at_ms=at_ms)

        job = self._cron.update_job(
            job_id,
            name=message[:30] if message else None,
            schedule=schedule,
            message=message or None,
        )
        if job is None:
            return ToolResult.fail(f"Job {job_id} not found")
        return ToolResult.ok(f"Updated job '{job.name}' ({job_id})")

    def _remove_job(self, job_id: str | None) -> ToolResult:
        if not job_id:
            return ToolResult.fail("Error: job_id is required for remove")
        if self._cron.remove_job(job_id):
            return ToolResult.ok(f"Removed job {job_id}")
        return ToolResult.fail(f"Job {job_id} not found")

    async def _run_job(self, job_id: str | None) -> ToolResult:
        if not job_id:
            return ToolResult.fail("Error: job_id is required for run")
        ok = await self._cron.run_job(job_id)
        if not ok:
            return ToolResult.fail(f"Job {job_id} not found or disabled")
        return ToolResult.ok(f"Job {job_id} executed. Check the delivery channel for results.")
