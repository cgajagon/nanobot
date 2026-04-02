import pytest

from nanobot.cron.service import CronService
from nanobot.cron.types import CronSchedule


def test_add_job_rejects_unknown_timezone(tmp_path) -> None:
    service = CronService(tmp_path / "cron" / "jobs.json")

    with pytest.raises(ValueError, match="unknown timezone 'America/Vancovuer'"):
        service.add_job(
            name="tz typo",
            schedule=CronSchedule(kind="cron", expr="0 9 * * *", tz="America/Vancovuer"),
            message="hello",
        )

    assert service.list_jobs(include_disabled=True) == []


def test_add_job_accepts_valid_timezone(tmp_path) -> None:
    service = CronService(tmp_path / "cron" / "jobs.json")

    job = service.add_job(
        name="tz ok",
        schedule=CronSchedule(kind="cron", expr="0 9 * * *", tz="America/Vancouver"),
        message="hello",
    )

    assert job.schedule.tz == "America/Vancouver"
    assert job.state.next_run_at_ms is not None


def test_update_job_changes_schedule(tmp_path) -> None:
    service = CronService(tmp_path / "cron" / "jobs.json")
    job = service.add_job(
        name="original",
        schedule=CronSchedule(kind="every", every_ms=60000),
        message="hello",
    )
    original_id = job.id

    updated = service.update_job(
        original_id,
        schedule=CronSchedule(kind="every", every_ms=120000),
    )
    assert updated is not None
    assert updated.id == original_id
    assert updated.schedule.every_ms == 120000
    assert updated.state.next_run_at_ms is not None


def test_update_job_changes_message(tmp_path) -> None:
    service = CronService(tmp_path / "cron" / "jobs.json")
    job = service.add_job(
        name="original",
        schedule=CronSchedule(kind="every", every_ms=60000),
        message="old prompt",
    )

    updated = service.update_job(job.id, message="new prompt")
    assert updated is not None
    assert updated.payload.message == "new prompt"
    assert updated.name == "original"  # name unchanged


def test_update_job_changes_name(tmp_path) -> None:
    service = CronService(tmp_path / "cron" / "jobs.json")
    job = service.add_job(
        name="original",
        schedule=CronSchedule(kind="every", every_ms=60000),
        message="hello",
    )

    updated = service.update_job(job.id, name="renamed")
    assert updated is not None
    assert updated.name == "renamed"


def test_update_job_not_found(tmp_path) -> None:
    service = CronService(tmp_path / "cron" / "jobs.json")
    assert service.update_job("nonexistent") is None


def test_update_job_no_changes(tmp_path) -> None:
    service = CronService(tmp_path / "cron" / "jobs.json")
    job = service.add_job(
        name="original",
        schedule=CronSchedule(kind="every", every_ms=60000),
        message="hello",
    )

    updated = service.update_job(job.id)
    assert updated is not None
    assert updated.name == "original"
    assert updated.payload.message == "hello"
