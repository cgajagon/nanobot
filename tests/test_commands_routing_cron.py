from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from nanobot.cli.commands import app
from nanobot.config.schema import Config

runner = CliRunner()


def test_routing_trace_branches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = Config()
    cfg.agents.defaults.workspace = str(tmp_path)
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda: cfg)

    missing = runner.invoke(app, ["routing", "trace"])
    assert missing.exit_code == 0
    assert "No legacy routing trace" in missing.stdout
    assert "Langfuse" in missing.stdout

    trace_path = tmp_path / "memory" / "routing_trace.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text("\n", encoding="utf-8")
    empty = runner.invoke(app, ["routing", "trace"])
    assert empty.exit_code == 0
    assert "Trace file is empty" in empty.stdout

    trace_path.write_text(
        "not-json\n"
        + json.dumps(
            {
                "timestamp": "2026-03-11T00:00:00",
                "event": "classify",
                "role": "general",
                "confidence": 0.9,
                "success": True,
                "message": "hi",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ok = runner.invoke(app, ["routing", "trace", "--last", "5"])
    assert ok.exit_code == 0
    assert "Routing Trace" in ok.stdout


def test_routing_metrics_branches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = Config()
    cfg.agents.defaults.workspace = str(tmp_path)
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda: cfg)

    missing = runner.invoke(app, ["routing", "metrics"])
    assert missing.exit_code == 0
    assert "No legacy routing metrics" in missing.stdout
    assert "Langfuse" in missing.stdout

    metrics_path = tmp_path / "memory" / "routing_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("bad-json", encoding="utf-8")
    bad = runner.invoke(app, ["routing", "metrics"])
    assert bad.exit_code == 1
    assert "Failed to read metrics" in bad.stdout

    metrics_path.write_text(
        json.dumps(
            {
                "routing_classifications": 2,
                "routing_delegations": 1,
                "routing_cycles_blocked": 0,
                "routing_classify_latency_sum_ms": 30,
                "routing_classify_latency_max_ms": 20,
                "delegation_latency_sum_ms": 50,
                "delegation_latency_max_ms": 50,
                "role_invocations:general": 2,
                "role_tool_calls:general": 3,
            }
        ),
        encoding="utf-8",
    )
    ok = runner.invoke(app, ["routing", "metrics"])
    assert ok.exit_code == 0
    assert "Routing Metrics" in ok.stdout
    assert "Per-Role Stats" in ok.stdout


def test_cron_run_success_and_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = Config()
    cfg.agents.defaults.workspace = str(tmp_path)

    monkeypatch.setattr("nanobot.config.loader.load_config", lambda: cfg)
    monkeypatch.setattr("nanobot.config.loader.get_data_dir", lambda: tmp_path)
    monkeypatch.setattr("nanobot.cli.cron._make_provider", lambda _cfg: object())

    class _Bus:
        pass

    class _AgentLoop:
        def __init__(self, **kwargs):
            pass

        async def process_direct(self, *args, **kwargs):
            return "cron-response"

    class _Payload:
        message = "job"
        channel = "cli"
        to = "direct"

    class _Job:
        id = "j1"
        payload = _Payload()

    class _CronService:
        should_run = True

        def __init__(self, _path: Path):
            self.on_job = None

        async def run_job(self, _job_id: str, force: bool = False):
            if self.should_run and self.on_job is not None:
                await self.on_job(_Job())
            return self.should_run

    monkeypatch.setattr("nanobot.bus.queue.MessageBus", _Bus)
    monkeypatch.setattr("nanobot.agent.loop.AgentLoop", _AgentLoop)
    monkeypatch.setattr("nanobot.cron.service.CronService", _CronService)

    ok = runner.invoke(app, ["cron", "run", "job-1", "--force"])
    assert ok.exit_code == 0
    assert "Job executed" in ok.stdout

    _CronService.should_run = False
    fail = runner.invoke(app, ["cron", "run", "job-1"])
    assert fail.exit_code == 0
    assert "Failed to run job" in fail.stdout


# ---------------------------------------------------------------------------
# routing dlq
# ---------------------------------------------------------------------------


def test_routing_dlq_no_trace_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """DLQ exits cleanly when no trace file exists."""
    cfg = Config()
    cfg.agents.defaults.workspace = str(tmp_path)
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda: cfg)

    result = runner.invoke(app, ["routing", "dlq"])
    assert result.exit_code == 0
    assert "No routing trace found" in result.stdout


def test_routing_dlq_empty_trace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """DLQ exits cleanly when trace file is empty."""
    cfg = Config()
    cfg.agents.defaults.workspace = str(tmp_path)
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda: cfg)

    trace_path = tmp_path / "memory" / "routing_trace.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text("\n", encoding="utf-8")

    result = runner.invoke(app, ["routing", "dlq"])
    assert result.exit_code == 0
    assert "Trace file is empty" in result.stdout


def test_routing_dlq_no_issues(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """DLQ reports no issues when all entries are healthy."""
    cfg = Config()
    cfg.agents.defaults.workspace = str(tmp_path)
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda: cfg)

    trace_path = tmp_path / "memory" / "routing_trace.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(
        json.dumps(
            {
                "timestamp": "2026-03-11T00:00:00",
                "event": "classify",
                "role": "general",
                "confidence": 0.95,
                "success": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["routing", "dlq"])
    assert result.exit_code == 0
    assert "No routing issues found" in result.stdout


def test_routing_dlq_flags_failures_and_low_confidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """DLQ detects failures, low-confidence, cycle blocks, and depth blocks."""
    cfg = Config()
    cfg.agents.defaults.workspace = str(tmp_path)
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda: cfg)

    trace_path = tmp_path / "memory" / "routing_trace.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    entries = [
        # Failure
        {
            "timestamp": "2026-03-11T00:00:01",
            "event": "delegate",
            "role": "code",
            "success": False,
            "from_role": "general",
        },
        # Low confidence
        {
            "timestamp": "2026-03-11T00:00:02",
            "event": "classify",
            "role": "research",
            "confidence": 0.3,
            "success": True,
        },
        # Cycle block
        {
            "timestamp": "2026-03-11T00:00:03",
            "event": "delegate_cycle_blocked",
            "role": "code",
            "depth": 3,
        },
        # Depth block
        {
            "timestamp": "2026-03-11T00:00:04",
            "event": "delegate_depth_blocked",
            "role": "general",
            "depth": 5,
        },
        # Healthy entry (should not be flagged)
        {
            "timestamp": "2026-03-11T00:00:05",
            "event": "classify",
            "role": "general",
            "confidence": 0.95,
            "success": True,
        },
    ]
    trace_path.write_text(
        "\n".join(json.dumps(e) for e in entries) + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["routing", "dlq", "--threshold", "0.5"])
    assert result.exit_code == 0
    assert "Routing DLQ" in result.stdout
    assert "4 issues" in result.stdout
    # Rich table rendering may split text across columns; check key fragments
    assert "cycle" in result.stdout
    assert "depth" in result.stdout
    assert "failed" in result.stdout
    assert "confidence" in result.stdout


# ---------------------------------------------------------------------------
# routing replay
# ---------------------------------------------------------------------------


def test_routing_replay_invalid_session_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Replay rejects session keys without a colon separator."""
    cfg = Config()
    cfg.agents.defaults.workspace = str(tmp_path)
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda: cfg)

    result = runner.invoke(
        app, ["routing", "replay", "--session", "no-colon", "--role", "code", "--message", "hi"]
    )
    assert result.exit_code == 1
    assert "Invalid session key" in result.stdout


def test_routing_replay_dry_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Replay --dry-run prints preview without executing."""
    cfg = Config()
    cfg.agents.defaults.workspace = str(tmp_path)
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda: cfg)

    result = runner.invoke(
        app,
        [
            "routing",
            "replay",
            "--session",
            "telegram:123",
            "--role",
            "code",
            "--message",
            "hello world",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "code" in result.stdout
    assert "telegram:123" in result.stdout
    assert "hello world" in result.stdout
    assert "Dry run" in result.stdout


def test_routing_replay_no_message_no_history(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Replay without --message fails when session has no user messages."""
    cfg = Config()
    cfg.agents.defaults.workspace = str(tmp_path)
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda: cfg)

    class _FakeSession:
        messages: list[dict] = []

    class _FakeSessionManager:
        def __init__(self, _workspace):
            pass

        def get_or_create(self, _key):
            return _FakeSession()

    monkeypatch.setattr("nanobot.session.manager.SessionManager", _FakeSessionManager)

    result = runner.invoke(
        app, ["routing", "replay", "--session", "telegram:123", "--role", "code"]
    )
    assert result.exit_code == 1
    assert "No user message found" in result.stdout


def test_routing_replay_loads_last_user_message_dry_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Replay loads the last user message from session history for dry-run."""
    cfg = Config()
    cfg.agents.defaults.workspace = str(tmp_path)
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda: cfg)

    class _FakeSession:
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second message"},
        ]

    class _FakeSessionManager:
        def __init__(self, _workspace):
            pass

        def get_or_create(self, _key):
            return _FakeSession()

    monkeypatch.setattr("nanobot.session.manager.SessionManager", _FakeSessionManager)

    result = runner.invoke(
        app,
        [
            "routing",
            "replay",
            "--session",
            "telegram:123",
            "--role",
            "research",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "second message" in result.stdout
    assert "Dry run" in result.stdout


def test_routing_replay_executes_agent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Replay executes the agent loop with the corrected role."""
    cfg = Config()
    cfg.agents.defaults.workspace = str(tmp_path)
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda: cfg)
    monkeypatch.setattr("nanobot.config.loader.get_data_dir", lambda: tmp_path)
    monkeypatch.setattr("nanobot.cli.routing._make_provider", lambda _cfg: object())
    monkeypatch.setattr("nanobot.agent.observability.init_langfuse", lambda _: None)
    monkeypatch.setattr("nanobot.agent.observability.shutdown", lambda: None)

    class _Bus:
        pass

    class _AgentLoop:
        def __init__(self, **kwargs):
            pass

        async def process_direct(self, content, **kwargs):
            return f"replayed: {content}"

        def stop(self):
            pass

        async def close_mcp(self):
            pass

    class _CronService:
        def __init__(self, _path):
            self.on_job = None

    monkeypatch.setattr("nanobot.bus.queue.MessageBus", _Bus)
    monkeypatch.setattr("nanobot.agent.loop.AgentLoop", _AgentLoop)
    monkeypatch.setattr("nanobot.cron.service.CronService", _CronService)

    result = runner.invoke(
        app,
        [
            "routing",
            "replay",
            "--session",
            "telegram:123",
            "--role",
            "code",
            "--message",
            "fix the bug",
        ],
    )
    assert result.exit_code == 0
    assert "replayed: fix the bug" in result.stdout


# ---------------------------------------------------------------------------
# replay-deadletters additional branches
# ---------------------------------------------------------------------------


def test_replay_deadletters_no_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """replay-deadletters exits cleanly when no dead-letter file exists."""
    cfg = Config()
    cfg.agents.defaults.workspace = str(tmp_path)
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda: cfg)

    result = runner.invoke(app, ["replay-deadletters"])
    assert result.exit_code == 0
    assert "No dead-letter file found" in result.stdout


def test_replay_deadletters_dry_run_with_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Dry run shows error field from dead-letter entries."""
    cfg = Config()
    cfg.agents.defaults.workspace = str(tmp_path)
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda: cfg)

    dead = tmp_path / "outbound_failed.jsonl"
    dead.write_text(
        json.dumps(
            {
                "channel": "telegram",
                "chat_id": "99",
                "content": "hello",
                "error": "connection refused",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["replay-deadletters", "--dry-run"])
    assert result.exit_code == 0
    assert "connection refused" in result.stdout


def test_replay_deadletters_no_channels(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """replay-deadletters fails when no channels are available."""
    cfg = Config()
    cfg.agents.defaults.workspace = str(tmp_path)
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda: cfg)

    dead = tmp_path / "outbound_failed.jsonl"
    dead.write_text(
        json.dumps({"channel": "telegram", "chat_id": "1", "content": "hi"}) + "\n",
        encoding="utf-8",
    )

    class _EmptyMgr:
        def __init__(self, _config, _bus):
            self.channels = {}
            self.enabled_channels = []

    monkeypatch.setattr("nanobot.channels.manager.ChannelManager", _EmptyMgr)

    result = runner.invoke(app, ["replay-deadletters"])
    assert result.exit_code == 1
    assert "No channels available" in result.stdout


def test_replay_deadletters_partial_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """replay-deadletters reports partial failures."""
    cfg = Config()
    cfg.agents.defaults.workspace = str(tmp_path)
    monkeypatch.setattr("nanobot.config.loader.load_config", lambda: cfg)

    dead = tmp_path / "outbound_failed.jsonl"
    dead.write_text(
        json.dumps({"channel": "telegram", "chat_id": "1", "content": "msg1"})
        + "\n"
        + json.dumps({"channel": "telegram", "chat_id": "2", "content": "msg2"})
        + "\n",
        encoding="utf-8",
    )

    class _PartialMgr:
        def __init__(self, _config, _bus):
            self.channels = {"telegram": object()}
            self.enabled_channels = ["telegram"]

        async def replay_dead_letters(self, dry_run=False):
            return (2, 1, 1)

    monkeypatch.setattr("nanobot.channels.manager.ChannelManager", _PartialMgr)

    result = runner.invoke(app, ["replay-deadletters"])
    assert result.exit_code == 0
    assert "1 sent" in result.stdout
    assert "1 failed" in result.stdout
    assert "Failed messages remain" in result.stdout
