from __future__ import annotations

import asyncio
import errno
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from nanobot.cli.commands import app
from nanobot.config.schema import Config

runner = CliRunner()


@dataclass
class _Payload:
    message: str
    channel: str | None
    to: str | None
    deliver: bool


@dataclass
class _CronJob:
    id: str
    payload: _Payload


class _Bus:
    def __init__(self):
        self.outbound = []

    async def publish_outbound(self, msg):
        self.outbound.append(msg)

    async def publish_inbound(self, _msg):
        return None

    async def consume_outbound(self):
        await asyncio.sleep(5)
        return SimpleNamespace(content="", metadata={})


class _AgentLoop:
    def __init__(self, **kwargs):
        self.model = "fake-model"
        self.channels_config = SimpleNamespace(send_tool_hints=True, send_progress=True)
        self._stopped = False
        self.context = SimpleNamespace(set_contacts_context=lambda contacts: None)
        self._capabilities = SimpleNamespace(refresh_health=lambda: None)

    def set_deliver_callback(self, callback):
        pass

    def set_contacts_provider(self, provider):
        pass

    async def process_direct(self, *args, **kwargs):
        return "ok-response"

    async def run(self):
        await asyncio.sleep(0)

    def stop(self):
        self._stopped = True

    async def close_mcp(self):
        return None


class _ChannelManager:
    def __init__(self, _config: Config, _bus: _Bus, enabled: list[str] | None = None):
        self.enabled_channels = enabled or ["telegram"]
        self.channels = {name: object() for name in self.enabled_channels}

    async def deliver(self, msg):
        return None

    async def start_all(self):
        await asyncio.sleep(0)

    async def stop_all(self):
        await asyncio.sleep(0)

    def get_email_contacts(self):
        return []

    def get_channel(self, name: str):
        return self.channels[name]


class _CronService:
    def __init__(self, _path: Path):
        self.on_job = None

    def status(self):
        return {"jobs": 1}

    async def start(self):
        if self.on_job is not None:
            await self.on_job(_CronJob("j1", _Payload("cron-msg", "telegram", "42", True)))

    def stop(self):
        return None


class _HeartbeatService:
    def __init__(self, *, on_execute, on_notify, on_health_refresh=None, **kwargs):
        self._on_execute = on_execute
        self._on_notify = on_notify

    async def start(self):
        out = await self._on_execute("heartbeat-task")
        await self._on_notify(out)

    def stop(self):
        return None


class _SessionManager:
    def __init__(self, _workspace: Path):
        self._items = [{"key": "telegram:42"}]

    def list_sessions(self):
        return list(self._items)


def test_gateway_runs_cron_and_heartbeat_callbacks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = Config()
    cfg.agents.defaults.workspace = str(tmp_path)
    cfg.gateway.heartbeat.enabled = True

    bus = _Bus()

    monkeypatch.setattr("nanobot.config.loader.load_config", lambda: cfg)
    monkeypatch.setattr("nanobot.config.loader.get_data_dir", lambda: tmp_path)
    monkeypatch.setattr("nanobot.cli.gateway._make_provider", lambda _cfg: object())
    monkeypatch.setattr("nanobot.bus.queue.MessageBus", lambda: bus)
    monkeypatch.setattr("nanobot.agent.loop.AgentLoop", _AgentLoop)
    monkeypatch.setattr("nanobot.session.manager.SessionManager", _SessionManager)
    monkeypatch.setattr("nanobot.cron.service.CronService", _CronService)
    monkeypatch.setattr("nanobot.heartbeat.service.HeartbeatService", _HeartbeatService)
    monkeypatch.setattr("nanobot.channels.manager.ChannelManager", _ChannelManager)

    out = runner.invoke(app, ["gateway", "--port", "19000"])
    assert out.exit_code == 0
    assert "Starting nanobot gateway" in out.stdout
    assert "Channels enabled" in out.stdout
    assert "Heartbeat" in out.stdout
    assert len(bus.outbound) >= 1


def test_gateway_continues_when_health_port_is_busy_with_web_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = Config()
    cfg.agents.defaults.workspace = str(tmp_path)
    cfg.channels.web.enabled = True
    cfg.channels.web.host = "127.0.0.1"
    cfg.channels.web.port = 8000

    bus = _Bus()

    monkeypatch.setattr("nanobot.config.loader.load_config", lambda: cfg)
    monkeypatch.setattr("nanobot.config.loader.get_data_dir", lambda: tmp_path)
    monkeypatch.setattr("nanobot.cli.gateway._make_provider", lambda _cfg: object())
    monkeypatch.setattr("nanobot.bus.queue.MessageBus", lambda: bus)
    monkeypatch.setattr("nanobot.agent.loop.AgentLoop", _AgentLoop)
    monkeypatch.setattr("nanobot.session.manager.SessionManager", _SessionManager)
    monkeypatch.setattr("nanobot.cron.service.CronService", _CronService)
    monkeypatch.setattr("nanobot.heartbeat.service.HeartbeatService", _HeartbeatService)
    monkeypatch.setattr(
        "nanobot.channels.manager.ChannelManager",
        lambda _config, _bus: _ChannelManager(_config, _bus, enabled=["telegram", "web"]),
    )

    class _WebChannel:
        pass

    web_channel = _WebChannel()

    class _FakeServer:
        async def serve(self):
            await asyncio.sleep(0)

    class _FakeConfig:
        def __init__(self, *args, **kwargs):
            pass

    def _create_app(*args, **kwargs):
        return object()

    async def _start_health_server(*args, **kwargs):
        raise OSError(errno.EADDRINUSE, "address already in use")

    monkeypatch.setattr("nanobot.channels.web.WebChannel", _WebChannel)
    monkeypatch.setattr("nanobot.web.app.create_app", _create_app)
    monkeypatch.setattr("uvicorn.Config", _FakeConfig)
    monkeypatch.setattr("uvicorn.Server", lambda *_args, **_kwargs: _FakeServer())
    monkeypatch.setattr("nanobot.web.health.start_health_server", _start_health_server)

    manager = _ChannelManager(cfg, bus, enabled=["telegram", "web"])
    manager.channels["web"] = web_channel
    monkeypatch.setattr("nanobot.channels.manager.ChannelManager", lambda _config, _bus: manager)

    out = runner.invoke(app, ["gateway", "--port", "19000"])
    assert out.exit_code == 0
    assert "Gateway health port 19000 is already in use" in out.stdout
    # "Web UI:" when frontend is built; "Web API:" when it is not — assert the URL only.
    assert "http://127.0.0.1:8000" in out.stdout


def test_agent_single_message_and_interactive_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = Config()
    cfg.agents.defaults.workspace = str(tmp_path)

    monkeypatch.setattr("nanobot.config.loader.load_config", lambda: cfg)
    monkeypatch.setattr("nanobot.config.loader.get_data_dir", lambda: tmp_path)
    monkeypatch.setattr("nanobot.cli.agent._make_provider", lambda _cfg: object())
    monkeypatch.setattr("nanobot.bus.queue.MessageBus", _Bus)
    monkeypatch.setattr("nanobot.agent.loop.AgentLoop", _AgentLoop)
    monkeypatch.setattr("nanobot.cron.service.CronService", _CronService)

    class _Timer:
        def __init__(self, *_args, **_kwargs):
            self.daemon = True

        def start(self):
            return None

        def cancel(self):
            return None

    monkeypatch.setattr("threading.Timer", _Timer)

    single = runner.invoke(app, ["agent", "-m", "hello", "--timeout", "1"])
    assert single.exit_code == 0
    assert "nanobot" in single.stdout

    monkeypatch.setattr("nanobot.cli.agent._init_prompt_session", lambda: None)
    monkeypatch.setattr("nanobot.cli.agent._flush_pending_tty_input", lambda: None)
    monkeypatch.setattr("nanobot.cli.agent._restore_terminal", lambda: None)

    calls = {"n": 0}

    async def _read_once():
        calls["n"] += 1
        return "exit"

    monkeypatch.setattr("nanobot.cli.agent._read_interactive_input_async", _read_once)

    interactive = runner.invoke(app, ["agent", "--session", "cli:direct", "--timeout", "0"])
    assert interactive.exit_code == 0
    assert "Interactive mode" in interactive.stdout


def test_agent_single_message_logs_enabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Agent single-message with --logs flag uses nullcontext (no spinner)."""
    cfg = Config()
    cfg.agents.defaults.workspace = str(tmp_path)

    monkeypatch.setattr("nanobot.config.loader.load_config", lambda: cfg)
    monkeypatch.setattr("nanobot.config.loader.get_data_dir", lambda: tmp_path)
    monkeypatch.setattr("nanobot.cli.agent._make_provider", lambda _cfg: object())
    monkeypatch.setattr("nanobot.bus.queue.MessageBus", _Bus)
    monkeypatch.setattr("nanobot.agent.loop.AgentLoop", _AgentLoop)
    monkeypatch.setattr("nanobot.cron.service.CronService", _CronService)

    class _Timer:
        def __init__(self, *_args, **_kwargs):
            self.daemon = True

        def start(self):
            return None

        def cancel(self):
            return None

    monkeypatch.setattr("threading.Timer", _Timer)

    result = runner.invoke(app, ["agent", "-m", "hello", "--logs", "--timeout", "1"])
    assert result.exit_code == 0


def test_agent_interactive_session_id_without_colon(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Interactive mode with a session ID that has no colon uses 'cli' as channel."""
    cfg = Config()
    cfg.agents.defaults.workspace = str(tmp_path)

    monkeypatch.setattr("nanobot.config.loader.load_config", lambda: cfg)
    monkeypatch.setattr("nanobot.config.loader.get_data_dir", lambda: tmp_path)
    monkeypatch.setattr("nanobot.cli.agent._make_provider", lambda _cfg: object())
    monkeypatch.setattr("nanobot.bus.queue.MessageBus", _Bus)
    monkeypatch.setattr("nanobot.agent.loop.AgentLoop", _AgentLoop)
    monkeypatch.setattr("nanobot.cron.service.CronService", _CronService)
    monkeypatch.setattr("nanobot.cli.agent._init_prompt_session", lambda: None)
    monkeypatch.setattr("nanobot.cli.agent._flush_pending_tty_input", lambda: None)
    monkeypatch.setattr("nanobot.cli.agent._restore_terminal", lambda: None)

    async def _read_exit():
        return "exit"

    monkeypatch.setattr("nanobot.cli.agent._read_interactive_input_async", _read_exit)

    result = runner.invoke(app, ["agent", "--session", "mysession", "--timeout", "0"])
    assert result.exit_code == 0
    assert "Interactive mode" in result.stdout


def test_agent_interactive_empty_input_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Interactive mode skips empty input and continues."""
    cfg = Config()
    cfg.agents.defaults.workspace = str(tmp_path)

    monkeypatch.setattr("nanobot.config.loader.load_config", lambda: cfg)
    monkeypatch.setattr("nanobot.config.loader.get_data_dir", lambda: tmp_path)
    monkeypatch.setattr("nanobot.cli.agent._make_provider", lambda _cfg: object())
    monkeypatch.setattr("nanobot.bus.queue.MessageBus", _Bus)
    monkeypatch.setattr("nanobot.agent.loop.AgentLoop", _AgentLoop)
    monkeypatch.setattr("nanobot.cron.service.CronService", _CronService)
    monkeypatch.setattr("nanobot.cli.agent._init_prompt_session", lambda: None)
    monkeypatch.setattr("nanobot.cli.agent._flush_pending_tty_input", lambda: None)
    monkeypatch.setattr("nanobot.cli.agent._restore_terminal", lambda: None)

    call_count = {"n": 0}

    async def _read_then_exit():
        call_count["n"] += 1
        if call_count["n"] == 1:
            return "   "  # empty/whitespace
        return "exit"

    monkeypatch.setattr("nanobot.cli.agent._read_interactive_input_async", _read_then_exit)

    result = runner.invoke(app, ["agent", "--session", "cli:direct", "--timeout", "0"])
    assert result.exit_code == 0
    assert call_count["n"] == 2  # called twice: empty, then exit


def test_agent_interactive_eof_exits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Interactive mode handles EOFError (Ctrl+D) by exiting cleanly."""
    cfg = Config()
    cfg.agents.defaults.workspace = str(tmp_path)

    monkeypatch.setattr("nanobot.config.loader.load_config", lambda: cfg)
    monkeypatch.setattr("nanobot.config.loader.get_data_dir", lambda: tmp_path)
    monkeypatch.setattr("nanobot.cli.agent._make_provider", lambda _cfg: object())
    monkeypatch.setattr("nanobot.bus.queue.MessageBus", _Bus)
    monkeypatch.setattr("nanobot.agent.loop.AgentLoop", _AgentLoop)
    monkeypatch.setattr("nanobot.cron.service.CronService", _CronService)
    monkeypatch.setattr("nanobot.cli.agent._init_prompt_session", lambda: None)
    monkeypatch.setattr("nanobot.cli.agent._flush_pending_tty_input", lambda: None)
    monkeypatch.setattr("nanobot.cli.agent._restore_terminal", lambda: None)

    async def _raise_eof():
        raise EOFError

    monkeypatch.setattr("nanobot.cli.agent._read_interactive_input_async", _raise_eof)

    result = runner.invoke(app, ["agent", "--session", "cli:direct", "--timeout", "0"])
    assert result.exit_code == 0
    assert "Goodbye" in result.stdout


def test_agent_single_message_no_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Agent single-message with --timeout 0 skips watchdog timer."""
    cfg = Config()
    cfg.agents.defaults.workspace = str(tmp_path)

    monkeypatch.setattr("nanobot.config.loader.load_config", lambda: cfg)
    monkeypatch.setattr("nanobot.config.loader.get_data_dir", lambda: tmp_path)
    monkeypatch.setattr("nanobot.cli.agent._make_provider", lambda _cfg: object())
    monkeypatch.setattr("nanobot.bus.queue.MessageBus", _Bus)
    monkeypatch.setattr("nanobot.agent.loop.AgentLoop", _AgentLoop)
    monkeypatch.setattr("nanobot.cron.service.CronService", _CronService)

    result = runner.invoke(app, ["agent", "-m", "hi", "--timeout", "0"])
    assert result.exit_code == 0


def test_agent_interactive_sends_message_and_gets_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Interactive mode sends a message through the bus and prints the response."""
    cfg = Config()
    cfg.agents.defaults.workspace = str(tmp_path)

    monkeypatch.setattr("nanobot.config.loader.load_config", lambda: cfg)
    monkeypatch.setattr("nanobot.config.loader.get_data_dir", lambda: tmp_path)
    monkeypatch.setattr("nanobot.cli.agent._make_provider", lambda _cfg: object())
    monkeypatch.setattr("nanobot.cron.service.CronService", _CronService)
    monkeypatch.setattr("nanobot.cli.agent._init_prompt_session", lambda: None)
    monkeypatch.setattr("nanobot.cli.agent._flush_pending_tty_input", lambda: None)
    monkeypatch.setattr("nanobot.cli.agent._restore_terminal", lambda: None)

    call_count = {"n": 0}

    async def _read_input():
        call_count["n"] += 1
        if call_count["n"] == 1:
            return "hello agent"
        return "exit"

    monkeypatch.setattr("nanobot.cli.agent._read_interactive_input_async", _read_input)

    # Bus that returns a response after receiving an inbound message
    class _ResponsiveBus:
        def __init__(self):
            self._inbound_received = asyncio.Event()
            self._response_sent = False

        async def publish_inbound(self, msg):
            self._inbound_received.set()

        async def consume_outbound(self):
            # Wait for an inbound message, then respond once
            await self._inbound_received.wait()
            if not self._response_sent:
                self._response_sent = True
                return SimpleNamespace(content="agent reply", metadata={})
            # After responding, return slowly so the outbound consumer doesn't spin
            await asyncio.sleep(5)
            return SimpleNamespace(content="", metadata={})

    monkeypatch.setattr("nanobot.bus.queue.MessageBus", _ResponsiveBus)

    class _RunAgentLoop(_AgentLoop):
        async def run(self):
            # Simulate agent loop running briefly
            await asyncio.sleep(0)

    monkeypatch.setattr("nanobot.agent.loop.AgentLoop", _RunAgentLoop)

    result = runner.invoke(app, ["agent", "--session", "cli:test", "--timeout", "0"])
    assert result.exit_code == 0
    assert "Interactive mode" in result.stdout
    assert "agent reply" in result.stdout


def test_agent_interactive_progress_messages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Interactive mode displays progress messages from the bus."""
    cfg = Config()
    cfg.agents.defaults.workspace = str(tmp_path)

    monkeypatch.setattr("nanobot.config.loader.load_config", lambda: cfg)
    monkeypatch.setattr("nanobot.config.loader.get_data_dir", lambda: tmp_path)
    monkeypatch.setattr("nanobot.cli.agent._make_provider", lambda _cfg: object())
    monkeypatch.setattr("nanobot.cron.service.CronService", _CronService)
    monkeypatch.setattr("nanobot.cli.agent._init_prompt_session", lambda: None)
    monkeypatch.setattr("nanobot.cli.agent._flush_pending_tty_input", lambda: None)
    monkeypatch.setattr("nanobot.cli.agent._restore_terminal", lambda: None)

    call_count = {"n": 0}

    async def _read_input():
        call_count["n"] += 1
        if call_count["n"] == 1:
            return "do something"
        # Give time for progress messages to be consumed
        await asyncio.sleep(0.1)
        return "exit"

    monkeypatch.setattr("nanobot.cli.agent._read_interactive_input_async", _read_input)

    # Bus that sends a progress message then a final response
    class _ProgressBus:
        def __init__(self):
            self._inbound_received = asyncio.Event()
            self._msg_index = 0

        async def publish_inbound(self, msg):
            self._inbound_received.set()

        async def consume_outbound(self):
            await self._inbound_received.wait()
            self._msg_index += 1
            if self._msg_index == 1:
                return SimpleNamespace(content="Searching files...", metadata={"_progress": True})
            if self._msg_index == 2:
                return SimpleNamespace(content="done result", metadata={})
            await asyncio.sleep(5)
            return SimpleNamespace(content="", metadata={})

    monkeypatch.setattr("nanobot.bus.queue.MessageBus", _ProgressBus)
    monkeypatch.setattr("nanobot.agent.loop.AgentLoop", _AgentLoop)

    result = runner.invoke(app, ["agent", "--session", "cli:test", "--timeout", "0"])
    assert result.exit_code == 0
    assert "Searching files" in result.stdout


def test_agent_interactive_tool_hint_suppressed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Interactive mode suppresses tool hints when channels_config disables them."""
    cfg = Config()
    cfg.agents.defaults.workspace = str(tmp_path)

    monkeypatch.setattr("nanobot.config.loader.load_config", lambda: cfg)
    monkeypatch.setattr("nanobot.config.loader.get_data_dir", lambda: tmp_path)
    monkeypatch.setattr("nanobot.cli.agent._make_provider", lambda _cfg: object())
    monkeypatch.setattr("nanobot.cron.service.CronService", _CronService)
    monkeypatch.setattr("nanobot.cli.agent._init_prompt_session", lambda: None)
    monkeypatch.setattr("nanobot.cli.agent._flush_pending_tty_input", lambda: None)
    monkeypatch.setattr("nanobot.cli.agent._restore_terminal", lambda: None)

    call_count = {"n": 0}

    async def _read_input():
        call_count["n"] += 1
        if call_count["n"] == 1:
            return "query"
        await asyncio.sleep(0.1)
        return "exit"

    monkeypatch.setattr("nanobot.cli.agent._read_interactive_input_async", _read_input)

    class _NoHintsAgentLoop(_AgentLoop):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # Disable tool hints
            self.channels_config = SimpleNamespace(send_tool_hints=False, send_progress=True)

    # Bus that sends a tool hint progress, then a regular progress, then response
    class _HintBus:
        def __init__(self):
            self._inbound_received = asyncio.Event()
            self._msg_index = 0

        async def publish_inbound(self, msg):
            self._inbound_received.set()

        async def consume_outbound(self):
            await self._inbound_received.wait()
            self._msg_index += 1
            if self._msg_index == 1:
                # Tool hint that should be suppressed
                return SimpleNamespace(
                    content="Using read_file...",
                    metadata={"_progress": True, "_tool_hint": True},
                )
            if self._msg_index == 2:
                # Regular progress that should be shown
                return SimpleNamespace(content="Analyzing...", metadata={"_progress": True})
            if self._msg_index == 3:
                return SimpleNamespace(content="final answer", metadata={})
            await asyncio.sleep(5)
            return SimpleNamespace(content="", metadata={})

    monkeypatch.setattr("nanobot.bus.queue.MessageBus", _HintBus)
    monkeypatch.setattr("nanobot.agent.loop.AgentLoop", _NoHintsAgentLoop)

    result = runner.invoke(app, ["agent", "--session", "cli:test", "--timeout", "0"])
    assert result.exit_code == 0
    # Tool hint should be suppressed
    assert "read_file" not in result.stdout
    # Regular progress should be shown
    assert "Analyzing" in result.stdout


def test_agent_interactive_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Interactive mode handles KeyboardInterrupt by exiting with Goodbye."""
    cfg = Config()
    cfg.agents.defaults.workspace = str(tmp_path)

    monkeypatch.setattr("nanobot.config.loader.load_config", lambda: cfg)
    monkeypatch.setattr("nanobot.config.loader.get_data_dir", lambda: tmp_path)
    monkeypatch.setattr("nanobot.cli.agent._make_provider", lambda _cfg: object())
    monkeypatch.setattr("nanobot.bus.queue.MessageBus", _Bus)
    monkeypatch.setattr("nanobot.agent.loop.AgentLoop", _AgentLoop)
    monkeypatch.setattr("nanobot.cron.service.CronService", _CronService)
    monkeypatch.setattr("nanobot.cli.agent._init_prompt_session", lambda: None)
    monkeypatch.setattr("nanobot.cli.agent._flush_pending_tty_input", lambda: None)
    monkeypatch.setattr("nanobot.cli.agent._restore_terminal", lambda: None)

    async def _raise_keyboard_interrupt():
        raise KeyboardInterrupt

    monkeypatch.setattr(
        "nanobot.cli.agent._read_interactive_input_async", _raise_keyboard_interrupt
    )

    result = runner.invoke(app, ["agent", "--session", "cli:direct", "--timeout", "0"])
    assert result.exit_code == 0
    assert "Goodbye" in result.stdout
