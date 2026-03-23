from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prompt_toolkit.formatted_text import HTML

from nanobot.cli import agent as _agent_mod


@pytest.fixture
def mock_prompt_session():
    """Mock the global prompt session."""
    mock_session = MagicMock()
    mock_session.prompt_async = AsyncMock()
    with (
        patch("nanobot.cli.agent._PROMPT_SESSION", mock_session),
        patch("nanobot.cli.agent.patch_stdout"),
    ):
        yield mock_session


async def test_read_interactive_input_async_returns_input(mock_prompt_session):
    """Test that _read_interactive_input_async returns the user input from prompt_session."""
    mock_prompt_session.prompt_async.return_value = "hello world"

    result = await _agent_mod._read_interactive_input_async()

    assert result == "hello world"
    mock_prompt_session.prompt_async.assert_called_once()
    args, _ = mock_prompt_session.prompt_async.call_args
    assert isinstance(args[0], HTML)  # Verify HTML prompt is used


async def test_read_interactive_input_async_handles_eof(mock_prompt_session):
    """Test that EOFError converts to KeyboardInterrupt."""
    mock_prompt_session.prompt_async.side_effect = EOFError()

    with pytest.raises(KeyboardInterrupt):
        await _agent_mod._read_interactive_input_async()


def test_init_prompt_session_creates_session():
    """Test that _init_prompt_session initializes the global session."""
    # Ensure global is None before test
    _agent_mod._PROMPT_SESSION = None

    with (
        patch("nanobot.cli.agent.PromptSession") as mock_session,
        patch("nanobot.cli.agent.FileHistory"),
        patch("pathlib.Path.home") as mock_home,
    ):
        mock_home.return_value = MagicMock()

        _agent_mod._init_prompt_session()

        assert _agent_mod._PROMPT_SESSION is not None
        mock_session.assert_called_once()
        _, kwargs = mock_session.call_args
        assert kwargs["multiline"] is False
        assert kwargs["enable_open_in_editor"] is False


async def test_read_interactive_input_async_raises_without_session():
    """_read_interactive_input_async raises RuntimeError when session is None."""
    original = _agent_mod._PROMPT_SESSION
    _agent_mod._PROMPT_SESSION = None
    try:
        with pytest.raises(RuntimeError, match="Call _init_prompt_session"):
            await _agent_mod._read_interactive_input_async()
    finally:
        _agent_mod._PROMPT_SESSION = original


def test_is_exit_command():
    """_is_exit_command recognises all exit variants."""
    for cmd in ("exit", "quit", "/exit", "/quit", ":q", "EXIT", "Quit"):
        assert _agent_mod._is_exit_command(cmd) is True
    assert _agent_mod._is_exit_command("hello") is False
    assert _agent_mod._is_exit_command("") is False


async def test_drain_pending_tasks_returns_immediately_when_no_tasks():
    """_drain_pending_tasks returns immediately when there are no background tasks."""
    await _agent_mod._drain_pending_tasks(timeout=0.1)


def test_restore_terminal_noop_when_no_saved_attrs():
    """_restore_terminal is a no-op when _SAVED_TERM_ATTRS is None."""
    original = _agent_mod._SAVED_TERM_ATTRS
    _agent_mod._SAVED_TERM_ATTRS = None
    try:
        _agent_mod._restore_terminal()  # should not raise
    finally:
        _agent_mod._SAVED_TERM_ATTRS = original


def test_flush_pending_tty_input_non_tty():
    """_flush_pending_tty_input returns early when stdin is not a TTY."""
    import io

    with patch("sys.stdin", io.StringIO("")):
        _agent_mod._flush_pending_tty_input()  # should not raise


def test_flush_pending_tty_input_fileno_exception():
    """_flush_pending_tty_input handles stdin without fileno."""
    mock_stdin = MagicMock()
    mock_stdin.fileno.side_effect = Exception("no fileno")
    with patch("sys.stdin", mock_stdin):
        _agent_mod._flush_pending_tty_input()  # should not raise


async def test_drain_pending_tasks_with_background_tasks():
    """_drain_pending_tasks waits for pending tasks."""
    completed = {"done": False}

    async def _bg():
        completed["done"] = True

    import asyncio

    task = asyncio.create_task(_bg())
    await _agent_mod._drain_pending_tasks(timeout=1.0)
    assert completed["done"] is True
    assert task.done()
