"""Contract tests: session persistence survives LLM errors."""

from __future__ import annotations

from pathlib import Path

import pytest

from nanobot.bus.events import InboundMessage
from nanobot.providers.base import LLMResponse, ToolCallRequest
from tests.helpers import ScriptedProvider, _make_loop


def _make_inbound(text: str = "find DS10540") -> InboundMessage:
    return InboundMessage(
        channel="cli",
        chat_id="test-user",
        sender_id="user-1",
        content=text,
    )


class TestUserMessagePersistedBeforeOrchestrator:
    """User message must be on disk before the orchestrator runs."""

    async def test_user_message_survives_provider_exception(self, tmp_path: Path):
        """If the provider raises, the user message is still in the session."""

        class FailingProvider(ScriptedProvider):
            async def chat(self, *args, **kwargs):  # type: ignore[override]
                raise RuntimeError("connection reset")

        provider = FailingProvider([])
        loop = _make_loop(tmp_path, provider)

        with pytest.raises(RuntimeError, match="connection reset"):
            await loop._process_message(_make_inbound("important question"))

        # Reload session from disk
        loop._processor.sessions.invalidate("cli:test-user")
        session = loop._processor.sessions.get_or_create("cli:test-user")

        user_msgs = [m for m in session.messages if m.get("role") == "user"]
        assert len(user_msgs) >= 1
        assert "important question" in user_msgs[-1]["content"]


class TestPartialStateSavedOnException:
    """Tool results accumulated before a crash are persisted."""

    async def test_tool_results_saved_when_second_llm_call_crashes(self, tmp_path: Path):
        """First LLM call returns tool_calls, second raises. Partial state saved."""
        call_count = 0

        class CrashOnSecondCall(ScriptedProvider):
            async def chat(self, *args, **kwargs):  # type: ignore[override]
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return LLMResponse(
                        content=None,
                        tool_calls=[
                            ToolCallRequest(
                                id="tc_1",
                                name="read_file",
                                arguments={"file_path": str(tmp_path / "data.txt")},
                            )
                        ],
                    )
                raise RuntimeError("API unavailable")

        # Create the file the tool will read
        (tmp_path / "data.txt").write_text("test data")

        provider = CrashOnSecondCall([])
        loop = _make_loop(tmp_path, provider)

        with pytest.raises(RuntimeError, match="API unavailable"):
            await loop._process_message(_make_inbound("read my file"))

        # Reload session from disk
        loop._processor.sessions.invalidate("cli:test-user")
        session = loop._processor.sessions.get_or_create("cli:test-user")

        # User message should be persisted
        user_msgs = [m for m in session.messages if m.get("role") == "user"]
        assert any("read my file" in m["content"] for m in user_msgs)

        # Tool result should be persisted (partial state from before crash)
        tool_msgs = [m for m in session.messages if m.get("role") == "tool"]
        assert len(tool_msgs) >= 1


class TestNoDuplicateUserMessage:
    """Pre-save + normal save must not produce duplicate user messages."""

    async def test_successful_turn_has_one_user_message(self, tmp_path: Path):
        """A normal successful turn stores exactly one user message."""
        provider = ScriptedProvider([LLMResponse(content="Hello!")])
        loop = _make_loop(tmp_path, provider)

        await loop._process_message(_make_inbound("hi there"))

        # Reload session from disk
        loop._processor.sessions.invalidate("cli:test-user")
        session = loop._processor.sessions.get_or_create("cli:test-user")

        user_msgs = [m for m in session.messages if m.get("role") == "user"]
        assert len(user_msgs) == 1
        assert "hi there" in user_msgs[0]["content"]
