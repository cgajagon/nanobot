"""Tests for GET /api/threads/{thread_id}/messages endpoint."""

from __future__ import annotations

import pytest


def _make_messages() -> list[dict]:
    """Return a realistic session history with mixed message types."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello there"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "tc1", "function": {"name": "exec", "arguments": "{}"}}],
        },
        {"role": "tool", "content": "command output", "tool_call_id": "tc1"},
        {"role": "assistant", "content": "Here is what I found."},
        {"role": "system", "content": "Guardrail: try a different approach"},
        {"role": "user", "content": "Thanks, can you also check X?"},
        {"role": "assistant", "content": "Sure, checking X now..."},
    ]


class FakeSession:
    """Minimal session stub for testing."""

    def __init__(self, messages: list[dict]) -> None:
        self._messages = messages

    def get_history(self, max_messages: int | None = None) -> list[dict]:
        return list(self._messages)


class FakeSessionManager:
    """Minimal session manager stub for testing."""

    def __init__(self, sessions: dict[str, FakeSession] | None = None) -> None:
        self._sessions = sessions or {}

    def get_or_create(self, key: str) -> FakeSession:
        if key not in self._sessions:
            self._sessions[key] = FakeSession([])
        return self._sessions[key]

    def list_sessions(self) -> list[dict]:
        return [{"key": k} for k in self._sessions]

    def save(self, session: object) -> None:
        pass

    def invalidate(self, key: str) -> None:
        pass

    def _get_session_path(self, key: str) -> type:
        from pathlib import Path

        return Path("/tmp/fake")


@pytest.fixture
def app_with_messages():
    """Create a test FastAPI app with a pre-populated session."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from nanobot.web.routes import router

    app = FastAPI()
    app.include_router(router)  # router already has prefix="/api"
    session = FakeSession(_make_messages())
    app.state.session_manager = FakeSessionManager({"web:test-thread-123": session})
    return TestClient(app)


@pytest.fixture
def app_empty():
    """Create a test FastAPI app with no sessions."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from nanobot.web.routes import router

    app = FastAPI()
    app.include_router(router)  # router already has prefix="/api"
    app.state.session_manager = FakeSessionManager()
    return TestClient(app)


def test_returns_only_user_and_assistant_messages(app_with_messages):
    """Endpoint filters out system and tool messages."""
    resp = app_with_messages.get("/api/threads/test-thread-123/messages")
    assert resp.status_code == 200
    data = resp.json()
    roles = [m["role"] for m in data["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert "system" not in roles
    assert "tool" not in roles


def test_filters_assistant_messages_without_content(app_with_messages):
    """Assistant messages with only tool_calls (no text) are excluded."""
    resp = app_with_messages.get("/api/threads/test-thread-123/messages")
    data = resp.json()
    for msg in data["messages"]:
        if msg["role"] == "assistant":
            assert len(msg["content"]) > 0
            assert msg["content"][0]["text"] != ""


def test_message_format_is_assistant_ui_compatible(app_with_messages):
    """Messages have id, role, and content as list of {type, text}."""
    resp = app_with_messages.get("/api/threads/test-thread-123/messages")
    data = resp.json()
    for msg in data["messages"]:
        assert "id" in msg
        assert "role" in msg
        assert isinstance(msg["content"], list)
        for part in msg["content"]:
            assert part["type"] == "text"
            assert isinstance(part["text"], str)


def test_messages_have_sequential_ids(app_with_messages):
    """Each message gets a unique sequential id."""
    resp = app_with_messages.get("/api/threads/test-thread-123/messages")
    data = resp.json()
    ids = [m["id"] for m in data["messages"]]
    assert ids == ["msg_0", "msg_1", "msg_2", "msg_3"]


def test_empty_thread_returns_empty_array(app_empty):
    """A thread with no messages returns an empty array."""
    resp = app_empty.get("/api/threads/nonexistent/messages")
    assert resp.status_code == 200
    assert resp.json() == {"messages": []}


def test_string_content_wrapped_in_text_part(app_with_messages):
    """Plain string content is wrapped in [{type: 'text', text: content}]."""
    resp = app_with_messages.get("/api/threads/test-thread-123/messages")
    data = resp.json()
    first_msg = data["messages"][0]
    assert first_msg["role"] == "user"
    assert first_msg["content"] == [{"type": "text", "text": "Hello there"}]
