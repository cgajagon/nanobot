# Web UI Session Persistence — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist web chat sessions so reloading the page and clicking a thread loads its full message history.

**Architecture:** Backend adds `GET /api/threads/{id}/messages` endpoint that filters and transforms session history to `ThreadMessageLike` format. Frontend wires `@assistant-ui/react`'s `ThreadHistoryAdapter` to fetch from this endpoint on thread switch. `thread-sync.ts` persists the local→server thread ID mapping to localStorage so it survives reloads.

**Tech Stack:** FastAPI (backend), `@assistant-ui/react` v0.12.17, TypeScript, localStorage

**Design Spec:** `docs/superpowers/specs/2026-04-02-web-session-persistence-design.md`

---

## File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Modify | `nanobot/web/routes.py` | Add `GET /api/threads/{thread_id}/messages` endpoint |
| Modify | `nanobot/web/models.py` | Add `ThreadMessageContent` and `ThreadMessagesResponse` models |
| Modify | `frontend/src/lib/thread-sync.ts` | Persist threadMap to localStorage; export `getServerThreadId()` |
| Modify | `frontend/src/App.tsx` | Add `history` adapter to `useDataStreamRuntime` |
| Create | `tests/test_web_thread_messages.py` | Backend endpoint tests |

---

### Task 1: Add response models for thread messages

**Files:**
- Modify: `nanobot/web/models.py`

- [ ] **Step 1: Add `ThreadMessageContent` and `ThreadMessagesResponse` models**

Add at the end of `nanobot/web/models.py`:

```python
class ThreadMessageContent(BaseModel):
    """A content part in a thread message (assistant-ui format)."""

    type: str = "text"
    text: str = ""


class ThreadMessageItem(BaseModel):
    """A user-visible message in assistant-ui ThreadMessageLike format."""

    id: str
    role: str
    content: list[ThreadMessageContent]


class ThreadMessagesResponse(BaseModel):
    """Response from the thread messages endpoint."""

    messages: list[ThreadMessageItem]
```

- [ ] **Step 2: Run lint**

Run: `make lint`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add nanobot/web/models.py
git commit -m "feat(web): add ThreadMessagesResponse model for thread history endpoint"
```

---

### Task 2: Add `GET /api/threads/{thread_id}/messages` endpoint

**Files:**
- Modify: `nanobot/web/routes.py`
- Create: `tests/test_web_thread_messages.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_web_thread_messages.py`:

```python
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
    app.include_router(router, prefix="/api")

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
    app.include_router(router, prefix="/api")
    app.state.session_manager = FakeSessionManager()

    return TestClient(app)


def test_returns_only_user_and_assistant_messages(app_with_messages):
    """Endpoint filters out system and tool messages."""
    resp = app_with_messages.get("/api/threads/test-thread-123/messages")
    assert resp.status_code == 200
    data = resp.json()
    roles = [m["role"] for m in data["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]
    # No system or tool messages
    assert "system" not in roles
    assert "tool" not in roles


def test_filters_assistant_messages_without_content(app_with_messages):
    """Assistant messages with only tool_calls (no text) are excluded."""
    resp = app_with_messages.get("/api/threads/test-thread-123/messages")
    data = resp.json()
    # The assistant message with tool_calls but no content should be gone
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_web_thread_messages.py -v`
Expected: FAIL (endpoint doesn't exist)

- [ ] **Step 3: Implement the endpoint**

In `nanobot/web/routes.py`, add after the `delete_thread` endpoint (after line 357):

```python
@router.get("/threads/{thread_id}/messages")
async def get_thread_messages(request: Request, thread_id: str):
    """Return user-visible messages for a thread in assistant-ui format.

    Filters out system messages, tool messages, and assistant messages
    that have only tool_calls (no displayable text content).  Transforms
    each message to ``ThreadMessageLike`` format with ``id``, ``role``,
    and ``content`` as a list of ``{type, text}`` parts.
    """
    from nanobot.web.models import (
        ThreadMessageContent,
        ThreadMessageItem,
        ThreadMessagesResponse,
    )

    session_manager = request.app.state.session_manager
    session_key = _session_key(thread_id)
    session = session_manager.get_or_create(session_key)
    history = session.get_history()

    items: list[ThreadMessageItem] = []
    idx = 0
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content")

        # Skip non-user-visible messages
        if role not in ("user", "assistant"):
            continue
        # Skip assistant messages with no text content (tool-call-only)
        if role == "assistant" and not content:
            continue

        # Transform content to assistant-ui format
        if isinstance(content, str):
            parts = [ThreadMessageContent(type="text", text=content)]
        elif isinstance(content, list):
            parts = [
                ThreadMessageContent(type=p.get("type", "text"), text=p.get("text", ""))
                for p in content
                if isinstance(p, dict) and p.get("text")
            ]
        else:
            continue

        if not parts:
            continue

        items.append(ThreadMessageItem(id=f"msg_{idx}", role=role, content=parts))
        idx += 1

    return ThreadMessagesResponse(messages=items)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_web_thread_messages.py -v`
Expected: PASS (all 6 tests)

- [ ] **Step 5: Run lint and typecheck**

Run: `make lint && make typecheck`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add nanobot/web/routes.py tests/test_web_thread_messages.py
git commit -m "feat(web): add GET /api/threads/{id}/messages endpoint

Returns user-visible messages for a thread in assistant-ui
ThreadMessageLike format. Filters out system, tool, and tool-call-only
assistant messages. Transforms content to [{type, text}] parts."
```

---

### Task 3: Persist threadMap to localStorage in `thread-sync.ts`

**Files:**
- Modify: `frontend/src/lib/thread-sync.ts`

- [ ] **Step 1: Add localStorage persistence and export `getServerThreadId`**

Replace the contents of `frontend/src/lib/thread-sync.ts` with:

```typescript
/**
 * Thread ID synchronization between assistant-ui local threads and the server.
 *
 * Problem: assistant-ui's useDataStreamRuntime may omit the `threadId` from the
 * first message in a new thread due to a React render timing issue (the remoteId
 * state update from initialize() hasn't propagated to the threadIdRef yet).
 * This causes the server to generate a UUID session for the first message,
 * while follow-up messages use `__LOCALID_xxx` — creating separate sessions.
 *
 * Solution: intercept fetch calls to `/api/chat` to:
 * 1. Capture the server's `X-Thread-Id` response header.
 * 2. Map local thread IDs to server-generated thread IDs.
 * 3. Replace local thread IDs with server IDs in subsequent requests.
 * 4. When the first message has no threadId, record a "pending" state so the
 *    next message (with a local ID) can be mapped to the server's UUID.
 *
 * The threadMap is persisted to localStorage so it survives page reloads.
 */

const CHAT_API = "/api/chat";
const STORAGE_KEY = "nanobot-thread-map";

/** Load threadMap from localStorage, or start empty. */
function loadMap(): Map<string, string> {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      return new Map<string, string>(JSON.parse(stored));
    }
  } catch {
    // Corrupted data — start fresh.
  }
  return new Map<string, string>();
}

/** Persist threadMap to localStorage. */
function persistMap(): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify([...threadMap]));
  } catch {
    // localStorage full or unavailable — no-op.
  }
}

/** Map from local thread ID (__LOCALID_xxx) to server thread ID (UUID). */
const threadMap = loadMap();

/** Server thread ID from the most recent request that had no local thread ID. */
let pendingServerThreadId: string | null = null;

/**
 * Look up the server thread ID for a local or server thread ID.
 * Returns the server UUID if a mapping exists, or the input unchanged.
 * Used by the history adapter to resolve thread IDs for API calls.
 */
export function getServerThreadId(threadId: string): string {
  return threadMap.get(threadId) ?? threadId;
}

/**
 * Install a global fetch interceptor that synchronizes thread IDs.
 * Call once at app startup (before any chat requests).
 */
export function installThreadSync(): void {
  const originalFetch = window.fetch.bind(window);

  window.fetch = async (
    input: RequestInfo | URL,
    init?: RequestInit
  ): Promise<Response> => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    if (!url.endsWith(CHAT_API) || !init?.body) {
      return originalFetch(input, init);
    }

    let body: Record<string, unknown>;
    try {
      body = JSON.parse(init.body as string);
    } catch {
      return originalFetch(input, init);
    }

    const localThreadId = body.threadId as string | undefined;

    if (localThreadId) {
      // We have a local thread ID — check if we have a server mapping for it.
      if (threadMap.has(localThreadId)) {
        body.threadId = threadMap.get(localThreadId);
        init = { ...init, body: JSON.stringify(body) };
      } else if (pendingServerThreadId) {
        // First message had no local ID, this is the follow-up.
        // Map this local ID → the server UUID from the first message.
        threadMap.set(localThreadId, pendingServerThreadId);
        body.threadId = pendingServerThreadId;
        init = { ...init, body: JSON.stringify(body) };
        pendingServerThreadId = null;
        persistMap();
      }
    }

    const response = await originalFetch(input, init);

    // Read the server's canonical thread ID from the response header.
    const serverThreadId = response.headers.get("x-thread-id");
    if (serverThreadId) {
      if (localThreadId) {
        // Direct mapping: local → server.
        threadMap.set(localThreadId, serverThreadId);
        persistMap();
      } else {
        // First message had no threadId — store as pending for the next request.
        pendingServerThreadId = serverThreadId;
      }
    }

    return response;
  };
}
```

Key changes from original:
- `threadMap` initialized from localStorage via `loadMap()`
- `persistMap()` called after every mapping update
- New export: `getServerThreadId()` — used by the history adapter (Task 4)
- `STORAGE_KEY = "nanobot-thread-map"` constant

- [ ] **Step 2: Verify the frontend builds**

Run: `cd frontend && npm run build`
Expected: BUILD SUCCESS (no type errors)

- [ ] **Step 3: Commit**

```bash
git add frontend/src/lib/thread-sync.ts
git commit -m "feat(web): persist thread ID map to localStorage

threadMap (local→server thread ID) now survives page reloads via
localStorage. Also exports getServerThreadId() for the history adapter
to resolve thread IDs when fetching historical messages."
```

---

### Task 4: Wire `ThreadHistoryAdapter` in App.tsx

**Files:**
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Add the history adapter to `useDataStreamRuntime`**

In `frontend/src/App.tsx`, add this import near the top (after the existing imports):

```typescript
import { getServerThreadId } from "@/lib/thread-sync";
import { ExportedMessageRepository } from "@assistant-ui/react";
```

Then modify the `useDataStreamRuntime` call (lines 226-241). Replace the `adapters` object:

```typescript
  const runtime = useDataStreamRuntime({
    api: "/api/chat",
    protocol: "ui-message-stream",
    /** Clone the response so we can read status events without disturbing the runtime. */
    onResponse: (response) => {
      if (!response.body) return;
      readStatusEvents(response.clone().body!, setAgentStatus);
    },
    adapters: {
      attachments: new CompositeAttachmentAdapter([
        new SimpleImageAttachmentAdapter(),
        new SimpleTextAttachmentAdapter(),
        fallbackFileAdapter,
      ]),
      history: {
        async load() {
          // Get the current thread's remote ID from the runtime URL
          // The thread ID is passed via the threadId body param on /api/chat
          // After a reload, we need to fetch messages from the server
          const threadId = runtime?.threads?.mainThread?.id;
          if (!threadId) return ExportedMessageRepository.fromArray([]);

          const serverId = getServerThreadId(threadId);
          try {
            const res = await fetch(`/api/threads/${serverId}/messages`);
            if (!res.ok) return ExportedMessageRepository.fromArray([]);
            const data = await res.json();
            return ExportedMessageRepository.fromArray(
              data.messages.map((m: { id: string; role: string; content: { type: string; text: string }[] }) => ({
                role: m.role as "user" | "assistant",
                content: m.content.map((c) => c.text).join(""),
                id: m.id,
              }))
            );
          } catch {
            return ExportedMessageRepository.fromArray([]);
          }
        },
        async append() {
          // No-op — server persists messages via the chat flow
        },
      },
    },
  });
```

Note: the `ExportedMessageRepository.fromArray()` helper converts a flat `ThreadMessageLike[]` array into the repository format that the `ThreadHistoryAdapter.load()` must return.

**IMPORTANT:** The `load()` function is called per-thread but receives no thread ID parameter. The approach of reading `runtime?.threads?.mainThread?.id` may need adjustment — verify at runtime. If the runtime object isn't accessible inside the adapter closure, an alternative is to track the current thread ID via a React ref or a module-level variable updated by a `useEffect` that watches thread switches. The implementer should test this and adapt.

- [ ] **Step 2: Verify the frontend builds**

Run: `cd frontend && npm run build`
Expected: BUILD SUCCESS

- [ ] **Step 3: Manual test**

1. Start the gateway: `nanobot gateway`
2. Open the web UI, send a message, note the thread appears in sidebar
3. Reload the page
4. Click the thread in the sidebar — messages should load

- [ ] **Step 4: Commit**

```bash
git add frontend/src/App.tsx
git commit -m "feat(web): wire ThreadHistoryAdapter to load messages on thread switch

Configures useDataStreamRuntime's adapters.history to fetch messages
from GET /api/threads/{id}/messages when switching threads. Uses
getServerThreadId() from thread-sync to resolve local thread IDs.
Messages load from server on page reload when clicking a thread."
```

---

### Task 5: Full validation

- [ ] **Step 1: Run backend checks**

Run: `make check`
Expected: PASS

- [ ] **Step 2: Run backend tests**

Run: `make test`
Expected: PASS

- [ ] **Step 3: Run frontend build**

Run: `cd frontend && npm run build`
Expected: BUILD SUCCESS

- [ ] **Step 4: Manual end-to-end test**

1. Start gateway: `nanobot gateway`
2. Send a multi-turn conversation (at least 3 messages)
3. Reload the page
4. Verify thread list shows the thread with correct title
5. Click the thread — verify all user and assistant messages appear
6. Send a new message in the restored thread — verify it continues the session
7. Create a new thread — verify it works independently
