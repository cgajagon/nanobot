# Web UI Session Persistence — Design Spec

> Persist chat sessions across page reloads so clicking a thread in the
> sidebar loads its full message history (ChatGPT-like behavior).

**Date:** 2026-04-02
**Status:** Draft

---

## Problem

When the user reloads the web UI, all conversation history is lost. The thread
list re-fetches from `/api/threads` (titles visible), but clicking a thread
shows an empty chat. Messages exist on disk in session JSONL files but the
frontend has no way to load them.

**Root cause:** Two missing pieces:
1. No backend endpoint to retrieve historical messages for a thread
2. No frontend `ThreadHistoryAdapter` configured to load messages on thread switch

## Solution

### Backend: Add `/api/threads/{thread_id}/messages` endpoint

New `GET` endpoint in `nanobot/web/routes.py` that:
1. Loads the session via `SessionManager.get_or_create(f"web:{thread_id}")`
2. Calls `session.get_history()` to retrieve stored messages
3. Filters to user-visible messages only (user + assistant with content)
4. Transforms to `@assistant-ui/react`'s `ThreadMessageLike` format
5. Returns the array

**Message filtering rules:**
- Include: `role=user` messages with content
- Include: `role=assistant` messages with non-empty text content
- Exclude: `role=system` (prompts, guardrail injections)
- Exclude: `role=tool` (tool results)
- Exclude: `role=assistant` messages with only `tool_calls` and no content

**Message format transformation (server-side):**

Session stores OpenAI format:
```json
{"role": "user", "content": "hello"}
```

Endpoint returns assistant-ui format:
```json
{
  "messages": [
    {
      "id": "msg_0",
      "role": "user",
      "content": [{"type": "text", "text": "hello"}]
    }
  ]
}
```

Each message gets a synthetic `id` field (`msg_{index}`) since session storage
doesn't track message IDs.

String content is wrapped in `[{"type": "text", "text": content}]`. Content
that is already a list (multimodal) is passed through.

### Frontend: Wire `ThreadHistoryAdapter` in App.tsx

Configure the `useDataStreamRuntime` hook's `adapters.history` with an adapter
whose `load()` method fetches from the new endpoint:

```typescript
adapters: {
  history: {
    async load() {
      const threadId = getCurrentThreadId(); // from runtime
      if (!threadId) return { messages: [] };
      const res = await fetch(`/api/threads/${threadId}/messages`);
      const data = await res.json();
      return { messages: data.messages };
    },
    async append(item) {
      // No-op — server persists messages via the chat flow
    },
  },
  // ... existing attachments adapter
}
```

The `@assistant-ui/react` library calls `load()` automatically when switching
threads. No custom thread-switching logic needed.

### Frontend: Persist threadMap to localStorage

Update `thread-sync.ts` to save/restore the `threadMap` (local ID to server ID
mapping) in localStorage. Without this, the mapping is lost on reload and the
frontend can't correlate thread list items with server sessions.

```typescript
const STORAGE_KEY = "nanobot-thread-map";

// On startup: restore from localStorage
const stored = localStorage.getItem(STORAGE_KEY);
const threadMap = new Map<string, string>(stored ? JSON.parse(stored) : []);

// After each mapping update: persist
function persistMap() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify([...threadMap]));
}
```

---

## File Changes

| Action | Path | Responsibility |
|--------|------|---------------|
| Modify | `nanobot/web/routes.py` | Add `GET /api/threads/{thread_id}/messages` endpoint |
| Modify | `nanobot/web/models.py` | Add `ThreadMessagesResponse` model |
| Modify | `frontend/src/App.tsx` | Add `history` adapter to `useDataStreamRuntime` |
| Modify | `frontend/src/lib/thread-sync.ts` | Persist threadMap to localStorage |

---

## What This Does NOT Change

- **Session storage** — messages continue to be saved via the existing chat flow
- **Thread list** — `/api/threads` endpoint and `ThreadListPrimitive` unchanged
- **Chat streaming** — `/api/chat` SSE flow unchanged
- **Thread creation/deletion** — existing endpoints unchanged
- **thread-sync.ts fetch interceptor** — stays, still needed for first-message ID sync

---

## Edge Cases

- **Empty thread:** User creates a thread but hasn't sent a message yet. Endpoint
  returns `{ messages: [] }`. UI shows empty chat correctly.
- **Thread with only tool calls:** All messages filtered out. Returns empty array.
  UI shows empty chat — acceptable (the agent was working but produced no text).
- **Very long threads:** No pagination in v1. Session history is bounded by the
  consolidation window (default 100 messages). If needed later, add `?limit=N&offset=M`.
- **Concurrent sessions:** User has the web UI open in two tabs. Both use the same
  session files. No conflict — messages are appended, and `load()` reads the latest state.

## Testing

- Backend: test the new endpoint with a pre-populated session (user + assistant +
  system + tool messages), verify only user/assistant returned in correct format
- Frontend: manual test — send messages, reload page, click thread, verify history loads
