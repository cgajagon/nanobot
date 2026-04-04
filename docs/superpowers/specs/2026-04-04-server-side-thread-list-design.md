# Server-Side Thread List with LLM Title Generation

> Design spec for connecting the web UI thread list to server-persisted sessions
> with automatic LLM-generated thread titles.
>
> Date: 2026-04-04

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Root Cause Analysis](#root-cause-analysis)
3. [Chosen Approach](#chosen-approach)
4. [Server-Side Changes](#server-side-changes)
5. [Frontend Changes](#frontend-changes)
6. [Data Flow](#data-flow)
7. [Error Handling and Edge Cases](#error-handling-and-edge-cases)
8. [Files to Create, Modify, and Delete](#files-to-create-modify-and-delete)
9. [Testing Strategy](#testing-strategy)

---

<a id="problem-statement"></a>
## 1. Problem Statement

The web UI thread list does not display previous sessions after a gateway restart, and
all threads show "New Chat" instead of meaningful titles.

**Expected behavior:** When the gateway restarts, the thread list sidebar should display
all previous web sessions with descriptive titles (e.g., "Debugging the auth middleware"
instead of "New Chat").

**Actual behavior:** The thread list only shows threads created in the current browser
session. All threads display "New Chat" as their title. Previous sessions are preserved
on disk but invisible to the user.

---

<a id="root-cause-analysis"></a>
## 2. Root Cause Analysis

The issue has two independent causes:

### Cause 1: No Server-Side Thread List Integration

The frontend uses `useDataStreamRuntime` from `@assistant-ui/react-data-stream` (App.tsx:227).
This runtime manages threads **client-side only** — thread metadata lives in the browser's
memory/localStorage and is not populated from the server.

The server has a `GET /api/threads` endpoint (routes.py:307-330) that returns all web
sessions with their metadata. However, **nothing in the frontend calls this endpoint**
to populate the thread list. The `ThreadListPrimitive` component renders threads from
the assistant-ui runtime's internal state, which starts empty on page load.

### Cause 2: No Title Metadata

The `ThreadListItemPrimitive.Title` component (thread-list.tsx:67) reads the title from
the assistant-ui runtime's thread state. When no title is available, it falls back to
the `fallback="New Chat"` prop.

The server computes titles on the fly in `_thread_title()` (routes.py:48-56) by
extracting the first user message, but this value is never stored in session metadata
and never reaches the frontend's thread list state.

### Why the Existing Architecture Cannot Fix This

The current frontend uses a `thread-sync.ts` fetch interceptor that patches
`window.fetch` to map local thread IDs to server UUIDs. This is a workaround for
thread ID synchronization but does not address thread list population or title
management. The interceptor operates at the wrong level of abstraction — it patches
individual requests rather than integrating with the library's thread list runtime.

---

<a id="chosen-approach"></a>
## 3. Chosen Approach

**`RemoteThreadListAdapter` + Server-Side Title Generation**

Replace the client-side-only thread management with `useRemoteThreadListRuntime`
wrapping `useDataStreamRuntime`. Implement a `RemoteThreadListAdapter` that calls
existing server endpoints plus two new ones (title generation and rename). The server
becomes the single source of truth for thread metadata.

### Why This Approach

- **Library-native**: `useRemoteThreadListRuntime` is the intended extension point in
  assistant-ui for server-backed thread lists. It handles local-to-remote ID mapping,
  optimistic UI updates, and thread lifecycle management.
- **Single source of truth**: Thread metadata (titles, timestamps, status) lives on the
  server in session JSONL files. No localStorage sync issues, no cross-tab divergence.
- **Eliminates the fetch interceptor**: The `thread-sync.ts` fetch interceptor that
  patches `window.fetch` is replaced by the library's native ID mapping via
  `initialize()`. This removes a fragile global side effect.

### Alternatives Considered

**Hybrid polling approach** — Keep `useDataStreamRuntime`, add a separate mechanism to
fetch `GET /api/threads` on page load and programmatically create threads in the runtime.
Rejected because it fights the library's design, creates two sources of truth for thread
state, and makes title sync a polling problem.

**LocalStorage adapter with server sync** — Use assistant-ui's built-in
`createLocalStorageAdapter` with a custom storage backend that syncs to the server.
Rejected because dual storage (localStorage + server) creates sync conflicts, and titles
can diverge between devices/browsers.

---

<a id="server-side-changes"></a>
## 4. Server-Side Changes

### 4a. Title Storage

Store generated titles in `session.metadata["title"]`. The `Session.metadata` field
(session/manager.py:52) is a `dict[str, Any]` that is already persisted in the JSONL
metadata line and loaded on restart. No schema changes or migrations needed.

### 4b. Title Generation Endpoint

**New endpoint:** `POST /api/threads/{thread_id}/generate-title`

**Behavior:**
1. Load session via `session_manager.get_or_create(f"web:{thread_id}")`
2. Extract the first user message and first assistant response from `session.messages`
3. Make a one-shot LLM call via `agent_loop.provider.chat()`:
   - Messages: A single user message with a prompt like:
     ```
     Generate a short, descriptive title (5-10 words) for this conversation.
     Do not use quotes or punctuation. Just the title text.

     User: {first_user_message}
     Assistant: {first_assistant_response_truncated}
     ```
   - Temperature: 0.7
   - Max tokens: 30
4. Store the result in `session.metadata["title"]`
5. Call `session_manager.save(session)` to persist
6. Return `{"title": "the generated title"}`

**LLM provider access:** Available via `request.app.state.agent_loop.provider`
(wired in web/app.py). The model to use is `request.app.state.agent_loop.model`.

**Response model:** `GenerateTitleResponse(BaseModel)` with a single `title: str` field.

### 4c. Rename Endpoint

**New endpoint:** `PATCH /api/threads/{thread_id}`

**Request body:** `{"title": "new title"}` — Pydantic model `RenameThreadRequest`
with a single `title: str` field.

**Behavior:**
1. Load session via `session_manager.get_or_create(f"web:{thread_id}")`
2. Set `session.metadata["title"] = request.title`
3. Call `session_manager.save(session)` to persist
4. Return `{"status": "ok"}`

This supports the `RemoteThreadListAdapter.rename()` method.

### 4d. Update `_thread_title()`

Modify the existing `_thread_title()` function (routes.py:48-56) to check session
metadata first:

```python
def _thread_title(session: object) -> str:
    metadata: dict = getattr(session, "metadata", {})
    if title := metadata.get("title"):
        return title
    # Existing fallback: first user message, truncated to 50 chars
    messages: list[dict] = getattr(session, "messages", [])
    for m in messages:
        if m.get("role") == "user":
            text: str = m.get("content", "")
            if len(text) > 50:
                return text[:50] + "..."
            return text or "New Chat"
    return "New Chat"
```

This means `GET /api/threads` automatically returns LLM-generated titles for sessions
that have them, and falls back to the first user message for older sessions.

### 4e. Update `GET /api/threads` Response

No structural changes to the endpoint. The existing `ThreadInfo` model already has a
`title: str` field. The only change is that `_thread_title()` now prefers
`metadata["title"]` over the first user message.

### 4f. No Changes to Streaming

Title generation is triggered by the frontend (via the adapter's `generateTitle()`
method), not by the SSE stream. The `streaming.py` file remains unchanged. The
assistant-ui library calls `generateTitle()` automatically after the first exchange
when the thread has no title.

---

<a id="frontend-changes"></a>
## 5. Frontend Changes

### 5a. New File: `frontend/src/lib/thread-list-adapter.ts`

Implements the `RemoteThreadListAdapter` interface from `@assistant-ui/core`. Each
method maps to a server endpoint:

#### `list(): Promise<RemoteThreadListResponse>`

```typescript
async list() {
  const response = await fetch("/api/threads");
  const data = await response.json();
  return {
    threads: data.threads.map((t) => ({
      remoteId: t.threadId,
      status: "regular" as const,
      title: t.title === "New Chat" ? undefined : t.title,
    })),
  };
}
```

Maps the server's `ThreadInfo` to `RemoteThreadMetadata`. Threads with title
"New Chat" have `title: undefined` so the library knows to call `generateTitle()`
when messages become available.

#### `initialize(threadId: string): Promise<RemoteThreadInitializeResponse>`

```typescript
async initialize(_threadId: string) {
  const response = await fetch("/api/threads", { method: "POST" });
  const data = await response.json();
  return { remoteId: data.threadId, externalId: undefined };
}
```

Called by the library when a new thread is created (user clicks "New Thread" or sends
the first message). Creates a server-side session and returns the UUID that the library
maps to the local thread ID.

#### `generateTitle(remoteId: string, messages: readonly ThreadMessage[]): Promise<AssistantStream>`

```typescript
async generateTitle(remoteId: string, _messages: readonly ThreadMessage[]) {
  const response = await fetch(`/api/threads/${remoteId}/generate-title`, {
    method: "POST",
  });
  const data = await response.json();
  const title = data.title || "New Chat";

  // Wrap the result in a single-chunk AssistantStream
  // The library expects a ReadableStream that produces text parts
  return new ReadableStream({
    start(controller) {
      controller.enqueue({
        type: "text-delta",
        textDelta: title,
      });
      controller.enqueue({
        type: "finish",
        finishReason: "stop",
      });
      controller.close();
    },
  });
}
```

The server does the actual LLM call. The adapter just wraps the response in the
`AssistantStream` format the library expects. We ignore the `messages` parameter
because the server already has the session messages.

**Note on `AssistantStream` format:** The library processes this stream via
`AssistantMessageStream.fromAssistantStream()`, which async-iterates over chunks
looking for text parts. A single `text-delta` chunk followed by a `finish` chunk
is sufficient for a non-streaming title.

#### `rename(remoteId: string, newTitle: string): Promise<void>`

```typescript
async rename(remoteId: string, newTitle: string) {
  await fetch(`/api/threads/${remoteId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: newTitle }),
  });
}
```

#### `delete(remoteId: string): Promise<void>`

```typescript
async delete(remoteId: string) {
  await fetch(`/api/threads/${remoteId}`, { method: "DELETE" });
}
```

Uses the existing `DELETE /api/threads/{id}` endpoint.

#### `archive(remoteId: string): Promise<void>` / `unarchive(remoteId: string): Promise<void>`

No-ops for now. The server has no archive concept. The adapter methods are required by
the interface but do nothing:

```typescript
async archive(_remoteId: string) {}
async unarchive(_remoteId: string) {}
```

#### `fetch(threadId: string): Promise<RemoteThreadMetadata>`

Called when switching to a thread not in the initial list (e.g., created in another tab).
Makes a lightweight call to get thread metadata:

```typescript
async fetch(threadId: string) {
  // Re-fetch the full list and find the thread
  // (no dedicated single-thread endpoint exists)
  const response = await fetch("/api/threads");
  const data = await response.json();
  const thread = data.threads.find((t) => t.threadId === threadId);
  return {
    remoteId: threadId,
    status: "regular" as const,
    title: thread?.title === "New Chat" ? undefined : thread?.title,
  };
}
```

### 5b. Modify `frontend/src/App.tsx`

**Replace the runtime setup:**

Current (lines 227-243):
```tsx
const runtime = useDataStreamRuntime({
  api: "/api/chat",
  protocol: "ui-message-stream",
  onResponse: (response) => { ... },
  adapters: {
    attachments: new CompositeAttachmentAdapter([...]),
    history: serverHistoryAdapter,
  },
});
```

New:
```tsx
import { useRemoteThreadListRuntime } from "@assistant-ui/react";
import { threadListAdapter } from "@/lib/thread-list-adapter";

// Inside App component:
const runtime = useRemoteThreadListRuntime({
  runtimeHook: () =>
    useDataStreamRuntime({
      api: "/api/chat",
      protocol: "ui-message-stream",
      onResponse: (response) => { ... },
      adapters: {
        attachments: new CompositeAttachmentAdapter([...]),
        history: serverHistoryAdapter,
      },
    }),
  adapter: threadListAdapter,
});
```

**Remove imports:**
- Remove `ThreadHistorySync` import from `@/lib/thread-history`
- Remove `<ThreadHistorySync />` component from the JSX tree (line 247)

### 5c. Simplify `frontend/src/lib/thread-history.ts`

**Remove:**
- The `_currentThreadRemoteId` mutable variable (line 29)
- The `ThreadHistorySync` component (lines 115-123) — the library provides the
  remoteId through its own thread list item state
- The synchronous-render hack (the comment block explaining the race condition)

**Simplify `serverHistoryAdapter.load()`:**

The history adapter needs to read the current thread's `remoteId` from the library's
state instead of the mutable variable. The exact mechanism depends on what assistant-ui
exposes — the adapter may need to accept the remoteId as a parameter from the runtime,
or read it via the `useThreadListItem()` hook context.

If the library doesn't provide a clean way to pass remoteId to `load()`, we keep a
simplified version of the mutable variable pattern but without the `ThreadHistorySync`
component — the `useRemoteThreadListRuntime` manages thread state internally.

**Keep:**
- `fetchThreadMessages()` function — still needed to load messages from
  `GET /api/threads/{id}/messages`
- `serverHistoryAdapter` export — still passed to `useDataStreamRuntime`

### 5d. Delete `frontend/src/lib/thread-sync.ts`

The entire file is removed. Its responsibilities are replaced by:

- **Thread ID mapping** (local -> server UUID): Handled by the library's
  `initialize()` call, which maps `__LOCALID_xxx` to the server's UUID internally
- **Fetch interceptor**: No longer needed — the library sends the correct threadId
  in requests after `initialize()` completes
- **localStorage persistence** (`nanobot-thread-map`): Orphaned. The library manages
  its own ID mapping. The old localStorage key (`nanobot-thread-map`) becomes inert
  data — no cleanup needed.

### 5e. No Changes to `frontend/src/components/thread-list.tsx`

The component already uses `ThreadListItemPrimitive.Title` with `fallback="New Chat"`.
With the new runtime, the `Title` component reads the title from
`RemoteThreadMetadata.title` (provided by `adapter.list()`). When the title is
`undefined`, it shows "New Chat" and the library triggers `generateTitle()` when
messages become available. No component changes needed.

---

<a id="data-flow"></a>
## 6. Data Flow

### 6a. Page Load (Restoring Previous Sessions)

```
Browser loads page
  -> React mounts App component
  -> useRemoteThreadListRuntime initializes
  -> Library calls adapter.list() once on mount
  -> adapter.list() fetches GET /api/threads
  -> Server reads all web:* session JSONL files
  -> For each session: check metadata["title"], fall back to first user message
  -> Returns ThreadListResponse with thread metadata
  -> adapter.list() maps to RemoteThreadMetadata[]
  -> Library populates internal thread list state
  -> ThreadListPrimitive renders all threads in sidebar
  -> Each ThreadListItemPrimitive.Title shows:
     - metadata["title"] if LLM-generated title exists
     - First user message (truncated) if no generated title
     - "New Chat" if no messages exist
```

### 6b. New Conversation

```
User clicks "New Thread"
  -> Library generates local ID __LOCALID_xxx
  -> Thread appears in sidebar as "New Chat"

User types and sends first message
  -> Library detects thread status === "new"
  -> Library calls adapter.initialize(__LOCALID_xxx)
  -> adapter.initialize() calls POST /api/threads
  -> Server creates session web:{uuid}, saves empty JSONL
  -> Returns {threadId: uuid}
  -> adapter returns {remoteId: uuid, externalId: undefined}
  -> Library maps __LOCALID_xxx -> uuid for all future operations

  -> useDataStreamRuntime sends POST /api/chat with threadId=uuid
  -> Server processes message through agent pipeline
  -> SSE stream delivers response to browser
  -> Stream completes with [DONE]

  -> Library detects thread has no title and messages are available
  -> Library calls adapter.generateTitle(uuid, messages)
  -> adapter calls POST /api/threads/{uuid}/generate-title
  -> Server loads session, extracts first exchange
  -> Server makes one-shot LLM call with title generation prompt
  -> Server stores title in session.metadata["title"], saves session
  -> Returns {title: "Debugging the auth middleware"}
  -> adapter wraps in single-chunk AssistantStream
  -> Library updates thread title in sidebar: "New Chat" -> "Debugging the auth middleware"
```

### 6c. Returning to a Previous Thread

```
User clicks existing thread in sidebar
  -> Library calls switchToThread(threadId)
  -> runtimeHook() is called to get thread runtime
  -> useDataStreamRuntime creates runtime for this thread
  -> History adapter's load() is called
  -> load() reads remoteId from thread list item state
  -> Fetches GET /api/threads/{remoteId}/messages
  -> Server loads session, filters to user/assistant messages
  -> Returns ThreadMessagesResponse
  -> History adapter converts to ExportedMessageRepository
  -> Thread renders with full conversation history
  -> User can continue the conversation in the same session
```

### 6d. Gateway Restart Recovery

```
Gateway process stops
  -> Session JSONL files persist on disk unchanged
  -> metadata["title"] values preserved in JSONL first line

Gateway process restarts
  -> SessionManager initializes with workspace/sessions/ directory

Browser reloads (or new browser session)
  -> useRemoteThreadListRuntime mounts
  -> adapter.list() fetches GET /api/threads
  -> Server reads all JSONL files from disk
  -> Sessions with metadata["title"] return their LLM-generated title
  -> Sessions without metadata["title"] return first user message (truncated)
  -> Thread list populates with all previous sessions and their titles
```

---

<a id="error-handling-and-edge-cases"></a>
## 7. Error Handling and Edge Cases

### 7a. Title Generation Failures

**LLM call fails** (rate limit, timeout, API error):
- The `POST /api/threads/{id}/generate-title` endpoint catches the exception and
  returns an HTTP error (500)
- The frontend adapter catches the error — the thread keeps showing the fallback
  title (first user message truncated to 50 chars from `GET /api/threads`)
- No retry mechanism. The user can still use the thread normally
- If the user wants a title, they can trigger a manual rename (future feature)

**Empty session** (no user messages):
- The endpoint returns early with a 200 and `{"title": "New Chat"}`
- This is stored in metadata — `GET /api/threads` returns it as the title
- When the user sends a message and the library calls `generateTitle()` again,
  it overwrites the placeholder

**LLM returns empty or unusable content**:
- If `response.content` is empty or None, use the first-user-message fallback
- If the title is excessively long (> 100 chars), truncate to 100 chars

### 7b. Thread Sync Edge Cases

**Browser tab open during gateway restart:**
- `adapter.list()` only runs on mount. If the gateway restarts while a tab is open,
  in-progress conversations may fail (the agent loop is gone)
- Existing threads continue to appear in the sidebar (the runtime state persists
  in memory until the page is reloaded)
- On page reload, `adapter.list()` re-fetches from the restarted server and all
  persisted sessions appear

**Two browser tabs:**
- Both tabs share the same server sessions via API calls
- Thread operations (create, delete, rename) from one tab won't auto-reflect in
  the other until page reload
- No WebSocket sync needed for v1 — acceptable trade-off

**`initialize()` fails** (server down or network error):
- The library handles this gracefully — the thread stays in "new" status
- The user sees an error when trying to send a message
- Messages are not lost — `useDataStreamRuntime` buffers locally
- On retry, `initialize()` creates the server session

**`list()` fails** (server down on page load):
- The thread list shows a loading skeleton, then renders empty
- The user can still create new threads (which calls `initialize()`)
- No automatic retry — the user reloads the page when the server is back

### 7c. Migration from Current Implementation

**Existing sessions without titles:**
- `GET /api/threads` falls back to the first user message (truncated to 50 chars)
  via the updated `_thread_title()` function
- These sessions appear in the thread list with their fallback title
- No automatic title generation for existing sessions — titles are only generated
  for new conversations going forward

**Existing `nanobot-thread-map` in localStorage:**
- The `thread-sync.ts` fetch interceptor stored a map of local-to-server thread IDs
  in localStorage under the key `nanobot-thread-map`
- Removing `thread-sync.ts` makes this data orphaned but harmless
- No cleanup code needed — it's inert

**Existing in-memory thread state in assistant-ui:**
- On the first page load after the migration, the library's internal state starts
  fresh and is populated from `adapter.list()`
- Any threads that were only in localStorage (never persisted to server) will
  be lost — but this is expected since those threads had no server-side session

### 7d. `AssistantStream` Format

The `generateTitle()` method must return a `ReadableStream` that the library processes
via `AssistantMessageStream.fromAssistantStream()`. The library expects chunks with:
- `{type: "text-delta", textDelta: "the title text"}` — text content
- `{type: "finish", finishReason: "stop"}` — stream termination

A single `text-delta` chunk followed by a `finish` chunk is sufficient for a
non-streaming title. The library reads text parts from each chunk and uses the last
one as the title.

If the stream format is wrong (unrecognized chunk types), the library silently ignores
the chunks and the title remains `undefined` (fallback shows "New Chat"). This is safe
but means a format mismatch would be silent — testing is important.

---

<a id="files-to-create-modify-and-delete"></a>
## 8. Files to Create, Modify, and Delete

### New Files

| File | Purpose |
|------|---------|
| `frontend/src/lib/thread-list-adapter.ts` | `RemoteThreadListAdapter` implementation — maps each adapter method to a server API call |

### Modified Files

| File | Changes |
|------|---------|
| `nanobot/web/routes.py` | (1) Add `POST /api/threads/{thread_id}/generate-title` endpoint — loads session, calls LLM, stores title in metadata. (2) Add `PATCH /api/threads/{thread_id}` endpoint — updates title in session metadata. (3) Update `_thread_title()` to check `session.metadata.get("title")` before falling back to first user message. |
| `nanobot/web/models.py` | Add `GenerateTitleResponse(BaseModel)` with `title: str` field. Add `RenameThreadRequest(BaseModel)` with `title: str` field. |
| `frontend/src/App.tsx` | (1) Replace `useDataStreamRuntime` with `useRemoteThreadListRuntime` wrapping it via the `runtimeHook` pattern. (2) Import `threadListAdapter` from new file. (3) Remove `ThreadHistorySync` import and `<ThreadHistorySync />` JSX element. (4) Remove `installThreadSync` import if present. |
| `frontend/src/lib/thread-history.ts` | (1) Remove `_currentThreadRemoteId` mutable variable. (2) Remove `ThreadHistorySync` component export. (3) Simplify `serverHistoryAdapter.load()` to read remoteId from thread list item state instead of the mutable variable. (4) Keep `fetchThreadMessages()` and the core adapter export. |

### Deleted Files

| File | Reason |
|------|--------|
| `frontend/src/lib/thread-sync.ts` | Fetch interceptor fully replaced by library-native ID mapping via `adapter.initialize()`. Thread ID synchronization, localStorage persistence, and the `pendingServerThreadId` mechanism are all handled internally by `useRemoteThreadListRuntime`. |

### Unchanged Files

| File | Why Unchanged |
|------|--------------|
| `frontend/src/components/thread-list.tsx` | Already uses `ThreadListItemPrimitive.Title` with `fallback="New Chat"` — works as-is with the new runtime providing titles |
| `nanobot/session/manager.py` | `Session.metadata` is already a `dict[str, Any]` that persists to JSONL — no changes needed to support the `title` key |
| `nanobot/web/streaming.py` | Title generation is triggered by the frontend adapter, not the SSE stream — streaming logic untouched |
| `nanobot/channels/web.py` | WebChannel streaming mechanics unchanged |
| `nanobot/web/app.py` | App state setup already exposes `agent_loop` and `session_manager` on `request.app.state` |

---

<a id="testing-strategy"></a>
## 9. Testing Strategy

### Server-Side Tests

**Title generation endpoint:**
- Test with a session that has user+assistant messages — verify LLM is called and
  title is stored in metadata
- Test with an empty session — verify graceful fallback
- Test with a session that already has a title — verify it overwrites
- Test LLM failure — verify error response, no metadata corruption

**Rename endpoint:**
- Test successful rename — verify metadata updated and persisted
- Test rename of non-existent thread — verify appropriate error

**Updated `_thread_title()`:**
- Test session with `metadata["title"]` — returns the metadata title
- Test session without metadata title but with user messages — returns first user
  message truncated
- Test session with no messages — returns "New Chat"

**`GET /api/threads` integration:**
- Test that threads with generated titles return those titles
- Test that threads without generated titles fall back to first user message

### Frontend Tests

**`thread-list-adapter.ts`:**
- Test `list()` — mock `fetch`, verify correct mapping to `RemoteThreadMetadata`
- Test `initialize()` — mock `fetch`, verify POST call and return shape
- Test `generateTitle()` — mock `fetch`, verify stream format
- Test `delete()` — mock `fetch`, verify DELETE call
- Test `rename()` — mock `fetch`, verify PATCH call with correct body
- Test error handling in each method — verify graceful failure

**`App.tsx` integration:**
- Verify `useRemoteThreadListRuntime` is used instead of `useDataStreamRuntime` alone
- Verify `ThreadHistorySync` is no longer rendered

**Removal verification:**
- Verify `thread-sync.ts` is deleted and no imports reference it
- Verify `installThreadSync()` is not called anywhere
