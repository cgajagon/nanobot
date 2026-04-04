# Server-Side Thread List Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect the web UI thread list to server-persisted sessions so previous conversations appear after gateway restart, with automatic LLM-generated titles.

**Architecture:** Replace client-side-only thread management (`useDataStreamRuntime`) with `useRemoteThreadListRuntime` wrapping it, backed by a `RemoteThreadListAdapter` that calls server endpoints. Title generation uses a new server endpoint that makes a one-shot LLM call. The `thread-sync.ts` fetch interceptor is removed in favor of the library's native ID mapping.

**Tech Stack:** FastAPI (Python backend), assistant-ui `@assistant-ui/react` ^0.12.17 (React frontend), Pydantic models, existing `LLMProvider` for title generation.

**Spec:** `docs/superpowers/specs/2026-04-04-server-side-thread-list-design.md`

---

## File Map

### New Files

| File | Responsibility |
|------|---------------|
| `frontend/src/lib/thread-list-adapter.ts` | `RemoteThreadListAdapter` implementation — maps adapter methods to server API calls |

### Modified Files

| File | Changes |
|------|---------|
| `nanobot/web/models.py` | Add `GenerateTitleResponse` and `RenameThreadRequest` Pydantic models |
| `nanobot/web/routes.py` | Add `PATCH /api/threads/{id}` rename endpoint, add `POST /api/threads/{id}/generate-title` endpoint, update `_thread_title()` to check metadata first |
| `frontend/src/App.tsx` | Replace `useDataStreamRuntime` with `useRemoteThreadListRuntime`, remove `ThreadHistorySync` |
| `frontend/src/lib/thread-history.ts` | Remove `ThreadHistorySync` component and `_currentThreadRemoteId` hack, simplify `load()` |
| `frontend/src/main.tsx` | Remove `installThreadSync()` call and import |

### Deleted Files

| File | Reason |
|------|--------|
| `frontend/src/lib/thread-sync.ts` | Replaced by library-native ID mapping via `adapter.initialize()` |

---

## Task 1: Add Pydantic Models for New Endpoints

**Files:**
- Modify: `nanobot/web/models.py`

- [ ] **Step 1: Add `GenerateTitleResponse` and `RenameThreadRequest` models**

Add these two models at the end of `nanobot/web/models.py`, after the `ThreadMessagesResponse` class:

```python
class GenerateTitleResponse(BaseModel):
    """Response from the title generation endpoint."""

    title: str


class RenameThreadRequest(BaseModel):
    """Request body for renaming a thread."""

    title: str
```

- [ ] **Step 2: Run lint and typecheck**

Run: `make lint && make typecheck`
Expected: PASS (no errors)

- [ ] **Step 3: Commit**

```bash
git add nanobot/web/models.py
git commit -m "feat(web): add Pydantic models for thread title endpoints"
```

---

## Task 2: Update `_thread_title()` to Check Metadata First

**Files:**
- Modify: `nanobot/web/routes.py:48-56`

- [ ] **Step 1: Update `_thread_title()` function**

Replace the existing `_thread_title()` function in `nanobot/web/routes.py` (lines 48-56) with:

```python
def _thread_title(session: object) -> str:
    metadata: dict = getattr(session, "metadata", {})
    if title := metadata.get("title"):
        return title
    messages: list[dict] = getattr(session, "messages", [])
    for m in messages:
        if m.get("role") == "user":
            text: str = m.get("content", "")
            if len(text) > 50:
                return text[:50] + "..."
            return text or "New Chat"
    return "New Chat"
```

- [ ] **Step 2: Run lint and typecheck**

Run: `make lint && make typecheck`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add nanobot/web/routes.py
git commit -m "feat(web): _thread_title() checks session metadata before fallback"
```

---

## Task 3: Add Rename Endpoint (`PATCH /api/threads/{id}`)

**Files:**
- Modify: `nanobot/web/routes.py`

- [ ] **Step 1: Add import for `RenameThreadRequest`**

In `nanobot/web/routes.py`, update the import from `nanobot.web.models` (lines 18-24) to include the new model:

```python
from nanobot.web.models import (
    ChatMessage,
    ChatRequest,
    GenerateTitleResponse,
    HistoryMessage,
    HistoryResponse,
    RenameThreadRequest,
    ThreadInfo,
    ThreadListResponse,
)
```

- [ ] **Step 2: Add the PATCH endpoint**

Add the following endpoint in `nanobot/web/routes.py`, after the `delete_thread` handler (after line 357):

```python
@router.patch("/threads/{thread_id}")
async def rename_thread(request: Request, thread_id: str, body: RenameThreadRequest):
    """Rename a thread by updating its title in session metadata."""
    session_manager = request.app.state.session_manager
    session_key = _session_key(thread_id)
    session = session_manager.get_or_create(session_key)
    session.metadata["title"] = body.title
    session_manager.save(session)
    return {"status": "ok"}
```

- [ ] **Step 3: Run lint and typecheck**

Run: `make lint && make typecheck`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add nanobot/web/routes.py
git commit -m "feat(web): add PATCH /api/threads/{id} endpoint for renaming threads"
```

---

## Task 4: Add Title Generation Endpoint (`POST /api/threads/{id}/generate-title`)

**Files:**
- Modify: `nanobot/web/routes.py`

- [ ] **Step 1: Add the generate-title endpoint**

Add the following endpoint in `nanobot/web/routes.py`, after the `rename_thread` handler:

```python
@router.post("/threads/{thread_id}/generate-title")
async def generate_title(request: Request, thread_id: str):
    """Generate an LLM-powered title for a thread from its first exchange."""
    session_manager = request.app.state.session_manager
    agent_loop = request.app.state.agent_loop

    session_key = _session_key(thread_id)
    session = session_manager.get_or_create(session_key)

    # Extract first user message and first assistant response
    first_user = ""
    first_assistant = ""
    for m in session.messages:
        if m.get("role") == "user" and not first_user:
            content = m.get("content", "")
            first_user = content if isinstance(content, str) else str(content)
        elif m.get("role") == "assistant" and first_user and not first_assistant:
            content = m.get("content", "")
            first_assistant = content if isinstance(content, str) else str(content)
            break

    if not first_user:
        return GenerateTitleResponse(title="New Chat")

    # Truncate to keep the prompt compact
    user_snippet = first_user[:300]
    assistant_snippet = first_assistant[:300] if first_assistant else ""

    prompt_parts = [
        "Generate a short, descriptive title (5-10 words) for this conversation.",
        "Do not use quotes or punctuation around the title. Just the title text.",
        "",
        f"User: {user_snippet}",
    ]
    if assistant_snippet:
        prompt_parts.append(f"Assistant: {assistant_snippet}")

    try:
        response = await agent_loop.provider.chat(
            messages=[{"role": "user", "content": "\n".join(prompt_parts)}],
            model=agent_loop.model,
            temperature=0.7,
            max_tokens=30,
        )
        title = (response.content or "").strip()
        if not title or len(title) > 100:
            title = first_user[:50] + ("..." if len(first_user) > 50 else "")
    except Exception:  # crash-barrier: title generation is non-critical
        logger.warning("Title generation failed for thread {}", thread_id)
        title = first_user[:50] + ("..." if len(first_user) > 50 else "")

    session.metadata["title"] = title
    session_manager.save(session)
    return GenerateTitleResponse(title=title)
```

- [ ] **Step 2: Run lint and typecheck**

Run: `make lint && make typecheck`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add nanobot/web/routes.py
git commit -m "feat(web): add POST /api/threads/{id}/generate-title endpoint"
```

---

## Task 5: Create `RemoteThreadListAdapter`

**Files:**
- Create: `frontend/src/lib/thread-list-adapter.ts`

- [ ] **Step 1: Create the adapter file**

Create `frontend/src/lib/thread-list-adapter.ts` with the following content:

```typescript
/**
 * RemoteThreadListAdapter implementation for nanobot.
 *
 * Maps assistant-ui's thread list adapter interface to the nanobot server API.
 * Each method calls a server endpoint — the server is the single source of
 * truth for thread metadata (titles, timestamps, status).
 */

import type { RemoteThreadListAdapter } from "@assistant-ui/react";

/** Response shape from GET /api/threads. */
interface ServerThreadInfo {
  threadId: string;
  title: string;
  createdAt: string | null;
  updatedAt: string | null;
}

interface ServerThreadListResponse {
  threads: ServerThreadInfo[];
}

export const threadListAdapter: RemoteThreadListAdapter = {
  async list() {
    const response = await fetch("/api/threads");
    if (!response.ok) {
      return { threads: [] };
    }
    const data: ServerThreadListResponse = await response.json();
    return {
      threads: data.threads.map((t) => ({
        remoteId: t.threadId,
        status: "regular" as const,
        title: t.title === "New Chat" ? undefined : t.title,
      })),
    };
  },

  async initialize(_threadId: string) {
    const response = await fetch("/api/threads", { method: "POST" });
    const data = await response.json();
    return { remoteId: data.threadId as string, externalId: undefined };
  },

  async generateTitle(remoteId: string) {
    const response = await fetch(`/api/threads/${remoteId}/generate-title`, {
      method: "POST",
    });
    if (!response.ok) {
      // Return a minimal stream that produces no title update
      return new ReadableStream({
        start(controller) {
          controller.close();
        },
      });
    }
    const data = await response.json();
    const title = data.title || "New Chat";

    // Wrap the result in a single-chunk AssistantStream.
    // The library processes this via AssistantMessageStream.fromAssistantStream(),
    // extracting text parts from each chunk.
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
  },

  async rename(remoteId: string, newTitle: string) {
    await fetch(`/api/threads/${remoteId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: newTitle }),
    });
  },

  async delete(remoteId: string) {
    await fetch(`/api/threads/${remoteId}`, { method: "DELETE" });
  },

  async archive(_remoteId: string) {
    // No archive concept on the server yet — no-op.
  },

  async unarchive(_remoteId: string) {
    // No unarchive concept on the server yet — no-op.
  },

  async fetch(threadId: string) {
    // Re-fetch the full list and find this thread.
    // A dedicated single-thread endpoint could be added later for efficiency.
    const response = await fetch("/api/threads");
    if (!response.ok) {
      return { remoteId: threadId, status: "regular" as const, title: undefined };
    }
    const data: ServerThreadListResponse = await response.json();
    const thread = data.threads.find((t) => t.threadId === threadId);
    return {
      remoteId: threadId,
      status: "regular" as const,
      title:
        thread?.title && thread.title !== "New Chat"
          ? thread.title
          : undefined,
    };
  },
};
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: PASS (or only pre-existing errors unrelated to new file)

- [ ] **Step 3: Commit**

```bash
git add frontend/src/lib/thread-list-adapter.ts
git commit -m "feat(web): add RemoteThreadListAdapter for server-backed thread list"
```

---

## Task 6: Simplify `thread-history.ts`

**Files:**
- Modify: `frontend/src/lib/thread-history.ts`

The `ThreadHistorySync` component and `_currentThreadRemoteId` mutable variable are
no longer needed — `useRemoteThreadListRuntime` manages thread state internally. The
history adapter still needs to load messages from the server, but it reads the remoteId
from the thread list item state provided by the library.

- [ ] **Step 1: Rewrite `thread-history.ts`**

Replace the entire contents of `frontend/src/lib/thread-history.ts` with:

```typescript
/**
 * Thread history adapter for assistant-ui.
 *
 * Loads historical messages from the server when switching to an existing thread.
 * The remoteId is read from the thread list item state provided by
 * useRemoteThreadListRuntime — no mutable variable hack needed.
 */

import { ExportedMessageRepository } from "@assistant-ui/react";
import type { ThreadHistoryAdapter } from "@assistant-ui/react";
import type { ExportedMessageRepositoryItem } from "@assistant-ui/react";

/** Response shape from GET /api/threads/{id}/messages. */
interface ThreadMessagesResponse {
  messages: Array<{
    id: string;
    role: "user" | "assistant";
    content: Array<{ type: string; text: string }>;
  }>;
}

/**
 * Fetch messages for a thread from the server and convert to assistant-ui format.
 */
async function fetchThreadMessages(
  serverThreadId: string,
): Promise<ExportedMessageRepository> {
  try {
    const response = await fetch(`/api/threads/${serverThreadId}/messages`);
    if (!response.ok) {
      return { messages: [] };
    }
    const data: ThreadMessagesResponse = await response.json();
    if (!data.messages || data.messages.length === 0) {
      return { messages: [] };
    }

    const messageLikes = data.messages.map((msg) => ({
      role: msg.role as "user" | "assistant",
      content: msg.content.map((c) => c.text).join("\n"),
      id: msg.id,
    }));

    return ExportedMessageRepository.fromArray(messageLikes);
  } catch {
    return { messages: [] };
  }
}

/**
 * Mutable variable tracking the remoteId of the thread being loaded.
 *
 * Updated by setCurrentThreadRemoteId() from the App component before the
 * history adapter's load() is called. This is necessary because the
 * ThreadHistoryAdapter.load() interface accepts no parameters.
 */
let _currentThreadRemoteId: string | undefined;

/**
 * Set the remoteId for the next history load.
 * Called from the App component's thread switch handler.
 */
export function setCurrentThreadRemoteId(remoteId: string | undefined): void {
  _currentThreadRemoteId = remoteId;
}

/**
 * ThreadHistoryAdapter implementation that loads messages from the server.
 */
export const serverHistoryAdapter: ThreadHistoryAdapter = {
  async load() {
    const remoteId = _currentThreadRemoteId;
    if (!remoteId) {
      return { messages: [] };
    }

    // Don't try to load history for brand-new local threads.
    if (remoteId.startsWith("__LOCALID_")) {
      return { messages: [] };
    }

    return fetchThreadMessages(remoteId);
  },

  async append(_item: ExportedMessageRepositoryItem) {
    // Messages are already persisted server-side by the chat endpoint.
  },
};
```

**Key changes from original:**
- Removed `ThreadHistorySync` component export
- Removed `getServerThreadId` import from `thread-sync`
- Kept `_currentThreadRemoteId` mutable variable (the `ThreadHistoryAdapter.load()` interface accepts no parameters, so we still need a way to pass the remoteId — but now it's set via an exported function instead of a React component)
- Added `setCurrentThreadRemoteId()` export for the App component to call

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: May show errors for App.tsx (still importing old exports) — that's fine, Task 8 will fix it.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/lib/thread-history.ts
git commit -m "refactor(web): simplify thread-history adapter, remove ThreadHistorySync"
```

---

## Task 7: Delete `thread-sync.ts` and Remove Its Usage

**Files:**
- Delete: `frontend/src/lib/thread-sync.ts`
- Modify: `frontend/src/main.tsx`

- [ ] **Step 1: Remove `installThreadSync` from `main.tsx`**

Edit `frontend/src/main.tsx` to remove the import and call. The file should become:

```typescript
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { TooltipProvider } from "@/components/ui/tooltip";
import "./index.css";
import App from "./App.tsx";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <TooltipProvider>
      <App />
    </TooltipProvider>
  </StrictMode>,
);
```

- [ ] **Step 2: Delete `thread-sync.ts`**

```bash
rm frontend/src/lib/thread-sync.ts
```

- [ ] **Step 3: Verify no remaining imports**

```bash
grep -rn "thread-sync" frontend/src/
```

Expected: Zero matches (thread-history.ts no longer imports from it after Task 6).

- [ ] **Step 4: Commit**

```bash
git add frontend/src/main.tsx
git rm frontend/src/lib/thread-sync.ts
git commit -m "refactor(web): remove thread-sync.ts fetch interceptor"
```

---

## Task 8: Wire `useRemoteThreadListRuntime` in `App.tsx`

**Files:**
- Modify: `frontend/src/App.tsx`

This is the critical integration task — replacing the runtime setup and removing the
old `ThreadHistorySync` component.

- [ ] **Step 1: Update imports**

In `frontend/src/App.tsx`, replace the current imports (lines 1-17) with:

```typescript
import { useState, type FC } from "react";
import { MenuIcon, PanelLeftIcon } from "lucide-react";
import {
  AssistantRuntimeProvider,
  CompositeAttachmentAdapter,
  SimpleImageAttachmentAdapter,
  SimpleTextAttachmentAdapter,
  useThreadListItem,
  type AttachmentAdapter,
} from "@assistant-ui/react";
import { useDataStreamRuntime } from "@assistant-ui/react-data-stream";
import { useRemoteThreadListRuntime } from "@assistant-ui/react";
import { Thread } from "@/components/thread";
import { ThreadList } from "@/components/thread-list";
import { TooltipIconButton } from "@/components/tooltip-icon-button";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { cn } from "@/lib/utils";
import { serverHistoryAdapter, setCurrentThreadRemoteId } from "@/lib/thread-history";
import { threadListAdapter } from "@/lib/thread-list-adapter";
```

**Key changes:**
- Added `useThreadListItem` and `useRemoteThreadListRuntime` imports
- Removed `ThreadHistorySync` import
- Added `setCurrentThreadRemoteId` import
- Added `threadListAdapter` import
- Removed `thread-sync` related imports

- [ ] **Step 2: Add `ThreadRemoteIdSync` component**

Add this component after the `readStatusEvents` function (before the `App` component), replacing the old `ThreadHistorySync`:

```typescript
/**
 * Syncs the current thread's remoteId to the history adapter.
 *
 * Reads remoteId from the thread list item state (provided by
 * useRemoteThreadListRuntime) and writes it to the history adapter's
 * mutable variable so load() can read it.
 */
function ThreadRemoteIdSync(): null {
  const threadListItem = useThreadListItem({ optional: true });
  setCurrentThreadRemoteId(threadListItem?.remoteId);
  return null;
}
```

- [ ] **Step 3: Replace the runtime setup in `App` component**

Replace the runtime initialization (the `const runtime = useDataStreamRuntime(...)` block) with:

```typescript
  const runtime = useRemoteThreadListRuntime({
    runtimeHook: () =>
      // eslint-disable-next-line react-hooks/rules-of-hooks
      useDataStreamRuntime({
        api: "/api/chat",
        protocol: "ui-message-stream",
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
          history: serverHistoryAdapter,
        },
      }),
    adapter: threadListAdapter,
  });
```

**Note:** The `runtimeHook` callback calls `useDataStreamRuntime` inside a non-component function. The `eslint-disable` comment is needed because this is an intentional pattern from the assistant-ui library — the hook is called inside a hook-compatible context managed by the library's internal `HookInstanceManager`.

- [ ] **Step 4: Replace `<ThreadHistorySync />` with `<ThreadRemoteIdSync />`**

In the JSX return, replace `<ThreadHistorySync />` (line 247) with:

```tsx
      <ThreadRemoteIdSync />
```

- [ ] **Step 5: Verify TypeScript compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: PASS

- [ ] **Step 6: Verify the dev server starts**

Run: `cd frontend && npx vite --host 127.0.0.1 &`
Expected: Vite dev server starts without build errors. Kill it after verification.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/App.tsx
git commit -m "feat(web): wire useRemoteThreadListRuntime for server-backed thread list"
```

---

## Task 9: Manual Integration Test

This task verifies the full flow end-to-end.

- [ ] **Step 1: Start the gateway**

```bash
nanobot gateway
```

Expected: Gateway starts, web UI accessible at the configured port.

- [ ] **Step 2: Open the web UI and verify thread list loads**

Open the web UI in a browser. The thread list sidebar should show any existing sessions
from `GET /api/threads`. Sessions with LLM-generated titles show those titles; sessions
without show the first user message (truncated); sessions with no messages show "New Chat".

- [ ] **Step 3: Start a new conversation**

Click "New Thread", type a message, send it. Verify:
- The message is sent and a response is received
- After the response completes, the thread title in the sidebar updates from "New Chat"
  to an LLM-generated title (may take 1-2 seconds)

- [ ] **Step 4: Restart the gateway and verify persistence**

Stop the gateway (Ctrl+C), restart it. Reload the browser page. Verify:
- The thread list shows the previous conversation
- The title is the LLM-generated title (not "New Chat")
- Clicking the thread loads the full conversation history

- [ ] **Step 5: Test thread deletion**

Hover over a thread in the sidebar, click the "..." menu, click "Delete". Verify:
- The thread disappears from the sidebar
- The session file is removed from disk

- [ ] **Step 6: Run structural checks**

Run: `make lint && make typecheck`
Expected: PASS

- [ ] **Step 7: Final commit (if any fixups needed)**

```bash
git add -A
git commit -m "fix(web): integration test fixups for thread list"
```

Only commit if there were fixes. Skip if everything passed cleanly.
