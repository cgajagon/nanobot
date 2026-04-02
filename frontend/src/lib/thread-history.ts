/**
 * Thread history adapter for assistant-ui.
 *
 * Loads historical messages from the server when switching to an existing thread.
 * The adapter's load() method receives no parameters (library limitation), so we
 * track the "current thread being loaded" via a mutable variable updated by a
 * React component placed inside AssistantRuntimeProvider.
 *
 * Flow:
 * 1. ThreadHistorySync component (React) reads the current thread's remoteId
 *    via useThreadListItem() and writes it to _currentThreadRemoteId.
 * 2. When assistant-ui creates a new thread runtime and calls load(), the adapter
 *    reads _currentThreadRemoteId, resolves it to a server UUID via
 *    getServerThreadId(), and fetches messages from /api/threads/{id}/messages.
 * 3. Messages are converted to ThreadMessageLike[] and returned as an
 *    ExportedMessageRepository.
 */

import { useThreadListItem } from "@assistant-ui/react";
import { ExportedMessageRepository } from "@assistant-ui/react";
import type { ThreadHistoryAdapter } from "@assistant-ui/react";
import type { ExportedMessageRepositoryItem } from "@assistant-ui/react";
import { getServerThreadId } from "./thread-sync";

/**
 * Mutable variable tracking the remoteId of the thread currently being loaded.
 * Updated by ThreadHistorySync (React component) before load() is called.
 */
let _currentThreadRemoteId: string | undefined;

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
      // Thread not found or no messages — return empty repository.
      return { messages: [] };
    }
    const data: ThreadMessagesResponse = await response.json();
    if (!data.messages || data.messages.length === 0) {
      return { messages: [] };
    }

    // Convert server messages to ThreadMessageLike format for fromArray().
    const messageLikes = data.messages.map((msg) => ({
      role: msg.role as "user" | "assistant",
      content: msg.content.map((c) => c.text).join("\n"),
      id: msg.id,
    }));

    return ExportedMessageRepository.fromArray(messageLikes);
  } catch {
    // Network error or parse failure — return empty.
    return { messages: [] };
  }
}

/**
 * ThreadHistoryAdapter implementation that loads messages from the server.
 *
 * Passed to useDataStreamRuntime via adapters.history.
 */
export const serverHistoryAdapter: ThreadHistoryAdapter = {
  async load() {
    const remoteId = _currentThreadRemoteId;
    if (!remoteId) {
      // New thread with no remoteId yet — nothing to load.
      return { messages: [] };
    }

    // Resolve local thread IDs to server UUIDs.
    const serverThreadId = getServerThreadId(remoteId);

    // Don't try to load history for brand-new local threads.
    if (serverThreadId.startsWith("__LOCALID_")) {
      return { messages: [] };
    }

    return fetchThreadMessages(serverThreadId);
  },

  async append(_item: ExportedMessageRepositoryItem) {
    // Messages are already persisted server-side by the chat endpoint.
    // No additional persistence needed from the frontend.
  },
};

/**
 * React component that syncs the current thread's remoteId to the mutable
 * variable read by serverHistoryAdapter.load().
 *
 * Must be placed inside <AssistantRuntimeProvider> so useThreadListItem()
 * has access to the thread context.
 *
 * The write happens during render (not in useEffect) to avoid a race condition:
 * the library's __internal_load() calls load() in a useEffect, and React fires
 * child effects before parent effects. Writing synchronously during render
 * ensures _currentThreadRemoteId is set before load() is called.
 *
 * Renders nothing — purely a side-effect component.
 */
export function ThreadHistorySync(): null {
  const threadListItem = useThreadListItem({ optional: true });

  // Synchronous write during render — not in useEffect — to beat the race
  // with __internal_load() which calls load() in a useEffect.
  _currentThreadRemoteId = threadListItem?.remoteId;

  return null;
}
