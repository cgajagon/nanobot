/**
 * Thread history adapter for assistant-ui.
 *
 * Loads historical messages from the server when switching to an existing thread.
 * The remoteId is set via setCurrentThreadRemoteId() from a sync component in
 * App.tsx, which reads it from useRemoteThreadListRuntime's thread list state.
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
