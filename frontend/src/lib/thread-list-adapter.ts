/**
 * RemoteThreadListAdapter implementation for nanobot.
 *
 * Maps assistant-ui's thread list adapter interface to the nanobot server API.
 * Each method calls a server endpoint — the server is the single source of
 * truth for thread metadata (titles, timestamps, status).
 */

import type {
  unstable_RemoteThreadListAdapter as RemoteThreadListAdapter,
} from "@assistant-ui/react";
import type { ThreadMessage } from "@assistant-ui/core";
import { createAssistantStream } from "assistant-stream";
import { setCurrentThreadRemoteId } from "./thread-history";

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
    // Clear immediately so that if load() fires before the POST completes,
    // it returns empty messages instead of the previous thread's messages.
    setCurrentThreadRemoteId(undefined);
    const response = await fetch("/api/threads", { method: "POST" });
    const data = await response.json();
    const remoteId = data.threadId as string;
    // Eagerly set the remoteId so the body() callback in useDataStreamRuntime
    // can inject it into the /api/chat request immediately after init resolves.
    setCurrentThreadRemoteId(remoteId);
    return { remoteId, externalId: undefined };
  },

  async generateTitle(remoteId: string, _messages: readonly ThreadMessage[]) {
    const response = await fetch(`/api/threads/${remoteId}/generate-title`, {
      method: "POST",
    });
    if (!response.ok) {
      return createAssistantStream(() => {});
    }
    const data = await response.json();
    const title = data.title || "New Chat";

    return createAssistantStream((controller) => {
      controller.appendText(title);
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
