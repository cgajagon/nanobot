/**
 * Shared helpers for Playwright E2E tests.
 *
 * All tests run against the real gateway with a real LLM API key.
 * Tests are gated behind the NANOBOT_E2E_LLM env var.
 */

import { type Page, type APIRequestContext, test } from "@playwright/test";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface CapturedChatRequest {
  threadId?: string;
}

export interface ThreadSetupResult {
  threadId: string;
  title: string;
  messageText: string;
}

// ---------------------------------------------------------------------------
// Gating
// ---------------------------------------------------------------------------

/**
 * Call in test.beforeEach to skip when the env var is not set.
 */
export function skipWithoutLLM(): void {
  test.skip(
    !process.env.NANOBOT_E2E_LLM,
    "Requires running gateway with LLM API key (set NANOBOT_E2E_LLM=1)",
  );
}

// ---------------------------------------------------------------------------
// Message helpers
// ---------------------------------------------------------------------------

/**
 * Type a message into the composer and press Enter to send.
 */
export async function sendMessage(page: Page, text: string): Promise<void> {
  const input = page.getByRole("textbox", { name: "Message input" });
  await input.fill(text);
  await input.press("Enter");
}

/**
 * Wait for an assistant response to appear and finish streaming.
 *
 * Looks for a new element with data-role="assistant", then waits for
 * its text content to stabilize (no new text for 2 seconds).
 */
export async function waitForResponse(page: Page): Promise<void> {
  // Wait for at least one assistant message to exist
  const assistantMessages = page.locator('[data-role="assistant"]');
  const lastMessage = assistantMessages.last();
  await lastMessage.waitFor({ state: "visible", timeout: 30_000 });

  // Wait for content to stabilize (streaming complete)
  const content = lastMessage.locator(".aui-assistant-message-content");
  let previousText = "";
  let stableCount = 0;

  while (stableCount < 4) {
    await page.waitForTimeout(500);
    const currentText = (await content.textContent()) ?? "";
    if (currentText === previousText && currentText.length > 0) {
      stableCount++;
    } else {
      stableCount = 0;
      previousText = currentText;
    }
  }
}

// ---------------------------------------------------------------------------
// Thread helpers
// ---------------------------------------------------------------------------

/**
 * Delete a thread via the API. Does not throw on failure.
 */
export async function deleteThread(
  request: APIRequestContext,
  threadId: string,
): Promise<void> {
  try {
    await request.delete(`/api/threads/${threadId}`);
  } catch {
    // Ignore — thread may not exist if test failed during creation.
  }
}

/**
 * Create a new thread, send a message, wait for the response, and return
 * the threadId and generated title.
 */
export async function createThreadWithMessage(
  page: Page,
  message: string,
): Promise<ThreadSetupResult> {
  // Capture the threadId from the initialize response
  let threadId = "";

  await page.route("**/api/threads", async (route) => {
    if (route.request().method() === "POST") {
      const response = await route.fetch();
      const body = await response.json();
      threadId = body.threadId ?? "";
      await route.fulfill({ response });
    } else {
      await route.continue();
    }
  });

  // Click "New Thread" and send the message
  await page.getByRole("button", { name: "New Thread" }).click();
  await sendMessage(page, message);

  // Wait for the assistant to finish responding
  await waitForResponse(page);

  // Wait for title to generate (up to 15s)
  const sidebar = page.locator(".aui-thread-list-root");
  const activeThread = sidebar.locator("[data-active]").first();

  let title = "New Chat";
  for (let i = 0; i < 30; i++) {
    await page.waitForTimeout(500);
    const text = (await activeThread.textContent())?.trim() ?? "";
    if (text && text !== "New Chat" && text !== "More options") {
      title = text.replace("More options", "").trim();
      break;
    }
  }

  // Clean up route handler
  await page.unroute("**/api/threads");

  return { threadId, title, messageText: message };
}

// ---------------------------------------------------------------------------
// Network capture
// ---------------------------------------------------------------------------

/**
 * Set up passthrough capture for /api/chat requests.
 * Returns the array that will be populated with captured requests.
 * Call page.unroute('** /api/chat') when done.
 */
export async function captureChatRequests(
  page: Page,
): Promise<CapturedChatRequest[]> {
  const requests: CapturedChatRequest[] = [];

  await page.route("**/api/chat", async (route) => {
    const postData = route.request().postData();
    if (postData) {
      try {
        const body = JSON.parse(postData);
        requests.push({ threadId: body.threadId });
      } catch {
        requests.push({});
      }
    }
    await route.continue();
  });

  return requests;
}
