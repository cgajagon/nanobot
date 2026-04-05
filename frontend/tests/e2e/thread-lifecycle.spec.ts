/**
 * Test 1: Full Thread Lifecycle
 *
 * New thread → send message → threadId in request → response renders
 * → title generates → send second message → same session continues.
 */

import { test, expect } from "@playwright/test";
import {
  skipWithoutLLM,
  sendMessage,
  waitForResponse,
  deleteThread,
  captureChatRequests,
} from "./helpers";

test.describe("Thread Lifecycle", () => {
  let createdThreadIds: string[] = [];

  test.beforeEach(() => {
    skipWithoutLLM();
  });

  test.afterEach(async ({ request }) => {
    for (const id of createdThreadIds) {
      await deleteThread(request, id);
    }
    createdThreadIds = [];
  });

  test("new thread: send message, verify threadId, title generates, second message same session", async ({
    page,
  }) => {
    await page.goto("/");

    // Wait for thread list to load
    await expect(
      page.getByRole("button", { name: "New Thread" }),
    ).toBeVisible();

    // Click New Thread
    await page.getByRole("button", { name: "New Thread" }).click();

    // Capture the initialize response to get threadId
    let initThreadId = "";
    await page.route("**/api/threads", async (route) => {
      if (route.request().method() === "POST" && !route.request().url().includes("generate-title")) {
        const response = await route.fetch();
        const body = await response.json();
        initThreadId = body.threadId ?? "";
        await route.fulfill({ response });
      } else {
        await route.continue();
      }
    });

    // Capture /api/chat requests to check threadId
    const chatRequests = await captureChatRequests(page);

    // Send first message
    await sendMessage(page, "What is the capital of France?");

    // Wait for the response to finish streaming
    await waitForResponse(page);

    // Assert: initialize was called and returned a threadId
    expect(initThreadId).toBeTruthy();
    createdThreadIds.push(initThreadId);

    // Assert: /api/chat included the correct threadId
    expect(chatRequests.length).toBeGreaterThanOrEqual(1);
    expect(chatRequests[0].threadId).toBe(initThreadId);

    // Assert: assistant response is visible
    const assistantMessages = page.locator('[data-role="assistant"]');
    await expect(assistantMessages.last()).toBeVisible();

    // Assert: thread title changed from "New Chat"
    const sidebar = page.locator(".aui-thread-list-root");
    await expect(async () => {
      const activeThread = sidebar.locator("[data-active]").first();
      const text = (await activeThread.textContent()) ?? "";
      const cleanText = text.replace("More options", "").trim();
      expect(cleanText).not.toBe("New Chat");
      expect(cleanText.length).toBeGreaterThan(0);
    }).toPass({ timeout: 15_000 });

    // Send second message
    await sendMessage(page, "And what is its population?");
    await waitForResponse(page);

    // Assert: second /api/chat has the same threadId
    expect(chatRequests.length).toBeGreaterThanOrEqual(2);
    expect(chatRequests[1].threadId).toBe(initThreadId);

    // Assert: both user messages are visible
    await expect(page.getByText("What is the capital of France?")).toBeVisible();
    await expect(page.getByText("And what is its population?")).toBeVisible();

    // Cleanup route handlers
    await page.unroute("**/api/threads");
    await page.unroute("**/api/chat");
  });
});
