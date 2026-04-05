/**
 * Test 3: Rapid Messages Stay in Same Thread
 *
 * Send 3 messages in quick succession → all /api/chat requests
 * contain the same threadId.
 */

import { test, expect } from "@playwright/test";
import {
  skipWithoutLLM,
  sendMessage,
  deleteThread,
  captureChatRequests,
} from "./helpers";

test.describe("Rapid Messages", () => {
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

  test("3 rapid messages all route to same threadId", async ({ page }) => {
    test.setTimeout(90_000);

    await page.goto("/");
    await expect(
      page.getByRole("button", { name: "New Thread" }),
    ).toBeVisible();

    // Click New Thread
    await page.getByRole("button", { name: "New Thread" }).click();

    // Capture /api/chat requests
    const chatRequests = await captureChatRequests(page);

    // Also capture the threadId from initialize
    let initThreadId = "";
    await page.route("**/api/threads", async (route) => {
      if (route.request().method() === "POST" && !route.request().url().includes("generate-title")) {
        const response = await route.fetch();
        const body = await response.json();
        if (!initThreadId) {
          initThreadId = body.threadId ?? "";
        }
        await route.fulfill({ response });
      } else {
        await route.continue();
      }
    });

    // Send 3 messages rapidly
    await sendMessage(page, "Rapid test message one: hello");
    await page.waitForTimeout(500);
    await sendMessage(page, "Rapid test message two: how are you");
    await page.waitForTimeout(500);
    await sendMessage(page, "Rapid test message three: goodbye");

    // Wait for all 3 assistant responses
    const assistantMessages = page.locator('[data-role="assistant"]');
    await expect(assistantMessages).toHaveCount(3, { timeout: 60_000 });

    // Track the threadId for cleanup
    if (initThreadId) {
      createdThreadIds.push(initThreadId);
    }

    // Assert: all 3 /api/chat requests have the same threadId
    expect(chatRequests.length).toBe(3);
    expect(chatRequests[0].threadId).toBeTruthy();
    expect(chatRequests[1].threadId).toBe(chatRequests[0].threadId);
    expect(chatRequests[2].threadId).toBe(chatRequests[0].threadId);

    // Assert: threadId matches the initialize response
    if (initThreadId) {
      expect(chatRequests[0].threadId).toBe(initThreadId);
    }

    // Assert: all 3 user messages are visible
    await expect(page.getByText("Rapid test message one")).toBeVisible();
    await expect(page.getByText("Rapid test message two")).toBeVisible();
    await expect(page.getByText("Rapid test message three")).toBeVisible();

    // Cleanup
    await page.unroute("**/api/chat");
    await page.unroute("**/api/threads");
  });
});
