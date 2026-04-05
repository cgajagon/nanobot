/**
 * Test 2: Persistence Across Page Reload
 *
 * After a conversation, reload the page → thread appears with title
 * → click it → history loads correctly.
 */

import { test, expect } from "@playwright/test";
import {
  skipWithoutLLM,
  createThreadWithMessage,
  deleteThread,
} from "./helpers";

test.describe("Thread Persistence", () => {
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

  test("thread survives page reload with title and history", async ({
    page,
  }) => {
    await page.goto("/");
    await expect(
      page.getByRole("button", { name: "New Thread" }),
    ).toBeVisible();

    // Create a thread with a distinctive message
    const { threadId, title } = await createThreadWithMessage(
      page,
      "E2E persistence test: explain quantum entanglement briefly",
    );
    createdThreadIds.push(threadId);

    // Reload the page
    await page.reload();

    // Wait for thread list to load
    await expect(
      page.getByRole("button", { name: "New Thread" }),
    ).toBeVisible();

    // Assert: the thread appears in the sidebar with its title
    const threadButton = page.getByRole("button", { name: title });
    await expect(threadButton).toBeVisible({ timeout: 10_000 });

    // Click the thread to load its history
    await threadButton.click();

    // Wait for messages to appear
    await expect(page.getByText("quantum entanglement")).toBeVisible({
      timeout: 15_000,
    });

    // Assert: an assistant response is also visible (history fully loaded)
    const assistantMessages = page.locator('[data-role="assistant"]');
    await expect(assistantMessages.first()).toBeVisible();
  });
});
