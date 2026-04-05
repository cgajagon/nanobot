/**
 * Test 4: Thread Navigation
 *
 * Click thread A → history loads → click thread B → different history
 * → click back to A → A's history returns. Plus rapid switching test.
 */

import { test, expect } from "@playwright/test";
import {
  skipWithoutLLM,
  createThreadWithMessage,
  deleteThread,
} from "./helpers";

test.describe("Thread Navigation", () => {
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

  test("switch between threads loads correct history, rapid switch resolves", async ({
    page,
  }) => {
    test.setTimeout(120_000);

    await page.goto("/");
    await expect(
      page.getByRole("button", { name: "New Thread" }),
    ).toBeVisible();

    // Create thread A
    const threadA = await createThreadWithMessage(
      page,
      "Thread A navigation test: what is two plus two?",
    );
    createdThreadIds.push(threadA.threadId);

    // Create thread B
    const threadB = await createThreadWithMessage(
      page,
      "Thread B navigation test: what is the capital of Japan?",
    );
    createdThreadIds.push(threadB.threadId);

    // Thread B should be active (just created)
    await expect(page.getByText("capital of Japan")).toBeVisible();

    // Click thread A in the sidebar
    const threadAButton = page.getByRole("button", { name: threadA.title });
    await threadAButton.click();

    // Wait for thread A's history to load
    await expect(page.getByText("two plus two")).toBeVisible({
      timeout: 15_000,
    });

    // Assert: thread A's message is visible
    await expect(page.getByText("two plus two")).toBeVisible();

    // Assert: thread B's message is NOT visible
    await expect(page.getByText("capital of Japan")).not.toBeVisible();

    // Assert: assistant response is visible (history fully loaded)
    const assistantMessages = page.locator('[data-role="assistant"]');
    await expect(assistantMessages.first()).toBeVisible();

    // Rapid switching test: click B then A within 200ms
    const threadBButton = page.getByRole("button", { name: threadB.title });
    await threadBButton.click();
    await page.waitForTimeout(200);
    await threadAButton.click();

    // Wait for history to settle
    await page.waitForTimeout(3_000);

    // Assert: thread A's messages are visible (last click wins)
    await expect(page.getByText("two plus two")).toBeVisible();

    // Assert: thread B's messages are NOT visible
    await expect(page.getByText("capital of Japan")).not.toBeVisible();
  });
});
