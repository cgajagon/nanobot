# Playwright E2E Test Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 4 Playwright E2E tests for the web chat UI covering thread lifecycle, persistence, rapid message routing, and navigation.

**Architecture:** Install Playwright in the frontend package, create a config pointing at the Vite dev server (localhost:5173), implement shared helpers for message sending/response waiting/cleanup, then implement 4 test files that run against the real gateway with LLM. All tests gated behind `NANOBOT_E2E_LLM` env var.

**Tech Stack:** Playwright Test, TypeScript, Chromium, real gateway + LLM

**Spec:** `docs/superpowers/specs/2026-04-05-e2e-playwright-tests-design.md`

---

## File Map

### New Files

| File | Responsibility |
|------|---------------|
| `frontend/playwright.config.ts` | Playwright configuration: timeouts, browser, reporters, base URL |
| `frontend/tests/e2e/helpers.ts` | Shared utilities: `sendMessage`, `waitForResponse`, `createThreadWithMessage`, `deleteThread`, types |
| `frontend/tests/e2e/thread-lifecycle.spec.ts` | Test 1: full lifecycle (init → chat → title → second message) |
| `frontend/tests/e2e/thread-persistence.spec.ts` | Test 2: persistence across page reload |
| `frontend/tests/e2e/rapid-messages.spec.ts` | Test 3: rapid messages stay in same thread |
| `frontend/tests/e2e/thread-navigation.spec.ts` | Test 4: thread switching + rapid click race condition |

### Modified Files

| File | Changes |
|------|---------|
| `frontend/package.json` | Add `@playwright/test` to devDependencies |
| `frontend/.gitignore` | Add Playwright artifacts |
| `Makefile` | Add `test-e2e` and `test-e2e-headed` targets |

---

## Task 1: Install Playwright and Configure

**Files:**
- Modify: `frontend/package.json`
- Create: `frontend/playwright.config.ts`
- Modify: `frontend/.gitignore`
- Modify: `Makefile`

- [ ] **Step 1: Add `@playwright/test` to devDependencies**

In `frontend/package.json`, add to the `devDependencies` object:

```json
"@playwright/test": "^1.52.0"
```

Then install:

```bash
cd frontend && npm install
```

- [ ] **Step 2: Install Chromium browser binary**

```bash
cd frontend && npx playwright install chromium
```

Expected: Downloads Chromium binary. Output includes "Downloading chromium..." and a success message.

- [ ] **Step 3: Create `frontend/playwright.config.ts`**

```typescript
import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests/e2e",
  timeout: 60_000,
  expect: {
    timeout: 30_000,
  },
  use: {
    baseURL: "http://127.0.0.1:5173",
    trace: "retain-on-failure",
  },
  retries: 0,
  reporter: [["list"], ["html", { open: "never" }]],
  projects: [
    {
      name: "chromium",
      use: { browserName: "chromium" },
    },
  ],
});
```

- [ ] **Step 4: Add Playwright artifacts to `frontend/.gitignore`**

Append these lines at the end of `frontend/.gitignore`:

```
# Playwright
test-results/
playwright-report/
.playwright-mcp/
```

- [ ] **Step 5: Add Makefile targets**

In `Makefile`, add to the `.PHONY` declaration on line 1 (append to the existing list):

```
test-e2e test-e2e-headed
```

Then add these targets after the `test-integration` target (after line 24):

```makefile
test-e2e:  ## Run Playwright E2E tests (requires gateway + Vite dev server + LLM API key)
	cd frontend && NANOBOT_E2E_LLM=1 npx playwright test

test-e2e-headed:  ## Run E2E tests with visible browser
	cd frontend && NANOBOT_E2E_LLM=1 npx playwright test --headed
```

- [ ] **Step 6: Verify Playwright runs (no tests yet)**

```bash
cd frontend && npx playwright test
```

Expected: Output says "no tests found" or similar (tests directory doesn't exist yet). No errors about config.

- [ ] **Step 7: Commit**

```bash
git add frontend/package.json frontend/package-lock.json frontend/playwright.config.ts frontend/.gitignore Makefile
git commit -m "chore(web): install Playwright and configure E2E test infrastructure"
```

---

## Task 2: Create Shared Test Helpers

**Files:**
- Create: `frontend/tests/e2e/helpers.ts`

- [ ] **Step 1: Create the helpers file**

Create `frontend/tests/e2e/helpers.ts`:

```typescript
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
 *
 * Sets up network capture, clicks "New Thread", sends the message, waits
 * for streaming to complete, captures the threadId from the /api/threads
 * POST response and the title from the sidebar.
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
 * Call page.unroute('**/api/chat') when done.
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
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd frontend && npx tsc --noEmit
```

Expected: PASS (or only pre-existing errors unrelated to the new file). The helpers file only imports from `@playwright/test`.

- [ ] **Step 3: Commit**

```bash
git add frontend/tests/e2e/helpers.ts
git commit -m "test(web): add shared Playwright E2E test helpers"
```

---

## Task 3: Test 1 — Full Thread Lifecycle

**Files:**
- Create: `frontend/tests/e2e/thread-lifecycle.spec.ts`

- [ ] **Step 1: Create the test file**

Create `frontend/tests/e2e/thread-lifecycle.spec.ts`:

```typescript
/**
 * Test 1: Full Thread Lifecycle
 *
 * New thread → send message → threadId in request → response renders
 * → title generates → send second message → same session continues.
 *
 * Exercises: adapter.initialize(), body() threadId injection, SSE streaming,
 * generateTitle() stream format, session continuity.
 */

import { test, expect } from "@playwright/test";
import {
  skipWithoutLLM,
  sendMessage,
  waitForResponse,
  deleteThread,
  captureChatRequests,
  type CapturedChatRequest,
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
    // Wait up to 15s for title generation
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
```

- [ ] **Step 2: Run the test (requires gateway + Vite running)**

```bash
cd frontend && NANOBOT_E2E_LLM=1 npx playwright test thread-lifecycle --headed
```

Expected: Test passes. Browser opens, creates thread, sends messages, verifies assertions.

If `NANOBOT_E2E_LLM` is not set:

```bash
cd frontend && npx playwright test thread-lifecycle
```

Expected: Test is skipped with message "Requires running gateway with LLM API key".

- [ ] **Step 3: Commit**

```bash
git add frontend/tests/e2e/thread-lifecycle.spec.ts
git commit -m "test(web): add E2E test for full thread lifecycle"
```

---

## Task 4: Test 2 — Persistence Across Page Reload

**Files:**
- Create: `frontend/tests/e2e/thread-persistence.spec.ts`

- [ ] **Step 1: Create the test file**

Create `frontend/tests/e2e/thread-persistence.spec.ts`:

```typescript
/**
 * Test 2: Persistence Across Page Reload
 *
 * After a conversation, reload the page → thread appears with title
 * → click it → history loads correctly.
 *
 * Exercises: JSONL persistence, session.metadata["title"], adapter.list(),
 * history adapter load(), GET /api/threads/{id}/messages.
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
    const { threadId, title, messageText } = await createThreadWithMessage(
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
    // The title may be the LLM-generated title or the fallback (first user message)
    // Either way, a button containing relevant text should exist
    const threadButton = page.getByRole("button", { name: title });
    await expect(threadButton).toBeVisible({ timeout: 10_000 });

    // Click the thread to load its history
    await threadButton.click();

    // Wait for messages to appear
    await expect(page.getByText("quantum entanglement")).toBeVisible({
      timeout: 15_000,
    });

    // Assert: the user message is visible
    await expect(
      page.getByText("quantum entanglement"),
    ).toBeVisible();

    // Assert: an assistant response is also visible (history fully loaded)
    const assistantMessages = page.locator('[data-role="assistant"]');
    await expect(assistantMessages.first()).toBeVisible();
  });
});
```

- [ ] **Step 2: Run the test**

```bash
cd frontend && NANOBOT_E2E_LLM=1 npx playwright test thread-persistence --headed
```

Expected: PASS — creates thread, reloads, finds thread, loads history.

- [ ] **Step 3: Commit**

```bash
git add frontend/tests/e2e/thread-persistence.spec.ts
git commit -m "test(web): add E2E test for thread persistence across reload"
```

---

## Task 5: Test 3 — Rapid Messages Stay in Same Thread

**Files:**
- Create: `frontend/tests/e2e/rapid-messages.spec.ts`

- [ ] **Step 1: Create the test file**

Create `frontend/tests/e2e/rapid-messages.spec.ts`:

```typescript
/**
 * Test 3: Rapid Messages Stay in Same Thread
 *
 * Send 3 messages in quick succession → all /api/chat requests
 * contain the same threadId.
 *
 * Exercises: _currentThreadRemoteId consistency under timing pressure,
 * body() callback reliability, session splitting prevention.
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
    test.setTimeout(90_000); // 3 LLM calls

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
    await page.waitForTimeout(500); // Wait for initialize() to resolve
    await sendMessage(page, "Rapid test message two: how are you");
    await page.waitForTimeout(500);
    await sendMessage(page, "Rapid test message three: goodbye");

    // Wait for all responses to complete
    // We need at least 3 assistant messages visible
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
```

- [ ] **Step 2: Run the test**

```bash
cd frontend && NANOBOT_E2E_LLM=1 npx playwright test rapid-messages --headed
```

Expected: PASS — sends 3 messages quickly, all have same threadId.

- [ ] **Step 3: Commit**

```bash
git add frontend/tests/e2e/rapid-messages.spec.ts
git commit -m "test(web): add E2E test for rapid message threadId consistency"
```

---

## Task 6: Test 4 — Thread Navigation

**Files:**
- Create: `frontend/tests/e2e/thread-navigation.spec.ts`

- [ ] **Step 1: Create the test file**

Create `frontend/tests/e2e/thread-navigation.spec.ts`:

```typescript
/**
 * Test 4: Thread Navigation
 *
 * Click thread A → history loads → click thread B → different history
 * → click back to A → A's history returns. Plus rapid switching test.
 *
 * Exercises: history adapter load(), _currentThreadRemoteId updates,
 * state cleanup on switch, rapid switching race condition.
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
    test.setTimeout(120_000); // 4 LLM calls across 2 threads

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
```

- [ ] **Step 2: Run the test**

```bash
cd frontend && NANOBOT_E2E_LLM=1 npx playwright test thread-navigation --headed
```

Expected: PASS — creates 2 threads, navigates between them, rapid switch resolves correctly.

- [ ] **Step 3: Commit**

```bash
git add frontend/tests/e2e/thread-navigation.spec.ts
git commit -m "test(web): add E2E test for thread navigation and rapid switching"
```

---

## Task 7: Run Full Suite and Final Verification

- [ ] **Step 1: Run all 4 tests**

```bash
cd frontend && NANOBOT_E2E_LLM=1 npx playwright test
```

Expected: 4 tests pass. Output shows:
```
  ✓ thread-lifecycle.spec.ts (thread lifecycle)
  ✓ thread-persistence.spec.ts (thread persistence)
  ✓ rapid-messages.spec.ts (rapid messages)
  ✓ thread-navigation.spec.ts (thread navigation)

  4 passed
```

- [ ] **Step 2: Verify tests skip without env var**

```bash
cd frontend && npx playwright test
```

Expected: 4 tests skipped with message "Requires running gateway with LLM API key".

- [ ] **Step 3: Verify Makefile targets work**

```bash
make test-e2e
```

Expected: Same as step 1 (4 tests pass).

- [ ] **Step 4: Run lint and typecheck on the Python side (ensure no regressions)**

```bash
make lint && make typecheck
```

Expected: PASS

- [ ] **Step 5: Final commit if any fixes needed**

```bash
git add -A
git commit -m "test(web): final E2E test suite fixups"
```

Only commit if there were fixes. Skip if everything passed cleanly.
