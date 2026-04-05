# Playwright E2E Test Suite for Web Chat UI

> Design spec for end-to-end Playwright tests covering thread lifecycle,
> persistence, rapid message routing, and thread navigation.
>
> Date: 2026-04-05

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Risk Assessment](#risk-assessment)
3. [Candidate Evaluation](#candidate-evaluation)
4. [Chosen Approach](#chosen-approach)
5. [Infrastructure](#infrastructure)
6. [Test Specifications](#test-specifications)
7. [Helpers and Conventions](#helpers-and-conventions)
8. [File Structure](#file-structure)
9. [Broader Test Portfolio by Layer](#broader-test-portfolio)

---

<a id="problem-statement"></a>
## 1. Problem Statement

The web chat UI has no automated test coverage. Recent feature work (server-side thread
list with LLM titles, PR #145 and #146) introduced bugs that were only caught through
manual browser debugging with Playwright MCP:

- `threadId` not included in `/api/chat` requests — the `useDataStreamRuntime` body
  callback read `_currentThreadRemoteId` before the adapter's `initialize()` had
  resolved, causing every new conversation to create two separate server sessions
  (one from initialize, one from chat with a random UUID).
- Wrong `AssistantStream` chunk format — `generateTitle()` returned a raw
  `ReadableStream` with `{type: "text-delta", textDelta: title}` chunks, but the
  library expects chunks produced by `createAssistantStream` from the `assistant-stream`
  package (with `path` field and proper chunk types). Title generation silently failed.
- Wrong import names — `RemoteThreadListAdapter` and `useRemoteThreadListRuntime` are
  exported with `unstable_` prefix in `@assistant-ui/react` v0.12.x. The frontend
  rendered a blank page with no console errors.

All three bugs were **integration seam failures** — the individual components worked
but the wiring between them was wrong. Unit tests would not have caught them. E2E tests
exercising the real browser → Vite proxy → gateway → LLM pipeline would.

---

<a id="risk-assessment"></a>
## 2. Risk Assessment

### Fragile Coupling Points in the Frontend

#### The Mutable Variable Pattern (HIGH RISK)

The `_currentThreadRemoteId` mutable variable in `thread-history.ts` is shared state
between React components and non-React code:

```
ThreadRemoteIdSync (React component, renders every cycle)
    → setCurrentThreadRemoteId(threadListItem?.remoteId)
        → _currentThreadRemoteId (module-level let)
            ← getCurrentThreadRemoteId() (read by body() callback in useDataStreamRuntime)
            ← _currentThreadRemoteId (read by serverHistoryAdapter.load())
```

This pattern has a **race condition window**: if the user sends a message before
`initialize()` resolves and `ThreadRemoteIdSync` renders with the new remoteId, the
`body()` callback returns `undefined` and the server generates a random UUID.

The fix in PR #146 closes this window by eagerly calling `setCurrentThreadRemoteId()`
inside the adapter's `initialize()` callback. But the pattern remains fragile — any
change to render timing, adapter lifecycle, or the assistant-ui library's internal
hook scheduling could reopen the race.

**What tests catch this:** Test 1 (threadId in request), Test 3 (rapid messages).

#### Thread List ↔ Server Session Consistency (HIGH RISK)

The `RemoteThreadListAdapter` maps between assistant-ui's internal thread state and the
server's JSONL sessions. The mapping flows through:

1. `adapter.list()` → `GET /api/threads` → maps `ThreadInfo` to `RemoteThreadMetadata`
2. `adapter.initialize()` → `POST /api/threads` → creates session, returns UUID
3. `adapter.generateTitle()` → `POST /api/threads/{id}/generate-title` → LLM call,
   wraps result in `AssistantStream` via `createAssistantStream`
4. `adapter.fetch()` → re-fetches `GET /api/threads` and finds one thread by ID

Each step has its own contract (request format, response shape, stream format). A
mismatch at any point causes silent failures — titles don't update, threads don't
appear, or sessions split.

**What tests catch this:** Test 1 (lifecycle), Test 2 (persistence), Test 4 (navigation).

#### SSE Streaming Integrity (MEDIUM RISK)

The response stream from `/api/chat` uses the Vercel AI SDK Data Stream Protocol
(`ui-message-stream`). The frontend's `readStatusEvents()` clones the response body
and parses SSE lines looking for `type: "status"`, `type: "finish"`, and `type: "error"`
events. The parser is lenient (ignores parse errors on non-JSON lines) but doesn't
differentiate between agent errors and transport errors.

The backend's `stream_agent_response()` has two event dispatch paths (canonical and
legacy) and a tool-call safety closure that handles unclosed tool calls at stream end.
The stream timeout is 5 minutes (20 intervals of 15s) after which it force-closes with
a hardcoded error text.

**What tests catch this:** Test 1 (response renders), Test 3 (multiple responses stream
correctly).

#### State Cleanup on Thread Switch (MEDIUM RISK)

When the user clicks a different thread in the sidebar:

1. `useRemoteThreadListRuntime` updates the active thread ID
2. `ThreadRemoteIdSync` reads the new `remoteId` and writes to `_currentThreadRemoteId`
3. `serverHistoryAdapter.load()` reads `_currentThreadRemoteId` and fetches messages
4. The chat area re-renders with the new thread's history

If step 2 hasn't executed before step 3 (render timing), the history adapter loads
messages for the wrong thread. Rapid switching (click A then immediately B) can cause
history from A to appear briefly before B's loads.

**What tests catch this:** Test 4 (navigation + rapid switching).

### What Has Broken Before (Git History)

| Commit | Issue | Root Cause |
|--------|-------|-----------|
| 904596fe | threadId not in /api/chat | remoteId not propagated before request fired |
| d48f4222 | Title generation silently failed | Wrong stream format (missing `path` field) |
| 81ee48ba | Exports didn't exist | assistant-ui v0.12 requires `unstable_` prefix |
| 6e85640a | File attachment + streaming broken | Data URI handling and text markers interaction |
| 69036f60 | Streaming duplicated messages | Ephemeral system messages persisted to history |
| d054933c | Messages streamed twice | Streaming boundary deduplication gaps |

**Pattern:** Most bugs are state synchronization, API contract mismatches, and
timing-dependent streaming issues.

---

<a id="candidate-evaluation"></a>
## 3. Candidate Evaluation

### Initial Candidate List (12 scenarios)

The initial proposal had 12 test scenarios. Each was evaluated for value, risk coverage,
correct test layer, and redundancy.

### Thread Lifecycle Group

**#1: New thread → send message → threadId in request → response renders**
- Risk covered: The exact bug from PR #146 (threadId missing from /api/chat)
- Value: **HIGH** — Most critical integration point. Exercises initialize() → body()
  injection → streaming → render.
- Would catch: Missing threadId, session mismatch, broken initialize(), streaming
  parse failures.
- Layer: **Correct (E2E)** — requires real server, real SSE, real session creation.

**#2: Thread title auto-generates after first exchange**
- Risk covered: The generateTitle adapter → server LLM call → stream format → UI update chain.
- Value: **MEDIUM** — Title generation is UX. The AssistantStream format bug was a
  one-time implementation error, not a recurring regression pattern.
- Would catch: Broken stream format, server endpoint failure, title not persisting.
- Layer: Correct (E2E) but **merged with #1**. Title generation is a natural
  continuation of "send first message." Testing it separately doubles the cost
  of thread creation + message send + wait for response.

**#3: Page reload → previous threads appear with correct titles**
- Risk covered: Session persistence, `GET /api/threads`, adapter `list()` mapping.
- Value: **HIGH** — Core feature. If it regresses, the entire thread list is broken.
- Would catch: Session file corruption, metadata not persisting, list() mapping errors,
  title fallback logic.
- Layer: **Correct (E2E)** — requires real page navigation cycle.

**#4: Delete thread → removed from sidebar and disk**
- Risk covered: DELETE endpoint, session file cleanup, UI removal.
- Value: **LOW** — Simple CRUD. The endpoint is 6 lines. The UI delegates to
  assistant-ui primitives. Very unlikely to regress.
- Would catch: File not deleted, session cache stale.
- Layer: **Over-tested as E2E.** A backend integration test (call DELETE, verify file
  gone) covers the real risk. The UI part is a button click → optimistic removal
  handled by the library.
- **Decision: Eliminated.**

### Thread Navigation Group

**#5: Click existing thread → history loads correctly**
- Risk covered: History adapter `load()`, `_currentThreadRemoteId` sync,
  `GET /api/threads/{id}/messages`, message format conversion.
- Value: **HIGH** — Fragile mutable variable pattern. Real regression risk.
- Would catch: Wrong remoteId, stale `_currentThreadRemoteId`, message format
  mismatch, empty history bug.
- Layer: **Correct (E2E).**

**#6: Switch between threads → correct messages shown for each**
- Risk covered: State cleanup between threads, `_currentThreadRemoteId` update, no
  message bleed.
- Value: **MEDIUM** — Overlaps significantly with #5. Incremental value is testing
  that switching *away* clears state correctly.
- Would catch: Stale messages from previous thread, `_currentThreadRemoteId` not updated.
- Layer: Correct (E2E) but **merged with #5** as a single "navigate threads" test:
  click thread A → verify messages → click thread B → verify different messages →
  click back to A.

**#7: Send message in existing thread → same session continues**
- Risk covered: threadId persistence across messages, session append (not create).
- Value: **MEDIUM-HIGH** — Tests that body() callback provides the correct threadId
  for follow-up messages in an existing thread.
- Would catch: threadId lost after first message, session splitting on second message.
- Layer: Correct (E2E) but **merged with #1** (send two messages in the new thread,
  verify both route to same session).

### Edge Cases Group

**#8: Gateway restart → threads persist**
- Risk covered: Identical to #3 (page reload + threads appear).
- Value: **REDUNDANT** — A gateway restart means the server process stops and starts.
  Session files are on disk. The frontend doesn't know the difference between "gateway
  restarted" and "page reloaded." Both result in `adapter.list()` being called fresh.
- Would catch: Nothing beyond #3.
- **Decision: Eliminated.**

**#9: Send message while offline / server down → error shown, no crash**
- Risk covered: Network failure resilience, error boundaries.
- Value: **LOW for E2E, MEDIUM as a concern** — The real risk is real (no error
  boundary around `readStatusEvents`, status leaks), but testing this in Playwright
  requires intercepting network requests to simulate failure, which makes the test
  fragile and environment-dependent.
- Would catch: Unhandled promise rejection, UI crash on network error.
- Layer: **Wrong layer.** Better as a unit test of error handling in adapter methods
  (mock fetch to throw) plus a manual smoke test.
- **Decision: Eliminated from E2E. Noted for unit test layer.**

**#10: Multiple rapid messages → all go to same thread**
- Risk covered: The `_currentThreadRemoteId` timing race, `body()` callback consistency.
- Value: **HIGH** — Highest-risk scenario not covered by other tests. Rapid interaction
  is exactly how the mutable variable pattern breaks. If a user types fast and hits
  Enter twice while the first response is streaming, the threadId must remain consistent.
- Would catch: Race condition in threadId sync, session splitting under load.
- Layer: **Correct (E2E)** — requires real async timing.

### Attachment Handling Group

**#11: Send message with file attachment → attachment processed**
- Risk covered: Text file upload (FileReader → attachment tags → server extraction →
  disk save).
- Value: **MEDIUM** — Meaningful complexity (regex extraction, dedup manifest, path
  traversal protection). But attachment handling has been stable.
- Would catch: Attachment not saved, content not stripped, file not accessible to agent.
- Layer: Correct (E2E) but **lower priority** than thread management tests.
- **Decision: Deferred to future expansion.**

**#12: Send message with image → image rendered in chat**
- Risk covered: Image upload → base64 → multipart → server extraction → rendering.
- Value: **LOW** — Overlaps with #11 significantly. Image path uses
  `SimpleImageAttachmentAdapter` (library code). Server extraction is the same
  `_extract_images` function.
- Would catch: Image-specific MIME handling.
- Layer: **Over-tested as separate E2E.** Merge into #11 if attachment testing is added.
- **Decision: Eliminated.**

### Coverage Gaps Identified

**Missing — HIGH priority (added to final suite):**
- **Rapid thread switching** — Click thread A, immediately click thread B before
  history loads. Tests that `_currentThreadRemoteId` updates correctly. Added to
  Test 4 (navigation) as a rapid-switch substep.

**Missing — MEDIUM priority (noted, not added):**
- **SSE stream interruption** — User clicks "stop" during generation. Tests clean
  stream closure and partial response preservation.
- **Second message after full streaming cycle** — Verifies threadId persists after
  init → chat → generate-title → second chat. Covered by Test 1 (merged #7).

### Final Test Selection

After evaluation: 12 candidates → 4 tests (3 merges, 5 eliminations, 1 gap filled).

| Test | Merged From | Priority |
|------|-------------|----------|
| Full thread lifecycle | #1 + #2 + #7 | P0 |
| Persistence across reload | #3 | P0 |
| Rapid messages same thread | #10 | P0 |
| Thread navigation | #5 + #6 + new rapid switching | P1 |

---

<a id="chosen-approach"></a>
## 4. Chosen Approach

### Approach Considered: Pure Playwright with Route Interception

Mock the `/api/chat` SSE response using Playwright's `page.route()` to return a
deterministic streamed response. Real server calls for thread management endpoints.
Title generation endpoint mocked to return a fixed title.

**Pros:** Deterministic chat responses, fast (~1s per test), thread management tested
against real server, clean separation between frontend wiring and agent testing.
**Cons:** Must maintain the mock SSE format if the streaming protocol changes.

**Rejected.** The user preferred real LLM testing for maximum realism.

### Approach Considered: Fixture-Based with Recorded Responses

Record real SSE responses once, save as fixtures, replay via route interception. Update
fixtures periodically.

**Pros:** Realistic response format, deterministic, no LLM cost.
**Cons:** Fixtures go stale when streaming format changes, extra maintenance burden.

**Rejected.** Maintenance overhead outweighs the determinism benefit for 4 tests.

### Chosen: Full Real Stack, All Tests Gated

All 4 tests use the real LLM. The entire suite is gated behind `NANOBOT_E2E_LLM`
env var. No mocking, no fixtures. Tests run against the full stack: browser → Vite dev
server → proxy → FastAPI gateway → LLM provider.

**Pros:** Most realistic, catches real integration issues across the entire stack,
no mock maintenance.
**Cons:** Slow (~10-20s per LLM call, total suite ~5 minutes), costs money per run,
can't run in CI without API key. Response content varies across runs (assertions
check structure, not exact text).

**Rationale:** The bugs we found were all real integration failures that only manifested
with the full stack running. Mocked tests would have passed while the real UI was
broken. For a 4-test suite run manually before pushing frontend changes, the cost
and speed trade-offs are acceptable.

### Test Gating Mechanism

```typescript
import { test } from '@playwright/test';

test.beforeEach(async () => {
  test.skip(!process.env.NANOBOT_E2E_LLM, 'Requires running gateway with LLM API key');
});
```

All tests skip gracefully when the env var is not set. Running `npx playwright test`
without the env var produces 4 skipped tests, not 4 failures.

---

<a id="infrastructure"></a>
## 5. Infrastructure

### Playwright Configuration

**Dependency:** `@playwright/test` as dev dependency in `frontend/package.json`.
After installation, `npx playwright install chromium` downloads the browser binary.

**Config file:** `frontend/playwright.config.ts` with:

```typescript
import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 60_000,           // Per-test timeout (agent responses take 10-20s)
  expect: {
    timeout: 30_000,         // Per-assertion timeout (waiting for streamed content)
  },
  use: {
    baseURL: 'http://127.0.0.1:5173',
    trace: 'on-first-retry', // Capture trace for debugging failures
  },
  retries: 0,                // No retries — tests should be deterministic
  reporter: [
    ['list'],                // Terminal output
    ['html', { open: 'never' }],  // HTML report for debugging
  ],
  projects: [
    {
      name: 'chromium',
      use: { browserName: 'chromium' },
    },
  ],
});
```

**Key configuration decisions:**

- **Single browser (Chromium only):** This is an internal tool, not a public website.
  Cross-browser testing adds no value — the bugs are in the JavaScript/React integration
  layer, not browser rendering differences.

- **60s test timeout:** A single LLM round-trip takes 10-20s. Tests with 2 round-trips
  need 40s+ plus setup/teardown. 60s gives headroom without hiding hung tests. Test 4
  (navigation) overrides to 120s.

- **30s expect timeout:** Individual assertions that wait for streamed content
  (e.g., "wait for assistant response to appear") may take up to 20s. 30s gives margin.

- **No retries:** With real LLM, flaky tests indicate real timing issues. Retries mask
  the problem. If a test fails, investigate rather than retry.

- **Trace on first retry:** Since retries are 0, this effectively means no automatic
  trace. Developers can run with `--trace on` when debugging.

### Prerequisites

Tests assume the developer has three things running:

1. **The gateway:** `nanobot gateway` — serves the FastAPI web app on port 8000
2. **The Vite dev server:** `cd frontend && npm run dev` — serves the React app on
   port 5173 with `/api/*` proxied to port 8000
3. **A valid LLM API key:** Configured in `~/.nanobot/config.json`

No automated server lifecycle management. The `test-e2e` Makefile target documents
these prerequisites. This matches the project's existing `make test-integration`
pattern where the developer starts the necessary services manually.

**Rationale for no auto-start:** The test suite is 4 tests. Adding a `webServer` config
or `globalSetup` script to start/stop the gateway and Vite would add significant
complexity (process management, port conflicts, startup sequencing, health checks)
for minimal benefit. When the suite grows beyond ~10 tests or CI is needed, this
decision should be revisited.

### Session Cleanup Strategy

Each test tracks the thread IDs it creates in a local array. `afterEach` iterates
the array and deletes each thread via `DELETE /api/threads/{id}` using Playwright's
`page.request` API (direct HTTP call, no browser navigation).

```typescript
let createdThreadIds: string[] = [];

test.afterEach(async ({ request }) => {
  for (const id of createdThreadIds) {
    await request.delete(`/api/threads/${id}`);
  }
  createdThreadIds = [];
});
```

This prevents cross-test contamination without requiring workspace isolation or
database resets. Each test starts with a clean state (its own threads) and ends
with cleanup (threads deleted).

**Edge case:** If a test fails mid-execution, `afterEach` still runs and cleans up
any threads created before the failure. Threads created by the server (e.g., from
`/api/chat` without a threadId due to a bug) are NOT cleaned up — they persist as
orphans. This is acceptable for a manual test suite.

### Makefile Targets

```makefile
test-e2e:           ## Run Playwright E2E tests (requires gateway + Vite dev server + LLM API key)
	cd frontend && NANOBOT_E2E_LLM=1 npx playwright test

test-e2e-headed:    ## Run E2E tests with visible browser
	cd frontend && NANOBOT_E2E_LLM=1 npx playwright test --headed
```

These targets are **NOT** part of `make check` or `make pre-push`. They are run
manually when the developer wants to validate frontend integration, similar to
`make test-integration` for Python LLM tests.

**`test-e2e-headed`** opens a visible browser window — useful for debugging test
failures and watching the interaction in real time.

---

<a id="test-specifications"></a>
## 6. Test Specifications

### Test 1: Full Thread Lifecycle (`thread-lifecycle.spec.ts`)

**What it tests:** New thread → send message → threadId in request → response renders
→ title generates → send second message → same session continues.

This is the highest-value test. It exercises the complete path from thread creation
through two full conversation turns, validating every integration point that broke
in PR #145 and #146.

**Merged scenarios:** #1 (threadId in request), #2 (title auto-generates), #7 (second
message same session).

**Steps:**

1. Navigate to `http://127.0.0.1:5173/`
2. Wait for thread list to load — assert the "New Thread" button is visible in the
   sidebar, indicating the thread list has been fetched from `GET /api/threads` and
   rendered
3. Click "New Thread" button
4. Set up network request observation using `page.route('**/api/chat', ...)` in
   passthrough mode and `page.route('**/api/threads', ...)` to capture the initialize
   response. Requests are observed but not intercepted — they continue to the real
   server.
5. Type a message (e.g., "What is the capital of France?") into the composer textbox
   and press Enter
6. Wait for both `/api/threads` (POST, initialize) and `/api/chat` (POST) requests
   to be captured
7. Assert: `POST /api/threads` was called — this confirms the adapter's `initialize()`
   fired for the new thread
8. Extract the `threadId` from the `POST /api/threads` response body
   (`data.threadId`)
9. Assert: `POST /api/chat` request body contains a `threadId` field matching the
   UUID from step 8 — this is the exact bug from PR #146 where `threadId` was
   missing, causing session mismatch
10. Wait for the assistant response to render in the chat — use a locator that watches
    for text content appearing in the assistant message area. Wait until the content
    stabilizes (no new text for 2s), indicating streaming is complete
11. Assert: At least some text content is visible in the assistant message area (we
    don't assert exact content since real LLM responses vary)
12. Wait up to 10s for the thread title in the sidebar to change — the library calls
    `adapter.generateTitle()` after the first "runEnd" event, which triggers
    `POST /api/threads/{id}/generate-title` on the server
13. Assert: The active thread's button text in the sidebar is NOT "New Chat" — it has
    been replaced by an LLM-generated title. We don't assert the exact title text.
    This validates the full chain: adapter `generateTitle()` → `createAssistantStream`
    → server LLM call → title stored in `session.metadata["title"]` → UI updates
14. Capture the title text for use in cleanup identification
15. Clear the network capture arrays
16. Type a second message (e.g., "And what is its population?") and press Enter
17. Wait for the second `/api/chat` request to be captured
18. Assert: Second `/api/chat` request body contains the same `threadId` as the first
    — this confirms session continuity (the body() callback returns the same threadId
    for follow-up messages)
19. Wait for the second assistant response to render
20. Assert: Both user messages are visible in the chat (the first and second)
21. Cleanup: delete the thread via `DELETE /api/threads/{threadId}`

**Timeout:** 60s (two full LLM round-trips at ~15s each + title generation + setup)

**Key assertions and what they catch:**

| Assertion | Bug it catches |
|-----------|---------------|
| `POST /api/threads` called | adapter.initialize() not firing |
| threadId in first /api/chat | PR #146 bug: missing threadId |
| Response renders | SSE streaming broken, parse failure |
| Title changes from "New Chat" | Wrong AssistantStream format (PR #145 bug) |
| Same threadId in second /api/chat | Session splitting on follow-up messages |
| Both user messages visible | Messages routed to different sessions |

---

### Test 2: Persistence Across Page Reload (`thread-persistence.spec.ts`)

**What it tests:** After a conversation, reload the page → thread appears in sidebar
with its generated title → click it → conversation history loads correctly.

This validates the core feature request: sessions surviving gateway restarts. Since
a page reload exercises the same code path (fresh `adapter.list()` call), this test
covers both "reload" and "restart" scenarios without redundancy.

**Steps:**

1. Navigate to the app
2. Create a new thread with a message using the `createThreadWithMessage` helper:
   - Clicks "New Thread"
   - Sets up network capture
   - Sends a distinctive message (e.g., "E2E persistence test: explain quantum
     entanglement briefly")
   - Waits for response to render
   - Captures the threadId from network and the thread title from the sidebar
   - Returns `{ threadId, title, messageText }`
3. Store the captured `threadId`, `title`, and `messageText` for later assertions
4. Reload the page using `page.reload()`
5. Wait for the thread list to load — assert the "New Thread" button is visible
   (indicates `GET /api/threads` completed and rendered)
6. Assert: A button in the sidebar contains the captured `title` text — this proves
   the title was persisted in `session.metadata["title"]` and returned by
   `GET /api/threads` after the reload
7. Click that thread button to switch to it
8. Wait for message content to appear in the main chat area — the history adapter's
   `load()` fetches `GET /api/threads/{id}/messages` and the library renders the
   conversation
9. Assert: The user message text from step 2 ("quantum entanglement") is visible in
   the chat area
10. Assert: An assistant response is also visible (not just the user message — the
    full conversation was loaded, not just the user's side)
11. Cleanup: delete the thread via `DELETE /api/threads/{threadId}`

**Timeout:** 60s (one LLM round-trip + page reload + history load)

**Key assertions and what they catch:**

| Assertion | Bug it catches |
|-----------|---------------|
| Thread appears after reload | JSONL file not saved, session manager not reading files |
| Title matches pre-reload title | metadata["title"] not persisted, _thread_title() fallback used instead |
| User message visible in history | GET /api/threads/{id}/messages returns wrong data |
| Assistant response visible | History adapter not loading, message format conversion broken |

---

### Test 3: Rapid Messages Stay in Same Thread (`rapid-messages.spec.ts`)

**What it tests:** Send 3 messages in quick succession without waiting for responses
→ all 3 `/api/chat` requests contain the same `threadId`.

This is the race condition test. It directly exercises the fragile
`_currentThreadRemoteId` mutable variable under timing pressure. If the body() callback
or the mutable variable state is inconsistent during rapid input, this test will catch
session splitting.

**Steps:**

1. Navigate to the app
2. Click "New Thread"
3. Set up network capture for all `/api/chat` requests — observe and passthrough,
   collecting `{ threadId, messageText }` from each request body
4. Type first message ("Rapid test message one: hello") and press Enter
5. Wait 500ms — this is enough time for `initialize()` to resolve and
   `_currentThreadRemoteId` to be set, but NOT enough for the full LLM response.
   The first response will still be streaming when the second message is sent.
6. Type second message ("Rapid test message two: how are you") and press Enter
7. Wait 500ms
8. Type third message ("Rapid test message three: goodbye") and press Enter
9. Wait for all 3 `/api/chat` requests to be captured (they may fire immediately
   or be queued by the library)
10. Wait for all responses to complete — up to 60s. The agent may process them
    sequentially (each response completes before the next starts) or in parallel
    depending on the model. Either way, we wait until 3 assistant responses are
    visible in the chat.
11. Extract all `threadId` values from the captured requests
12. Assert: All 3 `threadId` values are the same — no session splitting occurred
13. Assert: All 3 `threadId` values are non-empty (not undefined/null)
14. Assert: All 3 user messages are visible in the chat area (messages weren't lost)
15. Cleanup: delete the thread using the threadId from the first request

**Timeout:** 90s (three sequential LLM calls, each potentially 15-20s)

**Key assertions and what they catch:**

| Assertion | Bug it catches |
|-----------|---------------|
| All 3 threadIds identical | _currentThreadRemoteId race condition |
| All 3 threadIds non-empty | body() callback returns undefined before init resolves |
| All 3 user messages visible | Messages routed to different sessions and lost |

**Flakiness note:** This is the most flaky-prone test in the suite due to LLM response
time variability and the 500ms timing between messages. The 500ms delay was chosen
because:
- Too short (< 200ms): `initialize()` may not have resolved, making every run
  trigger the race condition (which is what we're testing, but we want to test the
  *fix* works, not reproduce the bug)
- Too long (> 2s): The first response might complete, making the test not actually
  exercise rapid-fire behavior
- 500ms is the sweet spot: init has resolved, but the response is still streaming

If this test becomes consistently flaky, the fix is to address the underlying mutable
variable pattern, not to adjust the timing.

---

### Test 4: Thread Navigation (`thread-navigation.spec.ts`)

**What it tests:** Click thread A → history loads → click thread B → different history
→ click back to A → A's history returns. Plus: rapid switching (click B then A within
200ms) → last click wins.

This tests state cleanup during thread switches and the rapid switching race condition
where `_currentThreadRemoteId` might not update fast enough.

**Merged scenarios:** #5 (history loads), #6 (switch between threads), new (rapid
switching).

**Steps:**

1. Navigate to the app
2. Create thread A using `createThreadWithMessage`:
   - Send "Thread A navigation test: what is two plus two?"
   - Wait for response
   - Capture `{ threadId: threadIdA, title: titleA }`
3. Create thread B:
   - Click "New Thread"
   - Send "Thread B navigation test: what is the capital of Japan?"
   - Wait for response
   - Capture `{ threadId: threadIdB, title: titleB }`
4. Thread B should be the active thread (just created and responded to)
5. Assert: Thread B's user message ("capital of Japan") is visible in the chat area
6. Assert: Thread A's user message ("two plus two") is NOT visible in the chat area
7. Click thread A in the sidebar — locate by title text (`titleA`)
8. Wait for history to load — thread A's content appears in the chat area
9. Assert: Thread A's user message ("two plus two") is visible
10. Assert: Thread B's user message ("capital of Japan") is NOT visible — no message
    bleed from the previous active thread
11. Assert: An assistant response is visible in thread A (history fully loaded, not
    just user message)
12. **Rapid switching test:** Click thread B's button in the sidebar, then immediately
    (within 200ms) click thread A's button. This simulates a user who accidentally
    clicks the wrong thread and quickly corrects.
13. Wait 3s for history to settle — the rapid clicks may trigger two history loads,
    and the last one should win
14. Assert: Thread A's messages are visible (last click wins)
15. Assert: Thread B's messages are NOT visible (the intermediate click was superseded)
16. Cleanup: delete both threads via API

**Timeout:** 120s (four LLM calls — two for setup, plus navigation overhead)

**Key assertions and what they catch:**

| Assertion | Bug it catches |
|-----------|---------------|
| Thread A history loads when clicked | History adapter not fetching, _currentThreadRemoteId not updated |
| No message bleed between threads | State not cleaned up on switch |
| Assistant response visible | GET /api/threads/{id}/messages broken or format mismatch |
| Last click wins on rapid switch | _currentThreadRemoteId race during rapid navigation |
| Intermediate click superseded | History from wrong thread displayed |

---

<a id="helpers-and-conventions"></a>
## 7. Helpers and Conventions

### Shared Test Utilities (`frontend/tests/e2e/helpers.ts`)

#### `sendMessage(page: Page, text: string): Promise<void>`

Types text into the composer and sends it.

```typescript
export async function sendMessage(page: Page, text: string): Promise<void> {
  const input = page.getByRole('textbox', { name: 'Message input' });
  await input.fill(text);
  await input.press('Enter');
}
```

Abstracts the input selector so it's defined in one place. If the DOM structure
changes (e.g., assistant-ui renames the textbox), only this function needs updating.

#### `waitForResponse(page: Page): Promise<void>`

Waits for an assistant message to appear and finish streaming.

Implementation approach: locate the assistant message container, wait for text content
to appear, then wait for the content to stabilize (no new text for 2 seconds). The
2-second stabilization window accounts for SSE streaming delays between chunks.

If no assistant response appears within the expect timeout (30s), the assertion fails
with a clear message indicating the response never arrived.

#### `getThreadIdsFromRequests(requests: CapturedRequest[]): string[]`

Extracts `threadId` values from captured `/api/chat` request bodies. Returns an
array of strings. Used by Test 1 (lifecycle) and Test 3 (rapid messages) to assert
threadId consistency.

```typescript
export interface CapturedRequest {
  threadId?: string;
  body?: Record<string, unknown>;
}

export function getThreadIdsFromRequests(requests: CapturedRequest[]): string[] {
  return requests.map(r => r.threadId).filter((id): id is string => !!id);
}
```

#### `deleteThread(request: APIRequestContext, threadId: string): Promise<void>`

Calls `DELETE /api/threads/{threadId}` via Playwright's request API. Used in
`afterEach` cleanup. Does not throw on 404 (thread may not exist if test failed
before creation completed).

```typescript
export async function deleteThread(
  request: APIRequestContext,
  threadId: string,
): Promise<void> {
  await request.delete(`/api/threads/${threadId}`);
}
```

#### `createThreadWithMessage(page: Page, message: string): Promise<ThreadSetupResult>`

Composite helper that creates a thread and sends a message in one call. Used by
Test 2 (persistence) and Test 4 (navigation) for setup.

```typescript
export interface ThreadSetupResult {
  threadId: string;
  title: string;
  messageText: string;
}
```

Implementation:
1. Set up network capture for `/api/threads` (POST) and `/api/chat` (POST)
2. Click "New Thread" button
3. Call `sendMessage(page, message)`
4. Wait for the `/api/threads` POST response to get the `threadId`
5. Call `waitForResponse(page)` to wait for the assistant to finish
6. Wait up to 10s for the thread title to change from "New Chat"
7. Read the title text from the active thread button in the sidebar
8. Clean up the route handlers
9. Return `{ threadId, title, messageText: message }`

This helper encapsulates ~15 lines of setup that multiple tests need. Without it,
Tests 2 and 4 would each duplicate this setup logic.

### Network Request Capture Pattern

Tests that need to inspect API calls use `page.route()` in **passthrough mode**.
The route handler captures request data but calls `route.continue()` so the request
proceeds to the real server unmodified.

```typescript
const chatRequests: CapturedRequest[] = [];

await page.route('**/api/chat', async (route) => {
  const postData = route.request().postData();
  if (postData) {
    const body = JSON.parse(postData);
    chatRequests.push({ threadId: body.threadId, body });
  }
  await route.continue();
});
```

**Why passthrough, not intercept:** We want real server responses. The capture is
purely observational — it records what the frontend sends so we can assert on request
structure (threadId present, consistent across requests) without interfering with the
actual request/response cycle.

**Cleanup:** Route handlers are removed after each test via `page.unroute()` or by
navigating away. The `afterEach` cleanup handles this implicitly.

### Locator Strategy

Tests use **accessible role selectors** matching the DOM structure produced by
assistant-ui primitives. These were verified using Playwright's `browser_snapshot`
tool during the debugging session:

| Element | Selector | Source |
|---------|----------|--------|
| Thread list items | `page.getByRole('button', { name: titleText })` | `ThreadListItemPrimitive.Trigger` renders as button |
| Message input | `page.getByRole('textbox', { name: 'Message input' })` | assistant-ui's composer textbox |
| New Thread button | `page.getByRole('button', { name: 'New Thread' })` | `ThreadListPrimitive.New` button |
| Send button | `page.getByRole('button', { name: 'Send message' })` | Composer submit button |
| User messages | `page.getByText(messageText)` within chat area | Message content |
| Assistant messages | Content blocks within the thread component | assistant-ui message blocks |

**No `data-testid` attributes** are added. The existing DOM uses accessible roles from
assistant-ui primitives which are stable across minor versions. If assistant-ui changes
its DOM structure in a major version, the locators will need updating — but so would
the entire integration layer.

### Test Independence

Each test is **fully independent**:

- Creates its own threads (no shared fixtures)
- Cleans up via `afterEach` (deletes created threads)
- No shared mutable state between tests
- Can run in any order
- Can run in parallel (though parallel execution may cause thread list visual noise
  since all tests share the same server — not recommended for the initial suite)

### Assertion Philosophy

Since tests use a real LLM, response **content** is non-deterministic. Assertions
focus on **structure and behavior**, not exact text:

| Assert this | Not this |
|------------|----------|
| Response renders (text content exists) | Response says "The capital of France is Paris" |
| Title is not "New Chat" | Title is "Conversation about France" |
| User message is visible | User message is at position 3 in the DOM |
| threadId is a UUID | threadId is "abc-123-def" |
| All threadIds match | threadIds match a specific value |

---

<a id="file-structure"></a>
## 8. File Structure

### New Files

| File | Purpose | Approximate Size |
|------|---------|-----------------|
| `frontend/playwright.config.ts` | Playwright configuration: timeouts, browser, reporters, base URL | ~25 lines |
| `frontend/tests/e2e/helpers.ts` | Shared test utilities: `sendMessage`, `waitForResponse`, `createThreadWithMessage`, `deleteThread`, `getThreadIdsFromRequests`, types | ~100 lines |
| `frontend/tests/e2e/thread-lifecycle.spec.ts` | Test 1: full lifecycle (init → chat → title → second message) | ~80 lines |
| `frontend/tests/e2e/thread-persistence.spec.ts` | Test 2: persistence across page reload | ~50 lines |
| `frontend/tests/e2e/rapid-messages.spec.ts` | Test 3: rapid messages stay in same thread | ~60 lines |
| `frontend/tests/e2e/thread-navigation.spec.ts` | Test 4: thread switching + rapid click race condition | ~90 lines |

### Modified Files

| File | Change |
|------|--------|
| `frontend/package.json` | Add `@playwright/test` to `devDependencies` |
| `Makefile` | Add `test-e2e` and `test-e2e-headed` targets |
| `frontend/.gitignore` | Add Playwright artifacts: `test-results/`, `playwright-report/`, `.playwright-mcp/` |

### Directory Structure

```
frontend/
  playwright.config.ts
  tests/
    e2e/
      helpers.ts                      # Shared utilities and types
      thread-lifecycle.spec.ts        # Test 1: P0
      thread-persistence.spec.ts      # Test 2: P0
      rapid-messages.spec.ts          # Test 3: P0
      thread-navigation.spec.ts       # Test 4: P1
```

---

<a id="broader-test-portfolio"></a>
## 9. Broader Test Portfolio by Layer

This spec covers only the E2E layer. For completeness, the recommended full test
portfolio across all layers:

| Layer | What to Test | Count | Status |
|-------|-------------|-------|--------|
| **E2E (Playwright)** | The 4 tests in this spec | 4 | This spec |
| **Backend integration** | Thread CRUD endpoints with real SessionManager | 5-6 | Future |
| **Backend unit** | `_thread_title()` variants, `_strip_attachments()`, SSE event formatting | 3-4 | Future |
| **Frontend unit** | Adapter methods with mocked fetch (list mapping, init return shape) | 4-5 | Future |
| **Frontend unit** | Error handling (fetch fails, server 500, network timeout) | 2-3 | Future |

The E2E layer is implemented first because it catches the integration seam failures
that actually broke in production. Lower layers add coverage for unit-level logic
that hasn't been a source of bugs yet.
