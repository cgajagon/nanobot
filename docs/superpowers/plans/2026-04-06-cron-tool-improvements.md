# Cron Tool Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix two defects in the cron tool: (1) the `message` parameter description misleads the LLM into summarizing the user's prompt, and (2) there is no way to manually trigger a cron job to test what it will actually do.

**Architecture:** Both changes are scoped to `nanobot/tools/builtin/cron.py`. Fix 1 changes the tool schema description. Fix 2 adds a `run` action that delegates to the existing `CronService.run_job()` async method. No changes to the cron service, gateway, or agent loop.

**Tech Stack:** Python 3.10+, pytest, pytest-asyncio

**Status:** COMPLETED — merged in PR #161.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `nanobot/tools/builtin/cron.py` | Modify | Update `message` description, add `run` to action enum, add `_run_job()` method |
| `tests/test_cron_tool.py` | Modify | Add tests for new description text and `run` action |

---

### Task 1: Fix the `message` parameter description

**Files:**
- Modify: `nanobot/tools/builtin/cron.py:25`
- Test: `tests/test_cron_tool.py`

- [x] **Step 1: Write the failing test**
- [x] **Step 2: Run test to verify it fails**
- [x] **Step 3: Update the description**
- [x] **Step 4: Run test to verify it passes**
- [x] **Step 5: Run lint and typecheck**
- [x] **Step 6: Commit**

---

### Task 2: Add `run` action to trigger a cron job on demand

**Files:**
- Modify: `nanobot/tools/builtin/cron.py:22,62-87`
- Test: `tests/test_cron_tool.py`

- [x] **Step 1: Write the failing tests**
- [x] **Step 2: Run tests to verify they fail**
- [x] **Step 3: Add `run` to the action enum**
- [x] **Step 4: Add the `run` action to the execute dispatch and the is_running guard**
- [x] **Step 5: Add the `_run_job` method**
- [x] **Step 6: Run tests to verify they pass**
- [x] **Step 7: Run full test suite and checks**
- [x] **Step 8: Update the tool description to mention `run`**
- [x] **Step 9: Run the description test to verify it still passes**
- [x] **Step 10: Commit**

---

### Task 3: Final validation

- [x] **Step 1: Run make check**
- [x] **Step 2: Run make test**
- [x] **Step 3: Verify the schema is consistent**
