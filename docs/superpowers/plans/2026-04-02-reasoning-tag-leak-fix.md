# Fix Reasoning Block Leak in Web UI

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop `[REASONING]...[/REASONING]` blocks from appearing in web UI responses by switching the reasoning prompt to use `<think>` tags, which `strip_think()` already removes.

**Architecture:** The reasoning prompt template (`reasoning.md`) instructs the LLM to output `[REASONING]...[/REASONING]` blocks, but `strip_think()` only removes `<think>...</think>` tags. Switching the prompt to `<think>` tags makes the existing stripping logic handle reasoning blocks automatically — both in the streaming path (`TextChunk` via `full_clean`) and the final response path (`turn_runner.py` lines 284, 320).

**Tech Stack:** Python 3.10+, pytest

**Worktree:** `../nanobot-fix-reasoning-leak` (branch `fix/reasoning-tag-leak`)

---

## File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Modify | `nanobot/templates/prompts/reasoning.md` | Switch `[REASONING]...[/REASONING]` to `<think>...</think>` |
| Modify | `tests/test_no_answer_recovery.py` | Add test for `<think>` reasoning block stripping |
| Modify | `tests/test_turn_runner.py:355-383` | Update test to use `<think>` tags |

---

### Task 1: Add strip_think test for reasoning blocks

**Files:**
- Modify: `tests/test_no_answer_recovery.py:209-227` (TestStripThink class)

- [ ] **Step 1: Write the failing test**

Add to the `TestStripThink` class in `tests/test_no_answer_recovery.py` after the existing `test_mixed_think_and_text` test (line 227):

```python
    def test_reasoning_block_stripped(self):
        """<think> reasoning blocks (from reasoning.md prompt) are stripped."""
        result = strip_think(
            "<think>\n1. What does the user need? Summarize a file.\n"
            "2. What am I looking for? A meeting transcript.\n</think>\n\n"
            "Here is the summary of the meeting."
        )
        assert result == "Here is the summary of the meeting."

    def test_reasoning_block_only_returns_none(self):
        """A response containing only a reasoning block returns None."""
        assert strip_think(
            "<think>\n1. What does the user need? Find a project.\n</think>"
        ) is None
```

- [ ] **Step 2: Run tests to verify they pass**

These tests should already pass because `strip_think()` handles `<think>` tags. This confirms the existing infrastructure works.

Run: `pytest tests/test_no_answer_recovery.py::TestStripThink -v`
Expected: PASS (all 6 tests)

- [ ] **Step 3: Commit**

```bash
git add tests/test_no_answer_recovery.py
git commit -m "test: add strip_think tests for reasoning block format"
```

---

### Task 2: Switch reasoning prompt from `[REASONING]` to `<think>` tags

**Files:**
- Modify: `nanobot/templates/prompts/reasoning.md`

- [ ] **Step 1: Update the prompt template**

Replace the entire content of `nanobot/templates/prompts/reasoning.md` with:

```markdown
# Reasoning Protocol

## Before Taking Action

When you receive a task, work through these steps before calling any tool.
**You MUST output your reasoning in a `<think>` block before your first
tool call.** This block is mandatory — never skip it.

Format:

```
<think>
1. What does the user need? <find, read, create, modify, summarize>
2. What am I looking for? <describe the target and its likely type>
   - A project code or identifier → likely a FOLDER or FILE NAME
   - A topic or keyword → likely FILE CONTENT
   - A tag, property, or date → likely METADATA
   - A specific document → likely a FILE PATH
3. Which tool or command matches, and why? <tool choice + reasoning>
   - Find by name → list_dir, or skill commands that list/browse
   - Search content → grep/search commands
   - Read known file → read_file
   - Explore structure → list_dir first, then narrow down
4. What will I try if this returns nothing? <a DIFFERENT approach, not tweaked arguments>
5. Source check: Am I about to cite memory or tool results? If memory, have I verified it with a tool?
</think>
```

Every question must be answered. Keep each answer to 1-2 lines.

## When a Tool Returns Empty Results

STOP. Do not report "not found" to the user.

"No results" means your APPROACH may be wrong — not that the data
doesn't exist. The user told you it exists.

Ask yourself:
- Could the search term be a folder name instead of file content?
- Could it be a file name instead of a tag?
- Should I list the directory structure instead of searching?

Try your fallback approach before responding.

## When a Tool Returns an Error

Read the error message. Classify it:
- Wrong arguments → fix the syntax and retry
- Command not found → use a different command
- Permission denied → try a different approach entirely
- Timeout → try a simpler operation

Do not retry the same failing command unchanged.

## Fallback Principle

Your base tools (list_dir, read_file) always work. If specialized
tools or skill commands fail, fall back to the filesystem.
The filesystem is ground truth.
```

- [ ] **Step 2: Run lint**

Run: `make lint`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add nanobot/templates/prompts/reasoning.md
git commit -m "fix(agent): switch reasoning prompt from [REASONING] to <think> tags

The [REASONING]...[/REASONING] blocks were leaking into web UI responses
because strip_think() only removes <think>...</think> tags. Switching the
prompt to use <think> tags makes the existing stripping logic handle
reasoning blocks in both the streaming and final response paths."
```

---

### Task 3: Update turn_runner test to use `<think>` tags

**Files:**
- Modify: `tests/test_turn_runner.py:355-383`

- [ ] **Step 1: Update the test**

In `tests/test_turn_runner.py`, find `test_tool_call_preserves_assistant_content` (line 356). Update lines 365 and 381 to use `<think>` tags:

Replace line 365:
```python
                _tool_response([tc1], content="[REASONING]\n1. Need: list files\n[/REASONING]"),
```
With:
```python
                _tool_response([tc1], content="<think>\n1. Need: list files\n</think>"),
```

Replace line 381:
```python
        assert asst_msg["content"] == "[REASONING]\n1. Need: list files\n[/REASONING]", (
```
With:
```python
        assert asst_msg["content"] == "<think>\n1. Need: list files\n</think>", (
```

Note: This test verifies that content is preserved in the **message history** (for context between LLM calls). The `<think>` block stays in `state.messages` so the model can see its prior reasoning. It is only stripped from `final_content` (the user-facing response) via `strip_think()` at turn_runner.py lines 284/320.

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_turn_runner.py::TestToolUseLoop::test_tool_call_preserves_assistant_content -v`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `make check && make test`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_turn_runner.py
git commit -m "test: update reasoning block test to use <think> tags"
```

---

## Verification

After all tasks, verify the fix end-to-end:

1. Start the web UI (`make web` or equivalent)
2. Send a message like "Summarize details in Obsidian for project DS10540"
3. Confirm the `<think>` block does NOT appear in the chat response
4. Check the session JSONL file — the `<think>` block SHOULD appear in assistant messages with `tool_calls` (preserved for context) but NOT in the final assistant response
