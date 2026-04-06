# Delete `nanobot ui` Command — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the legacy `nanobot ui` CLI command and all code that exists solely to support it (dual-mode WebChannel, `owns_lifecycle` in app factory).

**Architecture:** Three commits: (1) delete the command + registration + docs, (2) simplify WebChannel by removing unmanaged mode and its dispatcher, (3) simplify `create_app()` by removing `owns_lifecycle`. Each commit is independently valid.

**Tech Stack:** Python 3.10+, Typer, FastAPI, pytest, ruff, mypy

**Spec:** `docs/superpowers/specs/2026-04-06-delete-nanobot-ui-command-design.md`

---

### Task 1: Delete `ui()` Function and Command Registration

**Files:**
- Modify: `nanobot/cli/gateway.py` — delete lines 283-386
- Modify: `nanobot/cli/commands.py` — delete import and registration
- Modify: `docker-compose.yml` — delete `nanobot-ui` service
- Modify: `README.md` — delete `nanobot ui` row
- Modify: `docs/superpowers/specs/2026-03-23-readme-rewrite-design.md` — delete row
- Modify: `docs/superpowers/plans/2026-03-23-readme-rewrite.md` — delete row
- Modify: `docs/superpowers/specs/2026-03-22-cli-split-design.md` — add deprecation note
- Modify: `docs/superpowers/plans/2026-03-22-cli-split.md` — add deprecation note

- [ ] **Step 1: Delete `ui()` function from gateway.py**

Remove the entire `ui()` function (lines 283-386) from `nanobot/cli/gateway.py`.
Also remove the `os` import at line 5 if it becomes unused (check — `gateway()`
does not use `os`, only `ui()` uses `os.environ.get`).

After deletion, verify `gateway.py` still has:
```python
from nanobot.cli.gateway import gateway as _gateway_impl
```
as its only public export used by `commands.py`.

- [ ] **Step 2: Remove import and command registration from commands.py**

In `nanobot/cli/commands.py`:

Delete line 25:
```python
from nanobot.cli.gateway import ui as _ui_impl
```

Delete line 74:
```python
app.command()(_ui_impl)
```

The file should have `from nanobot.cli.gateway import gateway as _gateway_impl`
(line 24) and `app.command()(_gateway_impl)` (line 73) remaining.

- [ ] **Step 3: Delete `nanobot-ui` service from docker-compose.yml**

Remove lines 53-68 (the entire `nanobot-ui` service block):

```yaml
  nanobot-ui:
    <<: *common-config
    container_name: nanobot-ui
    command: ["ui", "--host", "0.0.0.0", "--port", "8000"]
    profiles:
      - ui
    ports:
      - "8000:8000"
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 1G
        reservations:
          cpus: '0.25'
          memory: 256M
```

- [ ] **Step 4: Delete `nanobot ui` from documentation**

In `README.md`, delete line 750:
```
| `nanobot ui` | Launch web UI |
```

In `docs/superpowers/specs/2026-03-23-readme-rewrite-design.md`, delete line 127:
```
| `nanobot ui` | Launch web UI |
```

In `docs/superpowers/plans/2026-03-23-readme-rewrite.md`, delete line 482:
```
| `nanobot ui` | Launch web UI |
```

- [ ] **Step 5: Annotate historical CLI split docs**

In `docs/superpowers/specs/2026-03-22-cli-split-design.md`, change line 4 from:
```
**Status:** Draft
```
to:
```
**Status:** Implemented (note: `nanobot ui` command was removed in 2026-04-06)
```

In `docs/superpowers/plans/2026-03-22-cli-split.md`, add after line 1:
```
> **Note:** The `nanobot ui` command referenced in this plan was removed on
> 2026-04-06. The web UI is served by `nanobot gateway` with `channels.web.enabled`.
```

- [ ] **Step 6: Run lint and typecheck**

Run: `make lint && make typecheck`
Expected: PASS — no references to deleted symbols remain.

- [ ] **Step 7: Run tests**

Run: `make test`
Expected: All tests pass. No test depends on the `ui` command.

- [ ] **Step 8: Verify no stale references**

Run:
```bash
grep -rn "_ui_impl\|\"nanobot ui\"\|'nanobot ui'" nanobot/ tests/ README.md docs/ docker-compose.yml --include="*.py" --include="*.md" --include="*.yml" --include="*.yaml"
```
Expected: Zero matches (historical docs may mention `ui` in context — that's fine,
but no functional references should remain).

- [ ] **Step 9: Commit**

```bash
git add nanobot/cli/gateway.py nanobot/cli/commands.py docker-compose.yml README.md \
  docs/superpowers/specs/2026-03-23-readme-rewrite-design.md \
  docs/superpowers/plans/2026-03-23-readme-rewrite.md \
  docs/superpowers/specs/2026-03-22-cli-split-design.md \
  docs/superpowers/plans/2026-03-22-cli-split.md
git commit -m "refactor(cli): remove legacy nanobot ui command

The web UI is served by nanobot gateway when channels.web.enabled is true.
The standalone ui command was a pre-integration leftover that lacked cron
execution, heartbeat, channel delivery, and outbound routing."
```

---

### Task 2: Simplify WebChannel — Remove Unmanaged Mode

**Files:**
- Modify: `nanobot/channels/web.py` — remove `managed` param, `_dispatcher_task`, `_dispatch_outbound()`
- Modify: `nanobot/channels/manager.py:121` — remove `managed=True` kwarg
- Modify: `tests/test_web_channel.py` — update start/stop tests, remove dispatcher tests

- [ ] **Step 1: Update test expectations for simplified start/stop**

In `tests/test_web_channel.py`, rewrite `TestWebChannelStartStop`:

```python
class TestWebChannelStartStop:
    async def test_start_sets_running(self, channel: WebChannel):
        await channel.start()
        assert channel._running is True
        await channel.stop()

    async def test_stop_clears_running(self, channel: WebChannel):
        await channel.start()
        await channel.stop()
        assert channel._running is False

    async def test_stop_noop_without_start(self, channel: WebChannel):
        await channel.stop()  # should not raise
```

- [ ] **Step 2: Delete dispatcher-dependent tests**

In `tests/test_web_channel.py`, delete the two tests that call
`channel._dispatch_outbound()` directly:

- `test_routes_web_messages` (around line 122-131) — tests dispatch routing;
  this behavior is already tested in `tests/test_channel_manager.py`
- `test_ignores_other_channels` (around line 133-141) — tests channel filtering;
  ChannelManager handles this

- [ ] **Step 3: Run tests to confirm test changes are valid**

Run: `pytest tests/test_web_channel.py -v`
Expected: FAIL — tests reference old behavior not yet removed from source.

- [ ] **Step 4: Simplify WebChannel class**

In `nanobot/channels/web.py`:

Replace the `__init__` method — remove `managed` parameter and `_dispatcher_task`:

```python
def __init__(self, config: Any, bus: MessageBus) -> None:
    super().__init__(config, bus)
    # chat_id → queue of outbound messages for that thread's SSE stream
    self._streams: dict[str, asyncio.Queue[OutboundMessage | None]] = {}
    # chat_ids whose SSE stream was closed (client disconnect / stop button).
    # Messages for these are silently dropped to avoid log spam from the
    # agent loop which may still be running.
    self._disconnected: set[str] = set()
```

Replace `start()`:

```python
async def start(self) -> None:
    """Mark the channel as running.

    Message routing is handled by :class:`ChannelManager`'s dispatcher
    which calls :meth:`send` directly.
    """
    self._running = True
```

Replace `stop()`:

```python
async def stop(self) -> None:
    """Mark the channel as stopped."""
    self._running = False
```

Delete the entire `_dispatch_outbound()` method (lines 127-148) and its
section comment (lines 123-125).

Update the class docstring to remove managed/unmanaged explanation:

```python
class WebChannel(BaseChannel):
    """Request-driven channel for the web UI.

    Unlike long-running channels (Telegram, Discord) this channel is driven
    by incoming HTTP requests.  Message routing is handled by
    :class:`ChannelManager`'s dispatcher which calls :meth:`send` directly.
    ``start()``/``stop()`` manage only the running state.
    """
```

- [ ] **Step 5: Remove `managed=True` from ChannelManager**

In `nanobot/channels/manager.py`, change lines 118-122 from:

```python
self.channels["web"] = WebChannel(
    self.config.channels.web,
    self.bus,
    managed=True,
)
```

to:

```python
self.channels["web"] = WebChannel(
    self.config.channels.web,
    self.bus,
)
```

- [ ] **Step 6: Run tests**

Run: `make test`
Expected: All tests pass.

- [ ] **Step 7: Run lint and typecheck**

Run: `make lint && make typecheck`
Expected: PASS.

- [ ] **Step 8: Verify no stale references**

Run:
```bash
grep -rn "managed\|_dispatch_outbound\|_dispatcher_task\|_managed" nanobot/channels/web.py tests/test_web_channel.py
```
Expected: Zero matches.

- [ ] **Step 9: Commit**

```bash
git add nanobot/channels/web.py nanobot/channels/manager.py tests/test_web_channel.py
git commit -m "refactor(channels): remove unmanaged mode from WebChannel

WebChannel is always managed by ChannelManager. Remove the managed
parameter, private dispatcher task, and _dispatch_outbound method.
ChannelManager's dispatcher handles all outbound message routing."
```

---

### Task 3: Simplify `create_app()` — Remove `owns_lifecycle`

**Files:**
- Modify: `nanobot/web/app.py` — remove `owns_lifecycle` param and conditional shutdown
- Modify: `nanobot/cli/gateway.py` — remove `owns_lifecycle=True` from call site (already gone after Task 1, but verify the gateway call site doesn't pass it either)

- [ ] **Step 1: Simplify `create_app()` in app.py**

In `nanobot/web/app.py`:

Remove `owns_lifecycle` from the function signature (line 50) and docstring
(lines 62-63).

Replace the lifespan context manager (lines 66-74):

```python
@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[misc]
    yield
```

Remove the `asynccontextmanager` import if no longer needed — check: it IS still
needed for the lifespan definition above.

- [ ] **Step 2: Verify gateway call site**

In `nanobot/cli/gateway.py`, find the `create_app()` call inside `gateway()`.
Confirm it does NOT pass `owns_lifecycle` (it passes `owns_lifecycle=False`
implicitly via the default). No change needed here — the parameter is simply
removed from the function signature.

Grep to confirm:
```bash
grep -n "owns_lifecycle" nanobot/
```
Expected: Zero matches after the edit.

- [ ] **Step 3: Run lint and typecheck**

Run: `make lint && make typecheck`
Expected: PASS.

- [ ] **Step 4: Run tests**

Run: `make test`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add nanobot/web/app.py
git commit -m "refactor(web): remove owns_lifecycle from create_app

The gateway always manages shutdown externally. The owns_lifecycle
parameter existed only for the now-deleted nanobot ui command."
```

---

### Task 4: Final Verification

- [ ] **Step 1: Run full check suite**

Run: `make check`
Expected: All checks pass (lint, typecheck, import-check, structure-check,
prompt-check, phase-todo-check, doc-check).

- [ ] **Step 2: Run full test suite**

Run: `make test`
Expected: All tests pass.

- [ ] **Step 3: Final grep for any remaining references**

Run:
```bash
grep -rn "nanobot ui\|_ui_impl\|owns_lifecycle\|_dispatch_outbound\|_managed\|managed=True" \
  nanobot/ tests/ README.md docker-compose.yml --include="*.py" --include="*.md" --include="*.yml"
```
Expected: Zero matches. (Historical docs may mention `ui` contextually — those
are acceptable as annotations, not functional references.)
