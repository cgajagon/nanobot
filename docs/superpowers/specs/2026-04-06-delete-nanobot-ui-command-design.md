# Delete `nanobot ui` Command

> Spec for removing the legacy `nanobot ui` CLI command and all supporting code.
> Date: 2026-04-06.

## Context

The `nanobot ui` command was the original entry point for the web UI, created on
2026-03-13 (`debb93d2`). Three days later (`daeec3f7`), the web UI was integrated
into `nanobot gateway` as a proper channel alongside Telegram, email, etc.

Since then, `nanobot ui` has been a degraded duplicate:
- No cron execution (jobs are created but never fire)
- No heartbeat
- No channel delivery (message tool, email)
- No ChannelManager (outbound routing broken)
- No contact provider

Production and staging deployments use `nanobot gateway` exclusively. The `ui`
command exists only as a leftover from before the gateway integration.

## Goal

Delete `nanobot ui` and all code that exists solely to support it. No backward
compatibility. No aliases. No deprecation warnings.

## Scope

### Delete

| File | Lines | What |
|------|-------|------|
| `nanobot/cli/gateway.py` | 283-386 | Entire `ui()` function |
| `nanobot/cli/commands.py` | 25 | `from nanobot.cli.gateway import ui as _ui_impl` |
| `nanobot/cli/commands.py` | 74 | `app.command()(_ui_impl)` |
| `docker-compose.yml` | 53-68 | `nanobot-ui` service section |
| `README.md` | 750 | `nanobot ui` row in CLI table |
| `docs/superpowers/specs/2026-03-23-readme-rewrite-design.md` | 127 | `nanobot ui` row |
| `docs/superpowers/plans/2026-03-23-readme-rewrite.md` | 482 | `nanobot ui` row |
| `nanobot/channels/web.py` | 127-148 | `_dispatch_outbound()` method (only used in unmanaged mode) |

### Simplify

**`nanobot/channels/web.py`** — Remove `managed` parameter and dual-mode logic:
- Remove `managed` parameter from `__init__` (line 39)
- Remove `self._managed` field (line 48)
- Remove `self._dispatcher_task` field (line 47) — no longer needed
- Remove conditional in `start()` (lines 61-62) — `start()` becomes `self._running = True`
- Simplify `stop()` — just set `self._running = False`, no dispatcher task to cancel
- Delete `_dispatch_outbound()` method entirely (lines 127-148)
- Update class docstring to remove managed/unmanaged explanation

**`nanobot/channels/manager.py`** — Remove `managed=True` from WebChannel
construction (line 121). WebChannel no longer accepts this parameter.

**`nanobot/web/app.py`** — Remove `owns_lifecycle` parameter and its logic:
- Remove `owns_lifecycle` parameter from `create_app()` (line 50)
- Remove docstring mention (line 62-63)
- Simplify lifespan to just `yield` (lines 66-74) — gateway manages shutdown

### Update Tests

**`tests/test_web_channel.py`**:
- `TestWebChannelStartStop` — Tests assert `_dispatcher_task` exists after `start()`.
  After simplification, `start()` just sets `_running = True`. Update these tests.
- `test_routes_web_messages` and `test_ignores_other_channels` call
  `channel._dispatch_outbound()` directly. Delete these tests — the method no longer
  exists. The dispatch behavior is tested in `test_channel_manager.py` via
  ChannelManager's dispatcher.
- Fixture creates `WebChannel(cfg, bus)` without `managed` — still valid since we're
  removing the parameter entirely.

**`tests/test_commands_gateway_agent.py`** — No changes needed. Tests only cover
`gateway()`, not `ui()`. The `_WebChannel` mock doesn't use `managed`.

### Update Historical Docs (annotate, not delete)

The following files are historical plan/spec documents. They reference `ui` as
part of the original CLI split work. Add a note at the top stating the `ui` command
was removed:

- `docs/superpowers/specs/2026-03-22-cli-split-design.md`
- `docs/superpowers/plans/2026-03-22-cli-split.md`

## Frontend Proxy

`frontend/vite.config.ts` proxies `/api` to `http://127.0.0.1:8000`. This port
matches the default `WebChannelConfig.port` in gateway mode, so no change is
needed. The proxy works identically whether the backend is `nanobot gateway` or
was `nanobot ui` — both serve on the configured web port.

## What Stays

- `WebChannelConfig` in `config/schema.py` — used by gateway
- `NANOBOT_WEB_API_KEY` env var — gateway reads it via config
- `create_app()` function — gateway calls it
- `WebChannel` class — gateway uses it via ChannelManager
- All gateway tests, deployment configs, Dockerfile
- Frontend code (unchanged)

## Verification

After all changes:
```bash
make check          # lint + typecheck + structural checks
make test           # unit tests pass
grep -rn "nanobot ui\|_ui_impl\|owns_lifecycle\|managed.*True\|_dispatch_outbound" \
  nanobot/ tests/ --include="*.py"   # zero matches
grep -rn "nanobot ui" README.md docs/ docker-compose.yml  # zero matches
```
