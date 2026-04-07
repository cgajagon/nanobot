"""Message sanitization for LLM providers.

Pure functions that normalize message lists before sending to LLM APIs.
Repairs orphaned tool_calls and tool_results to prevent API validation errors.
"""

from __future__ import annotations

from typing import Any

# Keys that LLM APIs accept in message dicts. Everything else is stripped.
_ALLOWED_MSG_KEYS: frozenset[str] = frozenset(
    {"role", "content", "tool_calls", "tool_call_id", "name"}
)


def sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip non-standard keys, ensure content key, and repair orphaned tool_calls.

    This is the single authoritative location for message sanitization before
    LLM API calls. Called by LiteLLMProvider.chat() and stream_chat().
    """
    sanitized = []
    for msg in messages:
        clean = {k: v for k, v in msg.items() if k in _ALLOWED_MSG_KEYS}
        # Strict providers require "content" even when assistant only has tool_calls
        if clean.get("role") == "assistant" and "content" not in clean:
            clean["content"] = None
        sanitized.append(clean)

    # Forward repair: strip assistant tool_calls that lack matching tool results.
    # This handles mid-turn crashes where tool execution was interrupted.
    tool_result_ids: set[str] = {
        m["tool_call_id"] for m in sanitized if m.get("role") == "tool" and "tool_call_id" in m
    }
    repaired = []
    for msg in sanitized:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            orphaned = [
                tc for tc in msg["tool_calls"] if tc.get("id") and tc["id"] not in tool_result_ids
            ]
            if orphaned:
                valid = [tc for tc in msg["tool_calls"] if tc["id"] in tool_result_ids]
                if valid:
                    repaired.append({**msg, "tool_calls": valid})
                else:
                    repaired.append({k: v for k, v in msg.items() if k != "tool_calls"})
                continue
        repaired.append(msg)
    return repaired
