"""Contract tests: tool-message pairing invariant across the full pipeline.

These tests verify that no combination of session truncation and provider
sanitization produces orphaned tool_result messages (tool results without
matching assistant tool_calls).
"""

from __future__ import annotations

from nanobot.context.compression import compress_context
from nanobot.providers.sanitize import sanitize_messages
from nanobot.session.manager import Session


def _assert_no_orphaned_tool_results(messages: list[dict]) -> None:
    """Assert that every tool result has a matching assistant tool_call."""
    assistant_tc_ids: set[str] = set()
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                if tc.get("id"):
                    assistant_tc_ids.add(tc["id"])

    for m in messages:
        if m.get("role") == "tool" and m.get("tool_call_id"):
            assert m["tool_call_id"] in assistant_tc_ids, (
                f"Orphaned tool_result: tool_call_id={m['tool_call_id']} "
                f"not in assistant tool_calls {assistant_tc_ids}"
            )


class TestMessagePairingInvariant:
    """The pipeline must never produce orphaned tool results."""

    def test_large_session_small_window(self):
        """39 messages, max_messages=25 — the exact incident scenario."""
        session = Session(key="test")
        session.messages = [
            {"role": "user", "content": "Set up cron job"},
        ]
        for i in range(15):
            tc_id = f"toolu_{i:04d}"
            session.messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc_id,
                            "type": "function",
                            "function": {"name": "exec", "arguments": "{}"},
                        }
                    ],
                }
            )
            session.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": "exec",
                    "content": f"result_{i}",
                }
            )
        session.messages.append({"role": "assistant", "content": "Done"})
        # 33 messages total

        history = session.get_history(max_messages=25)
        sanitized = sanitize_messages(history)
        _assert_no_orphaned_tool_results(sanitized)

    def test_all_tool_messages_no_user(self):
        """Session with only tool-call cycles, no user messages at all."""
        session = Session(key="test")
        session.messages = []
        for i in range(10):
            tc_id = f"tc_{i}"
            session.messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc_id,
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ],
                }
            )
            session.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": "f",
                    "content": f"r{i}",
                }
            )
        # 20 messages, all tool cycles

        history = session.get_history(max_messages=8)
        sanitized = sanitize_messages(history)
        _assert_no_orphaned_tool_results(sanitized)

    def test_multi_tool_batch_split(self):
        """Assistant with multiple tool_calls split across the window boundary."""
        session = Session(key="test")
        session.messages = [
            {"role": "user", "content": "start"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc_a",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    },
                    {
                        "id": "tc_b",
                        "type": "function",
                        "function": {"name": "g", "arguments": "{}"},
                    },
                    {
                        "id": "tc_c",
                        "type": "function",
                        "function": {"name": "h", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "tc_a", "name": "f", "content": "ra"},
            {"role": "tool", "tool_call_id": "tc_b", "name": "g", "content": "rb"},
            {"role": "tool", "tool_call_id": "tc_c", "name": "h", "content": "rc"},
            {"role": "assistant", "content": "final"},
        ]

        history = session.get_history(max_messages=4)
        sanitized = sanitize_messages(history)
        _assert_no_orphaned_tool_results(sanitized)

    def test_full_pipeline_with_compression(self):
        """Session truncation + sync compression never produces orphaned tool results."""
        session = Session(key="test")
        # Build a session with enough messages to trigger both truncation and compression
        session.messages = [{"role": "user", "content": "start"}]
        for i in range(20):
            tc_id = f"toolu_{i:04d}"
            session.messages.append(
                {
                    "role": "assistant",
                    "content": f"Let me check item {i}",
                    "tool_calls": [
                        {
                            "id": tc_id,
                            "type": "function",
                            "function": {"name": "exec", "arguments": "{}"},
                        }
                    ],
                }
            )
            session.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": "exec",
                    "content": f"Result for item {i}: " + "x" * 200,
                }
            )
        session.messages.append({"role": "assistant", "content": "All done."})
        # 43 messages total

        # Step 1: Session truncation
        history = session.get_history(max_messages=25)

        # Step 2: Build message list (simulate what build_messages does)
        messages = [{"role": "system", "content": "You are helpful."}] + history

        # Step 3: Compression (sync path, tight budget to force dropping)
        compressed = compress_context(messages, max_tokens=2000)

        # Step 4: Sanitize (last gate before API)
        final = sanitize_messages(compressed)

        # Invariant: no orphaned tool results at any stage
        _assert_no_orphaned_tool_results(history)
        _assert_no_orphaned_tool_results(compressed)
        _assert_no_orphaned_tool_results(final)
