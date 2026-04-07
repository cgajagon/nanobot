"""Tests for micro-extraction trivial-turn skip pre-filter."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from nanobot.memory.write.micro_extractor import (
    _TRIVIAL_ASSISTANT_MAX_LEN,
    _TRIVIAL_MAX_LEN,
    _TRIVIAL_PATTERNS,
    MicroExtractor,
)


def _make_extractor(*, enabled: bool = True) -> tuple[MicroExtractor, MagicMock]:
    """Build a MicroExtractor with mocked provider and ingester."""
    provider = MagicMock()
    # Make chat() an AsyncMock that blocks briefly, keeping the task alive
    # long enough for assertions on _pending_tasks.
    blocking_event = asyncio.Event()

    async def _blocking_chat(**kwargs: object) -> MagicMock:
        await blocking_event.wait()
        resp = MagicMock()
        resp.tool_calls = []
        return resp

    provider.chat = _blocking_chat
    ingester = MagicMock()
    extractor = MicroExtractor(
        provider=provider,
        ingester=ingester,
        model="gpt-4o-mini",
        enabled=enabled,
    )
    return extractor, provider


class TestTrivialTurnSkip:
    """Trivial user messages skip the LLM call entirely."""

    @pytest.mark.asyncio
    async def test_ok_skipped(self) -> None:
        ext, provider = _make_extractor()
        await ext.submit("ok", "Got it.", channel="cli")
        assert len(ext._pending_tasks) == 0

    @pytest.mark.asyncio
    async def test_thanks_with_punctuation_skipped(self) -> None:
        ext, provider = _make_extractor()
        await ext.submit("Thanks!", "You're welcome.", channel="cli")
        assert len(ext._pending_tasks) == 0

    @pytest.mark.asyncio
    async def test_yes_question_mark_skipped(self) -> None:
        ext, provider = _make_extractor()
        await ext.submit("yes?", "Proceeding.", channel="cli")
        assert len(ext._pending_tasks) == 0

    @pytest.mark.asyncio
    async def test_emoji_skipped(self) -> None:
        ext, provider = _make_extractor()
        await ext.submit("\U0001f44d", "Great!", channel="cli")
        assert len(ext._pending_tasks) == 0

    @pytest.mark.asyncio
    async def test_mixed_case_skipped(self) -> None:
        ext, provider = _make_extractor()
        await ext.submit("OKAY", "Fine.", channel="cli")
        assert len(ext._pending_tasks) == 0

    @pytest.mark.asyncio
    async def test_multiword_trivial_skipped(self) -> None:
        ext, provider = _make_extractor()
        await ext.submit("sounds good", "Thanks!", channel="cli")
        assert len(ext._pending_tasks) == 0

    @pytest.mark.asyncio
    async def test_go_ahead_skipped(self) -> None:
        ext, provider = _make_extractor()
        await ext.submit("go ahead", "Starting now.", channel="cli")
        assert len(ext._pending_tasks) == 0


class TestNonTrivialPassThrough:
    """Non-trivial messages must always pass through to LLM extraction."""

    @pytest.mark.asyncio
    async def test_meaningful_short_message_passes(self) -> None:
        ext, provider = _make_extractor()
        await ext.submit("no, use Python 3.12", "Updating to 3.12.", channel="cli")
        assert len(ext._pending_tasks) >= 1

    @pytest.mark.asyncio
    async def test_long_message_passes(self) -> None:
        ext, provider = _make_extractor()
        await ext.submit(
            "The vault is at C:\\Users\\me\\Documents\\PM",
            "Noted.",
            channel="cli",
        )
        assert len(ext._pending_tasks) >= 1

    @pytest.mark.asyncio
    async def test_trivial_user_long_assistant_passes(self) -> None:
        """Trivial user msg + long assistant response -> must NOT skip."""
        ext, provider = _make_extractor()
        long_assistant = "Actually, I realize DS10540 is in a different folder. " * 5
        assert len(long_assistant.strip()) > _TRIVIAL_ASSISTANT_MAX_LEN
        await ext.submit("ok", long_assistant, channel="cli")
        assert len(ext._pending_tasks) >= 1


class TestEmptyMessageSkip:
    """Empty or whitespace-only messages are skipped immediately."""

    @pytest.mark.asyncio
    async def test_empty_string_skipped(self) -> None:
        ext, provider = _make_extractor()
        await ext.submit("", "Hello.", channel="cli")
        assert len(ext._pending_tasks) == 0

    @pytest.mark.asyncio
    async def test_whitespace_only_skipped(self) -> None:
        ext, provider = _make_extractor()
        await ext.submit("   ", "Response.", channel="cli")
        assert len(ext._pending_tasks) == 0


class TestDisabledExtractor:
    """When disabled, submit() is a no-op regardless of message content."""

    @pytest.mark.asyncio
    async def test_disabled_skips_everything(self) -> None:
        ext, provider = _make_extractor(enabled=False)
        await ext.submit("important fact about the user", "Noted.", channel="cli")
        assert len(ext._pending_tasks) == 0


class TestTrivialConstants:
    """Verify the constants are well-formed."""

    def test_patterns_is_frozenset(self) -> None:
        assert isinstance(_TRIVIAL_PATTERNS, frozenset)

    def test_all_patterns_are_lowercase(self) -> None:
        for p in _TRIVIAL_PATTERNS:
            assert p == p.lower() or not p.isascii(), f"Pattern {p!r} is not lowercase"

    def test_max_len_is_positive(self) -> None:
        assert _TRIVIAL_MAX_LEN > 0

    def test_assistant_max_len_is_positive(self) -> None:
        assert _TRIVIAL_ASSISTANT_MAX_LEN > 0

    def test_all_patterns_within_max_len(self) -> None:
        for p in _TRIVIAL_PATTERNS:
            assert len(p) <= _TRIVIAL_MAX_LEN, (
                f"Pattern {p!r} ({len(p)} chars) exceeds _TRIVIAL_MAX_LEN ({_TRIVIAL_MAX_LEN})"
            )
