"""Tests for ConsolidationOrchestrator (rewritten for TaskGroup API)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from nanobot.agent.consolidation import ConsolidationOrchestrator


def _orch(archive_fn=None):
    memory = MagicMock()
    memory.consolidate = AsyncMock(return_value=True)
    return ConsolidationOrchestrator(
        memory=memory,
        archive_fn=archive_fn or MagicMock(),
        max_concurrent=2,
        memory_window=50,
        enable_contradiction_check=True,
    ), memory


class TestInProgressDeduplication:
    async def test_second_submit_for_same_session_is_noop(self):
        orch, memory = _orch()
        session = MagicMock()
        session.messages = []
        async with orch:
            orch.submit("s", session, MagicMock(), "m")
            orch.submit("s", session, MagicMock(), "m")
        assert memory.consolidate.call_count == 1


class TestSubmitAndConsolidateAndWait:
    async def test_submit_runs_in_background(self):
        orch, memory = _orch()
        session = MagicMock()
        session.messages = []
        async with orch:
            orch.submit("s", session, MagicMock(), "m")
        memory.consolidate.assert_called_once()

    async def test_consolidate_and_wait_is_awaitable(self):
        orch, memory = _orch()
        session = MagicMock()
        session.messages = []
        async with orch:
            result = await orch.consolidate_and_wait("s", session, MagicMock(), "m")
        assert result is True


class TestArchiveFnOnFailure:
    async def test_archive_fn_called_on_consolidate_failure(self):
        archive = MagicMock()
        orch, memory = _orch(archive_fn=archive)
        memory.consolidate = AsyncMock(side_effect=RuntimeError("boom"))
        session = MagicMock()
        session.messages = [{"role": "user", "content": "hi"}]
        async with orch:
            orch.submit("s", session, MagicMock(), "m")
        archive.assert_called_once()


class TestRateLimiterIntegration:
    """Consolidation respects the shared rate limiter."""

    async def test_submit_calls_wait_if_needed_before_consolidate(self):
        """Rate limiter wait is called before the LLM-backed consolidation."""
        rl = AsyncMock()
        rl.wait_if_needed = AsyncMock(return_value=0.0)
        rl.record = MagicMock()

        memory = MagicMock()
        memory.consolidate = AsyncMock(return_value=True)

        orch = ConsolidationOrchestrator(
            memory=memory,
            archive_fn=MagicMock(),
            max_concurrent=2,
            memory_window=50,
            enable_contradiction_check=True,
            rate_limiter=rl,
        )
        session = MagicMock()
        session.messages = []
        async with orch:
            orch.submit("s", session, MagicMock(), "m")
        rl.wait_if_needed.assert_awaited_once()
        memory.consolidate.assert_called_once()

    async def test_submit_records_tokens_after_consolidate(self):
        """Rate limiter records estimated tokens after consolidation completes."""
        rl = AsyncMock()
        rl.wait_if_needed = AsyncMock(return_value=0.0)
        rl.record = MagicMock()

        memory = MagicMock()
        memory.consolidate = AsyncMock(return_value=True)

        orch = ConsolidationOrchestrator(
            memory=memory,
            archive_fn=MagicMock(),
            max_concurrent=2,
            memory_window=50,
            enable_contradiction_check=True,
            rate_limiter=rl,
        )
        session = MagicMock()
        session.messages = [{"role": "user", "content": "hello " * 500}]
        async with orch:
            orch.submit("s", session, MagicMock(), "m")
        rl.record.assert_called_once()
        recorded_tokens = rl.record.call_args[0][0]
        assert recorded_tokens > 0

    async def test_submit_without_rate_limiter_still_works(self):
        """Consolidation without a rate limiter works as before (backward compat)."""
        memory = MagicMock()
        memory.consolidate = AsyncMock(return_value=True)

        orch = ConsolidationOrchestrator(
            memory=memory,
            archive_fn=MagicMock(),
            max_concurrent=2,
            memory_window=50,
            enable_contradiction_check=True,
        )
        session = MagicMock()
        session.messages = []
        async with orch:
            orch.submit("s", session, MagicMock(), "m")
        memory.consolidate.assert_called_once()

    async def test_consolidate_and_wait_also_respects_rate_limiter(self):
        """The blocking consolidate_and_wait path also checks the rate limiter."""
        rl = AsyncMock()
        rl.wait_if_needed = AsyncMock(return_value=0.0)
        rl.record = MagicMock()

        memory = MagicMock()
        memory.consolidate = AsyncMock(return_value=True)

        orch = ConsolidationOrchestrator(
            memory=memory,
            archive_fn=MagicMock(),
            max_concurrent=2,
            memory_window=50,
            enable_contradiction_check=True,
            rate_limiter=rl,
        )
        session = MagicMock()
        session.messages = []
        async with orch:
            await orch.consolidate_and_wait("s", session, MagicMock(), "m")
        rl.wait_if_needed.assert_awaited_once()
