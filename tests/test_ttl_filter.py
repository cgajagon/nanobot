"""Tests for TTL expiry filter in retrieval."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from nanobot.memory.read.retriever import filter_expired


class TestFilterExpired:
    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def test_no_ttl_passes_through(self) -> None:
        items = [{"id": "1", "summary": "No TTL", "timestamp": "2020-01-01T00:00:00Z"}]
        result = filter_expired(items, self._now())
        assert len(result) == 1

    def test_expired_event_excluded(self) -> None:
        old_ts = (self._now() - timedelta(days=60)).isoformat()
        items = [{"id": "1", "summary": "Old task", "timestamp": old_ts, "ttl_days": 30}]
        result = filter_expired(items, self._now())
        assert len(result) == 0

    def test_fresh_event_with_ttl_included(self) -> None:
        recent_ts = (self._now() - timedelta(days=5)).isoformat()
        items = [{"id": "1", "summary": "Fresh task", "timestamp": recent_ts, "ttl_days": 30}]
        result = filter_expired(items, self._now())
        assert len(result) == 1

    def test_ttl_uses_last_confirmed_when_available(self) -> None:
        old_ts = (self._now() - timedelta(days=60)).isoformat()
        recent_confirmed = (self._now() - timedelta(days=5)).isoformat()
        items = [
            {
                "id": "1",
                "summary": "Old but confirmed",
                "timestamp": old_ts,
                "last_confirmed": recent_confirmed,
                "ttl_days": 30,
            }
        ]
        result = filter_expired(items, self._now())
        assert len(result) == 1

    def test_invalid_ttl_ignored(self) -> None:
        old_ts = (self._now() - timedelta(days=60)).isoformat()
        items: list[dict[str, Any]] = [
            {"id": "1", "summary": "Bad TTL", "timestamp": old_ts, "ttl_days": -5},
            {"id": "2", "summary": "Zero TTL", "timestamp": old_ts, "ttl_days": 0},
            {"id": "3", "summary": "String TTL", "timestamp": old_ts, "ttl_days": "thirty"},
        ]
        result = filter_expired(items, self._now())
        assert len(result) == 3

    def test_expired_even_with_old_last_confirmed(self) -> None:
        """Events are excluded when both timestamp AND last_confirmed exceed TTL."""
        old_ts = (self._now() - timedelta(days=60)).isoformat()
        old_confirmed = (self._now() - timedelta(days=45)).isoformat()
        items = [
            {
                "id": "1",
                "summary": "Old confirmed but still expired",
                "timestamp": old_ts,
                "last_confirmed": old_confirmed,
                "ttl_days": 30,
            }
        ]
        result = filter_expired(items, self._now())
        assert len(result) == 0

    def test_no_timestamp_passes_through(self) -> None:
        items = [{"id": "1", "summary": "No ts", "ttl_days": 30}]
        result = filter_expired(items, self._now())
        assert len(result) == 1
