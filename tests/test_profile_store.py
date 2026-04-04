"""Tests for ProfileCache and ProfileStore."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from nanobot.memory.db import MemoryDatabase
from nanobot.memory.persistence.profile_io import ProfileCache, ProfileStore


class TestProfileCache:
    def test_read_returns_empty_dict_when_file_missing(self):
        cache = ProfileCache()
        result = cache.read()
        assert result == {}

    def test_read_loads_from_disk_on_first_call(self):
        cache = ProfileCache()
        cache.write({"preferences": ["tea"]})
        result = cache.read()
        assert result == {"preferences": ["tea"]}

    def test_read_uses_cache_on_second_call(self):
        cache = ProfileCache()
        cache.write({"preferences": ["tea"]})
        result1 = cache.read()
        result2 = cache.read()
        assert result1 == result2 == {"preferences": ["tea"]}

    def test_write_updates_cache_atomically(self):
        cache = ProfileCache()
        data = {"preferences": ["coffee"]}
        cache.write(data)
        result = cache.read()
        assert result == data

    def test_invalidate_forces_reload(self):
        cache = ProfileCache()
        cache.write({"preferences": ["tea"]})
        assert cache.read() == {"preferences": ["tea"]}
        cache.invalidate()
        assert cache.read() == {}


class TestProfileStoreReadWrite:
    def _make_store(self, tmp_path: Path) -> ProfileStore:
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir(exist_ok=True)
        db = MemoryDatabase(mem_dir / "memory.db", dims=32)
        return ProfileStore(db=db)

    def test_read_profile_returns_dict(self, tmp_path: Path):
        store = self._make_store(tmp_path)
        result = store.read_profile()
        assert isinstance(result, dict)

    def test_write_profile_and_read_back(self, tmp_path: Path):
        store = self._make_store(tmp_path)
        store.write_profile({"preferences": ["tea"], "stable_facts": []})
        result = store.read_profile()
        assert result["preferences"] == ["tea"]

    def test_fresh_db_read_does_not_warn(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, propagate_loguru_to_caplog: None
    ):
        """A fresh DB with no profile row is normal init, not a parse error."""
        store = self._make_store(tmp_path)
        with caplog.at_level(logging.WARNING):
            store.read_profile()
        assert "Failed to parse memory profile" not in caplog.text

    def test_fresh_db_read_persists_default_profile(self, tmp_path: Path):
        """First read on a fresh DB should persist the default so the row exists."""
        store = self._make_store(tmp_path)
        store.read_profile()
        # Second read should hit the DB row, not return None from DB
        row = store._db.read_profile("profile")
        assert row is not None
        assert isinstance(row, dict)
        assert "preferences" in row
