"""Wiring contract tests: memory subsystem construction and state propagation."""

from __future__ import annotations

from pathlib import Path

from nanobot.config.memory import MemoryConfig
from nanobot.memory.db.connection import MemoryDatabase
from nanobot.memory.store import MemoryStore


def test_strategies_ddl_importable():
    """STRATEGIES_DDL is a public constant in constants, used by test fixtures."""
    from nanobot.memory.constants import STRATEGIES_DDL

    assert "CREATE TABLE" in STRATEGIES_DDL
    assert "strategies" in STRATEGIES_DDL


def test_memory_database_connection_property(tmp_path):
    """MemoryDatabase exposes a shared connection for subsystem components."""
    import sqlite3

    store = _make_store(tmp_path)
    conn = store.db.connection
    assert isinstance(conn, sqlite3.Connection)
    # Verify strategies table exists in the shared database
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='strategies'")
    assert cursor.fetchone() is not None


def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(
        tmp_path,
        embedding_provider="hash",
        memory_config=MemoryConfig(graph_enabled=False),
    )


def test_conflict_resolve_gap_follows_memory_config(tmp_path):
    """ConflictManager receives MemoryConfig directly at construction."""
    store = _make_store(tmp_path)

    # Default gap is 0.25
    assert store.conflict_mgr._memory_config is not None
    assert store.conflict_mgr._memory_config.conflict_auto_resolve_gap == 0.25

    # Construct a store with a custom gap value.
    store2 = MemoryStore(
        tmp_path / "store2",
        embedding_provider="hash",
        memory_config=MemoryConfig(graph_enabled=False, conflict_auto_resolve_gap=0.5),
    )
    assert store2.conflict_mgr._memory_config.conflict_auto_resolve_gap == 0.5


def test_profile_mgr_has_conflict_mgr_after_construction(tmp_path):
    """ProfileStore._conflict_mgr is wired after MemoryStore construction."""
    store = _make_store(tmp_path)
    assert store.profile_mgr._conflict_mgr is not None
    assert store.profile_mgr._conflict_mgr is store.conflict_mgr


def test_profile_mgr_corrector_wired(tmp_path):
    """ProfileStore._corrector is wired to CorrectionOrchestrator after construction."""
    store = _make_store(tmp_path)
    assert store.profile_mgr._corrector is not None


def test_rollout_override_atomic_consistency(tmp_path):
    """Scorer receives MemoryConfig directly at construction."""
    store = _make_store(tmp_path)

    # Scorer holds a direct reference to the MemoryConfig passed at construction.
    scorer_config = store._scorer._memory_config
    assert scorer_config is store._memory_config


def test_maintenance_reindex_removed(tmp_path):
    """reindex_from_structured_memory was a no-op stub and has been removed."""
    store = _make_store(tmp_path)
    assert not hasattr(store.maintenance, "reindex_from_structured_memory")


def test_alias_registry_table_exists(tmp_path: Path) -> None:
    """alias_registry table is created by MemoryDatabase schema init."""
    store = _make_store(tmp_path)
    cursor = store.db.connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='alias_registry'"
    )
    assert cursor.fetchone() is not None


def test_alias_store_property(tmp_path: Path) -> None:
    """MemoryDatabase exposes alias_store lazy property."""
    store = _make_store(tmp_path)
    alias_store = store.db.alias_store
    assert alias_store is not None
    # Should be the same instance on second access
    assert store.db.alias_store is alias_store


def test_last_confirmed_field_on_memory_event() -> None:
    """MemoryEvent has last_confirmed and source_role fields (stored in _extra overflow)."""
    from nanobot.memory.event import MemoryEvent

    event = MemoryEvent(summary="test", last_confirmed="2025-01-01T00:00:00Z", source_role="user")
    assert event.last_confirmed == "2025-01-01T00:00:00Z"
    assert event.source_role == "user"


def test_source_role_in_micro_extract_schema() -> None:
    """_MICRO_EXTRACT_TOOL schema includes source_role field."""
    from nanobot.memory.write.micro_extractor import _MICRO_EXTRACT_TOOL

    props = _MICRO_EXTRACT_TOOL[0]["function"]["parameters"]["properties"]["events"]["items"][
        "properties"
    ]
    assert "source_role" in props
    assert props["source_role"]["type"] == "string"
    assert "user" in props["source_role"]["enum"]
    assert "assistant" in props["source_role"]["enum"]


def test_stability_half_life_constants_defined():
    """_STABILITY_HALF_LIFE has entries for all three stability levels."""
    from nanobot.memory.read.scoring import _STABILITY_HALF_LIFE

    assert "high" in _STABILITY_HALF_LIFE
    assert "medium" in _STABILITY_HALF_LIFE
    assert "low" in _STABILITY_HALF_LIFE
    assert _STABILITY_HALF_LIFE["high"] > _STABILITY_HALF_LIFE["medium"]
    assert _STABILITY_HALF_LIFE["medium"] > _STABILITY_HALF_LIFE["low"]


def test_onnx_unavailable_falls_back_to_composite(tmp_path: Path) -> None:
    """When ONNX reranker reports unavailable, store uses CompositeReranker."""
    import nanobot.memory.ranking.onnx_reranker as onnx_mod
    from nanobot.memory.ranking.reranker import CompositeReranker

    original = onnx_mod._ort
    try:
        onnx_mod._ort = None  # simulate missing onnxruntime
        store = MemoryStore(tmp_path)
        assert isinstance(store._reranker, CompositeReranker)
    finally:
        onnx_mod._ort = original


class TestBusyTimeout:
    """SQLite connection has busy_timeout set for concurrent access safety."""

    def test_busy_timeout_set(self, tmp_path: Path) -> None:
        """Connection should have busy_timeout >= 5000 for WAL concurrent writes."""
        db = MemoryDatabase(tmp_path / "test.db", dims=384)
        result = db.connection.execute("PRAGMA busy_timeout").fetchone()
        assert result is not None
        timeout = result[0]
        assert timeout >= 5000, f"Expected busy_timeout >= 5000, got {timeout}"
