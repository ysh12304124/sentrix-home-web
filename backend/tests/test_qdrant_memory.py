import json
import os
import tempfile

import pytest

from backend.db import MemoryStore
from backend.qdrant_memory import get_qdrant_index
from scripts.maintenance.sync_qdrant_vectors import sync


qdrant_client = pytest.importorskip("qdrant_client")


def test_qdrant_returns_none_when_dir_locked_by_another_process(monkeypatch):
    import fcntl

    with tempfile.TemporaryDirectory() as directory:
        monkeypatch.setenv("SENTRIX_VECTOR_BACKEND", "qdrant")
        monkeypatch.setenv("SENTRIX_QDRANT_PATH", os.path.join(directory, "qdrant"))
        monkeypatch.setenv("SENTRIX_QDRANT_COLLECTION_PREFIX", "test_locked")
        original_flock = fcntl.flock

        def rejecting_flock(fd, operation):
            if operation & fcntl.LOCK_NB and operation & fcntl.LOCK_EX:
                raise BlockingIOError("directory owned by another process")
            return original_flock(fd, operation)

        monkeypatch.setattr(fcntl, "flock", rejecting_flock)
        assert get_qdrant_index(os.path.join(directory, "memory.db")) is None


def test_qdrant_single_instance_lock_acquired(monkeypatch):
    import fcntl

    with tempfile.TemporaryDirectory() as directory:
        monkeypatch.setenv("SENTRIX_VECTOR_BACKEND", "qdrant")
        monkeypatch.setenv("SENTRIX_QDRANT_PATH", os.path.join(directory, "qdrant"))
        monkeypatch.setenv("SENTRIX_QDRANT_COLLECTION_PREFIX", "test_owner")
        index = get_qdrant_index(os.path.join(directory, "memory.db"))
        assert index is not None
        lock_path = os.path.join(directory, "qdrant", ".sentrix-qdrant.lock")
        assert os.path.exists(lock_path)
        assert get_qdrant_index(os.path.join(directory, "memory.db")) is index
        fd = os.open(lock_path, os.O_RDONLY)
        try:
            with pytest.raises((BlockingIOError, OSError)):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)



def test_qdrant_dual_write_search_and_scope_fallback(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        monkeypatch.setenv("SENTRIX_VECTOR_BACKEND", "qdrant")
        monkeypatch.setenv("SENTRIX_QDRANT_PATH", os.path.join(directory, "qdrant"))
        monkeypatch.setenv("SENTRIX_QDRANT_COLLECTION_PREFIX", "test_memory")
        store = MemoryStore(os.path.join(directory, "memory.db"))
        store.upsert_vector("visual", "asset", "asset-a", [1.0, 0.0], "clip", {"scope_id": "album-a"})
        store.upsert_vector("visual", "asset", "asset-b", [0.0, 1.0], "clip", {"scope_id": "album-b"})

        hits = store.search_vectors("visual", [1.0, 0.0], scope_id="album-a", model_name="clip")
        assert [item["source_id"] for item in hits] == ["asset-a"]
        assert store.vector_search_status()["active_backend"] == "qdrant"

        sqlite_hits = store.search_vectors_sqlite(
            "visual", [1.0, 0.0], scope_id="album-a", model_name="clip"
        )
        assert [item["source_id"] for item in sqlite_hits] == ["asset-a"]
        store.close()


def test_different_dimensions_and_models_are_isolated(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        monkeypatch.setenv("SENTRIX_VECTOR_BACKEND", "qdrant")
        monkeypatch.setenv("SENTRIX_QDRANT_PATH", os.path.join(directory, "qdrant"))
        monkeypatch.setenv("SENTRIX_QDRANT_COLLECTION_PREFIX", "test_dimensions")
        store = MemoryStore(os.path.join(directory, "memory.db"))
        store.upsert_vector("visual", "asset", "clip-asset", [1.0, 0.0], "clip")
        store.upsert_vector("visual", "asset", "cn-asset", [1.0, 0.0, 0.0], "chinese-clip")
        assert [item["source_id"] for item in store.search_vectors(
            "visual", [1.0, 0.0], model_name="clip"
        )] == ["clip-asset"]
        assert [item["source_id"] for item in store.search_vectors(
            "visual", [1.0, 0.0, 0.0], model_name="chinese-clip"
        )] == ["cn-asset"]
        assert store.vector_search_status()["collections"] == 2
        store.close()


def test_qdrant_never_returns_rows_deleted_from_sqlite(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        monkeypatch.setenv("SENTRIX_VECTOR_BACKEND", "qdrant")
        monkeypatch.setenv("SENTRIX_QDRANT_PATH", os.path.join(directory, "qdrant"))
        monkeypatch.setenv("SENTRIX_QDRANT_COLLECTION_PREFIX", "test_stale")
        store = MemoryStore(os.path.join(directory, "memory.db"))
        store.upsert_vector("visual", "asset", "deleted-asset", [1.0, 0.0], "clip")
        store.connection.execute(
            "DELETE FROM memory_vectors WHERE source_type = 'asset' AND source_id = ?",
            ("deleted-asset",),
        )
        store.connection.commit()

        assert store.search_vectors("visual", [1.0, 0.0], model_name="clip") == []
        assert store.vector_search_status()["active_backend"] == "sqlite_fallback"
        store.close()


def test_qdrant_missing_collection_is_explicit_in_fallback_status(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        monkeypatch.setenv("SENTRIX_VECTOR_BACKEND", "qdrant")
        monkeypatch.setenv("SENTRIX_QDRANT_PATH", os.path.join(directory, "qdrant"))
        monkeypatch.setenv("SENTRIX_QDRANT_COLLECTION_PREFIX", "test_missing")
        store = MemoryStore(os.path.join(directory, "memory.db"))
        assert store.search_vectors("visual", [1.0, 0.0], model_name="clip") == []
        status = store.vector_search_status()
        assert status["active_backend"] == "sqlite_fallback"
        assert status["error"] == "qdrant_no_collection"
        store.close()


def test_full_sync_removes_orphaned_qdrant_points(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        monkeypatch.setenv("SENTRIX_VECTOR_BACKEND", "qdrant")
        monkeypatch.setenv("SENTRIX_QDRANT_PATH", os.path.join(directory, "qdrant"))
        monkeypatch.setenv("SENTRIX_QDRANT_COLLECTION_PREFIX", "test_rebuild")
        store = MemoryStore(os.path.join(directory, "memory.db"))
        store.upsert_vector("visual", "asset", "current", [1.0, 0.0], "clip")
        index = get_qdrant_index(store.path)
        index.upsert(
            row_id="stale-row", scope_id="home-default", space="visual",
            source_type="asset", source_id="stale", vector=[0.9, 0.1],
            model_name="clip", metadata={},
        )

        result = sync(store, index)
        hits = index.search(space="visual", vector=[1.0, 0.0], limit=10, model_name="clip")

        assert result["cleared_collections"] == 1
        assert [item["source_id"] for item in hits] == ["current"]
        store.close()


def test_generic_start_script_keeps_sqlite_as_the_default_backend():
    script = os.path.join(
        os.path.dirname(__file__), "..", "..", "scripts", "runtime", "start_sentrix_api.sh"
    )
    content = open(script, encoding="utf-8").read()
    assert 'SENTRIX_VECTOR_BACKEND:-sqlite' in content
