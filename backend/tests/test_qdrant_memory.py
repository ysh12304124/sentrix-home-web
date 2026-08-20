import json
import os
import tempfile

import pytest

from backend.db import MemoryStore


qdrant_client = pytest.importorskip("qdrant_client")


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
