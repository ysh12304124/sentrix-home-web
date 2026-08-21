"""Optional Qdrant mirror for Sentrix memory vectors.

SQLite remains the source of truth.  Qdrant is a derived, rebuildable index:
new vectors are dual-written and searches fall back to SQLite when the client
or a collection is unavailable.  Collections are split by vector space,
model and dimension so incompatible embedding checkpoints can never mix.
Each collection still uses a named vector, following the delivered Ego4D
Qdrant payload/vector layout.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
import threading
import uuid
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None


_CLIENTS = {}
_CLIENTS_LOCK = threading.Lock()
_DIR_LOCKS = {}
_POINT_NAMESPACE = uuid.UUID("09cba006-7577-4e7b-b973-f58e005d6822")
_LOCK_FILENAME = ".sentrix-qdrant.lock"


def _enabled() -> bool:
    return os.getenv("SENTRIX_VECTOR_BACKEND", "sqlite").strip().lower() == "qdrant"


def _safe(value: str, limit: int = 18) -> str:
    text = re.sub(r"[^a-zA-Z0-9_]+", "_", str(value or "").lower()).strip("_")
    return (text or "default")[:limit]


class QdrantMemoryIndex:
    def __init__(self, path: str, prefix: str = "sentrix_memory"):
        self.path = str(Path(path).resolve())
        self.prefix = _safe(prefix, 30)
        self._client = None
        self._lock = threading.RLock()
        self.last_error = None

    @property
    def available(self) -> bool:
        try:
            self._get_client().get_collections()
            return True
        except Exception as error:
            self.last_error = f"{type(error).__name__}: {error}"
            return False

    def _get_client(self):
        if self._client is not None:
            return self._client
        with self._lock:
            if self._client is None:
                from qdrant_client import QdrantClient

                Path(self.path).mkdir(parents=True, exist_ok=True)
                self._client = QdrantClient(path=self.path)
        return self._client

    def _collection(self, space: str, model_name: str, dimension: int) -> str:
        identity = f"{space}\0{model_name}\0{dimension}".encode("utf-8")
        suffix = hashlib.sha1(identity).hexdigest()[:10]
        return f"{self.prefix}_{_safe(space)}_{_safe(model_name)}_{dimension}_{suffix}"[:255]

    def _ensure_collection(self, collection: str, space: str, dimension: int) -> None:
        from qdrant_client import models

        client = self._get_client()
        names = {item.name for item in client.get_collections().collections}
        if collection in names:
            return
        client.create_collection(
            collection_name=collection,
            vectors_config={
                space: models.VectorParams(size=dimension, distance=models.Distance.COSINE),
            },
        )

    @staticmethod
    def _point_id(space: str, source_type: str, source_id: str, model_name: str) -> str:
        return str(uuid.uuid5(
            _POINT_NAMESPACE,
            f"{space}\0{source_type}\0{source_id}\0{model_name}",
        ))

    def upsert(self, *, row_id: str, scope_id: str, space: str, source_type: str,
               source_id: str, vector: list[float], model_name: str,
               metadata: dict | None = None, created_at: str | None = None,
               updated_at: str | None = None) -> None:
        from qdrant_client import models

        if not vector:
            return
        collection = self._collection(space, model_name, len(vector))
        with self._lock:
            self._ensure_collection(collection, space, len(vector))
            payload = {
                "row_id": row_id,
                "scope_id": scope_id or "home-default",
                "space": space,
                "source_type": source_type,
                "source_id": source_id,
                "model_name": model_name,
                "metadata": dict(metadata or {}),
                "created_at": created_at,
                "updated_at": updated_at,
                "vector_names": [space],
                "level": source_type,
            }
            self._get_client().upsert(
                collection_name=collection,
                points=[models.PointStruct(
                    id=self._point_id(space, source_type, source_id, model_name),
                    vector={space: [float(value) for value in vector]},
                    payload=payload,
                )],
                wait=True,
            )
        self.last_error = None

    def _matching_collections(self, space: str, dimension: int,
                              model_name: str | None) -> list[str]:
        client = self._get_client()
        if model_name:
            expected = self._collection(space, model_name, dimension)
            names = {item.name for item in client.get_collections().collections}
            return [expected] if expected in names else []
        prefix = f"{self.prefix}_{_safe(space)}_"
        dim_marker = f"_{dimension}_"
        return [item.name for item in client.get_collections().collections
                if item.name.startswith(prefix) and dim_marker in item.name]

    def clear(self) -> int:
        """Drop only collections owned by this Sentrix index prefix."""
        with self._lock:
            client = self._get_client()
            collections = [
                item.name for item in client.get_collections().collections
                if item.name.startswith(f"{self.prefix}_")
            ]
            for collection in collections:
                client.delete_collection(collection_name=collection)
        self.last_error = None
        return len(collections)

    def search(self, *, space: str, vector: list[float], limit: int,
               scope_id: str | None = None, model_name: str | None = None) -> list[dict]:
        from qdrant_client import models

        client = self._get_client()
        query_filter = None
        if scope_id:
            query_filter = models.Filter(must=[models.FieldCondition(
                key="scope_id", match=models.MatchValue(value=scope_id),
            )])
        results = []
        for collection in self._matching_collections(space, len(vector), model_name):
            kwargs = {
                "collection_name": collection,
                "query": [float(value) for value in vector],
                "using": space,
                "query_filter": query_filter,
                "limit": max(1, int(limit)),
                "with_payload": True,
                "with_vectors": False,
            }
            response = client.query_points(**kwargs)
            for point in response.points:
                payload = dict(point.payload or {})
                results.append({
                    "id": payload.get("row_id"),
                    "scope_id": payload.get("scope_id") or "home-default",
                    "space": payload.get("space") or space,
                    "source_type": payload.get("source_type"),
                    "source_id": payload.get("source_id"),
                    "model_name": payload.get("model_name"),
                    "metadata_json": dict(payload.get("metadata") or {}),
                    "created_at": payload.get("created_at"),
                    "updated_at": payload.get("updated_at"),
                    "score": float(point.score),
                })
        results.sort(key=lambda item: item["score"], reverse=True)
        self.last_error = None
        return results[:max(1, int(limit))]

    def collection_stats(self) -> dict:
        client = self._get_client()
        collections = [item.name for item in client.get_collections().collections
                       if item.name.startswith(f"{self.prefix}_")]
        points = 0
        for name in collections:
            try:
                points += int(client.get_collection(name).points_count or 0)
            except Exception:
                pass
        return {"backend": "qdrant", "path": self.path,
                "collections": len(collections), "points": points,
                "last_error": self.last_error}


def _acquire_dir_lock(directory: str):
    """Take an exclusive flock on the Qdrant dir so only one API process owns it.

    Returns the open fd on success, or None when another process already holds
    the lock.  A POSIX advisory lock is released automatically when the fd is
    closed or the process exits, so no explicit unlock is needed on shutdown.
    """
    if fcntl is None:
        return True  # non-POSIX: no cross-process guard available
    lock_path = Path(directory) / _LOCK_FILENAME
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return fd


def get_qdrant_index(db_path: str | None = None) -> QdrantMemoryIndex | None:
    if not _enabled():
        return None
    configured = os.getenv("SENTRIX_QDRANT_PATH")
    if configured:
        path = configured
    elif db_path and db_path != ":memory:":
        path = str(Path(db_path).resolve().parent / "qdrant")
    else:
        return None
    prefix = os.getenv("SENTRIX_QDRANT_COLLECTION_PREFIX", "sentrix_memory")
    key = (str(Path(path).resolve()), prefix)
    with _CLIENTS_LOCK:
        if key not in _CLIENTS:
            lock_fd = _acquire_dir_lock(path)
            if lock_fd is None:
                return None
            index = QdrantMemoryIndex(path, prefix)
            if isinstance(lock_fd, int):
                _DIR_LOCKS[key] = lock_fd
            _CLIENTS[key] = index
        return _CLIENTS[key]


def close_qdrant_clients() -> None:
    with _CLIENTS_LOCK:
        clients = list(_CLIENTS.values())
        _CLIENTS.clear()
        locks = list(_DIR_LOCKS.values())
        _DIR_LOCKS.clear()
    for index in clients:
        try:
            if index._client is not None:
                index._client.close()
        except Exception:
            pass
    for fd in locks:
        try:
            os.close(fd)
        except Exception:
            pass


atexit.register(close_qdrant_clients)
