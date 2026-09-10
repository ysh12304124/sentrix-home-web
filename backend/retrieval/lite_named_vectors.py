"""Read migrated Sentrix Lite named vectors as native retrieval channels.

The legacy local Qdrant collection contains five named vectors per memory
unit.  This adapter exposes the four requested semantic routes as independent
native ``Retriever`` instances so the main backend's weighted RRF can fuse
them with its lexical channel.  Legacy asset ids are mapped to native assets
through ``metadata_json.legacy_asset_id``; no HTTP service on port 8091 is
required.
"""

from __future__ import annotations

import atexit
import os
import threading
from dataclasses import dataclass
from pathlib import Path

from .base import CandidateHit


ROUTES = (
    ("clip_image", "image", "visual"),
    ("visual_caption", "visual", "text"),
    ("object", "object", "text"),
    ("relation", "relation", "text"),
)


class LiteNamedVectorIndex:
    def __init__(self, path: str, collection: str):
        self.path = str(Path(path).resolve())
        self.collection = collection
        self._client = None
        self._lock = threading.RLock()
        self._embedding_cache: dict[tuple[int, str, str], list[float]] = {}
        self._embedding_cache_order: list[tuple[int, str, str]] = []
        self.last_error = None

    def _open(self):
        if self._client is None:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(path=self.path)
        return self._client

    def query(self, *, route: str, vector: list[float], scope_id: str,
              media_types: tuple[str, ...], limit: int):
        from qdrant_client import models

        must = [models.FieldCondition(
            key="scope_id", match=models.MatchValue(value=scope_id),
        )]
        if len(media_types) == 1:
            must.append(models.FieldCondition(
                key="media_type", match=models.MatchValue(value=media_types[0]),
            ))
        query_filter = models.Filter(must=must)
        with self._lock:
            response = self._open().query_points(
                collection_name=self.collection,
                query=[float(value) for value in vector],
                using=route,
                query_filter=query_filter,
                limit=max(1, int(limit)),
                with_payload=True,
                with_vectors=False,
            )
        self.last_error = None
        return list(response.points)

    def embed(self, embedding_router, slot: str, text: str) -> list[float]:
        """Reuse one query embedding across named-vector routes.

        ``visual_caption``, ``object`` and ``relation`` all use the same BGE
        query space.  Without this cache one RRF retrieval needlessly calls
        the local embedding sidecar three times for identical input.
        """
        key = (id(embedding_router), slot, text)
        with self._lock:
            cached = self._embedding_cache.get(key)
            if cached is not None:
                return list(cached)
        if slot == "visual":
            vector = embedding_router.embed_visual(text)
        else:
            vector = embedding_router.embed_text(text)
        vector = [float(value) for value in (vector or [])]
        if not vector:
            return []
        with self._lock:
            if key not in self._embedding_cache:
                self._embedding_cache[key] = vector
                self._embedding_cache_order.append(key)
                while len(self._embedding_cache_order) > 64:
                    expired = self._embedding_cache_order.pop(0)
                    self._embedding_cache.pop(expired, None)
        return list(vector)

    def close(self):
        with self._lock:
            if self._client is not None:
                self._client.close()
                self._client = None


_INDEXES: dict[tuple[str, str], LiteNamedVectorIndex] = {}
_INDEX_LOCK = threading.Lock()


def _shared_index(path: str, collection: str) -> LiteNamedVectorIndex:
    key = (str(Path(path).resolve()), collection)
    with _INDEX_LOCK:
        if key not in _INDEXES:
            _INDEXES[key] = LiteNamedVectorIndex(*key)
        return _INDEXES[key]


def close_lite_named_vector_indexes():
    with _INDEX_LOCK:
        values = list(_INDEXES.values())
        _INDEXES.clear()
    for value in values:
        try:
            value.close()
        except Exception:
            pass


atexit.register(close_lite_named_vector_indexes)


@dataclass
class LiteNamedVectorRetriever:
    name: str
    vector_name: str
    embedding_slot: str
    kind: str = "primary"

    def __init__(self, store, embedding_router, *, name: str, vector_name: str,
                 embedding_slot: str, index: LiteNamedVectorIndex, legacy_scope_id: str):
        self.store = store
        self.embedding_router = embedding_router
        self.name = name
        self.vector_name = vector_name
        self.embedding_slot = embedding_slot
        self.index = index
        self.legacy_scope_id = legacy_scope_id
        self._legacy_to_native = None
        self._status = "uninitialized"
        self.backend_used = "qdrant-lite-named-vectors"

    @property
    def status(self):
        return self._status

    def _asset_map(self) -> dict[str, str]:
        if self._legacy_to_native is not None:
            return self._legacy_to_native
        mapping = {}
        for asset in self.store.list_assets(limit=100_000):
            metadata = asset.get("metadata_json") or {}
            if isinstance(metadata, dict) and metadata.get("legacy_asset_id"):
                mapping[str(metadata["legacy_asset_id"])] = str(asset["id"])
        self._legacy_to_native = mapping
        return mapping

    def _embed(self, text: str) -> list[float]:
        if self.embedding_router is None:
            return []
        if hasattr(self.index, "embed"):
            return self.index.embed(self.embedding_router, self.embedding_slot, text)
        if self.embedding_slot == "visual":
            return self.embedding_router.embed_visual(text)
        return self.embedding_router.embed_text(text)

    def retrieve(self, query, filters, limit: int) -> list[CandidateHit]:
        text = query.whole_query or " ".join(facet.surface_text for facet in query.facets)
        vector = self._embed(text)
        if not vector:
            self._status = "embedder_unavailable"
            return []
        try:
            points = self.index.query(
                route=self.vector_name,
                vector=vector,
                scope_id=self.legacy_scope_id,
                media_types=tuple(filters.media_types or ()),
                # Multiple video segments may map to one native video.  Ask
                # for extra points before the asset-level de-duplication.
                limit=max(int(limit) * 4, int(limit), 24),
            )
        except Exception as error:
            self.index.last_error = f"{type(error).__name__}: {error}"
            self._status = "error"
            return []
        mapping = self._asset_map()
        hits = []
        seen = set()
        allowed_scopes = set(filters.scope_ids or ())
        for point in points:
            payload = dict(point.payload or {})
            native_id = mapping.get(str(payload.get("asset_id") or ""))
            if not native_id or native_id in seen:
                continue
            asset = self.store.get_asset(native_id) or {}
            if not filters.all_authorized and allowed_scopes \
                    and (asset.get("scope_id") or "home-default") not in allowed_scopes:
                continue
            if filters.media_types and asset.get("media_type") not in filters.media_types:
                continue
            seen.add(native_id)
            unit = {
                "unit_type": payload.get("unit_type"),
                "segment_index": payload.get("segment_index"),
                "start_sec": payload.get("start_sec"),
                "end_sec": payload.get("end_sec"),
            }
            hits.append(CandidateHit(
                asset_id=native_id,
                retriever=self.name,
                raw_score=float(point.score),
                score_kind="cosine_similarity",
                higher_is_better=True,
                rank=len(hits) + 1,
                source_id=str(point.id),
                source_revision=None,
                metadata={"route": self.vector_name, "legacy_unit": unit},
            ))
            if len(hits) >= max(1, int(limit)):
                break
        self._status = "ready" if hits else "no_candidates"
        return hits


def build_lite_named_vector_retrievers(store, *, embedding_router=None):
    path = os.getenv("SENTRIX_LITE_NAMED_VECTOR_PATH", "").strip()
    if not path:
        return []
    collection = os.getenv(
        "SENTRIX_LITE_NAMED_VECTOR_COLLECTION", "sentrix_memory_units_v1",
    ).strip()
    legacy_scope_id = os.getenv("SENTRIX_LITE_NAMED_VECTOR_SCOPE_ID", "").strip()
    if not legacy_scope_id:
        return []
    index = _shared_index(path, collection)
    return [
        LiteNamedVectorRetriever(
            store, embedding_router,
            name=name, vector_name=vector_name, embedding_slot=slot,
            index=index, legacy_scope_id=legacy_scope_id,
        )
        for name, vector_name, slot in ROUTES
    ]
