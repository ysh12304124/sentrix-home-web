"""VisualAnnRetriever — text-to-image ANN recall.

Query text -> VisualQueryEmbedder -> hnswlib visual index -> Asset candidates.

The visual index's source_type is ``asset`` so ``source_id`` is already an
Asset ID.  Scope filtering stays code-side (P0-5): the index returns more
candidates when a scope filter is active (oversample) and the Kernel applies
the authoritative scope/time/media pass afterwards.

P0-4: the index manifest must match the query embedder (model_id, dimension);
an incompatible index is recorded as ``index_incompatible`` and the channel is
skipped — never searched silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..retrieval_ann import create_index
from .base import CandidateHit, HardFilterContext, RetrievalQuery

_DEFAULT_ANN_DIR = Path(__file__).resolve().parents[2] / "data" / "ann"


@dataclass
class VisualAnnRetriever:
    name: str = "visual_ann"
    kind: str = "primary"
    space: str = "visual"

    def __init__(self, store, *, embedding_router=None, ann_dir=None, backend="hnswlib"):
        self.store = store
        self.embedding_router = embedding_router
        self.ann_dir = Path(ann_dir) if ann_dir else _DEFAULT_ANN_DIR
        self.backend = backend
        self._index = None
        self._load_failed = None
        self._status = "uninitialized"

    @property
    def status(self):
        return self._status

    def _load_index(self):
        if self._index is not None:
            return self._index
        path = self.ann_dir / self.space
        if not (self.ann_dir / f"{self.space}.hnsw").is_file():
            self._status = "unavailable"
            self._load_failed = "index_missing"
            return None
        try:
            index = create_index(self.backend, dim=None)
            index.load(str(path))
            expected_model = getattr(self.embedding_router.visual, "model_id", None) if self.embedding_router else None
            if not index.validate(expected_model_id=expected_model, expected_dim=None):
                self._status = "incompatible"
                self._load_failed = index.incompatible_reason
                self._index = index  # cached so we don't reload every request
                return None
            self._status = "ready"
            self._index = index
            return index
        except Exception as error:
            self._status = "unavailable"
            self._load_failed = str(error)
            return None

    def retrieve(self, query: RetrievalQuery, filters: HardFilterContext, limit: int) -> list[CandidateHit]:
        if self.embedding_router is None or not self.embedding_router.visual_available:
            self._status = "embedder_unavailable"
            return []
        index = self._load_index()
        if index is None:
            return []
        vector = self.embedding_router.embed_visual(query.whole_query or " ".join(
            f.surface_text for f in query.facets))
        if not vector:
            return []
        if index.manifest().get("dimension") and len(vector) != index.manifest()["dimension"]:
            # Dimension is the authoritative cross-check (P0-4): a query vector
            # whose length differs from the index dim would be meaningless.
            self._status = "incompatible"
            self._load_failed = f"dimension_mismatch:{len(vector)}vs{index.manifest()['dimension']}"
            return []
        scope = filters.scope_ids[0] if filters.scope_ids and not filters.all_authorized else None
        try:
            rows = index.search(vector, k=limit, scope_id=scope)
        except Exception as error:
            self._status = "search_error"
            self._load_failed = str(error)
            return []
        hits = []
        for rank, (asset_id, similarity, metadata) in enumerate(rows[:limit]):
            hits.append(CandidateHit(
                asset_id=asset_id,
                retriever=self.name,
                raw_score=float(similarity),
                score_kind="cosine_similarity",
                higher_is_better=True,
                rank=rank + 1,
                source_id=asset_id,
                source_revision=metadata.get("revision"),
                metadata={"scope_id": metadata.get("scope_id"), "space": self.space,
                          "model_id": getattr(self.embedding_router.visual, "model_id", None)},
            ))
        return hits
