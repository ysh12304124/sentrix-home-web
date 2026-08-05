"""TextAnnRetriever — text-to-observation/event ANN recall.

Query text -> TextQueryEmbedder -> hnswlib semantic/episodic index -> the
source Observation's owning Asset.

The semantic/episodic indices store ``source_type=observation`` and their
metadata carries ``asset_id`` (pipeline.upsert_vector), so this retriever maps
an observation hit back to its Asset for candidate merge.  P0-4 manifest
validation applies just like the visual channel.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..retrieval_ann import create_index
from .base import CandidateHit, HardFilterContext, RetrievalQuery

_DEFAULT_ANN_DIR = Path(__file__).resolve().parents[2] / "data" / "ann"


@dataclass
class TextAnnRetriever:
    name: str = "text_ann"
    kind: str = "primary"

    def __init__(self, store, *, embedding_router=None, ann_dir=None, backend="hnswlib", spaces=("semantic", "episodic")):
        self.store = store
        self.embedding_router = embedding_router
        self.ann_dir = Path(ann_dir) if ann_dir else _DEFAULT_ANN_DIR
        self.backend = backend
        self.spaces = spaces
        self._indices = {}
        self._load_failed = None
        self._status = "uninitialized"

    @property
    def status(self):
        return self._status

    def _load_index(self, space):
        if space in self._indices:
            return self._indices[space]
        path = self.ann_dir / space
        if not (self.ann_dir / f"{space}.hnsw").is_file():
            self._indices[space] = None
            return None
        try:
            index = create_index(self.backend, dim=None)
            index.load(str(path))
            expected_model = getattr(self.embedding_router.text, "model_id", None) if self.embedding_router else None
            if not index.validate(expected_model_id=expected_model, expected_dim=None):
                self._status = "incompatible"
                self._load_failed = index.incompatible_reason
                self._indices[space] = None
                return None
            self._indices[space] = index
            return index
        except Exception as error:
            self._status = "unavailable"
            self._load_failed = str(error)
            self._indices[space] = None
            return None

    def retrieve(self, query: RetrievalQuery, filters: HardFilterContext, limit: int) -> list[CandidateHit]:
        if self.embedding_router is None or not self.embedding_router.text_available:
            self._status = "embedder_unavailable"
            return []
        vector = self.embedding_router.embed_text(query.whole_query or " ".join(
            f.surface_text for f in query.facets))
        if not vector:
            return []
        scope = filters.scope_ids[0] if filters.scope_ids and not filters.all_authorized else None
        candidates: dict[str, tuple[float, int, dict]] = {}
        for space in self.spaces:
            index = self._load_index(space)
            if index is None:
                continue
            if index.manifest().get("dimension") and len(vector) != index.manifest()["dimension"]:
                self._status = "incompatible"
                self._load_failed = f"dimension_mismatch:{len(vector)}vs{index.manifest()['dimension']}"
                continue
            try:
                rows = index.search(vector, k=limit, scope_id=scope)
            except Exception as error:
                self._load_failed = str(error)
                self._status = "search_error"
                continue
            for rank, (source_id, similarity, metadata) in enumerate(rows):
                asset_id = metadata.get("asset_id") or source_id
                current = candidates.get(asset_id)
                if current is None or similarity > current[1]:
                    candidates[asset_id] = (similarity, rank + 1, metadata)
        if not candidates:
            if self._status in {"uninitialized", "unavailable", "incompatible"}:
                pass
            elif self._status != "ready":
                self._status = "no_candidates"
        else:
            self._status = "ready"
        hits = []
        for rank, (asset_id, (similarity, _, metadata)) in enumerate(sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)[:limit]):
            hits.append(CandidateHit(
                asset_id=asset_id,
                retriever=self.name,
                raw_score=float(similarity),
                score_kind="cosine_similarity",
                higher_is_better=True,
                rank=rank + 1,
                source_id=metadata.get("source_id") or asset_id,
                source_revision=metadata.get("revision"),
                metadata={"scope_id": metadata.get("scope_id"), "spaces": list(self.spaces),
                          "model_id": getattr(self.embedding_router.text, "model_id", None)},
            ))
        return hits
