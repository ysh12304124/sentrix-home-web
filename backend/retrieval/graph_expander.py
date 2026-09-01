"""Graph-backed late expansion for the normal evidence retrieval path."""

from __future__ import annotations

import os

from .base import CandidateHit, HardFilterContext, RetrievalQuery


class GraphExpander:
    """Use baseline asset hits as seeds for MAGMA-style graph traversal."""

    name = "graph"
    kind = "expander"

    def __init__(self, store, *, config=None):
        self.store = store
        self.config = config

    def retrieve(self, query: RetrievalQuery, filters: HardFilterContext, limit: int):
        return []

    def expand(self, seed_asset_ids: list[str], filters: HardFilterContext,
               limit: int, query: RetrievalQuery | None = None) -> list[CandidateHit]:
        if not seed_asset_ids or query is None:
            return []
        question = str(query.whole_query or "").strip()
        if not question:
            question = " ".join(str(f.surface_text or "").strip()
                                 for f in query.facets if f.surface_text).strip()
        if not question:
            return []
        scope_id = filters.scope_ids[0] if len(filters.scope_ids) == 1 else None
        try:
            from ..graph_memory import GraphMemoryService
            result = GraphMemoryService().expand_from_assets(
                question,
                list(dict.fromkeys(seed_asset_ids))[:max(8, min(len(seed_asset_ids), 40))],
                top_k=max(10, min(int(limit or 30), 60)),
                scope_id=scope_id,
            )
        except Exception:
            return []
        if not result.get("ok"):
            return []
        seeds = set(seed_asset_ids)
        hits = []
        for rank, item in enumerate(result.get("nodes") or [], 1):
            asset_id = str(item.get("asset_id") or "")
            if not asset_id or asset_id in seeds:
                continue
            score = float(item.get("score") or 0.0)
            hits.append(CandidateHit(
                asset_id=asset_id,
                retriever=self.name,
                raw_score=score,
                score_kind="graph_path",
                higher_is_better=True,
                rank=rank,
                source_id=str(item.get("node_id") or asset_id),
                metadata={"event_id": item.get("event_id"),
                          "query_type": result.get("query_type"),
                          "path_count": len(result.get("paths") or [])},
            ))
            if len(hits) >= limit:
                break
        return hits


def graph_retriever_enabled() -> bool:
    return os.getenv("SENTRIX_GRAPH_RETRIEVER_ENABLED", "1").strip().lower() in {
        "1", "true", "yes", "on"
    }
