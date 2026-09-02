"""Graph-backed late expansion for the normal evidence retrieval path."""

from __future__ import annotations

import os
import re

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
        # Graph traversal is most valuable for sequence/multi-hop questions.
        # Running it for every ordinary caption/location lookup used to add
        # 20–45 seconds and inject weak neighbours into an already good
        # lexical/metadata ranking.  Keep an explicit opt-in for debugging,
        # while routing temporal/causal/relationship questions through the
        # graph by default.
        graph_enabled = os.getenv("SENTRIX_GRAPH_EXPAND_ALL", "0").strip().lower() in {
            "1", "true", "yes", "on"
        }
        graph_query = bool(re.search(
            r"before|after|then|next|later|sequence|timeline|chronolog|cause|because|why|"
            r"multi.?hop|relationship|interact|state change|"
            r"之前|之后|然后|接着|先后|顺序|时间线|过程|步骤|因果|因为|为何|为什么|"
            r"关系|互动|状态变化|导致|结果|跨视频|多个视频|哪几个视频",
            question, re.I))
        if not graph_enabled and not graph_query:
            return []
        scope_id = filters.scope_ids[0] if len(filters.scope_ids) == 1 else None
        try:
            from ..graph_memory import GraphMemoryService
            result = GraphMemoryService().expand_from_assets(
                question,
                list(dict.fromkeys(seed_asset_ids))[:max(8, min(len(seed_asset_ids), 24))],
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
