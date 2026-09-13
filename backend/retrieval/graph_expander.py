"""Graph-backed late expansion for the normal evidence retrieval path."""

from __future__ import annotations

import os
import threading
from dataclasses import replace

from .base import CandidateHit, HardFilterContext, RetrievalQuery
from .graph_policy import graph_retrieval_policy


# Graph expansion is invoked once per QA turn.  Constructing a service for
# every turn used to reload the 30k-node SQLite graph and then close it again,
# adding ~1-2s before every query.  Keep one process-local service so its
# read-only in-memory indexes can be reused across turns.  The service itself
# serializes access and invalidates the builder when the derived graph changes.
_GRAPH_SERVICE_LOCK = threading.Lock()
_GRAPH_SERVICE = None
_GRAPH_SERVICE_KEY = None


def _shared_graph_service():
    global _GRAPH_SERVICE, _GRAPH_SERVICE_KEY
    from ..graph_memory import GraphMemoryService

    root = os.getenv("SENTRIX_DATA_DIR", "")
    db_path = os.getenv("SENTRIX_DB_PATH", "")
    graph_path = os.getenv("SENTRIX_GRAPH_DB_PATH", "")
    key = (root, db_path, graph_path)
    with _GRAPH_SERVICE_LOCK:
        if _GRAPH_SERVICE is None or _GRAPH_SERVICE_KEY != key:
            _GRAPH_SERVICE = GraphMemoryService()
            _GRAPH_SERVICE_KEY = key
        return _GRAPH_SERVICE


class GraphExpander:
    """Use baseline asset hits as seeds for MAGMA-style graph traversal."""

    name = "graph"
    kind = "expander"

    def __init__(self, store, *, config=None):
        self.store = store
        self.config = config

    def retrieve(self, query: RetrievalQuery, filters: HardFilterContext, limit: int):
        return []

    @staticmethod
    def _route_question(query: RetrievalQuery) -> str:
        """Build route text without changing the primary retriever query.

        ``RetrievalQuery.from_spec`` intentionally keeps its legacy
        constraint-only contract for ANN/FTS.  Semantic facets still contain
        the user's sequence wording when no hard constraint was emitted, so
        the graph policy must see them explicitly.
        """
        parts = [str(query.whole_query or "").strip()]
        parts.extend(str(f.surface_text or "").strip()
                     for f in (query.facets or []) if f.surface_text)
        return " ".join(dict.fromkeys(part for part in parts if part))

    @staticmethod
    def route(query: RetrievalQuery, filters: HardFilterContext) -> dict:
        """Expose the same deterministic route used by ``expand``."""
        question = GraphExpander._route_question(query)
        if not question:
            question = ""
        return graph_retrieval_policy(question, filters=filters, query=query)

    def expand(self, seed_asset_ids: list[str], filters: HardFilterContext,
               limit: int, query: RetrievalQuery | None = None) -> list[CandidateHit]:
        if query is None:
            return []
        question = self._route_question(query)
        if not question:
            question = ""
        if not question:
            return []
        route = graph_retrieval_policy(question, filters=filters, query=query)
        if not route["enabled"]:
            return []
        scope_id = filters.scope_ids[0] if len(filters.scope_ids) == 1 else None
        # The normal path is deliberately cheap: traverse only from ordinary
        # retrieval anchors.  For an explicit multi-hop/causal question this
        # can fail precisely when graph recall is needed most -- no ordinary
        # hit is a usable anchor.  In that narrowly-defined case we use the
        # graph query engine as an independent, bounded recall source instead
        # of silently returning zero graph candidates.
        graph_intent = str(route.get("intent") or "ordinary")
        seed_cap = 48 if graph_intent in {
            "temporal", "relationship", "causal", "multi_hop", "cross_media",
        } else 24
        seed_ids = list(dict.fromkeys(seed_asset_ids))[:max(8, min(len(seed_asset_ids), seed_cap))]
        # For an ordinary query in the full ``on`` experiment the independent
        # global graph channel is enough; doing an anchored traversal first
        # would double graph work without adding a second semantic signal.
        use_anchors = bool(seed_ids) and not (
            graph_intent == "ordinary" and str(route.get("mode") or "") == "on")
        try:
            service = _shared_graph_service()
            result = service.expand_from_assets(
                question, seed_ids,
                top_k=max(10, min(int(limit or 30), 60)), scope_id=scope_id,
            ) if use_anchors else {"ok": True, "nodes": [], "paths": []}
        except Exception:
            return []
        if not result.get("ok"):
            result = {"nodes": [], "paths": []}
        seeds = set(seed_ids)
        hits = []
        for rank, item in enumerate(result.get("nodes") or [], 1):
            asset_id = str(item.get("asset_id") or "")
            if not asset_id:
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
                          "graph_policy": route,
                          # Seed nodes are intentionally retained.  They are
                          # the graph's evidence that an ordinary candidate
                          # lies on a relevant path, so dropping them makes
                          # graph traversal unable to re-rank its own seeds.
                          "graph_seed": asset_id in seeds,
                          "path_count": len(result.get("paths") or [])},
            ))
            if len(hits) >= limit:
                break
        # Anchored expansion alone cannot recover a media item that ordinary
        # ANN/lexical retrieval missed.  For graph-semantic intents, merge one
        # bounded global graph query into the same graph channel.  This is a
        # recall source, not a second answer-generation pass; the shared
        # service keeps the graph index hot so the extra query is cheap.
        # In ``on`` mode the graph is an independent recall channel for every
        # question, including ordinary visual/place queries.  Auto mode keeps
        # ordinary queries seed-gated, so enabling the experiment cannot add a
        # graph scan to the production default accidentally.
        global_intents = {"temporal", "relationship", "causal", "multi_hop", "cross_media"}
        if str(route.get("mode") or "") == "on":
            global_intents.add("ordinary")
        if graph_intent not in global_intents:
            return hits
        try:
            # Keep enough graph-only candidates for the fusion head to choose
            # a missed event.  The previous hard cap of 12 meant that the
            # global fallback was effectively just another tiny reranker and
            # could not recover a GT event outside the ordinary ANN head.
            # Keep a wider independent graph head.  A 24-item head was often
            # filled by generic neighbours of the same event, so graph could
            # be active without recovering a missed media asset.  The bound
            # stays finite; the shared in-memory service makes this a bounded
            # index lookup rather than another model call.
            graph_top_k = max(
                8, min(int(os.getenv("SENTRIX_GRAPH_GLOBAL_TOP_K", "48")), 48))
            global_result = service.search(
                question, top_k=graph_top_k, scope_id=scope_id)
        except Exception:
            return hits
        if not global_result.get("ok"):
            return hits
        existing = {hit.asset_id for hit in hits}
        base_rank = len(hits)
        for rank, item in enumerate(global_result.get("nodes") or [], 1):
            asset_id = str(item.get("source_asset_id") or item.get("asset_id") or "")
            if not asset_id or asset_id in existing:
                continue
            hits.append(CandidateHit(
                asset_id=asset_id, retriever=self.name,
                raw_score=float(item.get("score") or 0.0),
                score_kind="graph_global_path", higher_is_better=True, rank=rank,
                source_id=str(item.get("id") or asset_id),
                metadata={"graph_policy": route, "graph_seed": asset_id in seeds,
                          "graph_source": "global_fallback",
                          "path_count": len(global_result.get("graph_paths") or [])},
            ))
            existing.add(asset_id)
            if len(hits) >= limit:
                break
        # Keep graph-channel rank monotonic after merging anchored and global
        # candidates.  RRF uses rank, so a global candidate with a stronger
        # graph score must not be penalized solely for being appended later.
        hits.sort(key=lambda hit: (-float(hit.raw_score or 0.0), bool(
            (hit.metadata or {}).get("graph_source") == "global_fallback"), hit.asset_id))
        # CandidateHit is a frozen dataclass.  Rebuild the ranked objects
        # instead of mutating ``rank`` in place; the old assignment raised
        # ``cannot assign to field 'rank'`` and made every graph query fail.
        hits = [replace(hit, rank=index) for index, hit in enumerate(hits, 1)]
        return hits[:limit]


def graph_retriever_enabled() -> bool:
    return os.getenv("SENTRIX_GRAPH_RETRIEVER_ENABLED", "1").strip().lower() in {
        "1", "true", "yes", "on"
    }
