"""Graph-backed late expansion for the normal evidence retrieval path."""

from __future__ import annotations

import os
import re
import threading
from dataclasses import replace

from .base import CandidateHit, HardFilterContext, RetrievalQuery, effective_media_type
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

    @staticmethod
    def _prefers_still_images(query: RetrievalQuery, filters: HardFilterContext) -> bool:
        """Return true only for an explicit photo/image request.

        Video frames are stored as image assets, so media_type=image alone is
        not enough to distinguish them from original photos.  Apply this
        narrow preference only to the graph channel; ambiguous event queries
        continue to search both photo and video evidence.
        """
        if filters.media_types and "image" in filters.media_types and "video" not in filters.media_types:
            return True
        if "video" in filters.negated_media:
            return True
        text = GraphExpander._route_question(query).lower()
        wants_image = bool(re.search(
            r"照片|相片|图片|圖片|图像|影像|合影|留影|拍照|image|photo|picture|photograph",
            text,
        ))
        wants_video = bool(re.search(
            r"视频|影片|录像|录影|片段|录屏|video|clip|footage",
            text,
        ))
        return wants_image and not wants_video

    def _is_video_evidence(self, asset_id: str) -> bool:
        try:
            asset = self.store.get_asset(asset_id) or {}
        except Exception:
            return False
        metadata = asset.get("metadata_json")
        if not isinstance(metadata, dict):
            metadata = {}
        derived = str(asset.get("derived_kind") or metadata.get("derived_kind") or "").lower()
        return (asset.get("media_type") == "video"
                or derived in {"video_keyframe", "video_keyframe_webp"})

    @staticmethod
    def _diversify_by_event(hits: list[CandidateHit], per_event_cap: int) -> list[CandidateHit]:
        """Bound repeated frames from one event so graph head covers more events."""
        cap = max(1, int(per_event_cap))
        counts: dict[str, int] = {}
        output: list[CandidateHit] = []
        seen_assets: set[str] = set()
        for hit in hits:
            if hit.asset_id in seen_assets:
                continue
            seen_assets.add(hit.asset_id)
            metadata = hit.metadata or {}
            event_key = str(
                metadata.get("event_id") or metadata.get("video_uid") or ""
            ).strip().casefold()
            if event_key:
                # A resolved event is already a narrow, query-specific unit;
                # retaining its full small member set is more useful than
                # dropping one of the few candidate photos at the generic
                # neighbour-diversification cap.
                event_cap = max(cap, 8) if metadata.get("graph_source") == "event_context" else cap
                if counts.get(event_key, 0) >= event_cap:
                    continue
                counts[event_key] = counts.get(event_key, 0) + 1
            output.append(hit)
        return output

    def _event_context_candidates(self, query: RetrievalQuery,
                                  filters: HardFilterContext,
                                  route: dict, limit: int) -> list[CandidateHit]:
        """Resolve a clearly named event to its existing member assets.

        The graph projection primarily indexes video frames; image event
        membership is already maintained by Sentrix's event store. Reuse that
        relation as a graph retrieval source, but only for an event route and
        only as bounded candidate additions (never as a replacement result).
        """
        if str(route.get("intent") or "") != "event":
            return []
        question = self._route_question(query)
        scope_id = filters.scope_ids[0] if len(filters.scope_ids) == 1 else None
        if not question or not scope_id:
            return []
        try:
            # Imported lazily to avoid a static dependency from the retrieval
            # kernel back into the Agent tool registry. The event resolver is
            # a deterministic retrieval helper and is already loaded when a
            # search request reaches this point.
            from ..agent_runtime.tools import _event_keyword_anchor
            event = _event_keyword_anchor(question, self.store, scope_id)
        except Exception:
            return []
        if not event:
            return []
        event_id = str(event.get("event_id") or "")
        hits = []
        for rank, asset_id in enumerate(event.get("asset_ids") or [], 1):
            asset_id = str(asset_id or "")
            if not asset_id:
                continue
            try:
                asset = self.store.get_asset(asset_id) or {}
            except Exception:
                asset = {}
            media_type = str(asset.get("media_type") or "")
            if filters.media_types and effective_media_type(asset) not in filters.media_types:
                continue
            if self._prefers_still_images(query, filters) and self._is_video_evidence(asset_id):
                continue
            hits.append(CandidateHit(
                asset_id=asset_id,
                retriever=self.name,
                # This source has a unique, scope-local event match. Keep its
                # members ahead of generic graph neighbours so they can use
                # the graph head's reserved recall slots.
                raw_score=max(0.75, 0.94 - (rank - 1) * 0.01),
                score_kind="event_member_context",
                higher_is_better=True,
                rank=rank,
                source_id=event_id or asset_id,
                metadata={
                    "graph_policy": route,
                    "graph_source": "event_context",
                    "event_id": event_id,
                    "event_title": event.get("event_title"),
                },
            ))
            if len(hits) >= max(1, min(int(limit or 8), 50)):
                break
        return hits

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
            "temporal", "relationship", "causal", "multi_hop", "cross_media", "event",
        } else 24
        seed_ids = list(dict.fromkeys(seed_asset_ids))[:max(8, min(len(seed_asset_ids), seed_cap))]
        # Even in the explicit ``on`` experiment, ordinary natural-language
        # questions need baseline assets as graph anchors.  The graph index's
        # global lexical query cannot reliably map a free-form Chinese phrase
        # to an entity/event node, so skipping anchors made every ordinary
        # graph expansion return zero candidates and produced no measurable
        # graph effect.
        use_anchors = bool(seed_ids)
        # Keep a wider graph head than the final evidence head.  If we stop
        # at ``limit`` while emitting anchored seed neighbours, the later
        # global event-summary query never gets a chance to contribute a
        # graph-only candidate, so graph rerank changes order but not GT set.
        graph_limit = max(int(limit or 30) * 2, 60)
        try:
            service = _shared_graph_service()
            result = service.expand_from_assets(
                question, seed_ids,
                top_k=max(10, min(graph_limit, 100)), scope_id=scope_id,
            ) if use_anchors else {"ok": True, "nodes": [], "paths": []}
        except Exception:
            result = {"ok": False, "nodes": [], "paths": []}
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
                          "video_uid": item.get("video_uid"),
                          "graph_source": item.get("graph_source"),
                          "event_summary_node_id": item.get("event_summary_node_id"),
                          "query_type": result.get("query_type"),
                          "graph_policy": route,
                          # Seed nodes are intentionally retained.  They are
                          # the graph's evidence that an ordinary candidate
                          # lies on a relevant path, so dropping them makes
                          # graph traversal unable to re-rank its own seeds.
                          "graph_seed": asset_id in seeds,
                          "path_count": len(result.get("paths") or [])},
            ))
            if len(hits) >= graph_limit:
                break
        existing = {hit.asset_id for hit in hits}
        for hit in self._event_context_candidates(query, filters, route, graph_limit):
            if hit.asset_id in existing:
                continue
            hits.append(hit)
            existing.add(hit.asset_id)
            if len(hits) >= graph_limit:
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
        global_intents = {"temporal", "relationship", "causal", "multi_hop", "cross_media", "event"}
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
        for rank, item in enumerate(global_result.get("nodes") or [], 1):
            asset_id = str(item.get("source_asset_id") or item.get("asset_id") or "")
            if not asset_id or asset_id in existing:
                continue
            attributes = item.get("attributes")
            attributes = attributes if isinstance(attributes, dict) else {}
            hits.append(CandidateHit(
                asset_id=asset_id, retriever=self.name,
                raw_score=float(item.get("score") or 0.0),
                score_kind="graph_global_path", higher_is_better=True, rank=rank,
                source_id=str(item.get("id") or asset_id),
                metadata={"graph_policy": route, "graph_seed": asset_id in seeds,
                          "graph_source": "global_fallback",
                          "event_id": item.get("event_id") or attributes.get("event_id"),
                          "video_uid": item.get("video_uid") or attributes.get("video_uid"),
                          "path_count": len(global_result.get("graph_paths") or [])},
            ))
            existing.add(asset_id)
            if len(hits) >= graph_limit:
                break
        # A graph EVENT for a video is persisted as a keyframe image because
        # the VLM/CLIP stack consumes frames.  For a video route, also expose
        # the parent source-video asset in the same event head; otherwise the
        # Agent can inspect a frame but the benchmark/UI can never deliver or
        # score the corresponding video.  Keep this projection route-scoped so
        # ordinary photo retrieval is not polluted by video parents.
        video_route = graph_intent == "cross_media" or "video" in (filters.media_types or ())
        if video_route:
            for hit in list(hits):
                try:
                    asset = self.store.get_asset(hit.asset_id) or {}
                except Exception:
                    asset = {}
                metadata = asset.get("metadata_json") if isinstance(asset.get("metadata_json"), dict) else {}
                derived = str(asset.get("derived_kind") or metadata.get("derived_kind") or "").lower()
                if derived not in {"video_keyframe", "video_keyframe_webp"}:
                    continue
                parent_id = str(asset.get("parent_asset_id") or metadata.get("parent_asset_id") or "")
                if not parent_id or parent_id in existing:
                    continue
                hits.append(CandidateHit(
                    asset_id=parent_id, retriever=self.name,
                    raw_score=float(hit.raw_score or 0.0) * 0.985,
                    score_kind="graph_video_parent", higher_is_better=True,
                    rank=len(hits) + 1, source_id=parent_id,
                    metadata={"graph_policy": route, "graph_source": "video_parent",
                              "event_id": (hit.metadata or {}).get("event_id"),
                              "video_uid": (hit.metadata or {}).get("video_uid"),
                              "parent_keyframe_asset_id": hit.asset_id},
                ))
                existing.add(parent_id)
                if len(hits) >= graph_limit:
                    break
        # Explicit photo requests should not be diluted by video keyframes,
        # which are indexed as image assets.  Do not infer image-only from the
        # absence of video wording: most event QA is intentionally cross-modal.
        if self._prefers_still_images(query, filters):
            hits = [hit for hit in hits if not self._is_video_evidence(hit.asset_id)]

        # Keep graph-channel rank monotonic after merging anchored and global
        # candidates.  RRF uses rank, so a global candidate with a stronger
        # graph score must not be penalized solely for being appended later.
        hits.sort(key=lambda hit: (-float(hit.raw_score or 0.0), bool(
            (hit.metadata or {}).get("graph_source") == "global_fallback"), hit.asset_id))
        # CandidateHit is a frozen dataclass.  Rebuild the ranked objects
        # instead of mutating ``rank`` in place; the old assignment raised
        # ``cannot assign to field 'rank'`` and made every graph query fail.
        hits = [replace(hit, rank=index) for index, hit in enumerate(hits, 1)]
        try:
            per_event_cap = int(os.getenv("SENTRIX_GRAPH_PER_EVENT_CAP", "4"))
        except (TypeError, ValueError):
            per_event_cap = 4
        hits = self._diversify_by_event(hits, per_event_cap)
        hits = [replace(hit, rank=index) for index, hit in enumerate(hits, 1)]
        # ``limit`` is the final evidence-head size requested by the caller.
        # The graph channel intentionally builds a wider head so fusion can
        # introduce graph-only candidates that ordinary retrieval missed.
        # Truncating back to ``limit`` here discards precisely those
        # candidates before the graph quota/fusion stage can see them.
        return hits[:graph_limit]


def graph_retriever_enabled() -> bool:
    return os.getenv("SENTRIX_GRAPH_RETRIEVER_ENABLED", "1").strip().lower() in {
        "1", "true", "yes", "on"
    }
