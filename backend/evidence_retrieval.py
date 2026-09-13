"""Correctness-first Asset-level Evidence Retrieval Kernel."""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
import time

from .geocoding import place_text_matches
from .query_contracts import HARD, SEMANTIC, QueryFacet, QuerySpec, parse_time_expression


def build_verifier_evidence_bundle(packet, claim_id):
    """Expose controlled canonical excerpts to the verifier only.

    Derived observation_fields are intentionally excluded from the proof list;
    they are a writing aid, not a second source of truth.
    """
    canonical = []
    for item in packet.assets:
        canonical.append({
            "evidence_id": item["asset_id"], "type": "asset", "source_text": item.get("file_name") or item["asset_id"],
            "subject_ids": [], "time": item.get("captured_at"), "scope_id": packet.scope_id, "is_canonical": True,
        })
        for observation_id in item.get("observation_ids", []):
            canonical.append({
                "evidence_id": observation_id, "type": "observation", "source_text": "受控 Observation 摘录",
                "subject_ids": [], "time": item.get("captured_at"), "scope_id": packet.scope_id, "is_canonical": True,
            })
    return {"claim_id": claim_id, "canonical_evidence": canonical, "derived_context": []}


@dataclass
class EvidencePacket:
    query_id: str
    scope_id: str
    answer_target: str
    assets: list[dict] = field(default_factory=list)
    exact_results: list[dict] = field(default_factory=list)
    strong_results: list[dict] = field(default_factory=list)
    approximate_results: list[dict] = field(default_factory=list)
    gaps: list[dict] = field(default_factory=list)
    excluded_count: int = 0
    # Phase R R2: per-channel recall trace so API/benchmark can prove each
    # retriever was invoked (or give an explicit unavailable reason).
    channel_trace: dict = field(default_factory=dict)
    retrieval_timing: dict = field(default_factory=dict)

    def as_dict(self):
        return {
            "query_id": self.query_id,
            "scope_id": self.scope_id,
            "answer_target": self.answer_target,
            "assets": self.assets,
            "exact_results": self.exact_results,
            "strong_results": self.strong_results,
            "approximate_results": self.approximate_results,
            "gaps": self.gaps,
            "excluded_count": self.excluded_count,
            "channel_trace": self.channel_trace,
            "retrieval_timing": self.retrieval_timing,
            "result_summary": {
                "exact_count": len(self.exact_results),
                "strong_count": len(self.strong_results),
                "approximate_count": len(self.approximate_results),
                "hard_constraint_violations": self.excluded_count,
            },
        }


def _parse_datetime(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except (TypeError, ValueError):
        return None


def _contains(haystack, needle):
    """Full-substring containment only (Phase R P0-6).

    Tokenized all-match produced the album1-01 false-positive storm (single
    colour/material characters matching unrelated captions).  A condition is
    only "supported" by a field when the full normalized value is present.
    Candidate recall has its own channel (LexicalRetriever); this function is
    the evidence check, not a recall path.
    """
    haystack, needle = str(haystack or "").lower(), str(needle or "").lower()
    if not needle:
        return False
    return needle in haystack


def _in_annual_window(captured, window):
    start_month, start_day, end_month, end_day = window
    current = (captured.month, captured.day)
    start = (start_month, start_day)
    end = (end_month, end_day)
    return start <= current < end if start < end else current >= start or current < end


class EvidenceRetrievalKernel:
    def __init__(self, store, *, retrievers=None, embedding_router=None, config=None, trace=None):
        self.store = store
        self.retrievers = list(retrievers or [])
        self.embedding_router = embedding_router
        self.config = config
        self._trace_sink = trace
        # A structured query can intentionally pass hundreds of candidates to
        # the condition verifier.  Keep request-local bulk projections so that
        # verification does not issue get_asset/list_observations/event SQL for
        # every candidate (the former N+1 path cost 25-35s on a 350-asset
        # benchmark scope).
        self._asset_cache = None
        self._observations_by_asset_cache = None
        self._event_text_by_observation_cache = None

    def retrieve(self, spec: QuerySpec):
        if self.retrievers and self._multi_retriever_enabled():
            return self._retrieve_multi(spec)
        return self._retrieve_single(spec)

    def _multi_retriever_enabled(self) -> bool:
        if self.config is not None:
            return self.config.multi_retriever
        import os
        return os.getenv("SENTRIX_EVIDENCE_MULTI_RETRIEVER_V1", "0").lower() in {"1", "true", "yes", "on"}

    def _retrieve_single(self, spec: QuerySpec):
        all_authorized = spec.scope_mode == "all_authorized"
        scope_id = spec.scope_id or (spec.scope_ids[0] if spec.scope_ids else "all_authorized")
        query_scope = None if all_authorized else scope_id
        # ``MemoryStore.list_assets`` defaults to 200 rows.  Using that default
        # here silently removed older media from the fallback/single-retriever
        # path, which is especially damaging for benchmark albums where the
        # answer image is near the end of the import order.
        try:
            asset_rows = self.store.list_assets(scope_id=query_scope, limit=100_000)
        except TypeError:
            # Keep lightweight test/integration stores compatible with the
            # production pagination argument while preserving the full-scope
            # behavior on MemoryStore.
            asset_rows = self.store.list_assets(scope_id=query_scope)
        assets = {item["id"]: item for item in asset_rows}
        try:
            observation_rows = self.store.list_observations(
                scope_id=query_scope, limit=100_000)
        except TypeError:
            observation_rows = self.store.list_observations(scope_id=query_scope)
        observations = [item for item in observation_rows if item.get("asset_id") in assets]
        by_asset = {}
        for observation in observations:
            by_asset.setdefault(observation["asset_id"], []).append(observation)
        packet = EvidencePacket(spec.query_id, scope_id, spec.answer_target)
        for asset_id, asset in assets.items():
            candidates = by_asset.get(asset_id, [])
            if not candidates:
                candidates = [{}]
            best = None
            for observation in candidates:
                result = self._evaluate(asset, observation, spec)
                if best is None or result["rank"] > best["rank"]:
                    best = result
            if best["excluded"]:
                packet.excluded_count += 1
                continue
            item = best["item"]
            packet.assets.append(item)
            if item["level"] == "exact":
                packet.exact_results.append(item)
            elif item["level"] == "strong":
                packet.strong_results.append(item)
            else:
                packet.approximate_results.append(item)
        packet.assets.sort(key=lambda item: ({"exact": 0, "strong": 1, "approximate": 2}[item["level"]], -item["score"]))
        for constraint in spec.constraints:
            if constraint.strictness == SEMANTIC and not any(item["condition_results"].get(constraint.key, {}).get("status") == "matched" for item in packet.assets):
                packet.gaps.append({"condition": constraint.key, "reason": "no_direct_support"})
        limit = int(spec.result_requirement.get("top_k", 10) or 10)
        packet.assets = packet.assets[:limit] if spec.result_requirement.get("mode") != "all_relevant" else packet.assets
        packet.exact_results = [item for item in packet.exact_results if item in packet.assets]
        packet.strong_results = [item for item in packet.strong_results if item in packet.assets]
        packet.approximate_results = [item for item in packet.approximate_results if item in packet.assets]
        return packet

    def _retrieve_multi(self, spec: QuerySpec):
        """Phase R R2 path: prefilter -> multi-channel recall -> merge -> seed
        gate/adjacency -> condition evidence -> hard postfilter -> fusion.

        Only candidate Assets recalled by at least one enabled retriever are
        evaluated, which is the semantic-difference this phase introduces.
        """
        total_started = time.monotonic()
        from .retrieval import HardFilterContext, RetrievalQuery
        from .retrieval.config import RetrievalConfig
        from .retrieval.fusion import DEFAULT_CHANNEL_WEIGHTS
        from .retrieval.ranking import VISUAL_ONLY, rank

        query_started = time.monotonic()
        config = self.config or RetrievalConfig()
        filters = HardFilterContext.from_spec(spec)
        query = RetrievalQuery.from_spec(spec, embedding_router=self.embedding_router)
        query_build_ms = round((time.monotonic() - query_started) * 1000, 1)
        requested_limit = int(spec.result_requirement.get("top_k", config.top_k) or config.top_k)
        # Candidate recall is threshold-based, not Top-K based. Ask each
        # enabled channel for the complete authorized scope so a relevant
        # asset cannot disappear merely because another channel ranked it
        # below an arbitrary head. Later reranking decides ordering; scope and
        # explicit media type remain the only hard boundaries.
        scope_for_count = None if spec.scope_mode == "all_authorized" else (spec.scope_id or (spec.scope_ids[0] if spec.scope_ids else None))
        try:
            # Ask for the complete authorized scope; the store's default
            # limit=200 is a presentation limit, not a retrieval boundary.
            authorized_count = len(self.store.list_assets(
                scope_id=scope_for_count, limit=100_000))
        except Exception:
            authorized_count = 0
        recall_limit = max(requested_limit, authorized_count, 1)
        # The fused ranking is the single post-recall reducer.  Keep every
        # channel's complete authorized recall for ranking, then hand the
        # model/Agent a bounded candidate head.  A fixed score cut is
        # deliberately avoided here: absolute thresholds cannot adapt to
        # per-query score distributions and silently drop recalled GT
        # (measured on album3-max 100QA: threshold prefilter dropped
        # asset recall from 0.971 to 0.893, while rank top-50 kept 0.971).
        import os
        candidate_limit = max(1, int(os.getenv("SENTRIX_SEARCH_CANDIDATE_TOP_K", "30")))
        # Structured anchors (calendar/place/media) are cheap, deterministic
        # recall channels.  Let their complete scope reach the condition pass;
        # otherwise a valid metadata hit at rank 200+ is discarded before it
        # can be fused with visual/text evidence.  Pure semantic searches keep
        # the bounded head for latency.
        if (filters.time_bounds or filters.annual_time_window
                or filters.place or filters.media_types):
            candidate_limit = max(candidate_limit, min(authorized_count, 1000))
        strategy = config.ranking_strategy
        all_relevant = spec.result_requirement.get("mode") == "all_relevant"
        min_retrieval_score = 0.0

        channel_hits = {}
        channel_trace = {}
        expanders = []
        for retriever in self.retrievers:
            if retriever.kind == "expander":
                expanders.append(retriever)
                continue
            if self.embedding_router and hasattr(self.embedding_router, "get_and_clear_timing_events"):
                self.embedding_router.get_and_clear_timing_events()
            channel_started = time.monotonic()
            try:
                hits = retriever.retrieve(query, filters, limit=recall_limit)
                channel_hits[retriever.name] = hits
                channel_status = getattr(retriever, "status", "ok")
                channel_reason = None
            except Exception as error:
                channel_hits[retriever.name] = []
                channel_status = "error"
                channel_reason = str(error)
            embedding_events = []
            if self.embedding_router and hasattr(self.embedding_router, "get_and_clear_timing_events"):
                embedding_events = self.embedding_router.get_and_clear_timing_events()
            trace = {
                "invoked": True,
                "candidate_count": len(channel_hits[retriever.name]),
                "status": channel_status,
                "backend": getattr(retriever, "backend_used", None),
                "latency_ms": round((time.monotonic() - channel_started) * 1000, 1),
                "embedding_ms": round(sum(event.get("latency_ms", 0) for event in embedding_events), 1),
                "embedding_events": embedding_events,
            }
            if self.embedding_router and hasattr(self.embedding_router, "status"):
                trace["embedding_status"] = self.embedding_router.status()
            if channel_reason:
                trace["reason"] = channel_reason
            channel_trace[retriever.name] = trace

        scope_id = spec.scope_id or (spec.scope_ids[0] if spec.scope_ids else "all_authorized")
        all_authorized = spec.scope_mode == "all_authorized"
        packet = EvidencePacket(spec.query_id, scope_id, spec.answer_target)
        packet.channel_trace = channel_trace
        # R8-3: expose per-channel asset order so the benchmark can trace each
        # GT's rank per channel (which channel moved it up or down).
        packet.channel_hits = {name: [hit.asset_id for hit in hits] for name, hits in channel_hits.items()}

        primary_fusion_started = time.monotonic()
        # Keep this baseline order only for selecting graph seeds and for
        # explaining the later graph-induced ranking delta.  It is *not* the
        # final order once an expander has returned graph evidence.
        baseline_fused = rank(
            channel_hits, strategy, candidate_limit,
            fusion_weights=DEFAULT_CHANNEL_WEIGHTS,
        )
        baseline_rank_by_asset = {
            candidate.asset_id: position
            for position, candidate in enumerate(baseline_fused, 1)
        }
        primary_items = self._evaluate_fused(
            baseline_fused,
            spec, packet, filters, all_authorized, scope_id, skip_assets=set())
        primary_fusion_ms = round((time.monotonic() - primary_fusion_started) * 1000, 1)

        # R3B seed-gated adjacency — R8-3: only expands when the strategy is
        # not visual_only, and only for all_relevant or reliable seeds.
        already = {item["asset_id"] for item in primary_items}
        graph_policy_for_fusion = None
        graph_expanders = [expander for expander in expanders
                           if getattr(expander, "name", "") == "graph"]
        if expanders and ((strategy != VISUAL_ONLY or all_relevant) or graph_expanders):
            reliable_seeds = [item["asset_id"] for item in primary_items
                              if item.get("level") in {"exact", "strong"}]
            adjacency_trace = {"invoked": True, "candidate_count": 0, "status": "no_seeds"}
            for expander in expanders:
                # Graph late-fusion intentionally starts from the wider
                # baseline head; adjacency keeps its stricter reliable-seed
                # gate to avoid changing existing photo retrieval semantics.
                # A graph seed must only have passed the *hard* scope/media
                # filters.  Requiring it to have already passed final
                # condition verification turns graph retrieval into a
                # post-processing step: a plausible anchor that needs its
                # neighbours to answer the question is discarded before the
                # graph can ever see it.  ``baseline_fused`` is already made
                # from hard-filtered retriever outputs; the expanded results
                # still go through _evaluate_fused below before being returned.
                seeds = ([candidate.asset_id for candidate in baseline_fused[:30]
                          if candidate.asset_id] if getattr(expander, "name", "") == "graph"
                         else reliable_seeds)
                graph_policy = None
                if getattr(expander, "name", "") == "graph" and hasattr(expander, "route"):
                    try:
                        graph_policy = expander.route(query, filters)
                    except Exception as error:
                        graph_policy = {
                            "enabled": False,
                            "mode": "auto",
                            "intent": "unknown",
                            "reason": f"policy_error:{type(error).__name__}",
                            "policy_version": "graph-route-v1",
                        }
                    graph_policy_for_fusion = graph_policy
                channel_started = time.monotonic()
                if graph_policy and not graph_policy.get("enabled"):
                    adjacency_hits = []
                    adjacency_trace = {
                        "invoked": False,
                        "candidate_count": 0,
                        "status": "skipped_policy",
                        "seeds": len(seeds),
                        "policy": graph_policy,
                        "latency_ms": 0.0,
                        "embedding_ms": 0.0,
                        "embedding_events": [],
                    }
                    channel_hits[expander.name] = adjacency_hits
                    channel_trace[expander.name] = adjacency_trace
                    continue
                try:
                    try:
                        adjacency_hits = expander.expand(
                            seeds, filters, limit=recall_limit, query=query)
                    except TypeError:
                        adjacency_hits = expander.expand(seeds, filters, limit=recall_limit)
                    adjacency_trace = {"invoked": True, "candidate_count": len(adjacency_hits),
                                       "status": "ok", "seeds": len(seeds)}
                    if graph_policy:
                        adjacency_trace["policy"] = graph_policy
                except Exception as error:
                    adjacency_hits = []
                    adjacency_trace = {"invoked": True, "candidate_count": 0, "status": "error", "reason": str(error)}
                    if graph_policy:
                        adjacency_trace["policy"] = graph_policy
                adjacency_trace["latency_ms"] = round((time.monotonic() - channel_started) * 1000, 1)
                adjacency_trace["embedding_ms"] = 0.0
                adjacency_trace["embedding_events"] = []
                channel_hits[expander.name] = adjacency_hits
                channel_trace[expander.name] = adjacency_trace
            packet.channel_hits[expander.name] = [hit.asset_id for hit in channel_hits.get(expander.name, [])]

        # This is the actual graph re-rank.  The old implementation ranked
        # graph hits in isolation and appended them after the ordinary list;
        # graph evidence could therefore never move an existing candidate.
        # Fuse every channel together so a path hit boosts an existing seed
        # and a newly reached node competes in the same candidate universe.
        adjacency_fusion_started = time.monotonic()
        # Graph candidates are useful evidence, but their safe weight depends
        # on the query intent.  A generic/ordinary query has no directional
        # path requirement, so broad graph neighbours must not displace
        # stronger visual/lexical hits.  Temporal, relation, causal and
        # multi-hop queries explicitly ask for graph semantics and get a
        # stronger graph contribution.  This keeps ``on`` useful for an
        # ablation while making the ranking itself intent-aware.
        graph_fusion_weights = dict(DEFAULT_CHANNEL_WEIGHTS)
        graph_intent = str((graph_policy_for_fusion or {}).get("intent") or "ordinary")
        # A graph-semantic route must be able to introduce a candidate that
        # no ordinary channel recalled.  With the old 1.35 weight, a rank-1
        # graph-only hit contributed less than a rank-1 visual hit and almost
        # never entered the final top-30, so telemetry showed many graph
        # promotions but essentially no GT recall change.  Use a bounded
        # intent-specific boost: explicit path questions get the strongest
        # boost, while ordinary/on-mode queries remain conservative.
        graph_fusion_weights["graph"] = {
            "multi_hop": 4.0,
            "causal": 3.5,
            "temporal": 3.0,
            "relationship": 3.0,
            "cross_media": 3.0,
        }.get(graph_intent, 0.35)
        # Rank a wider pool first, then reserve a small number of head slots
        # for graph-only candidates.  Weighted RRF alone cannot do this: a
        # graph hit that was absent from the visual/lexical channels gets one
        # rank contribution and is routinely pushed below the 30-item head.
        # That made graph telemetry look active while the returned candidate
        # set was effectively unchanged.  The quota is bounded and only
        # applies when graph retrieval is enabled; ordinary ``auto`` queries
        # still have no graph channel at all.
        graph_pool_limit = max(candidate_limit, candidate_limit * 3)
        ranked_pool = rank(
            channel_hits, strategy, graph_pool_limit,
            fusion_weights=graph_fusion_weights,
        )
        graph_hits = channel_hits.get("graph") or []
        graph_asset_ids = {hit.asset_id for hit in graph_hits}
        baseline_ids = set(baseline_rank_by_asset)
        graph_only_pool = [candidate for candidate in ranked_pool
                           if candidate.asset_id in graph_asset_ids
                           and candidate.asset_id not in baseline_ids]
        try:
            configured_graph_quota = int(os.getenv("SENTRIX_GRAPH_HEAD_QUOTA", "8"))
        except (TypeError, ValueError):
            configured_graph_quota = 6
        graph_quota = min(
            max(0, configured_graph_quota),
            candidate_limit // 3,
            len(graph_only_pool),
        )
        combined_fused = list(ranked_pool[:candidate_limit])
        forced_graph_ids = []
        if graph_quota:
            current_graph_count = sum(
                candidate.asset_id in graph_asset_ids for candidate in combined_fused)
            missing_quota = max(0, graph_quota - current_graph_count)
            if missing_quota:
                forced = graph_only_pool[:missing_quota]
                forced_graph_ids = [candidate.asset_id for candidate in forced]
                forced_set = set(forced_graph_ids)
                kept = [candidate for candidate in combined_fused
                        if candidate.asset_id not in forced_set]
                # Replace the weakest non-graph tail instead of prepending a
                # broad graph result.  This preserves most of the baseline
                # order while guaranteeing that graph recall is observable in
                # the actual returned head.
                combined_fused = (kept[:max(0, candidate_limit - len(forced))]
                                  + forced)
        combined_rank_by_asset = {
            candidate.asset_id: position
            for position, candidate in enumerate(combined_fused, 1)
        }
        graph_supported_top = [
            candidate.asset_id for candidate in combined_fused[:candidate_limit]
            if candidate.asset_id in graph_asset_ids
        ]
        graph_rerank = {
            "applied": bool(graph_hits),
            "baseline_candidate_count": len(baseline_fused),
            "combined_candidate_count": len(combined_fused),
            "graph_candidate_count": len(graph_hits),
            "graph_supported_top_count": len(graph_supported_top),
            "new_candidate_count": sum(
                asset_id not in baseline_rank_by_asset for asset_id in graph_asset_ids),
            "promoted_count": sum(
                1 for asset_id, position in combined_rank_by_asset.items()
                if asset_id in baseline_rank_by_asset
                and position < baseline_rank_by_asset[asset_id]),
            "demoted_count": sum(
                1 for asset_id, position in combined_rank_by_asset.items()
                if asset_id in baseline_rank_by_asset
                and position > baseline_rank_by_asset[asset_id]),
            "graph_fusion_weight": graph_fusion_weights["graph"],
            "graph_intent": graph_intent,
            "graph_head_quota": graph_quota,
            "graph_forced_head_count": len(forced_graph_ids),
            # Raw IDs stay in the sanitized-out debug observation only.  The
            # evaluator resolves them to media names and measures whether the
            # graph changed GT recall/precision for this exact question.
            "baseline_ranked_asset_ids": [candidate.asset_id for candidate in baseline_fused],
            "reranked_asset_ids": [candidate.asset_id for candidate in combined_fused],
        }
        combined_items = self._evaluate_fused(
            combined_fused, spec, packet, filters, all_authorized, scope_id,
            skip_assets=set(),
        )
        adjacency_fusion_ms = round((time.monotonic() - adjacency_fusion_started) * 1000, 1)

        postprocess_started = time.monotonic()
        packet.assets = combined_items
        for item in packet.assets:
            if item["level"] == "exact":
                packet.exact_results.append(item)
            elif item["level"] == "strong":
                packet.strong_results.append(item)
            else:
                packet.approximate_results.append(item)

        # Evidence level remains a safety boundary; within a level preserve
        # the unified multi-channel fusion order rather than comparing raw
        # graph-path and ANN scores that live on incompatible scales.
        packet.assets.sort(key=lambda item: (
            {"exact": 0, "strong": 1, "approximate": 2}[item["level"]],
            -float(item.get("fusion_score") or 0.0),
            item["asset_id"],
        ))
        # Optional confidence gate. It is disabled by default because score
        # scales differ by retriever. When calibrated, this threshold is the
        # only reduction mechanism; there is no fixed candidate Top-K.
        import os
        try:
            min_retrieval_score = float(os.getenv("SENTRIX_SEARCH_MIN_RETRIEVAL_SCORE", "0") or 0)
        except (TypeError, ValueError):
            min_retrieval_score = 0.0
        if min_retrieval_score > 0:
            packet.assets = [item for item in packet.assets
                             if float(item.get("retrieval_score") or 0) >= min_retrieval_score]
            packet.exact_results = [item for item in packet.exact_results if item in packet.assets]
            packet.strong_results = [item for item in packet.strong_results if item in packet.assets]
            packet.approximate_results = [item for item in packet.approximate_results if item in packet.assets]
        for constraint in spec.constraints:
            if constraint.strictness == SEMANTIC and not any(item["condition_results"].get(constraint.key, {}).get("status") == "matched" for item in packet.assets):
                packet.gaps.append({"condition": constraint.key, "reason": "no_direct_support"})
        # Multi-channel recall is confidence/threshold based.  ``recall_limit``
        # is only the per-channel request size (expanded to the authorized
        # scope above); never truncate the fused candidate universe by a
        # presentation-oriented Top-K here.  Delivery and evidence selection
        # happen in the Agent tool layer.
        packet.exact_results = [item for item in packet.exact_results if item in packet.assets]
        packet.strong_results = [item for item in packet.strong_results if item in packet.assets]
        packet.approximate_results = [item for item in packet.approximate_results if item in packet.assets]
        packet.retrieval_timing = {
            "total_ms": round((time.monotonic() - total_started) * 1000, 1),
            "query_build_ms": query_build_ms,
            "channels": channel_trace,
            "fusion_ms": round(primary_fusion_ms + adjacency_fusion_ms, 1),
            "postprocess_ms": round((time.monotonic() - postprocess_started) * 1000, 1),
            "min_retrieval_score": min_retrieval_score,
        }
        graph_trace = channel_trace.get("graph") or {}
        if graph_trace.get("policy"):
            packet.retrieval_timing["graph_policy"] = graph_trace["policy"]
            packet.retrieval_timing["graph_invoked"] = bool(graph_trace.get("invoked"))
            packet.retrieval_timing["graph_candidate_count"] = int(graph_trace.get("candidate_count") or 0)
            packet.retrieval_timing["graph_rerank"] = graph_rerank
        if self.embedding_router and hasattr(self.embedding_router, "status"):
            packet.retrieval_timing["embedding_status"] = self.embedding_router.status()
        return packet

    def _evaluate_fused(self, fused, spec, packet, filters, all_authorized, scope_id, *, skip_assets):
        """Run the condition pass over fused candidates; return non-excluded items."""
        fused = list(fused)
        self._prime_evaluation_cache(None if all_authorized else scope_id)
        items = []
        for candidate in fused:
            if candidate.asset_id in skip_assets:
                continue
            asset = (self._asset_cache or {}).get(candidate.asset_id)
            if asset is None and self._asset_cache is None:
                asset = self.store.get_asset(candidate.asset_id)
            if asset is None:
                continue
            if not all_authorized and (asset.get("scope_id") or "home-default") != scope_id:
                continue
            observations = self._observations_for_asset(candidate.asset_id)
            best = None
            for observation in observations or [{}]:
                result = self._evaluate(asset, observation, spec)
                if best is None or result["rank"] > best["rank"]:
                    best = result
            if best["excluded"]:
                packet.excluded_count += 1
                continue
            item = best["item"]
            item["attributions"] = [
                {"retriever": hit.retriever, "rank": hit.rank, "score": hit.raw_score,
                 "score_kind": hit.score_kind}
                for hit in candidate.retriever_hits
            ]
            item["fusion_score"] = round(candidate.rrf, 4)
            # Preserve the strongest channel score separately from RRF.  RRF
            # is only an ordering signal; an optional confidence threshold can
            # be calibrated against this raw score without reintroducing a
            # fixed candidate count.
            item["retrieval_score"] = round(max(
                (float(hit.raw_score) for hit in candidate.retriever_hits
                 if hit.raw_score is not None), default=0.0), 6)
            items.append(item)
        return items

    def _observations_for_asset(self, asset_id):
        if self._observations_by_asset_cache is not None:
            return self._observations_by_asset_cache.get(asset_id, [])
        # Compatibility fallback for lightweight stores that cannot serve the
        # bulk projection used by production retrieval.
        return self.store.list_observations(asset_id=asset_id, limit=1000)

    def _prime_evaluation_cache(self, scope_id):
        """Load verifier inputs once for the current retrieval request.

        ``list_assets`` and ``list_observations`` decode rows in one query each.
        Event evidence is also projected in one join.  This keeps full-scope
        recall semantics while making verifier cost linear in rows instead of
        linear in candidates multiplied by database round trips.
        """
        if self._asset_cache is not None:
            return
        try:
            assets = self.store.list_assets(scope_id=scope_id, limit=100_000)
            observations = self.store.list_observations(
                scope_id=scope_id, limit=100_000)
        except (AttributeError, TypeError):
            return
        self._asset_cache = {str(item.get("id")): item for item in assets if item.get("id")}
        by_asset = {}
        observation_ids = set()
        for observation in observations:
            asset_id = str(observation.get("asset_id") or "")
            if not asset_id or asset_id not in self._asset_cache:
                continue
            by_asset.setdefault(asset_id, []).append(observation)
            if observation.get("id"):
                observation_ids.add(str(observation["id"]))
        self._observations_by_asset_cache = by_asset
        event_text = {}
        try:
            rows = self.store.connection.execute(
                "SELECT eo.observation_id, e.place, e.title, e.summary "
                "FROM event_observations eo JOIN events e ON e.id=eo.event_id"
            ).fetchall()
            for row in rows:
                observation_id = str(row[0] or "")
                if observation_id not in observation_ids:
                    continue
                value = " ".join(str(item or "") for item in row[1:])
                if value:
                    event_text[observation_id] = (
                        event_text.get(observation_id, "") + " " + value
                    ).strip()
        except Exception:
            pass
        self._event_text_by_observation_cache = event_text

    def probe(self, raw_text: str, scope_id: str | None, viewer_id: str = "owner",
              *, focus=None, media_hint=None):
        """Neutral probe: run the shared retrievers under probe budgets (R4/R9-2).

        Returns ``(channel_hits, index_health)`` for the NeutralProbe to
        aggregate.  Scope / media come from the request context; no unconfirmed
        hard semantic constraints are fabricated (P0-7).  ``focus`` and
        ``media_hint`` are forwarded to the probe decision (session follow-up
        and media-aware weighting).  The confirmed-entity signal is covered by
        the primary ``entity`` retriever in the shared set.
        """
        if not self.retrievers:
            return {}, {}
        from .retrieval import HardFilterContext, RetrievalQuery
        from .retrieval.config import RetrievalConfig
        config = self.config or RetrievalConfig()
        filters = HardFilterContext(
            scope_ids=(scope_id,) if scope_id else (),
            viewer_id=viewer_id,
        )
        query = RetrievalQuery(
            whole_query=raw_text or "",
            facets=[QueryFacet("semantic", raw_text or "")],
        )
        channel_hits = {}
        index_health = {}
        for retriever in self.retrievers:
            if retriever.kind != "primary":
                continue
            try:
                hits = retriever.retrieve(query, filters, limit=config.probe_top_k)
                channel_hits[retriever.name] = hits
                index_health[retriever.name] = {"status": "ok", "hits": len(hits)}
            except Exception as exc:
                channel_hits[retriever.name] = []
                index_health[retriever.name] = {"status": "error", "detail": type(exc).__name__}
        return channel_hits, index_health

    # Phase R P1-2: a ``matched`` status is only allowed from evidence sources
    # that directly prove the condition.  Vector / FTS / generic pool hits can
    # only produce ``possible`` — a high cosine or a token overlap never proves
    # a household fact.
    _MATCHED_SOURCE_TYPES = frozenset({
        "asset_metadata", "observation_field_exact", "confirmed_bridge",
        "entity_bridge_confirmed", "subject_binding",
    })

    def _evaluate(self, asset, observation, spec):
        results = {}
        excluded = False
        for constraint in spec.constraints:
            status, source_type, source_id, confidence = self._condition(asset, observation, constraint)
            if status == "matched" and source_type not in self._MATCHED_SOURCE_TYPES:
                status = "possible"
            results[constraint.key] = {"status": status, "source_type": source_type, "source_id": source_id, "confidence": confidence}
            if constraint.strictness == HARD and (status != "matched" if not constraint.negated else status == "matched"):
                # Place metadata is open-world: a derived video keyframe often
                # has no reverse-geocode of its own even though the uploaded
                # parent carries GPS.  An unknown place must not erase a
                # semantically matched event from the candidate set; only an
                # authoritative, conflicting geocode is a hard contradiction.
                # The slot reranker applies the explicit place score when it
                # is available and the event-text score when it is not.
                if not (constraint.dimension == "place" and status == "unknown"):
                    excluded = True
            if constraint.strictness == SEMANTIC and status == "contradicted":
                excluded = True
        semantic = [item["status"] for item in results.values() if item["status"] in {"possible", "unknown", "contradicted"}]
        level = "exact" if not semantic else "approximate"
        if semantic and all(status == "possible" for status in semantic):
            level = "strong"
        evidence_ids = [asset["id"]] + ([observation.get("id")] if observation.get("id") else [])
        item = {
            "asset_id": asset["id"], "file_name": asset.get("file_name"), "media_type": asset.get("media_type"),
            "captured_at": asset.get("captured_at") or observation.get("captured_at"), "observation_ids": [observation["id"]] if observation.get("id") else [],
            "evidence_ids": evidence_ids, "condition_results": results, "level": level,
            "score": round(sum(1 for value in results.values() if value["status"] == "matched") / max(1, len(results)), 4),
            "observation_fields": {
                "place": observation.get("place"),
                "activity": observation.get("activity"),
                # Formation may provide this field when it has a real
                # face/body subject binding.  The scene clothing list stays
                # out of person summaries by design.
                "subject_clothing": observation.get("subject_clothing") or [],
            },
        }
        return {"excluded": excluded, "item": item, "rank": (0 if excluded else 3 if level == "exact" else 2 if level == "strong" else 1)}

    # Multi-value observation fields — a miss on these produces ``unknown``,
    # never ``contradicted``.  Only formation-provided subject bindings may
    # produce a real contradiction for the same subject.
    _OPEN_WORLD_LIST_DIMENSIONS = {"clothing", "object", "ocr"}
    # Single-value observation fields where a mismatch is a genuine reason
    # to contradict the constraint.
    _SINGLE_VALUE_DIMENSIONS = {"place"}

    def _condition(self, asset, observation, constraint):
        value = constraint.value
        if constraint.dimension == "time":
            bounds = parse_time_expression(value)
            captured = _parse_datetime(asset.get("captured_at") or observation.get("captured_at"))
            if bounds:
                matched = bool(captured and bounds[0] <= captured < bounds[1])
            else:
                from .query_contracts import parse_annual_time_expression
                window = parse_annual_time_expression(value)
                matched = bool(captured and window and _in_annual_window(captured, window))
            return ("matched", "asset_metadata", asset.get("id"), 1.0) if matched else ("contradicted", "asset_metadata", asset.get("id"), 1.0)
        if constraint.dimension == "media":
            return ("matched", "asset_metadata", asset.get("id"), 1.0) if asset.get("media_type") == value else ("contradicted", "asset_metadata", asset.get("id"), 1.0)
        if constraint.dimension == "person":
            # people entries may be plain names OR confirmed-entity dicts
            # ({entity_id, name, status}).  A confirmed bridge must match both,
            # and also observations linked to the confirmed person via
            # entity_mentions (the benchmark observations carry the person only
            # as a mention, not in the people text field).
            names, entity_ids = set(), set()
            for entry in observation.get("people") or []:
                if isinstance(entry, dict):
                    if entry.get("name"):
                        names.add(str(entry["name"]))
                    if entry.get("entity_id"):
                        entity_ids.add(str(entry["entity_id"]))
                else:
                    names.add(str(entry))
            if value in names or value in entity_ids:
                return ("matched", "confirmed_bridge", observation.get("id"), 1.0)
            if self._observation_has_confirmed_person(observation.get("id"), value):
                return ("matched", "confirmed_bridge", observation.get("id"), 1.0)
            return ("unknown", None, None, 0.0)
        if constraint.dimension in self._OPEN_WORLD_LIST_DIMENSIONS:
            return self._evaluate_open_world(observation, constraint)
        if constraint.dimension == "place":
            return self._evaluate_place(asset, observation, constraint)
        if constraint.dimension in self._SINGLE_VALUE_DIMENSIONS:
            return self._evaluate_single_value(observation, constraint)
        if constraint.dimension == "activity":
            return self._evaluate_activity(observation, constraint)
        # Unknown/other dimension — fall through to a full-text match on
        # caption + labels.  A miss is unknown; we never contradict from a
        # generic keyword pool.
        return self._evaluate_semantic_pool(observation, constraint)

    def _observation_has_confirmed_person(self, observation_id, name):
        """True when an observation carries an entity_mention to a confirmed
        person whose canonical name matches (benchmark person linkage lives in
        entity_mentions, not the people text field)."""
        if not observation_id or self.store is None:
            return False
        try:
            for mention in self.store.entity_mentions_for_observation(observation_id):
                entity = self.store.get_entity(mention["entity_id"])
                if entity and entity.get("status") == "confirmed" \
                        and entity.get("canonical_name") == name:
                    return True
            # Entity mentions are a sparse annotation projection.  A confirmed
            # face cluster attached to the same observation is already a valid
            # identity bridge and must participate in retrieval conditions.
            row = self.store.connection.execute(
                "SELECT 1 FROM face_instances fi "
                "JOIN face_clusters fc ON fc.id = fi.cluster_id "
                "JOIN entities e ON e.id = fc.entity_id "
                "WHERE fi.observation_id = ? AND fc.status = 'confirmed' "
                "AND e.status = 'confirmed' AND e.entity_type = 'person' "
                "AND e.canonical_name = ? LIMIT 1",
                (observation_id, name),
            ).fetchone()
            if row:
                return True
        except Exception:
            return False
        return False

    @staticmethod
    def _evaluate_open_world(observation, constraint):
        """Open-world semantics for list-shaped fields (plan §10)."""
        subject_field = {"clothing": "subject_clothing", "object": "subject_objects"}.get(constraint.dimension)
        bindings = observation.get(subject_field) or [] if subject_field else []
        if bindings:
            same_subject_match = False
            same_subject_other = False
            for binding in bindings:
                bound_value = str(binding.get("value") or "").strip()
                if not bound_value:
                    continue
                if _contains(bound_value, constraint.value):
                    same_subject_match = True
                else:
                    same_subject_other = True
            if same_subject_match:
                return ("matched", "subject_binding", observation.get("id"),
                        float(observation.get("confidence", 0) or 0))
            if same_subject_other:
                return ("contradicted", "subject_binding", observation.get("id"),
                        float(observation.get("confidence", 0) or 0))
        field = {"clothing": "clothing", "object": "objects", "ocr": "ocr_text"}[constraint.dimension]
        raw = observation.get(field) or ""
        joined = " ".join(str(item) for item in raw) if isinstance(raw, list) else str(raw)
        if _contains(joined, constraint.value):
            return ("possible", "observation", observation.get("id"),
                    float(observation.get("confidence", 0) or 0))
        return ("unknown", None, None, 0.0)

    @staticmethod
    def _asset_geocode(asset):
        """从资产元数据解析反地理编码记录（dict 或 JSON 字符串都兼容）。"""
        metadata = asset.get("metadata_json") or {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except (TypeError, ValueError):
                metadata = {}
        geocode = metadata.get("reverse_geocode") or {}
        if isinstance(geocode, str):
            try:
                geocode = json.loads(geocode)
            except (TypeError, ValueError):
                geocode = {}
        if isinstance(geocode, dict) and geocode:
            return geocode
        # Keep GPS-only imports searchable without changing the hard-filter
        # contract: resolve through the optional offline metadata/geocoder
        # fallback and let the normal place evaluator consume the result.
        location = str(asset.get("captured_location") or "")
        match = re.search(r"(-?\d+(?:\.\d+)?)\s*[,; ]\s*(-?\d+(?:\.\d+)?)", location)
        if match:
            try:
                from .geocoding import default_reverse_geocoder
                return default_reverse_geocoder().lookup(
                    {"latitude": match.group(1), "longitude": match.group(2)},
                    filename=asset.get("file_name"),
                ) or {}
            except Exception:
                pass
        return {}

    def _evaluate_place(self, asset, observation, constraint):
        """地理地点条件（D12）。

        权威来源是 assets.metadata_json.reverse_geocode（GPS 反地理编码）；
        observation.place 是场景类型（"室内餐厅或咖啡馆"），只作为弱文本信号，
        不能产生矛盾。有权威 geocode 但确不匹配才判 contradicted（排除）；
        无 geocode 时保持 unknown（开放世界，不能因为照片没 GPS 就剔除）。
        """
        value = str(constraint.value or "").strip()
        if not value:
            return ("unknown", None, None, 0.0)
        geocode = self._asset_geocode(asset)
        if geocode and place_text_matches(value, geocode):
            return ("matched", "asset_metadata", asset.get("id"),
                    float(geocode.get("confidence") or 0.9))
        # A normalized observation.place value is a direct field fact when it
        # exactly names the requested place.  Keep broader captions/scene
        # prose weak, but do not downgrade this precise field to possible.
        observation_place = str(observation.get("place") or "").strip()
        if observation_place and observation_place == value:
            return ("matched", "observation_field_exact", observation.get("id"),
                    float(observation.get("confidence") or 0))
        pool = " ".join(filter(None, [observation.get("place"), observation.get("caption")]))
        if _contains(pool, value):
            return ("possible", "observation", observation.get("id"),
                    float(observation.get("confidence") or 0))
        # Event summaries are separate from the observation projection and
        # often carry the only human-readable place for videos.  Treat a
        # direct place mention there as supported (not authoritative matched)
        # so the Agent receives a useful partial-support signal without
        # turning a generated summary into a hard geocode fact.
        try:
            observation_id = str(observation.get("id") or "")
            if self._event_text_by_observation_cache is not None:
                event_pool = self._event_text_by_observation_cache.get(observation_id, "")
            else:
                event_rows = self.store.connection.execute(
                    "SELECT e.place, e.title, e.summary FROM events e "
                    "JOIN event_observations eo ON eo.event_id=e.id "
                    "WHERE eo.observation_id=?", (observation_id,)
                ).fetchall()
                event_pool = " ".join(str(value or "") for row in event_rows for value in row)
            if _contains(event_pool, value):
                return ("possible", "event_summary", observation.get("id"), 0.7)
        except Exception:
            pass
        if geocode:
            return ("contradicted", "asset_metadata", asset.get("id"),
                    float(geocode.get("confidence") or 0.9))
        return ("unknown", None, None, 0.0)

    @staticmethod
    def _evaluate_single_value(observation, constraint):
        """Single-value field: mismatch may contradict, absent field is unknown."""
        field = {"place": "place"}[constraint.dimension]
        raw = observation.get(field) or ""
        text = " ".join(str(item) for item in raw) if isinstance(raw, list) else str(raw)
        if _contains(text, constraint.value):
            return ("matched", "observation_field_exact", observation.get("id"),
                    float(observation.get("confidence", 0) or 0))
        if text:
            return ("contradicted", "observation_field_exact", observation.get("id"),
                    float(observation.get("confidence", 0) or 0))
        return ("unknown", None, None, 0.0)

    @staticmethod
    def _evaluate_activity(observation, constraint):
        """Activity labels never contradict — the picture may show what the
        label omitted."""
        raw = observation.get("activity") or ""
        text = " ".join(str(item) for item in raw) if isinstance(raw, list) else str(raw)
        if _contains(text, constraint.value):
            return ("matched", "observation_field_exact", observation.get("id"),
                    float(observation.get("confidence", 0) or 0))
        return ("unknown", None, None, 0.0)

    @staticmethod
    def _evaluate_semantic_pool(observation, constraint):
        pool = " ".join(
            " ".join(str(item) for item in observation.get(key) or []) if isinstance(observation.get(key), list) else str(observation.get(key) or "")
            for key in ("caption", "activity", "place", "objects", "clothing", "ocr_text")
        )
        if _contains(pool, constraint.value):
            return ("possible", "observation", observation.get("id"),
                    float(observation.get("confidence", 0) or 0))
        return ("unknown", None, None, 0.0)
