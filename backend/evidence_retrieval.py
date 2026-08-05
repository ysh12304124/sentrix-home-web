"""Correctness-first Asset-level Evidence Retrieval Kernel."""

from dataclasses import dataclass, field
from datetime import datetime

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


class EvidenceRetrievalKernel:
    def __init__(self, store, *, retrievers=None, embedding_router=None, config=None, trace=None):
        self.store = store
        self.retrievers = list(retrievers or [])
        self.embedding_router = embedding_router
        self.config = config
        self._trace_sink = trace

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
        assets = {item["id"]: item for item in self.store.list_assets(scope_id=query_scope)}
        observations = [item for item in self.store.list_observations(scope_id=query_scope) if item.get("asset_id") in assets]
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
        from .retrieval import HardFilterContext, RetrievalQuery, fuse
        from .retrieval.config import RetrievalConfig

        config = self.config or RetrievalConfig()
        filters = HardFilterContext.from_spec(spec)
        query = RetrievalQuery.from_spec(spec, embedding_router=self.embedding_router)
        recall_limit = int(spec.result_requirement.get("top_k", config.top_k) or config.top_k)

        channel_hits = {}
        channel_trace = {}
        expanders = []
        for retriever in self.retrievers:
            if retriever.kind == "expander":
                expanders.append(retriever)
                continue
            try:
                hits = retriever.retrieve(query, filters, limit=recall_limit)
                channel_hits[retriever.name] = hits
                channel_trace[retriever.name] = {
                    "invoked": True, "candidate_count": len(hits), "status": getattr(retriever, "status", "ok"),
                }
            except Exception as error:
                channel_hits[retriever.name] = []
                channel_trace[retriever.name] = {"invoked": True, "candidate_count": 0,
                                                 "status": "error", "reason": str(error)}

        scope_id = spec.scope_id or (spec.scope_ids[0] if spec.scope_ids else "all_authorized")
        all_authorized = spec.scope_mode == "all_authorized"
        packet = EvidencePacket(spec.query_id, scope_id, spec.answer_target)
        packet.channel_trace = channel_trace

        primary_items = self._evaluate_fused(fuse(channel_hits), spec, packet,
                                             filters, all_authorized, scope_id,
                                             skip_assets=set())

        # R3B: seed-quality gate — only exact/strong primary results are seeds.
        seeds = [item["asset_id"] for item in primary_items if item.get("level") in {"exact", "strong"}]
        adjacency_trace = {"invoked": True, "candidate_count": 0, "status": "no_seeds"}
        for expander in expanders:
            try:
                adjacency_hits = expander.expand(seeds, filters, limit=recall_limit)
                adjacency_trace = {"invoked": True, "candidate_count": len(adjacency_hits),
                                   "status": "ok", "seeds": len(seeds)}
            except Exception as error:
                adjacency_hits = []
                adjacency_trace = {"invoked": True, "candidate_count": 0, "status": "error", "reason": str(error)}
            channel_hits[expander.name] = adjacency_hits
            channel_trace[expander.name] = adjacency_trace

        already = {item["asset_id"] for item in primary_items}
        if expanders:
            adjacency_items = self._evaluate_fused(fuse({expander.name: channel_hits[expander.name]
                                                         for expander in expanders}),
                                                   spec, packet, filters, all_authorized, scope_id,
                                                   skip_assets=already)
        else:
            adjacency_items = []

        packet.assets = primary_items + [item for item in adjacency_items if item["asset_id"] not in already]
        for item in packet.assets:
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
        if spec.result_requirement.get("mode") != "all_relevant":
            packet.assets = packet.assets[:recall_limit]
        packet.exact_results = [item for item in packet.exact_results if item in packet.assets]
        packet.strong_results = [item for item in packet.strong_results if item in packet.assets]
        packet.approximate_results = [item for item in packet.approximate_results if item in packet.assets]
        return packet

    def _evaluate_fused(self, fused, spec, packet, filters, all_authorized, scope_id, *, skip_assets):
        """Run the condition pass over fused candidates; return non-excluded items."""
        items = []
        for candidate in fused:
            if candidate.asset_id in skip_assets:
                continue
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
            items.append(item)
        return items

    def _observations_for_asset(self, asset_id):
        return [item for item in self.store.list_observations(limit=100_000) if item.get("asset_id") == asset_id]

    def probe(self, raw_text: str, scope_id: str | None, viewer_id: str = "owner"):
        """Neutral probe: run the shared retrievers under probe budgets (R4).

        Returns per-channel CandidateHits for the NeutralProbe to aggregate.
        Scope / media come from the request context; no unconfirmed hard
        semantic constraints are fabricated (P0-7).
        """
        if not self.retrievers:
            return {}
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
        for retriever in self.retrievers:
            if retriever.kind != "primary":
                continue
            try:
                channel_hits[retriever.name] = retriever.retrieve(query, filters, limit=config.probe_top_k)
            except Exception:
                channel_hits[retriever.name] = []
        return channel_hits

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
            return ("matched", "asset_metadata", asset.get("id"), 1.0) if bounds and captured and bounds[0] <= captured < bounds[1] else ("contradicted", "asset_metadata", asset.get("id"), 1.0)
        if constraint.dimension == "media":
            return ("matched", "asset_metadata", asset.get("id"), 1.0) if asset.get("media_type") == value else ("contradicted", "asset_metadata", asset.get("id"), 1.0)
        if constraint.dimension == "person":
            people = observation.get("people") or []
            return ("matched", "confirmed_bridge", observation.get("id"), 1.0) if value in people else ("unknown", None, None, 0.0)
        if constraint.dimension in self._OPEN_WORLD_LIST_DIMENSIONS:
            return self._evaluate_open_world(observation, constraint)
        if constraint.dimension in self._SINGLE_VALUE_DIMENSIONS:
            return self._evaluate_single_value(observation, constraint)
        if constraint.dimension == "activity":
            return self._evaluate_activity(observation, constraint)
        # Unknown/other dimension — fall through to a full-text match on
        # caption + labels.  A miss is unknown; we never contradict from a
        # generic keyword pool.
        return self._evaluate_semantic_pool(observation, constraint)

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
