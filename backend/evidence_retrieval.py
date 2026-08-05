"""Correctness-first Asset-level Evidence Retrieval Kernel."""

from dataclasses import dataclass, field
from datetime import datetime
import re

from .query_contracts import HARD, SEMANTIC, QuerySpec, parse_time_expression


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
    haystack, needle = str(haystack or "").lower(), str(needle or "").lower()
    if not needle:
        return False
    if needle in haystack:
        return True
    terms = [term for term in re.findall(r"[\w\u4e00-\u9fff]+", needle) if len(term) > 1]
    return bool(terms) and all(term in haystack for term in terms)


class EvidenceRetrievalKernel:
    def __init__(self, store):
        self.store = store

    def retrieve(self, spec: QuerySpec):
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

    def _evaluate(self, asset, observation, spec):
        results = {}
        excluded = False
        for constraint in spec.constraints:
            status, source_type, source_id, confidence = self._condition(asset, observation, constraint)
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
            return ("matched", "observation", observation.get("id"),
                    float(observation.get("confidence", 0) or 0))
        if text:
            return ("contradicted", "observation", observation.get("id"),
                    float(observation.get("confidence", 0) or 0))
        return ("unknown", None, None, 0.0)

    @staticmethod
    def _evaluate_activity(observation, constraint):
        """Activity labels never contradict — the picture may show what the
        label omitted."""
        raw = observation.get("activity") or ""
        text = " ".join(str(item) for item in raw) if isinstance(raw, list) else str(raw)
        if _contains(text, constraint.value):
            return ("matched", "observation", observation.get("id"),
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
