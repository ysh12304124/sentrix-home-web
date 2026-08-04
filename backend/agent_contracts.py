"""Typed boundaries for the Agent orchestration layer.

PydanticAI is optional at import time so SQLite-only tests and offline
maintenance commands remain usable. The framework can supply a plan, but the
deterministic validator below always owns permissions and fallback behavior.
"""

from dataclasses import dataclass
import json
import os
import re


ALLOWED_MODES = {"chat", "memory", "feedback", "clarify"}
ALLOWED_TOOLS = {
    "resolve_constraints", "describe_entity", "find_events", "trace_timeline",
    "compare_memories", "suggest_recall", "open_evidence", "request_clarification",
    "record_feedback",
}

CANONICAL_EVIDENCE_KINDS = {
    "event", "observation", "asset", "fact", "semantic_claim",
    "person", "relationship", "person_appearance",
}


def _split_claim_segments(text):
    """Split answer text while keeping offsets in the Python string space."""
    value = str(text or "")
    boundaries = []
    start = 0
    for match in re.finditer(r"[。！？!?；;\n]", value):
        end = match.end()
        boundaries.append((start, end))
        start = end
    if start < len(value):
        boundaries.append((start, len(value)))

    segments = []
    for start, end in boundaries:
        segment = value[start:end]
        split = re.search(r"[，,](?=(?:要不要|需要我|可以|是否))", segment)
        if split:
            first_end = start + split.end()
            segments.append((start, first_end, value[start:first_end]))
            segments.append((first_end, end, value[first_end:end]))
        else:
            segments.append((start, end, segment))
    return [(start, end, segment) for start, end, segment in segments if segment.strip()]


def _is_non_claim(text):
    value = str(text or "").strip(" \t\r\n。！？!?；;，,")
    return not value or value.startswith(("要不要", "需要我", "可以继续", "谢谢", "好的", "我在听"))


def _claim_kind(text):
    value = str(text or "")
    if any(token in value for token in ("不知道", "不足以", "无法判断", "不能判断", "记录不足", "没有足够", "没有找到", "未找到")):
        return "uncertainty"
    if any(token in value for token in ("可能", "应该", "似乎", "看起来", "喜欢", "性格", "总是")):
        return "family_inference"
    return "family_fact"


def extract_claims(text, *, source="answer"):
    """Scan complete text independently of any Writer-provided candidate list."""
    claims = []
    non_claims = []
    for index, (start, end, segment) in enumerate(_split_claim_segments(text), 1):
        item = {
            "claim_id": f"extract_claim_{index}",
            "start": start,
            "end": end,
            "text": segment,
            "claim_kind": "non_claim" if _is_non_claim(segment) else _claim_kind(segment),
            "candidate_evidence_ids": [],
            "source": source,
        }
        (non_claims if item["claim_kind"] == "non_claim" else claims).append(item)
    return {"claims": claims, "non_claim_spans": non_claims, "uncovered_spans": []}


def merge_claim_candidates(text, writer_candidates, *, follow_up_text=""):
    """Merge Writer hints into a complete independent extraction."""
    extracted = extract_claims(text)
    follow_up = extract_claims(follow_up_text, source="follow_up") if follow_up_text else {"claims": [], "non_claim_spans": [], "uncovered_spans": []}
    claims = [*extracted["claims"], *follow_up["claims"]]
    candidates = list(writer_candidates or [])
    for claim in claims:
        matches = [
            candidate for candidate in candidates
            if (
                str(candidate.get("text") or "").strip()
                and (
                    str(candidate.get("text") or "").strip() in str(claim["text"] or "")
                    or str(claim["text"] or "").strip() in str(candidate.get("text") or "")
                )
            )
        ]
        if not matches:
            continue
        candidate = matches[0]
        claim["writer_candidate_ids"] = [candidate.get("claim_id")] if candidate.get("claim_id") else []
        claim["intended_type"] = candidate.get("intended_type") or claim["claim_kind"]
        claim["candidate_evidence_ids"] = list(dict.fromkeys(candidate.get("candidate_evidence_ids") or []))
    return {
        "claims": claims,
        "non_claim_spans": [*extracted["non_claim_spans"], *follow_up["non_claim_spans"]],
        "uncovered_spans": [],
    }


def _source_text(item):
    clothing = item.get("clothing") or item.get("clothing_json")
    if isinstance(clothing, (list, tuple)) and clothing:
        clothing = "、".join(
            json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
            for value in clothing
        )
    return str(
        item.get("source_text")
        or item.get("summary")
        or item.get("caption")
        or item.get("transcript")
        or item.get("value_text")
        or clothing
        or "、".join(str(value) for value in (item.get("name"), item.get("family_role")) if value)
        or ""
    ).strip()


def build_evidence_bundle(claim, evidence_items, *, derived_context=(), scope_id=None, viewer_id=None):
    """Build verifier evidence from canonical excerpts, not narrative summaries."""
    candidate_ids = set(claim.get("candidate_evidence_ids") or [])
    canonical_evidence = []
    for item in evidence_items or []:
        evidence_id = item.get("id") or item.get("evidence_id")
        if not evidence_id or evidence_id not in candidate_ids:
            continue
        if item.get("kind") not in CANONICAL_EVIDENCE_KINDS:
            continue
        source_text = _source_text(item)
        if not source_text:
            continue
        subject_ids = list(dict.fromkeys(
            item.get("subject_ids") or [value for value in (
                item.get("person_id"), item.get("subject_entity_id"), item.get("object_entity_id")
            ) if value]
        ))
        canonical_evidence.append({
            "evidence_id": evidence_id,
            "type": item.get("kind"),
            "source_text": source_text,
            "subject_ids": subject_ids,
            "time": item.get("time") or item.get("time_start") or item.get("captured_at"),
            "scope_id": item.get("scope_id") or scope_id,
            "viewer_ids": list(item.get("viewer_ids") or ([viewer_id] if viewer_id else [])),
            "asset_id": item.get("asset_id"),
        })
    derived = []
    for item in derived_context or ():
        if item.get("is_canonical") is True:
            continue
        derived.append({
            "scene_id": item.get("scene_id"),
            "text": str(item.get("text") or item.get("narrative") or ""),
            "is_canonical": False,
            "source_evidence_ids": list(item.get("source_evidence_ids") or item.get("evidence_ids") or []),
        })
    return {
        "claim_id": claim.get("claim_id"),
        "canonical_evidence": canonical_evidence,
        "derived_context": derived,
    }


def verify_claims(claims, evidence_bundles, *, scope_id=None, viewer_id=None):
    """Run the deterministic evidence gate before any semantic model check."""
    bundles = {item.get("claim_id"): item for item in evidence_bundles or []}
    results = []
    for claim in claims or []:
        bundle = bundles.get(claim.get("claim_id"), {})
        supported = []
        for evidence in bundle.get("canonical_evidence", []):
            if scope_id and evidence.get("scope_id") != scope_id:
                continue
            viewer_ids = evidence.get("viewer_ids") or []
            if viewer_id and viewer_ids and viewer_id not in viewer_ids:
                continue
            supported.append(evidence.get("evidence_id"))
        status = "reasonable_summary" if supported else "unsupported"
        if not supported and claim.get("claim_kind") == "uncertainty":
            status = "abstention"
        if claim.get("claim_kind") == "non_claim":
            status = "not_required"
        results.append({
            "claim_id": claim.get("claim_id"),
            "status": status,
            "supported_evidence_ids": list(dict.fromkeys(item for item in supported if item)),
            "reason": "存在受控 canonical evidence" if supported else "明确保留意见，不把不确定内容写成事实" if status == "abstention" else "没有明确绑定的 canonical evidence",
            "replacement": None,
            "epistemic_type": claim.get("intended_type") or claim.get("claim_kind"),
        })
    return results


def repair_answer(text, claims, verifications, *, max_repairs=1):
    """Replace only failed spans; the caller must extract and verify again."""
    failures = {
        item.get("claim_id"): item for item in verifications or ()
        if item.get("status") in {"unsupported", "overstated", "contradicted", "privacy_blocked"}
    }
    replacements = []
    for claim in claims or ():
        if claim.get("claim_id") not in failures:
            continue
        replacement = "关于这点，目前的记录不足以确定。"
        if claim.get("claim_kind") == "family_fact":
            replacement = "这部分目前没有足够记录支持。"
        replacements.append((claim.get("start"), claim.get("end"), replacement, claim.get("claim_id")))
    replacements = [item for item in replacements if isinstance(item[0], int) and isinstance(item[1], int)]
    replacements = replacements[:max(0, int(max_repairs))]
    repaired_text = str(text or "")
    for start, end, replacement, _ in reversed(replacements):
        repaired_text = repaired_text[:start] + replacement + repaired_text[end:]
    return {
        "text": repaired_text,
        "repaired_claim_ids": [item[3] for item in replacements],
        "repair_count": len(replacements),
    }


def build_text_segments(text, claims, verifications):
    """Return claim-aware text pieces so clients never slice by Python offsets."""
    value = str(text or "")
    verification_map = {item.get("claim_id"): item for item in verifications or ()}
    spans = []
    for claim in claims or ():
        start, end = claim.get("start"), claim.get("end")
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
            continue
        if value[start:end] != str(claim.get("text") or ""):
            continue
        spans.append((start, end, claim))
    spans.sort(key=lambda item: (item[0], item[1]))
    segments = []
    cursor = 0
    for start, end, claim in spans:
        if start < cursor:
            continue
        if start > cursor:
            segments.append({"type": "text", "text": value[cursor:start]})
        verification = verification_map.get(claim.get("claim_id"), {})
        segments.append({
            "type": "claim",
            "claim_id": claim.get("claim_id"),
            "text": value[start:end],
            "status": verification.get("status", "unverified"),
        })
        cursor = end
    if cursor < len(value):
        segments.append({"type": "text", "text": value[cursor:]})
    return [item for item in segments if item.get("text")]


def claim_evidence_index(claims, evidence_bundles, verifications):
    """Expose stable claim_id to evidence links for API and UI consumers."""
    bundles = {item.get("claim_id"): item for item in evidence_bundles or ()}
    statuses = {item.get("claim_id"): item for item in verifications or ()}
    return {
        claim.get("claim_id"): {
            "claim_id": claim.get("claim_id"),
            "evidence_ids": list(dict.fromkeys(
                item.get("evidence_id") for item in bundles.get(claim.get("claim_id"), {}).get("canonical_evidence", [])
                if item.get("evidence_id")
            )),
            "status": statuses.get(claim.get("claim_id"), {}).get("status", "unverified"),
        }
        for claim in claims or () if claim.get("claim_id")
    }


def resolve_memory_intensity(mode, *, proactive_enabled=False):
    """Separate zero-memory chat, lightweight probing, and concrete retrieval."""
    if mode == "chat":
        return "probe" if proactive_enabled else "none"
    if mode == "feedback" or mode == "clarify":
        return "forensic"
    return "targeted"


try:
    from pydantic import BaseModel, Field

    class TurnPlanModel(BaseModel):
        mode: str = "chat"
        tools: list[str] = Field(default_factory=list)
        show_images: bool = False
        reason: str = ""
except ImportError:  # pragma: no cover - exercised by the dependency-free test environment.
    TurnPlanModel = None


try:
    from pydantic_ai import Agent as PydanticAIAgent
except ImportError:  # pragma: no cover - the production dependency is optional during migration.
    PydanticAIAgent = None


@dataclass(frozen=True)
class PlanValidation:
    mode: str
    tools: tuple[str, ...]
    show_images: bool
    reason: str

    def as_dict(self, planner):
        return {
            "mode": self.mode,
            "tools": list(self.tools),
            "show_images": self.show_images,
            "reason": self.reason,
            "planner": planner,
        }


def _fallback_plan_values(fallback):
    return {
        "mode": fallback.get("mode", "chat"),
        "tools": list(fallback.get("tools", [])),
        "show_images": bool(fallback.get("show_images")),
        "reason": str(fallback.get("reason") or ""),
    }


def validate_turn_plan(parsed, fallback):
    """Validate a model/framework plan without allowing permission escalation."""
    fallback_values = _fallback_plan_values(fallback)
    parsed = parsed if isinstance(parsed, dict) else {}
    mode = parsed.get("mode") if parsed.get("mode") in ALLOWED_MODES else fallback_values["mode"]
    if fallback_values["mode"] in {"memory", "feedback", "clarify"} and mode == "chat":
        mode = fallback_values["mode"]
    tools = [item for item in parsed.get("tools", []) if item in ALLOWED_TOOLS]
    if mode == "chat":
        tools = []
    elif mode == "memory":
        tools = list(dict.fromkeys(["resolve_constraints", *tools]))
        if len(tools) == 1:
            tools.append("find_events")
    elif mode == "feedback":
        tools = ["record_feedback"]
    else:
        tools = list(dict.fromkeys(["resolve_constraints", *tools, "request_clarification"]))
    show_images = bool(parsed.get("show_images")) and mode == "memory"
    if fallback_values["show_images"]:
        show_images = True
    return PlanValidation(
        mode=mode,
        tools=tuple(tools),
        show_images=show_images,
        reason=str(parsed.get("reason") or fallback_values["reason"])[:240],
    )


class PydanticAIPlanner:
    """Optional structured planner; never receives or owns the memory store."""

    def __init__(self, model=None):
        self.model = model or os.getenv("SENTRIX_AGENT_FRAMEWORK_MODEL")
        self._agent = None
        if PydanticAIAgent and TurnPlanModel and self.model:
            try:
                self._agent = PydanticAIAgent(self.model, output_type=TurnPlanModel)
            except Exception:
                self._agent = None

    @property
    def available(self):
        return self._agent is not None

    def plan(self, prompt):
        if not self._agent:
            return None
        try:
            result = self._agent.run_sync(prompt)
            output = getattr(result, "output", result)
            if hasattr(output, "model_dump"):
                return output.model_dump()
            return dict(output) if isinstance(output, dict) else None
        except Exception:
            return None
