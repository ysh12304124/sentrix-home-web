"""Response Consistency Validator (RX-5).

Deterministic post-generation checks over the answer, the AnswerBrief, the
ResponsePlan and the actual image count.  Violations are repaired locally at
most once; a still-invalid answer is replaced by the target-specific safe
fallback — never by a database report.

``scan_internal_leak`` is reused by the offline audit and the E2E scorer so a
single rule set guards every surface.
"""

from __future__ import annotations

import re

from .answer_brief import AnswerBrief
from .response_plan import ResponsePlan

_INTERNAL_ID_RE = re.compile(r"\b(?:asset|obs|event|entity|claim)_[A-Za-z0-9]{4,}")
_ENGLISH_LABEL_RE = re.compile(r"\b(matched|possible|unknown)\b", re.IGNORECASE)
_TABLE_NAME_RE = re.compile(r"\b(?:assets|observations|events|conditions|memory_vectors)\b", re.IGNORECASE)
_TRACE_TERM_RE = re.compile(r"(fusion_score|retrieval_trace|condition_key|recall_strength|source_type)")
_TEMPLATE_RE = re.compile(r"根据本地事件记忆检索到|本地事件记忆检索")
_UNABLE_RE = re.compile(r"无法提供|不能提供|给不了|无法展示|不能展示")
_DELIVERY_RE = re.compile(r"已找到并展示|已经展示|已展示|已给出")
_IMAGE_COUNT_RE = re.compile(r"(\d+)\s*张")
_FAMILY_PATTERN_RE = re.compile(r"多次出现|常常|经常|总是|喜欢|性格")
_DISCLOSURE_RE = re.compile(r"接近|不完全匹配|不能确认|不确定|没有完全匹配|无法确认|尚未确认|还不能确认|仅凭")
_FINDING_CLAIM_RE = re.compile(r"(?:我找到了|已找到|找到了\s*|找到这些|发现(?:了)?\s*(?:照片|图片|记录))")


def scan_internal_leak(text: str) -> list[str]:
    """Return every internal-token leak found in user-visible text."""
    value = str(text or "")
    hits: list[str] = []
    for pattern in (_INTERNAL_ID_RE, _ENGLISH_LABEL_RE, _TABLE_NAME_RE, _TRACE_TERM_RE, _TEMPLATE_RE):
        match = pattern.search(value)
        if match:
            hits.append(match.group(0))
    return list(dict.fromkeys(hits))


def _fact_by_id(brief: AnswerBrief):
    return {f.fact_id: f for f in brief.facts}


def _validate_statements(brief, statements) -> list[str]:
    failures = []
    facts = _fact_by_id(brief)
    for statement in statements or []:
        fact_id = statement.get("fact_id")
        certainty = statement.get("certainty")
        if not fact_id:
            continue
        fact = facts.get(fact_id)
        if fact is None:
            failures.append(f"unknown_fact_id:{fact_id}")
            continue
        if certainty == "confirmed" and fact.certainty == "possible":
            failures.append(f"overstated:{fact_id}")
    return failures


def validate_response(answer, brief: AnswerBrief, plan: ResponsePlan,
                      image_count: int, statements=None) -> dict:
    text = str(answer or "")
    failures: list[dict] = []
    reasons: list[str] = []

    leak = scan_internal_leak(text)
    if leak:
        failures.append({"rule": "internal_leak", "detail": "、".join(leak)})
        reasons.append("internal_leak")

    # must_not_say
    for banned in brief.must_not_say:
        if banned and banned in text:
            failures.append({"rule": "must_not_say", "detail": banned})
            reasons.append("must_not_say")

    # statements overstate or reference unknown facts
    for issue in _validate_statements(brief, statements):
        failures.append({"rule": "fact_consistency", "detail": issue})
        reasons.append("fact_consistency")

    # image consistency
    if image_count > 0 and _UNABLE_RE.search(text):
        failures.append({"rule": "cannot_provide_but_images", "detail": "正文说无法提供但图片非空"})
        reasons.append("image_contradiction")
    if image_count == 0 and _DELIVERY_RE.search(text) and brief.response_mode in {"asset_delivery", "exact_result", "approximate_result"}:
        failures.append({"rule": "claims_delivered_but_no_images", "detail": "正文说已展示但图片为空"})
        reasons.append("image_contradiction")
    count_matches = _IMAGE_COUNT_RE.findall(text)
    if count_matches and image_count > 0:
        if all(int(value) != image_count for value in count_matches):
            failures.append({"rule": "image_count_mismatch", "detail": f"正文数量{count_matches} vs 实际{image_count}"})
            reasons.append("image_count_mismatch")

    # mode consistency
    if brief.response_mode == "no_result" and image_count != 0:
        failures.append({"rule": "no_result_shows_images", "detail": f"no_result 但图片 {image_count}"})
        reasons.append("mode_consistency")
    if brief.response_mode == "asset_delivery" and image_count == 0:
        failures.append({"rule": "asset_delivery_no_images", "detail": "asset_delivery 但图片为空"})
        reasons.append("mode_consistency")
    if brief.response_mode == "approximate_result" and not _DISCLOSURE_RE.search(text):
        failures.append({"rule": "approximate_disclosure_missing", "detail": "近似结果未说明差异"})
        reasons.append("mode_consistency")
    if brief.response_mode == "person_summary" and not brief.facts and _FAMILY_PATTERN_RE.search(text):
        failures.append({"rule": "person_no_evidence_claim", "detail": "证据不足仍输出家庭主张"})
        reasons.append("mode_consistency")
    if brief.response_mode in {"no_result", "person_summary"} and not brief.facts \
            and _FINDING_CLAIM_RE.search(text):
        failures.append({"rule": "finding_claim_without_facts", "detail": "无事实依据却宣称找到"})
        reasons.append("mode_consistency")

    return {"valid": not failures, "failures": failures,
            "reasons": list(dict.fromkeys(reasons))}


def repair_response_once(answer, brief: AnswerBrief, plan: ResponsePlan,
                         image_count: int, result: dict) -> str | None:
    """Local repair for the violating sentences; None when unrecoverable."""
    text = str(answer or "")
    changed = False
    if any(f["rule"] == "internal_leak" for f in result["failures"]):
        text = _INTERNAL_ID_RE.sub("", text)
        text = _ENGLISH_LABEL_RE.sub("", text)
        text = _TABLE_NAME_RE.sub("", text)
        text = _TRACE_TERM_RE.sub("", text)
        text = _TEMPLATE_RE.sub("", text)
        text = re.sub(r"\(\s*\)|（\s*）", "", text)
        text = re.sub(r"\s{2,}", " ", text).strip(" ，。；、")
        changed = True
    if image_count > 0 and _UNABLE_RE.search(text):
        replacement = "这些照片已经找到，可以查看。" if brief.response_mode == "asset_delivery" else "这几张照片比较接近，可以查看。"
        text = _UNABLE_RE.sub(replacement, text)
        changed = True
    if not text.strip():
        return None
    return text if changed else text


def finalize_answer(answer, statements, brief: AnswerBrief, plan: ResponsePlan,
                    image_count: int, fallback_fn) -> tuple[str, list, dict]:
    """Validate, repair once, then fall back; return (answer, statements, result).

    The returned result dict carries ``fallback_used`` so E2E can distinguish a
    natural model answer from a deterministic safe fallback.
    """
    result = validate_response(answer, brief, plan, image_count, statements)
    if result["valid"]:
        result["fallback_used"] = False
        return answer, statements, result
    repaired = repair_response_once(answer, brief, plan, image_count, result)
    if repaired is not None:
        second = validate_response(repaired, brief, plan, image_count, statements)
        if second["valid"]:
            second["fallback_used"] = False
            return repaired, statements, second
    fallback_answer, fallback_statements = fallback_fn()
    fallback_result = validate_response(fallback_answer, brief, plan, image_count, fallback_statements)
    fallback_result["fallback_used"] = True
    return fallback_answer, fallback_statements, fallback_result
