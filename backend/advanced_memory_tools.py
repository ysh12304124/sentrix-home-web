"""Phase 7 — advanced memory tools.

Each tool takes an ``EvidencePacket`` (built by the Evidence Retrieval Kernel)
and returns a natural-language answer whose every claim can be traced back to
canonical evidence.  The tools compose the Phase 4 Writer/Verifier chain via
:class:`backend.complex_answer.ComplexAnswerBuilder`.

Plan §6/§7 invariants preserved:

- ``summarize_person`` refuses to invent personality, preference or family
  role from co-occurrence.
- ``trace_timeline`` never returns an Asset outside the QuerySpec time hard
  constraint.
- ``compare_memories`` reports each side independently — no fact bleed.
- ``build_pattern`` requires at least two distinct source events and refuses
  to promote the pattern to a confirmed fact.
"""

from __future__ import annotations

from datetime import datetime

from .complex_answer import ComplexAnswerBuilder
from .query_contracts import Constraint


def _tool_trace(tool, status, extra=None):
    trace = {"tool": tool, "permission": "read", "status": status}
    if extra:
        trace.update(extra)
    return trace


def summarize_person(spec, packet, gamma):
    """Return a canonical-evidence-backed person summary."""
    if not spec.entity_ids:
        return {"answer": "目前没有找到当前 scope 已确认的人物。",
                "statements": [], "claim_evidence_index": {}, "evidence_ids": [],
                "tool_trace": [_tool_trace("summarize_person", "requires_anchor")],
                "unknowns": ["confirmed_entity"]}
    builder = ComplexAnswerBuilder(gamma=gamma)
    result = builder.build(spec.answer_target, spec, packet)
    return {
        "answer": result["answer"], "statements": result["statements"],
        "claim_evidence_index": _index_from_statements(result["statements"]),
        "evidence_ids": _evidence_ids(packet),
        "tool_trace": [_tool_trace("summarize_person",
                                    "fallback" if result.get("fallback") else "complete",
                                    {"repair_count": 0})],
        "unknowns": result.get("unknowns", []),
    }


def summarize_event(spec, packet, gamma):
    """Summarise a specific Event using canonical Observation excerpts."""
    if not packet.assets:
        return {"answer": "没有找到相关事件的可靠证据。",
                "statements": [], "claim_evidence_index": {}, "evidence_ids": [],
                "tool_trace": [_tool_trace("summarize_event", "no_evidence")],
                "unknowns": []}
    builder = ComplexAnswerBuilder(gamma=gamma)
    result = builder.build(spec.answer_target, spec, packet)
    return {
        "answer": result["answer"], "statements": result["statements"],
        "claim_evidence_index": _index_from_statements(result["statements"]),
        "evidence_ids": _evidence_ids(packet),
        "tool_trace": [_tool_trace("summarize_event",
                                    "fallback" if result.get("fallback") else "complete")],
        "unknowns": result.get("unknowns", []),
    }


def trace_timeline(spec, packet, gamma):
    """Return Assets sorted by ``captured_at`` inside the hard time bound."""
    hard_time = _hard_time_bound(spec)
    filtered = []
    for asset in packet.assets:
        captured = _parse_dt(asset.get("captured_at"))
        if hard_time and captured is not None:
            start, end = hard_time
            if not (start <= captured < end):
                continue  # never violate the hard time constraint
        filtered.append(asset)
    filtered.sort(key=lambda item: item.get("captured_at") or "")
    lines = [f"{index + 1}. {asset.get('captured_at') or '未知时间'}：{asset.get('file_name') or asset.get('asset_id')}"
             for index, asset in enumerate(filtered[:20])]
    answer = "按时间排序的记录如下：\n" + "\n".join(lines) if lines else "在给定时间范围内没有找到可靠记录。"
    return {"answer": answer,
            "statements": [{"text": line, "status": "matched",
                             "evidence_ids": asset.get("evidence_ids", [])}
                            for asset, line in zip(filtered, lines)],
            "claim_evidence_index": {}, "evidence_ids": _evidence_ids_from_list(filtered),
            "tool_trace": [_tool_trace("trace_timeline", "complete",
                                         {"kept": len(filtered), "dropped": len(packet.assets) - len(filtered)})],
            "unknowns": []}


def compare_memories(spec_a, packet_a, spec_b, packet_b, gamma):
    """Compare two evidence packets independently — no cross-support."""
    def _summary(spec, packet):
        return {"scope_id": spec.scope_id, "asset_count": len(packet.assets),
                "exact": len(packet.exact_results),
                "approximate": len(packet.approximate_results),
                "constraints": [c.key for c in spec.constraints]}
    side_a = _summary(spec_a, packet_a)
    side_b = _summary(spec_b, packet_b)
    return {
        "answer": ("对比两组记录：\n" +
                    f"- 集合 A（{side_a['scope_id']}）共 {side_a['asset_count']} 条，"
                    f"精确 {side_a['exact']} / 近似 {side_a['approximate']}。\n" +
                    f"- 集合 B（{side_b['scope_id']}）共 {side_b['asset_count']} 条，"
                    f"精确 {side_b['exact']} / 近似 {side_b['approximate']}。"),
        "statements": [],  # each side's conclusions must be verified per side, not merged
        "claim_evidence_index": {},
        "evidence_ids": _evidence_ids(packet_a) + _evidence_ids(packet_b),
        "tool_trace": [_tool_trace("compare_memories", "complete",
                                     {"side_a_evidence_count": side_a["asset_count"],
                                      "side_b_evidence_count": side_b["asset_count"]})],
        "unknowns": [],
    }


def build_pattern(spec, packet, gamma, min_events=2):
    """Return a soft pattern only when the packet spans ≥ ``min_events`` events."""
    event_ids = set()
    for asset in packet.assets:
        for evidence_id in asset.get("evidence_ids") or []:
            if evidence_id.startswith("event") or evidence_id.startswith("obs"):
                event_ids.add(evidence_id)
    if len(event_ids) < min_events:
        return {"answer": "尚不足两次独立事件，无法归纳为模式。",
                "statements": [], "claim_evidence_index": {}, "evidence_ids": [],
                "tool_trace": [_tool_trace("build_pattern", "requires_more_events",
                                             {"observed": len(event_ids), "min_required": min_events})],
                "unknowns": ["insufficient_events"]}
    return {"answer": f"在 {len(event_ids)} 次独立事件中出现了一致的记录，但这只是柔性模式，不是确认事实。",
            "statements": [],
            "claim_evidence_index": {},
            "evidence_ids": _evidence_ids(packet),
            "tool_trace": [_tool_trace("build_pattern", "soft_pattern",
                                         {"event_count": len(event_ids)})],
            "unknowns": []}


def _hard_time_bound(spec):
    for constraint in spec.constraints:
        if constraint.dimension == "time" and constraint.strictness == "deterministic_hard":
            from .query_contracts import parse_time_expression
            bounds = parse_time_expression(constraint.value)
            if bounds:
                return bounds
    return None


def _parse_dt(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except (TypeError, ValueError):
        return None


def _index_from_statements(statements):
    return {f"claim_{index + 1}": {"evidence_ids": item.get("evidence_ids", []),
                                    "status": item.get("status", "unknown")}
            for index, item in enumerate(statements or [])}


def _evidence_ids(packet):
    ids = []
    for asset in packet.assets:
        ids.extend(asset.get("evidence_ids") or [])
    return ids


def _evidence_ids_from_list(assets):
    ids = []
    for asset in assets:
        ids.extend(asset.get("evidence_ids") or [])
    return ids
