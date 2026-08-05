"""Read-only contract evaluation for the evidence-backed memory steward."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.agent import MemoryAgent
from backend.db import MemoryStore


class RefusingGamma:
    model = "steward-evaluation"

    def answer(self, query, context):
        return {"answer": "证据不足", "confidence": 0.0, "evidence": [], "insufficient_evidence": True}

    def embed_text(self, text):
        return []


def _result_contract(result, expected_tools=(), require_evidence=False, follow_up=False):
    trace = result.get("tool_trace") or []
    tool_names = [item.get("tool") for item in trace]
    ordered = result.get("evidence_order") or []
    passed = bool(result.get("dialogue_plan")) and all(tool in tool_names for tool in expected_tools)
    memory_used = bool(result.get("memory_used"))
    evidence_required = bool(result.get("evidence_required"))
    if require_evidence:
        passed = passed and memory_used and evidence_required and result.get("evidence_status") == "anchored"
    if require_evidence:
        passed = passed and bool(result.get("evidence")) and bool(ordered)
    if follow_up:
        passed = passed and result.get("dialogue_plan", {}).get("mode") == "contextual_follow_up"
    return {
        "passed": passed,
        "mode": result.get("dialogue_plan", {}).get("mode"),
        "style": result.get("dialogue_plan", {}).get("style"),
        "tools": tool_names,
        "evidence_count": len(result.get("evidence") or []),
        "ordered_evidence": len(ordered),
        "insufficient_evidence": bool(result.get("insufficient_evidence")),
        "memory_used": memory_used,
        "evidence_required": evidence_required,
        "evidence_status": result.get("evidence_status"),
        "original_evidence_requested": bool(result.get("original_evidence_requested")),
        "image_count": len(result.get("image_results") or []),
    }


def evaluate(database):
    store = MemoryStore(str(database))
    try:
        agent = MemoryAgent(store, gamma=RefusingGamma())
        scope_id = "steward-evaluation"
        person = store.create_entity("妈妈", "person", "confirmed", confidence=1.0, scope_id=scope_id)
        asset = store.create_asset("steward_asset", "family.jpg", "image", "/tmp/family.jpg", scope_id=scope_id)
        observation = store.add_observation(asset["id"], {"caption": "妈妈在餐桌旁切蛋糕", "captured_at": "2025-05-01T10:00:00+00:00"})
        event = store.merge_observation_into_event(observation)
        store.upsert_event_participant(event["id"], person["id"], "visible_subject", [observation["id"]], 0.9)

        introduction = agent.answer_turn("介绍一下妈妈", "steward-introduction", scope_id=scope_id)
        follow_up = agent.answer_turn("她后来呢？", "steward-introduction", scope_id=scope_id)
        original = agent.answer_turn("请直接给我那次的原始照片", "steward-introduction", scope_id=scope_id)
        recommendation = agent.answer_turn("推荐一些妈妈的回忆", "steward-recommendation", scope_id=scope_id)
        unanchored = agent.answer_turn("推荐一些回忆", "steward-unanchored", scope_id=scope_id)
        chat = agent.answer_turn("今天有点累，想聊聊天", "steward-chat", scope_id=scope_id)
        ambiguous_store = MemoryStore(":memory:")
        try:
            ambiguous_store.create_entity("东湖边", "place", confidence=0.8, scope_id="ambiguous")
            ambiguous_store.create_entity("西湖边", "place", confidence=0.8, scope_id="ambiguous")
            ambiguous = MemoryAgent(ambiguous_store, gamma=RefusingGamma()).answer_turn("湖边在哪里", scope_id="ambiguous")
        finally:
            ambiguous_store.close()
        checks = {
            "introduction": _result_contract(introduction, ("resolve_constraints", "describe_entity", "open_evidence"), True),
            "follow_up": _result_contract(follow_up, ("resolve_constraints", "trace_timeline", "open_evidence"), True, True),
            "original_evidence": _result_contract(original, ("resolve_constraints", "trace_timeline", "open_evidence"), True),
            "recommendation": _result_contract(recommendation, ("resolve_constraints", "suggest_recall", "open_evidence"), True),
            "unanchored_recommendation": _result_contract(unanchored, ("resolve_constraints", "suggest_recall", "request_clarification")),
            "ordinary_chat": _result_contract(chat),
            "ambiguity": _result_contract(ambiguous, ("resolve_constraints", "request_clarification")),
        }
        checks["original_evidence"]["passed"] = checks["original_evidence"]["passed"] and original.get("original_evidence_requested") and bool(original.get("image_results"))
        checks["unanchored_recommendation"]["passed"] = checks["unanchored_recommendation"]["passed"] and checks["unanchored_recommendation"]["insufficient_evidence"]
        checks["unanchored_recommendation"]["passed"] = checks["unanchored_recommendation"]["passed"] and unanchored.get("evidence_status") == "gap" and bool((unanchored.get("evidence_layers") or {}).get("gaps"))
        checks["ordinary_chat"]["passed"] = checks["ordinary_chat"]["passed"] and not chat.get("memory_used") and not chat.get("evidence_required") and not chat.get("evidence")
        checks["ambiguity"]["passed"] = checks["ambiguity"]["passed"] and bool(ambiguous.get("clarification_candidates"))
        return {"passed": all(item["passed"] for item in checks.values()), "checks": checks}
    finally:
        store.close()


def main():
    parser = argparse.ArgumentParser(description="Run read-only memory steward dialogue contract checks.")
    parser.parse_args()
    # The contract fixture must never write to a household or production SQLite file.
    print(json.dumps(evaluate(":memory:"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
