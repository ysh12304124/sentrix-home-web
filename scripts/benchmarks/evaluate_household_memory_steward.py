#!/usr/bin/env python3
"""Run read-only dialogue acceptance checks against a copied household database."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.agent import MemoryAgent
from backend.db import MemoryStore


class EvidenceOnlyGamma:
    model = "household-steward-evaluation"

    def answer(self, query, context):
        return {"answer": "证据不足", "confidence": 0.0, "evidence": [], "insufficient_evidence": True}

    def embed_text(self, text):
        return []


def _check(result, tools=(), evidence=False, mode=None):
    trace = result.get("tool_trace") or []
    names = [item.get("tool") for item in trace]
    passed = all(tool in names for tool in tools)
    if evidence:
        passed = passed and bool(result.get("memory_used")) and bool(result.get("evidence_required"))
        passed = passed and result.get("evidence_status") == "anchored"
        passed = passed and bool(result.get("evidence")) and bool(result.get("evidence_order"))
    if mode:
        passed = passed and result.get("dialogue_plan", {}).get("mode") == mode
    return {
        "passed": passed,
        "mode": result.get("dialogue_plan", {}).get("mode"),
        "style": result.get("dialogue_plan", {}).get("style"),
        "tools": names,
        "evidence_count": len(result.get("evidence") or []),
        "ordered_evidence": len(result.get("evidence_order") or []),
        "insufficient_evidence": bool(result.get("insufficient_evidence")),
        "memory_used": bool(result.get("memory_used")),
        "evidence_required": bool(result.get("evidence_required")),
        "evidence_status": result.get("evidence_status"),
        "original_evidence_requested": bool(result.get("original_evidence_requested")),
        "image_count": len(result.get("image_results") or []),
    }


def evaluate(database_path):
    """Copy ``database_path`` before creating dialogue states or query gaps."""
    source = Path(database_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with tempfile.TemporaryDirectory(prefix="sentrix-household-steward-") as directory:
        copied = Path(directory) / "evaluation.db"
        shutil.copy2(source, copied)
        store = MemoryStore(str(copied))
        try:
            people = [item for item in store.list_entities(status="confirmed", scope_id="album2") if item.get("entity_type") == "person"]
            if not people:
                return {"passed": False, "reason": "album2 has no confirmed people for a real-memory dialogue check", "checks": {}}
            first = people[0]
            agent = MemoryAgent(store, gamma=EvidenceOnlyGamma())
            conversation = "household-steward-evaluation"
            introduction = agent.answer_turn(f"介绍一下{first['canonical_name']}", conversation, scope_id="album2")
            follow_up = agent.answer_turn("然后呢？", conversation, scope_id="album2")
            recommendation = agent.answer_turn(f"推荐一些{first['canonical_name']}的回忆", "household-steward-recall", scope_id="album2")
            unanchored = agent.answer_turn("推荐一些回忆", "household-steward-unanchored", scope_id="album2")
            cross_scope = agent.answer_turn("然后呢？", conversation, scope_id="album1")
            checks = {
                "introduction": _check(introduction, ("resolve_constraints", "describe_entity", "open_evidence"), True),
                "follow_up": _check(follow_up, ("resolve_constraints", "trace_timeline", "open_evidence"), True, "contextual_follow_up"),
                "recommendation": _check(recommendation, ("resolve_constraints", "suggest_recall", "open_evidence"), True),
                "unanchored_recommendation": _check(unanchored, ("resolve_constraints", "suggest_recall", "request_clarification")),
                "cross_scope_isolation": _check(cross_scope),
            }
            checks["unanchored_recommendation"]["passed"] = checks["unanchored_recommendation"]["passed"] and checks["unanchored_recommendation"]["insufficient_evidence"]
            checks["cross_scope_isolation"]["passed"] = checks["cross_scope_isolation"]["passed"] and checks["cross_scope_isolation"]["mode"] != "contextual_follow_up"
            return {
                "passed": all(item["passed"] for item in checks.values()),
                "scope_id": "album2", "person_id": first["id"],
                "checks": checks,
            }
        finally:
            store.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate the real household memory steward without writing to its source database.")
    parser.add_argument("database", type=Path, help="Source SQLite database; it is copied before evaluation.")
    args = parser.parse_args()
    result = evaluate(args.database)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result.get("passed") else 1)


if __name__ == "__main__":
    main()
