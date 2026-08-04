#!/usr/bin/env python3
"""Replay Agent behavior against a copied real-memory SQLite database."""

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.agent import MemoryAgent
from backend.db import MemoryStore
from backend.model_clients import GammaClient


class DeterministicGamma:
    model = "c1-deterministic-fallback"

    def answer(self, query, context):
        return {"answer": "证据不足", "confidence": 0.0, "evidence": [], "insufficient_evidence": True}

    def embed_text(self, text):
        return []


CASES = (
    {"id": "person_introduction", "query": "介绍一下明哥", "kind": "single"},
    {"id": "clothing", "query": "明哥穿什么颜色的衣服", "kind": "single"},
    {"id": "personality_boundary", "query": "明哥性格怎样", "kind": "single"},
    {"id": "relationship_boundary", "query": "明哥和我的关系", "kind": "single"},
    {"id": "preference_boundary", "query": "明哥喜欢吃什么", "kind": "single"},
    {"id": "original_evidence", "query": "请直接给我相关的原始照片", "kind": "single"},
    {"id": "role_ambiguity", "query": "介绍一下孩子", "kind": "single"},
    {"id": "no_evidence", "query": "火星上的生日回忆", "kind": "single"},
    {"id": "follow_up", "query": "然后呢？", "kind": "follow_up"},
    {"id": "scope_switch", "query": "然后呢？", "kind": "scope_switch"},
)


def _claim_summary(result):
    verifications = result.get("claim_verifications") or []
    statuses = [item.get("status") for item in verifications]
    family_claims = [item for item in statuses if item not in {"not_required"}]
    supported = sum(status in {"reasonable_summary", "abstention"} for status in family_claims)
    return {
        "claim_count": len(result.get("claims") or []),
        "verification_count": len(verifications),
        "supported_or_abstained": supported,
        "unsupported_count": sum(status == "unsupported" for status in statuses),
        "verification_status": result.get("claim_verification_status"),
        "repair_count": result.get("repair_count", 0),
        "evidence_bundle_count": len(result.get("evidence_bundles") or []),
        "claim_evidence_count": len(result.get("claim_evidence_index") or {}),
    }


def _result_record(case_id, query, result, elapsed_ms):
    return {
        "case_id": case_id,
        "query": query,
        "elapsed_ms": round(elapsed_ms, 2),
        "answer": result.get("answer", ""),
        "model": result.get("model"),
        "intent": result.get("intent"),
        "dialogue_plan": result.get("dialogue_plan"),
        "memory_intensity": result.get("memory_intensity"),
        "memory_used": result.get("memory_used"),
        "evidence_status": result.get("evidence_status"),
        "evidence_count": len(result.get("evidence") or []),
        "original_evidence_requested": bool(result.get("original_evidence_requested")),
        "image_count": len(result.get("image_results") or []),
        "clarification_candidates": result.get("clarification_candidates") or [],
        "claims": result.get("claims") or [],
        "claim_verifications": result.get("claim_verifications") or [],
        "evidence_bundles": result.get("evidence_bundles") or [],
        "claim_summary": _claim_summary(result),
    }


def _run_turn(agent, case, conversation_id, scope_id):
    started = time.perf_counter()
    result = agent.answer_turn(case["query"], conversation_id, scope_id=scope_id)
    return result, _result_record(case["id"], case["query"], result, (time.perf_counter() - started) * 1000)


def evaluate(database, repeats=3, deterministic=False, case_ids=None):
    source = Path(database).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    selected_cases = [case for case in CASES if not case_ids or case["id"] in set(case_ids)]
    all_runs = []
    for repeat in range(1, max(1, int(repeats)) + 1):
        with tempfile.TemporaryDirectory(prefix="sentrix-agent-c1-") as directory:
            copied = Path(directory) / "evaluation.db"
            shutil.copy2(source, copied)
            store = MemoryStore(str(copied))
            gamma = DeterministicGamma() if deterministic else GammaClient()
            agent = MemoryAgent(store, gamma=gamma)
            records = []
            try:
                for case in selected_cases:
                    if case["kind"] == "follow_up":
                        intro = {"id": "follow_up_intro", "query": "介绍一下明哥"}
                        _run_turn(agent, intro, f"c1-follow-up-{repeat}", "album2")
                        _, record = _run_turn(agent, case, f"c1-follow-up-{repeat}", "album2")
                    elif case["kind"] == "scope_switch":
                        intro = {"id": "scope_intro", "query": "介绍一下明哥"}
                        _run_turn(agent, intro, f"c1-scope-{repeat}", "album2")
                        _, record = _run_turn(agent, case, f"c1-scope-{repeat}", "album1")
                    else:
                        _, record = _run_turn(agent, case, f"c1-{case['id']}-{repeat}", "album2")
                    record["repeat"] = repeat
                    records.append(record)
            finally:
                store.close()
            all_runs.append({"repeat": repeat, "records": records})

    failures = []
    for run in all_runs:
        for record in run["records"]:
            cid = record["case_id"]
            if cid in {"person_introduction", "clothing"} and not record["memory_used"]:
                failures.append({"case_id": cid, "repeat": run["repeat"], "reason": "memory_not_used"})
            if cid in {"personality_boundary", "relationship_boundary", "preference_boundary"}:
                if "检索到" in record["answer"] or not record["answer"]:
                    failures.append({"case_id": cid, "repeat": run["repeat"], "reason": "unrelated_event_listing"})
            if cid == "original_evidence" and not record["original_evidence_requested"]:
                failures.append({"case_id": cid, "repeat": run["repeat"], "reason": "original_evidence_not_marked"})
            if cid == "role_ambiguity":
                names = {item.get("name") for item in record["clarification_candidates"]}
                if not {"明哥", "我"}.issubset(names) or "cluster_" in record["answer"]:
                    failures.append({"case_id": cid, "repeat": run["repeat"], "reason": "ambiguous_identity_not_human_readable"})
            if cid == "follow_up" and record["dialogue_plan"] and record["dialogue_plan"].get("mode") != "contextual_follow_up":
                failures.append({"case_id": cid, "repeat": run["repeat"], "reason": "follow_up_not_contextual"})
            if cid == "scope_switch" and record["dialogue_plan"] and record["dialogue_plan"].get("mode") == "contextual_follow_up":
                failures.append({"case_id": cid, "repeat": run["repeat"], "reason": "cross_scope_reused_context"})
            if record["claim_summary"]["unsupported_count"]:
                failures.append({"case_id": cid, "repeat": run["repeat"], "reason": "unsupported_claim"})

    return {
        "database": str(source),
        "readonly_source": True,
        "repeats": max(1, int(repeats)),
        "deterministic_gamma": bool(deterministic),
        "case_count": len(selected_cases),
        "passed": not failures,
        "failures": failures,
        "runs": all_runs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--deterministic", action="store_true", help="Use local fallback instead of calling Gamma")
    parser.add_argument("--case", action="append", dest="case_ids", help="Restrict replay to one or more case ids")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(args.database, args.repeats, args.deterministic, args.case_ids)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
