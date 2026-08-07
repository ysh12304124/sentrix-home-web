#!/usr/bin/env python3
"""TFPE v2: Structured Memory QA E2E.

Hits a validation instance (8092) with the structured QA set and asserts:
  - the model-judged route chose a structured strategy (no visual ANN),
  - the answer is exact (equals the deterministic executor's SQL ground truth),
  - zero images unless the query asked for assets,
  - no internal leak.

Ground truth is computed by running StructuredMemoryExecutor directly against
the same DB with the case's reference filter — never by re-asking the model.

Run on 153 (8092 RX + structured flags up, admin debug on):
  PYTHONPATH=. .venv/bin/python scripts/benchmarks/evaluate_structured_memory_qa.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from backend.db import MemoryStore
from backend.query_contracts import Constraint, HARD, QueryParseDraft, QuerySpec
from backend.response_validator import scan_internal_leak
from backend.structured_memory import StructuredMemoryExecutor

DB_PATH = REPO_ROOT / "data" / "sentrix.db"
CASES_PATH = REPO_ROOT / "scripts" / "benchmarks" / "structured_qa_cases.json"


def _resolve_entity(store, name, scope):
    row = store._row(
        "SELECT id FROM entities WHERE entity_type='person' AND status='confirmed' "
        "AND canonical_name=? AND scope_id=? LIMIT 1", (name, scope))
    if not row:
        row = store._row(
            "SELECT id FROM entities WHERE entity_type='person' AND status='confirmed' "
            "AND canonical_name=? LIMIT 1", (name,))
    return row["id"] if row else None


def _reference_draft_and_spec(case):
    draft = QueryParseDraft(answer_type=case["answer_type"],
                            structured=case.get("reference") or {})
    scope = case["scope"]
    spec = QuerySpec("ref", "single", [scope], "owner", "c", "answer", "general", constraints=[])
    entity = case.get("entity")
    if entity:
        entity_id = _resolve_entity(store, entity, scope)
        if entity_id:
            spec.entity_ids = [entity_id]
            spec.constraints.append(Constraint("person", entity, HARD, "confirmed_bridge"))
    return draft, spec


def _rows_equal(expected_rows, actual_rows):
    if actual_rows is None:
        return False
    expected = {(str(r["group"]), int(r.get("count", 0))) for r in expected_rows}
    actual = {(str(r["group"]), int(r.get("count", 0))) for r in actual_rows}
    return expected == actual


def _values_equal(answer_type, expected, actual) -> bool:
    if answer_type in {"count", "exists", "boolean"}:
        return expected.total == actual.get("total")
    if answer_type in {"first_occurrence", "last_occurrence", "date"}:
        return (expected.value or None) == (actual.get("value") or None)
    if answer_type == "date_range":
        return (expected.value or {}).get("first") == (actual.get("value") or {}).get("first") and \
               (expected.value or {}).get("last") == (actual.get("value") or {}).get("last")
    if answer_type in {"grouped_list", "list"}:
        return _rows_equal(expected.rows, actual.get("rows") or [])
    return False


def _answer_mentions_value(answer_type, expected, answer) -> bool:
    if answer_type == "count":
        return str(expected.total) in str(answer or "")
    if answer_type in {"exists", "boolean"}:
        text = str(answer or "")
        return ("有" in text or "存在" in text or "没有" in text or "无" in text)
    if answer_type in {"first_occurrence", "last_occurrence", "date"}:
        value = str(expected.value or "")
        return bool(value) and value[:10] in str(answer or "")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default=os.getenv("SENTRIX_API_URL", "http://127.0.0.1:8092"))
    parser.add_argument("--db", default=str(DB_PATH))
    args = parser.parse_args()

    global store
    store = MemoryStore(args.db)

    import httpx
    client = httpx.Client(timeout=300)
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]

    per_case = []
    totals = {"routed": 0, "exact": 0, "no_visual": 0, "no_images": 0, "no_leak": 0, "mention": 0}
    for case in cases:
        name = case["query"]
        try:
            body = client.post(f"{args.api}/api/assistant/turn",
                               json={"message": name, "scope_id": case["scope"]}).json()
        except Exception as exc:
            per_case.append({"query": name, "verdict": "FAIL", "reasons": [f"http_error:{exc}"], "answer": ""})
            continue
        answer = str(body.get("answer") or "")
        mode = body.get("response_mode") or ""
        strategy = (body.get("retrieval_strategy") or {}).get("chosen_strategy") or ""
        skipped = (body.get("retrieval_strategy") or {}).get("skipped_channels") or []
        actual = body.get("structured_result") or {}

        ref_draft, ref_spec = _reference_draft_and_spec(case)
        expected = StructuredMemoryExecutor(store).execute(ref_draft, ref_spec,
                                                           strategy=case["expected_strategy"])
        reasons = []
        routed = mode in {"structured_fact", "aggregate_answer"} and \
            strategy in {"structured_fact", "aggregation", "entity_fact"}
        exact = _values_equal(case["answer_type"], expected, actual)
        no_visual = "visual_ann" in skipped
        no_images = not bool(body.get("image_results"))
        no_leak = not scan_internal_leak(answer)
        mention = _answer_mentions_value(case["answer_type"], expected, answer)
        for ok, key, label in ((routed, "routed", "routed"), (exact, "exact", "exact_value"),
                               (no_visual, "no_visual", "no_visual_ann"), (no_images, "no_images", "zero_images"),
                               (no_leak, "no_leak", "no_leak"), (mention, "mention", "mentions_value")):
            if ok:
                totals[key] += 1
            else:
                reasons.append(label)
        verdict = "PASS" if not reasons else "FAIL"
        per_case.append({"query": name, "answer_type": case["answer_type"], "scope": case["scope"],
                         "verdict": verdict, "reasons": reasons,
                         "mode": mode, "strategy": strategy,
                         "expected": {"value": expected.value, "total": expected.total},
                         "actual": {"value": actual.get("value"), "total": actual.get("total")},
                         "answer": answer[:200]})
        print(f"[{verdict}] {name:38} mode={mode:>16} strategy={strategy:>14} reasons={reasons}", flush=True)

    count = len(cases)
    metrics = {key: round(value / count, 3) for key, value in totals.items()}
    print("\nmetrics:", json.dumps(metrics, ensure_ascii=False))
    passed = sum(1 for c in per_case if c["verdict"] == "PASS")
    print(f"{passed}/{count} passed")
    report = {"api": args.api, "count": count, "passed": passed,
              "metrics": metrics, "cases": per_case}
    out = REPO_ROOT / "docs" / "baseline" / "structured-qa-e2e.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0 if passed == count else 1


if __name__ == "__main__":
    sys.exit(main())
