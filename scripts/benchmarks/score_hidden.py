#!/usr/bin/env python3
"""Phase R9-5 — Hidden Acceptance offline scorer (USER side).

Reads ``hidden_predictions.json`` (produced by evaluate_hidden_acceptance.py,
contains NO GT) plus a user-held GT file and outputs the R9 §9.2 metrics.  The
code agent never sees the GT.

GT file format (``--gt``), keyed by hidden case key:
{
  "album1-04": {
    "gt_asset_ids": ["id1", "id2"],
    "expected_route": "evidence",     // evidence | contextual | none | clarify
    "empty": false,                    // expected empty result
    "all_relevant": false,
    "expected_mode": "evidence",       // parser mode expectation
    "expected_slots": {"actions": ["return_assets"], "facets": ["time"]}   // optional
  }, ...
}

Usage:
  PYTHONPATH=. .venv-mac/bin/python scripts/benchmarks/score_hidden.py \
      --predictions docs/baseline/hidden_predictions.json --gt ~/hidden_gt.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _mrr(ranks):
    if not ranks:
        return 0.0
    return round(sum(1.0 / r for r in ranks) / len(ranks), 4)


def _recall(hits, total):
    return round(hits / total, 4) if total else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default="docs/baseline/hidden_predictions.json")
    parser.add_argument("--gt", required=True)
    parser.add_argument("--report", default="docs/baseline/hidden_acceptance.json")
    args = parser.parse_args()

    preds = json.loads(Path(args.predictions).read_text(encoding="utf-8"))["cases"]
    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))

    rows = []
    rank_hits = {1: 0, 5: 0, 10: 0, 20: 0}
    rank_totals = 0
    first_ranks = []
    strict_empty_fp = 0
    strict_empty_total = 0
    route_ok = 0
    route_family_total = 0
    general_false = 0
    general_total = 0
    clarify_count = 0
    route_loss = 0
    mode_ok = 0
    mode_total = 0
    all_relevant_hits = 0
    all_relevant_total = 0
    approximate_legality_total = 0
    approximate_legality_ok = 0

    for case in preds:
        key = case["key"]
        label = gt.get(key)
        if label is None:
            continue
        expected_route = label.get("expected_route", "evidence")
        route = case["route"]
        empty = bool(label.get("empty"))
        expected_mode = label.get("expected_mode")

        if empty:
            strict_empty_total += 1
            if case["retrieved_asset_ids"]:
                strict_empty_fp += 1

        if expected_route in {"evidence", "contextual"}:
            route_family_total += 1
            route_ok += int(route in {"evidence", "contextual"})
        elif expected_route == "none":
            general_total += 1
            general_false += int(route == "evidence")
        if expected_route in {"evidence", "contextual"} and route in {"none", "clarify"}:
            route_loss += 1
        if route == "clarify":
            clarify_count += 1

        if expected_mode:
            mode_total += 1
            mode_ok += int(case["parser"]["mode"] == expected_mode)

        truth = [item for item in (label.get("gt_asset_ids") or [])]
        if truth and route == "evidence":
            retrieved = case["retrieved_asset_ids"]
            rank_totals += 1
            first = None
            for idx, asset_id in enumerate(retrieved, 1):
                if asset_id in truth:
                    first = idx
                    break
            if first is not None:
                first_ranks.append(first)
                for k in (1, 5, 10, 20):
                    if first <= k:
                        rank_hits[k] += 1
            if label.get("all_relevant"):
                all_relevant_total += 1
                all_relevant_hits += int(all(item in retrieved for item in truth))
            # approximate legality: every returned item has a known evidence level.
            approximate_legality_total += 1
            approximate_legality_ok += int(all(lvl in {"exact", "strong", "approximate"}
                                               for lvl in case["evidence_levels"]))

        rows.append({"key": key, "route": route, "expected_route": expected_route,
                     "empty": empty, "route_ok": route in {"evidence", "contextual"}
                     if expected_route in {"evidence", "contextual"} else None,
                     "first_rank": first if truth and route == "evidence" else None})

    report = {
        "count": len(preds),
        "retrieval": {
            "ranked_cases": rank_totals,
            "recall@1": _recall(rank_hits[1], rank_totals),
            "recall@5": _recall(rank_hits[5], rank_totals),
            "recall@10": _recall(rank_hits[10], rank_totals),
            "recall@20": _recall(rank_hits[20], rank_totals),
            "mrr": _mrr(first_ranks),
        },
        "router": {
            "family_to_evidence_rate": _recall(route_ok, route_family_total),
            "general_false_trigger": _recall(general_false, general_total),
            "clarify_count": clarify_count,
            "route_loss_count": route_loss,
        },
        "strict_empty_fp": strict_empty_fp,
        "strict_empty_total": strict_empty_total,
        "approximate_legality": _recall(approximate_legality_ok, approximate_legality_total),
        "all_relevant_recall": _recall(all_relevant_hits, all_relevant_total),
        "parser_mode_accuracy": _recall(mode_ok, mode_total),
        "cases": rows,
    }
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
