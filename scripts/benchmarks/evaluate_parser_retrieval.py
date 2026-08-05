#!/usr/bin/env python3
"""Parser + Retrieval combined evaluation (Phase R R1A).

Runs the same ground-truth case twice:
  - a cached/deterministic QuerySpec (no model)
  - the real QueryParser (needs GammaClient / Ollama)

and reports per-case and aggregate diffs: constraint-set agreement, mode
agreement, and Recall@10/20 delta.  When the parser is unavailable (no Ollama,
no GammaClient) the script still runs the cached side and records
``parser_unavailable`` so the 60-case regression can be scheduled without a
model dependency.

Benchmark tool — reads benchmark data, which is forbidden in runtime code.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from evaluate_retrieval_kernel import (
    DEFAULT_SAMPLES, _load_cases, _asset_ids_by_filename, _ranked_asset_ids,
    _build_deterministic_spec, _grade,
)


def _constraint_fingerprint(spec):
    return [(c.dimension, c.value, c.strictness, c.negated) for c in spec.constraints]


def _mode_fingerprint(spec):
    # Recover the gate-relevant mode from the spec's actions (best proxy offline).
    if any(a.type in {"return_assets", "summarize_person", "summarize_event"} for a in spec.actions):
        return "evidence"
    if spec.constraints:
        return "evidence"
    return "none"


def run(store, cases, parser, top_k=20, include=None, limit=None):
    from backend.evidence_retrieval import EvidenceRetrievalKernel
    kernel = EvidenceRetrievalKernel(store)
    include_set = set(item.strip() for item in include.split(",")) if include else None
    results = []
    for case in cases:
        if limit is not None and len(results) >= limit:
            break
        if include_set and case["key"] not in include_set:
            continue
        filename_to_id = _asset_ids_by_filename(store, case["album"])
        truth_ids = [filename_to_id.get(name) for name in (case.get("ground_truth") or [])]
        truth_ids = [item for item in truth_ids if item]

        cached_spec = _build_deterministic_spec(store, case)
        cached_packet = kernel.retrieve(cached_spec)
        cached_ranked = _ranked_asset_ids(cached_packet)
        cached_grade = _grade(case, cached_ranked, truth_ids, cached_packet)

        row = {
            "key": case["key"], "query": case.get("query_cn"),
            "cached": {**cached_grade,
                       "constraint_fingerprint": _constraint_fingerprint(cached_spec),
                       "mode": _mode_fingerprint(cached_spec)},
            "parser": None,
        }
        if parser is None:
            row["parser_unavailable"] = True
        else:
            try:
                from backend.query_contracts import build_query_spec
                draft = parser.parse(case.get("query_cn") or "", recent_turns="")
                parser_spec = build_query_spec(
                    draft, scope_id=case["album"], viewer_id="owner",
                    conversation_id=f"bench_{case['key']}", query_id=f"bench_{case['key']}",
                )
                packet = kernel.retrieve(parser_spec)
                ranked = _ranked_asset_ids(packet)
                grade = _grade(case, ranked, truth_ids, packet)
                row["parser"] = {
                    **grade,
                    "constraint_fingerprint": _constraint_fingerprint(parser_spec),
                    "mode": draft.mode,
                    "parser_confidence": draft.confidence,
                    "parser_actions": [a.type for a in draft.actions],
                }
            except Exception as error:
                row["parser"] = {"error": str(error)}
        results.append(row)
    return results


def _aggregate(results):
    if not results:
        return {"total": 0}
    with_parser = [item for item in results if item.get("parser") and "error" not in item["parser"]]
    agreement = sum(1 for item in with_parser
                    if item["cached"]["constraint_fingerprint"] == item["parser"]["constraint_fingerprint"]
                    and item["cached"]["mode"] == item["parser"]["mode"])
    recall_delta = None
    if with_parser:
        vals = [(item["cached"]["recall_at"].get(10), item["parser"]["recall_at"].get(10)) for item in with_parser]
        vals = [(c, p) for c, p in vals if c is not None and p is not None]
        if vals:
            recall_delta = round(sum(p - c for c, p in vals) / len(vals), 4)
    return {
        "total": len(results),
        "parser_available_count": len(with_parser),
        "spec_agreement_count": agreement,
        "spec_agreement_rate": round(agreement / len(with_parser), 4) if with_parser else None,
        "parser_vs_cached_recall@10_delta": recall_delta,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=os.getenv("SENTRIX_DB_PATH", "data/sentrix.db"))
    parser.add_argument("--samples-root", default=DEFAULT_SAMPLES)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include", default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from backend.db import MemoryStore
    from backend.query_parser import QueryParser

    store = MemoryStore(args.db)
    cases = _load_cases(args.samples_root)
    parser_obj = None
    try:
        from backend.model_clients import GammaClient
        gamma = GammaClient()
        parser_obj = QueryParser(gamma=gamma)
    except Exception as error:
        print(f"[eval] parser unavailable: {error}", file=sys.stderr)
    results = run(store, cases, parser_obj, top_k=args.top_k, include=args.include, limit=args.limit)
    store.close()
    summary = _aggregate(results)
    payload = {"summary": summary, "cases": results}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
        print(f"wrote {args.report}")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(text)


if __name__ == "__main__":
    main()
