#!/usr/bin/env python3
"""Annotate the 60 benchmark cases with acceptance policy (Phase R8-1).

Per case this emits:

  exact_asset_ids                 Ground-Truth file names resolved to asset ids
  acceptable_approximate_asset_ids  (empty by default — only human/rule approval)
  forbidden_asset_ids              assets that must never surface for this query
  empty_policy                     strict_empty | allow_approximate
  answerability                    reuse of the R0 audit categories
  gt_conflict_note                 freeze interpretation for GT count mismatches

empty_policy rules (user-defined, R8-1):
  - strict_empty: the query carries a concrete un-relaxable anchor —
    confirmed person, explicit city/district/province, explicit date,
    relationship+action combination, or the user asks for exact results only.
  - allow_approximate: mostly a pure visual description with no person/date/geo
    hard anchor, and the product allows showing approximate images WITH a
    difference explanation.

Output lives in docs/baseline/benchmark_annotations.json — never in runtime or
configs.  Human overrides can be supplied via --manual-override and are merged
on top of the heuristic classification.

This is a benchmark tool: it legitimately reads benchmark data, which is
forbidden in runtime code (backend/*.py, configs/) but allowed here.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from evaluate_retrieval_kernel import DEFAULT_SAMPLES, _load_cases

# Concrete anchors -> strict_empty.
_GEO_RE = re.compile(r"(市|区|省|县|镇|乡|湾|湖|山|路|街|城|岛)")
_DATE_RE = re.compile(r"(年|月|日|节|跨年|元旦|春节|圣诞|中秋)")
_PERSON_TOKENS = ("明哥", "王明", "八戒", "小黑", "自己", "妈妈", "爸爸", "爷爷", "奶奶", "合照", "我们")
_RELATION_ACTIONS = ("搂着", "抱着", "牵着", "靠着", "亲", "背")
_EXACT_ONLY = ("只要精确", "只要准确的", "只要确定", "精确", "必须")

# The 7 empty-GT cases from the original benchmark (GT=0) — these are the
# ones where empty_policy is most load-bearing.
EMPTY_GT_CASES = {"album1-07", "album1-16", "album2-02", "album2-06", "album2-12", "album3-14", "album3-17"}


def classify_empty_policy(query: str) -> str:
    value = str(query or "")
    if any(token in value for token in _EXACT_ONLY):
        return "strict_empty"
    person_anchor = any(token in value for token in _PERSON_TOKENS)
    geo_anchor = bool(_GEO_RE.search(value))
    date_anchor = bool(_DATE_RE.search(value))
    relation_anchor = any(token in value for token in _RELATION_ACTIONS)
    if person_anchor and (relation_anchor or geo_anchor or date_anchor):
        return "strict_empty"
    if person_anchor or geo_anchor or date_anchor:
        return "strict_empty"
    return "allow_approximate"


def gt_conflict_note(case) -> str:
    declared = case.get("ground_truth_count") or 0
    listed = len(case.get("ground_truth") or [])
    if declared and declared != listed:
        return f"GT 不一致：声明 {declared}，列出 {listed}。以列出文件为准（R8-1 冻结解释规则）。"
    return ""


def annotate(cases, samples_root):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from backend.db import MemoryStore
    import scripts.benchmarks.audit_benchmark_cases as audit

    db_path = os.getenv("SENTRIX_DB_PATH", "data/sentrix.db")
    store = MemoryStore(db_path)
    audit_rows = {}
    for case in cases:
        filename_to_id = audit._asset_ids_by_filename(store, case["album"])
        row = audit._audit_case(store, case, filename_to_id)
        audit_rows[case["key"]] = row
    store.close()

    annotations = []
    for case in cases:
        key = case["key"]
        arow = audit_rows[key]
        exact = arow["gt_resolved_asset_ids"]
        query = case.get("query_cn") or ""
        empty_policy = classify_empty_policy(query)
        # Empty-GT cases default to strict_empty unless pure-visual; keep the
        # heuristic but note the case is empty.
        if key in EMPTY_GT_CASES:
            empty_policy = empty_policy  # heuristic already handles anchors
        annotations.append({
            "key": key,
            "query_cn": query,
            "album": case["album"],
            "exact_asset_ids": exact,
            "acceptable_approximate_asset_ids": [],
            "forbidden_asset_ids": [],
            "empty_policy": empty_policy,
            "answerability": arow["answerability"],
            "gt_conflict_note": gt_conflict_note(case),
            "empty_gt": not bool(case.get("ground_truth")),
        })
    return annotations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-root", default=DEFAULT_SAMPLES)
    parser.add_argument("--manual-override", default=None, help="JSON of {key: {field: value}} human edits, merged on top")
    parser.add_argument("--report", default="docs/baseline/benchmark_annotations.json")
    args = parser.parse_args()

    cases = _load_cases(args.samples_root)
    annotations = annotate(cases, args.samples_root)

    if args.manual_override:
        overrides = json.loads(Path(args.manual_override).read_text(encoding="utf-8"))
        for entry in annotations:
            edits = overrides.get(entry["key"])
            if edits:
                entry.update(edits)

    summary = {
        "total": len(annotations),
        "strict_empty": sum(1 for a in annotations if a["empty_policy"] == "strict_empty"),
        "allow_approximate": sum(1 for a in annotations if a["empty_policy"] == "allow_approximate"),
        "empty_gt": sum(1 for a in annotations if a["empty_gt"]),
        "gt_conflicts": sum(1 for a in annotations if a["gt_conflict_note"]),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(
        {"schema_version": 2, "summary": summary, "cases": annotations},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.report}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
