#!/usr/bin/env python3
"""Independent Parser acceptance metrics (Phase R8-Parser).

Reports Parser behaviour SEPARATELY from Retrieval Recall:

  mode_accuracy              evidence cases -> evidence; general cases -> none
  action_facet_retention     evidence cases keep at least one action or facet
  hard_condition_loss        date/person/media/negation tokens lost from draft
  repair_rate                fraction of parses that triggered the repair call
  json_first_pass_valid      fraction where the first model JSON validated
  probe_rescue_rate          evidence cases the parser sent to none but whose
                             draft still carried household signals (R4 probe/gate
                             would rescue them)
  general_false_trigger      general cases the parser sent to evidence/contextual

Label set: the 60 benchmark household queries (expected evidence) + a synthetic
general-task set (writing/translation/explanation, expected none).  The parser
runs through the real QueryParser (e2b 2B backend) so this measures production
behaviour.  Output: docs/baseline/parser_acceptance.json
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from evaluate_retrieval_kernel import DEFAULT_SAMPLES, _load_cases

# General-task labels (synthetic, NOT benchmark queries) -> expected none.
GENERAL_CASES = [
    "帮我写一段生日祝福",
    "请写一篇关于春天的短文",
    "翻译成英文：今天天气很好",
    "解释一下量子纠缠",
    "为什么天空是蓝色的",
    "假设你是一台相机，拍个故事",
    "帮我起草一份给房东的信",
    "生成一段欢迎语",
    "写个推荐语",
    "讲讲你怎么看人工智能",
]

_DATE_RE = re.compile(r"(20\d{2}\s*年|去年|今年|元旦|春节|跨年|月|日)")
_PERSON_RE = re.compile(r"(明哥|王明|八戒|小黑|自己|合照|妈妈|爸爸)")
_MEDIA_RE = re.compile(r"(照片|图片|原图|视频)")
_NEGATION_RE = re.compile(r"(不要|排除|不是|别)")


def _hard_tokens(query):
    return {
        "date": bool(_DATE_RE.search(query)),
        "person": bool(_PERSON_RE.search(query)),
        "media": bool(_MEDIA_RE.search(query)),
        "negation": bool(_NEGATION_RE.search(query)),
    }


def _draft_has(query, draft, token_kind):
    if token_kind == "date":
        return bool(draft.time_expression)
    if token_kind == "media":
        return bool(draft.media_expressions)
    if token_kind == "negation":
        return bool(draft.negative_conditions)
    if token_kind == "person":
        return bool(draft.entity_names) or any(
            c.get("dimension") == "person" for c in draft.semantic_conditions)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-root", default=DEFAULT_SAMPLES)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--report", default="docs/baseline/parser_acceptance.json")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from backend.model_clients import GammaClient
    from backend.query_parser import QueryParser
    from backend.query_contracts import sanitize_query_parse

    gamma = GammaClient()
    qp = QueryParser(gamma=gamma)

    evidence_cases = []
    for case in _load_cases(args.samples_root):
        if args.limit and len(evidence_cases) >= args.limit:
            break
        evidence_cases.append(case)

    rows = []
    stats = {"evidence": len(evidence_cases), "general": len(GENERAL_CASES),
             "repair": 0, "json_first_valid": 0, "first_total": 0}
    correct_mode = 0
    general_false = 0
    probe_rescue = 0

    for case in evidence_cases:
        query = case.get("query_cn") or ""
        hard = _hard_tokens(query)
        # First model output JSON validity + repair trigger
        raw = qp._call_parser(query, "", None)
        stats["first_total"] += 1
        first_draft, first_errors = qp._draft_and_validate(raw) if raw else (qp._safe_fallback(), ["no_raw"])
        stats["json_first_valid"] += int(not first_errors)
        stats["repair"] += int(bool(first_errors))  # errors -> repair attempted
        # Full pipeline (repair + deterministic overlay)
        draft = qp.parse(query, "")
        correct = draft.mode == "evidence"
        correct_mode += int(correct)
        if not correct:
            # none mode: would the R4 gate/probe rescue it?
            if draft.mode == "none" and (draft.semantic_conditions or draft.facets or hard["date"] or hard["person"]):
                probe_rescue += 1
        hard_loss = []
        for kind, present in hard.items():
            if present and not _draft_has(query, draft, kind):
                hard_loss.append(kind)
        rows.append({"key": case["key"], "query": query, "mode": draft.mode,
                     "actions": len(draft.actions), "facets": len(draft.facets),
                     "hard_loss": hard_loss, "correct": correct})

    for query in GENERAL_CASES:
        draft = qp.parse(query, "")
        correct = draft.mode == "none"
        correct_mode += int(correct)
        general_false += int(not correct)
        rows.append({"key": f"gen_{len(rows)}", "query": query, "mode": draft.mode,
                     "actions": len(draft.actions), "facets": len(draft.facets),
                     "hard_loss": [], "correct": correct})

    total = len(rows)
    report = {
        "mode_accuracy": round(correct_mode / total, 4),
        "action_facet_retention": round(sum(1 for r in rows if r["correct"] and (r["actions"] or r["facets"])) / max(1, stats["evidence"]), 4),
        "hard_condition_loss": {},
        "general_false_trigger": round(general_false / len(GENERAL_CASES), 4),
        "probe_rescue_rate": round(probe_rescue / max(1, stats["evidence"]), 4),
        "repair_rate": round(stats["repair"] / stats["first_total"], 4),
        "json_first_pass_valid": round(stats["json_first_valid"] / max(1, stats["first_total"]), 4),
        "counts": stats,
    }
    loss = {}
    for r in rows:
        for kind in r["hard_loss"]:
            loss[kind] = loss.get(kind, 0) + 1
    report["hard_condition_loss"] = {k: round(v / max(1, stats["evidence"]), 4) for k, v in loss.items()}

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps({"summary": report, "cases": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.report}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
