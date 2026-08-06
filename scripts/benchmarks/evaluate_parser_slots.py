#!/usr/bin/env python3
"""Phase R9-3 — Parser slot acceptance (12B Agent Model Profile).

Evaluates the parser's semantic SLOTS (actions / facets / conditions / negation /
date / media), not just mode.  mode_accuracy is still recorded but is no longer
the primary metric (R9 §7.2).

Runs the real QueryParser (production code path) against a SYNTHETIC label set
that deliberately avoids Retrieval benchmark original sentences, so it never
leaks benchmark data into runtime or prompts.

Usage:
  python evaluate_parser_slots.py --candidate e2b   # 153 e2b 2B
  python evaluate_parser_slots.py --candidate 12b   # 153 quality_12b (default)
  python evaluate_parser_slots.py --candidate 7b    # optional mid-size via SENTRIX_PARSE_MODEL

Exit gates (12B): action recall >= 0.95, facet recall >= 0.95,
negative recall >= 0.98, invented hard = 0, JSON first-pass >= 0.98.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Synthetic slot labels — NOT retrieval benchmark sentences.  Structure mirrors
# the QueryParseDraft schema so expected fields are checked by dimension/kind.
SLOT_CASES = [
    {"query": "找一下去年春节我们拍的全家福照片", "mode": "evidence",
     "actions": ["return_assets"], "facets": ["time"], "date": True, "media": True,
     "negative": False, "hard": ["time", "media"]},
    {"query": "介绍一下明哥", "mode": "evidence", "actions": ["answer_question", "summarize_person"],
     "facets": ["person"], "date": False, "media": False, "negative": False},
    {"query": "不要视频，只要那天的照片", "mode": "evidence", "actions": ["return_assets"],
     "facets": [], "date": False, "media": True, "negative": True},
    {"query": "2024年5月我们在厨房做晚饭的照片", "mode": "evidence", "actions": ["return_assets"],
     "facets": ["time", "place", "activity"], "date": True, "media": True, "negative": False},
    {"query": "那次郊游大家都穿什么", "mode": "evidence", "actions": ["answer_question"],
     "facets": ["activity", "clothing"], "date": False, "media": False, "negative": False},
    {"query": "帮我写一首关于春天的诗", "mode": "none", "actions": [], "facets": [],
     "date": False, "media": False, "negative": False},
    {"query": "今天有点累", "mode": "none", "actions": [], "facets": [],
     "date": False, "media": False, "negative": False},
    {"query": "解释一下量子纠缠", "mode": "none", "actions": [], "facets": [],
     "date": False, "media": False, "negative": False},
    {"query": "把去年拍的照片都给我，不要视频", "mode": "evidence", "actions": ["return_assets"],
     "facets": ["time"], "date": True, "media": True, "negative": True},
    {"query": "泳池里那个孩子拍水花的画面", "mode": "evidence", "actions": ["answer_question"],
     "facets": ["place", "object", "visual"], "date": False, "media": False, "negative": False},
    {"query": "比较一下明哥和小黑去年谁去的公园多", "mode": "evidence", "actions": ["compare"],
     "facets": ["person", "place"], "date": True, "media": False, "negative": False},
    {"query": "这家人有没有养狗的照片", "mode": "evidence", "actions": ["answer_question"],
     "facets": ["object"], "date": False, "media": False, "negative": False},
    {"query": "上次说的那件黄色的外套在哪张照片里", "mode": "evidence",
     "actions": ["answer_question", "return_assets"], "facets": ["clothing", "time"],
     "date": False, "media": True, "negative": False},
    {"query": "不用查我的记忆，随便聊聊", "mode": "none", "actions": [], "facets": [],
     "date": False, "media": False, "negative": False},
    {"query": "把所有关于厨房的记录都列出来", "mode": "evidence", "actions": ["answer_question"],
     "facets": ["place"], "date": False, "media": False, "negative": False},
]

# Candidate -> env overrides applied before constructing the real client.
CANDIDATES = {
    "e2b": {"SENTRIX_PARSE_BACKEND": "e2b", "SENTRIX_PARSE_MODEL": "gemma-4-e2b-it+lora-v2",
            "SENTRIX_PARSE_BASE_URL": "http://127.0.0.1:8100"},
    "12b": {"SENTRIX_AGENT_MODEL_PROFILE": "quality_12b", "SENTRIX_PARSE_BACKEND": "ollama_local"},
    "7b": {"SENTRIX_PARSE_BACKEND": "ollama_local", "SENTRIX_PARSE_MODEL": "qwen2.5:7b"},
}

_NEGATIVE_RE = {"不要", "排除", "不是", "别"}
_MEDIA_RE = {"照片", "图片", "原图", "视频", "录像"}


def _draft_has_date(draft):
    return bool(draft.time_expression)


def _draft_has_media(draft):
    return bool(draft.media_expressions)


def _draft_has_negative(draft):
    return bool(draft.negative_conditions)


def _draft_has_action_type(draft, action_type):
    return any(a.type == action_type for a in draft.actions)


def _draft_has_facet_dimension(draft, dimension):
    return any(f.dimension == dimension for f in draft.facets)


def _draft_has_semantic_dimension(draft, dimension):
    return any(c.get("dimension") == dimension for c in draft.semantic_conditions)


def _facet_dims(draft):
    return {f.dimension for f in draft.facets}


def _semantic_dims(draft):
    return {c.get("dimension") for c in draft.semantic_conditions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=sorted(CANDIDATES), default="12b")
    parser.add_argument("--report", default="docs/baseline/parser_slots_{candidate}.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    for key, value in CANDIDATES[args.candidate].items():
        os.environ[key] = value

    sys.path.insert(0, str(REPO_ROOT))
    from backend.model_clients import GammaClient
    from backend.query_parser import QueryParser

    gamma = GammaClient()
    qp = QueryParser(gamma=gamma)

    cases = SLOT_CASES
    if args.limit:
        cases = cases[: args.limit]

    rows = []
    stats = {"cases": len(cases), "first_total": 0, "json_first_valid": 0,
             "repair": 0, "latency_sum": 0.0}
    totals = {"action": 0, "facet": 0, "semantic": 0, "negative": 0,
              "date": 0, "media": 0, "invented": 0}
    hits = {key: 0 for key in totals}

    for case in cases:
        query = case["query"]
        started = time.monotonic()
        raw = qp._call_parser(query, "", None)
        stats["first_total"] += 1
        first_draft, first_errors = qp._draft_and_validate(raw) if raw else (qp._safe_fallback(), ["no_raw"])
        stats["json_first_valid"] += int(not first_errors)
        stats["repair"] += int(bool(first_errors))
        draft = qp.parse(query, "")
        stats["latency_sum"] += time.monotonic() - started

        expected_actions = set(case["actions"])
        got_actions = {a.type for a in draft.actions}
        if expected_actions:
            totals["action"] += 1
            hits["action"] += int(expected_actions <= got_actions)
        expected_facets = set(case["facets"])
        if expected_facets:
            totals["facet"] += 1
            hits["facet"] += int(expected_facets <= _facet_dims(draft))
        if case.get("date"):
            totals["date"] += 1
            hits["date"] += int(_draft_has_date(draft))
        if case.get("media"):
            totals["media"] += 1
            hits["media"] += int(_draft_has_media(draft))
        if case.get("negative"):
            totals["negative"] += 1
            hits["negative"] += int(_draft_has_negative(draft))
        # invented hard condition: a hard structure the label says must be absent.
        invented = 0
        if not case.get("date") and _draft_has_date(draft):
            invented += 1
        if not case.get("media") and _draft_has_media(draft):
            invented += 1
        if not case.get("negative") and _draft_has_negative(draft):
            invented += 1
        totals["invented"] += 1
        hits["invented"] += int(invented == 0)

        rows.append({
            "query": query, "mode": draft.mode, "actions": got_actions,
            "facets": sorted(_facet_dims(draft)), "semantic_dims": sorted(_semantic_dims(draft)),
            "date": _draft_has_date(draft), "media": _draft_has_media(draft),
            "negative": _draft_has_negative(draft), "expected": case,
        })

    def rate(key):
        return round(hits[key] / max(1, totals[key]), 4) if totals[key] else None

    report = {
        "candidate": args.candidate,
        "action_recall": rate("action"),
        "facet_recall": rate("facet"),
        "semantic_condition_recall": rate("semantic"),
        "negative_recall": rate("negative"),
        "date_recall": rate("date"),
        "media_recall": rate("media"),
        "invented_hard_zero": rate("invented"),
        "json_first_pass_valid": round(stats["json_first_valid"] / max(1, stats["first_total"]), 4),
        "repair_rate": round(stats["repair"] / max(1, stats["first_total"]), 4),
        "avg_parse_seconds": round(stats["latency_sum"] / max(1, stats["cases"]), 3),
        "mode_accuracy": round(sum(1 for r in rows if r["mode"] == r["expected"]["mode"]) / max(1, len(rows)), 4),
        "counts": stats,
        "cases": rows,
    }
    path = Path(args.report.format(candidate=args.candidate))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    summary = {k: v for k, v in report.items() if k != "cases"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
