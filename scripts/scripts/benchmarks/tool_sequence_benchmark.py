#!/usr/bin/env python3
"""Phase F v2 F4 — Tool Sequence Benchmark（工具序列质量，规则检测，不调用模型）。

对 QA run 的每行统计：
  first_tool_accuracy      —— 首工具是否为检索入口（search_memories / query_memory_facts）
  full_sequence_success    —— complete 且（证据召回>=0.5 或 unanswerable 如实否认）
  premature_final          —— 证据已召回但未解析即结束 / 需要看图却直接给回避答案
  unnecessary_tool_call    —— 连续重复调用同一工具
  missed_required_resolution—— gold 证据存在但召回=0

用法：
  python tool_sequence_benchmark.py --run <qa_result.json> --out <out.json>
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

SEARCH_TOOLS = {"search_memories", "query_memory_facts"}
VISUAL_TOOLS = {"inspect_photo", "read_photo_text", "get_original_photos", "get_result_page"}
HEDGE_RE = re.compile(r"无法|不能确定|没有找到|如果需要|建议|不确定")


def score_row(r: dict) -> dict:
    tools = list(r.get("tools") or [])
    status = r.get("status") or ""
    ev = r.get("evidence") or {}
    has_gold = bool(ev.get("has_gold"))
    recall = ev.get("recall") or 0
    answer = (r.get("answer") or "").strip()
    answerable = r.get("answerability") == "answerable"
    decom = r.get("decom") or {}
    t_detail = ((decom.get("detail") or {}).get("T") or "")

    first_tool_ok = bool(tools) and tools[0] in SEARCH_TOOLS
    if not answerable and not has_gold:
        seq_ok = status == "complete" and bool(answer) and not HEDGE_RE.search(answer)
    elif recall >= 0.5:
        seq_ok = status == "complete"
    else:
        seq_ok = False

    premature = False
    if has_gold and recall >= 0.5 and not any(t in VISUAL_TOOLS for t in tools):
        if not answer or HEDGE_RE.search(answer):
            premature = True
    if "premature final" in t_detail:
        premature = True

    unnecessary = 0
    for i in range(1, len(tools)):
        if tools[i] == tools[i - 1]:
            unnecessary += 1

    missed_resolution = bool(has_gold and recall == 0 and status != "error")

    return {
        "qa_id": r.get("qa_id") or "",
        "status": status,
        "tools": tools,
        "first_tool_ok": first_tool_ok,
        "sequence_success": seq_ok,
        "premature_final": premature,
        "unnecessary_tool_calls": unnecessary,
        "missed_required_resolution": missed_resolution,
        "recall": recall,
        "detail_t": t_detail,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="QA run 的 qa_result.json")
    ap.add_argument("--out", default="~/Downloads/sentrix_qa_report/tool_sequence.json")
    args = ap.parse_args()

    data = json.loads(Path(args.run).expanduser().read_text(encoding="utf-8"))
    rows = data["rows"]
    scored = [score_row(r) for r in rows]
    n = max(1, len(scored))
    summary = {
        "total": len(scored),
        "first_tool_accuracy": round(sum(1 for s in scored if s["first_tool_ok"]) / n, 3),
        "full_sequence_success": round(sum(1 for s in scored if s["sequence_success"]) / n, 3),
        "premature_final": sum(1 for s in scored if s["premature_final"]),
        "unnecessary_tool_calls": sum(s["unnecessary_tool_calls"] for s in scored),
        "missed_required_resolution": sum(1 for s in scored if s["missed_required_resolution"]),
        "avg_tools_per_question": round(sum(len(s["tools"]) for s in scored) / n, 2),
        "tool_usage": dict(Counter(t for s in scored for t in s["tools"])),
    }
    print("Tool Sequence Benchmark（规则检测）")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    for s in scored:
        if s["premature_final"] or s["missed_required_resolution"] or not s["first_tool_ok"]:
            print(f"  - {s['qa_id']}: first_ok={s['first_tool_ok']} premature={s['premature_final']} "
                  f"missed={s['missed_required_resolution']} tools={s['tools']}")
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": scored},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"输出: {out}")


if __name__ == "__main__":
    main()
