#!/usr/bin/env python3
"""Phase E §5 — Answer Style Benchmark（回答风格，不依赖 12B Judge）。

对 QA run 里每题的真实 agent 回答做规则检测：
  1. Direct Answer Rate    —— 是否直接给答案，而非以"我找到 N 张候选/检索到…"开头
  2. Retrieval Jargon Leak —— 内部检索词汇是否泄漏到 final
  3. Unnecessary Hedging   —— 不需要不确定时是否过度保守/反问/承诺再看
  4. Core Fact Present     —— 标准答案关键实体是否保留在回答中

用法：
  evaluate_answer_style.py --run <qa_result.json>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phasee_oracle_decomposition import KEY_ENTITIES

JARGON_TERMS = [
    "候选照片", "候选", "partial_support", "candidate_only", "full_support", "no_match",
    "匹配程度", "检索结果", "相似候选", "query_satisfaction", "条件已确认", "部分确认",
    "相似匹配", "关键词的相似", "基于关键词",
]
HEDGE_TERMS = [
    "建议您查看", "建议您进一步确认", "如果需要", "可以让我为您", "可以告诉我您想了解",
    "让我查看这些照片", "您能提供更多", "我可以继续帮你核对", "我会为您进一步", "如果您需要更准确",
]
# 无实质内容的回避句式（回答几乎没有任何事实）
EMPTY_PATTERNS = [
    r"无法(?:完全)?确认", r"不能(?:完全)?确定", r"还(?:不)?无法确认", r"没有(?:找到|看到)",
    r"并不(?:能)?确认", r"尚未确认", r"暂时无法确定",
]


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def score_answer(qa_id: str, question: str, gold: str, answer: str) -> dict:
    ans = (answer or "").strip()
    a_norm = norm(ans)
    ents = KEY_ENTITIES.get(qa_id) or {}
    unanswerable = bool(ents.get("unanswerable"))

    # 1. Direct answer:不以检索过程开头（前 24 字符不含"找到 N 张/检索/候选"）
    head = a_norm[:24]
    direct = not re.search(r"找(?:到|了)?\d|候选|检索|搜索|查(?:询|到)|相关", head)
    # 2. Jargon leak
    jargon = [t for t in JARGON_TERMS if t in a_norm]
    # 3. Unnecessary hedge / empty回避
    hedged = [t for t in HEDGE_TERMS if t in ans]
    empty = bool(re.search("|".join(EMPTY_PATTERNS), ans))
    # 4. Core fact present（unanswerable 题要求如实否认；answerable 要求关键实体）
    if unanswerable:
        core = not empty or bool(re.search(r"无法(?:完全)?确认|不确定|没有足够|查不到", ans))
    else:
        hits = []
        for kind, values in ents.items():
            for v in values:
                if v and v.lower() in a_norm.lower():
                    hits.append(v)
        core = bool(hits)

    unnecessary_hedge = (not unanswerable) and (hedged or (empty and not core))
    return {
        "qa_id": qa_id, "question": question, "gold": gold, "answer": answer,
        "direct": direct, "jargon": jargon, "hedged": hedged, "empty": empty,
        "core_fact": core, "unnecessary_hedge": unnecessary_hedge,
        "unanswerable": unanswerable,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="QA run 的 qa_result.json")
    ap.add_argument("--out", default="~/Downloads/sentrix_qa_report/answer_style_report.json")
    args = ap.parse_args()

    data = json.loads(Path(args.run).expanduser().read_text(encoding="utf-8"))
    rows = data["rows"]
    scored = [score_answer(r["qa_id"], r["question"], r.get("gold_answer") or "",
                           r.get("answer") or "") for r in rows]
    n = len(scored)
    direct_n = sum(1 for s in scored if s["direct"])
    jargon_n = sum(1 for s in scored if s["jargon"])
    hedge_n = sum(1 for s in scored if s["unnecessary_hedge"])
    core_n = sum(1 for s in scored if s["core_fact"])
    empty_n = sum(1 for s in scored if s["empty"])
    summary = {
        "total": n,
        "direct_answer_rate": round(direct_n / max(1, n), 3),
        "jargon_leak_count": jargon_n,
        "unnecessary_hedge_count": hedge_n,
        "core_fact_present": round(core_n / max(1, n), 3),
        "empty_hedge_count": empty_n,
    }
    print("Answer Style（规则检测）")
    print(f"  Direct Answer Rate : {direct_n}/{n} = {summary['direct_answer_rate']}")
    print(f"  Jargon Leak        : {jargon_n} 题")
    print(f"  Unnecessary Hedge  : {hedge_n} 题")
    print(f"  Core Fact Present  : {core_n}/{n} = {summary['core_fact_present']}")
    print(f"  Empty Hedge(无事实) : {empty_n} 题")
    for s in scored:
        if s["jargon"] or s["unnecessary_hedge"] or not s["direct"]:
            print(f"  - {s['qa_id']}: direct={s['direct']} jargon={s['jargon']} hedge={s['hedged'][:1]}")
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": scored},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"输出: {out}")


if __name__ == "__main__":
    main()
