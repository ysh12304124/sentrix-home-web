#!/usr/bin/env python3
"""Phase F v2 F3 — Synthesis Faithfulness Benchmark（Observation→Answer 硬值保真）。

对 QA run 的每行做确定性硬值提取与比对（不调用模型）：
  1. 从标准答案(gold)提取硬值：电话 / 价格 / 年份 / 日期 / 数字
  2. 硬值只在“gold 里有但问题里没有”时才算相关（避免问题自带年份被误判）
  3. 检查 agent 回答是否保留每个硬值（Preservation）
  4. 检查 agent 回答是否出现 gold/问题里都没有的硬值（Contamination，潜在编造）
  5. unanswerable 题给出具体硬值 → Certainty Upgrade Error

用法：
  python synthesis_faithfulness.py --run <qa_result.json> --out <out.json>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PHONE_RE = re.compile(r"\d{7,}")
PRICE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:元|块|¥|￥)")
YEAR_RE = re.compile(r"(?:19|20)\d{2}年")
DATE_RE = re.compile(r"\d{1,2}月\d{1,2}日")
NUM_RE = re.compile(r"\d+")
UNCERTAIN_RE = re.compile(r"无法确定|不确定|没有足够|查不到|不能确定|信息不足")


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def hard_values(text: str) -> dict:
    text = norm(text)
    out = {"phone": [], "price": [], "year": [], "date": [], "number": []}
    for m in PHONE_RE.finditer(text):
        out["phone"].append(m.group(0))
    for m in PRICE_RE.finditer(text):
        out["price"].append(m.group(1))
    for m in YEAR_RE.finditer(text):
        out["year"].append(m.group(0)[:-1])
    for m in DATE_RE.finditer(text):
        out["date"].append(m.group(0))
    seen = set()
    for m in NUM_RE.finditer(text):
        v = m.group(0)
        if v in seen:
            continue
        seen.add(v)
        if v in out["phone"] or v in out["price"] or v in out["year"]:
            continue
        out["number"].append(v)
    return out


def category(gold_hv: dict, gold: str) -> str:
    if gold_hv["phone"]:
        return "phone"
    if gold_hv["price"]:
        return "price"
    if gold_hv["date"] or gold_hv["year"]:
        return "date"
    if gold_hv["number"]:
        return "number"
    g = norm(gold)
    if any(k in g for k in ("哪里", "地点", "位置", "在")):
        return "place"
    if UNCERTAIN_RE.search(g):
        return "unknown"
    return "named_entity"


def relevant_values(gold_hv: dict, question: str) -> dict:
    """gold 有、但问题文本里没有的硬值才算相关（避免重复问题锚点）。"""
    q = norm(question)
    out = {}
    for k, vs in gold_hv.items():
        out[k] = [v for v in vs if norm(v) not in q]
    return out


def score_row(r: dict) -> dict:
    question = r.get("question") or ""
    gold = r.get("gold_answer") or ""
    answer = r.get("answer") or ""
    answerable = r.get("answerability") == "answerable"
    gold_hv = hard_values(gold)
    answer_hv = hard_values(answer)
    rel = relevant_values(gold_hv, question)
    cat = category(gold_hv, gold)

    preserved, missing = [], []
    for k, vs in rel.items():
        for v in vs:
            if norm(v) in norm(answer):
                preserved.append((k, v))
            else:
                missing.append((k, v))

    # contamination：回答里的高价值硬值（≥4 位数字/价格/电话）不在 gold 也不在问题里
    q_norm = norm(question)
    g_norm = norm(gold)
    contamination = []
    for k in ("phone", "price", "year"):
        for v in answer_hv[k]:
            if norm(v) in q_norm or norm(v) in g_norm:
                continue
            contamination.append((k, v))
    for v in answer_hv["date"]:
        if v in q_norm or v in g_norm:
            continue
        contamination.append(("date", v))
    for v in answer_hv["number"]:
        if len(v) < 4 or v in q_norm or v in g_norm:
            continue
        contamination.append(("number", v))

    certainty_upgrade = False
    if not answerable and answer.strip():
        if any(answer_hv[k] for k in ("phone", "price", "year")) or \
           any(len(v) >= 4 for v in answer_hv["number"]):
            certainty_upgrade = True

    return {
        "qa_id": r.get("qa_id") or "",
        "question": question,
        "gold": gold,
        "answer": answer,
        "answerability": r.get("answerability") or "",
        "category": cat,
        "gold_hard": gold_hv,
        "relevant_hard": rel,
        "preserved": preserved,
        "missing": missing,
        "contamination": contamination,
        "certainty_upgrade": certainty_upgrade,
        "hard_preservation_ok": not missing,
        "contamination_free": not contamination,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="QA run 的 qa_result.json")
    ap.add_argument("--out", default="~/Downloads/sentrix_qa_report/synthesis_faithfulness.json")
    args = ap.parse_args()

    data = json.loads(Path(args.run).expanduser().read_text(encoding="utf-8"))
    rows = data["rows"]
    scored = [score_row(r) for r in rows]

    by_cat = defaultdict(lambda: {"total": 0, "ok": 0, "missing": 0, "contam": 0, "upgrade": 0})
    total_missing = 0
    total_contam = 0
    total_upgrade = 0
    for s in scored:
        c = by_cat[s["category"]]
        c["total"] += 1
        if s["hard_preservation_ok"]:
            c["ok"] += 1
        if s["missing"]:
            c["missing"] += 1
            total_missing += 1
        if s["contamination"]:
            c["contam"] += 1
            total_contam += 1
        if s["certainty_upgrade"]:
            c["upgrade"] += 1
            total_upgrade += 1
    n = len(scored)
    summary = {
        "total": n,
        "hard_value_preservation_ok": round((n - total_missing) / max(1, n), 3),
        "missing_questions": total_missing,
        "contamination_questions": total_contam,
        "certainty_upgrade_errors": total_upgrade,
        "by_category": {k: dict(v) for k, v in by_cat.items()},
    }
    print("Synthesis Faithfulness（硬值保真，规则检测）")
    print(f"  硬值完整保留: {n - total_missing}/{n} = {summary['hard_value_preservation_ok']}")
    print(f"  missing 硬值题: {total_missing}")
    print(f"  contamination(编造硬值)题: {total_contam}")
    print(f"  certainty upgrade 错误: {total_upgrade}")
    for s in scored:
        if s["missing"] or s["contamination"] or s["certainty_upgrade"]:
            print(f"  - {s['qa_id']} [{s['category']}] missing={s['missing']} contam={s['contamination']} upgrade={s['certainty_upgrade']}")
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": scored},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"输出: {out}")


if __name__ == "__main__":
    main()
