#!/usr/bin/env python3
"""Phase H H7 — inspect_photo capability 重评估：L2 模型判分（替代死代码中文数字/同义词表）。

原则（用户拍板）：评价层也用模型。数字等价（"三个人" vs "3"）、同义词等价由 12B judge
判断，代码不再维护中文数字表/同义词正则。代码只做结构性兜底（observation 为空/调用失败）。

输入：inspect_capability_benchmark.py 产出的 bench.json（rows[].observation 为 VLM 实际观察）。
判定：每条观察是否支持标准事实 expected。
  supported     —— 观察直接/等价确认该事实（数字等价、同义词、可推断）
  contradicted  —— 观察与标准事实明确矛盾
  unknown       —— 观察未提及且不矛盾（不确定）
用法（153）：
  cd /home/asus/Github/Sentrix-Home-Web
  PYTHONPATH=. .venv/bin/python scripts/benchmarks/inspect_capability_l2_eval.py \
      --bench /tmp/inspect_cap_bench.json --out /tmp/inspect_cap_l2_eval.json \
      --base http://127.0.0.1:8100/v1
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

JUDGE_SYSTEM = """你是照片复核能力评估员。你的任务是判断一段“VLM 照片观察文本”是否支持某个“标准事实”。

标准事实是照片里实际内容的 ground truth（人工/独立工具标注）；VLM 观察是多模态模型对同一张照片的文字描述。

请判定观察是否支持标准事实：
1. supported：观察直接确认或等价表达该事实。
   - 数字等价：标准“3”，观察“三个人/三名/3 人”都算支持；
   - 同义词/量词等价：标准“餐厅”，观察“店里/饭馆”且语境一致算支持；
   - 标准事实是数字时，观察中任意等价数字即支持。
2. contradicted：观察与标准事实明确矛盾（如标准“3 人”，观察明确说“一个人/没有人”）。
3. unknown：观察未提及该事实，且与标准事实不矛盾（如标准问人数，观察只描述场景没提人数）。

判定标准（严格遵守，宁可 unknown，不误判 supported）：
- 观察含糊（“多个人/一群人/一些人”）且标准是具体数字：若不能确定等价，判 unknown；
- 观察描述的是同一主体但数值不同：判 contradicted；
- 观察未涉及该能力维度：判 unknown。

只输出一个 JSON 对象，不要 markdown、不要多余文字：
{"verdict": "supported" 或 "contradicted" 或 "unknown", "reason": "一句话理由"}"""


def _chat_completion(base: str, system: str, user: str, timeout: int = 90) -> str:
    body = json.dumps({
        "model": "gemma4-12b-it",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "max_tokens": 200,
    }).encode("utf-8")
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return (payload["choices"][0]["message"]["content"] or "").strip()


def parse_verdict(raw: str) -> str | None:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except Exception:
        return None
    verdict = str(data.get("verdict") or "")
    if verdict in ("supported", "contradicted", "unknown"):
        return verdict
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, help="inspect_capability_benchmark.py 输出的 bench.json")
    ap.add_argument("--out", default="/tmp/inspect_cap_l2_eval.json")
    ap.add_argument("--base", default="http://127.0.0.1:8100/v1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.0)
    args = ap.parse_args()

    bench = json.load(open(args.bench, encoding="utf-8"))
    rows = bench.get("rows", [])
    if args.limit:
        rows = rows[:args.limit]

    results = []
    t0 = time.monotonic()
    for i, row in enumerate(rows, 1):
        cap = row.get("capability") or "?"
        obs = (row.get("observation") or "").strip()
        expected = row.get("expected") or []
        question = row.get("question") or ""
        print(f"[{i}/{len(rows)}] {cap} {row.get('file','')[:28]}", flush=True)
        entry = dict(row)
        if not obs or obs.lower() in {"", "空", "无", "无法描述"}:
            entry["l2_verdict"] = "unknown"
            entry["l2_reason"] = "observation 为空/不可用（结构性兜底，不调用模型）"
        else:
            user = (
                f"标准事实：{'、'.join(str(e) for e in expected)}\n"
                f"VLM 观察：{obs}\n"
                f"（原问题：{question}）\n"
                f"请判断观察是否支持标准事实。"
            )
            try:
                raw = _chat_completion(args.base, JUDGE_SYSTEM, user)
                verdict = parse_verdict(raw)
                entry["l2_verdict"] = verdict or "unknown"
                entry["l2_reason"] = (raw or "")[:200]
            except Exception as exc:
                entry["l2_verdict"] = "error"
                entry["l2_reason"] = f"model_call_error:{exc}"
        results.append(entry)
        if args.sleep:
            time.sleep(args.sleep)

    # 汇总
    by_cap: dict[str, Counter] = {}
    for r in results:
        by_cap.setdefault(r["capability"], Counter())[r["l2_verdict"]] += 1
    summary = {}
    for cap, c in sorted(by_cap.items()):
        total = sum(c.values())
        summary[cap] = {
            "tested": total,
            "supported": c.get("supported", 0),
            "contradicted": c.get("contradicted", 0),
            "unknown": c.get("unknown", 0),
            "error": c.get("error", 0),
            "support_rate": round(c.get("supported", 0) / max(total, 1), 3),
            # 旧死代码判分对比（如存在）
            "old_correct": sum(1 for r in results if r["capability"] == cap and r.get("state") == "correct"),
            "old_false_confident": sum(1 for r in results if r["capability"] == cap and r.get("state") == "false_confident"),
        }

    out = {
        "meta": {
            "scope": bench.get("meta", {}).get("scope"),
            "base": args.base,
            "labels": len(results),
            "elapsed_s": round(time.monotonic() - t0, 1),
            "judge": "gemma4-12b-it",
        },
        "capabilities": summary,
        "rows": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("=== SUMMARY ===")
    for cap, s in summary.items():
        print(f"{cap}: supported={s['supported']}/{s['tested']} "
              f"contradicted={s['contradicted']} unknown={s['unknown']} "
              f"error={s['error']} (旧判分 correct={s['old_correct']})")
    print(f"run 归档: {args.out}")


if __name__ == "__main__":
    main()
