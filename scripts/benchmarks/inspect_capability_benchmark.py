#!/usr/bin/env python3
"""Phase H H7 — inspect_photo capability benchmark v2（在 153 上运行）。

用真实 DB + 8100 VLM 直接调用 inspect_photo，按能力类别评估：
- people_count / clothing_color / large_object / small_object
- visible_text_short / ocr_number / scene

标注来源（诚实标注，不虚构）：qa_gold（人工 QA 金标）/ vision_faces（macOS Vision 人脸计数）
/ vision_ocr（macOS Vision OCR）/ db_metadata（导入期元数据）/ vision_class（Vision 分类）。

指标：accuracy / false_confident（supported 但答错）/ uncertain（显式不确定或空观察）。
样本量如实报告，不补齐、不为刷分而特判。

用法（153）：
  cd /home/asus/Github/Sentrix-Home-Web
  PYTHONPATH=. .venv/bin/python scripts/benchmarks/inspect_capability_benchmark.py \
      --labels /tmp/inspect_capability_labels.json --out /tmp/inspect_cap_bench.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from backend.db import MemoryStore
from backend.model_clients import GammaClient
from backend.agent_runtime import tools as runtime_tools


def _digits(text: str) -> list[str]:
    return re.findall(r"\d+", text or "")


def evaluate(capability: str, observation: str, expected: list[str]) -> bool:
    obs = re.sub(r"\s+", "", observation or "")
    exp = [re.sub(r"\s+", "", e) for e in expected]
    if capability == "people_count":
        ds = _digits(obs)
        return any(d in ds or d in obs for d in exp)
    if capability == "ocr_number":
        ds = _digits(obs)
        return all(d in ds for d in exp)
    # keyword categories
    return any(e in obs for e in exp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="/tmp/inspect_cap_bench.json")
    ap.add_argument("--scope", default="album3-v2")
    ap.add_argument("--base", default="http://127.0.0.1:8100/v1")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    labels = json.load(open(args.labels, encoding="utf-8"))
    if args.limit:
        labels = labels[:args.limit]

    store = MemoryStore(os.getenv("SENTRIX_DB_PATH", str(ROOT / "data" / "sentrix.db")))
    gamma = GammaClient(base_url=args.base)
    runtime_tools.bind_runtime(store, gamma=gamma)
    runtime_tools.register_tools()
    # 直接按 asset_id 调用 inspect_photo（handle -> asset_id 自映射）
    runtime_tools._RUNTIME["last_handles"] = {lb["asset_id"]: lb["asset_id"] for lb in labels}

    rows = []
    t0 = time.monotonic()
    for i, lb in enumerate(labels, 1):
        print(f"[{i}/{len(labels)}] {lb['capability']} {lb['file'][:30]}", flush=True)
        st = time.monotonic()
        try:
            res = runtime_tools._inspect_photo(
                {"asset_handle": lb["asset_id"], "question": lb["question"]},
                context={"scope_id": args.scope, "task_state": {}})
        except Exception as exc:
            res = {"summary": "", "observation": "", "certainty": "uncertain", "error": str(exc)}
        obs = res.get("observation") or res.get("summary") or ""
        certainty = str(res.get("certainty") or "uncertain")
        correct = evaluate(lb["capability"], obs, lb["expected"])
        if res.get("blocked") or (not obs and certainty == "uncertain"):
            state = "uncertain"
        elif certainty == "uncertain":
            state = "uncertain"
        elif correct:
            state = "correct"
        else:
            state = "false_confident"
        rows.append({
            "file": lb["file"], "asset_id": lb["asset_id"], "capability": lb["capability"],
            "question": lb["question"], "expected": lb["expected"], "source": lb.get("source"),
            "observation": obs[:300], "certainty": certainty, "state": state,
            "latency_s": round(time.monotonic() - st, 2),
        })

    # 聚合
    caps = {}
    for r in rows:
        c = caps.setdefault(r["capability"], {"tested": 0, "correct": 0, "false_confident": 0,
                                              "uncertain": 0, "error": 0, "cases": [], "failed": []})
        c["tested"] += 1
        c["cases"].append(r["file"])
        if r["state"] == "correct":
            c["correct"] += 1
        elif r["state"] == "uncertain":
            c["uncertain"] += 1
        elif r["state"] == "error":
            c["error"] += 1
        else:
            c["false_confident"] += 1
            c["failed"].append(r["file"])
    for c in caps.values():
        c["accuracy"] = round(c["correct"] / max(1, c["tested"]), 3)
        c["false_confident_rate"] = round(c["false_confident"] / max(1, c["tested"]), 3)
        c["uncertain_rate"] = round(c["uncertain"] / max(1, c["tested"]), 3)

    report = {"meta": {"scope": args.scope, "base": args.base, "labels": len(labels),
                       "elapsed_s": round(time.monotonic() - t0, 1)},
              "capabilities": caps, "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("=" * 80)
    for k, v in sorted(caps.items()):
        print(f"{k:20s} tested={v['tested']:3d} acc={v['accuracy']:.2f} "
              f"false_confident={v['false_confident']}/{v['tested']} "
              f"uncertain={v['uncertain']}/{v['tested']}")
    print(f"elapsed {report['meta']['elapsed_s']}s -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
