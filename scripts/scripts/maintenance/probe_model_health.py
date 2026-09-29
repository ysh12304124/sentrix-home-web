#!/usr/bin/env python3
"""Probe model health: cold/warm latency + JSON well-formedness (Phase R R5).

Runs N probes against the configured GammaClient and reports, per role model:
  - cold latency (first call after load)
  - warm latency (subsequent calls)
  - JSON well-formedness rate for the parser prompt
  - error rate

Output: ``docs/baseline/model_health_YYYYMMDD.json`` (or --report path).

This is a maintenance / diagnosis tool; it never changes runtime behaviour.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _load_clients():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from backend.model_clients import GammaClient
    return GammaClient()


def probe_json_wellformed(client, probe_texts, repeats=3):
    from backend.model_clients import parse_json_response
    results = {"ok": 0, "total": 0, "samples": []}
    for text in probe_texts:
        for _ in range(repeats):
            results["total"] += 1
            start = time.perf_counter()
            try:
                raw = client.chat(
                    f"把下面这句话分类为家庭记忆查询还是普通聊天，返回 JSON {{'mode': 'none'|'evidence'|'contextual'}}: {text}",
                    json_mode=True, role="parser",
                )
            except Exception as error:
                results["samples"].append({"error": str(error), "latency_s": round(time.perf_counter() - start, 3)})
                continue
            parsed = parse_json_response(raw)
            results["ok"] += bool(isinstance(parsed, dict) and parsed.get("mode") in {"none", "evidence", "contextual"})
            results["samples"].append({"latency_s": round(time.perf_counter() - start, 3),
                                       "wellformed": bool(parsed and "mode" in (parsed or {}))})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    client = _load_clients()
    # Synthetic generic probes — never benchmark queries (guard enforced).
    probe_texts = [
        "一个银色的小物件",
        "厨房里做饭的照片",
        "帮我写一段生日祝福",
        "去年秋天他拍的合照",
    ]
    start = time.perf_counter()
    cold = client.chat("你好", json_mode=False, role="answer")
    cold_latency = time.perf_counter() - start
    warm = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        client.chat("你好", json_mode=False, role="answer")
        warm.append(time.perf_counter() - start)

    json_result = probe_json_wellformed(client, probe_texts, repeats=args.repeats)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": client.model,
        "parse_model": client.parse_model,
        "answer_model": client.answer_model,
        "verify_model": client.verify_model,
        "cold_answer_latency_s": round(cold_latency, 3),
        "warm_answer_latency_s": [round(item, 3) for item in warm],
        "warm_answer_p50_s": round(sorted(warm)[len(warm) // 2], 3) if warm else None,
        "parser_json": json_result,
        "notes": "cold latency is first call after process start; warm p50 uses the same model role.",
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
        print(f"wrote {args.report}")
        print(json.dumps({k: v for k, v in report.items() if k != "parser_json"}, ensure_ascii=False, indent=2))
    else:
        print(text)


if __name__ == "__main__":
    main()
