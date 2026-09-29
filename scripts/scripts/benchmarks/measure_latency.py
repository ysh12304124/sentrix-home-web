#!/usr/bin/env python3
"""Latency / model-call budget measurement across the three user paths (Phase R8-6).

  normal_chat        writing prompt -> no retrieval, 1 answer call
  simple_evidence    household query -> parser + retrieval + answer
  complex_person     person intro -> parser + writer + claim + verify

Reports p50/p95 over >= repeats runs, model call counts, and whether any
timeout fallback fired.  Output: docs/baseline/latency_report.json
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

PROBES = {
    "normal_chat": ("帮我写一段生日祝福", "album1"),
    "simple_evidence": ("厨房里做晚饭", "album1"),
    "complex_person": ("介绍一下明哥", "album1"),
}


def _percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * p / 100.0))
    return round(ordered[index], 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default=os.getenv("SENTRIX_API_URL", "http://127.0.0.1:8091"))
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--report", default="docs/baseline/latency_report.json")
    args = parser.parse_args()

    import httpx
    client = httpx.Client(timeout=180)
    report = {"api": args.api, "repeats": args.repeats, "paths": {}}
    for name, (message, scope) in PROBES.items():
        latencies = []
        model_calls = []
        timeouts = 0
        errors = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            try:
                resp = client.post(f"{args.api}/api/assistant/turn",
                                   json={"message": message, "scope_id": scope})
                body = resp.json()
                latencies.append(time.perf_counter() - start)
                trace = body.get("retrieval_trace") or []
                parse_calls = sum(stage.get("counts", {}).get("query_parse", 0) for stage in trace)
                model_calls.append(1 + int(parse_calls))  # parser + answer approximation
            except Exception as error:
                errors.append(str(error)[:120])
                timeouts += 1
        report["paths"][name] = {
            "message": message,
            "p50_s": _percentile(latencies, 50),
            "p95_s": _percentile(latencies, 95),
            "avg_s": round(sum(latencies) / len(latencies), 3) if latencies else None,
            "model_calls_p50": _percentile(model_calls, 50),
            "timeout_or_error_count": timeouts,
            "sample_errors": errors[:2],
        }
        print(f"{name}: p50={report['paths'][name]['p50_s']}s p95={report['paths'][name]['p95_s']}s calls={report['paths'][name]['model_calls_p50']} errors={timeouts}")

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
