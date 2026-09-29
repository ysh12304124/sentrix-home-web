#!/usr/bin/env python3
"""Phase R9-6 — agent latency / stage-trace / model-call measurement.

Requires the API to run with SENTRIX_AGENT_STAGE_TRACE=1 so each response
carries a ``perf`` block (per-stage durations + model_calls).  Reports per path:

  - cold (first run) and warm p50/p95 request totals
  - model call counts (parser / repair / answer / claim) from the real response
  - per-stage p50 (explicit_detector / parser / router / probe / query_spec /
    retrieval / answer / claim)
  - timeout / fallback / error-route counts
  - a real-HTTP-call check (a warm normal_chat must include exactly 1 answer call)

Paths cover R9 §10.2.  Output: docs/baseline/latency_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

API_DEADLINE_S = 20.0

PATHS = {
    "normal_chat": {"message": "今天感觉怎么样", "scope": "album1", "expected_route": {"none", "not_applicable"}},
    "writing": {"message": "帮我写一段生日祝福", "scope": "album1", "expected_route": {"none", "not_applicable"}},
    "short_visual": {"message": "银色心形手镯", "scope": "album3", "expected_route": {"evidence", "clarify", "anchored", "gap"}},
    "simple_evidence": {"message": "2024年5月厨房里做晚饭", "scope": "album1", "expected_route": {"evidence", "anchored"}},
    "strict_empty": {"message": "贵阳夜晚步行街", "scope": "album1", "expected_route": {"gap", "anchored", "evidence"}},
    "allow_approximate": {"message": "水族馆海豚跃出水面", "scope": "album3", "expected_route": {"evidence", "anchored", "gap"}},
    "person_intro": {"message": "介绍一下明哥", "scope": "album1", "expected_route": {"evidence", "anchored", "gap"}},
    "composite_assets": {"message": "把去年拍的照片给我", "scope": "album1", "expected_route": {"evidence", "anchored"}},
    "parser_timeout": {"message": "把我所有关于做饭、厨房、年夜饭、出行的照片和视频都找出来整理成时间线，再比较明哥和小黑在厨房出现的次数", "scope": "album1", "expected_route": {"evidence", "anchored", "gap"}},
    "answer_timeout": {"message": "聊聊家里最近发生的事，随便展开讲讲", "scope": "album1", "expected_route": {"none", "not_applicable"}},
}

STAGES = ("explicit_detector", "parser", "router", "probe", "query_spec",
          "retrieval", "answer", "claim", "complex_chain")


def _p50(values):
    return round(statistics.median(values), 4) if values else None


def _p95(values):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * 0.95))
    return round(ordered[index], 4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default=os.getenv("SENTRIX_API_URL", "http://127.0.0.1:8091"))
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--report", default="docs/baseline/latency_report.json")
    args = parser.parse_args()

    import httpx
    client = httpx.Client(timeout=float(os.getenv("SENTRIX_API_TIMEOUT", "45")))
    report = {"api": args.api, "repeats": args.repeats, "deadline_s": API_DEADLINE_S, "paths": {}}

    for name, spec in PATHS.items():
        cold_total = None
        warm_totals = []
        model_calls = []
        stage_p50 = {}
        timeouts = 0
        route_errors = []
        for run_index in range(args.repeats):
            start = time.perf_counter()
            try:
                resp = client.post(f"{args.api}/api/assistant/turn",
                                   json={"message": spec["message"], "scope_id": spec["scope"]})
                body = resp.json()
            except Exception as error:
                timeouts += 1
                if run_index == 0:
                    cold_total = round(time.perf_counter() - start, 4)
                continue
            elapsed = round(time.perf_counter() - start, 4)
            if run_index == 0:
                cold_total = elapsed
            else:
                warm_totals.append(elapsed)
            perf = body.get("perf") or {}
            if not perf:
                route_errors.append({"run": run_index, "reason": "no perf block (SENTRIX_AGENT_STAGE_TRACE off)"})
            else:
                calls = perf.get("model_calls") or {}
                model_calls.append(calls)
                for stage in STAGES:
                    if stage in perf:
                        stage_p50.setdefault(stage, []).append(float(perf[stage]))
            status = body.get("evidence_status") or "none"
            if status not in spec["expected_route"]:
                route_errors.append({"run": run_index, "reason": f"route={status} expected={spec['expected_route']}"})

        calls_p50 = {}
        if model_calls:
            for key in ("parser", "repair", "answer", "claim"):
                values = [item.get(key, 0) for item in model_calls if item]
                if values:
                    calls_p50[key] = _p50(values)

        real_http = calls_p50 or None
        report["paths"][name] = {
            "message": spec["message"],
            "cold_total_s": cold_total,
            "warm_p50_s": _p50(warm_totals),
            "warm_p95_s": _p95(warm_totals),
            "model_calls_p50": calls_p50,
            "stage_p50_s": {stage: _p50(vals) for stage, vals in stage_p50.items()},
            "timeout_or_error_count": timeouts,
            "route_mismatch_count": len([e for e in route_errors if "no perf" not in e.get("reason", "")]),
            "sample_issues": route_errors[:2],
            "real_http_model_calls": bool(real_http),
        }
        print(f"{name}: cold={cold_total}s warm_p50={report['paths'][name]['warm_p50_s']}s "
              f"calls={calls_p50} timeouts={timeouts}", flush=True)

    # A warm normal_chat must include exactly 1 real answer model call.
    normal = report["paths"].get("normal_chat")
    if normal and normal.get("model_calls_p50", {}).get("answer", 0) < 1:
        normal["real_http_check"] = "FAIL: normal_chat missing answer model call"
    elif normal:
        normal["real_http_check"] = "ok"

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
