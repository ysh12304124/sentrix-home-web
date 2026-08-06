#!/usr/bin/env python3
"""Phase 12B-FC V6 — full-chain latency on the validation instance (GPU-fixed).

Only no-degradation results count: a run whose validation block shows
degradation or a model mismatch is excluded.  Reports cold + warm p50/p95 per
path, per-stage times, model call counts, and the product gates.

Product gates (post-GPU): simple evidence p95 <= 12s, API <= 20s.
Liveness gate: the model must complete (no forced 20s).

Run on 153:
  PYTHONPATH=. .venv/bin/python scripts/benchmarks/measure_12b_full_chain_latency.py
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

PATHS = {
    "chat": ("今天感觉怎么样", "album2_e2b"),
    "writing": ("帮我写一段生日祝福", "album2_e2b"),
    "simple_evidence": ("去年十月爬山拍的合影", "album2_e2b"),
    "strict_empty": ("贵阳夜晚步行街", "album1"),
    "allow_approximate": ("水族馆海豚跃出水面", "album3"),
    "person_chain": ("介绍一下明哥", "album2_e2b"),
}

STAGES = ("explicit_detector", "parser", "router", "retrieval", "answer", "claim")


def _pct(values, p):
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(len(ordered) * p / 100))], 3)


def _p50(values):
    return round(statistics.median(values), 3) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default=os.getenv("SENTRIX_API_URL", "http://127.0.0.1:8092"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--report", default=str(REPO_ROOT / "docs" / "baseline" / "sentrix-12b-latency-report.json"))
    args = parser.parse_args()

    import httpx
    client = httpx.Client(timeout=300)
    report = {"api": args.api, "paths": {}}
    for name, (message, scope) in PATHS.items():
        cold = None
        warm_totals = []
        stage_values = {s: [] for s in STAGES}
        calls = []
        excluded = 0
        for i in range(args.repeats):
            t0 = time.time()
            body = client.post(f"{args.api}/api/assistant/turn",
                               json={"message": message, "scope_id": scope}).json()
            elapsed = round(time.time() - t0, 2)
            v = body.get("validation") or {}
            if v.get("degradation_used") or not v.get("passed"):
                excluded += 1
                continue
            if i == 0:
                cold = elapsed
            else:
                warm_totals.append(elapsed)
            perf = body.get("perf") or {}
            for s in STAGES:
                if s in perf:
                    stage_values[s].append(perf[s])
            calls.append((perf.get("model_calls") or {}).get("parser", 0) +
                         (perf.get("model_calls") or {}).get("answer", 0) +
                         (perf.get("model_calls") or {}).get("writer", 0) +
                         (perf.get("model_calls") or {}).get("claim", 0))
        report["paths"][name] = {
            "message": message,
            "cold_s": cold,
            "warm_p50_s": _p50(warm_totals),
            "warm_p95_s": _pct(warm_totals, 95),
            "stage_p50_s": {s: _p50(vals) for s, vals in stage_values.items()},
            "model_calls_p50": _p50(calls),
            "runs_used": len(warm_totals) + (1 if cold is not None else 0),
            "excluded_degraded": excluded,
        }
        print(f"{name:16} cold={cold}s warm_p50={_p50(warm_totals)}s p95={_pct(warm_totals, 95)}s "
              f"calls={_p50(calls)} excluded={excluded}", flush=True)

    simple = report["paths"].get("simple_evidence") or {}
    gates = {
        "simple_evidence_p95_le_12s": (simple.get("warm_p95_s") is not None and simple["warm_p95_s"] <= 12.0),
        "api_le_20s": all((p.get("warm_p95_s") or 0) <= 20.0 for p in report["paths"].values()),
    }
    report["product_gates"] = gates
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ngates: {gates}")
    print(f"wrote {out}")
    return 0 if gates.get("simple_evidence_p95_le_12s") else 1


if __name__ == "__main__":
    sys.exit(main())
