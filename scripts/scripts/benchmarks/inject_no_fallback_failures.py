#!/usr/bin/env python3
"""Phase 12B-FC V5 — no-degradation fault injection proof.

Proves the validation profile does NOT silently degrade: when a fault is injected
(parser budget exhausted, model id mismatch, ...), the case must be marked
failed (validation.degradation_used=true / passed=false) instead of falling back
and counting as a pass.

Each probe points at a fault-injected validation instance:
  parser_timeout     SENTRIX_PARSER_BUDGET=0.5  -> parser times out -> failed
  model_mismatch     SENTRIX_PARSE_MODEL=<wrong> -> actual != expected -> failed

Expected: every probe verdict == FAIL (never PASS with a fallback).

Run on 153:
  PYTHONPATH=. .venv/bin/python scripts/benchmarks/inject_no_fallback_failures.py \
      --api-parser-timeout http://127.0.0.1:8094 --api-model-mismatch http://127.0.0.1:8095
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _probe(name, api, message, expected_fail):
    import httpx
    try:
        r = httpx.post(f"{api}/api/assistant/turn", json={"message": message, "scope_id": "album2_e2b"},
                       timeout=120)
        body = r.json()
    except Exception as exc:
        return {"name": name, "api": api, "verdict": "FAIL", "expected_fail": expected_fail,
                "reason": f"connection/error: {str(exc)[:120]}"}
    v = body.get("validation") or {}
    degraded = v.get("degradation_used")
    passed = v.get("passed")
    actual_roles = v.get("actual_roles")
    ledger = body.get("model_call_ledger") or []
    # Under an injected fault the case MUST fail (passed=False).  If it "passed"
    # anyway, the validation profile silently degraded — a violation.
    ok = (passed is False) and (degraded is True)
    return {
        "name": name, "api": api,
        "verdict": "NO_DEGRADATION_OK" if ok else "SILENT_PASS_VIOLATION",
        "expected_fail": expected_fail,
        "validation": {"passed": passed, "degradation_used": degraded,
                       "actual_roles": actual_roles, "issues": v.get("issues")},
        "ledger_roles": [c.get("role") for c in ledger],
        "evidence_status": body.get("evidence_status"),
        "answer": (body.get("answer") or "")[:120],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-parser-timeout", default="http://127.0.0.1:8094")
    parser.add_argument("--api-model-mismatch", default="http://127.0.0.1:8095")
    parser.add_argument("--report", default=str(REPO_ROOT / "docs" / "baseline" / "sentrix-12b-fault-injection.json"))
    args = parser.parse_args()

    results = [
        _probe("parser_timeout", args.api_parser_timeout, "介绍一下明哥", expected_fail=True),
        _probe("model_mismatch", args.api_model_mismatch, "去年十月爬山拍的合影", expected_fail=True),
    ]
    # A fault-injected case must fail (NO_DEGRADATION_OK); a silent pass is a violation.
    all_ok = all(r["verdict"] == "NO_DEGRADATION_OK" for r in results)
    report = {"probe_count": len(results), "all_faults_cause_failure": all_ok, "probes": results}
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
