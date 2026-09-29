#!/usr/bin/env python3
"""Phase 12B-FC V1 — 12B endpoint / model / GPU liveness probe.

Checks the base model is really reachable, the model ID matches config, and the
GPU can host the model.  Output: docs/baseline/sentrix-12b-liveness-report.md

Run on 153:
  PYTHONPATH=. .venv/bin/python scripts/maintenance/probe_12b_liveness.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _ollama_base():
    return os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("SENTRIX_PARSE_MODEL", "gemma4:12b"))
    parser.add_argument("--report", default=str(REPO_ROOT / "docs" / "baseline" / "sentrix-12b-liveness-report.md"))
    args = parser.parse_args()

    import httpx
    base = _ollama_base()
    report = {"model": args.model, "endpoint": base, "checks": {}}

    # 1. endpoint reachable + tags
    try:
        tags = httpx.get(f"{base}/api/tags", timeout=10).json().get("models", [])
        names = [m.get("name") for m in tags]
        report["checks"]["tags"] = {"ok": True, "models": names}
        model_found = args.model in names
        report["checks"]["model_present"] = {"ok": model_found,
                                             "expected": args.model, "actual": names}
    except Exception as exc:
        report["checks"]["tags"] = {"ok": False, "error": str(exc)}
        report["checks"]["model_present"] = {"ok": False, "error": "endpoint unreachable"}

    # 2. basic chat roundtrip + actual model echo
    try:
        t0 = time.monotonic()
        resp = httpx.post(f"{base}/api/chat", json={
            "model": args.model,
            "messages": [{"role": "user", "content": "你好，请回一句简短的话"}],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 64},
        }, timeout=90)
        body = resp.json()
        report["checks"]["chat_roundtrip"] = {
            "ok": resp.status_code == 200,
            "status": resp.status_code,
            "actual_model": body.get("model"),
            "latency_s": round(time.monotonic() - t0, 2),
            "sample": (body.get("message", {}).get("content") or "")[:80],
        }
    except Exception as exc:
        report["checks"]["chat_roundtrip"] = {"ok": False, "error": str(exc)}

    # 3. GPU residency (nvidia-smi + ollama ps)
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
                              "--format=csv"], capture_output=True, text=True, timeout=10)
        report["checks"]["nvidia_smi"] = {"ok": smi.returncode == 0, "output": smi.stdout.strip()[:300]}
    except Exception as exc:
        report["checks"]["nvidia_smi"] = {"ok": False, "error": str(exc)}
    try:
        ps = httpx.get(f"{base}/api/ps", timeout=10).json()
        resident = [{"name": m.get("name"), "size_vram": m.get("size_vram"),
                     "expires_at": m.get("expires_at")} for m in ps.get("models", [])]
        report["checks"]["ollama_residency"] = {"ok": True, "resident": resident}
    except Exception as exc:
        report["checks"]["ollama_residency"] = {"ok": False, "error": str(exc)}

    ok = all(c.get("ok", False) for c in report["checks"].values())
    report["verdict"] = "ALIVE" if ok else "BLOCKED"
    report["stop_condition"] = "proceed to V2" if ok else "output blocker report and stop"

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(
        "# 12B Liveness Report\n\n" + json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
