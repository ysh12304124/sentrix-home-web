#!/usr/bin/env python3
"""Phase 12B-FC V1 — GPU / model residency inspector.

Records nvidia-smi state, Ollama residency (VRAM / expires), and model load
behaviour before/after inference so the 12B validation can prove the model is
GPU-resident and not CPU-offloaded.

Run on 153:
  PYTHONPATH=. .venv/bin/python scripts/maintenance/inspect_gpu_model_residency.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")


def _nvidia_smi():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        return {"ok": out.returncode == 0, "raw": out.stdout.strip(),
                "error": out.stderr.strip()[:200] if out.returncode else None}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _ollama_ps():
    import httpx
    try:
        ps = httpx.get(f"{BASE}/api/ps", timeout=10).json()
        return {"ok": True, "models": [{
            "name": m.get("name"), "size_vram": m.get("size_vram"),
            "size_total": m.get("size"), "expires_at": m.get("expires_at"),
            "context_length": m.get("context_length"),
        } for m in ps.get("models", [])]}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=str(REPO_ROOT / "docs" / "baseline" / "sentrix-12b-gpu-residency.json"))
    args = parser.parse_args()

    import httpx
    report = {"base_url": BASE}
    report["nvidia_smi_before"] = _nvidia_smi()
    report["ollama_ps_before"] = _ollama_ps()
    try:
        resp = httpx.post(f"{BASE}/api/chat", json={
            "model": os.getenv("SENTRIX_PARSE_MODEL", "gemma4:12b"),
            "messages": [{"role": "user", "content": "你好，简短回一句"}],
            "stream": False, "options": {"num_predict": 32},
        }, timeout=120)
        report["inference"] = {"ok": resp.status_code == 200,
                               "actual_model": resp.json().get("model")}
    except Exception as exc:
        report["inference"] = {"ok": False, "error": str(exc)}
    report["nvidia_smi_after"] = _nvidia_smi()
    report["ollama_ps_after"] = _ollama_ps()

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
