#!/usr/bin/env python3
"""Probe the bge-m3 text embedder sidecar (Phase R9-4).

Health check + a sample embedding round-trip.  Exits non-zero when the sidecar
is down so orchestration can alert / restart it.
"""

from __future__ import annotations

import json
import os
import sys

try:
    import httpx
except Exception:
    httpx = None


def main():
    base = os.getenv("SENTRIX_TEXT_EMBEDDER_URL", "http://127.0.0.1:8101").rstrip("/")
    if httpx is None:
        print(json.dumps({"status": "error", "detail": "httpx not installed"}))
        return 1
    try:
        health = httpx.get(f"{base}/health", timeout=2.0)
        health.raise_for_status()
        body = health.json()
    except Exception as exc:
        print(json.dumps({"status": "down", "detail": str(exc)}))
        return 1
    if body.get("status") != "ok":
        print(json.dumps({"status": "loading", **body}))
        return 2
    try:
        embed = httpx.post(f"{base}/embed", json={"text": "厨房做晚饭"}, timeout=5.0)
        embed.raise_for_status()
        dim = len(embed.json().get("vector") or [])
    except Exception as exc:
        print(json.dumps({"status": "degraded", "detail": str(exc), **body}))
        return 3
    print(json.dumps({"status": "ok", "model": body.get("model"),
                      "dimension": body.get("dimension"), "sample_dim": dim}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
