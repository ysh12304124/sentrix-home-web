#!/usr/bin/env python3
"""Measure the complete image-memory pipeline in an isolated SQLite database."""

import argparse
import hashlib
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.db import MemoryStore
from backend.model_clients import ClipAdapter, FaceAdapter, GammaClient
from backend.pipeline import IngestionPipeline
from scripts.maintenance.rebuild_memory import benchmark_imports


def _fingerprint(files):
    digest = hashlib.sha256()
    for scope_id, path, _ in files:
        digest.update(scope_id.encode())
        digest.update(path.name.encode())
        digest.update(str(path.stat().st_size).encode())
    return digest.hexdigest()


def evaluate(manifest_path, baseline_seconds, limit=None, keep_alive="15m"):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    files = list(benchmark_imports(manifest))
    if limit is not None:
        files = files[:max(0, int(limit))]
    if not files:
        raise ValueError("manifest has no importable files")
    face = FaceAdapter()
    if face.enabled and not face.identity_configured:
        raise RuntimeError(f"identity model unavailable: {face.identity_error}")
    with tempfile.TemporaryDirectory(prefix="sentrix-e2e-") as directory:
        store = MemoryStore(str(Path(directory) / "benchmark.db"))
        gamma = GammaClient(keep_alive=keep_alive)
        pipeline = IngestionPipeline(store, gamma=gamma, face=face, clip=ClipAdapter())
        rows = []
        overall_started = time.perf_counter()
        try:
            for scope_id, path, metadata in files:
                asset = pipeline.create_asset(path, metadata=metadata)
                started = time.perf_counter()
                result = pipeline.process(asset["id"], summarize_event=False)
                elapsed = time.perf_counter() - started
                rows.append({
                    "scope_id": scope_id, "file_name": path.name, "seconds": round(elapsed, 4),
                    "status": result.get("status"),
                    "timings": (result.get("metadata_json") or {}).get("processing_timings", {}),
                })
            for scope_id in sorted({item[0] for item in files}):
                store.consolidate_events(scope_id)
                store.recluster_faces(scope_id=scope_id)
            completed = [row["seconds"] for row in rows if row["status"] == "processed"]
            mean = statistics.mean(completed) if completed else None
            speedup = baseline_seconds / mean if mean else 0.0
            return {
                "dataset_fingerprint": _fingerprint(files), "files_requested": len(files),
                "files_processed": len(completed), "files_failed": len(files) - len(completed),
                "overall_seconds": round(time.perf_counter() - overall_started, 4),
                "mean_seconds": round(mean, 4) if mean else None,
                "p95_seconds": round(sorted(completed)[max(0, int(len(completed) * .95) - 1)], 4) if completed else None,
                "baseline_seconds": baseline_seconds, "speedup": round(speedup, 4),
                "minimum_speedup": 5.0, "keep_alive": keep_alive,
                "pending_event_summaries": store.connection.execute(
                    "SELECT COUNT(*) FROM events WHERE title = '待总结事件'"
                ).fetchone()[0],
                "passed": len(completed) == len(files) and speedup >= 5.0,
                "rows": rows,
            }
        finally:
            store.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate complete household image processing in an isolated database.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--baseline-seconds", type=float, required=True, help="Measured legacy mean seconds per image, not an estimate.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--keep-alive", default="15m")
    args = parser.parse_args()
    result = evaluate(args.manifest, args.baseline_seconds, args.limit, args.keep_alive)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
