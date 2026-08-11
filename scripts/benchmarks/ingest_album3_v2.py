#!/usr/bin/env python3
"""导入 album3-v2 到 153 sentrix.db（只导原始照片 + EXIF 元数据，分析全部走生产管线）。

- 创建 scope album3-v2
- 对每张照片：create_asset（EXIF captured_at/GPS 由管线自动提取）+ pipeline.process
- 全部完成后：consolidate_events + summarize_events + recluster_faces

注意：本脚本不含任何 benchmark 答案/图片 ID 内容，只有数据导入逻辑。

用法（在 153 上，repo 根目录）:
  FACE_EMBEDDING_MODE=legacy SENTRIX_DB_PATH=data/sentrix.db \
  .venv/bin/python scripts/benchmarks/ingest_album3_v2.py \
    --source /home/asus/Github/Sentrix-Home-Web/data/album3-v2-source/photos \
    --scope album3-v2
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.db import MemoryStore
from backend.model_clients import ClipAdapter, FaceAdapter, GammaClient
from backend.pipeline import IngestionPipeline

SCOPE = "album3-v2"
DB_PATH = os.getenv("SENTRIX_DB_PATH", "")


def ingest(source_dir, scope_id, database_path, limit=0):
    photos = sorted(p for p in Path(source_dir).iterdir() if p.is_file())
    if limit:
        photos = photos[:limit]
    store = MemoryStore(database_path)
    store.create_memory_space(scope_id, scope_id, kind="benchmark",
                              source_path=str(source_dir))
    gamma = GammaClient()
    face = FaceAdapter()
    clip = ClipAdapter()
    print(f"face enabled={face.enabled} identity_configured={face.identity_configured} "
          f"identity_error={face.identity_error}")
    pipeline = IngestionPipeline(store, gamma=gamma, face=face, clip=clip)
    rows = []
    t0 = time.time()
    for i, path in enumerate(photos, 1):
        started = time.perf_counter()
        asset = pipeline.create_asset(path, metadata={"scope_id": scope_id,
                                                      "source_album_id": scope_id})
        if asset.get("status") == "processed":
            print(f"[{i}/{len(photos)}] {path.name} already processed, skip")
            continue
        result = pipeline.process(asset["id"], summarize_event=False)
        seconds = round(time.perf_counter() - started, 2)
        status = result.get("status")
        meta = result.get("metadata_json") or {}
        faces = len(meta.get("faces") or [])
        rows.append({"file": path.name, "status": status, "seconds": seconds, "faces": faces,
                     "event_id": meta.get("event_id")})
        print(f"[{i}/{len(photos)}] {path.name} -> {status} {seconds}s faces={faces}")
    elapsed = round(time.time() - t0, 1)
    merged = store.consolidate_events(scope_id)
    print(f"consolidate_events merged {len(merged)} pairs")
    pipeline.summarize_events(scope_id)
    recluster = store.recluster_faces(scope_id=scope_id)
    print(f"recluster_faces: {recluster}")
    ok = sum(1 for r in rows if r["status"] == "processed")
    print(f"DONE {ok}/{len(rows)} processed in {elapsed}s")
    return {"imported": len(rows), "processed": ok, "elapsed_s": elapsed, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--scope", default=SCOPE)
    ap.add_argument("--db", default=DB_PATH or str(Path(__file__).resolve().parents[2] / "data" / "sentrix.db"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if not Path(args.db).is_file():
        sys.exit(f"db not found: {args.db}")
    result = ingest(args.source, args.scope, args.db, args.limit)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
