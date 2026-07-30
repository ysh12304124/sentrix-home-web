#!/usr/bin/env python3
"""Discard derived Sentrix memory and rebuild it from a source directory."""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.db import MemoryStore
from backend.model_clients import ClipAdapter, FaceAdapter, FunASRClient, GammaClient
from backend.pipeline import IngestionPipeline


SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".wav", ".mp3", ".m4a", ".flac", ".txt", ".md"}


def rebuild(root, source):
    face = FaceAdapter()
    if face.enabled and face.identity_model in {"adaface", "magface"} and not face.identity_configured:
        raise RuntimeError(
            f"{face.identity_model} identity embedding is not configured: "
            f"{face.identity_error or 'missing model configuration'}"
        )
    data_dir = root / "data"
    db_path = data_dir / "sentrix.db"
    media_dir = data_dir / "media"
    if db_path.exists():
        db_path.unlink()
    if media_dir.exists():
        shutil.rmtree(media_dir)
    media_dir.mkdir(parents=True, exist_ok=True)

    store = MemoryStore(str(db_path))
    run = store.start_rebuild("sentrix-rebuild-v1", str(source))
    pipeline = IngestionPipeline(store, gamma=GammaClient(), asr=FunASRClient(), face=face, clip=ClipAdapter())
    metadata_path = source / "sentrix_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    files = sorted(path for path in source.rglob("*") if path.is_file() and path != metadata_path and path.suffix.lower() in SUPPORTED)
    processed = 0
    failed = 0
    for path in files:
        asset = pipeline.create_asset(path, metadata=metadata.get(path.name))
        result = pipeline.process(asset["id"], summarize_event=False)
        if result.get("status") == "failed":
            failed += 1
            print(f"FAILED {path}: {result.get('metadata_json', result)}")
        else:
            processed += 1
            print(f"OK {processed}/{len(files)} {path}")
    event_summaries = pipeline.summarize_events()
    recluster = store.recluster_faces()
    stats = {"files": len(files), "processed": processed, "failed": failed, "assets": store.count("assets"), "observations": store.count("observations"), "events": store.count("events"), "event_summaries": len(event_summaries), "entities": store.count("entities"), "clusters": store.count("face_clusters"), "facts": store.count("facts"), "recluster": recluster}
    store.finish_rebuild(run["id"], "completed" if failed == 0 else "completed_with_failures", stats)
    print(stats)
    store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--source", type=Path, default=None)
    args = parser.parse_args()
    source = args.source or (args.root / "data" / "test-albums")
    rebuild(args.root, source)
