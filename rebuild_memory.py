#!/usr/bin/env python3
"""Discard derived Sentrix memory and rebuild it from a source directory."""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.db import MemoryStore
from backend.model_clients import ClipAdapter, FaceAdapter, FunASRClient, GammaClient
from backend.pipeline import IngestionPipeline


SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".wav", ".mp3", ".m4a", ".flac", ".txt", ".md", ".json"}


def rebuild(root, source):
    data_dir = root / "data"
    db_path = data_dir / "sentrix.db"
    media_dir = data_dir / "media"
    if db_path.exists():
        db_path.unlink()
    if media_dir.exists():
        shutil.rmtree(media_dir)
    media_dir.mkdir(parents=True, exist_ok=True)

    store = MemoryStore(str(db_path))
    pipeline = IngestionPipeline(store, gamma=GammaClient(), asr=FunASRClient(), face=FaceAdapter(), clip=ClipAdapter())
    files = sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED)
    processed = 0
    failed = 0
    for path in files:
        asset = pipeline.create_asset(path)
        result = pipeline.process(asset["id"])
        if result.get("status") == "failed":
            failed += 1
            print(f"FAILED {path}: {result.get('metadata_json', result)}")
        else:
            processed += 1
            print(f"OK {processed}/{len(files)} {path}")
    print({"files": len(files), "processed": processed, "failed": failed, "assets": store.count("assets"), "observations": store.count("observations"), "events": store.count("events"), "entities": store.count("entities"), "clusters": store.count("face_clusters"), "facts": store.count("facts")})
    store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source", type=Path, default=None)
    args = parser.parse_args()
    source = args.source or (args.root / "data" / "test-albums")
    rebuild(args.root, source)
