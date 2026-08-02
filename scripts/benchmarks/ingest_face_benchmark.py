#!/usr/bin/env python3
"""Import a manifest-backed face benchmark into an explicitly isolated SQLite DB."""

import argparse
import json
import mimetypes
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.db import MemoryStore, make_id
from backend.model_clients import FaceAdapter


def _manifest_files(source, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    records = manifest.get("assets", [])
    if not records:
        raise ValueError("benchmark manifest has no assets")
    paths = []
    for record in records:
        path = Path(source) / str(record.get("file") or "")
        if not path.is_file():
            raise FileNotFoundError(f"benchmark image not found: {path}")
        paths.append(path)
    return paths


def ingest(database, source, manifest_path, face=None, limit=None):
    """Store detector output only; ground-truth identity stays in the manifest."""
    database = Path(database)
    source = Path(source)
    files = _manifest_files(source, manifest_path)
    if limit is not None:
        files = files[:max(0, int(limit))]
    store = MemoryStore(str(database))
    face = face or FaceAdapter()
    processed = faces = failed = 0
    try:
        for path in files:
            asset_id = make_id("asset")
            asset = store.create_asset(
                asset_id, path.name, "image", str(path), mimetypes.guess_type(path.name)[0], path.stat().st_size,
                {"benchmark": "lfw", "manifest": Path(manifest_path).name},
                scope_id="benchmark-lfw",
            )
            observation = store.add_observation(
                asset_id,
                {"source_type": "face_benchmark", "caption": "LFW 人脸聚类评估样本", "confidence": 1.0, "scope_id": "benchmark-lfw"},
                scope_id="benchmark-lfw",
            )
            try:
                detected = face.detect(path)
                cluster_ids = []
                for item in detected:
                    result = store.add_face_instance(asset_id, observation["id"], item)
                    if result and result.get("cluster_id"):
                        cluster_ids.append(result["cluster_id"])
                store.update_asset(asset_id, "processed", {"benchmark": "lfw", "observation_id": observation["id"], "cluster_ids": cluster_ids, "face_count": len(detected)})
                processed += 1
                faces += len(detected)
            except Exception as error:
                store.cleanup_asset_derivatives(asset_id)
                store.update_asset(asset_id, "failed", {"benchmark": "lfw", "error": str(error)})
                failed += 1
        recluster = store.recluster_faces(scope_id="benchmark-lfw")
        return {"processed": processed, "failed": failed, "faces": faces, "clusters": store.count("face_clusters"), "recluster": recluster}
    finally:
        store.close()


def main():
    parser = argparse.ArgumentParser(description="Import face benchmark samples into an isolated SQLite database.")
    parser.add_argument("--db", type=Path, required=True, help="New or isolated benchmark SQLite database; never the production database.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(json.dumps(ingest(args.db, args.source, args.manifest, limit=args.limit), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
