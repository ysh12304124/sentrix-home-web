#!/usr/bin/env python3
"""Build an isolated, face-only household benchmark database.

Evaluation labels in the manifest are intentionally never persisted.  This
imports only original image paths, scope provenance and model face outputs.
"""

import argparse
import json
import mimetypes
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.db import MemoryStore, make_id
from backend.model_clients import FaceAdapter


def ingest(manifest_path, database_path, face=None, threshold=0.30, minimum_quality=0.55):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    store = MemoryStore(str(database_path))
    face = face or FaceAdapter()
    processed = failed = detections = 0
    try:
        for space in manifest.get("spaces", []):
            scope_id = space["scope_id"]
            store.create_memory_space(scope_id, scope_id, kind="benchmark")
            for record in space.get("import", {}).get("files", []):
                path = Path(manifest["source_root"]) / record["relative_path"]
                asset = store.create_asset(
                    make_id("asset"), record["file_name"], "image", str(path),
                    mimetypes.guess_type(path.name)[0], path.stat().st_size,
                    {"scope_id": scope_id, "captured_at": record.get("captured_at"),
                     "captured_location": record.get("captured_location"),
                     "source_album_id": record.get("source_album_id")},
                    scope_id=scope_id,
                )
                observation = store.add_observation(
                    asset["id"], {"scope_id": scope_id, "captured_at": record.get("captured_at"),
                                  "source_type": "household_face_benchmark"}, scope_id=scope_id,
                )
                try:
                    rows = face.detect(path)
                    for row in rows:
                        store.add_face_instance(asset["id"], observation["id"], row)
                    store.update_asset(asset["id"], "processed", {
                        "benchmark": "household-face-only", "face_count": len(rows),
                    })
                    processed += 1
                    detections += len(rows)
                except Exception as error:
                    store.cleanup_asset_derivatives(asset["id"])
                    store.update_asset(asset["id"], "failed", {"benchmark": "household-face-only", "error": str(error)})
                    failed += 1
        recluster = {
            space["scope_id"]: store.recluster_faces(threshold=threshold, minimum_quality=minimum_quality, scope_id=space["scope_id"])
            for space in manifest.get("spaces", [])
        }
        return {"processed": processed, "failed": failed, "detections": detections, "recluster": recluster}
    finally:
        store.close()


def main():
    parser = argparse.ArgumentParser(description="Build an isolated household face benchmark database.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("database", type=Path)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--minimum-quality", type=float, default=0.55)
    args = parser.parse_args()
    print(json.dumps(ingest(args.manifest, args.database, threshold=args.threshold, minimum_quality=args.minimum_quality), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
