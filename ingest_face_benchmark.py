#!/usr/bin/env python3
"""Append a face-only benchmark without fabricating family events or facts."""

import argparse
import mimetypes
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.db import MemoryStore, make_id
from backend.model_clients import FaceAdapter


def ingest(root, source, limit):
    store = MemoryStore(str(root / "data" / "sentrix.db"))
    face = FaceAdapter()
    files = sorted(source.rglob("*.jpg"))[:limit]
    processed = 0
    faces = 0
    for path in files:
        asset_id = make_id("asset")
        asset = store.create_asset(asset_id, path.name, "image", str(path), mimetypes.guess_type(path.name)[0], path.stat().st_size, {"benchmark": "lfw", "source": "LFW"})
        observation = store.add_observation(asset_id, {"source_type": "face_benchmark", "caption": "LFW 人脸聚类评估样本", "confidence": 1.0, "raw": {"benchmark": "lfw", "model": "buffalo_l"}})
        detected = face.detect(path)
        cluster_ids = []
        for item in detected:
            result = store.add_face_instance(asset_id, observation["id"], item, model_name="buffalo_l")
            cluster_ids.append(result["cluster_id"])
        store.update_asset(asset_id, "processed", {"benchmark": "lfw", "observation_id": observation["id"], "cluster_ids": cluster_ids, "face_count": len(detected)})
        processed += 1
        faces += len(detected)
        if processed % 50 == 0:
            print({"processed": processed, "faces": faces, "clusters": store.count("face_clusters")})
    print({"processed": processed, "faces": faces, "clusters": store.count("face_clusters"), "entities": store.count("entities")})
    store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=300)
    args = parser.parse_args()
    ingest(args.root, args.source, args.limit)
