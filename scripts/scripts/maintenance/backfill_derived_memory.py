#!/usr/bin/env python3
"""Backfill visual/text vectors and face instances for already-processed assets.

Use after a directory scan where Gemma observations exist but CLIP/face were
unavailable (missing weights, HEIC unread by OpenCV, etc.).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.db import MemoryStore
from backend.embeddings.chinese_clip_visual import ChineseClipVisualEmbedder
from backend.image_io import ensure_heif_support
from backend.model_clients import ClipAdapter, FaceAdapter


def observation_text(observation: dict) -> str:
    clothing = " ".join(str(item) for item in (observation.get("clothing") or []))
    fact_text = ""
    raw = observation.get("raw_json") or observation.get("raw") or {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = {}
    gamma = raw.get("gamma") if isinstance(raw, dict) else {}
    facts = gamma.get("facts") if isinstance(gamma, dict) else observation.get("facts") or []
    if isinstance(facts, list):
        fact_text = " ".join(
            f"{item.get('subject', '')} {item.get('predicate', '')} {item.get('object', '')}"
            for item in facts
            if isinstance(item, dict)
        )
    return " ".join(
        filter(
            None,
            [
                observation.get("caption"),
                observation.get("activity"),
                observation.get("place"),
                observation.get("ocr_text"),
                observation.get("transcript"),
                clothing,
                fact_text,
            ],
        )
    )


def backfill(store: MemoryStore, *, scope_id: str | None, apply: bool) -> dict:
    ensure_heif_support()
    visual = ChineseClipVisualEmbedder()
    if not visual.available:
        raise RuntimeError(f"chinese_clip unavailable: {visual._error}")
    clip = ClipAdapter()
    if not clip.evidence_ready:
        # Force a load attempt so missing checkpoints surface clearly.
        clip.embed_text("probe")
        if not clip.evidence_ready:
            raise RuntimeError(f"clip text embedder unavailable: {clip.error}")
    face = FaceAdapter()
    assets = store.list_assets(media_type="image", limit=100_000, scope_id=scope_id)
    summary = {
        "assets": len(assets),
        "visual": 0,
        "text": 0,
        "faces": 0,
        "failed": 0,
        "apply": apply,
        "scope_id": scope_id,
    }
    for index, asset in enumerate(assets, start=1):
        path = asset.get("path") or ""
        asset_id = asset["id"]
        metadata = asset.get("metadata_json") or {}
        observation_id = metadata.get("observation_id")
        event_id = metadata.get("event_id")
        observation = store.get_observation(observation_id) if observation_id else None
        if not Path(path).is_file() or not observation:
            summary["failed"] += 1
            print(f"SKIP {index}/{len(assets)} {asset.get('file_name')} missing file/observation")
            continue
        try:
            visual_vector = visual.embed_image(path)
            text_vector = clip.embed_text(observation_text(observation))
            faces = face.detect(path)
            if not apply:
                summary["visual"] += 1 if visual_vector else 0
                summary["text"] += 1 if text_vector else 0
                summary["faces"] += len(faces)
                print(
                    f"DRY {index}/{len(assets)} {asset.get('file_name')} "
                    f"visual={len(visual_vector)} text={len(text_vector)} faces={len(faces)}"
                )
                continue
            if visual_vector:
                store.upsert_vector(
                    "visual",
                    "asset",
                    asset_id,
                    visual_vector,
                    visual.model_id,
                    {"observation_id": observation_id, "event_id": event_id, "scope_id": asset.get("scope_id")},
                )
                summary["visual"] += 1
            if text_vector:
                store.upsert_vector(
                    "episodic",
                    "observation",
                    observation_id,
                    text_vector,
                    clip.model_name,
                    {"asset_id": asset_id, "event_id": event_id, "scope_id": asset.get("scope_id")},
                )
                if event_id:
                    store.upsert_vector(
                        "episodic",
                        "event",
                        event_id,
                        text_vector,
                        clip.model_name,
                        {"observation_id": observation_id, "scope_id": asset.get("scope_id")},
                    )
                store.upsert_vector(
                    "semantic",
                    "observation",
                    observation_id,
                    text_vector,
                    clip.model_name,
                    {"asset_id": asset_id, "event_id": event_id, "scope_id": asset.get("scope_id")},
                )
                summary["text"] += 1
            cluster_ids = []
            saved_faces = 0
            for candidate in faces:
                instance = store.add_face_instance(asset_id, observation_id, candidate)
                if not instance:
                    continue
                saved_faces += 1
                if instance.get("cluster_id"):
                    cluster_ids.append(instance["cluster_id"])
            summary["faces"] += saved_faces
            if cluster_ids:
                metadata = dict(metadata)
                metadata["cluster_ids"] = list(dict.fromkeys((metadata.get("cluster_ids") or []) + cluster_ids))
                store.update_asset(asset_id, asset.get("status") or "processed", metadata)
            print(
                f"OK {index}/{len(assets)} {asset.get('file_name')} "
                f"visual={len(visual_vector)} text={len(text_vector)} faces={len(faces)}"
            )
        except Exception as error:
            summary["failed"] += 1
            print(f"FAILED {index}/{len(assets)} {asset.get('file_name')}: {error}")
    if apply:
        summary["recluster"] = store.recluster_faces(scope_id=scope_id)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=os.getenv("SENTRIX_DB_PATH", "data/sentrix.db"))
    parser.add_argument("--scope-id", default="album")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    store = MemoryStore(args.db)
    try:
        print(json.dumps(backfill(store, scope_id=args.scope_id, apply=args.apply), ensure_ascii=False, indent=2))
    finally:
        store.close()


if __name__ == "__main__":
    main()
