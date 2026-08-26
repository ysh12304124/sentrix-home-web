"""Finalize a package parent after all representative images are processed."""

from __future__ import annotations

import argparse
import json
import os

from backend.db import MemoryStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=os.getenv("SENTRIX_DB_PATH", "data/sentrix.db"))
    parser.add_argument("--asset-id", required=True)
    args = parser.parse_args()
    store = MemoryStore(args.db)
    parent = store.get_asset(args.asset_id)
    derived = store.list_derived_assets(args.asset_id)
    failed = [item for item in derived if item.get("status") != "processed"]
    if failed:
        raise RuntimeError(f"unprocessed derived assets: {len(failed)}")
    events = store._rows("SELECT id FROM events WHERE source_asset_id = ? ORDER BY source_scene_index", (args.asset_id,))
    metadata = dict(parent.get("metadata_json") or {})
    metadata.update({
        "video_stage": "processed", "keyframe_video_memory": True,
        "keyframe_video_memory_format": "webp", "keyframe_video_memory_frame_count": len(derived),
        "video_scene_event_ids": [item["id"] for item in events],
        "derived_keyframe_asset_ids": [item["id"] for item in derived],
        "error": None, "error_stage": None, "retryable": True,
    })
    store.update_asset(args.asset_id, "processed", metadata)
    print(json.dumps({"asset_id": args.asset_id, "events": len(events), "derived": len(derived)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
