"""Register and process a direct-source event WebP package in Sentrix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from backend.app import pipeline
from backend.db import MemoryStore, make_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=os.getenv("SENTRIX_DB_PATH", "data/sentrix.db"))
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    store = MemoryStore(args.db)
    store.create_memory_space(args.scope, args.label, kind="qa")
    frame_map = json.loads((args.package / "frame_map.json").read_text(encoding="utf-8"))
    asset_id = make_id("asset")
    descriptor = {
        "frame_map_path": str((args.package / "frame_map.json").resolve()),
        "semantic_path": str((args.package / "semantic.json").resolve()),
        "encoded_fps": frame_map.get("encoded_fps", 1.0),
        "source_fps": frame_map.get("source_fps", 0.0),
        "source_width": frame_map.get("source_width", 0),
        "source_height": frame_map.get("source_height", 0),
        "direct_source": True,
    }
    store.create_asset(
        asset_id, args.video.name, "video", str(args.video.resolve()), "video/mp4", args.video.stat().st_size,
        {"scope_id": args.scope, "keyframe_video_package": descriptor, "hippo_qa": True},
        scope_id=args.scope,
    )
    print(f"asset={asset_id} scope={args.scope} events={len(frame_map.get('frames') or [])}", flush=True)
    result = pipeline.process(asset_id)
    print(json.dumps({"asset_id": asset_id, "status": result.get("status"), "metadata": result.get("metadata_json")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
