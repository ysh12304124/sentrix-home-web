"""Backfill visual CLIP vectors for already-processed image assets."""

from __future__ import annotations

import argparse
import os
import time

from backend.db import MemoryStore
from backend.model_clients import ClipAdapter


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=os.getenv("SENTRIX_DB_PATH", "data/sentrix.db"))
    parser.add_argument("--parent-asset-id", action="append", required=True)
    args = parser.parse_args()

    store = MemoryStore(args.db)
    clip = ClipAdapter()
    assets = []
    for parent_asset_id in args.parent_asset_id:
        assets.extend(
            asset for asset in store.list_derived_assets(parent_asset_id)
            if asset.get("media_type") == "image"
        )
    visual_model_name = getattr(clip, "visual_model_name", clip.model_name)
    print(f"assets={len(assets)} model={visual_model_name} device={clip.device}", flush=True)
    written = 0
    started = time.perf_counter()
    for index, asset in enumerate(assets, 1):
        vector = clip.embed_image(asset["path"])
        if not vector:
            print(f"failed={asset['id']} error={clip.error}", flush=True)
            continue
        observation = store.connection.execute(
            "SELECT id FROM observations WHERE asset_id = ?", (asset["id"],)
        ).fetchone()
        metadata = {"asset_id": asset["id"], "backfilled": True}
        if observation:
            metadata["observation_id"] = observation[0]
        store.connection.execute(
            "DELETE FROM memory_vectors WHERE space = 'visual' AND source_type = 'asset' AND source_id = ?",
            (asset["id"],),
        )
        store.connection.commit()
        store.upsert_vector("visual", "asset", asset["id"], vector, visual_model_name, metadata)
        written += 1
        if index == 1 or index % 10 == 0 or index == len(assets):
            elapsed = time.perf_counter() - started
            print(f"progress={index}/{len(assets)} written={written} elapsed={elapsed:.1f}s", flush=True)
    print(f"done written={written} elapsed={time.perf_counter() - started:.1f}s", flush=True)
    return 0 if written == len(assets) else 1


if __name__ == "__main__":
    raise SystemExit(main())
