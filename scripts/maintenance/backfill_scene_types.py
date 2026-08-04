#!/usr/bin/env python3
"""Backfill model-selected scene types for existing image observations.

Dry-run is the default. Applying requires a SQLite backup because it updates
derived Observation fields and rebuilds only their non-person entity links.
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.db import MemoryStore
from backend.model_clients import GammaClient


def backup_database(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source) as input_connection, sqlite3.connect(destination) as output_connection:
        input_connection.backup(output_connection)


def missing_scene_type(observation):
    scene = (observation.get("canonical") or {}).get("scene_type")
    return not scene or scene == "其他或不确定"


def backfill_scene_types(store, gamma, apply=False, scope_id=None, limit=None):
    observations = [
        item for item in store.list_observations(100000, scope_id=scope_id)
        if (store.get_asset(item["asset_id"]) or {}).get("media_type") == "image" and missing_scene_type(item)
    ]
    if limit is not None:
        observations = observations[:limit]
    result = {"scope_id": scope_id, "scanned": len(observations), "updated": 0, "skipped": 0, "failed": 0}
    if not apply:
        return result
    for observation in observations:
        asset = store.get_asset(observation["asset_id"]) or {}
        if not asset.get("path"):
            result["skipped"] += 1
            continue
        try:
            analysis = gamma.analyze_image(asset["path"], {
                "file_name": asset.get("file_name"), "captured_at": asset.get("captured_at"),
                "captured_location": asset.get("captured_location") or "", "source_owner_id": asset.get("source_owner_id"),
            })
            store.enrich_observation(observation["id"], analysis, source="scene_type_backfill")
            event = store._row("SELECT event_id FROM event_observations WHERE observation_id = ? LIMIT 1", (observation["id"],))
            store.maintain_observation_entities(observation["id"], event["event_id"] if event else None)
            result["updated"] += 1
        except Exception:
            result["failed"] += 1
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path(__file__).resolve().parents[2] / "data" / "sentrix.db")
    parser.add_argument("--scope-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup", type=Path, help="required with --apply")
    args = parser.parse_args()
    if args.apply and not args.backup:
        parser.error("--apply requires --backup")
    if args.apply:
        backup_database(args.database, args.backup)
    store = MemoryStore(str(args.database))
    try:
        print(json.dumps(backfill_scene_types(store, GammaClient(), args.apply, args.scope_id, args.limit), ensure_ascii=False))
    finally:
        store.close()


if __name__ == "__main__":
    main()
