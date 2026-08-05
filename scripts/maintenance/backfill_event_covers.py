#!/usr/bin/env python3
"""Explicitly backfill evidence-backed event covers for an existing SQLite database."""

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.db import MemoryStore


def backup_database(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source) as input_connection, sqlite3.connect(destination) as output_connection:
        input_connection.backup(output_connection)


def backfill_event_covers(store, apply=False, scope_id=None):
    events = store.list_events(100000, scope_id=scope_id)
    eligible = []
    skipped = {"selected": 0, "no_image": 0}
    for event in events:
        selection = event.get("cover_selection") or {}
        if selection.get("source"):
            skipped["selected"] += 1
            continue
        image_count = sum(
            1 for observation_id in event.get("observation_ids", [])
            if (store.get_asset((store.get_observation(observation_id) or {}).get("asset_id")) or {}).get("media_type") == "image"
        )
        if not image_count:
            skipped["no_image"] += 1
            continue
        eligible.append(event["id"])
    updated = 0
    if apply:
        for event_id in eligible:
            selected = store.select_event_cover(event_id)
            if (selected or {}).get("cover_selection", {}).get("source") == "derived":
                updated += 1
    return {"scanned": len(events), "eligible": len(eligible), "updated": updated, "skipped": skipped, "scope_id": scope_id}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path(__file__).resolve().parents[2] / "data" / "sentrix.db")
    parser.add_argument("--scope-id")
    parser.add_argument("--apply", action="store_true", help="write only missing derived cover selections")
    parser.add_argument("--backup", type=Path, help="required with --apply; SQLite backup destination")
    args = parser.parse_args()
    if args.apply and not args.backup:
        parser.error("--apply requires --backup so the prior derived state can be restored")
    if args.apply:
        backup_database(args.database, args.backup)
    store = MemoryStore(str(args.database))
    try:
        print(json.dumps(backfill_event_covers(store, args.apply, args.scope_id), ensure_ascii=False))
    finally:
        store.close()


if __name__ == "__main__":
    main()
