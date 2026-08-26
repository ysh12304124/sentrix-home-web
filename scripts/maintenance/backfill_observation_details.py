#!/usr/bin/env python3
"""Backfill the versioned observation detail projection without model calls.

This migration only rehydrates fields already present in canonical/raw JSON. It
never invents visual facts; a later opt-in VLM re-enrichment can add details.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.db import MemoryStore, json_value, now_iso


def project(observation: dict) -> dict:
    raw = observation.get("raw") or {}
    source = raw.get("gamma") if isinstance(raw, dict) else {}
    source = source if isinstance(source, dict) else {}
    canonical = observation.get("canonical") or {}
    detail = {
        "schema_version": 1,
        "backfill_source": "existing_observation",
        "caption": observation.get("caption") or source.get("caption") or canonical.get("caption") or "",
        "activity": observation.get("activity") or source.get("activity") or canonical.get("activity") or "",
        "place": observation.get("place") or source.get("place") or canonical.get("place") or "",
        "people": observation.get("people") or source.get("people") or canonical.get("people") or [],
        "objects": observation.get("objects") or source.get("objects") or canonical.get("objects") or [],
        "clothing": observation.get("clothing") or source.get("clothing") or canonical.get("clothing") or [],
        "spatial_relations": observation.get("spatial_relations") or source.get("spatial_relations") or canonical.get("spatial_relations") or [],
        "ocr_text": observation.get("ocr_text") or source.get("ocr_text") or canonical.get("ocr_text") or "",
        "event_type": observation.get("event_type") or source.get("event_type") or canonical.get("event_type") or "",
        "facts": source.get("facts") or [],
        "legacy_raw_fields": {
            key: value for key, value in source.items()
            if key not in {"caption", "activity", "place", "people", "objects", "clothing",
                           "spatial_relations", "ocr_text", "event_type", "facts"}
        },
    }
    return detail


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/sentrix.db")
    parser.add_argument("--scope", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    store = MemoryStore(args.db)
    rows = store.list_observations(args.limit or 1_000_000, scope_id=args.scope or None)
    candidates = [row for row in rows if not (row.get("detail") or {}).get("schema_version")]
    if args.apply:
        with store.transaction():
            for row in candidates:
                detail = project(row)
                store.connection.execute(
                    "UPDATE observations SET detail_json = ?, revision = revision + 1, updated_at = ? WHERE id = ?",
                    (json_value(detail, {}), now_iso(), row["id"]),
                )
        status = "applied"
    else:
        status = "dry_run"
    print(json.dumps({
        "status": status,
        "scope": args.scope or None,
        "scanned": len(rows),
        "candidates": len(candidates),
        "would_update": len(candidates),
        "db": str(Path(args.db).resolve()),
    }, ensure_ascii=False))
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
