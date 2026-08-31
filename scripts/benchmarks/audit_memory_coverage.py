#!/usr/bin/env python3
"""Audit what survives from assets/observations into searchable memory.

This is deliberately read-only: it measures field coverage and index coverage
without calling a model or rewriting the database.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.db import MemoryStore


FIELDS = (
    "caption", "activity", "place", "people", "objects", "clothing",
    "spatial_relations", "ocr_text", "transcript", "detail",
)


def _has(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)


def audit(store: MemoryStore, scope: str | None = None) -> dict:
    observations = store.list_observations(limit=1_000_000, scope_id=scope)
    assets = store.list_assets(limit=1_000_000, scope_id=scope)
    field_counts = Counter()
    detail_versions = Counter()
    media_counts = Counter(str(a.get("media_type") or "unknown") for a in assets)
    for observation in observations:
        for field in FIELDS:
            if _has(observation.get(field)):
                field_counts[field] += 1
        detail = observation.get("detail") or {}
        if detail:
            detail_versions[str(detail.get("schema_version") or "legacy")] += 1
    n = len(observations)
    index_terms = 0
    try:
        row = store.connection.execute(
            "SELECT COUNT(DISTINCT observation_id) AS n FROM observation_search_terms" +
            (" WHERE scope_id = ?" if scope else ""),
            (scope,) if scope else (),
        ).fetchone()
        index_terms = int(row["n"] if row else 0)
    except Exception:
        index_terms = 0
    return {
        "scope_id": scope,
        "assets": len(assets),
        "observations": n,
        "media_types": dict(media_counts),
        "field_coverage": {
            field: {"count": field_counts[field],
                    "rate": round(field_counts[field] / n, 4) if n else None}
            for field in FIELDS
        },
        "detail_schema_versions": dict(detail_versions),
        "observation_search_terms": index_terms,
        "index_coverage_rate": round(index_terms / n, 4) if n else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--scope", default=None)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    store = MemoryStore(str(args.db))
    try:
        payload = audit(store, args.scope)
    finally:
        store.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
