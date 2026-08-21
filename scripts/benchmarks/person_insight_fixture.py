"""Build a strictly isolated verification copy of a memory space.

The copy is produced with the SQLite backup API (never file-level copying) and
only mutated on the working database. The source database is opened read-only
and never written. CLIP vectors and events are backfilled on the copy only.

Usage:
    SENTRIX_VECTOR_BACKEND=sqlite PYTHONPATH=. python3 \\
        scripts/benchmarks/person_insight_fixture.py \\
        --source-db sentrix.db --work-db /tmp/work.db --scope-id album3-max \\
        --backfill-clip --build-events
"""

import argparse
import json
import os
import sqlite3
from pathlib import Path


def ensure_sqlite_backend():
    if os.environ.get("SENTRIX_VECTOR_BACKEND", "sqlite").strip().lower() != "sqlite":
        raise RuntimeError("benchmark copy must use SQLite vector backend")


def backup_sqlite(source_path, destination_path):
    source_path = Path(source_path).resolve()
    destination_path = Path(destination_path).resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.unlink(missing_ok=True)
    source = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)
    source.execute("PRAGMA query_only = ON")
    destination = sqlite3.connect(str(destination_path))
    try:
        source.backup(destination)
        result = destination.execute("PRAGMA integrity_check").fetchone()
        if not result or result[0] != "ok":
            raise RuntimeError(f"SQLite backup integrity check failed: {result}")
        destination.commit()
    finally:
        destination.close()
        source.close()
    return str(destination_path)


def _source_stats(path):
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        face_instances = connection.execute("SELECT COUNT(*) FROM face_instances").fetchone()[0]
        active_clusters = connection.execute(
            "SELECT COUNT(*) FROM face_clusters WHERE status = 'active'"
        ).fetchone()[0]
    finally:
        connection.close()
    return {"face_instances": int(face_instances), "active_face_clusters": int(active_clusters)}


def backfill_visual_asset_vectors(store, scope_id):
    """Backfill missing visual/asset CLIP vectors on the working copy only."""
    rows = store.connection.execute(
        """SELECT id, path FROM assets
        WHERE scope_id = ? AND media_type = 'image'
        AND NOT EXISTS (
            SELECT 1 FROM memory_vectors mv
            WHERE mv.scope_id = assets.scope_id AND mv.space = 'visual'
            AND mv.source_type = 'asset' AND mv.source_id = assets.id
        )""",
        (scope_id,),
    ).fetchall()
    from backend.model_clients import ClipAdapter

    adapter = ClipAdapter()
    count = 0
    for row in rows:
        if not row["path"]:
            continue
        vector = adapter.embed_image(row["path"])
        if not vector:
            continue
        store.upsert_vector("visual", "asset", row["id"], vector, "chinese-clip", {})
        count += 1
    return {"visual_asset_vectors": count}


def build_missing_events(store, scope_id):
    """Build events on the working copy; refuse to run when events already exist."""
    existing = store.connection.execute(
        "SELECT COUNT(*) AS count FROM events WHERE scope_id = ?", (scope_id,)
    ).fetchone()["count"]
    if existing:
        raise RuntimeError(
            f"scope {scope_id} already has {existing} events; refusing to build"
        )
    observations = store.connection.execute(
        "SELECT * FROM observations WHERE scope_id = ? ORDER BY captured_at, id",
        (scope_id,),
    ).fetchall()
    for observation in observations:
        store.merge_observation_into_event(dict(observation))
    store.consolidate_events(scope_id)
    total = store.connection.execute(
        "SELECT COUNT(*) FROM events WHERE scope_id = ?", (scope_id,)
    ).fetchone()[0]
    return {"events": int(total)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", required=True)
    parser.add_argument("--work-db", required=True)
    parser.add_argument("--scope-id", required=True)
    parser.add_argument("--backfill-clip", action="store_true")
    parser.add_argument("--build-events", action="store_true")
    args = parser.parse_args(argv)

    ensure_sqlite_backend()
    stats = _source_stats(args.source_db)
    work_path = backup_sqlite(args.source_db, args.work_db)

    from backend.db import MemoryStore

    store = MemoryStore(work_path)
    result = {
        "backup_integrity": "ok",
        "source_writes": 0,
        "face_instances_unchanged": stats["face_instances"],
        "active_face_clusters_unchanged": stats["active_face_clusters"],
    }
    try:
        if args.backfill_clip:
            result.update(backfill_visual_asset_vectors(store, args.scope_id))
        if args.build_events:
            result.update(build_missing_events(store, args.scope_id))
    finally:
        store.close()
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
