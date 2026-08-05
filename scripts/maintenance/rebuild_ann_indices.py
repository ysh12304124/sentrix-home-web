#!/usr/bin/env python3
"""Rebuild hnswlib ANN indices from the ``memory_vectors`` table (Phase R).

Reads every vector row grouped by space (visual / semantic / episodic), builds
a fresh hnswlib index with a P0-4 manifest, and atomically swaps it into
``data/ann/{space}``.  The index is a derived artifact — this script is the
recovery path after a data-directory loss and the routine bulk rebuild.

Manifest ``model_id`` comes from the vector rows' ``model_name`` so the query
embedder compatibility check stays honest.

Usage:
  python scripts/maintenance/rebuild_ann_indices.py --db data/sentrix.db --ann-dir data/ann --apply
"""

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def _rows(store):
    return store.connection.execute(
        "SELECT id, scope_id, space, source_type, source_id, vector_json, model_name, "
        "metadata_json, created_at, updated_at FROM memory_vectors"
    ).fetchall()


def build(store, ann_dir, backend="hnswlib"):
    from backend.retrieval_ann import create_index
    vectors = _rows(store)
    by_space = {}
    for row in vectors:
        row = dict(row)
        by_space.setdefault(row["space"], []).append(row)
    summary = {}
    ann_dir = Path(ann_dir)
    ann_dir.mkdir(parents=True, exist_ok=True)
    for space, rows in sorted(by_space.items()):
        if space not in {"visual", "semantic", "episodic"}:
            continue
        dim = len(json.loads(rows[0]["vector_json"]))
        model = rows[0]["model_name"]
        index = create_index(backend, dim=dim, M=16, ef_construction=200, ef_search=50)
        index.set_manifest_extra(
            model_id=model, checkpoint_hash="", source_type="asset" if space == "visual" else "observation",
            normalized=True, source_revision=1,
        )
        payload = []
        for row in rows:
            try:
                vector = json.loads(row["vector_json"])
            except (TypeError, ValueError):
                continue
            metadata = {}
            try:
                metadata = json.loads(row["metadata_json"] or "{}")
            except (TypeError, ValueError):
                metadata = {}
            metadata["scope_id"] = row["scope_id"]
            metadata["revision"] = 1
            payload.append((row["source_id"], vector, metadata))
        index.build(payload)
        # Atomic swap: build into a temp path then rename.
        tmp = str(ann_dir / f"{space}.tmp")
        index.save(tmp)
        for suffix in (".hnsw", ".meta.json", ".manifest.json"):
            source = f"{tmp}{suffix}"
            if Path(source).is_file():
                os.replace(source, str(ann_dir / f"{space}{suffix}"))
        summary[space] = {"count": len(payload), "dim": dim, "model": model}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=os.getenv("SENTRIX_DB_PATH", "data/sentrix.db"))
    parser.add_argument("--ann-dir", default=os.getenv("SENTRIX_ANN_DIR", "data/ann"))
    parser.add_argument("--apply", action="store_true", help="required: write indices to disk")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from backend.db import MemoryStore

    if not args.apply:
        print("dry-run: pass --apply to write indices")
        return
    store = MemoryStore(args.db)
    summary = build(store, args.ann_dir)
    store.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
