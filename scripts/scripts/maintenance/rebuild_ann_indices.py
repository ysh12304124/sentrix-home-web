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


def build(store, ann_dir, backend="hnswlib", visual_embedder=None):
    """Rebuild semantic/episodic from stored vectors; optionally re-embed the
    visual space from image files when ``visual_embedder`` is provided."""
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
        if space == "visual" and visual_embedder is not None:
            payload = _visual_payload_from_images(store, visual_embedder)
            dim = len(payload[0][1]) if payload else visual_embedder.dimension
            model = visual_embedder.model_id
        else:
            # A space can contain vectors from several historical embedders
            # (and visual also contains face-instance rows).  An ANN index is
            # single-model/single-dimension; mixing rows silently drops the
            # current album or makes the query channel incompatible.  For
            # visual, prefer asset rows; for every space, keep the dominant
            # model among the eligible rows.
            usable = rows
            if space == "visual":
                asset_rows = [row for row in rows if row["source_type"] == "asset"]
                usable = asset_rows or rows
            by_model = {}
            for row in usable:
                by_model.setdefault(row["model_name"], []).append(row)
            usable = max(by_model.values(), key=len)
            dim = len(json.loads(usable[0]["vector_json"]))
            model = usable[0]["model_name"]
            payload = []
            for row in usable:
                try:
                    vector = json.loads(row["vector_json"])
                except (TypeError, ValueError):
                    continue
                if len(vector) != dim:
                    continue
                metadata = {}
                try:
                    metadata = json.loads(row["metadata_json"] or "{}")
                except (TypeError, ValueError):
                    metadata = {}
                metadata["scope_id"] = row["scope_id"]
                metadata["revision"] = 1
                payload.append((row["source_id"], vector, metadata))
        index = create_index(backend, dim=dim, M=16, ef_construction=200, ef_search=50)
        index.set_manifest_extra(
            model_id=model, checkpoint_hash="", source_type="asset" if space == "visual" else "observation",
            normalized=True, source_revision=1,
        )
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


def _visual_payload_from_images(store, embedder):
    payload = []
    for asset in store.list_assets(media_type="image", limit=100_000):
        path = asset.get("path") or ""
        if not Path(path).is_file():
            continue
        vector = embedder.embed_image(path)
        if not vector:
            continue
        payload.append((asset["id"], vector, {"scope_id": asset.get("scope_id") or "home-default", "revision": 1}))
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=os.getenv("SENTRIX_DB_PATH", "data/sentrix.db"))
    parser.add_argument("--ann-dir", default=os.getenv("SENTRIX_ANN_DIR", "data/ann"))
    parser.add_argument("--visual-embedder", choices=["none", "chinese_clip"], default="none",
                        help="re-embed the visual space from image files with this embedder")
    parser.add_argument("--apply", action="store_true", help="required: write indices to disk")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from backend.db import MemoryStore

    if not args.apply:
        print("dry-run: pass --apply to write indices")
        return
    visual_embedder = None
    if args.visual_embedder == "chinese_clip":
        from backend.embeddings.chinese_clip_visual import ChineseClipVisualEmbedder
        visual_embedder = ChineseClipVisualEmbedder()
        if not visual_embedder.available:
            print("chinese_clip embedder unavailable; aborting visual re-embed")
            return
    store = MemoryStore(args.db)
    summary = build(store, args.ann_dir, visual_embedder=visual_embedder)
    store.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
