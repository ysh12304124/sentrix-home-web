#!/usr/bin/env python3
"""Backfill and verify the rebuildable Qdrant memory-vector mirror."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path


def _rows(store, scope_id=None, model_names=None):
    query = (
        "SELECT id,scope_id,space,source_type,source_id,vector_json,model_name,"
        "metadata_json,created_at,updated_at FROM memory_vectors"
    )
    conditions = []
    parameters = []
    if scope_id:
        conditions.append("scope_id = ?")
        parameters.append(scope_id)
    model_names = [str(name) for name in (model_names or []) if str(name).strip()]
    if model_names:
        conditions.append(f"model_name IN ({','.join('?' for _ in model_names)})")
        parameters.extend(model_names)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY updated_at"
    return store.connection.execute(query, parameters).fetchall()


def _decode(value, fallback):
    try:
        return json.loads(value or "")
    except (TypeError, ValueError):
        return fallback


def sync(store, index, scope_id=None, model_names=None):
    payloads = []
    skipped = 0
    started = time.perf_counter()
    cleared_collections = (index.drop_scope(scope_id) if scope_id else index.clear())
    for raw in _rows(store, scope_id, model_names):
        row = dict(raw)
        vector = _decode(row["vector_json"], [])
        if not vector:
            skipped += 1
            continue
        payloads.append({
            "row_id": row["id"], "scope_id": row["scope_id"], "space": row["space"],
            "source_type": row["source_type"], "source_id": row["source_id"],
            "vector": vector, "model_name": row["model_name"],
            "metadata": _decode(row["metadata_json"], {}),
            "created_at": row["created_at"], "updated_at": row["updated_at"],
        })
    # This is an offline rebuild and all points are already materialized in
    # memory.  Larger batches dramatically reduce embedded-Qdrant fsync waits
    # on Windows without changing online single-row ingestion semantics.
    written = index.upsert_many(payloads, batch_size=2048) if hasattr(index, "upsert_many") else 0
    return {"written": written, "skipped": skipped,
            "cleared_collections": cleared_collections,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)}


def reembed_visual_assets(store, embedder, *, scope_id=None):
    started = time.perf_counter()
    written = 0
    skipped = 0
    assets = store.list_assets(media_type="image", limit=100_000, scope_id=scope_id)
    valid_assets = []
    for asset in assets:
        path = asset.get("path") or ""
        if not Path(path).is_file():
            skipped += 1
            continue
        valid_assets.append(asset)
    # Eight images fit comfortably on a 24 GiB card at the configured pixel
    # cap and avoid paying HTTP/model scheduling overhead for every asset.
    batch_size = 8
    for offset in range(0, len(valid_assets), batch_size):
        batch = valid_assets[offset:offset + batch_size]
        if hasattr(embedder, "embed_images"):
            vectors = embedder.embed_images([asset["path"] for asset in batch], batch_size=batch_size)
        else:
            vectors = [embedder.embed_image(batch[0]["path"])]
        if len(vectors) != len(batch):
            skipped += len(batch)
            continue
        for asset, vector in zip(batch, vectors):
            path = asset.get("path") or ""
            if not vector:
                skipped += 1
                continue
            observation_row = store._row(
                "SELECT id FROM observations WHERE asset_id = ? ORDER BY created_at LIMIT 1",
                (asset["id"],),
            )
            observation = store.get_observation(observation_row["id"]) if observation_row else {}
            store.upsert_vector(
                "visual", "asset", asset["id"], vector, embedder.model_id,
                {"scope_id": asset.get("scope_id") or "home-default",
                 "observation_id": observation.get("id"),
                 "event_id": observation.get("event_id"),
                 "reembedded": True},
            )
            written += 1
            if written % 20 == 0 or written == len(valid_assets):
                print(json.dumps({
                    "stage": "visual_reembed", "written": written,
                    "total": len(valid_assets), "skipped": skipped,
                    "elapsed_seconds": round(time.perf_counter() - started, 1),
                }, ensure_ascii=False), flush=True)
    return {"written": written, "skipped": skipped,
            "model": embedder.model_id,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)}


def benchmark(store, index, sample_count=20, limit=10, scope_id=None, model_names=None):
    rows = [dict(row) for row in _rows(store, scope_id, model_names)]
    if not rows:
        return {"queries": 0}
    step = max(1, len(rows) // max(1, sample_count))
    samples = rows[::step][:sample_count]
    sqlite_ms = []
    qdrant_ms = []
    overlaps = []
    for row in samples:
        vector = _decode(row["vector_json"], [])
        if not vector:
            continue
        started = time.perf_counter()
        sqlite_hits = store.search_vectors_sqlite(
            row["space"], vector, limit=limit, scope_id=row["scope_id"],
            model_name=row["model_name"],
        )
        sqlite_ms.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        qdrant_hits = index.search(
            space=row["space"], vector=vector, limit=limit,
            scope_id=row["scope_id"], model_name=row["model_name"],
        )
        qdrant_ms.append((time.perf_counter() - started) * 1000)
        left = {item["source_id"] for item in sqlite_hits}
        right = {item["source_id"] for item in qdrant_hits}
        overlaps.append(len(left & right) / max(1, len(left)))
    return {
        "queries": len(sqlite_ms),
        "limit": limit,
        "sqlite_avg_ms": round(statistics.mean(sqlite_ms), 3) if sqlite_ms else None,
        "qdrant_avg_ms": round(statistics.mean(qdrant_ms), 3) if qdrant_ms else None,
        "sqlite_p95_ms": round(sorted(sqlite_ms)[max(0, int(len(sqlite_ms) * .95) - 1)], 3) if sqlite_ms else None,
        "qdrant_p95_ms": round(sorted(qdrant_ms)[max(0, int(len(qdrant_ms) * .95) - 1)], 3) if qdrant_ms else None,
        "top_k_overlap": round(statistics.mean(overlaps), 4) if overlaps else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=os.getenv("SENTRIX_DB_PATH", "data/sentrix.db"))
    parser.add_argument("--qdrant-path", default=os.getenv("SENTRIX_QDRANT_PATH", "data/qdrant"))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--reembed-visual", choices=["none", "clip", "chinese_clip", "qwen3_vl"], default="none",
                        help="Rebuild image vectors in the deployed query model before Qdrant sync.")
    parser.add_argument("--scope", default="",
                        help="Only rebuild one scope; preserves all other derived collections.")
    parser.add_argument(
        "--model", action="append", default=[],
        help="Only mirror this exact model_name; repeat for multiple active models.",
    )
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    # Encode into SQLite first.  The final bulk sync mirrors every vector to
    # the new Qdrant directory in one pass; dual-writing each image here adds
    # hundreds of unnecessary fsyncs and makes an interrupted rebuild harder
    # to resume.
    os.environ["SENTRIX_VECTOR_BACKEND"] = "sqlite"
    os.environ["SENTRIX_QDRANT_PATH"] = str(Path(args.qdrant_path).resolve())

    from backend.db import MemoryStore

    store = MemoryStore(args.db)
    os.environ["SENTRIX_VECTOR_BACKEND"] = "qdrant"
    from backend.qdrant_memory import get_qdrant_index
    index = get_qdrant_index(args.db)
    try:
        rows = _rows(store, args.scope or None, args.model)
        result = {"sqlite_vectors": len(rows), "scope": args.scope or None,
                  "qdrant_path": str(Path(args.qdrant_path).resolve())}
        if index is None:
            result["qdrant"] = {
                "available": False,
                "reason": "qdrant_unavailable_or_locked",
            }
            if args.apply:
                raise RuntimeError(
                    "Qdrant unavailable or locked; stop the owning service before --apply"
                )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        if args.apply:
            if args.reembed_visual != "none":
                if args.reembed_visual == "qwen3_vl":
                    from backend.embeddings.qwen3_vl_remote import Qwen3VLVisualAdapter
                    embedder = Qwen3VLVisualAdapter()
                    # The generic rebuild helper records ``model_id`` while
                    # the runtime drop-in adapter exposes ``model_name``.
                    embedder.model_id = embedder.model_name
                elif args.reembed_visual == "chinese_clip":
                    from backend.embeddings.chinese_clip_visual import ChineseClipVisualEmbedder
                    embedder = ChineseClipVisualEmbedder()
                else:
                    from backend.model_clients import ClipAdapter
                    clip = ClipAdapter()

                    class ClipImageEmbedder:
                        model_id = clip.model_name

                        @staticmethod
                        def embed_image(path):
                            return clip.embed_image(path)

                    embedder = ClipImageEmbedder()
                result["visual_reembed"] = reembed_visual_assets(
                    store, embedder, scope_id=args.scope or None
                )
            result["sync"] = sync(store, index, args.scope or None, args.model)
        else:
            result["dry_run"] = True
        result["qdrant"] = index.collection_stats()
        if args.benchmark:
            result["benchmark"] = benchmark(
                store, index, sample_count=args.samples,
                scope_id=args.scope or None, model_names=args.model,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        store.close()


if __name__ == "__main__":
    main()
