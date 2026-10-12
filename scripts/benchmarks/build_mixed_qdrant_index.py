#!/usr/bin/env python3
"""Create a non-destructive Qwen-VL-image + BGE-text Qdrant index.

The three existing experiment indexes use identical point IDs.  This utility
copies the 2048-d image vector from the Qwen3-VL index and the four 1024-d text
route vectors from the baseline BGE index.  Source indexes are opened read-only
in practice and are never recreated or modified.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from qdrant_client import QdrantClient, models


COLLECTION = "sentrix_memory_units_v1"
TEXT_ROUTES = ("visual", "object", "relation", "speech")


def _all_points(client: QdrantClient) -> list:
    rows = []
    cursor = None
    while True:
        page, cursor = client.scroll(
            COLLECTION,
            limit=128,
            offset=cursor,
            with_payload=True,
            with_vectors=True,
        )
        rows.extend(page)
        if cursor is None:
            return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bge-index", required=True)
    parser.add_argument("--qwen-vl-index", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    bge_path = Path(args.bge_index).resolve()
    qwen_path = Path(args.qwen_vl_index).resolve()
    output_path = Path(args.out).resolve()
    if output_path in {bge_path, qwen_path}:
        raise ValueError("output index must not overwrite either source index")

    if output_path.exists() and any(output_path.iterdir()):
        existing = QdrantClient(path=str(output_path))
        try:
            if not existing.collection_exists(COLLECTION):
                raise RuntimeError(f"non-empty output has no {COLLECTION} collection")
            count = int(existing.count(COLLECTION, exact=True).count)
            if count != 374:
                raise RuntimeError(f"existing output has {count} points, expected 374")
            print(f"reuse validated mixed index: {output_path} ({count} points)")
            return 0
        finally:
            existing.close()

    bge = QdrantClient(path=str(bge_path))
    qwen = QdrantClient(path=str(qwen_path))
    try:
        bge_rows = {str(row.id): row for row in _all_points(bge)}
        qwen_rows = _all_points(qwen)
    finally:
        bge.close()
        qwen.close()

    if len(bge_rows) != len(qwen_rows):
        raise RuntimeError(
            f"source point counts differ: BGE={len(bge_rows)}, Qwen-VL={len(qwen_rows)}"
        )
    qwen_ids = {str(row.id) for row in qwen_rows}
    if set(bge_rows) != qwen_ids:
        raise RuntimeError("source point IDs differ; refusing to create a mixed index")

    output_path.mkdir(parents=True, exist_ok=True)
    out = QdrantClient(path=str(output_path))
    try:
        out.create_collection(
            COLLECTION,
            vectors_config={
                "image": models.VectorParams(size=2048, distance=models.Distance.COSINE),
                **{
                    route: models.VectorParams(size=1024, distance=models.Distance.COSINE)
                    for route in TEXT_ROUTES
                },
            },
        )
        points = []
        for row in qwen_rows:
            bge_row = bge_rows[str(row.id)]
            vectors = {"image": row.vector["image"]}
            vectors.update({route: bge_row.vector[route] for route in TEXT_ROUTES})
            points.append(models.PointStruct(
                id=row.id,
                vector=vectors,
                payload=dict(row.payload or {}),
            ))
        for offset in range(0, len(points), 64):
            out.upsert(COLLECTION, points=points[offset:offset + 64], wait=True)
        for field in (
            "scope_id", "asset_id", "media_type", "unit_type", "capture_date", "people",
        ):
            out.create_payload_index(
                COLLECTION, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD,
            )
        out.create_payload_index(
            COLLECTION,
            field_name="capture_year",
            field_schema=models.PayloadSchemaType.INTEGER,
        )
        count = int(out.count(COLLECTION, exact=True).count)
    finally:
        out.close()

    if count != len(qwen_rows):
        raise RuntimeError(f"mixed index validation failed: {count}/{len(qwen_rows)} points")
    print(f"created mixed index: {output_path} ({count} points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
