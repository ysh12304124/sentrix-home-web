#!/usr/bin/env python3
"""Build four auditable encoder A/B indexes without changing source data.

Every output contains the same point IDs and payloads.  Only the image and text
vector families are selected from the verified source indexes:

* E0: CLIP image + BGE-M3 text
* E1: CLIP image + Qwen3-Embedding-0.6B text
* E2: Qwen3-VL-Embedding-2B image + BGE-M3 text
* E3: Qwen3-VL-Embedding-2B image + Qwen3-Embedding-0.6B text

An ``index_manifest.json`` records model IDs, source paths, dimensions, point
count and deterministic vector fingerprints.  Existing non-empty outputs are
validated and reused; source indexes are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

from qdrant_client import QdrantClient, models


COLLECTION = "sentrix_memory_units_v1"
TEXT_ROUTES = ("visual", "object", "relation", "speech")
VARIANTS = {
    "E0-clip-bge": {
        "image_source": "clip_bge",
        "text_source": "clip_bge",
        "image_model": "openai/clip-vit-base-patch32",
        "text_model": "BAAI/bge-m3",
    },
    "E1-clip-qwen06": {
        "image_source": "clip_bge",
        "text_source": "clip_qwen",
        "image_model": "openai/clip-vit-base-patch32",
        "text_model": "Qwen/Qwen3-Embedding-0.6B",
    },
    "E2-qwenvl2b-bge": {
        "image_source": "qwenvl_qwen",
        "text_source": "clip_bge",
        "image_model": "Qwen/Qwen3-VL-Embedding-2B",
        "text_model": "BAAI/bge-m3",
    },
    "E3-qwenvl2b-qwen06": {
        "image_source": "qwenvl_qwen",
        "text_source": "clip_qwen",
        "image_model": "Qwen/Qwen3-VL-Embedding-2B",
        "text_model": "Qwen/Qwen3-Embedding-0.6B",
    },
}


def _all_points(client: QdrantClient) -> list:
    rows = []
    cursor = None
    while True:
        page, cursor = client.scroll(
            COLLECTION, limit=128, offset=cursor,
            with_payload=True, with_vectors=True,
        )
        rows.extend(page)
        if cursor is None:
            return rows


def _load(path: Path) -> dict[str, object]:
    client = QdrantClient(path=str(path))
    try:
        if not client.collection_exists(COLLECTION):
            raise RuntimeError(f"missing {COLLECTION}: {path}")
        return {str(row.id): row for row in _all_points(client)}
    finally:
        client.close()


def _fingerprints(rows: list) -> dict[str, str]:
    digests = {name: hashlib.sha256() for name in ("image", *TEXT_ROUTES)}
    for row in sorted(rows, key=lambda item: str(item.id)):
        point_id = str(row.id).encode("utf-8")
        for name, digest in digests.items():
            digest.update(point_id)
            for value in row.vector[name]:
                digest.update(struct.pack("<f", float(value)))
    return {name: digest.hexdigest() for name, digest in digests.items()}


def _validate_sources(sources: dict[str, dict[str, object]]) -> list[str]:
    id_sets = {name: set(rows) for name, rows in sources.items()}
    expected = next(iter(id_sets.values()))
    for name, ids in id_sets.items():
        if ids != expected:
            raise RuntimeError(
                f"point ID mismatch for {name}: {len(ids)} vs expected {len(expected)}"
            )
    for point_id in expected:
        reference = sources["clip_bge"][point_id].payload or {}
        for name, rows in sources.items():
            payload = rows[point_id].payload or {}
            for field in ("scope_id", "asset_id", "media_type", "unit_type", "segment_index"):
                if payload.get(field) != reference.get(field):
                    raise RuntimeError(f"payload mismatch {point_id} {name} field={field}")
    return sorted(expected)


def _build_one(
    output: Path,
    spec: dict,
    sources: dict[str, dict[str, object]],
    source_paths: dict[str, Path],
    point_ids: list[str],
) -> dict:
    manifest_path = output / "index_manifest.json"
    if output.exists() and any(output.iterdir()):
        if not manifest_path.is_file():
            raise RuntimeError(f"refusing unmanifested non-empty output: {output}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("point_count") != len(point_ids):
            raise RuntimeError(f"existing manifest count mismatch: {output}")
        return manifest

    image_rows = sources[spec["image_source"]]
    text_rows = sources[spec["text_source"]]
    first = image_rows[point_ids[0]]
    image_dimension = len(first.vector["image"])
    text_dimension = len(text_rows[point_ids[0]].vector[TEXT_ROUTES[0]])
    output.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(output))
    try:
        client.create_collection(
            COLLECTION,
            vectors_config={
                "image": models.VectorParams(
                    size=image_dimension, distance=models.Distance.COSINE,
                ),
                **{
                    route: models.VectorParams(
                        size=text_dimension, distance=models.Distance.COSINE,
                    )
                    for route in TEXT_ROUTES
                },
            },
        )
        batch = []
        written_rows = []
        for point_id in point_ids:
            image_row = image_rows[point_id]
            text_row = text_rows[point_id]
            vectors = {"image": image_row.vector["image"]}
            vectors.update({route: text_row.vector[route] for route in TEXT_ROUTES})
            point = models.PointStruct(
                id=image_row.id, vector=vectors, payload=dict(image_row.payload or {}),
            )
            batch.append(point)
            written_rows.append(point)
            if len(batch) == 64:
                client.upsert(COLLECTION, points=batch, wait=True)
                batch = []
        if batch:
            client.upsert(COLLECTION, points=batch, wait=True)
        count = int(client.count(COLLECTION, exact=True).count)
    finally:
        client.close()
    if count != len(point_ids):
        raise RuntimeError(f"output count mismatch {output}: {count}/{len(point_ids)}")

    manifest = {
        "schema_version": "sentrix_multiroute_ab_index_v1",
        "collection": COLLECTION,
        "point_count": count,
        "image": {
            "model": spec["image_model"],
            "dimension": image_dimension,
            "source_index": str(source_paths[spec["image_source"]]),
        },
        "text": {
            "model": spec["text_model"],
            "dimension": text_dimension,
            "routes": list(TEXT_ROUTES),
            "source_index": str(source_paths[spec["text_source"]]),
        },
        "vector_sha256": _fingerprints(written_rows),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-bge-index", required=True)
    parser.add_argument("--clip-qwen-index", required=True)
    parser.add_argument("--qwenvl-qwen-index", required=True)
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()

    source_paths = {
        "clip_bge": Path(args.clip_bge_index).resolve(),
        "clip_qwen": Path(args.clip_qwen_index).resolve(),
        "qwenvl_qwen": Path(args.qwenvl_qwen_index).resolve(),
    }
    output_root = Path(args.out_root).resolve()
    if output_root in set(source_paths.values()):
        raise ValueError("output root must not equal a source index")
    sources = {name: _load(path) for name, path in source_paths.items()}
    point_ids = _validate_sources(sources)

    results = {}
    for variant, spec in VARIANTS.items():
        output = output_root / variant
        results[variant] = {
            "path": str(output),
            **_build_one(output, spec, sources, source_paths, point_ids),
        }
        print(f"ready: {variant} ({results[variant]['point_count']} points)", flush=True)
    (output_root / "ab_manifest.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"saved: {output_root / 'ab_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
