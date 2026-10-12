#!/usr/bin/env python3
"""Add one embedding model's text vectors without deleting baseline vectors.

The source row universe is copied from an existing model (normally BGE-M3),
so an embedding comparison changes the encoder but not the indexed sources.
Rows are distinguished by ``model_name`` and can coexist in SQLite.  Qdrant
sync is intentionally separate, allowing each experiment to use its own
directory and leaving the baseline directory untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _flatten(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_flatten(item))
        return out
    if isinstance(value, dict):
        out = []
        for key in ("label", "primary", "name", "details", "text", "description"):
            out.extend(_flatten(value.get(key)))
        return out
    return []


def _observation_text(observation: dict) -> str:
    detail = observation.get("detail") or observation.get("detail_json") or {}
    pieces = [
        observation.get("caption"), observation.get("activity"), observation.get("place"),
        observation.get("ocr_text"), observation.get("transcript"),
        *_flatten(observation.get("clothing") or observation.get("clothing_json")),
        *_flatten((detail or {}).get("visible_details") if isinstance(detail, dict) else None),
    ]
    return " ".join(str(value).strip() for value in pieces if str(value or "").strip())


def _field_desc_text(observation: dict) -> str:
    objects = observation.get("objects") or observation.get("objects_json") or []
    return " ".join(filter(None, [
        str(observation.get("caption") or "").strip(),
        str(observation.get("place") or "").strip(),
        " ".join(_flatten(objects)),
        str(observation.get("event_type") or "").strip(),
        str(observation.get("ocr_text") or "").strip(),
    ]))


def _source_text(store, row: dict) -> str:
    if row["source_type"] == "observation":
        return _observation_text(store.get_observation(row["source_id"]) or {})
    if row["source_type"] == "event":
        event = store.get_event(row["source_id"]) or {}
        return " ".join(filter(None, [
            str(event.get("title") or "").strip(),
            str(event.get("summary") or "").strip(),
            str(event.get("activity") or "").strip(),
            str(event.get("place") or "").strip(),
        ]))
    if row["space"] == "field_desc" and row["source_type"] == "asset":
        observations = store.list_observations(asset_id=row["source_id"], limit=1) or []
        return _field_desc_text(observations[0] if observations else {})
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/sentrix.db")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--source-model-contains", default="bge-m3")
    parser.add_argument("--scope", default="", help="optional single scope id")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    os.environ["SENTRIX_VECTOR_BACKEND"] = "sqlite"
    from backend.db import MemoryStore

    store = MemoryStore(args.db)
    where = "space != 'visual' AND model_name LIKE ?"
    values: list[object] = [f"%{args.source_model_contains}%"]
    if args.scope:
        where += " AND scope_id = ?"
        values.append(args.scope)
    source_rows = [dict(row) for row in store.connection.execute(
        f"SELECT * FROM memory_vectors WHERE {where} ORDER BY created_at", values,
    ).fetchall()]
    prepared = []
    for row in source_rows:
        text = _source_text(store, row)
        if text:
            prepared.append((row, text))
    summary = {
        "source_rows": len(source_rows), "encodable_rows": len(prepared),
        "source_model_contains": args.source_model_contains,
        "target_model": args.model_id, "scope": args.scope or None,
    }
    if not args.apply:
        print(json.dumps({**summary, "dry_run": True}, ensure_ascii=False, indent=2))
        store.close()
        return 0

    from sentence_transformers import SentenceTransformer

    started = time.perf_counter()
    model = SentenceTransformer(args.model_path, device=args.device, local_files_only=True)
    model.max_seq_length = 512
    prompts = getattr(model, "prompts", {}) or {}
    kwargs = {
        "batch_size": max(1, args.batch_size), "normalize_embeddings": True,
        "show_progress_bar": True,
    }
    if "document" in prompts:
        kwargs["prompt_name"] = "document"
    vectors = model.encode([text for _, text in prepared], **kwargs)
    written = 0
    for (row, text), vector in zip(prepared, vectors):
        metadata = json.loads(row.get("metadata_json") or "{}")
        metadata.update({
            "reembedded": True, "reembedded_from_model": row["model_name"],
            "embedding_text_sha1": __import__("hashlib").sha1(text.encode("utf-8")).hexdigest(),
        })
        store.upsert_vector(
            row["space"], row["source_type"], row["source_id"],
            [float(value) for value in vector], args.model_id, metadata,
        )
        written += 1
    store.close()
    print(json.dumps({
        **summary, "written": written,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
