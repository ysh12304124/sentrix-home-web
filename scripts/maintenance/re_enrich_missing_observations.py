#!/usr/bin/env python3
"""Fill missing visual fields and refresh their retrieval vectors.

Only empty fields are filled; existing canonical facts are preserved.  Without
--apply this only counts eligible rows and sends no model requests.  It targets
one scope at a time so benchmark runs keep an auditable memory snapshot.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.db import MemoryStore
from backend.model_clients import GammaClient
from backend.pipeline import IngestionPipeline
from backend.semantic_taxonomy import normalize_semantic_analysis


FIELDS = ("caption", "activity", "place", "event_type", "ocr_text",
          "people", "objects", "clothing", "spatial_relations")


def _missing(observation: dict) -> list[str]:
    missing = []
    for key in FIELDS:
        value = observation.get(key)
        if value in (None, "", [], {}):
            missing.append(key)
    detail = observation.get("detail") or {}
    if not any(detail.get(key) for key in ("visible_details", "regions", "text_blocks")):
        missing.append("detail")
    return missing


def _merge(observation: dict, analysis: dict) -> dict:
    analysis = normalize_semantic_analysis(analysis or {})
    patch: dict = {}
    for key in FIELDS:
        if observation.get(key) in (None, "", [], {}) and analysis.get(key) not in (None, "", [], {}):
            patch[key] = analysis[key]
    old_detail = observation.get("detail") or {}
    new_detail = analysis.get("detail") or {}
    if new_detail:
        merged_detail = dict(old_detail)
        for key, value in new_detail.items():
            if key == "schema_version":
                merged_detail[key] = 1
            elif not merged_detail.get(key) and value not in (None, "", [], {}):
                merged_detail[key] = value
        if merged_detail != old_detail:
            patch["detail"] = merged_detail
    return patch


def _blank(observation: dict) -> bool:
    """A photo with no searchable visual or OCR description at all."""
    return (not any(str(observation.get(key) or "").strip()
                    for key in ("caption", "activity", "event_type", "ocr_text"))
            and not observation.get("people") and not observation.get("objects")
            and not any((observation.get("detail") or {}).get(key)
                        for key in ("visible_details", "regions", "text_blocks")))


def _user_media(asset: dict) -> bool:
    """Exclude identity seeds and generated event cards from album repair."""
    name = Path(str(asset.get("file_name") or "")).name
    metadata = asset.get("metadata_json") or {}
    return not (re.match(r"^(?:faceid_|event_event_)", name, re.IGNORECASE)
                or str(metadata.get("source_type") or "").lower() == "identity_seed"
                or str(asset.get("derived_kind") or metadata.get("derived_kind") or "").lower()
                in {"face_crop", "face_id_crop", "face_identity_crop", "face_reference"})


def _embedding_text(observation: dict) -> str:
    objects = observation.get("objects") or []
    object_text = " ".join(
        " ".join(str(item.get(key) or "") for key in ("label", "primary", "details"))
        if isinstance(item, dict) else str(item) for item in objects)
    details = (observation.get("detail") or {}).get("visible_details") or []
    detail_text = " ".join(
        str(item.get("text") or item.get("label") or "")
        if isinstance(item, dict) else str(item) for item in details)
    return " ".join(str(part) for part in (
        observation.get("caption"), observation.get("activity"),
        observation.get("place"), observation.get("ocr_text"),
        object_text, detail_text) if part)


def _write_reenrichment(store, pipeline, observation, asset, patch):
    """Build embeddings first, then atomically update text and its indexes."""
    merged = {**observation, **patch}
    text = _embedding_text(merged)
    if not text.strip():
        return False
    vector, model = pipeline._text_embed(text)
    field_text = pipeline._field_desc_text(merged)
    field_vector, field_model = pipeline._field_desc_embed(field_text)
    event_id = (asset.get("metadata_json") or {}).get("event_id")
    metadata = {"asset_id": asset["id"], "event_id": event_id}
    with store.transaction():
        store.enrich_observation(observation["id"], patch,
                                 source="missing_field_re_enrichment")
        for space in ("episodic", "semantic"):
            store.upsert_vector(space, "observation", observation["id"],
                                vector, model, metadata)
        if field_vector:
            store.upsert_vector("field_desc", "asset", asset["id"],
                                field_vector, field_model, {
                                    "observation_id": observation["id"],
                                    "event_id": event_id, "text": field_text,
                                })
    return True


def run(*, db: str, scope_id: str, apply: bool, limit: int = 0,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str = "qwen3-vl-4b-instruct", workers: int = 4,
        only_blank: bool = False, asset_id: str | None = None) -> dict:
    store = MemoryStore(db)
    rows = store.list_observations(limit or 1_000_000, scope_id=scope_id)
    assets = {row.get("asset_id"): store.get_asset(row.get("asset_id")) or {}
              for row in rows}
    candidates = [row for row in rows if _missing(row)
                  and (not only_blank or _blank(row))
                  and (not asset_id or row.get("asset_id") == asset_id)
                  and _user_media(assets.get(row.get("asset_id")) or {})]
    summary = {
        "scope_id": scope_id, "scanned": len(rows), "candidates": len(candidates),
        "processed": 0, "updated": 0, "failed": 0, "apply": apply,
        "model": model, "base_url": base_url,
        "only_blank": only_blank, "asset_id": asset_id,
    }
    if limit:
        candidates = candidates[:limit]
    if not apply:
        store.close()
        return summary
    timeout = float(os.getenv("SENTRIX_REENRICH_TIMEOUT", "240"))
    assets_by_id = {row.get("asset_id"): assets.get(row.get("asset_id")) or {}
                    for row in candidates}

    def analyze_one(observation):
        asset = assets_by_id.get(observation.get("asset_id")) or {}
        path = asset.get("path") or ""
        if not Path(path).is_file():
            return observation, asset, None, f"missing {path}", 0.0
        started = time.perf_counter()
        gamma = GammaClient(base_url=base_url, model=model, backend="openai", timeout=timeout)
        analysis = gamma.analyze_image(path, {
            "file_name": asset.get("file_name") or "",
            "captured_at": asset.get("captured_at") or "",
            "captured_location": asset.get("captured_location") or "",
            "location_context": (asset.get("metadata_json") or {}).get("reverse_geocode") or {},
            "missing_fields": _missing(observation),
        })
        return observation, asset, _merge(observation, analysis), None, time.perf_counter() - started

    try:
        # The maintenance path needs only embedding methods, not ingestion's
        # face/video adapters.  No asset status or existing facts are changed.
        pipeline = object.__new__(IngestionPipeline)
        with ThreadPoolExecutor(max_workers=max(1, min(int(workers), 8)),
                                thread_name_prefix="sentrix-reenrich") as executor:
            futures = [executor.submit(analyze_one, observation) for observation in candidates]
            for index, future in enumerate(as_completed(futures), start=1):
                try:
                    observation, asset, patch, error, seconds = future.result()
                    if error:
                        summary["failed"] += 1
                        print(f"SKIP {index}/{len(candidates)} {error}", flush=True)
                        continue
                    summary["processed"] += 1
                    if patch and _write_reenrichment(
                            store, pipeline, observation, asset, patch):
                        summary["updated"] += 1
                    print(f"{'OK' if patch else 'NOOP'} {index}/{len(candidates)} "
                          f"{asset.get('file_name') or observation.get('asset_id')} "
                          f"fields={','.join(patch) or '-'} seconds={seconds:.1f}", flush=True)
                except Exception as exc:
                    summary["failed"] += 1
                    print(f"FAILED {index}/{len(candidates)}: {exc}", flush=True)
    finally:
        store.close()
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/sentrix.db")
    parser.add_argument("--scope", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--base-url", default=os.getenv("SENTRIX_VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model", default=os.getenv("SENTRIX_VLLM_MODEL", "qwen3-vl-4b-instruct"))
    parser.add_argument("--workers", type=int, default=int(os.getenv("SENTRIX_REENRICH_WORKERS", "4")))
    parser.add_argument("--only-blank", action="store_true")
    parser.add_argument("--asset-id")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(db=args.db, scope_id=args.scope, apply=args.apply,
                         limit=args.limit, base_url=args.base_url, model=args.model,
                         workers=args.workers, only_blank=args.only_blank,
                         asset_id=args.asset_id),
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
