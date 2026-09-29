#!/usr/bin/env python3
"""Fill missing visual observation fields with the active 12B VLM.

Only empty fields are filled; existing canonical facts are preserved.  The
script is checkpointed by the observation's revision and can be rerun safely.
It intentionally targets one scope at a time so benchmark A/B runs can use a
controlled, auditable memory snapshot.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.db import MemoryStore
from backend.model_clients import GammaClient
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


def run(*, db: str, scope_id: str, apply: bool, limit: int = 0,
        base_url: str = "http://127.0.0.1:8100/v1",
        model: str = "gemma4-12b-it", workers: int = 4) -> dict:
    store = MemoryStore(db)
    rows = store.list_observations(limit or 1_000_000, scope_id=scope_id)
    candidates = [row for row in rows if _missing(row)]
    summary = {
        "scope_id": scope_id, "scanned": len(rows), "candidates": len(candidates),
        "processed": 0, "updated": 0, "failed": 0, "apply": apply,
        "model": model, "base_url": base_url,
    }
    if limit:
        candidates = candidates[:limit]
    timeout = float(os.getenv("SENTRIX_REENRICH_TIMEOUT", "240"))
    assets_by_id = {
        observation.get("asset_id"): store.get_asset(observation.get("asset_id")) or {}
        for observation in candidates
    }

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
                    if apply and patch:
                        # Only this main thread mutates SQLite; model requests run in parallel.
                        store.enrich_observation(observation["id"], patch,
                                                 source="12b_missing_field_re_enrichment")
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
    parser.add_argument("--base-url", default=os.getenv("SENTRIX_VLLM_BASE_URL", "http://127.0.0.1:8100/v1"))
    parser.add_argument("--model", default=os.getenv("SENTRIX_VLLM_MODEL", "gemma4-12b-it"))
    parser.add_argument("--workers", type=int, default=int(os.getenv("SENTRIX_REENRICH_WORKERS", "4")))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(db=args.db, scope_id=args.scope, apply=args.apply,
                         limit=args.limit, base_url=args.base_url, model=args.model,
                         workers=args.workers),
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
