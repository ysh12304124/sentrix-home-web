#!/usr/bin/env python3
"""Benchmark case answerability + GT consistency audit (Phase R R0).

Reads the three-album ground-truth sets and, against a real MemoryStore,
labels every case with:

  answerability:
    metadata               GT assets exist with captured_at / media_type
    confirmed_entity       query mentions a person resolvable to a confirmed entity
    lexical_observation    at least one Observation on the GT assets has a
                           caption / activity / place / object / clothing / ocr
    visual_semantic        a visual (asset) vector exists for the GT assets
    text_semantic          a semantic/episodic (observation) vector exists
    external_geo_required  the query or GT depends on geo/POI labels
    formation_missing      GT assets have no queryable Observation at all
    ambiguous_gt           GT count mismatches the listed file count

Also freezes the interpretation of GT inconsistencies so the retrieval
benchmark is not judged against ambiguous labels.

Run on 153 against the real DB (the only authoritative copy of the data).
Local runs against a fixture are for tooling smoke tests only.

This is a benchmark tool: it legitimately reads benchmark data, which is
forbidden in runtime code (backend/*.py, configs/) but allowed here.
"""

import argparse
import json
import os
import sys
from pathlib import Path


ALBUMS = ("album1", "album2", "album3")
DEFAULT_SAMPLES = os.getenv("SENTRIX_BENCHMARK_SAMPLES", str(Path.home() / "Downloads" / "samples"))


def _load_queries(samples_root):
    cases = []
    for album in ALBUMS:
        path = Path(samples_root) / album / "query.json"
        if not path.is_file():
            print(f"[audit] missing {path}", file=sys.stderr)
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for index, case in enumerate(data, 1):
            case["album"] = album
            case["key"] = f"{album}-{index:02d}"
            cases.append(case)
    return cases


def _asset_ids_by_filename(store, scope_id):
    """Map file_name -> asset_id for the given scope (album)."""
    mapping = {}
    for asset in store.list_assets(scope_id=scope_id, limit=10_000):
        mapping[asset.get("file_name")] = asset.get("id")
    return mapping


def _vector_exists(store, space, source_type, source_id):
    row = store.connection.execute(
        "SELECT 1 FROM memory_vectors WHERE space = ? AND source_type = ? AND source_id = ? LIMIT 1",
        (space, source_type, source_id),
    ).fetchone()
    return bool(row)


def _has_text_vector(store, asset_id, observation_ids):
    for source_id in [asset_id] + observation_ids:
        row = store.connection.execute(
            "SELECT 1 FROM memory_vectors WHERE space IN ('semantic','episodic') AND source_type = 'observation' AND source_id = ? LIMIT 1",
            (source_id,),
        ).fetchone()
        if row:
            return True
    return False


def _person_named(case):
    for token in ("明哥", "王明", "八戒", "小黑"):
        if token in case.get("query_cn", ""):
            return token
    return None


def _resolve_observation_ids(store, asset_id):
    return [obs.get("id") for obs in store.list_observations(limit=10_000) if obs.get("asset_id") == asset_id]


def _audit_case(store, case, filename_to_id):
    key = case["key"]
    files = list(case.get("ground_truth") or [])
    listed_count = len(files)
    declared_count = case.get("ground_truth_count") or listed_count
    resolved = [(file_name, filename_to_id.get(file_name)) for file_name in files]
    missing_in_db = [file_name for file_name, asset_id in resolved if asset_id is None]
    asset_ids = [asset_id for _, asset_id in resolved if asset_id]

    observations = []
    vectors_visual = []
    vectors_text = []
    for asset_id in asset_ids:
        obs_ids = _resolve_observation_ids(store, asset_id)
        observations.extend(obs_ids)
        vectors_visual.append(asset_id if _vector_exists(store, "visual", "asset", asset_id) else None)
        if _has_text_vector(store, asset_id, obs_ids):
            vectors_text.append(asset_id)

    has_any_observation_text = False
    for asset_id in asset_ids:
        for obs in store.list_observations(limit=10_000):
            if obs.get("asset_id") != asset_id:
                continue
            if obs.get("caption") or obs.get("activity") or obs.get("place") or obs.get("ocr_text") or obs.get("objects_json") != "[]" or obs.get("clothing_json") != "[]":
                has_any_observation_text = True
                break
        if has_any_observation_text:
            break

    person = _person_named(case)
    confirmed_entity = False
    if person:
        try:
            confirmed_entity = any(
                entity.get("canonical_name") == person
                for entity in store.list_entities(status="confirmed", scope_id=case["album"])
            )
        except Exception:
            confirmed_entity = False

    external_geo = (case.get("Location") == "cognitive") or any(
        token in case.get("query_cn", "") for token in ("市", "区", "省", "镇", "湾", "湖", "城")
    )

    answerability = {
        "metadata": bool(asset_ids),
        "confirmed_entity": confirmed_entity,
        "lexical_observation": has_any_observation_text,
        "visual_semantic": any(vectors_visual),
        "text_semantic": bool(vectors_text),
        "external_geo_required": external_geo,
        "formation_missing": bool(asset_ids) and not has_any_observation_text and not vectors_visual and not vectors_text,
        "ambiguous_gt": declared_count != listed_count,
    }
    return {
        "key": key, "query_cn": case.get("query_cn"), "album": case["album"],
        "declared_count": declared_count, "listed_count": listed_count,
        "gt_files": files, "gt_resolved_asset_ids": asset_ids,
        "missing_in_db": missing_in_db,
        "person_token": person,
        "source_labels": {"Location": case.get("Location"), "Person": case.get("Person"),
                          "Object": case.get("Object"), "Time": case.get("Time"), "Source": case.get("Source")},
        "answerability": answerability,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=os.getenv("SENTRIX_DB_PATH", "data/sentrix.db"))
    parser.add_argument("--samples-root", default=DEFAULT_SAMPLES)
    parser.add_argument("--report", default=None, help="write audit JSON to this path")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from backend.db import MemoryStore

    cases = _load_queries(args.samples_root)
    print(f"[audit] loaded {len(cases)} cases from {args.samples_root}")

    store = MemoryStore(args.db)
    audit = []
    for case in cases:
        filename_to_id = _asset_ids_by_filename(store, case["album"])
        audit.append(_audit_case(store, case, filename_to_id))
    store.close()

    summary = {
        "total": len(audit),
        "by_answerability": {key: sum(1 for item in audit if item["answerability"][key]) for key in (
            "metadata", "confirmed_entity", "lexical_observation", "visual_semantic",
            "text_semantic", "external_geo_required", "formation_missing", "ambiguous_gt")},
        "missing_in_db_count": sum(1 for item in audit if item["missing_in_db"]),
        "gt_inconsistency_count": sum(1 for item in audit if item["answerability"]["ambiguous_gt"]),
        "theoretically_answerable": sum(1 for item in audit if item["answerability"]["metadata"]
                                        and (item["answerability"]["lexical_observation"] or item["answerability"]["visual_semantic"])),
    }
    payload = {"summary": summary, "cases": audit}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
        print(f"wrote {args.report}")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(text)


if __name__ == "__main__":
    main()
