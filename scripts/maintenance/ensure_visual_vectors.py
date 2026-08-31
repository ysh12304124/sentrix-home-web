#!/usr/bin/env python3
"""Ensure every scope's image assets carry the production embedder's visual vector.

Detect: for each scope, compare image-asset count against chinese-clip visual
vector count.  Fix: reembed missing scopes with ChineseClipVisualEmbedder and
sync the Qdrant mirror.  Integrated into the 8091 restart SOP so that after any
restart all scopes are encoded with the current production embedder — a scope
imported under an older embedder (e.g. ViT-B-32) would otherwise keep a silent
``visual_ann`` no-candidate path even though production queries use chinese_clip.

``--field-desc`` additionally ensures every image asset carries the bge-m3
full-description vector (``memory_vectors.space = 'field_desc'``): the text is
the observation's caption + place + objects + event_type + ocr_text, and it is
written to SQLite directly (Qdrant mirror best-effort via upsert_vector), so the
backfill works even when Qdrant is stopped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _observation_desc_text(observation: dict) -> str:
    """完整描述文本：caption + place + objects + event_type + ocr_text（observations 字段）。

    必须与 backend/pipeline.py 的 _field_desc_text 拼法一致，检索侧才能复用同一
    语义通道做余弦。
    """
    caption = str(observation.get("caption") or "")
    place = str(observation.get("place") or "")
    objects = observation.get("objects") or []

    def _object_text(item):
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            parts = [
                str(item.get(key, "")).strip()
                for key in ("label", "primary", "details")
                if str(item.get(key, "")).strip() not in ("", "[]", "{}", "其他或不确定")
            ]
            return " ".join(parts)
        return str(item)

    object_text = " ".join(_object_text(item) for item in objects)
    event_type = str(observation.get("event_type") or "")
    ocr_text = str(observation.get("ocr_text") or "")
    return " ".join(filter(None, [caption, place, object_text, event_type, ocr_text]))


def _field_desc_missing(store, scopes):
    """每个 scope：图片资产数 > field_desc 资产向量数 → 缺失。"""
    missing = []
    for scope_id in scopes:
        image_assets = store.connection.execute(
            "SELECT COUNT(*) FROM assets WHERE scope_id=? AND media_type='image'",
            (scope_id,),
        ).fetchone()[0]
        field_vectors = store.connection.execute(
            "SELECT COUNT(*) FROM memory_vectors WHERE scope_id=? AND space='field_desc' AND source_type='asset'",
            (scope_id,),
        ).fetchone()[0]
        if image_assets > field_vectors:
            missing.append({"scope_id": scope_id, "image_assets": image_assets,
                            "field_desc_vectors": field_vectors})
    return missing


def _backfill_field_desc(store, embedder, scope_id=None):
    """为 scope 内每张图片生成 field_desc 向量；无观察或编码失败跳过。"""
    generated = 0
    skipped_no_observation = 0
    skipped_embed_fail = 0
    if scope_id:
        rows = store.connection.execute(
            "SELECT id FROM assets WHERE scope_id=? AND media_type='image' ORDER BY created_at",
            (scope_id,),
        ).fetchall()
    else:
        rows = store.connection.execute(
            "SELECT id FROM assets WHERE media_type='image' ORDER BY created_at",
        ).fetchall()
    for row in rows:
        asset_id = row["id"]
        observations = store.list_observations(asset_id=asset_id, limit=1)
        if not observations:
            skipped_no_observation += 1
            continue
        observation = observations[0]
        desc_text = _observation_desc_text(observation)
        vector = embedder.embed_query(desc_text)
        if not vector:
            skipped_embed_fail += 1
            continue
        store.upsert_vector(
            "field_desc", "asset", asset_id, vector, embedder.model_id,
            {"observation_id": observation["id"], "scope_id": scope_id,
             "text": desc_text},
        )
        generated += 1
    return {"generated": generated, "skipped_no_observation": skipped_no_observation,
            "skipped_embed_fail": skipped_embed_fail}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/sentrix.db")
    parser.add_argument("--apply", action="store_true", help="reembed missing scopes and sync Qdrant")
    parser.add_argument("--scope", default="",
                        help="comma-separated scope ids to ensure; empty = every scope")
    parser.add_argument("--field-desc", action="store_true",
                        help="also ensure bge-m3 field_desc description vectors for image assets")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    os.environ["SENTRIX_VECTOR_BACKEND"] = "qdrant"

    from backend.db import MemoryStore
    from backend.qdrant_memory import get_qdrant_index
    from backend.embeddings.chinese_clip_visual import ChineseClipVisualEmbedder

    MODEL = "chinese-clip-ViT-L-14"
    store = MemoryStore(args.db)
    index = get_qdrant_index(args.db)

    if args.scope:
        scopes = [scope_id.strip() for scope_id in args.scope.split(",") if scope_id.strip()]
    else:
        scopes = [row[0] for row in store.connection.execute("SELECT id FROM memory_spaces").fetchall()]
    missing = []
    for scope_id in scopes:
        image_assets = store.connection.execute(
            "SELECT COUNT(*) FROM assets WHERE scope_id=? AND media_type='image'",
            (scope_id,),
        ).fetchone()[0]
        visual_vectors = store.connection.execute(
            "SELECT COUNT(*) FROM memory_vectors WHERE scope_id=? AND space='visual' AND model_name=?",
            (scope_id, MODEL),
        ).fetchone()[0]
        if image_assets > visual_vectors:
            missing.append({"scope_id": scope_id, "image_assets": image_assets,
                            "visual_vectors": visual_vectors})

    from backend.retrieval_indexes import RetrievalIndex
    fts_missing = []
    for scope_id in scopes:
        observations = store.connection.execute(
            "SELECT COUNT(*) FROM observations WHERE scope_id=?", (scope_id,),
        ).fetchone()[0]
        fts_rows = store.connection.execute(
            "SELECT COUNT(*) FROM observation_search_fts WHERE scope_id=?", (scope_id,),
        ).fetchone()[0]
        if observations > fts_rows:
            fts_missing.append({"scope_id": scope_id, "observations": observations,
                                "fts_rows": fts_rows})

    field_missing = _field_desc_missing(store, scopes) if args.field_desc else []

    result = {"model": MODEL, "missing_scope_count": len(missing), "missing_scopes": missing,
              "fts_missing_scope_count": len(fts_missing), "fts_missing_scopes": fts_missing}
    if args.field_desc:
        result["field_desc_missing_scope_count"] = len(field_missing)
        result["field_desc_missing_scopes"] = field_missing
    if not args.apply:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("DRY_RUN: pass --apply to reembed missing visual vectors, rebuild FTS, and sync Qdrant")
        return

    if index is None and missing:
        raise RuntimeError("Qdrant unavailable or locked; stop 8091 before --apply")

    if missing:
        from scripts.maintenance.sync_qdrant_vectors import reembed_visual_assets, sync
        embedder = ChineseClipVisualEmbedder()
        fixed = []
        for missing_scope in missing:
            scope_id = missing_scope["scope_id"]
            re = reembed_visual_assets(store, embedder, scope_id=scope_id)
            sy = sync(store, index, scope_id=scope_id)
            fixed.append({"scope_id": scope_id, "reembed": re, "sync": sy})
        result["fixed"] = fixed

    if fts_missing:
        fts_index = RetrievalIndex(store)
        fixed_fts = []
        for missing_scope in fts_missing:
            scope_id = missing_scope["scope_id"]
            rebuilt = fts_index.rebuild_all(scope_id)
            fixed_fts.append({"scope_id": scope_id, "rebuilt_observations": rebuilt})
        result["fixed_fts"] = fixed_fts

    if args.field_desc:
        from backend.embeddings.bge_text import BgeM3TextQueryEmbedder
        fd_embedder = BgeM3TextQueryEmbedder()
        field_fixed = []
        if args.scope:
            targets = scopes
        else:
            targets = [m["scope_id"] for m in field_missing]
        for scope_id in targets:
            counts = _backfill_field_desc(store, fd_embedder, scope_id=scope_id)
            field_fixed.append({"scope_id": scope_id, **counts})
        result["field_desc_fixed"] = field_fixed
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
