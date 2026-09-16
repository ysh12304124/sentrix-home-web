"""Scope 收尾：让一个新建相册的检索资产完全落到单一方案（embeddings/scheme.py）。

历史坑：albums 建完只写了 clip/bge 之类"当时的"向量 + 常常漏了 FTS/索引，之后复用测评
检索失效。收尾统一做三件事（旧相册不跑，新相册构建后必跑）：

1) 给该 scope 每个 image asset 补写 chinese-clip 的 asset visual 向量
   （model_name = chinese-clip-ViT-L-14）。clip(ViT-B-32) 等旧模型行保留不删、不再新增。
2) 重建该 scope 的 observation_search_terms / FTS，并把 reverse_geocode 地名 +
   captured_at 时间 + 文件名写进 token（地名/时间题的词法锚）。
3) 返回统计，供调用方把 scope 加入后续 ANN 索引重建清单。

用法（在仓库根目录用带 hnswlib/chinese-clip 的 .venv python）：
  python -m backend.scope_finalize --db data/sentrix.db --scope <scope_id>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def ensure_chinese_clip_asset_vectors(store, scope_id: str) -> dict:
    from .embeddings.scheme import IMAGE_MODEL
    embedder = None
    rows = store.connection.execute(
        "SELECT id, file_name, path FROM assets WHERE scope_id=? AND media_type='image'",
        (scope_id,),
    ).fetchall()
    added = skipped = failed = 0
    existing = {
        row["source_id"]
        for row in store.connection.execute(
            "SELECT source_id FROM memory_vectors WHERE scope_id=? AND space='visual' "
            "AND source_type='asset' AND model_name=?",
            (scope_id, IMAGE_MODEL),
        ).fetchall()
    }
    for row in rows:
        asset_id = row["id"]
        if asset_id in existing:
            skipped += 1
            continue
        path = row["path"] or ""
        if not path or not os.path.isfile(path):
            skipped += 1
            continue
        try:
            if embedder is None:
                from .embeddings.chinese_clip_visual import ChineseClipVisualEmbedder
                embedder = ChineseClipVisualEmbedder.shared()
                if not getattr(embedder, "available", False):
                    return {"added": added, "skipped": skipped, "failed": failed,
                            "error": "chinese_clip embedder unavailable"}
            vector = embedder.embed_image(path)
            if not vector:
                failed += 1
                continue
            store.upsert_vector("visual", "asset", asset_id, vector, IMAGE_MODEL,
                                {"scope_id": scope_id})
            added += 1
        except Exception as exc:  # noqa: BLE001 - per-asset failure must not abort scope
            failed += 1
            if added == 0 and failed == 1:
                raise
    return {"added": added, "skipped": skipped, "failed": failed}


def run(store, scope_id: str) -> dict:
    from .retrieval_indexes import RetrievalIndex
    t0 = time.perf_counter()
    visual = ensure_chinese_clip_asset_vectors(store, scope_id)
    terms = RetrievalIndex(store).rebuild_all(scope_id=scope_id)
    return {
        "scope_id": scope_id,
        "visual": visual,
        "search_terms_rebuilt": int(terms or 0),
        "seconds": round(time.perf_counter() - t0, 1),
    }


def finalize_ingest_scope(store, scope_id):
    """Finish derived retrieval data before publishing batch completion."""
    if not scope_id:
        return None
    import logging
    try:
        result = run(store, scope_id)
        visual = result.get("visual") or {}
        if visual.get("error") or visual.get("failed"):
            logging.getLogger(__name__).error("scope retrieval finalization incomplete: %s", result)
        return result
    except Exception as exc:
        logging.getLogger(__name__).exception("scope retrieval finalization failed: %s", scope_id)
        return {"scope_id": scope_id, "error": str(exc)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=os.getenv("SENTRIX_DB_PATH", "data/sentrix.db"))
    parser.add_argument("--scope", required=True)
    args = parser.parse_args(argv)
    sys.path.insert(0, os.getcwd())
    from .db import MemoryStore  # noqa: PLC0415
    store = MemoryStore(args.db)
    try:
        print(json.dumps(run(store, args.scope), ensure_ascii=False, indent=2))
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
