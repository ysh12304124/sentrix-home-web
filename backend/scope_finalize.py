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
from pathlib import Path
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


def _visual_index_health() -> dict:
    """visual ANN 索引是否存在、与当前图像嵌入方案是否匹配。"""
    import json as _json

    from .embeddings.scheme import IMAGE_MODEL
    from .retrieval.text_ann import _DEFAULT_ANN_DIR

    ann_dir = Path(_DEFAULT_ANN_DIR)
    index = ann_dir / "visual.hnsw"
    manifest = ann_dir / "visual.manifest.json"
    if not index.is_file():
        return {"status": "missing",
                "detail": f"visual ANN 索引不存在（{index}）—— 视觉检索永远返回空候选"}
    if not manifest.is_file():
        return {"status": "no_manifest", "detail": f"缺少 {manifest.name}"}
    try:
        data = _json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"status": "unreadable", "detail": f"{manifest.name} 无法解析：{exc}"}
    if str(data.get("model_id")) != str(IMAGE_MODEL):
        return {"status": "model_mismatch",
                "detail": f"索引用的嵌入模型是 {data.get('model_id')}，当前是 {IMAGE_MODEL}"
                          " —— 查询向量与索引不在同一空间，检索结果无意义"}
    return {"status": "ok", "source_count": int(data.get("source_count") or 0)}


def verify_scope_retrieval(store, scope_id: str) -> dict:
    """核对这个 scope 的检索数据是否真的可用。

    为什么必须显式核对：视觉向量与 visual ANN **不是**处理资产时产生的，而是收尾时
    补的。批次中途退出（OOM / 重启 / 部署）时这一步不跑，结果就是相册建好了、资产
    全是 processed、却一条视觉向量都没有、检索永远返回空 —— 而链路上没有任何一处
    会报错。46 上踩过一次：223 张图没有向量，直到手工清点才发现。
    """
    def _count(sql, params):
        row = store.connection.execute(sql, params).fetchone()
        if row is None:
            return 0
        try:
            return int(row["n"] or 0)
        except (TypeError, IndexError, KeyError):
            return int(row[0] or 0)

    images = _count(
        "SELECT COUNT(*) AS n FROM assets WHERE scope_id=? AND media_type='image'", (scope_id,))
    visual = _count(
        "SELECT COUNT(*) AS n FROM memory_vectors WHERE scope_id=? AND space='visual'", (scope_id,))
    problems = []
    if images and not visual:
        problems.append(f"该 scope 有 {images} 张图但视觉向量为 0 —— 视觉检索会永远返回空")
    elif images and visual < images:
        problems.append(f"视觉向量 {visual} < 图片 {images}，有 {images - visual} 张图检索不到")

    index = _visual_index_health()
    if index.get("status") != "ok":
        problems.append(index.get("detail") or "visual ANN 索引不可用")

    return {
        "ok": not problems,
        "scope_id": scope_id,
        "images": images,
        "visual_vectors": visual,
        "visual_index": index,
        "problems": problems,
    }


def run(store, scope_id: str) -> dict:
    from .retrieval_indexes import RetrievalIndex
    t0 = time.perf_counter()
    visual = ensure_chinese_clip_asset_vectors(store, scope_id)
    terms = RetrievalIndex(store).rebuild_all(scope_id=scope_id)
    health = verify_scope_retrieval(store, scope_id)
    return {
        "scope_id": scope_id,
        "visual": visual,
        "search_terms_rebuilt": int(terms or 0),
        "seconds": round(time.perf_counter() - t0, 1),
        "health": health,
        "ok": bool(health["ok"]),
    }


def finalize_ingest_scope(store, scope_id):
    """Finish derived retrieval data before publishing batch completion."""
    if not scope_id:
        return None
    import logging
    log = logging.getLogger(__name__)
    try:
        result = run(store, scope_id)
        visual = result.get("visual") or {}
        if visual.get("error") or visual.get("failed"):
            log.error("scope retrieval finalization incomplete: %s", result)
        # 收尾"没报错"不等于"检索可用"：视觉向量为 0 或 ANN 缺失时链路一声不响，
        # 所以这里按核对结果显式告警，而不是只看有没有抛异常。
        if not (result.get("health") or {}).get("ok", True):
            log.error(
                "scope=%s 检索数据不完整，相册建好但检索会失效：%s",
                scope_id, (result.get("health") or {}).get("problems"),
            )
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
