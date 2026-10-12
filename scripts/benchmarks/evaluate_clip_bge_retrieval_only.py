#!/usr/bin/env python3
"""Direct CLIP coarse retrieval + BGE reranking, without Agent or Judge.

This is the baseline counterpart of ``evaluate_qwen3_vl_retrieval_only.py``:
questions are encoded directly by CLIP, ranked against the frozen CLIP visual
vectors, then the same Top-N candidates are passed to BGE Reranker v2 M3.
Video vectors are built temporarily from eight uniformly sampled frames and
are never written to SQLite or Qdrant.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.benchmarks.evaluate_qwen3_vl_retrieval_only import (  # noqa: E402
    asset_maps,
    group_name,
    gt_ids,
    rank_metrics,
    retrieval_query,
    select_cases,
    summarize,
)


def _normalise(vector) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    return value / norm if norm else value


def _stored_clip_vectors(db_path: Path, scope_id: str) -> dict[str, np.ndarray]:
    connection = sqlite3.connect(str(db_path))
    try:
        rows = connection.execute(
            "SELECT source_id, vector_json FROM memory_vectors "
            "WHERE scope_id=? AND space='visual' AND source_type='asset' "
            "AND model_name='ViT-B-32'",
            (scope_id,),
        ).fetchall()
    finally:
        connection.close()
    return {str(source_id): _normalise(json.loads(vector_json))
            for source_id, vector_json in rows}


def _temporary_video_vectors(clip, assets: list[dict], frames_per_video: int = 8):
    import cv2
    import torch
    from PIL import Image

    model, preprocess = clip._load()
    if model is None:
        raise RuntimeError(f"CLIP failed to load: {clip.error}")
    result: dict[str, np.ndarray] = {}
    for index, asset in enumerate(assets, 1):
        path = str(asset.get("path") or "")
        capture = cv2.VideoCapture(path)
        frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
        positions = np.linspace(0, frame_count - 1, frames_per_video).astype(int)
        tensors = []
        for position in positions:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
            ok, frame = capture.read()
            if ok and frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensors.append(preprocess(Image.fromarray(rgb)))
        capture.release()
        if not tensors:
            print(f"video vector skipped (no readable frames): {path}", flush=True)
            continue
        batch = torch.stack(tensors).to(clip.device)
        with torch.inference_mode():
            frame_vectors = model.encode_image(batch).float()
            frame_vectors = frame_vectors / frame_vectors.norm(dim=1, keepdim=True).clamp_min(1e-12)
            vector = frame_vectors.mean(dim=0)
            vector = vector / vector.norm().clamp_min(1e-12)
        result[str(asset["id"])] = vector.detach().cpu().numpy().astype(np.float32)
        print(f"temporary CLIP video vectors [{index}/{len(assets)}]", flush=True)
    return result


def _rank(query_vector: np.ndarray, candidates: dict[str, np.ndarray], limit: int):
    return [asset_id for asset_id, _score in sorted(
        ((asset_id, float(np.dot(query_vector, vector)))
         for asset_id, vector in candidates.items()),
        key=lambda item: (-item[1], item[0]),
    )[:limit]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--qa-dir", default=str(
        ROOT / "services" / "photobench" / "data" / "album3-max-video10" / "qa"
    ))
    parser.add_argument("--scope", default="album3-max-video10-migrated")
    parser.add_argument("--coarse-top-k", type=int, default=50)
    parser.add_argument("--frames-per-video", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("CLIP_CHECKPOINT", str(Path.home() / ".cache" / "clip" / "ViT-B-32.pt"))
    os.environ.setdefault("CLIP_DEVICE", "cuda")
    os.environ.setdefault("SENTRIX_RERANKER_DEVICE", "cuda")
    os.environ.setdefault("SENTRIX_RERANKER_BATCH_SIZE", "8")
    os.environ.setdefault("SENTRIX_RERANKER_MAX_LENGTH", "512")
    os.environ.setdefault("SENTRIX_RERANKER_IDLE_UNLOAD_SECONDS", "0")
    os.environ.setdefault("SENTRIX_RERANKER_COARSE_RANK_WEIGHT", "1.0")
    os.environ.setdefault("SENTRIX_RERANKER_BGE_RANK_WEIGHT", "0.1")
    os.environ.setdefault("SENTRIX_RERANKER_FUSION_RRF_K", "60")

    from backend.db import MemoryStore
    from backend.model_clients import ClipAdapter
    from backend.retrieval.reranker import get_local_reranker, rerank_candidates

    db_path = Path(args.db)
    store = MemoryStore(str(db_path))
    clip = ClipAdapter()
    cases = select_cases(Path(args.qa_dir), "balanced98")
    by_name, name_by_id = asset_maps(store, args.scope)
    scope_assets = store.list_assets(scope_id=args.scope, limit=100000)
    image_ids = {str(asset["id"]) for asset in scope_assets
                 if str(asset.get("media_type") or "").lower() == "image"}
    videos = [asset for asset in scope_assets
              if str(asset.get("media_type") or "").lower() == "video"]
    stored = _stored_clip_vectors(db_path, args.scope)
    image_vectors = {asset_id: vector for asset_id, vector in stored.items()
                     if asset_id in image_ids}
    video_vectors = _temporary_video_vectors(clip, videos, args.frames_per_video)
    scorer = get_local_reranker()

    rows, missing_gt, errors = [], [], []
    print(f"direct baseline: {len(image_vectors)} image vectors, "
          f"{len(video_vectors)} temporary video vectors, {len(cases)} questions", flush=True)
    for index, case in enumerate(cases, 1):
        question = retrieval_query(case)
        expected = gt_ids(case, by_name)
        if not expected:
            missing_gt.append(case.get("qa_id"))
            continue
        group = group_name(case)
        pool = video_vectors if group == "video" else image_vectors
        coarse_started = time.perf_counter()
        query_vector = _normalise(clip.embed_text(question))
        coarse_ids = _rank(query_vector, pool, args.coarse_top_k)
        coarse_ms = (time.perf_counter() - coarse_started) * 1000

        rerank_started = time.perf_counter()
        try:
            reranked = rerank_candidates(
                question, coarse_ids, store, scorer=scorer,
                window_seconds=float(os.getenv("SENTRIX_RERANKER_CLUSTER_WINDOW_SECONDS", "1")),
            )
            reranked_ids = reranked.asset_ids
            cluster_count = len(reranked.clusters)
            rerank_status = reranked.telemetry.get("status")
            telemetry = reranked.telemetry
        except Exception as exc:
            reranked_ids = list(coarse_ids)
            cluster_count = len(coarse_ids)
            rerank_status = "error"
            telemetry = {}
            errors.append({"qa_id": case.get("qa_id"), "error": repr(exc)})
        rerank_ms = (time.perf_counter() - rerank_started) * 1000
        rows.append({
            "qa_id": case.get("qa_id"), "group": group, "question": question,
            "gt_asset_ids": expected,
            "gt_file_names": [name_by_id.get(value, value) for value in expected],
            "coarse_ids": coarse_ids,
            "coarse_top8": [name_by_id.get(value, value) for value in coarse_ids[:8]],
            "coarse": rank_metrics(coarse_ids, expected),
            "coarse_ms": round(coarse_ms, 1),
            "reranked_ids": reranked_ids,
            "reranked_top8": [name_by_id.get(value, value) for value in reranked_ids[:8]],
            "reranked": rank_metrics(reranked_ids, expected),
            "rerank_ms": round(rerank_ms, 1),
            "rerank_status": rerank_status,
            "rerank_telemetry": telemetry,
            "temporal_clusters": cluster_count,
        })
        if index == 1 or index % 10 == 0 or index == len(cases):
            print(f"[{index}/{len(cases)}] {group} {case.get('qa_id')}", flush=True)

    groups = {}
    for group in ("image", "video", "all"):
        selected = rows if group == "all" else [row for row in rows if row["group"] == group]
        coarse = summarize(selected, "coarse")
        reranked = summarize(selected, "reranked")
        groups[group] = {
            "coarse_embedding": coarse,
            "after_reranker": reranked,
            "quality_delta": {
                key: round(reranked[key] - coarse[key], 4)
                for key in ("r_at_1", "r_at_3", "r_at_5", "r_at_8", "mrr",
                            "precision_at_8_macro", "recall_at_8_macro", "f1_at_8_macro")
            },
            "average_candidates": round(statistics.mean(len(row["coarse_ids"])
                                                          for row in selected), 1),
            "average_temporal_clusters": round(statistics.mean(row["temporal_clusters"]
                                                                 for row in selected), 1),
        }

    result = {
        "schema_version": "clip_bge_retrieval_only_v1",
        "selected": len(cases), "evaluated": len(rows), "scope_id": args.scope,
        "coarse_top_k": args.coarse_top_k, "frames_per_video": args.frames_per_video,
        "embedding_model": clip.model_name,
        "reranker_model": Path(os.getenv("SENTRIX_RERANKER_MODEL_PATH", "")).name,
        "ground_truth_policy": "metrics_only_after_ranking",
        "precision_recall_cutoff": 8,
        "vector_counts": {"image": len(image_vectors), "video_temporary": len(video_vectors)},
        "missing_gt": missing_gt, "errors": errors,
        "groups": groups, "rows": rows,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(groups, ensure_ascii=False, indent=2), flush=True)
    print(f"saved: {output.resolve()}", flush=True)
    scorer.unload()
    return 0 if not missing_gt and not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
