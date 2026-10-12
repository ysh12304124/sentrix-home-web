#!/usr/bin/env python3
"""Compare the existing multi-channel RRF head with temporal+BGE reranking.

The two variants share the exact same coarse candidates for every question,
so the measured quality/latency delta is isolated to the second-stage
reranker.  No GT field is passed to clustering, passage construction, or the
model.

Default evaluation set:
  * all 48 questions from single-video-48q.jsonl
  * first 50 image-only questions in image-related-439q.jsonl (file order)

Usage (from the repository root, conda ``memory``):
  python scripts/benchmarks/evaluate_bge_reranker.py --out docs/reports/reranker.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _media_refs(row: dict) -> list[dict]:
    refs = row.get("retrieval_media_refs") or row.get("answer_evidence_media_refs") or []
    if refs:
        return [item for item in refs if isinstance(item, dict)]
    refs = []
    refs.extend({"media_type": "image", "media_id": value}
                for value in (row.get("retrieval_image_ids") or []))
    refs.extend({"media_type": "video", "media_id": value}
                for value in (row.get("retrieval_video_ids") or []))
    return refs


def _select_cases(
    qa_dir: Path, video_limit: int, image_limit: int, *, selection: str,
) -> list[dict]:
    video_rows = _read_jsonl(qa_dir / "single-video-48q.jsonl")
    image_rows = _read_jsonl(qa_dir / "image-related-439q.jsonl")
    video_cases = []
    for row in video_rows:
        refs = _media_refs(row)
        if refs and all(item.get("media_type") == "video" for item in refs):
            video_cases.append({**row, "benchmark_group": "video"})
        if len(video_cases) >= video_limit:
            break
    if selection == "legacy_ab":
        # Reproduce the previously saved `retrieval-ab-image-sample50.json`.
        # Despite its historical filename this is the first 50 source rows and
        # intentionally includes five video-referenced questions.
        image_cases = [
            {**row, "benchmark_group": "image_sample50"}
            for row in image_rows[:image_limit]
        ]
    else:
        image_cases = []
        for row in image_rows:
            refs = _media_refs(row)
            if refs and all(item.get("media_type") == "image" for item in refs):
                image_cases.append({**row, "benchmark_group": "image"})
            if len(image_cases) >= image_limit:
                break
    return video_cases + image_cases


def _retrieval_query(item: dict) -> str:
    messages = [str(turn.get("message") or "").strip()
                for turn in (item.get("conversation") or [])]
    messages = [message for message in messages if message]
    return " ".join(messages[-3:]) or str(item.get("question") or "").strip()


def _remote_coarse(
    base_url: str,
    scope_id: str,
    query: str,
    limit: int = 50,
    *,
    backend: str = "hybrid",
) -> dict:
    payload = json.dumps({
        "scope_id": scope_id, "query": query, "backend": backend, "limit": limit,
    }, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/search",
        data=payload,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))


def _remote_health(base_url: str) -> dict:
    request = urllib.request.Request(f"{base_url.rstrip('/')}/health", method="GET")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _asset_lookup(store, scope_id: str) -> tuple[dict[str, str], dict[str, str]]:
    by_name: dict[str, str] = {}
    name_by_id: dict[str, str] = {}
    for asset in store.list_assets(scope_id=scope_id, limit=100000):
        asset_id = str(asset.get("id") or "")
        file_name = str(asset.get("file_name") or "")
        if asset_id and file_name:
            by_name.setdefault(file_name.casefold(), asset_id)
            name_by_id[asset_id] = file_name
    return by_name, name_by_id


def _gt_asset_ids(row: dict, by_name: dict[str, str]) -> list[str]:
    out = []
    for ref in _media_refs(row):
        media_id = str(ref.get("media_id") or "").replace("\\", "/")
        name = media_id.rsplit("/", 1)[-1]
        if ref.get("media_type") == "video" and not Path(name).suffix:
            name += ".mp4"
        asset_id = by_name.get(name.casefold())
        if asset_id and asset_id not in out:
            out.append(asset_id)
    return out


def _spec(query: str, scope_id: str, media_type: str):
    from backend.query_contracts import QueryParseDraft, build_query_spec

    draft = QueryParseDraft(intent="search", answer_target="asset_set")
    draft.semantic_conditions.append({
        "dimension": "semantic",
        "value": query,
        "source_text": query,
        "strictness": "semantic_required",
    })
    # The two benchmark tracks have an explicit media boundary.  This is not
    # an answer-derived hint: it is equivalent to the UI dataset selector and
    # prevents 334 photos from crowding out a ten-video track (and vice versa).
    draft.media_expressions.append("视频" if media_type == "video" else "照片")
    draft.result_requirement = {"mode": "best", "top_k": 50}
    return build_query_spec(
        draft,
        scope_id=scope_id,
        viewer_id="owner",
        conversation_id="reranker-benchmark",
        query_id="reranker-benchmark",
    )


def _rank_metrics(ranked: list[str], gt_ids: list[str]) -> dict:
    gt_set = set(gt_ids)
    first_rank = next((index for index, asset_id in enumerate(ranked, 1)
                       if asset_id in gt_set), None)
    top8 = ranked[:8]
    hits_at_8 = len(set(top8) & gt_set)
    precision_at_8 = hits_at_8 / len(top8) if top8 else 0.0
    recall_at_8 = hits_at_8 / len(gt_set) if gt_set else 0.0
    f1_at_8 = (
        2.0 * precision_at_8 * recall_at_8 / (precision_at_8 + recall_at_8)
        if precision_at_8 + recall_at_8 else 0.0
    )
    return {
        "r1": int(first_rank is not None and first_rank <= 1),
        "r3": int(first_rank is not None and first_rank <= 3),
        "r5": int(first_rank is not None and first_rank <= 5),
        "r8": int(first_rank is not None and first_rank <= 8),
        "mrr": (1.0 / first_rank) if first_rank else 0.0,
        "first_rank": first_rank,
        "precision_at_8": precision_at_8,
        "recall_at_8": recall_at_8,
        "f1_at_8": f1_at_8,
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summarize(rows: list[dict], variant: str) -> dict:
    metrics = [row[variant] for row in rows]
    latencies = [row[f"{variant}_latency_ms"] for row in rows]
    count = max(1, len(rows))
    return {
        "questions": len(rows),
        "r_at_1": round(sum(item["r1"] for item in metrics) / count, 4),
        "r_at_3": round(sum(item["r3"] for item in metrics) / count, 4),
        "r_at_5": round(sum(item["r5"] for item in metrics) / count, 4),
        "r_at_8": round(sum(item["r8"] for item in metrics) / count, 4),
        "mrr": round(sum(item["mrr"] for item in metrics) / count, 4),
        "precision_at_8_macro": round(
            sum(item["precision_at_8"] for item in metrics) / count, 4
        ),
        "recall_at_8_macro": round(
            sum(item["recall_at_8"] for item in metrics) / count, 4
        ),
        "f1_at_8_macro": round(
            sum(item["f1_at_8"] for item in metrics) / count, 4
        ),
        "latency_median_ms": round(statistics.median(latencies), 1) if latencies else 0.0,
        "latency_p95_ms": round(_percentile(latencies, 0.95), 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(ROOT / "data" / "sentrix.db"))
    parser.add_argument("--qa-dir", default=str(
        ROOT / "services" / "photobench" / "data" / "album3-max-video10" / "qa"
    ))
    parser.add_argument("--scope", default="album3-max-video10-migrated")
    parser.add_argument("--selection", choices=("legacy_ab", "media_only"),
                        default="legacy_ab")
    parser.add_argument("--coarse-url", default="",
                        help="Optional compatibility endpoint; default uses the native kernel")
    parser.add_argument(
        "--coarse-backend", choices=("hybrid", "multiroute_rrf", "vector"),
        default="hybrid", help="Remote Sentrix Lite retrieval policy",
    )
    parser.add_argument(
        "--candidate-cache-in", default="",
        help="Reuse full coarse candidate IDs from a previous evaluator report",
    )
    parser.add_argument(
        "--skip-rerank", action="store_true",
        help="Only collect/evaluate the coarse Top-K candidate cache",
    )
    parser.add_argument("--coarse-scope", default="scope_06a94e4483964088")
    parser.add_argument("--video-limit", type=int, default=48)
    parser.add_argument("--image-limit", type=int, default=50)
    parser.add_argument(
        "--qa-id", action="append", default=[],
        help="Evaluate only the selected QA ID; may be repeated for targeted retries",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    _load_dotenv(ROOT / ".env.local")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("SENTRIX_EVIDENCE_MULTI_RETRIEVER_V1", "1")
    os.environ.setdefault("SENTRIX_RETRIEVER_RANKING", "weighted_rrf")
    os.environ.setdefault("SENTRIX_SEARCH_CANDIDATE_TOP_K", "50")
    os.environ.setdefault("SENTRIX_IMAGE_EMBEDDER", "clip")
    os.environ.setdefault("SENTRIX_TEXT_EMBEDDER", "bge")
    os.environ.setdefault("SENTRIX_TEXT_EMBEDDER_URL", "http://127.0.0.1:8101")
    os.environ.setdefault(
        "SENTRIX_TEXT_EMBED_MODEL", os.getenv("LOCAL_BGE_MODEL_PATH", "BAAI/bge-m3")
    )
    os.environ.setdefault("SENTRIX_VECTOR_BACKEND", "qdrant")
    os.environ.setdefault("SENTRIX_QDRANT_PATH", str(ROOT / "data" / "qdrant"))
    os.environ.setdefault("SENTRIX_QDRANT_COLLECTION_PREFIX", "sentrix_local")

    from backend.db import MemoryStore
    from backend.embeddings import EmbeddingRouter
    from backend.evidence_retrieval import EvidenceRetrievalKernel
    from backend.model_clients import ClipAdapter
    from backend.retrieval import RetrievalConfig, build_default_retrievers
    from backend.retrieval.reranker import get_local_reranker, rerank_candidates

    store = MemoryStore(args.db)
    clip = ClipAdapter()
    router = EmbeddingRouter.from_clip(clip)
    config = RetrievalConfig()
    retrievers = build_default_retrievers(store, embedding_router=router, config=config)
    kernel = EvidenceRetrievalKernel(
        store, retrievers=retrievers, embedding_router=router, config=config,
    )
    coarse_service_health = _remote_health(args.coarse_url) if args.coarse_url else None
    cases = _select_cases(
        Path(args.qa_dir), args.video_limit, args.image_limit, selection=args.selection,
    )
    if args.qa_id:
        requested_qa_ids = {str(value) for value in args.qa_id}
        cases = [case for case in cases if str(case.get("qa_id")) in requested_qa_ids]
    by_name, name_by_id = _asset_lookup(store, args.scope)
    cached_rows: dict[str, dict] = {}
    if args.candidate_cache_in:
        cache_payload = json.loads(
            Path(args.candidate_cache_in).read_text(encoding="utf-8-sig")
        )
        cached_rows = {
            str(row.get("qa_id")): row
            for row in (cache_payload.get("rows") or [])
            if row.get("qa_id") and row.get("coarse_candidate_ids")
        }
    scorer = None if args.skip_rerank else get_local_reranker()
    reranker_model_name = Path(
        os.getenv("SENTRIX_RERANKER_MODEL_PATH", "")
    ).name.casefold()
    if os.getenv("SENTRIX_RERANKER_URL", "").strip():
        reranker_variant = "rrf_plus_temporal_qwen3_vl"
    elif "qwen" in reranker_model_name:
        reranker_variant = "rrf_plus_temporal_qwen3_text"
    else:
        reranker_variant = "rrf_plus_temporal_bge"
    rows = []
    missing_gt = []
    rerank_errors = []

    for index, case in enumerate(cases, 1):
        query = _retrieval_query(case)
        gt_ids = _gt_asset_ids(case, by_name)
        if not gt_ids:
            missing_gt.append(case.get("qa_id"))
            continue
        coarse_started = time.perf_counter()
        cached = cached_rows.get(str(case.get("qa_id")))
        if cached:
            coarse_ids = [
                str(asset_id) for asset_id in cached.get("coarse_candidate_ids") or []
                if asset_id
            ][:50]
            coarse_ms = float(cached.get("rrf_latency_ms") or 0.0)
        elif args.coarse_url:
            response = _remote_coarse(
                args.coarse_url, args.coarse_scope, query, 50,
                backend=args.coarse_backend,
            )
            coarse_ids = []
            for item in response.get("results") or []:
                native_id = by_name.get(str(item.get("file_name") or "").casefold())
                if native_id and native_id not in coarse_ids:
                    coarse_ids.append(native_id)
            coarse_ms = float(response.get("latency_ms") or (
                (time.perf_counter() - coarse_started) * 1000
            ))
        else:
            media_group = "video" if case["benchmark_group"] == "video" else "image"
            packet = kernel.retrieve(_spec(query, args.scope, media_group))
            coarse_ids = [item.get("asset_id") for item in (packet.assets or [])
                          if item.get("asset_id")][:50]
            coarse_ms = (time.perf_counter() - coarse_started) * 1000
        rerank_started = time.perf_counter()
        try:
            if args.skip_rerank:
                reranked_ids = list(coarse_ids)
                rerank_ms = 0.0
                cluster_count = len(coarse_ids)
                rerank_status = "skipped"
            else:
                reranked = rerank_candidates(
                    query,
                    coarse_ids,
                    store,
                    scorer=scorer,
                    window_seconds=float(os.getenv("SENTRIX_RERANKER_CLUSTER_WINDOW_SECONDS", "1")),
                )
                reranked_ids = reranked.asset_ids
                rerank_ms = (time.perf_counter() - rerank_started) * 1000
                cluster_count = len(reranked.clusters)
                rerank_status = reranked.telemetry.get("status")
        except Exception as error:
            reranked_ids = list(coarse_ids)
            rerank_ms = (time.perf_counter() - rerank_started) * 1000
            cluster_count = len(coarse_ids)
            rerank_status = "error"
            rerank_errors.append({"qa_id": case.get("qa_id"), "error": repr(error)})
        rows.append({
            "qa_id": case.get("qa_id"),
            "group": case.get("benchmark_group"),
            "question": query,
            "gt_asset_ids": gt_ids,
            "gt_file_names": [name_by_id.get(asset_id) for asset_id in gt_ids],
            "coarse_candidate_count": len(coarse_ids),
            "temporal_cluster_count": cluster_count,
            "rerank_status": rerank_status,
            "coarse_candidate_ids": coarse_ids,
            "reranked_candidate_ids": reranked_ids,
            "rrf": _rank_metrics(coarse_ids, gt_ids),
            "reranked": _rank_metrics(reranked_ids, gt_ids),
            "rrf_latency_ms": round(coarse_ms, 1),
            "reranked_latency_ms": round(coarse_ms + rerank_ms, 1),
            "rerank_incremental_ms": round(rerank_ms, 1),
            "rrf_top8": [name_by_id.get(asset_id, asset_id) for asset_id in coarse_ids[:8]],
            "reranked_top8": [name_by_id.get(asset_id, asset_id) for asset_id in reranked_ids[:8]],
        })
        if index == 1 or index % 10 == 0 or index == len(cases):
            print(f"[{index}/{len(cases)}] {case.get('benchmark_group')} {case.get('qa_id')}", flush=True)

    groups = {}
    group_names = list(dict.fromkeys(row["group"] for row in rows))
    for group in (*group_names, "all"):
        selected = rows if group == "all" else [row for row in rows if row["group"] == group]
        rrf = _summarize(selected, "rrf")
        groups[group] = {
            "rrf": rrf,
            "average_coarse_candidates": round(statistics.mean(
                row["coarse_candidate_count"] for row in selected
            ), 1) if selected else 0.0,
        }
        if not args.skip_rerank:
            reranked = _summarize(selected, "reranked")
            groups[group].update({
                reranker_variant: reranked,
                "delta": {
                    key: round(reranked[key] - rrf[key], 4)
                    for key in ("r_at_1", "r_at_3", "r_at_5", "r_at_8", "mrr")
                },
                "rerank_incremental_median_ms": round(statistics.median(
                    row["rerank_incremental_ms"] for row in selected
                ), 1) if selected else 0.0,
                "average_temporal_clusters": round(statistics.mean(
                    row["temporal_cluster_count"] for row in selected
                ), 1) if selected else 0.0,
            })

    result = {
        "schema_version": "sentrix_bge_reranker_ablation_v1",
        "dataset": {
            "qa_dir": str(Path(args.qa_dir).resolve()),
            "scope_id": args.scope,
            "selection": (
                "all 48 single-video rows + first 50 image-related source rows (legacy AB compatible)"
                if args.selection == "legacy_ab" else
                "all 48 single-video rows + first 50 image-only rows in source file order"
            ),
            "selected": len(cases),
            "evaluated": len(rows),
            "missing_gt": missing_gt,
        },
        "configuration": {
            "coarse_ranking": (
                f"sentrix_lite_{args.coarse_backend}"
                if args.coarse_url else config.ranking_strategy
            ),
            "coarse_top_k": 50,
            "channels": (
                ["image", "visual_caption", "object", "relation", "speech", "lexical"]
                if args.coarse_url and args.coarse_backend == "multiroute_rrf"
                else (["image", "visual_caption", "object", "relation", "speech"]
                      if args.coarse_url and args.coarse_backend == "vector"
                      else (["image", "visual_caption", "object", "relation", "lexical"]
                            if args.coarse_url
                            else [retriever.name for retriever in retrievers]))
            ),
            "coarse_url": args.coarse_url or None,
            "coarse_backend": args.coarse_backend if args.coarse_url else None,
            "coarse_service_health": coarse_service_health,
            "candidate_cache_in": str(Path(args.candidate_cache_in).resolve())
                                  if args.candidate_cache_in else None,
            "rerank_enabled": not args.skip_rerank,
            "reranker_model_path": os.getenv("SENTRIX_RERANKER_MODEL_PATH"),
            "reranker_url": os.getenv("SENTRIX_RERANKER_URL") or None,
            "reranker_variant": reranker_variant,
            "reranker_device": os.getenv("SENTRIX_RERANKER_DEVICE", "auto"),
            "ranking_policy": "coarse_bge_rank_fusion",
            "coarse_rank_weight": float(os.getenv(
                "SENTRIX_RERANKER_COARSE_RANK_WEIGHT", "1"
            )),
            "bge_rank_weight": float(os.getenv(
                "SENTRIX_RERANKER_BGE_RANK_WEIGHT", "0.25"
            )),
            "fusion_rrf_k": float(os.getenv("SENTRIX_RERANKER_FUSION_RRF_K", "60")),
            "passage_fields": [
                "time", "place", "people", "florence_caption", "ocr", "object",
                "object_description", "relation", "action_event",
            ],
            "visual_text_policy": "fallback_only_when_structured_semantic_fields_are_empty",
            "cluster_window_seconds": float(os.getenv(
                "SENTRIX_RERANKER_CLUSTER_WINDOW_SECONDS", "1"
            )),
        },
        "groups": groups,
        "rerank_errors": rerank_errors,
        "rows": rows,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(groups, ensure_ascii=False, indent=2), flush=True)
    print(f"saved: {output.resolve()}", flush=True)
    if scorer is not None:
        scorer.unload()
    return 0 if not missing_gt and not rerank_errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
