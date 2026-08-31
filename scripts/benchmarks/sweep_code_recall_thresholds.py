#!/usr/bin/env python3
"""Measure pure code retrieval thresholds for an album QA set.

This harness deliberately does not instantiate an LLM client or call
``search_memories``. It evaluates the production retrieval channels and their
code-only ranking/threshold behaviour without an LLM call.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path


def _percentile(values: list[int], point: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * point)))
    return float(ordered[index])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/sentrix.db")
    parser.add_argument("--qa", required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--thresholds", default="0,0.55,0.60,0.65,0.70,0.72,0.74,0.76,0.78,0.80,0.82,0.84,0.86")
    args = parser.parse_args()

    # A second process must not open the live local-Qdrant store for writing.
    # Its lock-safe code path is the production SQLite cosine fallback, which
    # uses the same persisted vectors and returns raw ANN scores.
    os.environ.setdefault("SENTRIX_VECTOR_BACKEND", "qdrant")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from backend.agent_runtime import tools
    from backend.db import MemoryStore
    from backend.embeddings import EmbeddingRouter
    from backend.model_clients import ClipAdapter
    from backend.retrieval import RetrievalConfig
    from backend.retrieval.base import HardFilterContext, RetrievalQuery
    from backend.retrieval.fusion import DEFAULT_CHANNEL_WEIGHTS, fuse
    from backend.retrieval.ranking import LATE_FUSION, rank
    from backend.retrieval.lexical import LexicalRetriever
    from backend.retrieval.text_ann import TextAnnRetriever
    from backend.retrieval.visual_ann import VisualAnnRetriever

    store = MemoryStore(args.db)
    router = EmbeddingRouter.from_clip(ClipAdapter())
    tools.bind_runtime(store, gamma=None, embedding_router=router,
                       retrieval_config=RetrievalConfig())
    tools.register_tools()
    text_retriever = TextAnnRetriever(store, embedding_router=router)
    visual_retriever = VisualAnnRetriever(store, embedding_router=router)
    lexical_retriever = LexicalRetriever(store)
    filters = HardFilterContext(scope_ids=(args.scope,), viewer_id="owner")
    thresholds = [float(value) for value in args.thresholds.split(",")]

    by_file = {
        str(asset.get("file_name") or ""): str(asset.get("id"))
        for asset in store.list_assets(scope_id=args.scope, limit=100_000)
    }
    rows = [json.loads(line) for line in Path(args.qa).read_text(encoding="utf-8").splitlines() if line.strip()]
    per_question: list[dict] = []

    for row in rows:
        question = str(row.get("question") or "")
        gt_ids = {
            by_file.get(Path(str(image_id)).name)
            for image_id in row.get("retrieval_image_ids") or []
        }
        gt_ids.discard(None)
        query = RetrievalQuery.from_spec(
            tools._spec_for(tools._draft_from_filters({"query": question}, answer_type="asset_set"),
                            args.scope, "owner"),
            embedding_router=router,
        )
        text_hits = text_retriever.retrieve(query, filters, limit=100_000)
        visual_hits = visual_retriever.retrieve(query, filters, limit=100_000)
        lexical_hits = lexical_retriever.retrieve(query, filters, limit=100_000)
        text_scores = {hit.asset_id: float(hit.raw_score) for hit in text_hits}
        visual_scores = {hit.asset_id: float(hit.raw_score) for hit in visual_hits}
        lexical_scores = {hit.asset_id: float(hit.raw_score) for hit in lexical_hits}
        ranked_ids = [hit.asset_id for hit in text_hits]
        channel_hits = {
            "lexical": lexical_hits,
            "text_ann": text_hits,
            "visual_ann": visual_hits,
        }
        visual_backbone = rank(
            channel_hits,
            RetrievalConfig().ranking_strategy,
            100_000,
            fusion_weights=DEFAULT_CHANNEL_WEIGHTS,
        )
        rrf_ranked = fuse(channel_hits, channel_weights=DEFAULT_CHANNEL_WEIGHTS)
        late_fusion = rank(channel_hits, LATE_FUSION, 100_000,
                           fusion_weights=DEFAULT_CHANNEL_WEIGHTS)
        event = tools._event_resolution(question, store, args.scope)
        if event is None:
            event = tools._event_keyword_anchor(question, store, args.scope)
        event_ids = set((event or {}).get("asset_ids") or [])
        gt_ranks = [ranked_ids.index(asset_id) + 1 for asset_id in gt_ids if asset_id in text_scores]
        per_question.append({
            "qa_id": row.get("qa_id"),
            "question": question,
            "gt_asset_ids": sorted(gt_ids),
            "gt_count": len(gt_ids),
            "gt_ranks": gt_ranks,
            "gt_scores": {asset_id: text_scores.get(asset_id) for asset_id in gt_ids},
            "gt_visual_scores": {asset_id: visual_scores.get(asset_id) for asset_id in gt_ids},
            "gt_lexical_scores": {asset_id: lexical_scores.get(asset_id) for asset_id in gt_ids},
            "ann_candidate_count": len(ranked_ids),
            "visual_candidate_count": len(visual_hits),
            "lexical_candidate_count": len(lexical_hits),
            "visual_backbone_ranks": [candidate.asset_id for candidate in visual_backbone],
            "weighted_rrf_ranks": [candidate.asset_id for candidate in rrf_ranked],
            "late_fusion_ranks": [candidate.asset_id for candidate in late_fusion],
            "event_id": (event or {}).get("event_id"),
            "event_asset_ids": sorted(event_ids),
            "event_candidate_count": len(event_ids),
            "event_gt_hit": bool(event_ids & gt_ids),
            "scores": text_scores,
            "visual_scores": visual_scores,
            "lexical_scores": lexical_scores,
        })

    curve = []
    scored_questions = [item for item in per_question if item["gt_count"]]
    for threshold in thresholds:
        candidate_counts = []
        gt_total = gt_hit_total = 0
        question_hits = 0
        event_augmented_question_hits = 0
        for item in scored_questions:
            candidates = {asset_id for asset_id, score in item["scores"].items() if score >= threshold}
            candidate_counts.append(len(candidates))
            gt = set(item["gt_asset_ids"])
            matched = candidates & gt
            gt_total += len(gt)
            gt_hit_total += len(matched)
            question_hits += bool(matched)
            event_augmented_question_hits += bool((candidates | set(item["event_asset_ids"])) & gt)
        curve.append({
            "threshold": threshold,
            "questions": len(scored_questions),
            "asset_recall": round(gt_hit_total / gt_total, 4) if gt_total else None,
            "question_recall": round(question_hits / len(scored_questions), 4) if scored_questions else None,
            "question_recall_with_event_hit": round(event_augmented_question_hits / len(scored_questions), 4) if scored_questions else None,
            "candidate_mean": round(statistics.mean(candidate_counts), 2) if candidate_counts else 0,
            "candidate_p50": _percentile(candidate_counts, 0.5),
            "candidate_p90": _percentile(candidate_counts, 0.9),
            "candidate_max": max(candidate_counts, default=0),
        })

    relative_curve = []
    for delta in (0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15):
        candidate_counts = []
        gt_total = gt_hit_total = question_hits = 0
        for item in scored_questions:
            top_score = max(item["scores"].values(), default=1.0)
            candidates = {
                asset_id for asset_id, score in item["scores"].items()
                if score >= top_score - delta
            }
            gt = set(item["gt_asset_ids"])
            matched = candidates & gt
            candidate_counts.append(len(candidates))
            gt_total += len(gt)
            gt_hit_total += len(matched)
            question_hits += bool(matched)
        relative_curve.append({
            "top_score_delta": delta,
            "asset_recall": round(gt_hit_total / gt_total, 4) if gt_total else None,
            "question_recall": round(question_hits / len(scored_questions), 4) if scored_questions else None,
            "candidate_mean": round(statistics.mean(candidate_counts), 2) if candidate_counts else 0,
            "candidate_p50": _percentile(candidate_counts, 0.5),
            "candidate_p90": _percentile(candidate_counts, 0.9),
            "candidate_max": max(candidate_counts, default=0),
        })

    ranking_curves = {"visual_backbone": [], "weighted_rrf": [], "late_fusion": []}
    for limit in (1, 3, 5, 10, 20, 30, 50, 100):
        for name, rank_key in (
            ("visual_backbone", "visual_backbone_ranks"),
            ("weighted_rrf", "weighted_rrf_ranks"),
            ("late_fusion", "late_fusion_ranks"),
        ):
            gt_total = gt_hit_total = question_hits = 0
            for item in scored_questions:
                candidates = set(item[rank_key][:limit])
                gt = set(item["gt_asset_ids"])
                matched = candidates & gt
                gt_total += len(gt)
                gt_hit_total += len(matched)
                question_hits += bool(matched)
            ranking_curves[name].append({
                "diagnostic_rank_limit": limit,
                "asset_recall": round(gt_hit_total / gt_total, 4) if gt_total else None,
                "question_recall": round(question_hits / len(scored_questions), 4) if scored_questions else None,
            })

    dual_channel_curve = []
    for text_threshold in (0.68, 0.70, 0.72, 0.74, 0.76):
        for visual_threshold in (0.28, 0.30, 0.32, 0.34, 0.36, 0.38):
            candidate_counts = []
            gt_total = gt_hit_total = question_hits = 0
            for item in scored_questions:
                candidates = {
                    asset_id for asset_id, score in item["scores"].items()
                    if score >= text_threshold
                } | {
                    asset_id for asset_id, score in item["visual_scores"].items()
                    if score >= visual_threshold
                }
                gt = set(item["gt_asset_ids"])
                matched = candidates & gt
                candidate_counts.append(len(candidates))
                gt_total += len(gt)
                gt_hit_total += len(matched)
                question_hits += bool(matched)
            dual_channel_curve.append({
                "text_threshold": text_threshold,
                "visual_threshold": visual_threshold,
                "asset_recall": round(gt_hit_total / gt_total, 4) if gt_total else None,
                "question_recall": round(question_hits / len(scored_questions), 4) if scored_questions else None,
                "candidate_mean": round(statistics.mean(candidate_counts), 2) if candidate_counts else 0,
                "candidate_p50": _percentile(candidate_counts, 0.5),
                "candidate_p90": _percentile(candidate_counts, 0.9),
                "candidate_max": max(candidate_counts, default=0),
            })

    payload = {
        "scope_id": args.scope,
        "qa_count": len(rows),
        "retriever": "lexical + text_ann + visual_ann + code event-anchor diagnostic",
        "curve": curve,
        "relative_curve": relative_curve,
        "ranking_curves": ranking_curves,
        "dual_channel_curve": dual_channel_curve,
        "questions": per_question,
    }
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"curve": curve, "out": args.out}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
