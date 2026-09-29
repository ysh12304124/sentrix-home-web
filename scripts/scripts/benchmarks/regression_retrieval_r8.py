#!/usr/bin/env python3
"""R8 retrieval regression re-run (Phase B B1).

Replays the recorded R8 regression set (queries + GT asset ids embedded in
gt_channel_ranks) against the live EvidenceRetrievalKernel and compares
Recall@1/5/10/20 / MRR / P@5 with the stored baseline summary.

Usage:
  python regression_retrieval_r8.py \
      --regression docs/baseline/retrieval_R8_regression_visual_backbone.json \
      --out /tmp/r8_regression_rerun.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))


_TIME_RE = __import__("re").compile(r"20\d{2}\s*(?:年|[-/.])\s*\d{1,2}\s*(?:月|[-/.])?(?:\s*\d{1,2}\s*日?)?")


def build_kernel(db_path):
    import os
    os.environ.setdefault("SENTRIX_EVIDENCE_MULTI_RETRIEVER_V1", "1")
    os.environ.setdefault("SENTRIX_RETRIEVER_RANKING", "visual_backbone")
    os.environ.setdefault("SENTRIX_IMAGE_EMBEDDER", "chinese_clip")
    os.environ.setdefault("CLIP_DEVICE", "cuda")
    from backend.db import MemoryStore
    from backend.evidence_retrieval import EvidenceRetrievalKernel
    from backend.retrieval import RetrievalConfig, build_default_retrievers
    try:
        from backend.embeddings import EmbeddingRouter
        from backend.model_clients import ClipAdapter
        router = EmbeddingRouter.from_clip(ClipAdapter())
    except Exception:
        router = None
    store = MemoryStore(db_path)
    cfg = RetrievalConfig()
    retrievers = build_default_retrievers(store, embedding_router=router, config=cfg)
    return store, EvidenceRetrievalKernel(store, retrievers=retrievers,
                                          embedding_router=router, config=cfg)


def spec_for(store, query, scope_id):
    from backend.query_contracts import QueryParseDraft, build_query_spec
    draft = QueryParseDraft(intent="answer", answer_target="general")
    match = _TIME_RE.search(query)
    if match:
        draft.time_expression = match.group(0)
    if any(token in query for token in ("照片", "图片", "原图", "视频")):
        draft.media_expressions.append("照片" if "视频" not in query else "视频")
    draft.semantic_conditions.append({"dimension": "semantic", "value": query,
                                      "source_text": query, "strictness": "semantic_required"})
    return build_query_spec(draft, scope_id=scope_id, viewer_id="owner",
                            conversation_id="regression", query_id="r8")


def scope_for(key):
    prefix = key.split("-", 1)[0]
    return prefix  # album1/album2/album3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regression", default="docs/baseline/retrieval_R8_regression_visual_backbone.json")
    ap.add_argument("--db", default="data/sentrix.db")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rec = json.load(open(args.regression, encoding="utf-8"))
    store, kernel = build_kernel(args.db)
    rows = []
    totals = {"total": 0, "r1": 0.0, "r5": 0.0, "r10": 0.0, "r20": 0.0,
              "mrr": 0.0, "p5": 0.0, "all_relevant": 0, "empty_gt_fp": 0,
              "hard_violation": 0, "lat": 0.0}
    for case in rec["cases"]:
        gt_ids = list((case.get("gt_channel_ranks") or {}).keys())
        if not gt_ids:
            continue
        query = case["query"]
        scope = scope_for(case["key"])
        t0 = time.time()
        packet = kernel.retrieve(spec_for(store, query, scope))
        dt = time.time() - t0
        ranked = [a.get("asset_id") for a in (packet.assets or [])]
        hits = {k: 0 for k in (1, 5, 10, 20)}
        for k in hits:
            hits[k] = sum(1 for g in gt_ids if g in ranked[:k])
        rr = 0.0
        for i, aid in enumerate(ranked, 1):
            if aid in gt_ids:
                rr = 1.0 / i
                break
        p5 = sum(1 for aid in ranked[:5] if aid in gt_ids) / 5.0
        all_rel = all(g in ranked[:20] for g in gt_ids)
        empty_gt_fp = (not gt_ids) and bool(ranked)
        rows.append({
            "key": case["key"], "query": query, "gt_count": len(gt_ids),
            "recall_at": {"1": hits[1], "5": hits[5], "10": hits[10], "20": hits[20]},
            "recall_rate_at": {str(k): (hits[k] / len(gt_ids)) for k in (1, 5, 10, 20)},
            "mrr": round(rr, 4), "precision_at_5": round(p5, 4),
            "all_relevant": all_rel, "empty_gt_fp": empty_gt_fp,
            "latency_s": round(dt, 3), "ranked_ids": ranked[:20],
        })
        n = len(gt_ids)
        totals["total"] += 1
        totals["r1"] += hits[1] / n
        totals["r5"] += hits[5] / n
        totals["r10"] += hits[10] / n
        totals["r20"] += hits[20] / n
        totals["mrr"] += rr
        totals["p5"] += p5
        totals["all_relevant"] += 1 if all_rel else 0
        totals["empty_gt_fp"] += 1 if empty_gt_fp else 0
        totals["hard_violation"] += len(packet.gaps or [])
        totals["lat"] += dt
    t = max(1, totals["total"])
    summary = {
        "total": totals["total"],
        "recall_at_1": round(totals["r1"] / t, 4),
        "recall_at_5": round(totals["r5"] / t, 4),
        "recall_at_10": round(totals["r10"] / t, 4),
        "recall_at_20": round(totals["r20"] / t, 4),
        "mrr": round(totals["mrr"] / t, 4),
        "precision_at_5": round(totals["p5"] / t, 4),
        "all_relevant_count": totals["all_relevant"],
        "empty_gt_fp_count": totals["empty_gt_fp"],
        "hard_violation_count": totals["hard_violation"],
        "avg_latency_s": round(totals["lat"] / t, 4),
    }
    recorded = rec.get("summary") or {}
    out = {"rerun": summary, "recorded": recorded,
           "delta": {k: (round(summary[k] - (recorded.get(k) or 0), 4) if k in summary else None)
                     for k in ("recall_at_1", "recall_at_5", "recall_at_10", "recall_at_20",
                               "mrr", "precision_at_5", "all_relevant_count")},
           "cases": rows}
    print(json.dumps({"rerun": summary, "recorded": recorded, "delta": out["delta"]}, ensure_ascii=False, indent=1))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
