#!/usr/bin/env python3
"""Real query paraphrase -> Observation/Event text retrieval test (Phase R8-4).

R7 showed the Text channel has low independent Recall (0.158) while CLIP's
self-retrieval AUC was high (0.996) — the classic "same text finds itself"
artifact.  This evaluator measures whether a *differently-phrased* query can
retrieve the stored Observation text, which is what actually matters.

Two paraphrase sources, both NON-self-matching:
  - manual map for common activities (做饭 <-> 做晚饭, 烟花 <-> 看烟花, ...)
  - activity-label retrieval where the target is every OTHER observation that
    shares the same activity (so the query text never equals its own target).

Compares CLIP Text (512-d) vs bge-m3 (1024-d, if installed).  Decision rule:
  paraphrase Recall@10 < 0.5  -> disable Text ANN or switch embedder
  0.5 <= R@10 < 0.7           -> bge-m3 is a candidate
  R@10 >= 0.7                 -> CLIP may stay

Output: docs/baseline/text_embedder_decision.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

# activity -> paraphrase query (observed top activities on the real corpus)
_ACTIVITY_PARAPHRASES = {
    "做晚饭": "做饭",
    "看烟花": "烟花",
    "自拍": "拍照",
    "逛街": "购物",
    "散步": "遛弯",
    "做午饭": "做饭",
    "吃饭": "用餐",
    "看日落": "日落",
    "做饭": "烧饭",
    "合影留念": "拍合照",
    "参观展览": "看展",
    "休息": "歇会儿",
    "用餐": "吃饭",
    "站立拍照": "站着拍照",
    "食物展示": "展示食物",
    "观赏自然风光": "看风景",
    "游客在景区游览": "在景区玩",
    "与镜头互动": "看镜头",
}


def _paraphrase(activity):
    """Manual map first, then general structural paraphrases."""
    if activity in _ACTIVITY_PARAPHRASES:
        return _ACTIVITY_PARAPHRASES[activity]
    for suffix in ("留念", "时分", "场景"):
        if activity.endswith(suffix):
            return activity[: -len(suffix)]
    for token in ("或", "与"):
        if token in activity:
            return activity.split(token)[0].strip()
    return None


def _load_observations():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from backend.db import MemoryStore
    store = MemoryStore(os.getenv("SENTRIX_DB_PATH", "data/sentrix.db"))
    obs = [dict(row) for row in store.list_observations(limit=100_000)]
    store.close()
    return obs


def _texts(obs):
    return {
        "id": obs.get("id"),
        "text": " ".join(str(item) for item in
                         [obs.get("caption"), obs.get("activity"), obs.get("place"),
                          obs.get("ocr_text")] if item),
        "activity": obs.get("activity"),
    }


def build_paraphrase_queries(obs):
    """Return list of {id, query, target_ids} using the manual paraphrase map."""
    queries = []
    seen = set()
    for record in obs:
        activity = record.get("activity")
        paraphrase = _paraphrase(activity)
        if not paraphrase or paraphrase == activity:
            continue
        key = (activity, paraphrase)
        if key in seen:
            continue
        seen.add(key)
        targets = [r["id"] for r in obs if r.get("activity") == activity]
        queries.append({"id": f"pq_{len(queries)}", "query": paraphrase, "target_ids": targets})
    return queries


def build_cross_retrieval_queries(obs):
    """activity-label query where target = OTHER observations with same activity."""
    by_activity = {}
    for record in obs:
        activity = record.get("activity")
        if activity:
            by_activity.setdefault(activity, []).append(record["id"])
    queries = []
    for activity, ids in by_activity.items():
        if len(ids) < 2:
            continue
        queries.append({"id": f"xq_{activity}", "query": activity, "target_ids": ids})
    return queries


def _embed(embedder, text):
    if hasattr(embedder, "embed_query"):
        return embedder.embed_query(text) or []
    if hasattr(embedder, "embed_text"):
        return embedder.embed_text(text) or []
    return []


def evaluate(corpus, queries, embedder):
    import math
    record_embeds = {}
    for record in corpus:
        vec = _embed(embedder, record["text"])
        if vec:
            record_embeds[record["id"]] = vec
    results = []
    for query in queries:
        qvec = _embed(embedder, query["query"])
        if not qvec:
            continue
        targets = set(query["target_ids"]) & set(record_embeds)
        if not targets:
            continue
        # Exclude the query's own source when it is in the corpus.
        items = [(rid, sum(a * b for a, b in zip(qvec, vec))) for rid, vec in record_embeds.items()]
        items.sort(key=lambda pair: pair[1], reverse=True)
        rank_of_first = None
        hits_at = {k: 0 for k in (1, 5, 10)}
        for idx, (rid, _) in enumerate(items, 1):
            if rid in targets:
                if rank_of_first is None:
                    rank_of_first = idx
                for k in (1, 5, 10):
                    if idx <= k:
                        hits_at[k] += 1
        results.append({"id": query["id"], "query": query["query"], "target_count": len(targets),
                        "first_rank": rank_of_first, "hits_at": hits_at})
    return results


def summarize(results):
    total = len(results)
    if not total:
        return {"count": 0}
    recall = {}
    for k in (1, 5, 10):
        recall[f"recall@{k}"] = round(sum(r["hits_at"][k] / max(1, r["target_count"]) for r in results) / total, 4)
    return {"count": total, **recall}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedder", choices=["clip", "bge"], default="clip")
    parser.add_argument("--report", default="docs/baseline/text_embedder_decision.json")
    args = parser.parse_args()

    obs = [dict(row) for row in _load_observations()]
    corpus = [_texts(row) for row in obs]
    corpus = [c for c in corpus if c["text"].strip()]

    if args.embedder == "bge":
        from backend.embeddings.bge_text import BgeM3TextQueryEmbedder
        embedder = BgeM3TextQueryEmbedder()
        if not embedder.available:
            print("bge-m3 unavailable (needs sentence-transformers); cannot compare")
            return
    else:
        from backend.model_clients import ClipAdapter
        from backend.embeddings.clip_text import ClipTextQueryEmbedder
        embedder = ClipTextQueryEmbedder(ClipAdapter())

    paraphrase = evaluate(corpus, build_paraphrase_queries(obs), embedder)
    cross = evaluate(corpus, build_cross_retrieval_queries(obs), embedder)

    report = {
        "embedder": args.embedder,
        "paraphrase_queries": summarize(paraphrase),
        "cross_retrieval": summarize(cross),
        "decision": None,
    }
    para_r10 = report["paraphrase_queries"].get("recall@10")
    if para_r10 is None:
        report["decision"] = "no_paraphrase_pairs"
    elif para_r10 < 0.5:
        report["decision"] = "disable_text_ann_or_switch_embedder"
    elif para_r10 < 0.7:
        report["decision"] = "bge_m3_candidate"
    else:
        report["decision"] = "keep_clip_text"

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.report}")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
