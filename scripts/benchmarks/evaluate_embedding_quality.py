#!/usr/bin/env python3
"""Embedding capability evaluation — two independent evaluators (Phase R R1B).

P0-2 requires Visual cross-modal and Text retrieval to be judged separately;
a text-only cosine experiment must never be used to claim CLIP image search
works.

  visual_crossmodal(images, queries, embedder):
      query text -> text embed -> brute-force cosine over image embeds
      -> rank of the correct image.  Metrics: Recall@1/5/10, MRR, AUC.
      ``images``  = [{"id": "...", "text": "canonical text for the image"}]
      ``queries`` = [{"id": "...", "query": "user-style query", "target": image_id}]

  text_retrieval(corpus, queries, embedder):
      query -> text embed -> cosine over corpus records
      -> rank of the target record.  Metrics: Recall@1/5/10, MRR, AUC.
      ``corpus`` = [{"id": "...", "text": "caption|activity|place|object|ocr|event summary", "field": "..."}]

CLI entrypoints read JSON files; run on 153 with the real ClipAdapter
(``--embedder clip``).  ``--embedder stub`` is for offline/CI smoke only.

Input data for the *acceptance* run must be an independent Development Set,
NOT the Regression Set queries — see Phase R plan §R1B.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path


def _cosine(a, b):
    if not a or not b:
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def _rank_positions(embeddings, queries):
    """Return per-query (rank_of_target, scores, target_index)."""
    outcomes = []
    for query in queries:
        qvec = embeddings["query"][query["id"]]
        target = query.get("target")
        items = [(item_id, _cosine(qvec, vec)) for item_id, vec in embeddings["candidate"].items()]
        items.sort(key=lambda pair: pair[1], reverse=True)
        target_index = next((i for i, (item_id, _) in enumerate(items) if item_id == target), None)
        outcomes.append({"query": query["id"], "rank": (target_index + 1) if target_index is not None else None,
                         "target": target, "scores": items})
    return outcomes


def _summarize(outcomes):
    n = len(outcomes)
    hits = [out for out in outcomes if out["rank"] is not None]
    recall = {}
    for k in (1, 5, 10):
        recall[f"recall@{k}"] = round(sum(1 for out in hits if out["rank"] <= k) / n, 4) if n else None
    mrr = round(sum(1.0 / out["rank"] for out in hits if out["rank"]) / n, 4) if n else None
    auc = _separation_auc(outcomes)
    return {"count": n, **recall, "mrr": mrr, "auc": round(auc, 4) if auc is not None else None}


def _separation_auc(outcomes):
    """AUC over all (target-score, non-target-score) pairs.

    Computed from each query's ranked score list: the fraction of pairs where
    the target score beats a non-target score.
    """
    positive = []
    negative = []
    for out in outcomes:
        if out["rank"] is None:
            continue
        target_score = None
        for item_id, score in out["scores"]:
            if item_id == out["target"]:
                target_score = score
                break
        if target_score is None:
            continue
        for item_id, score in out["scores"]:
            if item_id != out["target"]:
                negative.append((target_score, score))
    if not negative:
        return None
    wins = sum(1 for t, s in negative if t > s)
    ties = sum(1 for t, s in negative if t == s)
    return (wins + 0.5 * ties) / len(negative)


def _embed_text(embedder, text):
    """VisualQueryEmbedder (embed_query) or ClipAdapter (embed_text) both work."""
    if hasattr(embedder, "embed_query"):
        return embedder.embed_query(text) or []
    if hasattr(embedder, "embed_text"):
        return embedder.embed_text(text) or []
    return []


def _embed_image(embedder, source):
    if hasattr(embedder, "embed_image"):
        return embedder.embed_image(source) or []
    return []


def visual_crossmodal(images, queries, embedder=None):
    """Query text -> text embed; candidate = image embedding.

    ``images`` entries carry ``path`` (real file, used by ClipAdapter) or
    ``text`` (canonical text, used by stubs) for the candidate encoding.
    """
    if embedder is None:
        embedder = _make_embedder("clip")
    image_embeds = {}
    for image in images:
        source = image.get("path") or image.get("text") or ""
        vec = _embed_image(embedder, source)
        if vec:
            image_embeds[image["id"]] = vec
    query_embeds = {}
    for query in queries:
        vec = _embed_text(embedder, query.get("query") or "")
        if vec:
            query_embeds[query["id"]] = vec
    embeddings = {"query": query_embeds, "candidate": image_embeds}
    return _summarize(_rank_positions(embeddings, queries))


def text_retrieval(corpus, queries, embedder=None):
    """Query text -> text embed; candidate = corpus record text embed."""
    if embedder is None:
        embedder = _make_embedder("clip")
    record_embeds = {}
    for record in corpus:
        vec = _embed_text(embedder, record.get("text") or "")
        if vec:
            record_embeds[record["id"]] = vec
    query_embeds = {}
    for query in queries:
        vec = _embed_text(embedder, query.get("query") or "")
        if vec:
            query_embeds[query["id"]] = vec
    embeddings = {"query": query_embeds, "candidate": record_embeds}
    return _summarize(_rank_positions(embeddings, queries))


class _StubEmbedder:
    dim = 64

    def embed_text(self, text):
        return self._embed(str(text or ""))

    def embed_image(self, text):
        return self._embed(str(text or ""))

    def _embed(self, text):
        vector = [0.0] * self.dim
        padded = f" {text} "
        for index in range(len(padded) - 1):
            bucket = (sum(ord(c) for c in padded[index:index + 2]) * 2654435761) % self.dim
            vector[bucket] += 1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]


def _make_embedder(kind):
    if kind == "stub":
        return _StubEmbedder()
    if kind == "clip":
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from backend.model_clients import ClipAdapter
        return ClipAdapter()
    raise ValueError(f"unknown embedder: {kind}")


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-json", default=None, help='[{"id","path"|"text"}] candidate images')
    parser.add_argument("--corpus-json", default=None, help='[{"id","text","field"}] text retrieval corpus')
    parser.add_argument("--queries-json", default=None, help="Development labels used for BOTH when the two are not split")
    parser.add_argument("--text-queries-json", default=None, help="Development labels for text retrieval")
    parser.add_argument("--visual-queries-json", default=None, help="Development labels for visual cross-modal")
    parser.add_argument("--embedder", choices=["clip", "stub"], default="clip")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    embedder = _make_embedder(args.embedder)
    report = {"embedder": args.embedder}
    text_queries = args.text_queries_json or args.queries_json
    visual_queries = args.visual_queries_json or args.queries_json
    if args.corpus_json and text_queries:
        corpus = _load_json(args.corpus_json)
        queries = _load_json(text_queries)
        report["text_retrieval"] = text_retrieval(corpus, queries, embedder=embedder)
    if args.images_json and visual_queries:
        images = _load_json(args.images_json)
        queries = _load_json(visual_queries)
        report["visual_crossmodal"] = visual_crossmodal(images, queries, embedder=embedder)
    if not report.get("text_retrieval") and not report.get("visual_crossmodal"):
        parser.error("need --corpus-json and/or --images-json with matching query labels")
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
        print(f"wrote {args.report}")
        print(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
