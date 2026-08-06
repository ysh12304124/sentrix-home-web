#!/usr/bin/env python3
"""Retrieval-only benchmark runner (Phase R R1A + channel ablation).

Runs the EvidenceRetrievalKernel over a ground-truth set WITHOUT calling the
answer model, so the full set can finish in seconds instead of the 240s/round
end-to-end latency.  Two spec sources:

  --spec-source cached        deterministic extraction (no model at all)
  --spec-source parser        real QueryParser (needs GammaClient / Ollama)

Channel ablation is requested via --channels; before Phase R2 every channel
mode collapses to the current single Kernel and the report records that in
``channel_note``.  From R2 onward the flag selects real retriever subsets and
the same report shape keeps working, so Regression Set numbers are comparable.

Metrics per case: Recall@1/5/10/20, MRR, Precision@5, all_relevant recall,
empty-GT false positives, hard-constraint violations, GT rank distribution.
Also records per-channel contribution once retrievers exist.

This is a benchmark tool: it reads benchmark data, which is forbidden in
runtime code (backend/*.py, configs/) but allowed here.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path


ALBUMS = ("album1", "album2", "album3")
DEFAULT_SAMPLES = os.getenv("SENTRIX_BENCHMARK_SAMPLES", str(Path.home() / "Downloads" / "samples"))
_TIME_RE = re.compile(r"20\d{2}\s*(?:年|[-/.])\s*\d{1,2}\s*(?:月|[-/.])?(?:\s*\d{1,2}\s*日?)?")


def _load_cases(samples_root):
    cases = []
    for album in ALBUMS:
        path = Path(samples_root) / album / "query.json"
        if not path.is_file():
            print(f"[eval] missing {path}", file=sys.stderr)
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for index, case in enumerate(data, 1):
            case["album"] = album
            case["key"] = f"{album}-{index:02d}"
            cases.append(case)
    return cases


def _asset_ids_by_filename(store, scope_id):
    return {asset.get("file_name"): asset.get("id")
            for asset in store.list_assets(scope_id=scope_id, limit=10_000)}


def _build_deterministic_spec(store, case):
    """Rough cached QuerySpec without any model call.

    Structured extractions mirror the parser's deterministic overlay; the whole
    query is kept as a single visual/semantic condition.  This is intentionally
    *not* production-equivalent — it exists so the 60-case runner has a
    no-model path.  The production-faithful path is ``--spec-source parser``.
    """
    from backend.query_contracts import QueryParseDraft, build_query_spec
    query = case.get("query_cn") or ""
    draft = QueryParseDraft(intent="answer", answer_target="general")
    match = _TIME_RE.search(query)
    if match:
        draft.time_expression = match.group(0)
    if any(token in query for token in ("照片", "图片", "原图", "视频")):
        draft.media_expressions.append("照片" if "视频" not in query else "视频")
    for token in ("不要", "排除", "不是"):
        idx = query.find(token)
        if idx >= 0 and "视频" in query[idx:idx + 20]:
            draft.negative_conditions.append({"dimension": "media", "value": "video", "source_text": query[idx:idx + 20]})
    draft.semantic_conditions.append({"dimension": "semantic", "value": query,
                                      "source_text": query, "strictness": "semantic_required"})
    return build_query_spec(
        draft, scope_id=case["album"], viewer_id="owner",
        conversation_id=f"bench_{case['key']}", query_id=f"bench_{case['key']}",
    )


def _build_parser_spec(store, case, parser):
    draft = parser.parse(case.get("query_cn") or "", recent_turns="")
    from backend.query_contracts import build_query_spec
    return build_query_spec(
        draft, scope_id=case["album"], viewer_id="owner",
        conversation_id=f"bench_{case['key']}", query_id=f"bench_{case['key']}",
    )


def _ranked_asset_ids(packet):
    return [item["asset_id"] for item in packet.assets]


def _metrics(case, ranked, truth_ids, packet):
    truth = set(truth_ids)
    returned = set(ranked)
    hits_at = {}
    for k in (1, 5, 10, 20):
        prefix = set(ranked[:k])
        hits_at[k] = len(truth & prefix)
    recall = {k: (hits_at[k] / len(truth)) if truth else None for k in hits_at}
    mrr = 0.0
    for index, asset_id in enumerate(ranked):
        if asset_id in truth:
            mrr = 1.0 / (index + 1)
            break
    p5 = len(truth & set(ranked[:5])) / 5.0
    all_relevant = bool(truth) and truth.issubset(set(ranked[:20]))
    empty_fp = (not truth) and bool(returned)
    hard_violation = _count_hard_violations(packet)
    return {
        "key": case["key"], "query": case.get("query_cn"), "gt_count": len(truth),
        "returned_count": len(ranked), "hits_at": hits_at,
        "recall_at": recall, "mrr": round(mrr, 4), "precision_at_5": round(p5, 4),
        "all_relevant": all_relevant, "empty_gt_fp": empty_fp,
        "hard_violation": hard_violation, "ranked_ids": ranked,
    }


def _count_hard_violations(packet):
    """A packet that satisfies negated constraints is a hard violation.

    The kernel already excludes hard violations; this metric independently
    re-checks the emitted packet so a kernel regression cannot silently ship
    ``must_not`` content in a user-visible result.
    """
    violations = 0
    for item in packet.assets:
        condition_results = item.get("condition_results", {})
        for key, condition in condition_results.items():
            if condition.get("status") == "matched" and condition.get("negated"):
                violations += 1
    return violations


def _grade(case, ranked, truth_ids, packet):
    return _metrics(case, ranked, truth_ids, packet)


CHANNEL_MAP = {
    "lexical": {"lexical"},
    "visual": {"visual_ann"},
    "text": {"text_ann"},
    "structured": {"metadata", "entity"},
    "hybrid_no_adjacency": {"metadata", "entity", "lexical", "visual_ann", "text_ann"},
    "full_hybrid": {"metadata", "entity", "lexical", "visual_ann", "text_ann", "adjacency"},
}


def _make_embedding_router():
    try:
        from backend.embeddings import EmbeddingRouter
        from backend.model_clients import ClipAdapter
        return EmbeddingRouter.from_clip(ClipAdapter())
    except Exception:
        return None


def _build_kernel(store, channels, embedding_router):
    import os
    os.environ.setdefault("SENTRIX_EVIDENCE_MULTI_RETRIEVER_V1", "1")
    from backend.evidence_retrieval import EvidenceRetrievalKernel
    from backend.retrieval import RetrievalConfig, build_default_retrievers
    config = RetrievalConfig()
    if not config.multi_retriever:
        return EvidenceRetrievalKernel(store)
    subset = CHANNEL_MAP.get(channels or "full_hybrid", CHANNEL_MAP["full_hybrid"])
    retrievers = [r for r in build_default_retrievers(store, embedding_router=embedding_router, config=config)
                  if r.name in subset]
    return EvidenceRetrievalKernel(store, retrievers=retrievers,
                                   embedding_router=embedding_router, config=config)


def run(store, cases, spec_source, parser=None, top_k=20, include=None, limit=None,
        exclude_hidden=None, channels="full_hybrid"):
    embedding_router = _make_embedding_router()
    kernel = _build_kernel(store, channels, embedding_router)
    include_set = set(item.strip() for item in include.split(",")) if include else None
    exclude_set = set()
    if exclude_hidden:
        manifest = json.loads(Path(exclude_hidden).read_text(encoding="utf-8"))
        exclude_set = {entry["key"] for entry in manifest.get("hidden_keys", [])}
    results = []
    for case in cases:
        if limit is not None and len(results) >= limit:
            break
        if include_set and case["key"] not in include_set:
            continue
        if case["key"] in exclude_set:
            continue
        filename_to_id = _asset_ids_by_filename(store, case["album"])
        truth_ids = [filename_to_id.get(name) for name in (case.get("ground_truth") or [])]
        truth_ids = [item for item in truth_ids if item]
        spec = _build_parser_spec(store, case, parser) if spec_source == "parser" else _build_deterministic_spec(store, case)
        start = time.perf_counter()
        packet = kernel.retrieve(spec)
        elapsed = time.perf_counter() - start
        ranked = _ranked_asset_ids(packet)
        grade = _grade(case, ranked, truth_ids, packet)
        grade["latency_s"] = round(elapsed, 4)
        results.append(grade)
    return results


def run_dev(store, dev_cases, *, channels="full_hybrid", top_k=20):
    """Grade the independent Development Set (Phase R8-2).

    Dev cases carry ``query_cn``, ``exact`` (asset ids already resolved) and
    ``empty_policy``; scope is derived from the first exact asset so no
    benchmark-file resolution is involved.
    """
    embedding_router = _make_embedding_router()
    kernel = _build_kernel(store, channels, embedding_router)
    results = []
    for index, case in enumerate(dev_cases):
        query = case.get("query_cn") or ""
        truth_ids = [str(item) for item in (case.get("exact") or [])]
        acceptable = {str(item) for item in (case.get("acceptable_approximate") or [])}
        scope_id = case.get("scope_id")
        if not scope_id and truth_ids:
            asset = store.get_asset(truth_ids[0])
            if asset:
                scope_id = asset.get("scope_id")
        from backend.query_contracts import QueryParseDraft, build_query_spec
        draft = QueryParseDraft(intent="answer", answer_target="general")
        draft.semantic_conditions.append({"dimension": "semantic", "value": query, "source_text": query})
        spec = build_query_spec(draft, scope_id=scope_id or "home-default", viewer_id="owner",
                                conversation_id=f"dev_{index}", query_id=f"dev_{index}")
        start = time.perf_counter()
        packet = kernel.retrieve(spec)
        ranked = _ranked_asset_ids(packet)
        grade = _grade(_dev_case_shell(case), ranked, truth_ids, packet)
        grade["latency_s"] = round(time.perf_counter() - start, 4)
        grade["empty_policy"] = case.get("empty_policy", "allow_approximate")
        grade["category"] = case.get("category")
        grade["acceptable_hits"] = sorted(acceptable & set(ranked))
        # strict-empty: any returned asset that is NOT an acceptable approximate is an FP.
        if case.get("empty_policy") == "strict_empty" and not truth_ids:
            grade["strict_empty_fp"] = len([aid for aid in ranked if aid not in acceptable])
        results.append(grade)
    return results


def _dev_case_shell(case):
    return {"key": case.get("key", "dev"), "query_cn": case.get("query_cn") or ""}


def _aggregate(results):
    if not results:
        return {"total": 0}
    n = len(results)
    def _avg(key):
        vals = [item[key] for item in results if item[key] is not None]
        return round(sum(vals) / len(vals), 4) if vals else None
    return {
        "total": n,
        "recall_at_1": _avg_of_key(results, "recall_at", 1),
        "recall_at_5": _avg_of_key(results, "recall_at", 5),
        "recall_at_10": _avg_of_key(results, "recall_at", 10),
        "recall_at_20": _avg_of_key(results, "recall_at", 20),
        "mrr": _avg("mrr"),
        "precision_at_5": _avg("precision_at_5"),
        "all_relevant_count": sum(1 for item in results if item["all_relevant"]),
        "empty_gt_fp_count": sum(1 for item in results if item["empty_gt_fp"]),
        "strict_empty_fp_count": sum(item.get("strict_empty_fp", 0) for item in results),
        "hard_violation_count": sum(item["hard_violation"] for item in results),
        "avg_latency_s": round(sum(item["latency_s"] for item in results) / n, 4),
    }


def _avg_of_key(results, key, subkey):
    vals = []
    for item in results:
        value = item.get(key, {}).get(subkey)
        if value is not None:
            vals.append(value)
    return round(sum(vals) / len(vals), 4) if vals else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=os.getenv("SENTRIX_DB_PATH", "data/sentrix.db"))
    parser.add_argument("--samples-root", default=DEFAULT_SAMPLES)
    parser.add_argument("--spec-source", choices=["cached", "parser"], default="cached")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include", default=None, help="comma-separated keys e.g. album1-01,album3-14")
    parser.add_argument("--channels", default="full_hybrid",
                        help="lexical|visual|text|structured|hybrid_no_adjacency|full_hybrid (pre-R2 all map to current kernel)")
    parser.add_argument("--exclude-hidden", default=None, help="path to hidden_set_manifest.json to exclude hidden keys from grading")
    parser.add_argument("--dev-set", default=None, help="path to development_set.json to grade the independent Dev Set")
    parser.add_argument("--report", default=None, help="write JSON report to this path")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from backend.db import MemoryStore

    store = MemoryStore(args.db)
    if args.dev_set:
        dev_cases = json.loads(Path(args.dev_set).read_text(encoding="utf-8"))["cases"]
        print(f"[eval] Dev Set: {len(dev_cases)} cases, channels={args.channels}")
        results = run_dev(store, dev_cases, channels=args.channels, top_k=args.top_k)
        store.close()
        summary = _aggregate(results)
        payload = {"summary": summary, "dataset": "development", "channels": args.channels, "cases": results}
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        if args.report:
            Path(args.report).write_text(text, encoding="utf-8")
            print(f"wrote {args.report}")
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            print(text)
        return

    cases = _load_cases(args.samples_root)
    print(f"[eval] {len(cases)} cases, spec-source={args.spec_source}, channels={args.channels}")
    parser_obj = None
    if args.spec_source == "parser":
        from backend.model_clients import GammaClient
        from backend.query_parser import QueryParser
        parser_obj = QueryParser(gamma=GammaClient())
    results = run(store, cases, args.spec_source, parser=parser_obj, top_k=args.top_k,
                  include=args.include, limit=args.limit, exclude_hidden=args.exclude_hidden,
                  channels=args.channels)
    store.close()
    summary = _aggregate(results)
    payload = {"summary": summary,
               "channel_note": ("R2+: channels param selects real retriever subsets; "
                                "pre-R2 all modes collapsed to the single Kernel."),
               "channels": args.channels,
               "exclude_hidden": args.exclude_hidden or None,
               "cases": results}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
        print(f"wrote {args.report}")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(text)


if __name__ == "__main__":
    main()
