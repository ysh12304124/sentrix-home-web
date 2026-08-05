#!/usr/bin/env python3
"""Phase 3.5.1 — compare ANN libraries on the 153 vector snapshot.

Run this on 153 with the production database available.  It samples up to N
vectors per space (visual/asset, semantic/observation, episodic/event) from
``memory_vectors`` and benchmarks:

- SQLite full scan (baseline, always available)
- FAISS-CPU HNSW / IVFFlat / Flat (if ``faiss`` importable)
- HNSWlib (if ``hnswlib`` importable)

Each configuration records: build time, query p50/p95, memory footprint,
Recall@10 against the SQLite baseline and support for incremental
upsert / delete.  Output is a JSON decision report the user reviews before
selecting a library.
"""

import argparse
import json
import os
import random
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.db import MemoryStore


def _load_vectors(store, space, limit):
    rows = store.connection.execute(
        "SELECT source_type, source_id, vector_json, scope_id FROM memory_vectors WHERE space = ? LIMIT ?",
        (space, limit),
    ).fetchall()
    vectors, meta = [], []
    for row in rows:
        try:
            values = json.loads(row["vector_json"])
        except (TypeError, ValueError):
            continue
        if not isinstance(values, list) or not values:
            continue
        vectors.append(values)
        meta.append({"source_type": row["source_type"], "source_id": row["source_id"], "scope_id": row["scope_id"]})
    return vectors, meta


def _cosine_top_k(vectors, query, k):
    import math

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    def norm(a):
        return math.sqrt(sum(x * x for x in a)) or 1.0

    q_norm = norm(query)
    scored = []
    for index, vector in enumerate(vectors):
        v_norm = norm(vector)
        score = dot(query, vector) / (q_norm * v_norm)
        scored.append((score, index))
    scored.sort(reverse=True)
    return [index for _, index in scored[:k]]


def _bench_sqlite(vectors, queries, k):
    latencies = []
    baseline_hits = []
    tracemalloc.start()
    for query in queries:
        start = time.perf_counter()
        hits = _cosine_top_k(vectors, query, k)
        latencies.append((time.perf_counter() - start) * 1000)
        baseline_hits.append(hits)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "build_ms": 0,
        "p50_query_ms": statistics.median(latencies),
        "p95_query_ms": statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies),
        "peak_memory_kb": int(peak / 1024),
        "recall_at_k": 1.0,
        "supports_incremental_add": True,
        "supports_delete": True,
        "notes": "SQLite full cosine scan — always the ground truth for Recall@k",
    }, baseline_hits


def _try_import(name):
    try:
        return __import__(name)
    except Exception:
        return None


def _bench_faiss(vectors, queries, k, baseline_hits, index_type="HNSW32"):
    faiss = _try_import("faiss")
    if faiss is None:
        return {"skipped": True, "reason": "faiss module not installed"}
    import numpy as np
    xs = np.array(vectors, dtype="float32")
    faiss.normalize_L2(xs)
    dim = xs.shape[1]
    start = time.perf_counter()
    if index_type.startswith("HNSW"):
        index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efConstruction = 40
    elif index_type == "IVFFlat":
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, min(64, max(4, len(vectors) // 16)))
        index.train(xs)
    else:
        index = faiss.IndexFlatIP(dim)
    index.add(xs)
    build_ms = (time.perf_counter() - start) * 1000
    latencies, matches = [], 0
    tracemalloc.start()
    for query, baseline in zip(queries, baseline_hits):
        q = np.array([query], dtype="float32")
        faiss.normalize_L2(q)
        start = time.perf_counter()
        _, ids = index.search(q, k)
        latencies.append((time.perf_counter() - start) * 1000)
        matches += len(set(ids[0].tolist()) & set(baseline))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "build_ms": build_ms,
        "p50_query_ms": statistics.median(latencies),
        "p95_query_ms": statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies),
        "peak_memory_kb": int(peak / 1024),
        "recall_at_k": matches / (len(queries) * k) if queries else 0.0,
        "supports_incremental_add": True,
        "supports_delete": False,  # FAISS HNSW/IVFFlat require rebuild for delete
        "notes": f"FAISS {index_type}",
    }


def _bench_hnswlib(vectors, queries, k, baseline_hits):
    hnswlib = _try_import("hnswlib")
    if hnswlib is None:
        return {"skipped": True, "reason": "hnswlib module not installed"}
    import numpy as np
    xs = np.array(vectors, dtype="float32")
    xs /= (np.linalg.norm(xs, axis=1, keepdims=True) + 1e-8)
    dim = xs.shape[1]
    start = time.perf_counter()
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=len(vectors) * 2, ef_construction=200, M=16)
    index.add_items(xs, list(range(len(vectors))))
    index.set_ef(50)
    build_ms = (time.perf_counter() - start) * 1000
    latencies, matches = [], 0
    tracemalloc.start()
    for query, baseline in zip(queries, baseline_hits):
        q = np.array([query], dtype="float32")
        q /= (np.linalg.norm(q) + 1e-8)
        start = time.perf_counter()
        ids, _ = index.knn_query(q, k=k)
        latencies.append((time.perf_counter() - start) * 1000)
        matches += len(set(ids[0].tolist()) & set(baseline))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "build_ms": build_ms,
        "p50_query_ms": statistics.median(latencies),
        "p95_query_ms": statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies),
        "peak_memory_kb": int(peak / 1024),
        "recall_at_k": matches / (len(queries) * k) if queries else 0.0,
        "supports_incremental_add": True,
        "supports_delete": True,
        "notes": "hnswlib cosine",
    }


def evaluate_space(store, space, limit, num_queries, k, seed):
    vectors, meta = _load_vectors(store, space, limit)
    if not vectors:
        return {"space": space, "skipped": True, "reason": "no vectors"}
    random.seed(seed)
    query_indices = random.sample(range(len(vectors)), min(num_queries, len(vectors)))
    queries = [vectors[i] for i in query_indices]
    sqlite_stats, baseline_hits = _bench_sqlite(vectors, queries, k)
    return {
        "space": space,
        "sample_size": len(vectors),
        "queries": len(queries),
        "k": k,
        "libraries": {
            "sqlite_full_scan": sqlite_stats,
            "faiss_hnsw32": _bench_faiss(vectors, queries, k, baseline_hits, "HNSW32"),
            "faiss_ivfflat": _bench_faiss(vectors, queries, k, baseline_hits, "IVFFlat"),
            "faiss_flat": _bench_faiss(vectors, queries, k, baseline_hits, "Flat"),
            "hnswlib": _bench_hnswlib(vectors, queries, k, baseline_hits),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db_path")
    parser.add_argument("--report", default=None)
    parser.add_argument("--limit-per-space", type=int, default=500)
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    store = MemoryStore(args.db_path)
    try:
        report = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "db_path": args.db_path,
            "spaces": [evaluate_space(store, space, args.limit_per_space, args.queries, args.k, args.seed)
                        for space in ("visual", "semantic", "episodic")],
        }
    finally:
        store.close()
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        Path(args.report).write_text(payload)
        print(f"wrote {args.report}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
