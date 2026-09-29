#!/usr/bin/env python3
"""Phase 3.5.3 — 100k-vector ANN scale replay (scaffold).

The real backend is chosen after ``evaluate_ann_libraries.py`` runs on 153.
Until then this script exercises the baseline SQLite path so the harness is
runnable and can be pointed at the picked library later.

Targets from the plan:
- p50 <= 2 seconds, p95 <= 5 seconds at 100k vectors
- persistence: index rebuilt from disk after a service restart
"""

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.retrieval_ann import create_index


def _generate_vectors(count, dim, seed):
    random.seed(seed)
    vectors = []
    for index in range(count):
        vector = [random.gauss(0, 1) for _ in range(dim)]
        vectors.append((f"v{index}", vector, {"scope_id": "home", "index": index}))
    return vectors


def _bench(index, queries, k):
    latencies = []
    for query in queries:
        start = time.perf_counter()
        index.search(query, k=k)
        latencies.append((time.perf_counter() - start) * 1000)
    p50 = statistics.median(latencies)
    p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)
    return {"p50_ms": p50, "p95_ms": p95, "n": len(queries)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100_000)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", default="hnswlib",
                         help="AnnIndex backend (default hnswlib per Phase 3.5.1 selection)")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()
    print(f"generating {args.count} synthetic vectors (dim={args.dim})…")
    vectors = _generate_vectors(args.count, args.dim, args.seed)
    print(f"building {args.backend} index…")
    index_kwargs = {}
    if args.backend == "hnswlib":
        # Pre-size to avoid mid-build resizes; matches the plan target of 100k.
        index_kwargs["max_elements"] = max(args.count, 100_000)
    index = create_index(args.backend, **index_kwargs)
    build_start = time.perf_counter()
    index.build(vectors)
    build_ms = (time.perf_counter() - build_start) * 1000
    query_indices = random.sample(range(args.count), args.queries)
    queries = [vectors[i][1] for i in query_indices]
    print("running queries…")
    stats = _bench(index, queries, args.k)
    report = {
        "backend": args.backend, "count": args.count, "dim": args.dim,
        "build_ms": build_ms, **stats,
        "meets_p50_target": stats["p50_ms"] <= 2000,
        "meets_p95_target": stats["p95_ms"] <= 5000,
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        Path(args.report).write_text(payload)
        print(f"wrote {args.report}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
