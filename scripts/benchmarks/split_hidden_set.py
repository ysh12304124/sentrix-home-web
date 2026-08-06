#!/usr/bin/env python3
"""Partition the 60-case set into Regression + Development + Hidden (Phase R R1A).

Hidden Acceptance Set is drawn from the existing 60 cases per user decision
D2 (15-20 cases).  The manifest records case KEYS and query-type categories
only — never the Ground-Truth file names or asset IDs — so the implementing
agent cannot memorise the hidden answers.  The final R7 acceptance run must
grade the hidden keys with a separate script that holds the real query.json.

Stratified sampling covers: time / location / person / object / clothing /
composite / empty-GT categories with at least one hidden case each.

Benchmark tool — this legitimately reads benchmark data.
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from evaluate_retrieval_kernel import DEFAULT_SAMPLES, _load_cases


def _category(case):
    query = case.get("query_cn") or ""
    labels = {key: case.get(key) for key in ("Location", "Time", "Person", "Object", "Concept", "Genre")}
    if not (case.get("ground_truth") or []):
        return "empty_gt"
    if labels.get("Person") or any(token in query for token in ("明哥", "王明", "八戒", "小黑", "自己", "合照")):
        return "person"
    if labels.get("Location") or any(token in query for token in ("市", "区", "省", "镇", "湾", "湖", "城")):
        return "location"
    if labels.get("Time") or any(token in query for token in ("年", "月", "日", "节", "跨年", "元旦")):
        return "time"
    if labels.get("Object") in {"fact", "cognitive"} or any(token in query for token in ("手镯", "戒指", "可颂", "项链")):
        return "object"
    if any(token in query for token in ("衣", "睡衣", "裤", "裙", "鞋", "色", "花")):
        return "clothing"
    if len(query) >= 8 and sum(1 for key, val in labels.items() if val) >= 2:
        return "composite"
    return "other"


def partition(cases, hidden_count, seed):
    rng = random.Random(seed)
    buckets = {}
    for case in cases:
        buckets.setdefault(_category(case), []).append(case)
    hidden_keys = []
    # Guarantee at least one per category, then top up randomly.
    for category, members in sorted(buckets.items()):
        if len(hidden_keys) >= hidden_count:
            break
        chosen = rng.choice(members)
        hidden_keys.append(chosen["key"])
    remaining = [case for case in cases if case["key"] not in hidden_keys]
    rng.shuffle(remaining)
    for case in remaining:
        if len(hidden_keys) >= hidden_count:
            break
        hidden_keys.append(case["key"])
    hidden_keys = sorted(hidden_keys)
    regression_keys = sorted(case["key"] for case in cases if case["key"] not in hidden_keys)
    return hidden_keys, regression_keys, buckets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-root", default=DEFAULT_SAMPLES)
    parser.add_argument("--hidden-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--manifest", default="docs/baseline/hidden_set_manifest.json")
    args = parser.parse_args()

    cases = _load_cases(args.samples_root)
    hidden_keys, regression_keys, buckets = partition(cases, args.hidden_count, args.seed)

    # Re-derive the manifest entries with category only (no GT, no file names).
    by_key = {case["key"]: case for case in cases}
    hidden_entries = [
        {"key": key, "category": _category(by_key[key]), "query_cn": by_key[key].get("query_cn")}
        for key in hidden_keys
    ]
    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "hidden_count": len(hidden_keys),
        "regression_count": len(regression_keys),
        "category_counts": {cat: len(members) for cat, members in sorted(buckets.items())},
        "hidden_keys": hidden_entries,
        "regression_keys": regression_keys,
        "note": "GT file names / asset IDs intentionally absent — grade hidden keys via a separate script holding query.json.",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.manifest}")
    print(json.dumps({k: manifest[k] for k in ("seed", "hidden_count", "regression_count", "category_counts")}, ensure_ascii=False, indent=2))
    print("hidden_keys:", ", ".join(hidden_keys))
    print("regression_keys:", ", ".join(regression_keys))


if __name__ == "__main__":
    main()
