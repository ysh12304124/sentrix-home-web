#!/usr/bin/env python3
"""Read-only autopsy of face clusters in the production SQLite DB.

Excludes evaluation / benchmark spaces (PhotoBench-*, photobench-*, *_e2b) whose
repeated imports pollute statistics, and keeps only ONE observation per asset so
duplicate re-imports of the same photo cannot inflate cluster sizes.

Answers:
  1. Same-asset multi-face clusters in real user data (cannot-link violations).
  2. Does within-cluster similarity drop as cluster size grows?
  3. Where does match_threshold 0.30 sit on the real cosine distributions?

Read-only: opens the DB with mode=ro and never writes.
"""

import argparse
import json
import re
import sqlite3
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path

EXCLUDE_NAME = re.compile(r"photobench", re.IGNORECASE)
EXCLUDE_ID = re.compile(r"_e2b$")


def _cosine(left, right):
    n = min(len(left), len(right))
    if not n:
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = sum(float(a) * float(a) for a in left) ** 0.5
    right_norm = sum(float(a) * float(a) for a in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _summary(values):
    if not values:
        return None
    ordered = sorted(values)
    return {
        "min": round(ordered[0], 4),
        "p10": round(ordered[max(0, int(len(ordered) * 0.10) - 1)], 4),
        "median": round(statistics.median(ordered), 4),
        "mean": round(statistics.mean(ordered), 4),
        "p90": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.90))], 4),
        "max": round(ordered[-1], 4),
    }


def analyze(db_path):
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        spaces = connection.execute("SELECT id, name FROM memory_spaces").fetchall()
        keep = {
            s["id"] for s in spaces
            if not EXCLUDE_NAME.search(s["name"] or "") and not EXCLUDE_ID.search(s["id"] or "")
        }
        placeholders = ",".join("?" * len(keep))
        ranked = """WITH ranked AS (
            SELECT id, asset_id, ROW_NUMBER() OVER (
                PARTITION BY asset_id ORDER BY created_at, id) rn
            FROM observations)
            SELECT fi.id, fi.cluster_id, fi.embedding_json, fi.quality,
            fi.pose_bucket, fi.embedding_model, fi.asset_id, a.scope_id
            FROM face_instances fi
            JOIN ranked r ON r.id = fi.observation_id AND r.rn = 1
            JOIN assets a ON a.id = fi.asset_id
            WHERE a.scope_id IN ({0})
            AND fi.cluster_id IS NOT NULL AND fi.embedding_json != '[]'""".format(placeholders)
        instances = connection.execute(ranked, tuple(keep)).fetchall()
    finally:
        connection.close()

    members_by_cluster = defaultdict(list)
    for row in instances:
        members_by_cluster[row["cluster_id"]].append(row)

    model_counts = defaultdict(int)
    for row in instances:
        model_counts[row["embedding_model"] or "unknown"] += 1

    same_asset_conflicts = 0
    conflict_examples = []
    within = defaultdict(list)
    by_size_bucket = defaultdict(lambda: defaultdict(list))
    all_within = []
    all_cross = []

    cluster_ids = sorted(members_by_cluster)
    for cluster_id in cluster_ids:
        members = members_by_cluster[cluster_id]
        size = len(members)
        assets = defaultdict(int)
        for member in members:
            assets[member["asset_id"]] += 1
        for asset_id, count in assets.items():
            if count > 1:
                same_asset_conflicts += 1
                if len(conflict_examples) < 8:
                    conflict_examples.append({
                        "cluster_id": cluster_id,
                        "asset_id": asset_id,
                        "faces": count,
                        "scope_id": next(m["scope_id"] for m in members if m["asset_id"] == asset_id),
                        "models": sorted({m["embedding_model"] for m in members}),
                        "qualities": sorted(round(float(m["quality"]), 2) for m in members)[:4],
                        "poses": sorted({m["pose_bucket"] for m in members}),
                    })
        embeddings = []
        for member in members:
            try:
                embeddings.append((member, [float(x) for x in json.loads(member["embedding_json"])]))
            except (ValueError, TypeError):
                pass
        sims = [_cosine(a, b) for (_, a), (_, b) in combinations(embeddings, 2)]
        if sims:
            within[size].append(sims)
            all_within.extend(sims)
            bucket = 1 if size <= 1 else (2 if size <= 3 else (5 if size <= 8 else (15 if size <= 30 else 100)))
            by_size_bucket[bucket]["sims"].extend(sims)
            by_size_bucket[bucket]["clusters"].append(size)

    reps = []
    for cluster_id in cluster_ids:
        members = members_by_cluster[cluster_id]
        if not members:
            continue
        best = max(members, key=lambda m: float(m["quality"] or 0))
        try:
            vec = [float(x) for x in json.loads(best["embedding_json"])]
        except (ValueError, TypeError):
            continue
        reps.append((cluster_id, vec))
    for (id_a, vec_a), (id_b, vec_b) in combinations(reps, 2):
        all_cross.append(_cosine(vec_a, vec_b))

    size_vs_sim = {
        str(size): _summary([s for group in groups for s in group])
        for size, groups in sorted(within.items())
        if size > 1
    }
    bucket_table = {}
    for bucket, data in sorted(by_size_bucket.items()):
        if bucket <= 1:
            continue
        bucket_table[str(bucket)] = {
            "clusters": len(data["clusters"]),
            "sim": _summary(data["sims"]),
        }

    return {
        "included_spaces": len(keep),
        "faces_after_asset_dedup": len(instances),
        "clusters": len(cluster_ids),
        "embedding_model_counts": dict(model_counts),
        "same_asset_conflicts": same_asset_conflicts,
        "conflict_examples": conflict_examples,
        "cluster_size_vs_within_similarity": size_vs_sim,
        "size_bucket_summary": bucket_table,
        "within_cluster_similarity": _summary(all_within),
        "cross_cluster_similarity": _summary(all_cross),
        "threshold_0_30_position": {
            "within_below_0_30": round(sum(1 for s in all_within if s < 0.30) / max(1, len(all_within)), 4),
            "cross_above_0_30": round(sum(1 for s in all_cross if s >= 0.30) / max(1, len(all_cross)), 4),
            "within_pairs": len(all_within),
            "cross_pairs": len(all_cross),
        },
    }


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Autopsy real-user face clusters read-only.")
    parser.add_argument("--db", type=Path, default=root / "data" / "sentrix.db")
    args = parser.parse_args()
    print(json.dumps(analyze(args.db), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
