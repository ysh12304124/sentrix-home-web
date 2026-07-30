#!/usr/bin/env python3
"""Evaluate a controlled LFW-backed album without leaking labels into Sentrix."""

import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.face_clustering import pairwise_metrics


def _truth_from_manifest(manifest_path):
    if not manifest_path:
        return {}
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return {item["file"]: item["source_identity"] for item in manifest.get("assets", [])}


def evaluate(db_path, manifest_path=None):
    connection = sqlite3.connect(db_path)
    truth_by_file = _truth_from_manifest(manifest_path)
    if truth_by_file:
        rows = connection.execute("""SELECT fi.id, fi.cluster_id, a.file_name
            FROM face_instances fi JOIN assets a ON a.id = fi.asset_id
            WHERE fi.cluster_id IS NOT NULL""").fetchall()
        rows = [row for row in rows if row[2] in truth_by_file]
        predicted = {face_id: cluster_id for face_id, cluster_id, _ in rows}
        truth = {face_id: truth_by_file[file_name] for face_id, _, file_name in rows}
        metrics = pairwise_metrics(predicted, truth)
        clusters = defaultdict(list)
        for _, cluster_id, file_name in rows:
            clusters[cluster_id].append(truth_by_file[file_name])
        return {
            "identity_samples": len(rows),
            "known_identities": len(set(truth.values())),
            "clusters": len(clusters),
            **{key: round(value, 4) if isinstance(value, float) else value for key, value in metrics.items()},
            "largest_clusters": sorted(
                (
                    {"cluster_id": cluster_id, "members": len(labels), "labels": Counter(labels).most_common(3)}
                    for cluster_id, labels in clusters.items()
                ),
                key=lambda item: item["members"], reverse=True,
            )[:10],
        }
    rows = connection.execute("""SELECT fi.cluster_id, a.path
        FROM face_instances fi JOIN assets a ON a.id = fi.asset_id
        WHERE json_extract(a.metadata_json, '$.benchmark') = 'lfw'""").fetchall()
    clusters = defaultdict(list)
    for cluster_id, path in rows:
        clusters[cluster_id].append(Path(path).parent.name)
    if not clusters:
        return {"images": 0, "clusters": 0, "purity": 0.0, "coverage": 0.0}
    total = sum(len(labels) for labels in clusters.values())
    correctly_clustered = sum(Counter(labels).most_common(1)[0][1] for labels in clusters.values())
    multi_image_clusters = sum(1 for labels in clusters.values() if len(labels) > 1)
    return {
        "images": total,
        "clusters": len(clusters),
        "multi_image_clusters": multi_image_clusters,
        "purity": round(correctly_clustered / total, 4),
        "coverage": round(multi_image_clusters / len(clusters), 4),
        "largest_clusters": sorted(({"cluster_id": cluster_id, "members": len(labels), "labels": Counter(labels).most_common(3)} for cluster_id, labels in clusters.items()), key=lambda item: item["members"], reverse=True)[:10],
    }


if __name__ == "__main__":
    import argparse

    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=root / "data" / "sentrix.db")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.db, args.manifest), ensure_ascii=False, indent=2))
