#!/usr/bin/env python3
"""Evaluate unsupervised face clusters against LFW directory labels."""

import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


def evaluate(db_path):
    connection = sqlite3.connect(db_path)
    rows = connection.execute("""SELECT fi.cluster_id, a.path
        FROM face_instances fi JOIN assets a ON a.id = fi.asset_id
        WHERE a.path LIKE '%/test-run/lfw/%'""").fetchall()
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
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(evaluate(root / "data" / "sentrix.db"), ensure_ascii=False, indent=2))
