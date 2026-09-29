#!/usr/bin/env python3
"""Read-only identity clustering gate for an LFW-backed evaluation manifest."""

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.face_clustering import pairwise_metrics


def _truth_from_manifest(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return {str(Path(item["file"]).name): str(item["source_identity"]) for item in manifest.get("assets", [])}


def _round_metrics(metrics):
    return {key: round(value, 4) if isinstance(value, float) else value for key, value in metrics.items()}


def evaluate(db_path, manifest_path):
    """Evaluate persisted clusters without storing benchmark labels in SQLite.

    LFW manifests label one primary identity per image. The highest-confidence
    face is therefore the labeled sample; other faces stay in diagnostics and
    never inherit a label that the manifest did not provide. A missing primary
    detection receives a private predicted label so missing same-identity pairs
    count as false negatives instead of silently inflating recall.
    """
    truth_by_file = _truth_from_manifest(manifest_path)
    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(
            """SELECT fi.id, fi.cluster_id, fi.detection_confidence, fi.area_ratio,
            fi.quality, a.file_name FROM face_instances fi
            JOIN assets a ON a.id = fi.asset_id
            WHERE fi.cluster_id IS NOT NULL"""
        ).fetchall()
    finally:
        connection.close()
    predicted_by_file = defaultdict(list)
    for face_id, cluster_id, confidence, area_ratio, quality, file_name in rows:
        if file_name in truth_by_file:
            predicted_by_file[file_name].append({
                "face_id": str(face_id), "cluster_id": str(cluster_id),
                "confidence": float(confidence or 0), "area_ratio": float(area_ratio or 0),
                "quality": float(quality or 0),
            })

    predicted = {}
    truth = {}
    detected_samples = 0
    extra_detections = 0
    for file_name, identity in sorted(truth_by_file.items()):
        candidates = predicted_by_file.get(file_name, [])
        sample_id = f"expected:{file_name}"
        truth[sample_id] = identity
        if candidates:
            primary = max(
                candidates,
                key=lambda item: (item["confidence"], item["area_ratio"], item["quality"], item["face_id"]),
            )
            predicted[sample_id] = primary["cluster_id"]
            detected_samples += 1
            extra_detections += max(0, len(candidates) - 1)
        else:
            predicted[sample_id] = f"missing:{file_name}"

    metrics = pairwise_metrics(predicted, truth)
    clusters = defaultdict(list)
    for file_name, candidates in predicted_by_file.items():
        for candidate in candidates:
            clusters[candidate["cluster_id"]].append(truth_by_file[file_name])
    expected_samples = len(truth)
    result = {
        "expected_samples": expected_samples,
        "detected_samples": detected_samples,
        "missing_samples": expected_samples - detected_samples,
        "extra_detections": extra_detections,
        "coverage": detected_samples / expected_samples if expected_samples else 0.0,
        "known_identities": len(set(truth.values())),
        "clusters": len(clusters),
        **metrics,
        "pairwise_f1": metrics["f1"],
        "largest_clusters": sorted(
            (
                {"cluster_id": cluster_id, "members": len(labels), "labels": Counter(labels).most_common(3)}
                for cluster_id, labels in clusters.items()
            ),
            key=lambda item: item["members"], reverse=True,
        )[:10],
    }
    return _round_metrics(result)


def meets_gate(result, minimum_f1=0.95, minimum_coverage=0.95):
    return result.get("pairwise_f1", 0.0) >= minimum_f1 and result.get("coverage", 0.0) >= minimum_coverage


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Validate read-only LFW identity clustering quality.")
    parser.add_argument("--db", type=Path, default=root / "data" / "sentrix.db")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--minimum-f1", type=float, default=0.95)
    parser.add_argument("--minimum-coverage", type=float, default=0.95)
    args = parser.parse_args()
    result = evaluate(args.db, args.manifest)
    result["gate"] = {
        "minimum_pairwise_f1": args.minimum_f1,
        "minimum_coverage": args.minimum_coverage,
        "passed": meets_gate(result, args.minimum_f1, args.minimum_coverage),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
