#!/usr/bin/env python3
"""Sweep FaceClusterer parameters against a benchmark manifest, read-only.

Re-clusters the already-detected face instances of an isolated benchmark DB with
different match_threshold / minimum_quality and reports pairwise metrics using
the manifest identities. No re-detection, no writes.
"""

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.face_clustering import FaceClusterer, FaceSample, pairwise_metrics


def main():
    parser = argparse.ArgumentParser(description="Sweep face clustering parameters.")
    parser.add_argument("--db", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--thresholds", default="0.25,0.30,0.35,0.40,0.45")
    parser.add_argument("--min-qualities", default="0.45,0.55,0.65")
    parser.add_argument("--strategies", default="all")
    parser.add_argument("--min-det-score", type=float, help="Only cluster faces whose detection confidence reaches this (detected-but-weak faces are kept as evidence, excluded from clustering).")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    truth_by_file = {Path(a["file"]).name: a["source_identity"] for a in manifest["assets"]}

    connection = sqlite3.connect(args.db)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """SELECT fi.id, fi.embedding_json, fi.quality, fi.pose_bucket,
        fi.embedding_model, fi.embedding_version, fi.detection_confidence,
        fi.area_ratio, a.file_name
        FROM face_instances fi JOIN assets a ON a.id = fi.asset_id"""
    ).fetchall()
    connection.close()

    cluster_rows = rows
    if args.min_det_score is not None:
        cluster_rows = [r for r in rows if float(r["detection_confidence"] or 0) >= args.min_det_score]

    samples = []
    by_file = defaultdict(list)
    for row in cluster_rows:
        samples.append(FaceSample(
            row["id"],
            [float(x) for x in json.loads(row["embedding_json"])],
            quality=float(row["quality"] or 0),
            pose_bucket=row["pose_bucket"] or "unknown",
            model_name=row["embedding_model"] or "unknown",
            model_version=row["embedding_version"] or "unknown",
        ))
        by_file[row["file_name"]].append((
            row["id"], float(row["detection_confidence"] or 0),
            float(row["area_ratio"] or 0), float(row["quality"] or 0),
        ))

    header = f"{'strategy':>8} {'threshold':>9} {'min_qual':>8} {'prec':>7} {'rec':>7} {'f1':>7} {'sing':>7} {'clusters':>8} {'fp':>4} {'fn':>5}"
    print(header)
    for strategy in [x.strip() for x in args.strategies.split(",") if x.strip()]:
        for threshold in [float(x) for x in args.thresholds.split(",")]:
            for minq in [float(x) for x in args.min_qualities.split(",")]:
                result = FaceClusterer(match_threshold=threshold, minimum_quality=minq, match_strategy=strategy).fit(samples)
                labels = result.labels
                predicted = {}
                for file_name, candidates in by_file.items():
                    primary = max(candidates, key=lambda c: (c[1], c[2], c[3], c[0]))
                    predicted[file_name] = labels.get(primary[0], f"missing:{file_name}")
                truth = {file_name: truth_by_file[file_name] for file_name in by_file if file_name in truth_by_file}
                metrics = pairwise_metrics(predicted, truth)
                print(
                    f"{strategy:>8} {threshold:>9.2f} {minq:>8.2f} {metrics['precision']:>7.4f} "
                    f"{metrics['recall']:>7.4f} {metrics['f1']:>7.4f} {metrics['singleton_ratio']:>7.4f} "
                    f"{len(result.clusters):>8} {metrics['false_positive']:>4} {metrics['false_negative']:>5}"
                )


if __name__ == "__main__":
    main()
