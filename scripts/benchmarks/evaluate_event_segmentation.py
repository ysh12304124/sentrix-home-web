#!/usr/bin/env python3
"""Evaluate Sentrix event grouping against external, non-imported labels."""

import argparse
import itertools
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


def _truth_from_manifest(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    assets = manifest.get("assets", []) if isinstance(manifest, dict) else []
    if assets:
        return {
            item["file"]: item["event_id"]
            for item in assets
            if isinstance(item, dict) and item.get("file") and item.get("event_id")
        }
    if not isinstance(manifest, dict):
        return {}
    return {
        filename: metadata["event_id"]
        for filename, metadata in manifest.items()
        if isinstance(metadata, dict) and metadata.get("event_id")
    }


def _pairwise_metrics(predicted, truth):
    counts = {"true_positive": 0, "false_positive": 0, "false_negative": 0, "true_negative": 0}
    ids = sorted(set(predicted).intersection(truth))
    for left, right in itertools.combinations(ids, 2):
        predicted_same = predicted[left] == predicted[right]
        truth_same = truth[left] == truth[right]
        if predicted_same and truth_same:
            counts["true_positive"] += 1
        elif predicted_same:
            counts["false_positive"] += 1
        elif truth_same:
            counts["false_negative"] += 1
        else:
            counts["true_negative"] += 1
    precision_denominator = counts["true_positive"] + counts["false_positive"]
    recall_denominator = counts["true_positive"] + counts["false_negative"]
    precision = counts["true_positive"] / precision_denominator if precision_denominator else 0.0
    recall = counts["true_positive"] / recall_denominator if recall_denominator else 0.0
    return {
        **counts,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def _asset_keys(path, file_name, manifest_root):
    keys = []
    try:
        keys.append(Path(path).resolve().relative_to(manifest_root.resolve()).as_posix())
    except (OSError, ValueError):
        pass
    if file_name not in keys:
        keys.append(file_name)
    return keys


def evaluate(db_path, manifest_path):
    manifest_path = Path(manifest_path)
    truth_by_file = _truth_from_manifest(manifest_path)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """SELECT a.id AS asset_id, a.file_name, a.path, eo.event_id FROM assets a
        JOIN observations o ON o.asset_id = a.id
        JOIN event_observations eo ON eo.observation_id = o.id"""
    ).fetchall()
    filename_counts = Counter(row["file_name"] for row in rows)
    manifest_basename_counts = Counter(Path(key).name for key in truth_by_file)
    predicted = {}
    truth = {}
    matched_manifest_keys = set()
    for asset_id, file_name, path, event_id in rows:
        key = next((candidate for candidate in _asset_keys(path, file_name, manifest_path.parent)
                    if candidate in truth_by_file and candidate != file_name), None)
        if not key and filename_counts[file_name] == 1 and manifest_basename_counts[file_name] == 1:
            key = file_name if file_name in truth_by_file else None
        if key:
            predicted[asset_id] = event_id
            truth[asset_id] = truth_by_file[key]
            matched_manifest_keys.add(key)
    true_groups = defaultdict(set)
    predicted_groups = defaultdict(set)
    for filename, truth_event in truth.items():
        true_groups[truth_event].add(filename)
        predicted_groups[predicted[filename]].add(filename)
    split_events = {
        label: sorted({predicted[filename] for filename in files})
        for label, files in true_groups.items()
    }
    merged_events = {
        label: sorted({truth[filename] for filename in files})
        for label, files in predicted_groups.items()
    }
    metrics = _pairwise_metrics(predicted, truth)
    result = {
        "assets_evaluated": len(predicted),
        "unmatched_manifest_assets": sorted(set(truth_by_file) - matched_manifest_keys),
        "truth_event_count": len(true_groups),
        "predicted_event_count": len(predicted_groups),
        **{key: round(value, 4) if isinstance(value, float) else value for key, value in metrics.items()},
        "split_truth_events": sum(len(events) > 1 for events in split_events.values()),
        "merged_predicted_events": sum(len(events) > 1 for events in merged_events.values()),
        "splits": {label: events for label, events in split_events.items() if len(events) > 1},
        "merges": {label: events for label, events in merged_events.items() if len(events) > 1},
    }
    connection.close()
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=root / "data" / "sentrix.db")
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.db, args.manifest), ensure_ascii=False, indent=2))
