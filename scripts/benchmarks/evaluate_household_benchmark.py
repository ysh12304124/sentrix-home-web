"""Evaluate the household benchmark without writing its labels to memory."""

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.agent import MemoryAgent
from backend.db import MemoryStore


def _pair_metrics(samples):
    truth_pairs = {tuple(sorted((left, right))) for left, right in itertools.combinations(samples, 2) if left[0] == right[0]}
    predicted_pairs = {tuple(sorted((left, right))) for left, right in itertools.combinations(samples, 2) if left[1] and left[1] == right[1]}
    true_positive = len(truth_pairs & predicted_pairs)
    precision = true_positive / len(predicted_pairs) if predicted_pairs else 0.0
    recall = true_positive / len(truth_pairs) if truth_pairs else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4), "truth_pairs": len(truth_pairs), "predicted_pairs": len(predicted_pairs)}


def _input_diagnostics(manifest):
    result = {}
    for space in manifest.get("spaces", []):
        diagnostics = space.get("diagnostics", {})
        result[space["scope_id"]] = {
            "images": len(space.get("import", {}).get("files", [])),
            "metadata_missing_images": len(diagnostics.get("metadata_missing_images", [])),
            "face_missing_images": len(diagnostics.get("face_missing_images", [])),
            "images_without_metadata": len(diagnostics.get("images_without_metadata", [])),
            "images_without_face_map": len(diagnostics.get("images_without_face_map", [])),
            "queries": len(space.get("evaluation", {}).get("queries", [])),
            "query_ground_truth_missing": sum(len(item.get("missing_ground_truth", [])) for item in space.get("evaluation", {}).get("queries", [])),
        }
    return result


def _face_metrics(store, space):
    scope_id = space["scope_id"]
    truth_by_file = space.get("evaluation", {}).get("image_to_face_ids", {})
    rows = store._rows(
        """SELECT a.file_name, fi.id, fi.cluster_id FROM face_instances fi
        JOIN assets a ON a.id = fi.asset_id WHERE a.scope_id = ? AND fi.cluster_id IS NOT NULL
        ORDER BY a.file_name, fi.id""",
        (scope_id,),
    )
    predicted_by_file = {}
    for row in rows:
        predicted_by_file.setdefault(row["file_name"], []).append(row)
    # The provided household labels identify people per image, not face boxes.
    # A detector's order is not a valid identity alignment.  Only an image with
    # exactly one authorized label and one clustered detection can establish a
    # direct, order-independent evaluation sample.  Multi-person images remain
    # visible in coverage diagnostics instead of inflating or corrupting F1.
    samples = []
    direct_rows = []
    matched_faces = 0
    unmatched_truth = 0
    unmatched_prediction = 0
    for file_name, face_ids in truth_by_file.items():
        labels = [str(value) for value in face_ids]
        predicted = predicted_by_file.get(file_name, [])
        if len(labels) == len(predicted) == 1:
            direct_rows.append((file_name, labels[0], predicted[0]["cluster_id"]))
        else:
            unmatched_truth += len(labels)
            unmatched_prediction += len(predicted)

    label_clusters = {}
    cluster_labels = {}
    for _, label, cluster_id in direct_rows:
        label_clusters.setdefault(label, set()).add(cluster_id)
        cluster_labels.setdefault(cluster_id, set()).add(label)
    stable_labels = {
        label for label, cluster_ids in label_clusters.items()
        if len(cluster_ids) == 1 and len(cluster_labels[next(iter(cluster_ids))]) == 1
    }
    for _, label, cluster_id in direct_rows:
        if label not in stable_labels:
            unmatched_truth += 1
            unmatched_prediction += 1
            continue
        samples.append((label, cluster_id))
        matched_faces += 1
    metrics = _pair_metrics(samples)
    truth_occurrences = sum(len(value) for value in truth_by_file.values())
    predicted_occurrences = sum(len(value) for value in predicted_by_file.values())
    ambiguous_occurrences = truth_occurrences - matched_faces
    metrics.update({
        "truth_occurrences": truth_occurrences,
        "predicted_occurrences": predicted_occurrences,
        "matched_occurrences": matched_faces,
        "validated_occurrences": matched_faces,
        "ambiguous_occurrences": ambiguous_occurrences,
        "unmatched_truth_occurrences": unmatched_truth,
        "unmatched_prediction_occurrences": unmatched_prediction,
        "detection_coverage": round(sum(min(len(value), len(predicted_by_file.get(file_name, []))) for file_name, value in truth_by_file.items()) / truth_occurrences, 4) if truth_occurrences else 0.0,
        "validation_coverage": round(matched_faces / truth_occurrences, 4) if truth_occurrences else 0.0,
        "assignment": "仅单标签且单检测的图片建立无序身份映射；多人或数量不一致样本只计入覆盖率",
    })
    return metrics


def _query_metrics(store, space, top_k=20):
    agent = MemoryAgent(store)
    scope_id = space["scope_id"]
    rows = []
    for query in space.get("evaluation", {}).get("queries", []):
        result = agent.retrieve(query.get("query_cn") or query.get("query_en") or "", scope_id=scope_id)
        asset_names = []
        for observation in result.get("observations", []):
            asset = store.get_asset(observation.get("asset_id")) or {}
            if asset.get("file_name") and asset["file_name"] not in asset_names:
                asset_names.append(asset["file_name"])
        ground_truth = set(query.get("ground_truth", []))
        predicted = asset_names[:top_k]
        hits = len(ground_truth.intersection(predicted))
        precision = hits / len(predicted) if predicted else 0.0
        recall = hits / len(ground_truth) if ground_truth else 0.0
        rows.append({
            "query": query.get("query_cn") or query.get("query_en"),
            "ground_truth_count": len(ground_truth),
            "predicted_count": len(predicted),
            "hits": hits,
            "precision_at_k": round(precision, 4),
            "recall_at_k": round(recall, 4),
            "hit": bool(hits),
        })
    valid = [item for item in rows if item["ground_truth_count"]]
    return {
        "top_k": top_k,
        "queries": rows,
        "hit_rate": round(sum(item["hit"] for item in valid) / len(valid), 4) if valid else 0.0,
        "mean_precision": round(sum(item["precision_at_k"] for item in valid) / len(valid), 4) if valid else 0.0,
        "mean_recall": round(sum(item["recall_at_k"] for item in valid) / len(valid), 4) if valid else 0.0,
    }


def _event_diagnostics(store, space):
    scope_id = space["scope_id"]
    events = store.list_events(100000, scope_id=scope_id)
    event_by_asset = {}
    for event in events:
        for observation_id in event.get("observation_ids", []):
            observation = store.get_observation(observation_id)
            if observation:
                asset = store.get_asset(observation["asset_id"])
                if asset:
                    event_by_asset.setdefault(asset["file_name"], set()).add(event["id"])
    fragmentation = []
    for query in space.get("evaluation", {}).get("queries", []):
        event_ids = set().union(*(event_by_asset.get(name, set()) for name in query.get("ground_truth", [])))
        fragmentation.append({"query": query.get("query_cn"), "ground_truth_events": len(event_ids)})
    return {
        "events": len(events),
        "observations": sum(len(event.get("observation_ids", [])) for event in events),
        "singleton_events": sum(len(event.get("observation_ids", [])) == 1 for event in events),
        "query_ground_truth_fragmentation": fragmentation,
    }


def _scope_isolation(store):
    checks = {
        "observation_asset_scope_violations": store.connection.execute("SELECT COUNT(*) FROM observations o JOIN assets a ON a.id = o.asset_id WHERE o.scope_id != a.scope_id").fetchone()[0],
        "event_observation_scope_violations": store.connection.execute("""SELECT COUNT(*) FROM event_observations eo
            JOIN events e ON e.id = eo.event_id JOIN observations o ON o.id = eo.observation_id
            WHERE e.scope_id != o.scope_id""").fetchone()[0],
        "vector_asset_scope_violations": store.connection.execute("""SELECT COUNT(*) FROM memory_vectors v JOIN assets a
            ON a.id = json_extract(v.metadata_json, '$.asset_id') WHERE v.scope_id != a.scope_id""").fetchone()[0],
    }
    checks["passed"] = all(value == 0 for value in checks.values())
    return checks


def evaluate(manifest_path, db_path, top_k=20):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    store = MemoryStore(str(db_path))
    try:
        spaces = {}
        for space in manifest.get("spaces", []):
            scope_id = space["scope_id"]
            confirmed_people = store.list_entities(status="confirmed", scope_id=scope_id)
            spaces[scope_id] = {
                "face_clustering": _face_metrics(store, space),
                "events": _event_diagnostics(store, space),
                "queries": _query_metrics(store, space, top_k),
                "person_memory": {
                    "confirmed_people": len([item for item in confirmed_people if item.get("entity_type") == "person"]),
                    "event_memory_rows": store.connection.execute("SELECT COUNT(*) FROM person_event_memory WHERE scope_id = ?", (scope_id,)).fetchone()[0],
                    "pattern_rows": store.connection.execute("SELECT COUNT(*) FROM person_patterns WHERE scope_id = ?", (scope_id,)).fetchone()[0],
                },
            }
        return {"input_diagnostics": _input_diagnostics(manifest), "spaces": spaces, "scope_isolation": _scope_isolation(store)}
    finally:
        store.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("database", type=Path)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.manifest, args.database, max(1, args.top_k)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
