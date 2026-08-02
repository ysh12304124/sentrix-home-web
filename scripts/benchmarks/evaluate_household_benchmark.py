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
    samples = []
    matched_faces = 0
    unmatched_truth = 0
    unmatched_prediction = 0
    for file_name, face_ids in truth_by_file.items():
        predicted = predicted_by_file.get(file_name, [])
        for truth_id, prediction in zip(sorted(map(str, face_ids)), predicted):
            samples.append((truth_id, prediction["cluster_id"]))
            matched_faces += 1
        unmatched_truth += max(0, len(face_ids) - len(predicted))
        unmatched_prediction += max(0, len(predicted) - len(face_ids))
    metrics = _pair_metrics(samples)
    metrics.update({
        "truth_occurrences": sum(len(value) for value in truth_by_file.values()),
        "predicted_occurrences": sum(len(value) for value in predicted_by_file.values()),
        "matched_occurrences": matched_faces,
        "unmatched_truth_occurrences": unmatched_truth,
        "unmatched_prediction_occurrences": unmatched_prediction,
        "assignment": "按文件名和排序后的 face instance 配对；缺少 bbox 对齐标注时仅评估可配对样本",
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
