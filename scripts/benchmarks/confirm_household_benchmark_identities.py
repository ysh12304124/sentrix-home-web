#!/usr/bin/env python3
"""Confirm only unambiguous household benchmark identities with an audit trail.

The household labels do not contain face bounding boxes. This tool never relies
on detector ordering: it starts from one-label/one-cluster samples, then uses
already-resolved people to eliminate the remaining label in multi-person images.
Conflicts stay in the report for human review.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.db import MemoryStore


def _display_name(nicknames):
    return str((nicknames or [""])[0]).strip()


def _labeled_rows(store, space):
    scope_id = space["scope_id"]
    labels = space.get("evaluation", {}).get("image_to_face_ids", {})
    rows = store._rows(
        """SELECT a.file_name, fi.id, fi.cluster_id
        FROM face_instances fi JOIN assets a ON a.id = fi.asset_id
        WHERE a.scope_id = ? AND fi.cluster_id IS NOT NULL
        ORDER BY a.file_name, fi.id""",
        (scope_id,),
    )
    clusters_by_file = defaultdict(list)
    for row in rows:
        clusters_by_file[row["file_name"]].append(row)
    return [
        {
            "file_name": file_name,
            "face_ids": [str(value) for value in face_ids],
            "detections": clusters_by_file.get(file_name, []),
        }
        for file_name, face_ids in sorted(labels.items())
    ]


def infer_confirmations(store, manifest):
    """Infer a strict one-to-one face-ID to cluster mapping without applying it."""
    resolved = {}
    support = defaultdict(list)
    unresolved = []
    all_rows = []
    aliases = {}

    for space in manifest.get("spaces", []):
        nicknames = space.get("evaluation", {}).get("face_id_to_nicknames", {})
        for face_id, values in nicknames.items():
            aliases[(space["scope_id"], str(face_id))] = list(values or [])
        all_rows.extend((space["scope_id"], row) for row in _labeled_rows(store, space))

    # A one-label/one-cluster image is the only direct seed. Detection order
    # cannot affect it.
    for scope_id, row in all_rows:
        clusters = {item["cluster_id"] for item in row["detections"]}
        if len(row["face_ids"]) == len(clusters) == 1:
            key = (scope_id, row["face_ids"][0])
            cluster_id = next(iter(clusters))
            support[key].append({"file_name": row["file_name"], "face_instance_ids": [item["id"] for item in row["detections"]]})
            resolved.setdefault((scope_id, cluster_id), key)

    # Resolve a multi-person image only after all but one identity and cluster
    # are already known. Repeating this handles chains of album evidence.
    changed = True
    while changed:
        changed = False
        for scope_id, row in all_rows:
            labels = set(row["face_ids"])
            clusters = {item["cluster_id"] for item in row["detections"]}
            if len(labels) != len(clusters) or len(labels) < 2:
                continue
            known_labels = {label for (known_scope, _), (_, label) in resolved.items() if known_scope == scope_id}
            known_clusters = {cluster for (known_scope, cluster) in resolved if known_scope == scope_id}
            unknown_labels = labels - known_labels
            unknown_clusters = clusters - known_clusters
            if len(unknown_labels) == len(unknown_clusters) == 1:
                label, cluster = next(iter(unknown_labels)), next(iter(unknown_clusters))
                key = (scope_id, label)
                mapping_key = (scope_id, cluster)
                if mapping_key not in resolved:
                    resolved[mapping_key] = key
                    support[key].append({"file_name": row["file_name"], "face_instance_ids": [item["id"] for item in row["detections"] if item["cluster_id"] == cluster]})
                    changed = True

    inverse = defaultdict(list)
    for (scope_id, cluster_id), (_, face_id) in resolved.items():
        inverse[(scope_id, face_id)].append(cluster_id)
    confirmed = []
    for (scope_id, cluster_id), (_, face_id) in sorted(resolved.items()):
        display_name = _display_name(aliases.get((scope_id, face_id)))
        if not display_name or len(inverse[(scope_id, face_id)]) != 1:
            continue
        evidence = support[(scope_id, face_id)]
        confirmed.append({
            "scope_id": scope_id,
            "face_id": face_id,
            "name": display_name,
            "aliases": aliases.get((scope_id, face_id), []),
            "cluster_id": cluster_id,
            "support_files": [item["file_name"] for item in evidence],
            "face_instance_ids": [face_id for item in evidence for face_id in item["face_instance_ids"]],
        })

    confirmed_keys = {(item["scope_id"], item["cluster_id"]) for item in confirmed}
    for scope_id, row in all_rows:
        clusters = {item["cluster_id"] for item in row["detections"]}
        if row["face_ids"] and not clusters:
            unresolved.append({"scope_id": scope_id, "file_name": row["file_name"], "reason": "标注人物未检测到可聚类人脸"})
        elif len(row["face_ids"]) != len(clusters):
            unresolved.append({"scope_id": scope_id, "file_name": row["file_name"], "reason": "标注人数与可聚类检测数不一致"})
        elif any((scope_id, cluster_id) not in confirmed_keys for cluster_id in clusters):
            unresolved.append({"scope_id": scope_id, "file_name": row["file_name"], "reason": "缺少无歧义簇到标注人物映射"})
    return {"confirmations": confirmed, "unresolved": unresolved}


def apply_confirmations(store, report):
    applied = []
    for item in report["confirmations"]:
        cluster = store._row("SELECT * FROM face_clusters WHERE id = ? AND scope_id = ?", (item["cluster_id"], item["scope_id"]))
        if not cluster:
            continue
        previous = store.get_entity(cluster.get("entity_id")) if cluster.get("entity_id") else None
        result = store.confirm_face_cluster(item["cluster_id"], item["name"])
        if not result:
            continue
        entity = result["entity"]
        store._record_entity_revision(
            entity["id"], "benchmark_identity_confirmation",
            previous.get("canonical_name") if previous else "",
            item["name"], "authorized_household_face_info",
            item["face_instance_ids"],
        )
        store.connection.execute(
            "UPDATE entities SET summary = ?, updated_at = ? WHERE id = ?",
            ("依据用户授权的相册 face_info 标注自动确认；可回溯到标注样本与人脸证据", __import__("backend.db", fromlist=["now_iso"]).now_iso(), entity["id"]),
        )
        store.connection.commit()
        applied.append({**item, "entity_id": entity["id"]})
    return applied


def run(manifest_path, database_path, apply=False):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    store = MemoryStore(str(database_path))
    try:
        report = infer_confirmations(store, manifest)
        report["applied"] = apply_confirmations(store, report) if apply else []
        report["summary"] = {
            "eligible_confirmations": len(report["confirmations"]),
            "applied_confirmations": len(report["applied"]),
            "unresolved_samples": len(report["unresolved"]),
        }
        return report
    finally:
        store.close()


def main():
    parser = argparse.ArgumentParser(description="Confirm only unambiguous authorized household face labels.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("database", type=Path)
    parser.add_argument("--apply", action="store_true", help="Apply the inferred confirmations; default is read-only report.")
    args = parser.parse_args()
    print(json.dumps(run(args.manifest, args.database, args.apply), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
