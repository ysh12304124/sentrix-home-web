#!/usr/bin/env python3
"""Rebuild face instances/clusters for album1/2/3 with the current identity
embedding (buffalo_l legacy) and the tiered detection policy (detect at 0.5,
cluster only det_score >= 0.72).

Keeps observations/events/entities untouched. Removes the old AdaFace face
derivatives (instances, prototypes, vectors, mention/evidence links, orphan
clusters) and re-runs detection + clustering so the three household albums use
the improved model. A database backup is expected before running.
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.db import MemoryStore
from backend.model_clients import FaceAdapter

SCOPES = ["album1", "album2", "album3"]


def rebuild(db_path):
    store = MemoryStore(str(db_path))
    face = FaceAdapter()
    if face.enabled and not face.identity_ready:
        raise RuntimeError(
            f"face identity not ready: {face.identity_error or face.error}"
        )
    report = {}
    for scope in SCOPES:
        assets = store._rows("SELECT id, path FROM assets WHERE scope_id = ?", (scope,))
        removed_faces = removed_clusters = added_faces = 0
        started = time.perf_counter()
        for asset in assets:
            obs_ids = [
                row["id"]
                for row in store._rows(
                    "SELECT id FROM observations WHERE asset_id = ?", (asset["id"],)
                )
            ]
            if not obs_ids:
                continue
            face_ids, cluster_ids = [], set()
            for oid in obs_ids:
                for row in store._rows(
                    "SELECT id, cluster_id FROM face_instances WHERE observation_id = ?",
                    (oid,),
                ):
                    face_ids.append(row["id"])
                    if row["cluster_id"]:
                        cluster_ids.add(row["cluster_id"])
            if face_ids:
                placeholders = ",".join("?" * len(face_ids))
                store.connection.execute(
                    f"DELETE FROM person_appearance_evidence WHERE face_instance_id IN ({placeholders})",
                    face_ids,
                )
                store.connection.execute(
                    f"DELETE FROM entity_mentions WHERE face_instance_id IN ({placeholders})",
                    face_ids,
                )
                store.connection.execute(
                    f"DELETE FROM face_prototypes WHERE face_instance_id IN ({placeholders})",
                    face_ids,
                )
                store.connection.execute(
                    "DELETE FROM memory_vectors WHERE source_type = 'face_instance' AND source_id IN ({})".format(
                        placeholders
                    ),
                    face_ids,
                )
                store.connection.execute(
                    f"DELETE FROM face_instances WHERE id IN ({placeholders})", face_ids
                )
                removed_faces += len(face_ids)
            for cluster_id in cluster_ids:
                if not store._row(
                    "SELECT id FROM face_instances WHERE cluster_id = ?", (cluster_id,)
                ):
                    store.connection.execute(
                        "DELETE FROM face_clusters WHERE id = ?", (cluster_id,)
                    )
                    removed_clusters += 1
            store.connection.commit()

            path = asset.get("path")
            if not path or not os.path.isfile(path):
                continue
            detections = face.detect(path)
            for det in detections:
                store.add_face_instance(asset["id"], obs_ids[0], det)
            added_faces += len(detections)
            store.connection.commit()

        recluster = store.recluster_faces(scope_id=scope)
        report[scope] = {
            "assets": len(assets),
            "removed_faces": removed_faces,
            "removed_clusters": removed_clusters,
            "added_faces": added_faces,
            "seconds": round(time.perf_counter() - started, 2),
            "recluster": recluster,
        }
    store.close()
    return report


def main():
    root = Path(__file__).resolve().parents[2]
    parser = __import__("argparse").ArgumentParser(description="Rebuild album1/2/3 faces.")
    parser.add_argument("--db", default=root / "data" / "sentrix.db")
    args = parser.parse_args()
    print(json.dumps(rebuild(args.db), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
