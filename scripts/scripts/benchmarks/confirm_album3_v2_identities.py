#!/usr/bin/env python3
"""album3-v2 身份确认：用 manifest 人脸参考图(用户授权的人物名/关系)做实例级实体绑定。

流程：
  1. 对 8 张 ref 图跑人脸检测(40% 灰边填充)取最高分 embedding，L2 归一化
  2. 对 scope 内所有 face_instances 的 embedding 做归一化 cosine 匹配
     （阈值 --threshold 默认 0.45，top1-top2 margin 默认 0.08）
  3. 对高置信实例直接挂 entity_mention（不走聚类，避免聚类纯度问题）
  4. 刷新 event participants + 重建 person memory + 重分段事件

只写人物名/关系（人工授权数据），不写任何 benchmark 答案。

用法（153 上）:
  FACE_EMBEDDING_MODE=legacy SENTRIX_DB_PATH=data/sentrix.db \
  .venv/bin/python scripts/benchmarks/confirm_album3_v2_identities.py \
    --faces data/album3-v2-source/faces --manifest /tmp/album3-manifest.json \
    --scope album3-v2 --apply
"""
import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.db import MemoryStore, now_iso
from backend.model_clients import FaceAdapter


def _cosine(left, right):
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def _normalize(values):
    norm = math.sqrt(sum(v * v for v in (values or [])))
    return [v / norm for v in values] if norm else []


def _pad_face_crop(path, pad_ratio=0.4):
    try:
        import cv2
        import numpy as np
        image = cv2.imread(str(path))
        height, width = image.shape[:2]
        pad_x, pad_y = int(width * pad_ratio), int(height * pad_ratio)
        canvas = np.full((height + 2 * pad_y, width + 2 * pad_x, 3), 128, dtype=np.uint8)
        canvas[pad_y:pad_y + height, pad_x:pad_x + width] = image
        temp = Path("/tmp") / f"padded_{path.name}"
        cv2.imwrite(str(temp), canvas)
        return temp
    except Exception:
        return path


def detect_ref_embeddings(face_adapter, faces_dir):
    refs = {}
    for path in sorted(Path(faces_dir).glob("faceid_*.jpg")):
        face_id = path.stem.replace("faceid_", "")
        detections = face_adapter.detect(str(_pad_face_crop(path))) or []
        if not detections:
            print(f"[warn] 未在 {path.name} 检测到人脸")
            continue
        best = max(detections, key=lambda d: d.get("confidence", 0))
        emb = _normalize(best.get("embedding"))
        if emb:
            refs[face_id] = emb
            print(f"ref faceid_{face_id}: detected, quality={best.get('quality'):.2f}")
    return refs


def label_instances(store, scope_id, refs, threshold, margin):
    rows = store._rows(
        """SELECT fi.id, fi.embedding_json, fi.observation_id, a.file_name
        FROM face_instances fi JOIN assets a ON a.id = fi.asset_id
        WHERE a.scope_id = ? AND fi.embedding_json != '[]'""",
        (scope_id,),
    )
    labels = {}
    by_file = {}
    for row in rows:
        emb = _normalize(json.loads(row["embedding_json"] or "[]"))
        if not emb:
            continue
        scored = sorted(((face_id, _cosine(emb, ref)) for face_id, ref in refs.items()),
                        key=lambda item: item[1], reverse=True)
        top_id, top_score = scored[0]
        second_score = scored[1][1] if len(scored) > 1 else -1.0
        if top_score >= threshold and (top_score - second_score) >= margin:
            labels[row["id"]] = {
                "face_id": top_id, "score": round(top_score, 3),
                "observation_id": row["observation_id"], "file_name": row["file_name"],
            }
            by_file.setdefault(row["file_name"], []).append(top_id)
    return labels, by_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faces", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--scope", default="album3-v2")
    ap.add_argument("--db", default=os.getenv("SENTRIX_DB_PATH", ""))
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--margin", type=float, default=0.08)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    manifest_data = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    faces = manifest_data["faces"]
    face_info = {str(f["face_id"]): f for f in faces}
    store = MemoryStore(args.db)
    face = FaceAdapter()
    print(f"face enabled={face.enabled} identity_ready={face.identity_ready}")

    refs = detect_ref_embeddings(face, args.faces)
    print(f"ref 图匹配到 {len(refs)} 个人脸身份: {sorted(refs)}")
    labels, by_file = label_instances(store, args.scope, refs, args.threshold, args.margin)
    print(f"高置信标记实例: {len(labels)}，覆盖照片: {len(by_file)}")
    per_face = Counter(item["face_id"] for item in labels.values())
    print(f"按身份分布: {dict(per_face)}")
    for file_name, ids in sorted(by_file.items()):
        names = ", ".join(f"{iid}({face_info.get(iid, {}).get('canonical_name', '?')})" for iid in ids)
        print(f"  {file_name}: {names}")

    if not args.apply:
        print("(dry-run，未写库；加 --apply 生效)")
        return

    entities = {}
    linked = 0
    affected_observations = set()
    affected_clusters = set()
    for instance_id, item in labels.items():
        info = face_info.get(item["face_id"])
        if not info:
            continue
        name = info["canonical_name"]
        if name not in entities:
            existing = store.find_confirmed_person_by_name(args.scope, name)
            if existing:
                entity = store.get_entity(existing["entity_id"]) if isinstance(existing, dict) else store.get_entity(existing)
            else:
                entity = store.create_entity(name, "person", "confirmed",
                                             info.get("family_role"), 1.0,
                                             "依据用户授权相册身份标注确认", scope_id=args.scope)
            aliases = list(info.get("aliases") or [])
            if aliases:
                store.set_person_aliases(entity["id"], aliases)
            entities[name] = entity
        # A production cluster assignment is only a candidate.  Once the user
        # has authorized the reference-face mapping, move the concrete face
        # instance to that confirmed cluster and remove stale names attached
        # by an earlier low-margin match (e.g. the child in album3v4-097).
        target_cluster = store._row(
            "SELECT id FROM face_clusters WHERE scope_id = ? AND entity_id = ? AND status = 'confirmed' ORDER BY updated_at DESC LIMIT 1",
            (args.scope, entity["id"]),
        )
        if target_cluster:
            current = store._row("SELECT cluster_id FROM face_instances WHERE id = ?", (instance_id,))
            if current and current.get("cluster_id"):
                affected_clusters.add(current["cluster_id"])
            if current and current.get("cluster_id") != target_cluster["id"]:
                store.connection.execute(
                    "UPDATE face_instances SET cluster_id = ? WHERE id = ?",
                    (target_cluster["id"], instance_id),
                )
            affected_clusters.add(target_cluster["id"])
        store.connection.execute(
            "DELETE FROM entity_mentions WHERE face_instance_id = ? AND entity_id != ?",
            (instance_id, entity["id"]),
        )
        store._link_confirmed_entity_mention(entities[name], item["observation_id"],
                                             instance_id, item["score"])
        affected_observations.add(item["observation_id"])
        linked += 1

    for cluster in affected_clusters:
        store._refresh_face_prototypes(cluster)

    face_to_name = {str(f["face_id"]): f["canonical_name"] for f in faces}
    for name, entity in entities.items():
        obs_ids = [item["observation_id"] for item in labels.values()
                   if face_to_name.get(item["face_id"]) == name]
        store._refresh_event_participants(obs_ids)
        store.rebuild_person_memory(entity["id"])
        try:
            store.resegment_events_for_confirmed_entity(entity["id"])
        except Exception as exc:
            print(f"[warn] resegment {name} failed: {exc}")

    # Refreshing participants is intentionally additive in the general store
    # API.  For an authorized reference mapping, however, a stale prior name
    # on the same observation is an error, so remove only participants whose
    # evidence is wholly within the reconciled observations.
    for observation_id in affected_observations:
        valid = {row["entity_id"] for row in store._rows(
            "SELECT DISTINCT entity_id FROM entity_mentions WHERE observation_id = ?",
            (observation_id,),
        )}
        event_rows = store._rows(
            "SELECT event_id FROM event_observations WHERE observation_id = ?",
            (observation_id,),
        )
        for event_row in event_rows:
            event_id = event_row["event_id"]
            for participant in store._rows(
                "SELECT id, person_id, evidence_ids_json FROM event_participants WHERE event_id = ?",
                (event_id,),
            ):
                try:
                    evidence_ids = set(json.loads(participant.get("evidence_ids_json") or "[]"))
                except (TypeError, ValueError):
                    evidence_ids = set()
                if observation_id in evidence_ids and evidence_ids.issubset(affected_observations) and participant["person_id"] not in valid:
                    store.connection.execute("DELETE FROM event_participants WHERE id = ?", (participant["id"],))
            event = store.get_event(event_id) or {}
            participants = [
                {"entity_id": row["person_id"], "name": (store.get_entity(row["person_id"]) or {}).get("canonical_name", ""), "status": "confirmed"}
                for row in store._rows("SELECT person_id FROM event_participants WHERE event_id = ?", (event_id,))
            ]
            store.connection.execute(
                "UPDATE events SET participants_json = ?, revision = revision + 1, updated_at = ? WHERE id = ?",
                (json.dumps(participants, ensure_ascii=False), now_iso(), event_id),
            )
    # Final idempotent pass after event/person projections: those projections
    # are allowed to merge events, but they must never undo the user-authorized
    # face-to-entity binding above.
    for instance_id, item in labels.items():
        info = face_info.get(item["face_id"])
        entity = entities.get(info.get("canonical_name")) if info else None
        if not entity:
            continue
        target_cluster = store._row(
            "SELECT id FROM face_clusters WHERE scope_id = ? AND entity_id = ? AND status = 'confirmed' ORDER BY updated_at DESC LIMIT 1",
            (args.scope, entity["id"]),
        )
        if not target_cluster:
            continue
        store.connection.execute(
            "UPDATE face_instances SET cluster_id = ? WHERE id = ?",
            (target_cluster["id"], instance_id),
        )
        store.connection.execute(
            "DELETE FROM entity_mentions WHERE face_instance_id = ? AND entity_id != ?",
            (instance_id, entity["id"]),
        )
        store._link_confirmed_entity_mention(entity, item["observation_id"], instance_id, item["score"])
        store._refresh_face_prototypes(target_cluster["id"])
    store.connection.commit()
    store.connection.commit()

    print(f"APPLIED: 绑定 {linked} 个实例 -> {len(entities)} 个已确认人物")
    for name, entity in entities.items():
        print(f"  {name} entity={entity['id']}")
    store.close()


if __name__ == "__main__":
    main()
