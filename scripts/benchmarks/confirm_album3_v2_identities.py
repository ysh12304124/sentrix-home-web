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

from backend.db import MemoryStore
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
        store._link_confirmed_entity_mention(entities[name], item["observation_id"],
                                             instance_id, item["score"])
        linked += 1

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

    print(f"APPLIED: 绑定 {linked} 个实例 -> {len(entities)} 个已确认人物")
    for name, entity in entities.items():
        print(f"  {name} entity={entity['id']}")
    store.close()


if __name__ == "__main__":
    main()
