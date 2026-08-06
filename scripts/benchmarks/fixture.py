#!/usr/bin/env python3
"""Small offline fixture for benchmark script smoke tests.

Seeds a MemoryStore with a handful of synthetic assets/observations so the
benchmark runners can be exercised end-to-end without the real 153 database.
Synthetic data only — never benchmark GT filenames or real family data.

Runtime code (backend/*.py) must never import this module.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


def seed_fixture(store) -> list[dict]:
    """Insert a deterministic synthetic corpus; return created asset dicts."""
    now = _now()
    scope = "album1"
    rows = [
        # (file_name, captured_at, caption, activity, place, clothing, objects, ocr)
        ("IMG_DEMO_01.JPG", "2023-11-16T14:00:00+00:00", "卧室睡衣自拍", "自拍", None, ["毛绒睡衣"], [], ""),
        ("IMG_DEMO_02.JPG", "2024-05-03T18:30:00+00:00", "厨房做饭", "做晚饭", "厨房", ["围裙"], ["锅", "菜"], ""),
        ("IMG_DEMO_03.JPG", "2025-01-01T00:10:00+00:00", "跨年烟花", "看烟花", None, [], [], "新年快乐"),
        ("IMG_DEMO_04.JPG", "2024-12-31T23:50:00+00:00", "阳台", "聊天", None, [], [], ""),
        ("IMG_DEMO_05.JPG", "2023-06-01T09:00:00+00:00", "银手镯特写", "拍饰品", None, [], ["手镯"], ""),
    ]
    created = []
    for index, (file_name, captured, caption, activity, place, clothing, objects, ocr) in enumerate(rows):
        asset_id = f"asset_demo_{index + 1}"
        store.create_asset(asset_id, file_name, "image", f"/tmp/fixture/{file_name}", "image/jpeg", 1024, {
            "captured_at": captured, "scope_id": scope, "source_album_id": scope,
        })
        obs_id = f"obs_demo_{index + 1}"
        store.connection.execute(
            """INSERT INTO observations
               (id, scope_id, asset_id, captured_at, source_type, caption, activity, place,
                people_json, objects_json, ocr_text, event_type, transcript, confidence,
                raw_json, canonical_json, clothing_json, spatial_relations_json, revision, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (obs_id, scope, asset_id, captured, "vision", caption, activity, place,
             "[]", json.dumps(objects, ensure_ascii=False), ocr, "daily", None, 0.9,
             "{}", "{}", json.dumps(clothing, ensure_ascii=False), "[]", 1, now, now),
        )
        store.connection.execute(
            """INSERT INTO memory_vectors
               (id, scope_id, space, source_type, source_id, vector_json, model_name, metadata_json, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (f"vec_demo_{index + 1}", scope, "visual", "asset", asset_id, json.dumps([0.1 * (index + 1)] * 3), "ViT-B-32", "{}", now, now),
        )
        created.append({"asset_id": asset_id, "file_name": file_name, "observation_id": obs_id})
    store.connection.commit()
    return created
