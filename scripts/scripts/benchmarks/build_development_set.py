#!/usr/bin/env python3
"""Build the independent Development Set from real DB content (Phase R8-2).

Generates 30-50 synthetic cases — NOT copies of the benchmark query.json — that
cover the twelve required categories and resolve their own Ground Truth from the
current DB:

  视觉对象 / 颜色材质 / 活动 / 场景 / 地点 / 时间 / 人物 / 纯短语 /
  复合查询 / 严格空结果 / 允许近似 / all_relevant

The Development Set is allowed to be seen by the implementing agent (unlike the
Hidden Set): it exists so fusion weights / probe thresholds / approximate gates
are calibrated on *general* data rather than the 44 regression cases.

Queries are phrased as user-style variations of the stored fields (not exact
copies) so the set measures query->memory generalization, not self-retrieval.

Output: docs/baseline/development_set.json
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

TARGET = 48


def _load_store():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from backend.db import MemoryStore
    return MemoryStore(os.getenv("SENTRIX_DB_PATH", "data/sentrix.db"))


def _field_list(obs, parsed_key, raw_key):
    values = obs.get(parsed_key)
    if isinstance(values, list):
        return [str(item) for item in values if str(item or "").strip()]
    return _parse_json_list(obs.get(raw_key))


def _obs_pool(store):
    pool = []
    for obs in store.list_observations(limit=100_000):
        pool.append({
            "id": obs.get("id"),
            "asset_id": obs.get("asset_id"),
            "scope_id": obs.get("scope_id"),
            "caption": obs.get("caption"),
            "activity": obs.get("activity"),
            "place": obs.get("place"),
            "objects": _field_list(obs, "objects", "objects_json"),
            "clothing": _field_list(obs, "clothing", "clothing_json"),
            "people": _field_list(obs, "people", "people_json"),
            "captured_at": obs.get("captured_at"),
            "ocr": (obs.get("ocr_text") or "")[:80],
        })
    return [row for row in pool if row["asset_id"]]


def _parse_json_list(raw):
    try:
        data = json.loads(raw or "[]")
    except (TypeError, ValueError):
        return []
    return [str(item) for item in data if str(item or "").strip()]


def _assets_with_value(store, field, value):
    """All asset ids whose observation field equals value (exact GT)."""
    ids = set()
    for obs in store.list_observations(limit=100_000):
        if field == "activity" and obs.get("activity") == value:
            ids.add(obs.get("asset_id"))
        elif field == "place" and obs.get("place") == value:
            ids.add(obs.get("asset_id"))
        elif field == "people":
            people = obs.get("people")
            if isinstance(people, list):
                if value in people:
                    ids.add(obs.get("asset_id"))
            elif value in _parse_json_list(obs.get("people_json")):
                ids.add(obs.get("asset_id"))
    return sorted(ids)


def _month_year(captured):
    try:
        dt = datetime.fromisoformat(str(captured).replace("Z", "+00:00"))
        return dt.year, dt.month
    except (TypeError, ValueError):
        return None, None


def _pick(pool, key, rng):
    candidates = [row for row in pool if row.get(key)]
    return rng.choice(candidates) if candidates else None


def _distinct_values(pool, key):
    """Distinct non-empty values of a field, stable order."""
    seen = []
    for row in pool:
        value = row.get(key)
        if value and value not in seen:
            seen.append(value)
    return seen


def build(store, target=TARGET, seed=20260806):
    rng = random.Random(seed)
    pool = _obs_pool(store)
    cases = []

    # ---- activity (with paraphrase) ----
    for activity in _distinct_values(pool, "activity")[:6]:
        if len(cases) >= target:
            break
        paraphrase = activity
        if activity.endswith("饭"):
            paraphrase = activity[:-1] + "餐"
        elif len(activity) >= 3:
            paraphrase = activity[:2]
        cases.append({"key": f"dev_a{len(cases):02d}", "category": "activity",
                      "query_cn": f"{paraphrase}的照片", "exact": _assets_with_value(store, "activity", activity),
                      "empty_policy": "strict_empty", "note": f"activity paraphrase {activity}->{paraphrase}"})

    # ---- place ----
    for place in _distinct_values(pool, "place")[:6]:
        if len(cases) >= target:
            break
        cases.append({"key": f"dev_p{len(cases):02d}", "category": "place",
                      "query_cn": f"在{place}的照片", "exact": _assets_with_value(store, "place", place),
                      "empty_policy": "strict_empty", "note": f"place {place}"})

    # ---- people ----
    person_seen = set()
    for row in pool:
        if len(cases) >= target or len(person_seen) >= 5:
            break
        people = row.get("people") or []
        if not people:
            continue
        person = people[0]
        if person in person_seen or person in {"未知", "无"}:
            continue
        person_seen.add(person)
        cases.append({"key": f"dev_m{len(cases):02d}", "category": "person",
                      "query_cn": f"{person}的照片", "exact": _assets_with_value(store, "people", person),
                      "empty_policy": "strict_empty", "note": f"person {person}"})

    # ---- objects (visual object) ----
    object_seen = set()
    for row in pool:
        if len(cases) >= target or len(object_seen) >= 5:
            break
        objects = row.get("objects") or []
        if not objects:
            continue
        obj = objects[0]
        if obj in object_seen:
            continue
        object_seen.add(obj)
        cases.append({"key": f"dev_o{len(cases):02d}", "category": "visual_object",
                      "query_cn": f"找{obj}的照片", "exact": [row["asset_id"]],
                      "empty_policy": "allow_approximate", "note": f"object {obj}"})

    # ---- color/material (from caption/objects) ----
    color_pool = [row for row in pool if any("色" in item or "毛绒" in item or "绒" in item for item in row["objects"] or [str(row["caption"] or "")])]
    color_pool = color_pool[:6]
    for i, row in enumerate(color_pool[:4]):
        words = [item for item in row["objects"] if "色" in item or "绒" in item]
        if not words:
            continue
        cases.append({"key": f"dev_c{len(cases):02d}", "category": "color_material",
                      "query_cn": f"{words[0]}的物件", "exact": [row["asset_id"]],
                      "empty_policy": "allow_approximate", "note": f"color {words[0]}"})

    # ---- scene (from caption) ----
    scene_pool = [row for row in pool if row.get("caption") and any(tok in row["caption"] for tok in ("海边", "公园", "厨房", "阳台", "街道", "山顶", "车内"))]
    for i, row in enumerate(scene_pool[:3]):
        token = next(t for t in ("海边", "公园", "厨房", "阳台", "街道", "山顶", "车内") if t in row["caption"])
        cases.append({"key": f"dev_s{len(cases):02d}", "category": "scene",
                      "query_cn": f"{token}场景的照片", "exact": [row["asset_id"]],
                      "empty_policy": "allow_approximate", "note": f"scene {token}"})

    # ---- time ----
    time_seen = set()
    for row in pool:
        if len(cases) >= target or len(time_seen) >= 5:
            break
        year, month = _month_year(row.get("captured_at"))
        if not year or (year, month) in time_seen:
            continue
        time_seen.add((year, month))
        cases.append({"key": f"dev_t{len(cases):02d}", "category": "time",
                      "query_cn": f"{year} 年 {month} 月的照片",
                      "exact": [r["asset_id"] for r in pool if _month_year(r.get("captured_at")) == (year, month)],
                      "empty_policy": "strict_empty", "note": f"{year}-{month}"})

    # ---- bare phrase ----
    for i, row in enumerate(pool[:2]):
        obj = (row["objects"] or ["这个"])[0]
        cases.append({"key": f"dev_b{len(cases):02d}", "category": "bare_phrase",
                      "query_cn": obj, "exact": [row["asset_id"]],
                      "empty_policy": "allow_approximate", "note": f"bare {obj}"})

    # ---- composite (person + place + activity) ----
    comp_rows = [row for row in pool if row.get("people") and row.get("place") and row.get("activity")]
    for i, row in enumerate(comp_rows[:2]):
        person = row["people"][0]
        cases.append({"key": f"dev_x{len(cases):02d}", "category": "composite",
                      "query_cn": f"{person}在{row['place']}{row['activity']}的照片",
                      "exact": [row["asset_id"]], "empty_policy": "strict_empty",
                      "note": "person+place+activity"})

    # ---- strict empty (query for something not in DB) ----
    synthetic = ("红色直升机", "蓝色钢琴", "紫色长颈鹿", "金色雪橇")
    for i, phrase in enumerate(synthetic):
        cases.append({"key": f"dev_e{len(cases):02d}", "category": "strict_empty",
                      "query_cn": f"找{phrase}", "exact": [], "empty_policy": "strict_empty",
                      "note": f"synthetic absent {phrase}"})

    # ---- allow approximate (related but not exact) ----
    approx_rows = [row for row in pool if row.get("activity")]
    for i, row in enumerate(approx_rows[:2]):
        activity = row["activity"]
        cases.append({"key": f"dev_q{len(cases):02d}", "category": "allow_approximate",
                      "query_cn": f"{activity}的场景", "exact": [],
                      "acceptable_approximate": [row["asset_id"]], "empty_policy": "allow_approximate",
                      "note": "near activity"})

    # ---- all_relevant ----
    distinct_activities = _distinct_values(pool, "activity")
    if distinct_activities:
        relevant_activity = distinct_activities[0]
        cases.append({"key": f"dev_r{len(cases):02d}", "category": "all_relevant",
                      "query_cn": f"所有{relevant_activity}的照片", "exact": _assets_with_value(store, "activity", relevant_activity),
                      "all_relevant": True, "empty_policy": "strict_empty", "note": f"all {relevant_activity}"})

    cases = cases[:target]
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=TARGET)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--report", default="docs/baseline/development_set.json")
    args = parser.parse_args()

    store = _load_store()
    cases = build(store, target=args.target, seed=args.seed)
    store.close()

    category_counts = {}
    for case in cases:
        category_counts[case["category"]] = category_counts.get(case["category"], 0) + 1
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(
        {"schema_version": 1, "seed": args.seed, "total": len(cases),
         "category_counts": category_counts, "cases": cases},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.report} ({len(cases)} cases)")
    print(json.dumps(category_counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
