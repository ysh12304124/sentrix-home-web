#!/usr/bin/env python3
"""Publish one fresh v2 package into a dedicated, page-only memory space.

This does not run the old video pipeline and does not touch old scopes.  It
only registers the already-produced WebP representatives and their semantic
event spans so the timeline can display the fresh package without mixing it
with legacy QA imports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from backend.db import MemoryStore, make_id


def read(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default


def labels(values):
    result = []
    for item in values or []:
        value = item.get("label") or item.get("name") if isinstance(item, dict) else item
        value = str(value or "").strip()
        if value and value not in result:
            result.append(value)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    store = MemoryStore(str(args.db))
    store.create_memory_space(args.scope, args.label, kind="benchmark")
    frame_map = read(args.package / "frame_map.json", {})
    semantic_payload = read(args.package / "semantic.json", {})
    frames = list(frame_map.get("frames") or [])
    semantic_frames = list(semantic_payload.get("frames") or []) if isinstance(semantic_payload, dict) else []
    events = {str(row.get("event_id")): row for row in (semantic_payload.get("events") or [])} if isinstance(semantic_payload, dict) else {}
    semantic_by_index = {int(row.get("encoded_frame_index")): row for row in semantic_frames if str(row.get("encoded_frame_index", "")).lstrip("-").isdigit()}
    descriptions = read(args.package.parent.parent / "descriptions" / f"{args.video.stem}.json", [])
    descriptions = {str(row.get("event_id")): row for row in descriptions if isinstance(row, dict) and row.get("event_id")}

    parent_id = f"asset_latest_{args.video.stem}"
    parent = store.get_asset(parent_id)
    if not parent:
        parent = store.create_asset(parent_id, args.video.name, "video", str(args.video), "video/mp4", args.video.stat().st_size, {"scope_id": args.scope, "captured_at": datetime.fromtimestamp(args.video.stat().st_mtime, timezone.utc).isoformat(), "content_sha256": hashlib.sha256(args.video.name.encode()).hexdigest()}, scope_id=args.scope)
        store.update_asset(parent_id, "processed", {"video_stage": "processed", "keyframe_video_memory": True, "keyframe_video_memory_format": "webp", "keyframe_video_memory_frame_count": len(frames)})
    base_time = parent.get("captured_at") or datetime.fromtimestamp(args.video.stat().st_mtime, timezone.utc).isoformat()

    created_events = 0
    created_frames = 0
    for index, frame in enumerate(frames):
        encoded = int(frame.get("encoded_frame_index", index))
        info = semantic_by_index.get(encoded, {})
        event_id = str(info.get("event_id") or frame.get("event_id") or f"event_{encoded:05d}")
        event = events.get(event_id, {})
        image = args.package / "webp" / f"event_{event_id}.webp"
        if not image.is_file():
            candidates = sorted((args.package / "webp").glob(f"*{event_id}*.webp"))
            image = candidates[0] if candidates else Path("")
        if not image.is_file():
            continue
        source_sec = float(frame.get("source_timestamp_sec", 0) or 0)
        start_sec = float(event.get("start_sec", frame.get("event_start_sec", source_sec)) or 0)
        end_sec = float(event.get("end_sec", frame.get("event_end_sec", source_sec)) or 0)
        captured = datetime.fromisoformat(base_time.replace("Z", "+00:00"))
        start_iso = (captured + __import__("datetime").timedelta(seconds=start_sec)).isoformat()
        end_iso = (captured + __import__("datetime").timedelta(seconds=end_sec)).isoformat()
        objects = labels(info.get("objects"))
        actions = labels(info.get("actions"))
        description = descriptions.get(event_id, {})
        objects = list(dict.fromkeys(objects + [str(value) for value in description.get("objects") or []]))
        actions = list(dict.fromkeys(actions + [str(value) for value in description.get("actions") or []]))
        caption = str(description.get("caption") or "").strip()
        semantic_label = str(event.get("label") or frame.get("event_label") or "场景")
        summary = "；".join(filter(None, [caption, semantic_label, ", ".join(objects), ", ".join(actions)]))
        child_id = f"asset_latest_{args.video.stem}_{encoded:05d}"
        if not store.get_asset(child_id):
            store.create_asset(child_id, image.name, "image", str(image), "image/webp", image.stat().st_size, {"scope_id": args.scope, "parent_asset_id": parent_id, "derived_kind": "video_keyframe", "source_timestamp_sec": source_sec, "source_frame_index": int(frame.get("source_frame_index", 0) or 0), "source_scene_index": encoded, "captured_at": (captured + __import__("datetime").timedelta(seconds=source_sec)).isoformat(), "content_sha256": hashlib.sha256(image.read_bytes()).hexdigest(), "webp": True, "method": "yolo_prefilter_nvdec_targeted_katna_webp_v2"}, scope_id=args.scope)
            store.update_asset(child_id, "processed", {})
            created_frames += 1
        page_event = store.create_video_scene_event({"scope_id": args.scope, "title": f"{args.video.stem} · 场景：{semantic_label}", "summary": summary or f"{start_sec:.1f}s~{end_sec:.1f}s", "time_start": start_iso, "time_end": end_iso, "place": "其他或不确定", "source_asset_id": parent_id, "source_scene_index": encoded, "source_start_sec": start_sec, "source_end_sec": end_sec, "source_metadata": {"method": "yolo_prefilter_nvdec_targeted_katna_webp_v2", "memory_image_format": "webp", "semantic_substitution": True, "objects": objects, "actions": actions, "image_path": str(image)}})
        if page_event and not store._rows("SELECT observation_id FROM event_observations WHERE event_id=?", (page_event["id"],)):
            observation = store.add_observation(child_id, {"captured_at": (captured + __import__("datetime").timedelta(seconds=source_sec)).isoformat(), "source_type": "image", "caption": caption or summary, "activity": semantic_label, "objects": objects, "event_type": "视频场景", "confidence": 1.0}, scope_id=args.scope)
            store.attach_observation_to_event(page_event["id"], observation["id"])
        created_events += 1
    print(json.dumps({"scope": args.scope, "video": args.video.stem, "events": created_events, "frames": created_frames}, ensure_ascii=False))


if __name__ == "__main__":
    main()
