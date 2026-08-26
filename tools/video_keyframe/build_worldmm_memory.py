#!/usr/bin/env python3
"""Build WorldMM-style memories from the existing WebP keyframe package.

This deliberately does not run, modify, or re-extract keyframes.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import Counter
from pathlib import Path


def read_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--package", required=True, help="Existing hybrid-keyframes directory")
    parser.add_argument("--transcript", required=True, help="Whisper transcript.json")
    parser.add_argument("--descriptions", default="", help="Optional VLM descriptions keyed by package event_id")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    package = Path(args.package)
    frame_map = read_json(package / "frame_map.json", {})
    semantic = read_json(package / "semantic.json", [])
    semantics = semantic if isinstance(semantic, list) else list(semantic.get("frames") or [])
    semantic_by_index = {int(row.get("encoded_frame_index")): row for row in semantics if str(row.get("encoded_frame_index", "")).lstrip("-").isdigit()}
    frame_rows = list(frame_map.get("frames") or [])
    transcript = read_json(args.transcript, {})
    transcript_rows = transcript.get("segments") if isinstance(transcript, dict) else []
    descriptions = read_json(args.descriptions, []) if args.descriptions else []
    description_by_event = {str(row.get("event_id")): row for row in descriptions if isinstance(row, dict) and row.get("event_id")}
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    video = conn.execute("select id,path,file_name,metadata_json from assets where scope_id=? and media_type='video' order by created_at limit 1", (args.scope,)).fetchone()
    if not video:
        raise SystemExit(f"no video asset for scope {args.scope}")
    asset_meta = read_json(video["metadata_json"], {})
    observations = conn.execute("""
      select o.id,o.asset_id,o.caption,o.activity,o.place,o.transcript,o.objects_json,o.people_json,
             eo.event_id,e.title,e.summary,e.source_start_sec,e.source_end_sec
      from observations o join event_observations eo on eo.observation_id=o.id
      left join events e on e.id=eo.event_id where o.scope_id=? order by e.source_start_sec,o.captured_at
    """, (args.scope,)).fetchall()
    obs_by_asset = {row["asset_id"]: row for row in observations}
    event_rows = {}
    for row in observations:
        event_rows.setdefault(row["event_id"], row)
    db_events = sorted(event_rows.values(), key=lambda row: float(row["source_start_sec"] or 0))

    def json_labels(value):
        try:
            raw = json.loads(value or "[]")
        except (TypeError, ValueError):
            raw = []
        values = []
        for item in raw if isinstance(raw, list) else []:
            if isinstance(item, dict):
                item = item.get("label") or item.get("name") or item.get("primary") or ""
            if str(item).strip():
                values.append(str(item).strip())
        return values

    def db_event_at(timestamp):
        candidates = [row for row in db_events if row["source_start_sec"] is not None and row["source_end_sec"] is not None and float(row["source_start_sec"]) - 1.0 <= timestamp <= float(row["source_end_sec"]) + 1.0]
        return candidates[0] if candidates else None
    visual = []
    caption = []
    semantic_memory = []
    for index, frame in enumerate(frame_rows):
        encoded_index = int(frame.get("encoded_frame_index", index))
        info = semantic_by_index.get(encoded_index, {})
        event_id = info.get("event_id") or frame.get("event_id") or f"frame_{encoded_index:06d}"
        event = db_event_at(float(frame.get("source_timestamp_sec", 0) or 0))
        event_dict = dict(event) if event else {}
        image_path = str(package / "webp" / f"event_{event_id}.webp")
        # The package names files by event. Fall back to its event ordinal when
        # an old package used a different filename convention.
        if not Path(image_path).is_file():
            candidates = sorted((package / "webp").glob(f"*{event_id}*.webp"))
            image_path = str(candidates[0]) if candidates else ""
        objects = info.get("objects") or []
        actions = info.get("actions") or []
        def labels(values):
            return [str(x.get("label") or x.get("name") or x) if isinstance(x, dict) else str(x) for x in values]
        object_labels = labels(objects)
        action_labels = labels(actions)
        description = description_by_event.get(str(event_id), {})
        object_labels = list(dict.fromkeys(object_labels + [str(x) for x in description.get("objects") or []]))
        action_labels = list(dict.fromkeys(action_labels + [str(x) for x in description.get("actions") or []]))
        obs = obs_by_asset.get(info.get("asset_id"))
        if obs is None and event:
            obs = event
        if not object_labels and obs:
            object_labels = json_labels(obs["objects_json"])
        caption_text = "；".join(filter(None, [
            (obs["caption"] if obs else ""), (obs["activity"] if obs else ""),
            "、".join(object_labels), "、".join(action_labels), (event["summary"] if event else ""),
            description.get("caption") or description.get("text") or "",
        ]))
        start = float((event["source_start_sec"] if event and event["source_start_sec"] is not None else frame.get("source_timestamp_sec", 0)) or 0)
        end = float((event["source_end_sec"] if event and event["source_end_sec"] is not None else frame.get("source_timestamp_sec", 0)) or 0)
        row = {"id": f"visual_{encoded_index:06d}", "event_id": event_id, "timestamp_sec": float(frame.get("source_timestamp_sec", 0) or 0), "start_sec": start, "end_sec": end, "image_path": image_path, "caption": caption_text, "objects": object_labels, "actions": action_labels, "labels": object_labels + action_labels}
        visual.append(row)
        cap = {"id": f"caption_{event_id}", "event_id": event_id, "start_sec": start, "end_sec": end, "caption": caption_text, "objects": object_labels, "actions": action_labels, "image_path": image_path}
        caption.append(cap)
        for obj in dict.fromkeys(object_labels):
            semantic_memory.append({"id": f"sem_{event_id}_obj_{len(semantic_memory)}", "subject": "scene", "predicate": "contains", "object": obj, "event_id": event_id, "start_sec": start, "end_sec": end, "evidence_id": cap["id"]})
        for action in dict.fromkeys(action_labels):
            semantic_memory.append({"id": f"sem_{event_id}_act_{len(semantic_memory)}", "subject": "main character", "predicate": "performs", "object": action, "event_id": event_id, "start_sec": start, "end_sec": end, "evidence_id": cap["id"]})
    # Collapse duplicated visual records to one representative per event while
    # retaining all selected-frame timestamps in the manifest.
    visual_by_event = {}
    for row in visual:
        visual_by_event.setdefault(row["event_id"], row)
    visual = list(visual_by_event.values())
    episodic = []
    event_metadata = {item["event_id"]: db_event_at(float(item["start_sec"])) for item in caption}
    for event_id, matching in sorted(((item["event_id"], item) for item in caption), key=lambda pair: float(pair[1]["start_sec"] or 0)):
        if not matching:
            continue
        row = event_metadata.get(event_id)
        visual_id = next((item["id"] for item in visual if item["event_id"] == event_id), "")
        episodic.append({"id": f"episode_{event_id}", "event_id": event_id, "start_sec": matching["start_sec"], "end_sec": matching["end_sec"], "title": (row["title"] if row else "scene"), "summary": (row["summary"] if row else "") or matching["caption"], "caption_ids": [matching["id"]], "visual_ids": [visual_id], "objects": matching["objects"], "actions": matching["actions"]})
    audio = []
    for index, row in enumerate(transcript_rows or []):
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        audio.append({"id": f"audio_{index:04d}", "start_sec": float(row.get("start", 0) or 0), "end_sec": float(row.get("end", 0) or 0), "text": text, "source": "whisper-distil-large-v3.5"})
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": "sentrix-worldmm-memory-v1",
        "architecture": {"caption_memory": "time-aligned captions and audio transcript", "episodic_memory": "event/time episodes", "semantic_memory": "entity-action-object triples", "visual_memory": "WebP representative frames", "retrieval": "adaptive late fusion"},
        "video": {"asset_id": video["id"], "file_name": video["file_name"], "path": video["path"], "duration_sec": (asset_meta.get("video_metadata") or {}).get("duration_sec"), "source_width": frame_map.get("source_width"), "source_height": frame_map.get("source_height")},
        "caption_memory": caption, "audio_memory": audio, "episodic_memory": episodic, "semantic_memory": semantic_memory, "visual_memory": visual,
        "manifest": {"scope_id": args.scope, "package": str(package), "source_keyframe_count": len(frame_rows), "visual_representative_count": len(visual), "caption_count": len(caption), "episode_count": len(episodic), "semantic_triple_count": len(semantic_memory), "audio_segment_count": len(audio), "built_seconds": round(time.perf_counter() - started, 3), "keyframe_extraction_untouched": True},
    }
    (out / "memory.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(artifact["manifest"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(artifact["manifest"], ensure_ascii=False))


if __name__ == "__main__":
    main()
