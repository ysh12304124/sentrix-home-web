"""Import and project a precomputed keyframe-video package.

The package keeps the selected full-resolution frames in one video instead of
writing one JPEG per frame.  ``frame_map.json`` maps the compact video frame
index back to the source timestamp and ``semantic.json`` contains the YOLO /
pose result for each selected frame.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default
    except (OSError, ValueError, TypeError):
        return default


def load_keyframe_package(package: dict | None):
    """Load and validate package sidecars, returning a compact normalized view."""
    package = package if isinstance(package, dict) else {}
    map_path = Path(str(package.get("frame_map_path") or ""))
    semantic_path = Path(str(package.get("semantic_path") or ""))
    if not map_path.is_file() or not semantic_path.is_file():
        raise ValueError("keyframe video package requires frame_map.json and semantic.json")
    frame_map = _read_json(map_path, {})
    frames = frame_map.get("frames") if isinstance(frame_map, dict) else frame_map
    semantics = _read_json(semantic_path, [])
    semantic_events = list(semantics.get("events") or []) if isinstance(semantics, dict) else []
    if not isinstance(frames, list) or not frames:
        raise ValueError("frame_map.json has no frames")
    if not isinstance(semantics, list):
        semantics = semantics.get("frames") or [] if isinstance(semantics, dict) else []
    semantic_by_index = {
        int(item.get("encoded_frame_index")): item
        for item in semantics
        if isinstance(item, dict) and str(item.get("encoded_frame_index", "")).lstrip("-").isdigit()
    }
    normalized = []
    for ordinal, raw in enumerate(frames):
        if not isinstance(raw, dict):
            continue
        try:
            encoded_index = int(raw.get("encoded_frame_index", ordinal))
            timestamp = float(raw.get("source_timestamp_sec", raw.get("timestamp_sec", 0)) or 0)
            source_index = int(raw.get("source_frame_index", 0) or 0)
        except (TypeError, ValueError):
            continue
        semantic = semantic_by_index.get(encoded_index, {})
        normalized.append({
            "encoded_frame_index": encoded_index,
            "source_frame_index": source_index,
            "source_timestamp_sec": timestamp,
            "objects": list(semantic.get("objects") or []),
            "actions": list(semantic.get("actions") or []),
            "expressions": list(semantic.get("expressions") or []),
            "event_id": semantic.get("event_id") or raw.get("event_id"),
            "event_start_sec": semantic.get("event_start_sec", raw.get("event_start_sec")),
            "event_end_sec": semantic.get("event_end_sec", raw.get("event_end_sec")),
            "event_kind": semantic.get("event_kind", raw.get("event_kind")),
            "event_label": semantic.get("event_label", raw.get("event_label")),
        })
    if not normalized:
        raise ValueError("frame_map.json has no valid frame entries")
    normalized.sort(key=lambda item: item["encoded_frame_index"])
    return {
        "frames": normalized,
        "encoded_fps": float(package.get("encoded_fps") or frame_map.get("encoded_fps") or 25),
        "source_fps": float(package.get("source_fps") or frame_map.get("source_fps") or 0),
        "source_width": int(package.get("source_width") or frame_map.get("source_width") or 0),
        "source_height": int(package.get("source_height") or frame_map.get("source_height") or 0),
        "direct_source": bool(package.get("direct_source") or frame_map.get("direct_source")),
        "frame_map_path": str(map_path),
        "semantic_path": str(semantic_path),
        "events": semantic_events,
    }


def aggregate_semantics(frames: list[dict]):
    object_counts = Counter()
    action_counts = Counter()
    expression_counts = Counter()
    object_confidence = {}
    for frame in frames:
        for item in frame.get("objects") or []:
            label = str(item.get("label") or item.get("name") or "").strip()
            if label:
                object_counts[label] += 1
                object_confidence[label] = max(object_confidence.get(label, 0.0), float(item.get("confidence") or 0))
        for item in frame.get("actions") or []:
            label = str(item.get("label") or item.get("name") or "").strip()
            if label:
                action_counts[label] += 1
        for item in frame.get("expressions") or []:
            label = str(item.get("label") or item.get("name") or "").strip()
            if label:
                expression_counts[label] += 1
    objects = [
        {"label": label, "count": count, "confidence": round(object_confidence.get(label, 0.0), 4), "source": "yolo"}
        for label, count in object_counts.most_common(40)
    ]
    actions = [{"label": label, "count": count, "source": "pose"} for label, count in action_counts.most_common(20)]
    expressions = [{"label": label, "count": count, "source": "expression"} for label, count in expression_counts.most_common(20)]
    labels = [item["label"] for item in objects + actions + expressions]
    return objects, actions, expressions, labels
