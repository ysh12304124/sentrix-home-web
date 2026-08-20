from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def _value(item, key, default=""):
    return item.get(key, default) if isinstance(item, dict) else default


def _label_strings(values):
    return list(dict.fromkeys(
        str(value).strip() for value in values if value is not None and str(value).strip()
    ))


def _labels(row):
    values = []
    for key in ("objects", "actions", "expressions"):
        for item in row.get(key) or []:
            label = _value(item, "label") or _value(item, "name") or (item if isinstance(item, str) else "")
            if str(label).strip():
                values.append(str(label).strip().lower())
    event_label = str(row.get("event_label") or "").strip().lower()
    if event_label:
        values.append(event_label)
    return set(values)


_MEAL_LABELS = {
    "eat", "eating", "dining", "meal", "food", "吃饭", "用餐", "进食", "就餐",
    "吃东西", "吃饭场景",
}
_MEAL_OBJECTS = {
    "dining table", "bowl", "plate", "pizza", "donut", "food", "fork", "knife",
    "cup", "orange", "sandwich", "cake", "bottle",
}
_CONVERSATION_LABELS = {
    "talk", "talking", "speak", "speaking", "conversation", "chat", "discussion",
    "交谈", "说话", "聊天", "对话", "交流",
}
_LOW_INFO_ACTIONS = {
    "standing", "sitting", "raising hand", "stand", "sit", "walking", "走路",
    "站立", "坐着", "抬手", "挥手", "举手",
}
_STATIC_CONTEXT_OBJECTS = {"person", "chair", "couch", "bed", "potted plant", "vase"}


def _is_meal(row):
    labels = _labels(row)
    if labels & _MEAL_LABELS:
        return True
    # The detector often reports the tableware/food but not the verb "eating".
    # Two meal-context objects plus a person is a safer sustained-meal signal
    # than treating every person-only conversation as a meal.
    return "person" in labels and len(labels & _MEAL_OBJECTS) >= 2


def _is_conversation(row):
    labels = _labels(row)
    return bool(labels & _CONVERSATION_LABELS) or ("person" in labels and any(
        token in str(row.get("event_label") or "").lower() for token in _CONVERSATION_LABELS
    ))


def _meaningful_labels(row):
    labels = _labels(row)
    return labels - _LOW_INFO_ACTIONS


def _person_signature(row):
    people = []
    for item in row.get("objects") or []:
        label = str(_value(item, "label") or "").lower()
        if label != "person":
            continue
        bbox = _value(item, "bbox", []) or []
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(value) for value in bbox]
        people.append(((x1 + x2) / 2, (y1 + y2) / 2, max(0.0, x2 - x1), max(0.0, y2 - y1)))
    return sorted(people)


def _person_change(left, right):
    a, b = _person_signature(left), _person_signature(right)
    if not a or not b:
        return 1.0 if a != b else 0.0
    count_change = abs(len(a) - len(b)) / max(len(a), len(b), 1)
    paired = []
    for first, second in zip(a, b):
        position = min(1.0, float(np.linalg.norm(np.subtract(first[:2], second[:2]))) / 0.7)
        size = min(1.0, float(np.mean(np.abs(np.subtract(first[2:], second[2:])))) / 0.7)
        paired.append(0.7 * position + 0.3 * size)
    return min(1.0, 0.6 * count_change + 0.4 * (sum(paired) / max(1, len(paired))))


def _similar(left, right):
    a, b = _labels(left), _labels(right)
    if not a or not b:
        return False
    return _merge_compatible(left, right)


def _valid_image(row):
    path = Path(str(row.get("webp_path") or ""))
    return path if path.is_file() else None


def _info_score(row):
    labels = _labels(row)
    confidence = 0.0
    for key in ("objects", "actions", "expressions"):
        for item in row.get(key) or []:
            try:
                confidence += float(_value(item, "confidence", 0) or 0)
            except (TypeError, ValueError):
                pass
    # More independent semantic labels and confident detections are a better
    # proxy for information density than frame position.  WebP size is only a
    # small tie-breaker so it cannot select a noisy frame by itself.
    return 2.0 * len(labels) + confidence + min(1.0, float(row.get("webp_bytes") or 0) / 250000.0)


def _visual_signature(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    image = cv2.resize(image, (32, 18), interpolation=cv2.INTER_AREA).astype(np.float32)
    return (image - image.mean()) / max(float(image.std()), 1.0)


def _visual_distance(left, right):
    a = _visual_signature(left)
    b = _visual_signature(right)
    if a is None or b is None:
        return 1.0
    return float(np.mean(np.abs(a - b)))


def _background_signature(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    image = cv2.resize(image, (32, 18), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    # People and foreground actions change most in the centre. The border is
    # a lightweight fingerprint for conversations in one room.
    return np.concatenate([
        image[:4].reshape(-1), image[-4:].reshape(-1),
        image[:, :5].reshape(-1), image[:, -5:].reshape(-1),
    ])


def _same_background(left, right):
    left_path, right_path = _valid_image(left), _valid_image(right)
    if left_path is None or right_path is None:
        return False
    a, b = _background_signature(left_path), _background_signature(right_path)
    return a is not None and b is not None and float(np.mean(np.abs(a - b))) <= 0.18


def _background_relation(left, right):
    left_path, right_path = _valid_image(left), _valid_image(right)
    if left_path is None or right_path is None:
        return "environment_change"
    a, b = _background_signature(left_path), _background_signature(right_path)
    if a is None or b is None:
        return "environment_change"
    border_distance = float(np.mean(np.abs(a - b)))
    if border_distance > 0.30:
        return "environment_change"
    full_distance = _visual_distance(left_path, right_path)
    return "same_view" if full_distance < 0.45 else "same_environment_view_change"


def _merge_compatible(left, right):
    a, b = _labels(left), _labels(right)
    if not a or not b:
        return False
    relation = _background_relation(left, right)
    person_present = "person" in a and "person" in b
    person_delta = _person_change(left, right)
    meaningful_left = _meaningful_labels(left)
    meaningful_right = _meaningful_labels(right)
    overlap = len(meaningful_left & meaningful_right) / max(1, len(meaningful_left | meaningful_right))
    if relation == "environment_change":
        return False
    if person_present and person_delta <= 0.48:
        # Same view can carry a long conversation; a viewpoint change can too
        # when the meaningful subject remains stable. Low-value actions never
        # participate in this decision.
        return relation == "same_view" or overlap >= 0.12
    anchor = bool(meaningful_left & meaningful_right) and (person_present or len(meaningful_left & meaningful_right) >= 2)
    return anchor and overlap >= 0.35


def _choose_representatives(valid, group):
    """Keep exactly one highest-information frame for each merged event."""
    if not valid:
        return []
    return [max(valid, key=_info_score)]


def _merge_frames(frames, max_duration=300.0, max_gap=30.0):
    ordered = sorted(frames, key=lambda row: float(row.get("event_start_sec", row.get("source_timestamp_sec", 0)) or 0))
    groups = []
    for frame in ordered:
        current = groups[-1] if groups else None
        if current:
            previous = current[-1]
            previous_end = float(previous.get("event_end_sec", previous.get("source_timestamp_sec", 0)) or 0)
            current_start = float(current[0].get("event_start_sec", current[0].get("source_timestamp_sec", 0)) or 0)
            start = float(frame.get("event_start_sec", frame.get("source_timestamp_sec", 0)) or 0)
            duration = float(frame.get("event_end_sec", start) or start) - current_start
            if start - previous_end <= max_gap and duration <= max_duration and _similar(previous, frame):
                current.append(frame)
                continue
        groups.append([frame])

    result = []
    for index, group in enumerate(groups, 1):
        seen_hashes = set()
        valid = []
        duplicate_count = 0
        for row in group:
            path = _valid_image(row)
            if path is None:
                continue
            digest = hashlib.sha1(path.read_bytes()).hexdigest()
            if digest in seen_hashes:
                duplicate_count += 1
                continue
            seen_hashes.add(digest)
            valid.append(row)
        if not valid:
            continue
        representatives = _choose_representatives(valid, group)
        representative = representatives[0]
        objects = []
        actions = []
        expressions = []
        for row in group:
            for key, target in (("objects", objects), ("actions", actions), ("expressions", expressions)):
                for item in row.get(key) or []:
                    label = _value(item, "label") or _value(item, "name") or (item if isinstance(item, str) else "")
                    if str(label).strip() and str(label) not in target:
                        target.append(str(label))
        result.append({
            "event_id": f"merged_event_{index:05d}",
            "source_event_ids": [str(row.get("event_id") or "") for row in group],
            "start_sec": min(float(row.get("event_start_sec", row.get("source_timestamp_sec", 0)) or 0) for row in group),
            "end_sec": max(float(row.get("event_end_sec", row.get("source_timestamp_sec", 0)) or 0) for row in group),
            "representative": representative,
            "representatives": representatives,
            "objects": objects,
            "actions": actions,
            "expressions": expressions,
            "source_frame_count": len(group),
            "duplicate_frame_count": duplicate_count,
            "visual_duplicate_count": max(0, len(valid) - len(representatives)),
            "memory_keyframe_count": len(representatives),
        })
    return result


def run(video_path, output_dir, video_id):
    root = Path(__file__).resolve().parents[2] / "tools" / "video_keyframe"
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    script = root / "katna" / "run_yolo_prefilter_event_webp.py"
    yolo = Path(os.getenv("SENTRIX_VIDEO_YOLO_MODEL", root / "models/keyframe/yolo11n.pt"))
    pose = Path(os.getenv("SENTRIX_VIDEO_POSE_MODEL", root / "models/keyframe/yolo11n-pose.pt"))
    command = [
        os.getenv("SENTRIX_VIDEO_PYTHON", sys.executable), str(script),
        "--video", str(Path(video_path).resolve()), "--output", str(output), "--video-id", str(video_id),
        "--katna-root", str(root / "katna"), "--pipeline-root", str(root),
        "--yolo-model", str(yolo), "--pose-model", str(pose),
        "--scan-fps", os.getenv("SENTRIX_VIDEO_SAMPLE_FPS", "10"),
        "--yolo-batch-size", os.getenv("SENTRIX_VIDEO_YOLO_BATCH_SIZE", "16"),
        "--target-decode-workers", os.getenv("SENTRIX_VIDEO_TARGET_DECODE_WORKERS", "4"),
        "--merge-max-sec", os.getenv("SENTRIX_VIDEO_PREFILTER_MERGE_MAX_SEC", "12"),
        "--device", os.getenv("SENTRIX_VIDEO_DEVICE", "0"),
    ]
    process = subprocess.run(command, check=False, capture_output=True, text=True,
                             timeout=int(os.getenv("SENTRIX_VIDEO_TIMEOUT_SECONDS", "7200")))
    (output / "sentrix-hybrid.log").write_text(
        process.stdout + ("\nSTDERR\n" + process.stderr if process.stderr else ""), encoding="utf-8")
    if process.returncode:
        raise RuntimeError(f"hybrid WebP keyframe extraction failed ({process.returncode}): {process.stderr[-2000:]}")
    frame_map = json.loads((output / "frame_map.json").read_text(encoding="utf-8"))
    frames = list(frame_map.get("frames") or [])
    for row in frames:
        if not row.get("webp_path"):
            event_id = row.get("event_id")
            row["webp_path"] = str(output / "webp" / f"event_{event_id}.webp")
    # The extractor has already merged source-order events before NVDEC and
    # writes exactly one WebP for each final event.  Do not create a second,
    # image-level merge pass here.
    merged = []
    for row in frames:
        objects = _label_strings(row.get("event_objects") or [
            _value(item, "label") for item in row.get("objects") or []
        ])
        actions = _label_strings(row.get("event_actions") or [
            _value(item, "label") for item in row.get("actions") or []
        ])
        expressions = _label_strings([
            _value(item, "label") for item in row.get("expressions") or []
        ])
        merged.append({
            "event_id": str(row.get("event_id") or ""),
            "source_event_ids": [str(row.get("event_id") or "")],
            "start_sec": float(row.get("event_start_sec", row.get("source_timestamp_sec", 0)) or 0),
            "end_sec": float(row.get("event_end_sec", row.get("source_timestamp_sec", 0)) or 0),
            "representative": row, "representatives": [row],
            "objects": objects, "actions": actions, "expressions": expressions,
            "yolo_timeline": list(row.get("yolo_timeline") or []),
            "source_frame_count": int(row.get("source_frame_count") or 1),
            "duplicate_frame_count": 0, "visual_duplicate_count": 0,
            "memory_keyframe_count": 1,
        })
    stats = json.loads((output / "stats.json").read_text(encoding="utf-8"))
    manifest = {
        "method": "yolo10_premerge_katna_nvdec_multivlm_adaptive_webp_v9",
        "event_merge_before_image_write": True,
        "keyframe_extraction_untouched_by_memory_merge": False,
        "source_frame_count": len(frames), "merged_event_count": len(merged),
        "preliminary_event_count": int(stats.get("preliminary_event_count") or len(merged)),
        "events_merged_away": max(0, int(stats.get("preliminary_event_count") or len(merged)) - len(merged)),
        "duplicate_frames_removed": sum(item["duplicate_frame_count"] + item.get("visual_duplicate_count", 0) for item in merged),
        "missing_representative_images": sum(not _valid_image(item["representative"]) for item in merged),
        "image_integrity_passed": all(_valid_image(item["representative"]) for item in merged),
        "source_stats": str(output / "stats.json"),
        "groups": [{key: value for key, value in item.items() if key != "representative"} for item in merged],
    }
    (output / "memory_merge.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return frames, merged, manifest
