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


def _similar(left, right):
    a, b = _labels(left), _labels(right)
    if not a or not b:
        return False
    # Sustained meals are one semantic state even when the detector changes
    # between bowl/plate/food.  Conversations can also span long intervals;
    # visual diversity is preserved later by _choose_representatives().
    if _is_meal(left) and _is_meal(right):
        return True
    if _is_conversation(left) and _is_conversation(right) and "person" in a and "person" in b:
        return True
    # General long-form rule: repeated people in the same visual environment
    # may be split into many short detector spans, but belong to one memory
    # event. Representative selection below still keeps different views.
    if "person" in a and "person" in b and _same_background(left, right):
        return True
    overlap = len(a & b) / max(1, len(a | b))
    anchor = bool(a & b) and (("person" in a and "person" in b) or len(a & b) >= 2)
    return overlap >= 0.45 and anchor


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


def _choose_representatives(valid, group):
    """Keep the most informative frame plus visually different evidence.

    A sustained meal is intentionally represented by one frame.  For other
    long events, the cap grows with duration and selected frames must be both
    semantically informative and visually different.
    """
    if not valid:
        return []
    if all(_is_meal(row) for row in group if row):
        return [max(valid, key=_info_score)]
    start = float(group[0].get("event_start_sec", group[0].get("source_timestamp_sec", 0)) or 0)
    end = float(group[-1].get("event_end_sec", group[-1].get("source_timestamp_sec", start)) or start)
    duration = max(0.0, end - start)
    max_count = min(8, max(3, int(np.ceil(duration / 30.0))))
    min_gap = max(2.0, min(12.0, duration / max_count / 2.0))
    ranked = sorted(valid, key=_info_score, reverse=True)
    chosen = []
    for row in ranked:
        timestamp = float(row.get("source_timestamp_sec", row.get("event_start_sec", 0)) or 0)
        if any(abs(timestamp - float(item.get("source_timestamp_sec", item.get("event_start_sec", 0)) or 0)) < min_gap for item in chosen):
            continue
        path = _valid_image(row)
        if path is None or all(_visual_distance(path, _valid_image(item)) >= 0.08 for item in chosen):
            chosen.append(row)
        if len(chosen) >= max_count:
            break
    if not chosen:
        chosen = [max(valid, key=_info_score)]
    best = max(chosen, key=_info_score)
    ordered = sorted(chosen, key=lambda row: float(row.get("source_timestamp_sec", row.get("event_start_sec", 0)) or 0))
    return [best] + [row for row in ordered if row is not best]


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
    merged = _merge_frames(frames)
    manifest = {
        "method": "yolo10_batch_targeted_katna_nvdec_webp_v2",
        "keyframe_extraction_untouched_by_memory_merge": True,
        "source_frame_count": len(frames), "merged_event_count": len(merged),
        "events_merged_away": len(frames) - len(merged),
        "duplicate_frames_removed": sum(item["duplicate_frame_count"] + item.get("visual_duplicate_count", 0) for item in merged),
        "missing_representative_images": sum(not _valid_image(item["representative"]) for item in merged),
        "image_integrity_passed": all(_valid_image(item["representative"]) for item in merged),
        "source_stats": str(output / "stats.json"),
        "groups": [{key: value for key, value in item.items() if key != "representative"} for item in merged],
    }
    (output / "memory_merge.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return frames, merged, manifest
