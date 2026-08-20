from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def _value(item, key, default=""):
    return item.get(key, default) if isinstance(item, dict) else default


def _labels(row):
    values = []
    for key in ("objects", "actions", "expressions"):
        for item in row.get(key) or []:
            label = _value(item, "label") or _value(item, "name") or (item if isinstance(item, str) else "")
            if str(label).strip():
                values.append(str(label).strip().lower())
    return set(values)


def _similar(left, right):
    a, b = _labels(left), _labels(right)
    if not a or not b:
        return False
    overlap = len(a & b) / max(1, len(a | b))
    anchor = bool(a & b) and (("person" in a and "person" in b) or len(a & b) >= 2)
    return overlap >= 0.45 and anchor


def _valid_image(row):
    path = Path(str(row.get("webp_path") or ""))
    return path if path.is_file() else None


def _merge_frames(frames, max_duration=45.0, max_gap=0.75):
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
        representative = valid[0]
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
            "objects": objects,
            "actions": actions,
            "expressions": expressions,
            "source_frame_count": len(group),
            "duplicate_frame_count": duplicate_count,
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
        "duplicate_frames_removed": sum(item["duplicate_frame_count"] for item in merged),
        "missing_representative_images": sum(not _valid_image(item["representative"]) for item in merged),
        "image_integrity_passed": all(_valid_image(item["representative"]) for item in merged),
        "source_stats": str(output / "stats.json"),
        "groups": [{key: value for key, value in item.items() if key != "representative"} for item in merged],
    }
    (output / "memory_merge.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return frames, merged, manifest
