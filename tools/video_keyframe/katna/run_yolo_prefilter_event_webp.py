#!/usr/bin/env python3
"""YOLO/Pose prefilter -> event merge -> transient multi-frame VLM package.

The coarse pass samples the video without writing images.  Stable intervals are
represented by semantic spans and one anchor; only change windows are sent to
the Katna quality gate.  The output is a keyframe package consumed by the
normal WebP memory importer. Only one representative WebP is persisted.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def _value(item, key, default=None):
    return item.get(key, default) if isinstance(item, dict) else getattr(item, key, default)


def _labels(semantic):
    values = []
    for item in list(semantic.get("detections") or []) + list(semantic.get("actions") or []):
        label = str(_value(item, "label") or "").strip()
        if label:
            values.append(label)
    return tuple(sorted(set(values)))


_LOW_INFO_ACTIONS = {
    "standing", "sitting", "raising hand", "stand", "sit", "walking",
    "走路", "站立", "坐着", "抬手", "挥手", "举手",
}
_STATIC_MERGE_LABELS = {"person", "chair", "couch", "bed", "potted plant", "vase", "tv"}


def _meaningful_sample_labels(sample):
    return set(sample.get("labels") or ()) - _LOW_INFO_ACTIONS


def _person_count(sample):
    return sum(
        1 for item in sample.get("semantic", {}).get("detections") or []
        if str(_value(item, "label") or "").strip().lower() == "person"
    )


def _segment_background(segment):
    if not segment:
        return None
    anchors = [segment[0], segment[len(segment) // 2], segment[-1]]
    values = [item.get("background") for item in anchors if item.get("background") is not None]
    return np.median(np.stack(values), axis=0) if values else None


def _segment_visual(segment):
    if not segment:
        return None
    return segment[len(segment) // 2].get("visual")


def _memory_merge_compatible(left, right):
    left_background = _segment_background(left)
    right_background = _segment_background(right)
    if left_background is None or right_background is None:
        return False
    border_distance = float(np.mean(np.abs(left_background - right_background)))
    if border_distance > 0.30:
        return False
    left_visual, right_visual = _segment_visual(left), _segment_visual(right)
    full_distance = 1.0 if left_visual is None or right_visual is None else float(np.mean(np.abs(left_visual - right_visual)))
    same_view = full_distance < 0.22
    left_labels = set().union(*(_meaningful_sample_labels(item) for item in left))
    right_labels = set().union(*(_meaningful_sample_labels(item) for item in right))
    overlap = len(left_labels & right_labels) / max(1, len(left_labels | right_labels))
    left_people = int(np.median([_person_count(item) for item in left]))
    right_people = int(np.median([_person_count(item) for item in right]))
    repeated_people = left_people > 0 and right_people > 0 and abs(left_people - right_people) <= 1
    if repeated_people:
        return same_view or overlap >= 0.12
    return bool(left_labels & right_labels) and overlap >= 0.35


def merge_memory_segments(segments, max_duration_sec=300.0):
    """Merge in source order before any full-resolution image is written."""
    if not segments:
        return []
    merged = [list(segments[0])]
    for segment in segments[1:]:
        previous = merged[-1]
        duration = float(segment[-1]["timestamp"]) - float(previous[0]["timestamp"])
        if duration <= max_duration_sec and _memory_merge_compatible(previous, segment):
            previous.extend(segment)
        else:
            merged.append(list(segment))

    # A final short close-up can be a detail of the preceding memory (for
    # example a grill close-up after a poolside barbecue). Apply this once to
    # already-final groups so it cannot alter or propagate the main merge.
    result = []
    for group in merged:
        group_duration = float(group[-1]["timestamp"]) - float(group[0]["timestamp"])
        group_labels = set().union(*(_meaningful_sample_labels(item) for item in group)) - _STATIC_MERGE_LABELS
        if result and group_duration <= 6.0:
            previous_labels = set().union(*(_meaningful_sample_labels(item) for item in result[-1])) - _STATIC_MERGE_LABELS
            if previous_labels & group_labels:
                result[-1].extend(group)
                continue
        result.append(group)
    return result


def _signature(semantic):
    objects = tuple(sorted(str(_value(item, "label") or "") for item in semantic.get("detections") or []))
    actions = tuple(sorted(str(_value(item, "label") or "") for item in semantic.get("actions") or []))
    return (objects, actions)


def _sharpness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _tenengrad(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    return float((sx * sx + sy * sy).mean())


def _quality_key(candidate):
    # The old path used np.var(image), which measures contrast/colour spread,
    # not focus.  Rank by edge sharpness instead.
    return (_sharpness(candidate.image), _tenengrad(candidate.image))


def quality_select(candidates, brightness, entropy_score):
    """Unlimited selection with an explicit anti-motion-blur gate.

    Candidates are decoded at 384px wide.  A Laplacian variance of 150 is a
    conservative floor at that scale; if an event has no candidate above it,
    the event-level fallback still chooses its sharpest coarse sample.
    """
    min_sharpness = 150.0
    filtered = [candidate for candidate in candidates
                if 10.0 < brightness(candidate.image) < 90.0
                and 1.0 < entropy_score(candidate.image) < 10.0
                and _sharpness(candidate.image) >= min_sharpness]
    selected = sorted({item.frame_index: item for item in filtered}.values(), key=lambda item: item.timestamp)
    return selected, {
        "candidate_count": len(candidates), "filtered_count": len(filtered),
        "target_k": "unlimited", "brightness_range": [10.0, 90.0],
        "entropy_range": [1.0, 10.0], "min_laplacian_384": min_sharpness,
        "selection": "LUV+Hanning+brightness-entropy+Laplacian+Tenengrad-unlimited",
    }


def _predict_yolo_batch(analyzer, frames, confidence):
    if analyzer.detector is None:
        return [None] * len(frames)
    try:
        results = analyzer.detector.predict(
            source=frames,
            conf=confidence,
            iou=0.70,
            imgsz=max(frames[0].shape[:2]),
            device=analyzer.config.device,
            verbose=False,
        )
        return list(results)
    except Exception as exc:
        print(f"Ultralytics batch inference failed; disabling detector: {exc}", flush=True)
        analyzer.detector = None
        return [None] * len(frames)


def coarse_scan(video, analyzer, scan_fps, width, batch_size=16):
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    # ``round(23.976 / 10)`` becomes 2, which is actually ~12 FPS.  Ceiling
    # keeps the requested scan rate as an upper bound.
    stride = max(1, math.ceil(fps / max(scan_fps, 0.1)))
    samples = []
    index = -1
    source_width = 0
    source_height = 0
    previous_gray = None
    pending = []

    def flush_batch():
        nonlocal previous_gray
        if not pending:
            return
        results = _predict_yolo_batch(analyzer, [item["frame"] for item in pending], 0.30)
        for item, result in zip(pending, results):
            detections = analyzer._extract_detections(
                result, "yolo", item["frame"].shape[1], item["frame"].shape[0],
            )
            semantic = {"detections": detections, "actions": [], "expressions": []}
            change = float(cv2.absdiff(item["gray"], previous_gray).mean()) if previous_gray is not None else 0.0
            samples.append({
                "frame_index": item["frame_index"], "timestamp": item["timestamp"], "semantic": semantic,
                "labels": list(_labels(semantic)), "signature": _signature(semantic),
                "change": round(change, 4), "sharpness": round(_sharpness(item["frame"]), 4),
                "visual": item["visual"], "background": item["background"],
            })
            previous_gray = item["gray"]
        pending.clear()

    try:
        while True:
            ok, source = capture.read()
            if not ok:
                break
            index += 1
            if not source_width:
                source_height, source_width = source.shape[:2]
            if index % stride:
                continue
            frame = source
            if frame.shape[1] > width:
                scale = width / frame.shape[1]
                frame = cv2.resize(frame, (width, max(1, round(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)
            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (96, 54), interpolation=cv2.INTER_AREA)
            visual = cv2.resize(gray, (32, 18), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            background = np.concatenate([
                visual[:4].reshape(-1), visual[-4:].reshape(-1),
                visual[:, :5].reshape(-1), visual[:, -5:].reshape(-1),
            ])
            pending.append({
                "frame": frame, "gray": gray, "visual": visual, "background": background,
                "frame_index": index, "timestamp": index / fps,
            })
            if len(pending) >= max(1, int(batch_size)):
                flush_batch()
    finally:
        flush_batch()
        capture.release()
    if not samples:
        raise RuntimeError("YOLO prefilter produced no samples")

    segments = []
    start = 0
    min_event_sec = 4.0
    max_event_sec = 20.0
    for position in range(1, len(samples)):
        previous = samples[position - 1]
        current = samples[position]
        signature_changed = current["signature"] != previous["signature"]
        long_gap = current["timestamp"] - samples[start]["timestamp"] >= max_event_sec
        if (signature_changed and current["timestamp"] - samples[start]["timestamp"] >= min_event_sec) or long_gap:
            segments.append(samples[start:position])
            start = position
    segments.append(samples[start:])
    return fps, frame_count, samples, segments, source_width, source_height


def _windows(segments, fps):
    windows = []
    for segment in segments:
        first, last = segment[0], segment[-1]
        duration = last["timestamp"] - first["timestamp"]
        if duration > 12.0:
            anchors = [segment[int(i)] for i in np.linspace(0, len(segment) - 1, max(2, round(duration / 8.0))).astype(int)]
            for anchor in anchors:
                windows.append((max(0, anchor["frame_index"] - round(fps)), anchor["frame_index"] + round(fps)))
        else:
            windows.append((max(0, first["frame_index"] - round(fps)), last["frame_index"] + round(fps)))
    return windows


def _in_windows(index, windows):
    return any(start <= index <= end for start, end in windows)


def _segment_label(segment):
    labels = Counter(
        label
        for sample in segment
        for label in sample.get("labels") or []
        if label
    )
    return labels.most_common(1)[0][0] if labels else "scene"


def merge_segments(segments, max_merge_sec=12.0):
    """Merge adjacent short spans that describe the same dominant scene label."""
    if not segments:
        return segments
    merged = [list(segments[0])]
    for segment in segments[1:]:
        previous = merged[-1]
        combined_duration = segment[-1]["timestamp"] - previous[0]["timestamp"]
        if combined_duration <= max_merge_sec and _segment_label(previous) == _segment_label(segment):
            previous.extend(segment)
        else:
            merged.append(list(segment))
    return merged


def _segment_motion_score(segment):
    values = [float(sample.get("change") or 0.0) for sample in segment]
    return float(np.percentile(values, 90)) if values else 0.0


def select_unstable_segments(segments, percentile=75.0):
    """Keep Katna only for the high-motion quarter of YOLO segments."""
    if not segments:
        return [], 0.0
    scores = [_segment_motion_score(segment) for segment in segments]
    threshold = float(np.percentile(scores, percentile)) if len(scores) > 1 else scores[0]
    return [segment for segment, score in zip(segments, scores) if score >= threshold], threshold


def _decode_one_target(video, frame_index, fps, width, height):
    timestamp = max(0.0, float(frame_index) / max(float(fps), 0.1))
    frame_bytes = int(width) * int(height) * 3
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{timestamp:.6f}",
        "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
        "-i", str(video), "-an", "-frames:v", "1",
        "-vf", "hwdownload,format=nv12,format=bgr24",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
    ]
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode != 0 or len(completed.stdout) < frame_bytes:
        error = completed.stderr.decode("utf-8", errors="replace")[-800:]
        raise RuntimeError(f"NVDEC target frame {frame_index} failed: {error}")
    frame = np.frombuffer(completed.stdout[:frame_bytes], dtype=np.uint8).reshape((height, width, 3)).copy()
    return int(frame_index), frame


def _gpu_decode_target_frames(video, target_frames, width, height, fps, workers=4):
    """Decode only representative timestamps through parallel NVDEC seeks."""
    targets = sorted({int(item) for item in target_frames})
    if not targets:
        return
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = [pool.submit(_decode_one_target, video, index, fps, width, height) for index in targets]
        for future in futures:
            yield future.result()


def _frame_index(candidate):
    return int(candidate.frame_index) if hasattr(candidate, "frame_index") else int(candidate["frame_index"])


def _timestamp(candidate):
    return float(candidate.timestamp) if hasattr(candidate, "timestamp") else float(candidate["timestamp"])


def _candidate_sample(candidate, segment):
    index = _frame_index(candidate)
    return min(segment, key=lambda item: abs(int(item["frame_index"]) - index))


def _candidate_information(candidate, segment):
    nearest = _candidate_sample(candidate, segment)
    semantic = nearest.get("semantic") or {}
    labels = _meaningful_sample_labels(nearest)
    confidence = sum(float(_value(item, "confidence", 0) or 0) for item in semantic.get("detections") or [])
    sharpness = _quality_key(candidate)[0] if hasattr(candidate, "image") else float(nearest.get("sharpness") or 0)
    return 2.0 * len(labels) + confidence + min(2.0, math.log1p(max(0.0, sharpness)) / 4.0)


def _candidate_visual(candidate, segment):
    if hasattr(candidate, "image"):
        gray = cv2.cvtColor(candidate.image, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (32, 18), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return _candidate_sample(candidate, segment).get("visual")


def _pick_evidence_candidates(candidates, segment):
    """Select 3-5 diverse transient VLM frames and one primary frame."""
    first, last = float(segment[0]["timestamp"]), float(segment[-1]["timestamp"])
    inside = [item for item in candidates if first - 1.0 <= _timestamp(item) <= last + 1.0]
    # Add source-order anchors so stable events and the beginning/end of long
    # events are represented even when Katna only returns motion windows.
    anchor_count = min(12, max(3, int(math.ceil(max(0.0, last - first) / 20.0)) + 2))
    anchor_positions = np.linspace(0, len(segment) - 1, anchor_count).astype(int)
    pool = inside + [segment[int(position)] for position in anchor_positions]
    unique = {}
    for item in pool:
        unique.setdefault(_frame_index(item), item)
    pool = list(unique.values())
    duration = max(0.0, last - first)
    target_count = 3 if duration < 45.0 else 4 if duration < 150.0 else 5
    target_count = min(target_count, len(pool))
    primary = max(pool, key=lambda item: _candidate_information(item, segment))
    chosen = [primary]
    if duration >= 30.0:
        edge_span = max(1, len(segment) // 5)
        edge_anchors = [
            max(segment[:edge_span], key=lambda item: _candidate_information(item, segment)),
            max(segment[-edge_span:], key=lambda item: _candidate_information(item, segment)),
        ]
        for anchor in edge_anchors:
            if _frame_index(anchor) not in {_frame_index(value) for value in chosen}:
                chosen.append(anchor)
    while len(chosen) < target_count:
        best = None
        best_score = -1.0
        for item in pool:
            if _frame_index(item) in {_frame_index(value) for value in chosen}:
                continue
            visual = _candidate_visual(item, segment)
            visual_distance = min(
                float(np.mean(np.abs(visual - _candidate_visual(value, segment))))
                for value in chosen if visual is not None and _candidate_visual(value, segment) is not None
            ) if visual is not None else 0.0
            temporal_distance = min(abs(_timestamp(item) - _timestamp(value)) for value in chosen) / max(duration, 1.0)
            score = _candidate_information(item, segment) + 5.0 * visual_distance + 2.0 * temporal_distance
            if score > best_score:
                best, best_score = item, score
        if best is None:
            break
        chosen.append(best)
    return primary, sorted(chosen, key=_timestamp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--katna-root", required=True, type=Path)
    parser.add_argument("--pipeline-root", required=True, type=Path)
    parser.add_argument("--yolo-model", required=True, type=Path)
    parser.add_argument("--pose-model", required=True, type=Path)
    # Ten FPS is enough for semantic prefiltering; Katna still scans the source
    # video independently for temporal quality maxima.
    parser.add_argument("--scan-fps", type=float, default=10.0)
    parser.add_argument("--yolo-batch-size", type=int, default=16)
    parser.add_argument("--target-decode-workers", type=int, default=4)
    parser.add_argument("--merge-max-sec", type=float, default=12.0)
    parser.add_argument("--katna-unstable-percentile", type=float, default=75.0)
    parser.add_argument("--semantic-width", type=int, default=640)
    parser.add_argument("--katna-resize", type=int, default=384)
    parser.add_argument("--katna-chunk", type=int, default=500)
    parser.add_argument("--katna-scan-fps", type=float, default=10.0)
    parser.add_argument("--webp-quality", type=int, default=80)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.pipeline_root.resolve()))
    sys.path.insert(0, str(args.katna_root.resolve()))
    from worldmm_keyframe_pipeline import PipelineConfig, SemanticAnalyzer  # type: ignore
    from extract_keyframes import brightness, entropy_score  # type: ignore
    from run_katna_yolo_single import gpu_katna_candidates  # type: ignore

    analyzer = SemanticAnalyzer(PipelineConfig(
        video=str(args.video), output=str(args.output), video_id=args.video_id,
        device=args.device, yolo_model=str(args.yolo_model), pose_model=str(args.pose_model),
        width=args.semantic_width,
    ))
    coarse_started = time.perf_counter()
    fps, frame_count, samples, segments, source_width, source_height = coarse_scan(
        args.video, analyzer, args.scan_fps, args.semantic_width, args.yolo_batch_size,
    )
    coarse_seconds = time.perf_counter() - coarse_started
    segments = merge_segments(segments, args.merge_max_sec)
    preliminary_event_count = len(segments)
    segments = merge_memory_segments(segments)
    unstable_segments, motion_threshold = select_unstable_segments(
        segments, args.katna_unstable_percentile,
    )
    katna_windows = _windows(unstable_segments, fps)

    katna_started = time.perf_counter()
    candidates = gpu_katna_candidates(
        args.video, args.katna_resize, args.katna_chunk, fps,
        source_width, source_height, args.katna_scan_fps, katna_windows,
    ) if katna_windows else []
    targeted = candidates
    selected, katna_detail = quality_select(targeted, brightness, entropy_score)
    katna_seconds = time.perf_counter() - katna_started

    events = []
    frame_records = []
    for event_index, segment in enumerate(segments, 1):
        representative, evidence = _pick_evidence_candidates(selected, segment)
        rep_index = _frame_index(representative)
        rep_timestamp = _timestamp(representative)
        action_labels = Counter(str(_value(item, "label") or "") for sample in segment for item in sample["semantic"].get("actions") or [])
        object_labels = Counter(str(_value(item, "label") or "") for sample in segment for item in sample["semantic"].get("detections") or [])
        actions = [label for label, _ in action_labels.most_common(8) if label]
        objects = [label for label, _ in object_labels.most_common(20) if label]
        kind = "event" if actions else "scene"
        event_id = f"event_{event_index:05d}"
        yolo_timeline = []
        previous_labels = None
        timeline_stride = max(1, int(math.ceil(len(segment) / 40.0)))
        for sample_index, sample in enumerate(segment):
            labels = list(sample.get("labels") or [])
            if sample_index % timeline_stride == 0 or labels != previous_labels or sample_index == len(segment) - 1:
                yolo_timeline.append({"sec": round(float(sample["timestamp"]), 2), "labels": labels[:12]})
            previous_labels = labels
        if len(yolo_timeline) > 80:
            yolo_timeline = [yolo_timeline[int(position)] for position in np.linspace(0, len(yolo_timeline) - 1, 80)]
        summary = {
            "event_id": event_id, "kind": kind,
            "label": actions[0] if actions else (objects[0] if objects else "场景"),
            "start_sec": float(segment[0]["timestamp"]), "end_sec": float(segment[-1]["timestamp"]),
            "sample_count": len(segment), "objects": objects, "actions": actions,
            "yolo_timeline": yolo_timeline,
            "semantic_substitution": True,
            "substituted_sample_count": max(0, len(segment) - 1),
        }
        events.append(summary)
        semantic = samples[min(range(len(samples)), key=lambda i: abs(samples[i]["frame_index"] - rep_index))]["semantic"]
        frame_records.append({
            "encoded_frame_index": event_index - 1, "source_frame_index": rep_index,
            "source_timestamp_sec": rep_timestamp, "event_id": event_id,
            "event_start_sec": summary["start_sec"], "event_end_sec": summary["end_sec"],
            "source_frame_count": summary["sample_count"],
            "event_kind": kind, "event_label": summary["label"],
            "event_objects": objects, "event_actions": actions,
            "yolo_timeline": yolo_timeline,
            "vlm_evidence": [
                {"source_frame_index": _frame_index(item), "source_timestamp_sec": _timestamp(item)}
                for item in evidence
            ],
            "objects": [item if isinstance(item, dict) else {"label": item.label, "confidence": item.confidence, "bbox": item.bbox, "source": item.source, "track_id": item.track_id} for item in semantic.get("detections") or []],
            "actions": [dict(item) if isinstance(item, dict) else {"label": _value(item, "label", ""), "confidence": _value(item, "confidence", 0), "source": _value(item, "source", "pose")} for item in semantic.get("actions") or []],
            "expressions": [dict(item) if isinstance(item, dict) else {"label": _value(item, "label", ""), "confidence": _value(item, "confidence", 0), "source": _value(item, "source", "expression")} for item in semantic.get("expressions") or []],
        })

    # One NVDEC pass: decode the already-selected representative indices at
    # full resolution, run final semantics, and write WebP immediately.  This
    # removes both the old full-resolution OpenCV refinement scan and the old
    # second OpenCV scan used only for WebP writing.
    webp_dir = args.output / "webp"
    webp_dir.mkdir(parents=True, exist_ok=True)
    for stale in webp_dir.glob("event_*.webp"):
        stale.unlink()
    evidence_dir = args.output / "vlm-evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for stale in evidence_dir.glob("*.webp"):
        stale.unlink()
    wanted = {}
    for record in frame_records:
        for evidence_index, evidence in enumerate(record["vlm_evidence"]):
            wanted.setdefault(evidence["source_frame_index"], []).append((record, evidence, evidence_index))
    target_decode_started = time.perf_counter()
    decoded_targets = 0
    for index, frame in _gpu_decode_target_frames(
        args.video, wanted.keys(), source_width, source_height, fps, args.target_decode_workers,
    ):
        targets = wanted.pop(index, None)
        if targets is None:
            continue
        decoded_targets += len(targets)
        final_semantic = analyzer.analyze(frame)
        encoded, buffer = cv2.imencode(".webp", frame, [cv2.IMWRITE_WEBP_QUALITY, args.webp_quality])
        if not encoded:
            raise RuntimeError(f"WebP encoding failed at {index}")
        for record, evidence, evidence_index in targets:
            is_primary = index == int(record["source_frame_index"])
            target = (
                webp_dir / f"event_{record['event_id']}.webp"
                if is_primary else evidence_dir / f"{record['event_id']}_{evidence_index:02d}.webp"
            )
            target.write_bytes(buffer.tobytes())
            evidence["webp_path"] = str(target)
            evidence["webp_bytes"] = target.stat().st_size
            if is_primary:
                record["objects"] = [item if isinstance(item, dict) else {"label": item.label, "confidence": item.confidence, "bbox": item.bbox, "source": item.source, "track_id": item.track_id} for item in final_semantic.get("detections") or []]
                record["actions"] = [dict(item) if isinstance(item, dict) else {"label": _value(item, "label", ""), "confidence": _value(item, "confidence", 0), "source": _value(item, "source", "pose")} for item in final_semantic.get("actions") or []]
                record["expressions"] = [dict(item) if isinstance(item, dict) else {"label": _value(item, "label", ""), "confidence": _value(item, "confidence", 0), "source": _value(item, "source", "expression")} for item in final_semantic.get("expressions") or []]
                record["webp_path"] = str(target)
                record["webp_bytes"] = target.stat().st_size
    target_decode_seconds = time.perf_counter() - target_decode_started
    if wanted:
        raise RuntimeError(f"missing representative frames: {sum(len(items) for items in wanted.values())}")

    semantic_payload = {"events": events, "frames": frame_records}
    (args.output / "semantic.json").write_text(json.dumps(semantic_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "frame_map.json").write_text(json.dumps({
        "source_video": str(args.video), "source_fps": fps,
        "encoded_fps": 1.0, "source_width": source_width, "source_height": source_height,
        "encoded_image_format": "webp", "direct_source": True, "frames": frame_records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    stats = {
        "video_id": args.video_id, "source_video": str(args.video),
        "source_frames": frame_count, "source_duration_sec": frame_count / max(fps, 1),
        "coarse_scan_fps": args.scan_fps, "coarse_sample_count": len(samples),
        "yolo_batch_size": args.yolo_batch_size,
        "target_decode_workers": args.target_decode_workers,
        "merge_max_sec": args.merge_max_sec,
        "katna_unstable_percentile": args.katna_unstable_percentile,
        "katna_unstable_segment_count": len(unstable_segments),
        "katna_motion_threshold": motion_threshold,
        "katna_window_count": len(katna_windows),
        "preliminary_event_count": preliminary_event_count,
        "event_count": len(events), "memory_frames": len(frame_records),
        "event_merge_before_image_write": True,
        "transient_vlm_frames": sum(len(item.get("vlm_evidence") or []) for item in frame_records),
        "persistent_webp_frames": len(frame_records),
        "katna_candidate_count": len(candidates), "katna_targeted_candidate_count": len(targeted),
        "katna_selected_count": len(selected), "semantic_substituted_samples": sum(item["substituted_sample_count"] for item in events),
        "katna_scan_fps": args.katna_scan_fps,
        "webp_total_bytes": sum(item.get("webp_bytes", 0) for item in frame_records),
        "timings_sec": {"yolo_prefilter": round(coarse_seconds, 3), "katna_targeted": round(katna_seconds, 3), "package_write": round(time.perf_counter() - katna_started - katna_seconds, 3)},
        "gpu_target_decode_sec": round(target_decode_seconds, 3),
        "gpu_target_decode_frames": decoded_targets,
        "full_resolution_refinement": "removed; KATNA quality gate selects non-motion representatives",
        "katna_selection": katna_detail,
        "implementation": "yolo_prefilter_premerge_nvdec_single_event_webp_v3",
        "total_sec": round(time.perf_counter() - started, 3),
    }
    (args.output / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
