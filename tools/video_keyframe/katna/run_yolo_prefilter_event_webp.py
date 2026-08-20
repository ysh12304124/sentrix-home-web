#!/usr/bin/env python3
"""YOLO/Pose prefilter -> targeted Katna -> one WebP/VLM frame per event.

The coarse pass samples the video without writing images.  Stable intervals are
represented by semantic spans and one anchor; only change windows are sent to
the Katna quality gate.  The output is a keyframe package consumed by the
normal WebP memory importer.
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
            pending.append({"frame": frame, "gray": gray, "frame_index": index, "timestamp": index / fps})
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


def _pick_candidate(candidates, segment):
    if not candidates:
        return max(segment, key=lambda item: item["sharpness"])
    first, last = segment[0]["timestamp"], segment[-1]["timestamp"]
    inside = [item for item in candidates if first - 1.0 <= item.timestamp <= last + 1.0]
    return max(inside or candidates, key=_quality_key)


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
        segment_candidates = [item for item in selected if segment[0]["frame_index"] - 1 <= item.frame_index <= segment[-1]["frame_index"] + 1]
        representative = _pick_candidate(segment_candidates, segment)
        if hasattr(representative, "frame_index"):
            rep_index = int(representative.frame_index)
            rep_timestamp = float(representative.timestamp)
        else:
            rep_index = int(representative["frame_index"])
            rep_timestamp = float(representative["timestamp"])
        action_labels = Counter(str(_value(item, "label") or "") for sample in segment for item in sample["semantic"].get("actions") or [])
        object_labels = Counter(str(_value(item, "label") or "") for sample in segment for item in sample["semantic"].get("detections") or [])
        actions = [label for label, _ in action_labels.most_common(8) if label]
        objects = [label for label, _ in object_labels.most_common(20) if label]
        kind = "event" if actions else "scene"
        event_id = f"event_{event_index:05d}"
        summary = {
            "event_id": event_id, "kind": kind,
            "label": actions[0] if actions else (objects[0] if objects else "场景"),
            "start_sec": float(segment[0]["timestamp"]), "end_sec": float(segment[-1]["timestamp"]),
            "sample_count": len(segment), "objects": objects, "actions": actions,
            "semantic_substitution": True,
            "substituted_sample_count": max(0, len(segment) - 1),
        }
        events.append(summary)
        semantic = samples[min(range(len(samples)), key=lambda i: abs(samples[i]["frame_index"] - rep_index))]["semantic"]
        frame_records.append({
            "encoded_frame_index": event_index - 1, "source_frame_index": rep_index,
            "source_timestamp_sec": rep_timestamp, "event_id": event_id,
            "event_start_sec": summary["start_sec"], "event_end_sec": summary["end_sec"],
            "event_kind": kind, "event_label": summary["label"],
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
    wanted = {}
    for item in frame_records:
        wanted.setdefault(item["source_frame_index"], []).append(item)
    target_decode_started = time.perf_counter()
    decoded_targets = 0
    for index, frame in _gpu_decode_target_frames(
        args.video, wanted.keys(), source_width, source_height, fps, args.target_decode_workers,
    ):
        records = wanted.pop(index, None)
        if records is None:
            continue
        decoded_targets += len(records)
        for record in records:
            final_semantic = analyzer.analyze(frame)
            record["objects"] = [item if isinstance(item, dict) else {"label": item.label, "confidence": item.confidence, "bbox": item.bbox, "source": item.source, "track_id": item.track_id} for item in final_semantic.get("detections") or []]
            record["actions"] = [dict(item) if isinstance(item, dict) else {"label": _value(item, "label", ""), "confidence": _value(item, "confidence", 0), "source": _value(item, "source", "pose")} for item in final_semantic.get("actions") or []]
            record["expressions"] = [dict(item) if isinstance(item, dict) else {"label": _value(item, "label", ""), "confidence": _value(item, "confidence", 0), "source": _value(item, "source", "expression")} for item in final_semantic.get("expressions") or []]
            target = webp_dir / f"event_{record['event_id']}.webp"
            encoded, buffer = cv2.imencode(".webp", frame, [cv2.IMWRITE_WEBP_QUALITY, args.webp_quality])
            if not encoded:
                raise RuntimeError(f"WebP encoding failed at {index}")
            target.write_bytes(buffer.tobytes())
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
        "event_count": len(events), "memory_frames": len(frame_records),
        "katna_candidate_count": len(candidates), "katna_targeted_candidate_count": len(targeted),
        "katna_selected_count": len(selected), "semantic_substituted_samples": sum(item["substituted_sample_count"] for item in events),
        "katna_scan_fps": args.katna_scan_fps,
        "webp_total_bytes": sum(item.get("webp_bytes", 0) for item in frame_records),
        "timings_sec": {"yolo_prefilter": round(coarse_seconds, 3), "katna_targeted": round(katna_seconds, 3), "package_write": round(time.perf_counter() - katna_started - katna_seconds, 3)},
        "gpu_target_decode_sec": round(target_decode_seconds, 3),
        "gpu_target_decode_frames": decoded_targets,
        "full_resolution_refinement": "removed; KATNA quality gate selects non-motion representatives",
        "katna_selection": katna_detail,
        "implementation": "yolo_prefilter_nvdec_katna_event_webp_v2",
        "total_sec": round(time.perf_counter() - started, 3),
    }
    (args.output / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
