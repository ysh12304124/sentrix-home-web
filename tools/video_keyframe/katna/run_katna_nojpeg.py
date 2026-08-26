#!/usr/bin/env python3
"""KATNA -> YOLO/Pose -> one GPU encoded keyframe video.

The selected frames are never written as JPEG files.  KATNA supplies source
frame indices, the second sequential pass sends selected BGR frames directly
to an NVENC HEVC pipe and runs semantic inference on the in-memory frame.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--katna-engine", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--katna-resize", type=int, default=384)
    parser.add_argument("--katna-chunk", type=int, default=500)
    parser.add_argument("--semantic-width", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--yolo-model", required=True)
    parser.add_argument("--pose-model", required=True)
    args = parser.parse_args()
    args.video = args.video.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    katna_root = Path(os.environ.get("SENTRIX_KEYFRAME_KATNA_ROOT", str(Path(__file__).resolve().parent))).expanduser().resolve()
    pipeline_root = Path(os.environ.get("SENTRIX_KEYFRAME_PIPELINE_ROOT", str(Path(__file__).resolve().parents[1]))).expanduser().resolve()
    sys.path.insert(0, str(katna_root))
    sys.path.insert(0, str(pipeline_root))
    from run_katna_yolo_single import (  # type: ignore
        gpu_katna_candidates,
        katna_unlimited_quality_select,
    )
    from extract_keyframes import video_info  # type: ignore
    from worldmm_keyframe_pipeline import PipelineConfig, SemanticAnalyzer  # type: ignore

    started = time.perf_counter()
    info = video_info(args.video)
    phase = {}
    phase_started = time.perf_counter()
    if args.katna_engine != "gpu":
        raise RuntimeError("The no-JPEG production path requires the NVIDIA KATNA decoder")
    candidates = gpu_katna_candidates(
        args.video, args.katna_resize, args.katna_chunk,
        info["fps"], info["width"], info["height"],
    )
    phase["katna_scan_sec"] = round(time.perf_counter() - phase_started, 3)
    phase_started = time.perf_counter()
    selected, selection = katna_unlimited_quality_select(candidates)
    phase["quality_select_sec"] = round(time.perf_counter() - phase_started, 3)
    if not selected:
        raise RuntimeError("KATNA selected no frames")

    analyzer = SemanticAnalyzer(PipelineConfig(
        video=str(args.video), output=str(args.output), video_id=args.video_id,
        device=args.device, yolo_model=args.yolo_model, pose_model=args.pose_model,
    ))
    encoded_path = args.output / "selected_frames.hevc.mp4"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{info['width']}x{info['height']}",
        "-r", "25", "-i", "pipe:0", "-an", "-vf", "format=nv12,hwupload_cuda",
        "-c:v", "hevc_nvenc", "-preset", "fast", "-rc", "constqp", "-qp", "23",
        "-movflags", "+faststart", str(encoded_path),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    selected_by_index = {int(item.frame_index): item for item in selected}
    capture = cv2.VideoCapture(str(args.video))
    records = []
    source_index = 0
    phase_started = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            candidate = selected_by_index.get(source_index)
            if candidate is not None:
                if encoder.stdin is None:
                    raise RuntimeError("NVENC pipe is unavailable")
                encoder.stdin.write(frame.tobytes())
                semantic_frame = frame
                if semantic_frame.shape[1] > args.semantic_width:
                    scale = args.semantic_width / semantic_frame.shape[1]
                    semantic_frame = cv2.resize(semantic_frame, (args.semantic_width, max(1, round(semantic_frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)
                semantic = analyzer.analyze(semantic_frame)
                records.append({
                    "encoded_frame_index": len(records),
                    "source_frame_index": source_index,
                    "source_timestamp_sec": round(source_index / info["fps"], 6),
                    "objects": [asdict(item) for item in semantic.get("detections", [])],
                    "actions": semantic.get("actions", []),
                    "expressions": semantic.get("expressions", []),
                })
            source_index += 1
    finally:
        capture.release()
        if encoder.stdin:
            encoder.stdin.close()
        stderr = encoder.stderr.read().decode("utf-8", errors="replace") if encoder.stderr else ""
        return_code = encoder.wait()
    phase["semantic_and_encode_sec"] = round(time.perf_counter() - phase_started, 3)
    if return_code != 0 or not encoded_path.is_file():
        raise RuntimeError(f"NVENC keyframe encode failed ({return_code}): {stderr[-1200:]}")

    frame_map = {
        "source_video": str(args.video), "source_fps": info["fps"],
        "source_width": info["width"], "source_height": info["height"],
        "encoded_video": str(encoded_path), "encoded_fps": 25,
        "frames": records,
    }
    (args.output / "frame_map.json").write_text(json.dumps(frame_map, ensure_ascii=False), encoding="utf-8")
    (args.output / "semantic.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    stats = {
        "source_frames": info["frame_count"], "selected_frames": len(records),
        "encoded_frames": len(records), "source_width": info["width"],
        "source_height": info["height"], "encoded_codec": "hevc_nvenc",
        "output_bytes": encoded_path.stat().st_size,
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "katna_scan_sec": phase["katna_scan_sec"],
        "quality_select_sec": phase["quality_select_sec"],
        "semantic_and_encode_sec": phase["semantic_and_encode_sec"],
        "katna_selection": selection,
        "semantic_components": analyzer.loaded_components,
    }
    (args.output / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
