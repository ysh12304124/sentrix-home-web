#!/usr/bin/env python3
"""Benchmark the JPEG-style memory path with WebP evidence frames.

The source video remains untouched.  KATNA selects frames with the NVIDIA
decode path, selected frames are saved as WebP, YOLO/Pose runs on resized
in-memory frames, and the existing Gamma image-description path reads the
WebP assets.  The output is an isolated benchmark package, not an ingest.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np


def _uniform_sample(items, count):
    if count <= 0 or len(items) <= count:
        return list(items)
    positions = np.linspace(0, len(items) - 1, count, dtype=int).tolist()
    return [items[position] for position in positions]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--katna-root", required=True, type=Path)
    parser.add_argument("--pipeline-root", required=True, type=Path)
    parser.add_argument("--yolo-model", required=True, type=Path)
    parser.add_argument("--pose-model", required=True, type=Path)
    parser.add_argument("--katna-resize", type=int, default=384)
    parser.add_argument("--katna-chunk", type=int, default=500)
    parser.add_argument("--semantic-width", type=int, default=640)
    parser.add_argument("--memory-frames", type=int, default=160)
    parser.add_argument("--webp-quality", type=int, default=80)
    parser.add_argument("--device", default="0")
    parser.add_argument("--reuse-descriptions", action="store_true")
    args = parser.parse_args()

    video = args.video.resolve()
    output = args.output.resolve()
    webp_dir = output / "webp"
    output.mkdir(parents=True, exist_ok=True)
    webp_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.katna_root.resolve()))
    sys.path.insert(0, str(args.pipeline_root.resolve()))
    project_root = args.pipeline_root.resolve().parents[1]
    sys.path.insert(0, str(project_root))

    from extract_keyframes import video_info  # type: ignore
    from run_katna_yolo_single import (  # type: ignore
        gpu_katna_candidates,
        katna_unlimited_quality_select,
    )
    from worldmm_keyframe_pipeline import PipelineConfig, SemanticAnalyzer  # type: ignore

    started = time.perf_counter()
    info = video_info(video)
    phase = {}

    phase_started = time.perf_counter()
    candidates = gpu_katna_candidates(
        video, args.katna_resize, args.katna_chunk,
        info["fps"], info["width"], info["height"],
    )
    phase["katna_optical_flow_sec"] = round(time.perf_counter() - phase_started, 3)

    phase_started = time.perf_counter()
    selected, selection = katna_unlimited_quality_select(candidates)
    phase["katna_quality_select_sec"] = round(time.perf_counter() - phase_started, 3)
    selected = _uniform_sample(selected, args.memory_frames)
    if not selected:
        raise RuntimeError("KATNA selected no frames")
    selected_by_index = {int(item.frame_index): item for item in selected}

    analyzer = SemanticAnalyzer(PipelineConfig(
        video=str(video), output=str(output), video_id=args.video_id,
        device=args.device, yolo_model=str(args.yolo_model),
        pose_model=str(args.pose_model), width=args.semantic_width,
    ))
    semantic_records = []
    encode_seconds = 0.0
    yolopose_seconds = 0.0
    decode_pass_started = time.perf_counter()
    capture = cv2.VideoCapture(str(video))
    source_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            candidate = selected_by_index.get(source_index)
            if candidate is None:
                source_index += 1
                continue

            webp_path = webp_dir / f"frame_{source_index:08d}_t{source_index / info['fps']:010.3f}.webp"
            encode_started = time.perf_counter()
            ok_encode, encoded = cv2.imencode(
                ".webp", frame,
                [cv2.IMWRITE_WEBP_QUALITY, max(1, min(100, args.webp_quality))],
            )
            if not ok_encode:
                raise RuntimeError(f"WebP encode failed at source frame {source_index}")
            webp_path.write_bytes(encoded.tobytes())
            encode_seconds += time.perf_counter() - encode_started

            semantic_frame = frame
            if semantic_frame.shape[1] > args.semantic_width:
                scale = args.semantic_width / semantic_frame.shape[1]
                semantic_frame = cv2.resize(
                    semantic_frame,
                    (args.semantic_width, max(1, round(semantic_frame.shape[0] * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            yolopose_started = time.perf_counter()
            semantic = analyzer.analyze(semantic_frame)
            yolopose_seconds += time.perf_counter() - yolopose_started
            semantic_records.append({
                "encoded_frame_index": len(semantic_records),
                "source_frame_index": source_index,
                "source_timestamp_sec": round(source_index / info["fps"], 6),
                "webp_path": str(webp_path),
                "webp_bytes": webp_path.stat().st_size,
                "objects": [asdict(item) for item in semantic.get("detections", [])],
                "actions": semantic.get("actions", []),
                "expressions": semantic.get("expressions", []),
            })
            source_index += 1
    finally:
        capture.release()
    phase["decode_and_webp_encode_sec"] = round(time.perf_counter() - decode_pass_started, 3)
    phase["webp_encode_only_sec"] = round(encode_seconds, 3)
    phase["yolopose_sec"] = round(yolopose_seconds, 3)
    semantic_components = dict(analyzer.loaded_components)
    (output / "semantic.json").write_text(json.dumps(semantic_records, ensure_ascii=False), encoding="utf-8")

    # Release YOLO/Pose before loading the VLM so the benchmark does not keep
    # both GPU model families resident at once.
    del analyzer
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    descriptions_path = output / "descriptions.json"
    descriptions = []
    description_seconds = 0.0
    description_started = time.perf_counter()
    if args.reuse_descriptions and descriptions_path.is_file():
        descriptions = json.loads(descriptions_path.read_text(encoding="utf-8"))
        description_seconds = sum(float(item.get("elapsed_sec") or 0) for item in descriptions)
        print(json.dumps({"reused_descriptions": len(descriptions)}, ensure_ascii=False), flush=True)
    else:
        from backend.model_clients import GammaClient

        gamma = GammaClient()
        description_started = time.perf_counter()
        for index, record in enumerate(semantic_records, 1):
            one_started = time.perf_counter()
            error = None
            analysis = {}
            try:
                analysis = gamma.analyze_image(record["webp_path"], {
                    "file_name": Path(record["webp_path"]).name,
                    "captured_at": None,
                    "captured_location": "",
                    "video_id": args.video_id,
                    "source_timestamp_sec": record["source_timestamp_sec"],
                })
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            elapsed = time.perf_counter() - one_started
            description_seconds += elapsed
            descriptions.append({
                "encoded_frame_index": record["encoded_frame_index"],
                "source_frame_index": record["source_frame_index"],
                "source_timestamp_sec": record["source_timestamp_sec"],
                "webp_path": record["webp_path"],
                "description": analysis,
                "elapsed_sec": round(elapsed, 3),
                "error": error,
            })
            if index % 10 == 0:
                print(json.dumps({
                    "described": index, "total": len(semantic_records),
                    "elapsed_sec": round(time.perf_counter() - description_started, 3),
                }, ensure_ascii=False), flush=True)
    phase["image_description_sec"] = round(description_seconds, 3)
    phase["image_description_wall_sec"] = round(
        description_seconds if args.reuse_descriptions else time.perf_counter() - description_started, 3
    )

    (output / "descriptions.json").write_text(json.dumps(descriptions, ensure_ascii=False), encoding="utf-8")
    frame_map = {
        "source_video": str(video), "source_fps": info["fps"],
        "source_width": info["width"], "source_height": info["height"],
        "encoded_image_format": "webp", "webp_quality": args.webp_quality,
        "frames": semantic_records,
    }
    (output / "frame_map.json").write_text(json.dumps(frame_map, ensure_ascii=False), encoding="utf-8")
    sizes = [item["webp_bytes"] for item in semantic_records]
    end_to_end_stage_sum = sum(
        float(phase.get(key, 0.0))
        for key in (
            "katna_optical_flow_sec", "katna_quality_select_sec",
            "decode_and_webp_encode_sec", "image_description_wall_sec",
        )
    )
    stats = {
        "source_video": str(video), "source_duration_sec": info.get("duration", info["frame_count"] / info["fps"]),
        "source_frames": info["frame_count"], "memory_frames": len(semantic_records),
        "source_width": info["width"], "source_height": info["height"],
        "webp_quality": args.webp_quality,
        "webp_total_bytes": sum(sizes),
        "webp_average_bytes": round(sum(sizes) / max(len(sizes), 1), 1),
        "webp_min_bytes": min(sizes), "webp_max_bytes": max(sizes),
        "katna_selection": selection,
        "timings_sec": phase,
        "total_sec": round(end_to_end_stage_sum, 3),
        "observed_wall_sec": round(time.perf_counter() - started, 3),
        "description_success_count": sum(not item["error"] for item in descriptions),
        "description_error_count": sum(bool(item["error"]) for item in descriptions),
        "semantic_components": semantic_components,
    }
    (output / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
