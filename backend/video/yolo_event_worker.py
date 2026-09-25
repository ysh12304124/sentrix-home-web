"""Sparse YOLO scan used by the two-pass SVD keyframe extractor.

This worker intentionally evaluates only the representative frame indices
selected by the first SVD fold. It shares the project's SemanticAnalyzer and
YOLO model configuration without running the rest of the WorldMM pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2


def _resize(frame, width):
    if frame.shape[1] <= width:
        return frame
    scale = width / frame.shape[1]
    return cv2.resize(frame, (width, max(1, round(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)


def scan(video_path: Path, candidates_path: Path, output_path: Path, model: str, device: str,
         width: int, batch_size: int) -> dict:
    video_tools = Path(__file__).resolve().parents[2] / "tools" / "video_keyframe"
    sys.path.insert(0, str(video_tools))
    from worldmm_keyframe_pipeline import PipelineConfig, SemanticAnalyzer  # type: ignore

    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError("first-pass SVD produced no YOLO candidate frames")
    config = PipelineConfig(
        video=str(video_path), output=str(output_path.parent), width=width,
        device=device, yolo_model=model, pose_model="", disable_semantics=False,
    )
    analyzer = SemanticAnalyzer(config)
    if analyzer.detector is None:
        raise RuntimeError(f"YOLO detector unavailable for configured model: {model}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video for sparse YOLO scan: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        capture.release()
        raise RuntimeError("video FPS is unavailable for YOLO frame timestamps")

    ordered = sorted(candidates, key=lambda item: int(item["frame_index"]))
    target_cursor = 0
    frame_index = 0
    frames, batch_candidates, records = [], [], []

    def flush_batch():
        if not frames:
            return
        results = analyzer.detector.predict(
            source=frames, conf=0.30, iou=0.70, imgsz=width,
            device=device, verbose=False,
        )
        if len(results) != len(batch_candidates):
            raise RuntimeError("YOLO returned a different number of results than input frames")
        for candidate, result, frame in zip(batch_candidates, results, frames):
            detections = analyzer._extract_detections(result, "yolo", frame.shape[1], frame.shape[0])
            records.append({
                "sample_index": int(candidate["sample_index"]),
                "frame_index": int(candidate["frame_index"]),
                "timestamp_sec": float(candidate["timestamp_sec"]),
                "detections": [
                    {"label": item.label, "confidence": float(item.confidence), "bbox": list(item.bbox)}
                    for item in detections
                ],
                "labels": sorted({item.label for item in detections}),
            })
        frames.clear()
        batch_candidates.clear()

    try:
        last_target = int(ordered[-1]["frame_index"])
        while target_cursor < len(ordered) and frame_index <= last_target:
            ok, source = capture.read()
            if not ok:
                break
            candidate = ordered[target_cursor]
            target_index = int(candidate["frame_index"])
            if frame_index == target_index:
                frames.append(_resize(source, width))
                batch_candidates.append(candidate)
                target_cursor += 1
                if len(frames) >= max(1, int(batch_size)):
                    flush_batch()
            frame_index += 1
        flush_batch()
    finally:
        capture.release()

    if target_cursor != len(ordered):
        raise RuntimeError(f"decoded {target_cursor}/{len(ordered)} YOLO candidates")
    result = {
        "method": "yolo_object_change_v1", "model": model, "device": device,
        "fps": fps, "input_frame_count": len(ordered), "samples": records,
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = scan(args.video, args.candidates, args.output, args.model, args.device, args.width, args.batch_size)
    print(json.dumps({"status": "ok", "sample_count": len(result["samples"]), "model": result["model"]}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"YOLO event scan failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
