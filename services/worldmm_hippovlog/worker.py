#!/usr/bin/env python3
"""Per-video extraction and official WorldMM caption preparation."""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from threading import Event, Thread
from datetime import datetime

from runtime import configure_worldmm, dataset_root, worldmm_root

SERVICE = Path(__file__).resolve().parent
PROJECT = SERVICE.parents[1]


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    pending.replace(path)


def extract(video_id: str, run_dir: Path) -> None:
    sys.path.insert(0, str(PROJECT))
    from backend.video.svd_keyframe import SVDLowRankKeyframeExtractor

    output = run_dir / "svd_esvd" / "keyframes" / video_id
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        print(f"[resume] keyframes {video_id}", flush=True)
        return
    video = dataset_root() / "videos" / f"{video_id}.mp4"
    stopped = Event()
    started = time.monotonic()

    def heartbeat() -> None:
        status_path = run_dir / "status.json"
        while not stopped.wait(30):
            try:
                state = json.loads(status_path.read_text(encoding="utf-8"))
                if state.get("stage") != "svd_esvd_keyframes":
                    continue
                elapsed = int(time.monotonic() - started)
                state["message"] = f"SVD/eSVD CUDA 抽帧：{video_id} · 已运行 {elapsed // 60}分{elapsed % 60}秒"
                state["updated_at"] = datetime.now().isoformat(timespec="seconds")
                save_json(status_path, state)
            except (OSError, ValueError):
                continue

    monitor = Thread(target=heartbeat, daemon=True)
    monitor.start()
    try:
        result = SVDLowRankKeyframeExtractor().extract(video, video_id, output)
    finally:
        stopped.set()
        monitor.join(timeout=1)
    events = result.manifest.get("metrics", {}).get("event_generation", {})
    if events.get("status") != "complete" or events.get("device") != "cuda":
        raise RuntimeError(f"required CUDA YOLO scan did not complete for {video_id}: {events}")
    frames = sorted(
        (asdict(frame) for scene in result.scenes for frame in scene.keyframes),
        key=lambda row: row["timestamp_sec"],
    )
    if not frames:
        raise RuntimeError(f"SVD/eSVD returned no keyframes for {video_id}")
    save_json(manifest_path, {
        "video_id": video_id, "method": result.manifest["method"], "keyframes": frames,
        "selected_keyframe_count": len(frames),
        "sampled_frame_count": result.full_keyframe_count,
        "runtime_seconds": result.manifest["metrics"]["runtime_seconds"],
        "yolo_device": events["device"],
    })
    print(f"[keyframes] {video_id}: {len(frames)} selected in {result.manifest['metrics']['runtime_seconds']}s", flush=True)


def load_transcripts(video_id: str, fine_module):
    source = dataset_root() / "metadata" / "transcripts" / "full" / video_id / "transcript.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    entries = []
    for row in payload["segments"]:
        start, end = float(row["start"]), float(row["end"])
        text = str(row.get("text", "")).strip()
        if text and end > start:
            entries.append(fine_module.SubtitleEntry(
                start_seconds=start, end_seconds=end,
                start_timestamp=fine_module.format_clock(start),
                end_timestamp=fine_module.format_clock(end), text=text,
            ))
    return entries


@lru_cache(maxsize=30)
def svd_candidates(video_id: str, run_dir: Path):
    manifest = json.loads((run_dir / "svd_esvd" / "keyframes" / video_id / "manifest.json").read_text(encoding="utf-8"))
    return manifest["keyframes"]


def svd_frames(video_id: str, run_dir: Path, segment, fine_module, video_reader_ctx):
    from PIL import Image
    candidates = svd_candidates(video_id, run_dir)
    selected = [row for row in candidates if segment.start_seconds <= row["timestamp_sec"] < segment.end_seconds]
    if not selected:
        midpoint = (segment.start_seconds + segment.end_seconds) / 2.0
        index = min(max(int(round(midpoint * video_reader_ctx.average_fps)), 0), video_reader_ctx.total_frames - 1)
        with video_reader_ctx.lock:
            image = video_reader_ctx.reader[index].asnumpy()
        return [fine_module.FrameSample(timestamp_seconds=midpoint, image=fine_module.frame_to_image(image))]
    return [fine_module.FrameSample(
        timestamp_seconds=float(row["timestamp_sec"]),
        image=Image.open(row["path"]).convert("RGB"),
    ) for row in selected]


def caption(video_id: str, algorithm: str, run_dir: Path) -> None:
    output = run_dir / algorithm / "captions" / video_id
    if (output / "metrics.json").is_file() and all(
        (output / f"{name}.json").is_file() for name in ("10sec", "30sec", "3min", "10min")
    ):
        print(f"[resume] captions {algorithm} {video_id}", flush=True)
        return
    configure_worldmm()
    from decord import VideoReader, cpu
    from threading import Lock
    from worldmm.llm import LLMModel
    fine = importlib.import_module("generate_fine_caption")

    video = dataset_root() / "videos" / f"{video_id}.mp4"
    output.mkdir(parents=True, exist_ok=True)
    final = output / "10sec.json"
    records = output / "10sec.partial.jsonl"
    model = LLMModel(model_name="qwen3vl-4b")
    reader = VideoReader(str(video), ctx=cpu(0))
    duration, fps = fine.get_video_duration(reader)
    ctx = fine.VideoReaderContext(reader=reader, average_fps=fps, total_frames=len(reader), lock=Lock())
    segments = fine.build_segments(duration, load_transcripts(video_id, fine), unit_time=10)
    done = {}
    if records.is_file():
        for line in records.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                done[int(row["index"])] = row
            except (ValueError, KeyError):
                continue
    started = time.perf_counter()
    selection_seconds = 0.0
    if not final.is_file():
        for index, segment in enumerate(segments):
            if index in done:
                continue
            selection_start = time.perf_counter()
            fallback = algorithm == "svd_esvd" and not any(
                segment.start_seconds <= row["timestamp_sec"] < segment.end_seconds
                for row in svd_candidates(video_id, run_dir)
            )
            frames = (fine.sample_segment_frames(ctx, segment.start_seconds, segment.end_seconds)
                      if algorithm == "worldmm_uniform" else svd_frames(video_id, run_dir, segment, fine, ctx))
            selection_seconds += time.perf_counter() - selection_start
            try:
                response = model.generate(fine.build_segment_prompt(segment, frames))
                caption_text = str(response or "").strip()
                if not caption_text:
                    raise RuntimeError(f"empty caption for {video_id} segment {index}")
                row = {
                    "index": index, "entry": fine.build_caption_entry(segment, video, caption_text),
                    "input_frame_count": len(frames),
                    "fallback_midpoint_frame": fallback,
                }
                with records.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                done[index] = row
            finally:
                fine.release_frames(frames)
            if (index + 1) % 10 == 0 or index + 1 == len(segments):
                print(f"[caption] {algorithm} {video_id}: {index + 1}/{len(segments)}", flush=True)
        if len(done) != len(segments):
            raise RuntimeError(f"caption checkpoint has {len(done)}/{len(segments)} segments")
        save_json(final, [done[index]["entry"] for index in range(len(segments))])
    if not (output / "10min.json").is_file():
        multiscale = importlib.import_module("worldmm.memory.episodic.gen_multiscale")
        multiscale.ThreadPoolExecutor = lambda: ThreadPoolExecutor(max_workers=1)
        multiscale.gen_multiscale(str(final), str(output), model, perspective="general")
    metrics = {
        "video_id": video_id, "algorithm": algorithm,
        "caption_count": len(segments),
        "input_frame_count": sum(int(row["input_frame_count"]) for row in done.values()),
        "fallback_window_count": sum(bool(row.get("fallback_midpoint_frame")) for row in done.values()),
        "selection_seconds": round(selection_seconds, 3),
        "caption_and_multiscale_seconds": round(time.perf_counter() - started, 3),
    }
    save_json(output / "metrics.json", metrics)
    print(f"[caption complete] {algorithm} {video_id}: {len(segments)} captions", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["extract", "caption"])
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--algorithm", choices=["worldmm_uniform", "svd_esvd"])
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.stage == "extract":
        extract(args.video_id, args.run_dir)
    else:
        if not args.algorithm:
            parser.error("--algorithm is required for captions")
        caption(args.video_id, args.algorithm, args.run_dir)


if __name__ == "__main__":
    main()
