#!/usr/bin/env python3
"""Run the VLM-free v2.3 MTSW keyframe pipeline.

Two modes are deliberately exposed:

* ``--semantic``: fair event-only ablation over the existing 180 WebP package;
* ``--video``: low-rate state scan, local NVDEC dense decode and new WebP
  evidence under a new v2.3 method run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.db import MemoryStore, make_id
from backend.video.event_aggregator import DINOEmbedder, EventAggregator
from backend.video.mtsw import MTSW_METHOD_VERSION, MTSW_CONFIG, FrameState, MTSWEngine, records_from_images


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_existing(store, media_id, semantic_path, cache_dir):
    asset = store.get_asset(media_id)
    if not asset:
        raise ValueError(f"media asset not found: {media_id}")
    package = json.loads(Path(semantic_path).read_text(encoding="utf-8"))
    records = EventAggregator.records_from_package(package, store.list_derived_assets(media_id))
    embedder = DINOEmbedder(cache_dir=cache_dir, device=os.getenv("SENTRIX_EVENTAGG_DEVICE"))
    embeddings, embedding_metrics = embedder.embed(media_id, records, 16)
    states = records_from_images([])
    for index, record in enumerate(records):
        image = cv2.imread(str(record.image_path))
        state = FrameState(
            frame_index=int((record.raw or {}).get("source_frame_index") or index),
            timestamp=float(record.timestamp), image_path=str(record.image_path), image=image,
            objects=list(record.objects or []), scene_embedding=embeddings.get(record.frame_id, []),
            sharpness=float(record.sharpness or 0),
        )
        states.append(state)
    return asset, states, {"embedding": embedding_metrics, "source": "existing_180_webp"}


class LowRateScanner:
    def __init__(self, video, output_dir, scan_fps=2.0, width=640, batch_size=32, pose_stride=4):
        self.video = Path(video)
        self.output_dir = Path(output_dir)
        self.scan_dir = self.output_dir / "scan"
        self.scan_dir.mkdir(parents=True, exist_ok=True)
        self.scan_fps, self.width = float(scan_fps), int(width)
        self.batch_size, self.pose_stride = max(1, int(batch_size)), max(1, int(pose_stride))
        self.model = None
        self.model_error = None
        self.pose_model = None
        self.pose_error = None

    def _load_models(self):
        try:
            from ultralytics import YOLO
            model_path = os.getenv("SENTRIX_VIDEO_YOLO_MODEL", str(ROOT / "tools/video_keyframe/models/keyframe/yolo11n.pt"))
            self.model = YOLO(model_path)
            pose_path = os.getenv("SENTRIX_VIDEO_POSE_MODEL", str(ROOT / "tools/video_keyframe/models/keyframe/yolo11n-pose.pt"))
            self.pose_model = YOLO(pose_path)
        except Exception as error:
            self.model_error = f"{type(error).__name__}: {error}"
            self.pose_error = self.model_error

    @staticmethod
    def _result_objects(result, width, height):
        values = []
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return values
        xyxy = getattr(boxes, "xyxy", None)
        cls = getattr(boxes, "cls", None)
        conf = getattr(boxes, "conf", None)
        names = getattr(result, "names", {}) or {}
        if xyxy is None:
            return values
        for index, box in enumerate(xyxy.detach().cpu().tolist()):
            label_index = int(cls[index].item()) if cls is not None else -1
            label = str(names.get(label_index, label_index))
            confidence = float(conf[index].item()) if conf is not None else 1.0
            values.append({"label": label, "confidence": round(confidence, 4), "bbox": [round(float(item), 2) for item in box], "bbox_norm": [round(float(box[0]) / max(width, 1), 5), round(float(box[1]) / max(height, 1), 5), round(float(box[2]) / max(width, 1), 5), round(float(box[3]) / max(height, 1), 5)]})
        return values

    def scan(self):
        started = time.perf_counter()
        self._load_models()
        capture = cv2.VideoCapture(str(self.video))
        if not capture.isOpened():
            raise RuntimeError(f"unable to open video: {self.video}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        stride = max(1, int(np.ceil(fps / max(self.scan_fps, 0.1))))
        pending, samples, frame_index = [], [], -1
        yolo_seconds, pose_seconds = 0.0, 0.0

        def flush():
            nonlocal yolo_seconds, pose_seconds
            if not pending:
                return
            images = [item["image"] for item in pending]
            yolo_started = time.perf_counter()
            results = []
            if self.model is not None:
                try:
                    # The scan images are already width-normalized to 640.  A
                    # fixed stride-compatible size avoids Ultralytics warning
                    # and resize churn for every batch on 16:9/near-16:9 video.
                    results = list(self.model.predict(source=images, conf=0.25, imgsz=640, device=os.getenv("SENTRIX_VIDEO_DEVICE", "0"), verbose=False))
                except Exception as error:
                    self.model_error = f"{type(error).__name__}: {error}"
                    self.model = None
            yolo_seconds += time.perf_counter() - yolo_started
            pose_results = {}
            pose_started = time.perf_counter()
            pose_items = [(index, item) for index, item in enumerate(pending) if index % self.pose_stride == 0]
            if self.pose_model is not None and pose_items:
                try:
                    pose_output = list(self.pose_model.predict(source=[item["image"] for _, item in pose_items], conf=0.25, imgsz=640, device=os.getenv("SENTRIX_VIDEO_DEVICE", "0"), verbose=False))
                    pose_results = {index: value for (index, _), value in zip(pose_items, pose_output)}
                except Exception as error:
                    self.pose_error = f"{type(error).__name__}: {error}"
                    self.pose_model = None
            pose_seconds += time.perf_counter() - pose_started
            for item_index, (item, result) in enumerate(zip(pending, results or [None] * len(pending))):
                objects = self._result_objects(result, item["image"].shape[1], item["image"].shape[0]) if result is not None else []
                persons = [obj for obj in objects if obj["label"] == "person"]
                pose_result = pose_results.get(item_index)
                pose_state = "unknown"
                if persons:
                    largest = max(persons, key=lambda obj: (obj["bbox"][2] - obj["bbox"][0]) * (obj["bbox"][3] - obj["bbox"][1]))
                    box = largest["bbox"]
                    ratio = (box[3] - box[1]) / max(1.0, box[2] - box[0])
                    pose_state = "standing" if ratio >= 1.35 else "sitting_or_close"
                keypoints = getattr(pose_result, "keypoints", None) if pose_result is not None else None
                if keypoints is not None and getattr(keypoints, "xy", None) is not None:
                    points = keypoints.xy.detach().cpu().numpy()
                    if len(points):
                        visible = points.reshape(-1, 2)
                        if len(visible) and float(np.mean(visible[:, 1])) < item["image"].shape[0] * 0.42:
                            pose_state = "arm_up_or_hand_near_face"
                image_path = self.scan_dir / f"scan_{item['frame_index']:08d}.jpg"
                cv2.imwrite(str(image_path), item["image"], [cv2.IMWRITE_JPEG_QUALITY, 62])
                samples.append(FrameState(frame_index=item["frame_index"], timestamp=item["timestamp"], image_path=str(image_path), image=item["image"], objects=objects, people=persons, pose_state=pose_state))
            pending.clear()

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_index += 1
                if frame_index % stride:
                    continue
                if frame.shape[1] > self.width:
                    scale = self.width / frame.shape[1]
                    frame = cv2.resize(frame, (self.width, max(1, round(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)
                pending.append({"frame_index": frame_index, "timestamp": frame_index / fps, "image": frame})
                if len(pending) >= self.batch_size:
                    flush()
        finally:
            flush()
            capture.release()
        return samples, {
            "scan_decode_backend": "opencv_sequential_low_rate",
            "scan_fps": self.scan_fps, "scan_stride": stride, "scan_frames": len(samples),
            "source_frame_count": frame_count, "source_fps": fps, "source_width": width, "source_height": height,
            "yolo_seconds": round(yolo_seconds, 4), "pose_seconds": round(pose_seconds, 4),
            "yolo_model_error": self.model_error, "pose_model_error": self.pose_error,
            "scan_wall_seconds": round(time.perf_counter() - started, 4),
        }


def _dense_decode(video, state, width, height, output_path):
    timestamp = max(0.0, float(state.timestamp))
    frame_bytes = int(width) * int(height) * 3
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{timestamp:.6f}", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-i", str(video), "-an", "-frames:v", "1", "-vf", "hwdownload,format=nv12,format=bgr24", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode == 0 and len(completed.stdout) >= frame_bytes:
        frame = np.frombuffer(completed.stdout[:frame_bytes], dtype=np.uint8).reshape((height, width, 3)).copy()
        backend = "nvdec"
    else:
        capture = cv2.VideoCapture(str(video)); capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000); ok, frame = capture.read(); capture.release()
        if not ok:
            raise RuntimeError(completed.stderr.decode("utf-8", errors="replace")[-800:])
        backend = "opencv_fallback"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_started = time.perf_counter()
    ok, encoded = cv2.imencode(".webp", frame, [cv2.IMWRITE_WEBP_QUALITY, 80])
    if not ok:
        raise RuntimeError(f"WebP encode failed at {timestamp}")
    output_path.write_bytes(encoded.tobytes())
    return backend, frame, round(time.perf_counter() - write_started, 6)


def _event_payload(event, frame_ids, run_id, captured_at=None):
    members = event["members"]
    objects = sorted({label for state in members for label in state.object_labels})
    representative = max(members, key=lambda state: (state.sharpness, len(state.object_labels)))
    return {"event_id": f"{run_id}_event_{event['event_index'] + 1:04d}", "start_sec": members[0].timestamp, "end_sec": members[-1].timestamp, "start_time": captured_at, "end_time": captured_at, "representative_frame_id": frame_ids[representative.frame_index], "place_id": None, "merge_score": 1.0, "frame_count": len(members), "object_summary": objects, "person_count_range": [min(state.person_count for state in members), max(state.person_count for state in members)], "state_count": len({state.state_signature for state in members}), "state_reasons": sorted({reason for state in members for reason in state.selection_reason}), "bridge_interruptions": event.get("interruptions", []), "member_frames": [{"frame_id": frame_ids[state.frame_index], "timestamp_sec": state.timestamp, "frame_hash": hashlib.sha256(Path(state.image_path).read_bytes()).hexdigest() if state.image_path and Path(state.image_path).is_file() else ""} for state in members], "metadata": {"method_version": MTSW_METHOD_VERSION, "state_count": len({state.state_signature for state in members}), "selection_reasons": sorted({reason for state in members for reason in state.selection_reason})}}


def _write_reports(output_dir, result, runtime, mode):
    output_dir.mkdir(parents=True, exist_ok=True)
    report_events = []
    for event in result["events"]:
        members = event.get("members", [])
        report_events.append({
            "event_index": event.get("event_index"),
            "start_sec": members[0].timestamp if members else None,
            "end_sec": members[-1].timestamp if members else None,
            "frame_indices": [state.frame_index for state in members],
            "state_count": len({state.state_signature for state in members}),
            "interruptions": event.get("interruptions", []),
        })
    payload = {"method_version": MTSW_METHOD_VERSION, "mode": mode, "config": result["config"], "metrics": result["metrics"], "runtime": runtime, "events": report_events, "transitions": result["transitions"], "dedup_cases": result["dedup_cases"], "bridge_cases": result["bridge_cases"]}
    (output_dir / "v23_ablation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    timeline_rows = "".join(f"<tr><td>{item['timestamp']:.2f}</td><td>{item['score']:.4f}</td><td>{item['z_score']:.3f}</td><td>{'KEYFRAME' if item['frame_index'] in {state.frame_index for state in result['selected']} else 'candidate'}</td><td>{', '.join(item['reasons'])}</td></tr>" for item in result["transitions"])
    state_rows = "".join(f"<tr><td>{state.timestamp:.2f}</td><td>{state.change_score:.4f}</td><td>{state.z_score:.3f}</td><td>{'selected' if state.selected else ('duplicate' if state.duplicate_of else '')}</td><td>{', '.join(state.selection_reason)}</td><td>{', '.join(state.object_labels)}</td></tr>" for state in result["states"])
    base_style = "body{font:14px system-ui;margin:24px;background:#f7faf4;color:#172016}table{border-collapse:collapse;background:#fff;width:100%}th,td{padding:7px;border:1px solid #dfe8d7;text-align:left}th{background:#eaf4df}.panel{background:#fff;padding:16px;margin:16px 0;border-radius:12px}"
    (output_dir / "v23_keyframe_timeline.html").write_text(f"<!doctype html><meta charset='utf-8'><style>{base_style}</style><h1>v2.3 MTSW keyframe timeline</h1><div class='panel'><pre>{json.dumps({'metrics': result['metrics'], 'runtime': runtime}, ensure_ascii=False, indent=2)}</pre></div><table><tr><th>time</th><th>change</th><th>z</th><th>decision</th><th>reason</th></tr>{timeline_rows}</table>", encoding="utf-8")
    (output_dir / "v23_window_debug.html").write_text(f"<!doctype html><meta charset='utf-8'><style>{base_style}</style><h1>v2.3 window debug</h1><p>Micro={result['config']['micro_window_sec']}s · Event={result['config']['event_short_window_sec']}s/{result['config']['event_reference_window_sec']}s · Context={result['config']['context_max_events']} events/{result['config']['context_max_time_sec']}s</p><table><tr><th>time</th><th>state score</th><th>adaptive z</th><th>selection</th><th>reason</th><th>objects</th></tr>{state_rows}</table>", encoding="utf-8")
    (output_dir / "v23_bridge_cases.html").write_text(f"<!doctype html><meta charset='utf-8'><style>{base_style}</style><h1>v2.3 bridge cases</h1><pre>{json.dumps(result['bridge_cases'], ensure_ascii=False, indent=2)}</pre>", encoding="utf-8")
    (output_dir / "v23_dedup_cases.html").write_text(f"<!doctype html><meta charset='utf-8'><style>{base_style}</style><h1>v2.3 state-aware dedup cases</h1><pre>{json.dumps(result['dedup_cases'], ensure_ascii=False, indent=2)}</pre>", encoding="utf-8")
    return payload


def _store_run(store, asset, scope_id, result, output_dir, run_id, selected_asset_ids, runtime):
    events = [_event_payload(event, selected_asset_ids, run_id, asset.get("captured_at")) for event in result["events"]]
    memory_started = time.perf_counter()
    metrics = {**result["metrics"], **runtime, "vlm_calls": 0, "llm_calls": 0, "event_count": len(events), "timeline_compression": round(1.0 - len(events) / max(1, int(result["metrics"].get("final_keyframes") or 1)), 6)}
    config = {**result["config"], "output_dir": str(output_dir), "method_version": MTSW_METHOD_VERSION}
    store.replace_eventagg_run(run_id, asset["id"], scope_id or asset.get("scope_id") or "home-default", MTSW_METHOD_VERSION, events, metrics, config)
    metrics["memory_runtime_seconds"] = round(time.perf_counter() - memory_started, 4)
    store.finish_method_run(run_id, "completed", metrics)
    return events, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--media-id", required=True)
    parser.add_argument("--scope-id", default="loan-shoe-v2-mtsw")
    parser.add_argument("--video")
    parser.add_argument("--semantic")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default="/ssd/sscy/eventagg-cache/embeddings")
    parser.add_argument("--scan-fps", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--z-threshold", type=float, default=1.7)
    parser.add_argument("--dry-db", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    store = MemoryStore(args.db)
    asset = store.get_asset(args.media_id)
    if not asset:
        raise SystemExit(f"media asset not found: {args.media_id}")
    config = {**MTSW_CONFIG, "scan_fps": args.scan_fps, "z_threshold": args.z_threshold}
    if args.semantic:
        asset, states, source_runtime = _load_existing(store, args.media_id, args.semantic, args.cache_dir)
        mode = "existing_180_ablation"
    elif args.video:
        scanner = LowRateScanner(args.video, args.output, args.scan_fps, batch_size=args.batch_size)
        states, source_runtime = scanner.scan()
        mode = "full_video_mtsw"
    else:
        raise SystemExit("one of --semantic or --video is required")
    engine = MTSWEngine(config)
    result = engine.analyze(states)
    result["config"] = config
    Path(args.output).mkdir(parents=True, exist_ok=True)
    state_cache = []
    for state in states:
        state_cache.append({"frame_index": state.frame_index, "timestamp": state.timestamp, "image_path": state.image_path, "objects": state.objects, "people": state.people, "pose_state": state.pose_state, "scene_embedding": state.scene_embedding, "appearance_embedding": state.appearance_embedding, "sharpness": state.sharpness, "phash": state.phash, "black_frame": state.black_frame})
    Path(args.output, "state_cache.json").write_text(json.dumps({"cache_version": "mtsw-state-v1", "method_version": MTSW_METHOD_VERSION, "states": state_cache}, ensure_ascii=False), encoding="utf-8")
    selected_asset_ids = {}
    dense_runtime = {"dense_decode_seconds": 0.0, "dense_decode_windows": len(result["transitions"]), "dense_decode_backend": "not_run", "webp_write_seconds": 0.0}
    if args.video:
        video_meta = asset.get("metadata_json", {}).get("video_metadata", {})
        width, height = int(video_meta.get("width") or 1280), int(video_meta.get("height") or 720)
        dense_started = time.perf_counter(); backends = []; webp_seconds = 0.0
        for ordinal, state in enumerate(result["selected"], 1):
            target = Path(args.output) / "keyframes" / f"mtsw_{ordinal:04d}_{state.timestamp:.3f}.webp"
            backend, _, write_seconds = _dense_decode(args.video, state, width, height, target)
            backends.append(backend); state.image_path = str(target)
            webp_seconds += write_seconds
            selected_asset_ids[state.frame_index] = make_id("asset")
        dense_runtime["dense_decode_seconds"] = round(time.perf_counter() - dense_started, 4)
        dense_runtime["webp_write_seconds"] = round(webp_seconds, 4)
        dense_runtime["dense_decode_backend"] = ",".join(sorted(set(backends))) if backends else "none"
        # Create derived assets only for v2.3-selected WebP; baseline/evidence remains untouched.
        for state in result["selected"]:
            frame_id = selected_asset_ids[state.frame_index]
            path = Path(state.image_path)
            store.create_asset(frame_id, path.name, "image", str(path), "image/webp", path.stat().st_size, {"scope_id": args.scope_id, "parent_asset_id": args.media_id, "derived_kind": "video_mtsw_keyframe", "source_timestamp_sec": state.timestamp, "source_frame_index": state.frame_index, "mtsw_method_version": MTSW_METHOD_VERSION, "selection_reason": state.selection_reason, "state_score": state.change_score, "z_score": state.z_score, "selected": True}, scope_id=args.scope_id)
    else:
        # Existing package records already have stable evidence Asset IDs.
        derived = store.list_derived_assets(args.media_id)
        by_path = {str(Path(item.get("path", "")).resolve()): item["id"] for item in derived}
        for state in result["selected"]:
            selected_asset_ids[state.frame_index] = by_path.get(str(Path(state.image_path).resolve()), make_id("mtsw"))
    runtime = {**source_runtime, **dense_runtime, "total_wall_seconds": round(time.perf_counter() - started, 4), "vlm_calls": 0, "llm_calls": 0}
    run_id = f"mtsw_{args.media_id}_{'ablation' if args.semantic else 'full'}"
    if not args.dry_db:
        events, metrics = _store_run(store, asset, args.scope_id, result, Path(args.output), run_id, selected_asset_ids, runtime)
        result["events"] = events
        result["metrics"].update(metrics)
    payload = _write_reports(Path(args.output), result, runtime, mode)
    # Stable comparison artifact is generated from measured values only.
    comparison = {"method_version": MTSW_METHOD_VERSION, "baseline_v20": {"keyframes": 180, "events": 180}, "baseline_v21": {"keyframes": 180, "events": 131, "timeline_compression": 0.272222}, "v23": result["metrics"], "runtime": runtime}
    Path(args.output, "v23_comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 53); print("Sentrix Home v2.3 MTSW"); print("=" * 53); print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
