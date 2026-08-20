from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..db import make_id
from ..geocoding import format_gps_prefix
from .hybrid_keyframe import run as run_hybrid_keyframes
from .metadata import probe_video_metadata
from .worldmm_adapter import WorldMMAdapter


def _captured_at(value, offset=0.0):
    if not value:
        return None
    try:
        normalized = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", str(value)).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return (parsed + timedelta(seconds=float(offset))).isoformat()
    except (TypeError, ValueError):
        return str(value)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_LOW_INFO_ACTIONS = {
    "standing", "sitting", "raising hand", "stand", "sit", "walking", "走路",
    "站立", "坐着", "抬手", "挥手", "举手",
}


def _event_title(item):
    actions = [str(value) for value in item.get("actions") or []
               if str(value).strip().lower() not in _LOW_INFO_ACTIONS]
    objects = [str(value) for value in item.get("objects") or []
               if str(value).strip().lower() not in {"person", "chair", "couch", "bed", "potted plant", "vase"}]
    if actions:
        return f"事件：{actions[0]}"
    if objects:
        return f"场景：{objects[0]}"
    return "视频片段"


def _browser_preview(video_path, target, codec):
    if str(codec or "").lower() in {"h264", "avc1"}:
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-i", str(video_path),
        "-map", "0:v:0", "-map", "0:a:0?", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", str(target),
    ], check=False, capture_output=True, text=True, timeout=3600)
    if process.returncode or not target.is_file():
        raise RuntimeError(process.stderr.strip()[-1000:] or "ffmpeg preview transcode failed")
    return str(target)


class VideoMemoryAdapter:
    def __init__(self, worldmm=None):
        self.worldmm = worldmm or WorldMMAdapter()

    def process(self, asset, pipeline):
        store = pipeline.store
        asset_id = asset["id"]
        started = time.perf_counter()
        stage = "video-metadata"
        try:
            store.update_asset(asset_id, stage, {"video_stage": stage, "retryable": True})
            metadata = probe_video_metadata(asset["path"])
            captured_at = metadata.captured_at or asset.get("captured_at")
            if not captured_at:
                captured_at = datetime.fromtimestamp(Path(asset["path"]).stat().st_mtime, timezone.utc).isoformat()
                metadata.creation_source = "file_mtime_fallback"
            location = metadata.captured_location or asset.get("captured_location")
            reverse_geocode = {}
            if metadata.latitude is not None and metadata.longitude is not None:
                reverse_geocode = pipeline.geocoder.lookup({
                    "latitude": metadata.latitude, "longitude": metadata.longitude,
                }) or {}
            location_label = format_gps_prefix(reverse_geocode) or location or "其他或不确定"
            store.update_asset(asset_id, "video-keyframe-extracting", {
                "video_stage": "video-keyframe-extracting", "video_metadata": metadata.as_dict(),
                "captured_at": captured_at, "captured_location": location,
                "source_device_id": metadata.device or asset.get("source_device_id"),
                "reverse_geocode": reverse_geocode, "location_source": "video_metadata" if metadata.latitude is not None else "upload_metadata",
            })

            if os.getenv("SENTRIX_VIDEO_KEYFRAME_ALGORITHM", "hybrid_webp").lower() == "hybrid_webp":
                return self._process_hybrid_webp(
                    asset, pipeline, metadata, captured_at, location_label, reverse_geocode, started,
                )

            data_root = Path(os.getenv("SENTRIX_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
            derived_root = data_root / "derived" / "video" / asset_id
            worldmm_root = derived_root / "worldmm"
            preview_path = None
            preview_error = None
            try:
                preview_path = _browser_preview(asset["path"], derived_root / "preview.mp4", metadata.codec)
            except Exception as error:
                preview_error = str(error)
            store.update_asset(asset_id, "video-keyframe-extracting", {
                "browser_preview_path": preview_path, "browser_preview_error": preview_error,
            })
            stage = "video-keyframe-extracting"
            result = self.worldmm.run(asset["path"], asset_id, worldmm_root)
            if not result.scenes or not result.keyframe_count:
                raise RuntimeError("WorldMM returned no scenes or keyframes")

            stage = "video-scene-importing"
            store.update_asset(asset_id, stage, {
                "video_stage": stage, "worldmm_scene_count": len(result.scenes),
                "worldmm_keyframe_count": result.keyframe_count,
                "worldmm_full_keyframe_count": result.full_keyframe_count,
                "worldmm_summary_keyframe_count": result.summary_keyframe_count,
                "worldmm_selected_keyframe_count": result.selected_keyframe_count,
            })
            scene_ids = []
            keyframe_asset_ids = []
            for scene in result.scenes:
                scene_start = _captured_at(captured_at, scene.start_sec)
                scene_end = _captured_at(captured_at, scene.end_sec)
                event = store.create_video_scene_event({
                    "scope_id": asset.get("scope_id"), "title": f"视频场景 {scene.index + 1}",
                    "summary": f"视频场景 · {scene.start_sec:.1f}s~{scene.end_sec:.1f}s",
                    "time_start": scene_start, "time_end": scene_end, "place": location_label,
                    "source_asset_id": asset_id, "source_scene_index": scene.index,
                    "source_start_sec": scene.start_sec, "source_end_sec": scene.end_sec,
                    "source_metadata": {
                        "worldmm_scene_id": scene.scene_id, "semantic_labels": scene.semantic_labels,
                        "keyframe_count": len(scene.keyframes), "location_source": "video_metadata",
                    },
                })
                scene_ids.append(event["id"])
                scene_dir = derived_root / "scenes" / f"{scene.index:03d}"
                scene_dir.mkdir(parents=True, exist_ok=True)
                for ordinal, frame in enumerate(scene.keyframes, 1):
                    suffix = Path(frame.path).suffix or ".jpg"
                    target = scene_dir / f"kf_{ordinal:04d}{suffix}"
                    if Path(frame.path).resolve() != target.resolve():
                        shutil.copy2(frame.path, target)
                    keyframe_id = make_id("asset")
                    frame_captured_at = _captured_at(captured_at, frame.timestamp_sec)
                    provenance = {
                        "scope_id": asset.get("scope_id"), "batch_id": asset.get("batch_id"),
                        "parent_asset_id": asset_id, "derived_kind": "video_keyframe",
                        "source_timestamp_sec": frame.timestamp_sec, "source_frame_index": frame.frame_index,
                        "source_scene_index": scene.index, "captured_at": frame_captured_at,
                        "captured_location": location, "source_device_id": metadata.device or asset.get("source_device_id"),
                        "latitude": metadata.latitude, "longitude": metadata.longitude,
                        "source_captured_location": location,
                        "content_sha256": _sha256(target), "location_source": "video_metadata",
                        "worldmm_scene_id": scene.scene_id, "worldmm_keyframe_code": frame.code,
                        "worldmm_score": frame.score, "worldmm_selection_reason": frame.selection_reason,
                        "worldmm_semantics": {"objects": frame.objects, "actions": frame.actions, "expressions": frame.expressions},
                        "reverse_geocode": reverse_geocode,
                    }
                    store.create_asset(
                        keyframe_id, target.name, "image", str(target), "image/jpeg", target.stat().st_size,
                        provenance, scope_id=asset.get("scope_id"),
                    )
                    processed = pipeline.process(keyframe_id, summarize_event=False, forced_event_id=event["id"])
                    if processed.get("status") != "processed":
                        raise RuntimeError(f"keyframe semantic processing failed: {processed.get('metadata_json', {}).get('error', keyframe_id)}")
                    keyframe_asset_ids.append(keyframe_id)
                pipeline.summarize_event(event["id"])

            elapsed = round(time.perf_counter() - started, 3)
            return store.update_asset(asset_id, "processed", {
                "video_stage": "processed", "video_metadata": metadata.as_dict(),
                "latitude": metadata.latitude, "longitude": metadata.longitude,
                "location_source": "video_metadata" if metadata.latitude is not None else "upload_metadata",
                "worldmm_output": str(worldmm_root), "worldmm_scene_count": len(result.scenes),
                "worldmm_keyframe_count": result.keyframe_count,
                "worldmm_full_keyframe_count": result.full_keyframe_count,
                "worldmm_summary_keyframe_count": result.summary_keyframe_count,
                "worldmm_selected_keyframe_count": result.selected_keyframe_count,
                "video_scene_event_ids": scene_ids,
                "derived_keyframe_asset_ids": keyframe_asset_ids, "video_processing_seconds": elapsed,
                "worldmm_device": os.getenv("SENTRIX_VIDEO_DEVICE", "cpu"),
                "vlm_device": os.getenv("SENTRIX_QWEN3_VL_DEVICE", "cpu"),
                "error_stage": None, "error": None, "retryable": True,
            })
        except Exception as error:
            data_root = Path(os.getenv("SENTRIX_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
            shutil.rmtree(data_root / "derived" / "video" / asset_id / "hybrid-webp" / "vlm-evidence", ignore_errors=True)
            return store.update_asset(asset_id, "video-processing-failed", {
                "video_stage": stage, "error_stage": stage, "error": f"{type(error).__name__}: {error}",
                "retryable": True, "video_processing_seconds": round(time.perf_counter() - started, 3),
            })

    def _process_hybrid_webp(self, asset, pipeline, metadata, captured_at, location_label, reverse_geocode, started):
        """Run the fixed hybrid extractor and import only valid WebP representatives."""
        store = pipeline.store
        asset_id = asset["id"]
        data_root = Path(os.getenv("SENTRIX_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
        output = data_root / "derived" / "video" / asset_id / "hybrid-webp"
        frames, merged, manifest = run_hybrid_keyframes(asset["path"], output, asset_id)
        if not merged or not manifest.get("image_integrity_passed"):
            raise RuntimeError("hybrid extractor produced no valid WebP representatives")
        store.update_asset(asset_id, "video-scene-importing", {
            "video_stage": "video-scene-importing", "keyframe_algorithm": manifest["method"],
            "keyframe_source_count": len(frames), "worldmm_scene_count": len(merged),
            "worldmm_keyframe_count": len(merged), "worldmm_selected_keyframe_count": len(merged),
            "memory_event_merge": True, "memory_duplicate_frame_removal": True,
        })
        scene_ids = []
        keyframe_asset_ids = []
        event_vlm_seconds = 0.0
        transient_vlm_frame_count = 0
        for scene_index, item in enumerate(merged):
            representatives = list(item.get("representatives") or [item["representative"]])
            representative = representatives[0]
            target = Path(str(representative["webp_path"])).resolve()
            if not target.is_file():
                raise RuntimeError(f"WebP representative is missing: {target}")
            start_sec = float(item["start_sec"])
            end_sec = float(item["end_sec"])
            labels = list(dict.fromkeys(item["objects"] + item["actions"] + item["expressions"]))
            evidence_records = [
                value for value in representative.get("vlm_evidence") or []
                if value.get("webp_path") and Path(str(value.get("webp_path"))).is_file()
            ][:5]
            evidence_paths = [Path(str(value["webp_path"])).resolve() for value in evidence_records]
            if target not in evidence_paths:
                evidence_paths.insert(0, target)
                evidence_records.insert(0, {
                    "webp_path": str(target),
                    "source_timestamp_sec": representative.get("source_timestamp_sec", start_sec),
                    "source_frame_index": representative.get("source_frame_index", 0),
                })
            evidence_paths = evidence_paths[:5]
            evidence_records = evidence_records[:5]
            transient_vlm_frame_count += len(evidence_paths)
            vlm_started = time.perf_counter()
            try:
                if hasattr(pipeline.gamma, "analyze_video_event"):
                    event_analysis = pipeline.gamma.analyze_video_event(
                        evidence_paths,
                        {
                            "source_video": asset.get("file_name"),
                            "start_sec": start_sec, "end_sec": end_sec,
                            "evidence_timestamps_sec": [
                                float(value.get("source_timestamp_sec", start_sec) or start_sec)
                                for value in representative.get("vlm_evidence") or []
                            ][:5],
                            "captured_at": _captured_at(captured_at, start_sec),
                            "captured_location": location_label,
                        },
                        {
                            "objects": item["objects"], "actions": item["actions"],
                            "expressions": item["expressions"],
                            "timeline": item.get("yolo_timeline") or [],
                        },
                    )
                else:
                    event_analysis = pipeline.gamma.analyze_image(target)
                fallback_index = evidence_paths.index(target) if target in evidence_paths else 0
                selected_indices = event_analysis.pop("representative_indices", None)
                if not isinstance(selected_indices, list):
                    selected_indices = [event_analysis.pop("representative_index", fallback_index)]
                normalized_indices = []
                for value in selected_indices:
                    try:
                        selected_index = max(0, min(len(evidence_paths) - 1, int(value)))
                    except (TypeError, ValueError):
                        continue
                    if selected_index not in normalized_indices:
                        normalized_indices.append(selected_index)
                event_duration = max(0.0, end_sec - start_sec)
                max_persistent = 1 if event_duration < 12.0 else 2 if event_duration < 90.0 else 3
                min_persistent = 1 if event_duration < 45.0 else 2 if event_duration < 150.0 else 3
                selected_indices = (normalized_indices or [fallback_index])[:max_persistent]
                evidence_times = [
                    float(value.get("source_timestamp_sec", start_sec) or start_sec)
                    for value in evidence_records
                ]
                if event_duration >= 45.0 and len(evidence_paths) >= 2:
                    edge_indices = [0, len(evidence_paths) - 1]
                    middle_choices = [index for index in selected_indices if index not in edge_indices]
                    selected_indices = edge_indices + middle_choices[:max(0, max_persistent - 2)]
                while len(selected_indices) < min(min_persistent, len(evidence_paths)):
                    candidates = [index for index in range(len(evidence_paths)) if index not in selected_indices]
                    if not candidates:
                        break
                    selected_indices.append(max(
                        candidates,
                        key=lambda index: min(abs(evidence_times[index] - evidence_times[chosen]) for chosen in selected_indices),
                    ))
                selected_indices = sorted(selected_indices, key=lambda index: evidence_times[index])
                # Read every chosen image before overwriting the old primary;
                # the previous primary may itself be retained as a support.
                selected_payloads = [evidence_paths[index].read_bytes() for index in selected_indices]
                representatives = []
                for ordinal, (selected_index, payload) in enumerate(zip(selected_indices, selected_payloads)):
                    selected_record = dict(evidence_records[selected_index])
                    persistent_target = target if ordinal == 0 else target.with_name(
                        f"{target.stem}_support_{ordinal:02d}{target.suffix}"
                    )
                    persistent_target.write_bytes(payload)
                    selected_record.update({
                        "webp_path": str(persistent_target), "webp_bytes": len(payload),
                        "vlm_selected_evidence_index": selected_index,
                    })
                    representatives.append(selected_record)
                representative = representatives[0]
                item["representatives"] = representatives
                item["representative"] = representative
                item["memory_keyframe_count"] = len(representatives)
            finally:
                for evidence_path in evidence_paths:
                    if evidence_path != target and evidence_path.is_file():
                        evidence_path.unlink()
            vision_seconds = time.perf_counter() - vlm_started
            event_vlm_seconds += vision_seconds
            event_analysis["_vision_seconds"] = round(vision_seconds, 4)
            event = store.create_video_scene_event({
                "scope_id": asset.get("scope_id"),
                "title": _event_title(item),
                "summary": f"{start_sec:.1f}s~{end_sec:.1f}s；合并 {item['source_frame_count']} 个片段，保留信息帧 {len(representatives)} 张",
                "time_start": _captured_at(captured_at, start_sec), "time_end": _captured_at(captured_at, end_sec),
                "place": location_label, "source_asset_id": asset_id, "source_scene_index": scene_index,
                "source_start_sec": start_sec, "source_end_sec": end_sec,
                "source_metadata": {
                    "keyframe_algorithm": manifest["method"], "memory_event_merge": True,
                    "memory_duplicate_frame_removal": True, "source_event_ids": item["source_event_ids"],
                    "source_frame_count": item["source_frame_count"], "duplicate_frame_count": item["duplicate_frame_count"],
                    "memory_keyframe_count": len(representatives), "semantic_labels": labels[:80],
                    "vlm_evidence_count": len(evidence_paths),
                    "vlm_evidence_persisted": len(representatives),
                    "vlm_selected_evidence_indices": [
                        value.get("vlm_selected_evidence_index", 0) for value in representatives
                    ],
                    "image_path": str(target), "location_source": "video_metadata",
                },
            })
            scene_ids.append(event["id"])
            for evidence_index, representative in enumerate(representatives):
                target = Path(str(representative["webp_path"])).resolve()
                keyframe_id = make_id("asset")
                provenance = {
                    "scope_id": asset.get("scope_id"), "batch_id": asset.get("batch_id"),
                    "parent_asset_id": asset_id, "derived_kind": "video_keyframe_webp",
                    "source_timestamp_sec": float(representative.get("source_timestamp_sec", start_sec) or start_sec),
                    "source_frame_index": int(representative.get("source_frame_index", 0) or 0),
                    "source_scene_index": scene_index, "evidence_index": evidence_index,
                    "captured_at": _captured_at(captured_at, float(representative.get("source_timestamp_sec", start_sec) or start_sec)), "captured_location": asset.get("captured_location"),
                    "source_device_id": metadata.device or asset.get("source_device_id"),
                    "latitude": metadata.latitude, "longitude": metadata.longitude,
                    "content_sha256": _sha256(target), "location_source": "video_metadata",
                    "keyframe_algorithm": manifest["method"], "memory_event_merge": True,
                    "memory_duplicate_frame_removal": True,
                    "source_event_ids": item["source_event_ids"], "worldmm_semantics": {
                        "objects": item["objects"], "actions": item["actions"], "expressions": item["expressions"],
                    }, "reverse_geocode": reverse_geocode,
                }
                store.create_asset(keyframe_id, target.name, "image", str(target), "image/webp", target.stat().st_size, provenance, scope_id=asset.get("scope_id"))
                processed = pipeline.process(
                    keyframe_id, summarize_event=False, forced_event_id=event["id"],
                    image_analysis=event_analysis,
                )
                if processed.get("status") != "processed":
                    raise RuntimeError(f"WebP keyframe processing failed: {keyframe_id}")
                keyframe_asset_ids.append(keyframe_id)
            pipeline.summarize_event(event["id"])
        shutil.rmtree(output / "vlm-evidence", ignore_errors=True)
        elapsed = round(time.perf_counter() - started, 3)
        return store.update_asset(asset_id, "processed", {
            "video_stage": "processed", "video_metadata": metadata.as_dict(),
            "latitude": metadata.latitude, "longitude": metadata.longitude,
            "location_source": "video_metadata" if metadata.latitude is not None else "upload_metadata",
            "worldmm_output": str(output), "keyframe_algorithm": manifest["method"],
            "worldmm_scene_count": len(merged), "worldmm_keyframe_count": sum(item.get("memory_keyframe_count", 1) for item in merged),
            "worldmm_full_keyframe_count": transient_vlm_frame_count,
            "worldmm_selected_keyframe_count": len(keyframe_asset_ids),
            "video_scene_event_ids": scene_ids, "derived_keyframe_asset_ids": keyframe_asset_ids,
            "video_processing_seconds": elapsed, "memory_event_merge": True,
            "memory_duplicate_frame_removal": True, "memory_image_integrity_passed": True,
            "event_vlm_seconds": round(event_vlm_seconds, 3),
            "transient_vlm_frame_count": transient_vlm_frame_count,
            "persistent_keyframe_count": len(keyframe_asset_ids),
            "worldmm_device": os.getenv("SENTRIX_VIDEO_DEVICE", "0"), "vlm_device": "per-keyframe-pipeline",
            "error_stage": None, "error": None, "retryable": True,
        })
