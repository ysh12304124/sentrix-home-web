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
        for scene_index, item in enumerate(merged):
            representatives = list(item.get("representatives") or [item["representative"]])
            representative = representatives[0]
            target = Path(str(representative["webp_path"])).resolve()
            if not target.is_file():
                raise RuntimeError(f"WebP representative is missing: {target}")
            start_sec = float(item["start_sec"])
            end_sec = float(item["end_sec"])
            labels = list(dict.fromkeys(item["objects"] + item["actions"] + item["expressions"]))
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
                processed = pipeline.process(keyframe_id, summarize_event=False, forced_event_id=event["id"])
                if processed.get("status") != "processed":
                    raise RuntimeError(f"WebP keyframe processing failed: {keyframe_id}")
                keyframe_asset_ids.append(keyframe_id)
            pipeline.summarize_event(event["id"])
        elapsed = round(time.perf_counter() - started, 3)
        return store.update_asset(asset_id, "processed", {
            "video_stage": "processed", "video_metadata": metadata.as_dict(),
            "latitude": metadata.latitude, "longitude": metadata.longitude,
            "location_source": "video_metadata" if metadata.latitude is not None else "upload_metadata",
            "worldmm_output": str(output), "keyframe_algorithm": manifest["method"],
            "worldmm_scene_count": len(merged), "worldmm_keyframe_count": sum(item.get("memory_keyframe_count", 1) for item in merged),
            "worldmm_full_keyframe_count": len(frames), "worldmm_selected_keyframe_count": sum(item.get("memory_keyframe_count", 1) for item in merged),
            "video_scene_event_ids": scene_ids, "derived_keyframe_asset_ids": keyframe_asset_ids,
            "video_processing_seconds": elapsed, "memory_event_merge": True,
            "memory_duplicate_frame_removal": True, "memory_image_integrity_passed": True,
            "worldmm_device": os.getenv("SENTRIX_VIDEO_DEVICE", "0"), "vlm_device": "per-keyframe-pipeline",
            "error_stage": None, "error": None, "retryable": True,
        })
