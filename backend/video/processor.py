from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..db import make_id
from ..geocoding import format_gps_prefix
from .hybrid_keyframe import run as run_hybrid_keyframes
from .metadata import probe_video_metadata
from .keyframe_package import aggregate_semantics, load_keyframe_package
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

            package_descriptor = (asset.get("metadata_json") or {}).get("keyframe_video_package")
            if package_descriptor:
                data_root = Path(os.getenv("SENTRIX_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
                preview_path = None
                preview_error = None
                try:
                    preview_path = _browser_preview(asset["path"], data_root / "derived" / "video" / asset_id / "preview.mp4", metadata.codec)
                except Exception as error:
                    preview_error = str(error)
                store.update_asset(asset_id, "video-keyframe-extracting", {
                    "browser_preview_path": preview_path,
                    "browser_preview_error": preview_error,
                })
                return self._process_keyframe_package(
                    asset, pipeline, metadata, captured_at, location_label, reverse_geocode,
                    package_descriptor, started,
                )

            if os.getenv("SENTRIX_VIDEO_NOJPEG", "0").lower() in {"1", "true", "yes", "on"}:
                compacted = self._compact_to_keyframe_video(asset, pipeline, metadata, started)
                if compacted:
                    asset, package_descriptor = compacted
                    compacted_metadata = probe_video_metadata(asset["path"])
                    compacted_metadata.captured_at = metadata.captured_at
                    compacted_metadata.creation_source = metadata.creation_source
                    data_root = Path(os.getenv("SENTRIX_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
                    preview_path = None
                    preview_error = None
                    try:
                        preview_path = _browser_preview(
                            asset["path"],
                            data_root / "derived" / "video" / asset_id / "preview.mp4",
                            compacted_metadata.codec,
                        )
                    except Exception as error:
                        preview_error = str(error)
                    asset = store.update_asset(asset_id, "video-keyframe-extracting", {
                        "browser_preview_path": preview_path,
                        "browser_preview_error": preview_error,
                    })
                    return self._process_keyframe_package(
                        asset, pipeline, compacted_metadata, captured_at, location_label, reverse_geocode,
                        package_descriptor, started,
                    )

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
            semantic_fallback_count = 0
            result_stats = (result.manifest or {}).get("stats") or {}
            result_timings = result_stats.get("timings_sec") or {}
            for scene in result.scenes:
                scene_start = _captured_at(captured_at, scene.start_sec)
                scene_end = _captured_at(captured_at, scene.end_sec)
                scene_kind = str((scene.raw or {}).get("event_kind") or "scene")
                scene_label = next((label for label in scene.semantic_labels if label), "")
                scene_title = f"事件：{scene_label}" if scene_kind == "event" and scene_label else f"场景：{scene_label}" if scene_label else f"视频场景 {scene.index + 1}"
                event = store.create_video_scene_event({
                    "scope_id": asset.get("scope_id"), "title": scene_title,
                    "summary": f"{scene_title} · {scene.start_sec:.1f}s~{scene.end_sec:.1f}s",
                    "time_start": scene_start, "time_end": scene_end, "place": location_label,
                    "source_asset_id": asset_id, "source_scene_index": scene.index,
                    "source_start_sec": scene.start_sec, "source_end_sec": scene.end_sec,
                    "source_metadata": {
                        "worldmm_scene_id": scene.scene_id, "semantic_labels": scene.semantic_labels,
                        "keyframe_count": len(scene.keyframes), "location_source": "video_metadata",
                        "keyframe_method_version": (result.manifest or {}).get("method_version", "legacy-worldmm"),
                        "event_kind": scene_kind,
                    },
                })
                scene_ids.append(event["id"])
                scene_dir = derived_root / "scenes" / f"{scene.index:03d}"
                scene_dir.mkdir(parents=True, exist_ok=True)
                for ordinal, frame in enumerate(scene.keyframes, 1):
                    suffix = Path(frame.path).suffix or ".jpg"
                    mime_type = "image/webp" if suffix.lower() == ".webp" else "image/jpeg"
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
                        "keyframe_method_version": (result.manifest or {}).get("method_version", "legacy-worldmm"),
                        "reverse_geocode": reverse_geocode,
                    }
                    store.create_asset(
                        keyframe_id, target.name, "image", str(target), mime_type, target.stat().st_size,
                        provenance, scope_id=asset.get("scope_id"),
                    )
                    processed = pipeline.process(keyframe_id, summarize_event=False, forced_event_id=event["id"])
                    if processed.get("status") != "processed":
                        error_text = (processed.get("metadata_json", {}) or {}).get("error", keyframe_id)
                        if os.getenv("SENTRIX_VIDEO_SEMANTIC_FALLBACK", "0").lower() in {"1", "true", "yes", "on"}:
                            processed = self._persist_semantic_fallback(
                                keyframe_id, event["id"], frame, pipeline, error_text,
                            )
                            semantic_fallback_count += 1
                        else:
                            raise RuntimeError(f"keyframe semantic processing failed: {error_text}")
                    keyframe_asset_ids.append(keyframe_id)
                if not semantic_fallback_count:
                    pipeline.summarize_event(event["id"])

            elapsed = round(time.perf_counter() - started, 3)
            eventagg_metadata = {}
            if os.getenv("SENTRIX_VIDEO_METHOD", "hybrid_v2").strip().lower() in {"hybrid_v2.1_eventagg", "eventagg", "eventagg_v21"}:
                eventagg_metadata = self._build_eventagg_index(
                    asset, result, keyframe_asset_ids, pipeline, captured_at,
                )
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
                "keyframe_method_version": (result.manifest or {}).get("method_version", "legacy-worldmm"),
                "keyframe_extraction_seconds": round(float(result_stats.get("total_sec") or elapsed), 3),
                "keyframe_stage_timings": result_timings,
                "worldmm_device": os.getenv("SENTRIX_VIDEO_DEVICE", "cpu"),
                "vlm_device": os.getenv("SENTRIX_QWEN3_VL_DEVICE", "cpu"),
                "semantic_fallback_count": semantic_fallback_count,
                "semantic_fallback_enabled": os.getenv("SENTRIX_VIDEO_SEMANTIC_FALLBACK", "0").lower() in {"1", "true", "yes", "on"},
                **eventagg_metadata,
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

    def _build_eventagg_index(self, asset, result, keyframe_asset_ids, pipeline, captured_at):
        """Build the v2.1 event index after v2.0 evidence has been persisted.

        This post-layer is intentionally additive: baseline video_scene events,
        WebP assets and observations remain available for rollback.  A failure
        records ``eventagg_status=failed`` and leaves the baseline usable.
        """
        started = time.perf_counter()
        from .event_aggregator import EVENTAGG_METHOD_VERSION, DINOEmbedder, EventAggregator

        data_root = Path(os.getenv("SENTRIX_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
        semantic_path = Path(str((result.manifest or {}).get("semantic") or ""))
        if not semantic_path.is_file():
            return {"eventagg_status": "failed", "eventagg_fallback": "baseline", "eventagg_error": "semantic.json missing"}
        package = json.loads(semantic_path.read_text(encoding="utf-8"))
        frame_assets = [pipeline.store.get_asset(item) for item in keyframe_asset_ids]
        records = EventAggregator.records_from_package(package, frame_assets)
        cache_dir = Path(os.getenv("SENTRIX_EVENTAGG_CACHE", str(data_root / "cache" / "eventagg")))
        embedder = DINOEmbedder(cache_dir=cache_dir, device=os.getenv("SENTRIX_EVENTAGG_DEVICE"))
        embeddings, embedding_metrics = embedder.embed(asset["id"], records, int(os.getenv("SENTRIX_EVENTAGG_BATCH_SIZE", "16")))
        for record in records:
            record.visual_embedding = embeddings.get(record.frame_id, [])
        threshold = float(os.getenv("SENTRIX_EVENTAGG_MERGE_THRESHOLD", "0.68"))
        aggregator = EventAggregator({"merge_threshold": threshold}, cache_dir=cache_dir)
        groups, metrics = aggregator.aggregate(records, asset["id"], threshold=threshold)
        metrics.update(embedding_metrics)
        metrics["baseline_scene_count"] = len(keyframe_asset_ids)
        metrics["eventagg_wall_seconds"] = round(time.perf_counter() - started, 4)
        run_id = f"eventagg_{asset['id']}_upload"
        events = []
        for group in groups:
            data = EventAggregator.group_to_dict(group)
            data["event_id"] = f"{run_id}_{data['event_id'].rsplit('_', 1)[-1]}"
            data["start_time"] = _captured_at(captured_at, data["start_sec"])
            data["end_time"] = _captured_at(captured_at, data["end_sec"])
            data["member_frames"] = [
                {"frame_id": item.frame_id, "timestamp_sec": item.timestamp, "frame_hash": item.frame_hash}
                for item in group.members
            ]
            events.append(data)
        pipeline.store.replace_eventagg_run(
            run_id, asset["id"], asset.get("scope_id") or "home-default", EVENTAGG_METHOD_VERSION,
            events, metrics, {**aggregator.config, "cache_dir": str(cache_dir)},
        )
        return {
            "eventagg_status": "completed", "eventagg_fallback": None,
            "eventagg_run_id": run_id, "eventagg_event_count": len(events),
            "eventagg_metrics": metrics, "eventagg_memory_build_seconds": metrics["eventagg_wall_seconds"],
            "eventagg_method_version": EVENTAGG_METHOD_VERSION,
        }

    def _persist_semantic_fallback(self, asset_id, event_id, frame, pipeline, error_text):
        """Persist detector semantics when the optional VLM is unavailable.

        The keyframe package already contains YOLO/Pose objects and actions.
        Keeping that evidence makes video import useful and deterministic when
        the model service is down, while recording the exact degraded reason.
        A later semantic re-enrichment can replace this observation.
        """
        store = pipeline.store
        objects = list(frame.objects or [])
        actions = list(frame.actions or [])
        expressions = list(frame.expressions or [])
        labels = list(dict.fromkeys(
            [str(item.get("label") or "") for item in objects + actions + expressions if isinstance(item, dict)]
        ))
        label_text = "、".join(item for item in labels if item)
        caption = f"检测到：{label_text}" if label_text else "视频关键帧"
        people = ["人物"] if any(item.lower() == "person" for item in labels) else []
        observation = store.add_observation(asset_id, {
            "captured_at": _captured_at(None, frame.timestamp_sec),
            "source_type": "video_yolo_semantic_fallback",
            "caption": caption,
            "activity": "、".join(str(item.get("label") or "") for item in actions if isinstance(item, dict)),
            "place": "",
            "people": people,
            "objects": objects,
            "event_type": "video_scene",
            "confidence": 0.5,
            "canonical": {
                "caption": caption, "semantic_status": "detector_fallback",
                "objects": objects, "actions": actions, "expressions": expressions,
            },
            "raw": {
                "worldmm": frame.raw or {},
                "semantic_fallback": True,
                "semantic_fallback_error": str(error_text or "")[-1000:],
            },
        })
        store.attach_observation_to_event(event_id, observation["id"])
        entity_ids = [item["id"] for item in store.maintain_observation_entities(observation["id"], event_id)]
        metadata = {
            "observation_id": observation["id"], "event_id": event_id,
            "entity_ids": entity_ids, "semantic_status": "detector_fallback",
            "semantic_fallback": True, "semantic_fallback_error": str(error_text or "")[-1000:],
        }
        return store.update_asset(asset_id, "processed", metadata)

    def _compact_to_keyframe_video(self, asset, pipeline, metadata, started):
        """Run KATNA and in-memory YOLO, then atomically replace the source."""
        asset_id = asset["id"]
        data_root = Path(os.getenv("SENTRIX_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
        output = data_root / "derived" / "video" / asset_id / "nojpeg"
        output.mkdir(parents=True, exist_ok=True)
        script = Path(__file__).resolve().parents[2] / "tools" / "video_keyframe" / "katna" / "run_katna_nojpeg.py"
        if not script.is_file():
            return None
        root = self.worldmm.root
        yolo = Path(os.getenv("SENTRIX_VIDEO_YOLO_MODEL", root / "models/keyframe/yolo11n.pt"))
        pose = Path(os.getenv("SENTRIX_VIDEO_POSE_MODEL", root / "models/keyframe/yolo11n-pose.pt"))
        katna_root = Path(os.getenv("SENTRIX_KEYFRAME_KATNA_ROOT", "/home/sscy/GitHub/hpq/katna-bench-20260817/katna"))
        env = os.environ.copy()
        env.setdefault("SENTRIX_KEYFRAME_KATNA_ROOT", str(katna_root))
        env.setdefault("SENTRIX_KEYFRAME_PIPELINE_ROOT", str(root))
        command = [
            os.getenv("SENTRIX_VIDEO_PYTHON", sys.executable), str(script),
            "--video", str(Path(asset["path"]).resolve()), "--video-id", asset_id,
            "--output", str(output), "--katna-engine", os.getenv("SENTRIX_KATNA_ENGINE", "gpu"),
            "--katna-resize", os.getenv("SENTRIX_KATNA_RESIZE", "384"),
            "--katna-chunk", os.getenv("SENTRIX_KATNA_CHUNK", "500"),
            "--semantic-width", os.getenv("SENTRIX_KATNA_SEMANTIC_WIDTH", "640"),
            "--device", os.getenv("SENTRIX_VIDEO_DEVICE", "0"),
            "--yolo-model", str(yolo), "--pose-model", str(pose),
        ]
        timeout = int(os.getenv("SENTRIX_VIDEO_TIMEOUT_SECONDS", "7200"))
        process = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout, env=env)
        (output / "sentrix-katna-nojpeg.log").write_text(
            process.stdout + ("\nSTDERR\n" + process.stderr if process.stderr else ""), encoding="utf-8")
        if process.returncode != 0:
            raise RuntimeError(f"KATNA no-JPEG pipeline failed ({process.returncode}): {process.stderr.strip()[-1200:]}")
        encoded = output / "selected_frames.hevc.mp4"
        frame_map = output / "frame_map.json"
        semantic = output / "semantic.json"
        if not encoded.is_file() or not frame_map.is_file() or not semantic.is_file():
            raise RuntimeError("KATNA no-JPEG pipeline did not produce a complete keyframe package")

        target = Path(asset["path"]).resolve()
        os.replace(encoded, target)
        package_dir = target.parent
        map_target = package_dir / "frame_map.json"
        semantic_target = package_dir / "semantic.json"
        shutil.move(str(frame_map), str(map_target))
        shutil.move(str(semantic), str(semantic_target))
        pipeline.store.update_asset_file(
            asset_id, target, mime_type="video/mp4", size_bytes=target.stat().st_size,
            content_sha256=_sha256(target),
        )
        descriptor = {
            "frame_map_path": str(map_target), "semantic_path": str(semantic_target),
            "encoded_fps": 25, "source_fps": metadata.fps,
            "source_width": metadata.width, "source_height": metadata.height,
            "source_duration_sec": metadata.duration_sec,
        }
        updated = pipeline.store.update_asset(asset_id, "video-keyframe-extracting", {
            "keyframe_video": True, "keyframe_video_package": descriptor,
            "source_video_duration_sec": metadata.duration_sec,
            "source_video_size_bytes": asset.get("size_bytes"),
            "video_stage": "video-keyframe-extracting",
        })
        return updated, descriptor

    def _process_keyframe_package(
        self, asset, pipeline, metadata, captured_at, location_label, reverse_geocode,
        package_descriptor, started,
    ):
        """Import precomputed keyframes without JPEG assets or model reruns."""
        if os.getenv("SENTRIX_VIDEO_WEBP_MEMORY", "0").lower() in {"1", "true", "yes", "on"}:
            return self._process_keyframe_package_webp_memory(
                asset, pipeline, metadata, captured_at, location_label, reverse_geocode,
                package_descriptor, started,
            )
        store = pipeline.store
        asset_id = asset["id"]
        package = load_keyframe_package(package_descriptor)
        frames = package["frames"]
        objects, actions, expressions, labels = aggregate_semantics(frames)
        first_timestamp = float(frames[0]["source_timestamp_sec"])
        last_timestamp = float(frames[-1]["source_timestamp_sec"])
        duration = max(float(metadata.duration_sec or 0), last_timestamp)
        store.update_asset(asset_id, "video-scene-importing", {
            "video_stage": "video-scene-importing",
            "keyframe_video": True,
            "keyframe_video_frame_count": len(frames),
            "keyframe_video_encoded_fps": package["encoded_fps"],
            "keyframe_video_source_duration_sec": duration,
            "worldmm_scene_count": 1,
            "worldmm_keyframe_count": len(frames),
            "worldmm_full_keyframe_count": len(frames),
            "worldmm_summary_keyframe_count": len(frames),
            "worldmm_selected_keyframe_count": len(frames),
        })
        scene_event = store.create_video_scene_event({
            "scope_id": asset.get("scope_id"),
            "title": "关键帧视频场景",
            "summary": f"关键帧视频包含 {len(frames)} 个关键画面",
            "time_start": _captured_at(captured_at, first_timestamp),
            "time_end": _captured_at(captured_at, last_timestamp),
            "place": location_label,
            "source_asset_id": asset_id,
            "source_scene_index": 0,
            "source_start_sec": first_timestamp,
            "source_end_sec": last_timestamp,
            "source_metadata": {
                "keyframe_video": True,
                "keyframe_count": len(frames),
                "encoded_fps": package["encoded_fps"],
                "source_duration_sec": duration,
                "semantic_labels": labels[:80],
                "keyframe_map_path": package["frame_map_path"],
                "semantic_path": package["semantic_path"],
                "location_source": "video_metadata",
            },
        })
        observation = store.add_observation(asset_id, {
            "captured_at": _captured_at(captured_at, first_timestamp),
            "source_type": "video_keyframe_package",
            "caption": f"关键帧视频，共 {len(frames)} 个关键画面",
            "activity": "；".join(item["label"] for item in actions[:5]) or "视频关键画面",
            "place": location_label,
            "objects": objects,
            "event_type": "视频关键帧事件",
            "confidence": 0.8,
            "raw": {
                "keyframe_video": True,
                "frame_count": len(frames),
                "encoded_fps": package["encoded_fps"],
                "source_fps": package["source_fps"],
                "labels": labels[:80],
                "actions": actions,
                "expressions": expressions,
            },
            "canonical": {"objects": objects, "actions": actions, "expressions": expressions},
        })
        store.attach_observation_to_event(scene_event["id"], observation["id"])
        # This is intentionally a deterministic summary: the semantic package
        # already contains YOLO/Pose output, so importing it must not invoke the
        # VLM once per frame.
        summary_labels = "、".join(labels[:8]) or "暂无稳定目标标签"
        store.update_event(scene_event["id"], {
            "title": f"关键帧视频：{summary_labels}"[:80],
            "event_type": "视频关键帧事件",
            "activity": "、".join(item["label"] for item in actions[:5]) or "视频关键画面",
            "summary": f"保留 {len(frames)} 个关键画面；检测到：{summary_labels}。",
        })
        elapsed = round(time.perf_counter() - started, 3)
        return store.update_asset(asset_id, "processed", {
            "video_stage": "processed",
            "video_metadata": metadata.as_dict(),
            "latitude": metadata.latitude, "longitude": metadata.longitude,
            "location_source": "video_metadata" if metadata.latitude is not None else "upload_metadata",
            "worldmm_scene_count": 1,
            "worldmm_keyframe_count": len(frames),
            "worldmm_full_keyframe_count": len(frames),
            "worldmm_summary_keyframe_count": len(frames),
            "worldmm_selected_keyframe_count": len(frames),
            "video_scene_event_ids": [scene_event["id"]],
            "derived_keyframe_asset_ids": [],
            "video_processing_seconds": elapsed,
            "keyframe_video_import": True,
            "worldmm_device": "precomputed",
            "vlm_device": "not-run",
            "error_stage": None, "error": None, "retryable": True,
        })

    def _process_keyframe_package_webp_memory(
        self, asset, pipeline, metadata, captured_at, location_label, reverse_geocode,
        package_descriptor, started,
    ):
        """Build one WebP/VLM observation per detected event or fallback scene."""
        import cv2

        store = pipeline.store
        asset_id = asset["id"]
        package = load_keyframe_package(package_descriptor)
        frames = package["frames"]
        if not frames:
            raise RuntimeError("keyframe package contains no frames")
        package_events = {str(item.get("event_id")): item for item in package.get("events") or [] if isinstance(item, dict) and item.get("event_id")}
        groups = {}
        for index, frame in enumerate(frames):
            event_id = str(frame.get("event_id") or f"scene_{index:05d}")
            groups.setdefault(event_id, []).append(frame)
        event_specs = []
        for event_id, members in groups.items():
            spec = dict(package_events.get(event_id) or {})
            spec.setdefault("event_id", event_id)
            spec.setdefault("kind", "event" if any(member.get("actions") for member in members) else "scene")
            spec.setdefault("label", next((member.get("event_label") for member in members if member.get("event_label")), "场景"))
            spec.setdefault("start_sec", min(float(member.get("event_start_sec") or member.get("source_timestamp_sec") or 0) for member in members))
            spec.setdefault("end_sec", max(float(member.get("event_end_sec") or member.get("source_timestamp_sec") or 0) for member in members))
            spec.setdefault("sample_count", len(members))
            spec.setdefault("substituted_sample_count", max(0, int(spec.get("sample_count") or len(members)) - len(members)))
            event_specs.append(spec)
        event_specs.sort(key=lambda item: float(item.get("start_sec") or 0))
        selected = [min(groups[str(spec["event_id"])], key=lambda item: float(item.get("source_timestamp_sec") or 0)) for spec in event_specs]
        objects, actions, expressions, labels = aggregate_semantics(frames)
        first_timestamp = float(frames[0].get("source_timestamp_sec", 0) or 0)
        last_timestamp = float(frames[-1].get("source_timestamp_sec", 0) or 0)
        duration = max(float(metadata.duration_sec or 0), last_timestamp)
        data_root = Path(os.getenv("SENTRIX_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
        derived_root = data_root / "derived" / "video" / asset_id
        webp_root = derived_root / "webp-memory"
        webp_root.mkdir(parents=True, exist_ok=True)
        scene_events = {}
        for scene_index, spec in enumerate(event_specs):
            start_sec = float(spec.get("start_sec") or first_timestamp)
            end_sec = float(spec.get("end_sec") or start_sec)
            semantic_labels = list(dict.fromkeys(list(spec.get("objects") or []) + list(spec.get("actions") or [])))
            title_prefix = "事件" if spec.get("kind") == "event" else "场景"
            scene_events[str(spec["event_id"])] = store.create_video_scene_event({
                "scope_id": asset.get("scope_id"),
                "title": f"{title_prefix}：{spec.get('label') or '场景'}",
                "summary": f"{start_sec:.1f}s~{end_sec:.1f}s；语义采样 {int(spec.get('sample_count') or 1)} 帧，代表帧 1 张",
                "time_start": _captured_at(captured_at, start_sec),
                "time_end": _captured_at(captured_at, end_sec),
                "place": location_label, "source_asset_id": asset_id,
                "source_scene_index": scene_index, "source_start_sec": start_sec, "source_end_sec": end_sec,
                "source_metadata": {
                    "keyframe_video": True, "memory_image_format": "webp", "event_kind": spec.get("kind"),
                    "keyframe_count": int(spec.get("sample_count") or 1), "memory_keyframe_count": 1,
                    "semantic_substitution": True, "substituted_sample_count": int(spec.get("substituted_sample_count") or 0),
                    "encoded_fps": package["encoded_fps"], "source_duration_sec": duration,
                    "semantic_labels": semantic_labels[:80], "keyframe_map_path": package["frame_map_path"],
                    "semantic_path": package["semantic_path"], "location_source": "video_metadata",
                },
            })
        capture = cv2.VideoCapture(str(asset["path"]))
        if not capture.isOpened():
            raise RuntimeError(f"unable to decode keyframe video: {asset['path']}")
        # Multiple events can legitimately select the same source frame after
        # full-resolution refinement.  Keep all representatives instead of
        # letting a dict assignment silently overwrite earlier events.
        selected_by_index = {}
        for index, frame in enumerate(selected):
            source_index = int((frame.get("source_frame_index") if package.get("direct_source") else frame.get("encoded_frame_index", index)) or 0)
            selected_by_index.setdefault(source_index, []).append(frame)
        keyframe_asset_ids = []
        encoded_index = 0
        try:
            while selected_by_index:
                ok, image = capture.read()
                if not ok:
                    break
                frames_at_index = selected_by_index.pop(encoded_index, [])
                for frame in frames_at_index:
                    target = webp_root / f"kf_{int(frame.get('encoded_frame_index', encoded_index)):06d}.webp"
                    quality = max(1, min(100, int(os.getenv("SENTRIX_VIDEO_WEBP_QUALITY", "80"))))
                    encoded, buffer = cv2.imencode(".webp", image, [cv2.IMWRITE_WEBP_QUALITY, quality])
                    if not encoded:
                        raise RuntimeError(f"WebP encode failed at frame {encoded_index}")
                    target.write_bytes(buffer.tobytes())
                    timestamp = float(frame.get("source_timestamp_sec", 0) or 0)
                    event_key = str(frame.get("event_id") or "")
                    event_spec = next((item for item in event_specs if str(item.get("event_id")) == event_key), {})
                    scene_event = scene_events[event_key]
                    provenance = {
                        "scope_id": asset.get("scope_id"), "batch_id": asset.get("batch_id"),
                        "parent_asset_id": asset_id, "derived_kind": "video_keyframe_webp",
                        "source_timestamp_sec": timestamp,
                        "source_frame_index": int(frame.get("source_frame_index", 0) or 0),
                        "encoded_frame_index": int(frame.get("encoded_frame_index", encoded_index) or 0),
                        "source_scene_index": int(scene_event.get("source_scene_index") or 0), "captured_at": _captured_at(captured_at, timestamp),
                        "captured_location": asset.get("captured_location"),
                        "source_device_id": metadata.device or asset.get("source_device_id"),
                        "latitude": metadata.latitude, "longitude": metadata.longitude,
                        "content_sha256": _sha256(target), "location_source": "video_metadata",
                        "worldmm_semantics": {"objects": frame.get("objects", []), "actions": frame.get("actions", []), "expressions": frame.get("expressions", [])},
                        "reverse_geocode": reverse_geocode, "memory_image_format": "webp",
                        "event_id": event_key, "event_kind": event_spec.get("kind"),
                        "event_label": event_spec.get("label"), "semantic_substitution": True,
                        "substituted_sample_count": int(event_spec.get("substituted_sample_count") or 0),
                    }
                    keyframe_id = make_id("asset")
                    store.create_asset(
                        keyframe_id, target.name, "image", str(target), "image/webp", target.stat().st_size,
                        provenance, scope_id=asset.get("scope_id"),
                    )
                    processed = pipeline.process(keyframe_id, summarize_event=False, forced_event_id=scene_event["id"])
                    if processed.get("status") != "processed":
                        raise RuntimeError(f"WebP semantic processing failed: {processed.get('metadata_json', {}).get('error', keyframe_id)}")
                    keyframe_asset_ids.append(keyframe_id)
                encoded_index += 1
        finally:
            capture.release()
        if selected_by_index:
            remaining = sum(len(items) for items in selected_by_index.values())
            raise RuntimeError(f"keyframe video ended before {remaining} WebP frames were decoded")
        # The representative image has already gone through VLLM in
        # pipeline.process().  Keep the interval/object/action summary from
        # the detector package so each event makes exactly one VLLM image call.
        elapsed = round(time.perf_counter() - started, 3)
        elapsed = round(time.perf_counter() - started, 3)
        return store.update_asset(asset_id, "processed", {
            "video_stage": "processed", "video_metadata": metadata.as_dict(),
            "latitude": metadata.latitude, "longitude": metadata.longitude,
            "location_source": "video_metadata" if metadata.latitude is not None else "upload_metadata",
            "worldmm_scene_count": len(event_specs), "worldmm_keyframe_count": len(frames),
            "worldmm_full_keyframe_count": len(frames), "worldmm_summary_keyframe_count": len(frames),
            "worldmm_selected_keyframe_count": len(selected),
            "video_scene_event_ids": [item["id"] for item in scene_events.values()],
            "derived_keyframe_asset_ids": keyframe_asset_ids,
            "video_processing_seconds": elapsed, "keyframe_video_import": True,
            "keyframe_video_memory": True, "keyframe_video_memory_format": "webp",
            "keyframe_video_memory_frame_count": len(keyframe_asset_ids),
            "worldmm_device": "precomputed", "vlm_device": os.getenv("SENTRIX_QWEN3_VL_DEVICE", "cuda:0"),
            "error_stage": None, "error": None, "retryable": True,
        })
