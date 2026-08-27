from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..db import make_id
from ..geocoding import format_gps_prefix
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


def _semantic_scene_transition(event_analysis, evidence_paths, selected_indices, max_persistent):
    """Add endpoint evidence only for an algorithmic visual transition.

    The event VLM sees several ordered candidate images.  Its selected index
    list is authoritative for ordinary repeated views, but small models can
    still return one index across a visible camera/environment transition. In
    that case, use the first and last candidates only when generic image
    comparison confirms the change. GPS, place names, and content-specific
    vocabularies are intentionally absent from this rule.
    """
    evidence_count = len(evidence_paths or [])
    if evidence_count < 2 or len(selected_indices) >= 2:
        return selected_indices, False
    relation = "unknown"
    visual_signal = False
    try:
        from .hybrid_keyframe import _background_relation, _visual_distance
        visual_inputs_valid = all(Path(str(path)).is_file() for path in (evidence_paths[0], evidence_paths[-1]))
        if visual_inputs_valid:
            relation = _background_relation(
                {"webp_path": str(evidence_paths[0])},
                {"webp_path": str(evidence_paths[-1])},
            )
        visual_signal = visual_inputs_valid and relation == "environment_change"
        if visual_inputs_valid and relation == "same_environment_view_change":
            visual_signal = _visual_distance(evidence_paths[0], evidence_paths[-1]) >= 0.90
    except Exception:
        visual_signal = False
    if not visual_signal:
        return selected_indices, False
    forced = list(dict.fromkeys([0, evidence_count - 1, *selected_indices]))
    return sorted(forced[:max_persistent]), True


def _timeline_people_count(timeline, timestamp):
    """Return the nearest detector count for an evidence timestamp."""
    nearest = _timeline_row(timeline, timestamp)
    if nearest is None:
        return None
    try:
        if nearest.get("people_count") is not None:
            return int(nearest["people_count"])
    except (TypeError, ValueError):
        pass
    labels = {str(value).strip().lower() for value in nearest.get("labels") or []}
    return 1 if labels & {"person", "people", "child", "children", "人", "儿童", "小孩"} else 0


def _timeline_row(timeline, timestamp):
    """Return the nearest coarse semantic row for an evidence timestamp."""
    nearest = None
    nearest_delta = None
    for row in timeline or []:
        if not isinstance(row, dict):
            continue
        try:
            row_time = float(row.get("sec", row.get("timestamp", 0)) or 0)
        except (TypeError, ValueError):
            continue
        delta = abs(row_time - float(timestamp))
        if nearest_delta is None or delta < nearest_delta:
            nearest, nearest_delta = row, delta
    return nearest


def _collapse_same_view_people_change(evidence_paths, evidence_records, selected_indices, timeline):
    """Collapse a same-view selection to one information-dense frame.

    This is deliberately a visual-background rule.  It does not use GPS,
    place, or the event's merged people list. People count is the primary
    score; visible detector elements and encoded sharpness break ties. This
    also handles no-person scenes where a single frame has richer content.
    """
    if len(selected_indices) < 2 or len(evidence_paths) < 2:
        return selected_indices, False
    try:
        from .hybrid_keyframe import _background_relation
        relation = _background_relation(
            {"webp_path": str(evidence_paths[0])},
            {"webp_path": str(evidence_paths[-1])},
        )
    except Exception:
        return selected_indices, False
    if relation != "same_view":
        return selected_indices, False
    scores = {}
    for index in selected_indices:
        timestamp = (evidence_records[index] or {}).get("source_timestamp_sec", 0)
        row = _timeline_row(timeline, timestamp) or {}
        people = _timeline_people_count(timeline, timestamp)
        try:
            elements = int(row.get("element_count", len(row.get("labels") or [])) or 0)
        except (TypeError, ValueError):
            elements = len(row.get("labels") or [])
        try:
            bytes_score = Path(str(evidence_paths[index])).stat().st_size / 250000.0
        except OSError:
            bytes_score = 0.0
        scores[index] = (people if people is not None else 0, elements, bytes_score)
    if len(scores) < 2:
        return selected_indices, False
    best = max(selected_indices, key=lambda index: (*scores[index], -index))
    return [best], True


def _boundary_evidence(item, side):
    """Return the closest transient evidence image to a proposed boundary."""
    representative = item.get("representative") or {}
    records = [value for value in representative.get("vlm_evidence") or [] if value.get("webp_path")]
    if records:
        records.sort(key=lambda value: float(value.get("source_timestamp_sec", 0) or 0))
        record = records[-1] if side == "right" else records[0]
        path = Path(str(record["webp_path"])).resolve()
        if path.is_file():
            return path
    path = Path(str(representative.get("webp_path") or "")).resolve()
    return path if path.is_file() else None


def _review_scene_boundaries(merged, gamma):
    """Let the vision model audit proposed boundaries before frame selection.

    The prefilter remains the recall-oriented proposal stage.  The model is a
    second, boundary-aware judge: it can join adjacent states when the view
    and background are continuous, even if an object/action label changed.
    """
    if len(merged) < 2 or not hasattr(gamma, "review_video_scene_boundary"):
        return merged, []
    reviewed = []
    result = []
    for current in merged:
        if not result:
            result.append(current)
            continue
        previous = result[-1]
        left_path = _boundary_evidence(previous, "right")
        right_path = _boundary_evidence(current, "left")
        review = None
        same_visual_view = False
        relation = "unknown"
        try:
            from .hybrid_keyframe import _background_relation
            if left_path and right_path:
                relation = _background_relation(
                    {"webp_path": str(left_path)}, {"webp_path": str(right_path)},
                )
                same_visual_view = relation != "environment_change"
        except Exception:
            relation = "unknown"
        if left_path and right_path and same_visual_view:
            try:
                review = gamma.review_video_scene_boundary(
                    left_path, right_path,
                    {"left_end_sec": previous.get("end_sec"), "right_start_sec": current.get("start_sec")},
                )
            except Exception as error:
                review = {"same_scene": None, "error": str(error)}
        approved = bool(
            review and review.get("same_scene") is True
            and (
                float(review.get("confidence", 0) or 0) >= 0.55
                or float(review.get("background_continuity", 0) or 0) >= 0.60
            )
        )
        reviewed.append({
            "left_event_id": previous.get("event_id"),
            "right_event_id": current.get("event_id"),
            "visual_relation": relation,
            "model": review or {"same_scene": None, "reason": "no usable boundary evidence"},
            "merged": approved,
        })
        if not approved:
            result.append(current)
            continue
        previous["end_sec"] = current.get("end_sec", previous.get("end_sec"))
        previous["source_event_ids"] = list(dict.fromkeys(
            list(previous.get("source_event_ids") or []) + list(current.get("source_event_ids") or [])
        ))
        previous["source_frame_count"] = int(previous.get("source_frame_count") or 0) + int(current.get("source_frame_count") or 0)
        previous["duplicate_frame_count"] = int(previous.get("duplicate_frame_count") or 0) + int(current.get("duplicate_frame_count") or 0)
        previous["objects"] = list(dict.fromkeys(list(previous.get("objects") or []) + list(current.get("objects") or [])))
        previous["actions"] = list(dict.fromkeys(list(previous.get("actions") or []) + list(current.get("actions") or [])))
        previous["expressions"] = list(dict.fromkeys(list(previous.get("expressions") or []) + list(current.get("expressions") or [])))
        previous["yolo_timeline"] = list(previous.get("yolo_timeline") or []) + list(current.get("yolo_timeline") or [])
        left_ev = (previous.get("representative") or {}).get("vlm_evidence") or []
        right_ev = (current.get("representative") or {}).get("vlm_evidence") or []
        combined = {str(value.get("webp_path")): value for value in (left_ev + right_ev) if value.get("webp_path")}
        combined_values = sorted(combined.values(), key=lambda value: float(value.get("source_timestamp_sec", 0) or 0))
        if previous.get("representative") is not None:
            previous["representative"]["vlm_evidence"] = combined_values[-5:]
        previous["representatives"] = [previous.get("representative")]
        previous["boundary_model_review"] = list(previous.get("boundary_model_review") or []) + [review]
    return result, reviewed


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


def _frame_analysis_from_event(event_analysis, evidence_index):
    """Ground one retained image in its own window observation, not the whole event."""
    observation = None
    for value in event_analysis.get("frame_observations") or []:
        if not isinstance(value, dict):
            continue
        try:
            if int(value.get("index")) == int(evidence_index):
                observation = value
                break
        except (TypeError, ValueError):
            continue
    if observation is None:
        return event_analysis
    frame_analysis = dict(event_analysis)
    for key in ("caption", "activity", "place"):
        frame_analysis[key] = str(observation.get(key) or "")
    frame_analysis["people"] = list(observation.get("people") or [])
    frame_analysis["objects"] = list(observation.get("objects") or [])
    frame_analysis["clothing"] = []
    frame_analysis["emotions"] = []
    frame_analysis["spatial_relations"] = []
    frame_analysis["ocr_text"] = ""
    frame_analysis["semantic"] = {}
    frame_analysis["facts"] = []
    frame_analysis["detail"] = {
        "schema_version": 1, "visible_details": [], "regions": [], "text_blocks": [],
        "uncertainties": ["该观察仅对应当前保留帧"],
    }
    frame_analysis.pop("representative_indices", None)
    frame_analysis.pop("coverage_required_indices", None)
    return frame_analysis


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
    def __init__(self, worldmm=None, keyframe_algorithm=None):
        self.worldmm = worldmm or WorldMMAdapter()
        self.keyframe_algorithm = keyframe_algorithm

    def _keyframe_algorithm(self):
        return str(
            self.keyframe_algorithm
            or os.getenv("SENTRIX_VIDEO_KEYFRAME_ALGORITHM", "worldmm")
        ).strip().lower()

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

            data_root = Path(os.getenv("SENTRIX_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
            derived_root = data_root / "derived" / "video" / asset_id
            preview_path = None
            preview_error = None
            try:
                preview_path = _browser_preview(asset["path"], derived_root / "preview.mp4", metadata.codec)
            except Exception as error:
                preview_error = str(error)
            store.update_asset(asset_id, "video-keyframe-extracting", {
                "browser_preview_path": preview_path, "browser_preview_error": preview_error,
            })

            algorithm = self._keyframe_algorithm()
            if algorithm == "hybrid_webp":
                return self._process_hybrid_webp(
                    asset, pipeline, metadata, captured_at, location_label, reverse_geocode, started,
                )
            if algorithm == "mlt_semantic":
                return self._process_mlt_semantic(
                    asset, pipeline, metadata, captured_at, location_label, reverse_geocode, started,
                )
            if algorithm != "worldmm":
                raise ValueError(f"unsupported video keyframe algorithm: {algorithm}")

            worldmm_root = derived_root / "worldmm"
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
            shutil.rmtree(data_root / "derived" / "video" / asset_id / "mlt-semantic" / "vlm-evidence", ignore_errors=True)
            return store.update_asset(asset_id, "video-processing-failed", {
                "video_stage": stage, "error_stage": stage, "error": f"{type(error).__name__}: {error}",
                "retryable": True, "video_processing_seconds": round(time.perf_counter() - started, 3),
            })

    def _process_hybrid_webp(self, asset, pipeline, metadata, captured_at, location_label, reverse_geocode, started):
        return self._process_webp_keyframes(
            asset, pipeline, metadata, captured_at, location_label, reverse_geocode, started,
            extractor="hybrid_webp",
        )

    def _process_mlt_semantic(self, asset, pipeline, metadata, captured_at, location_label, reverse_geocode, started):
        return self._process_webp_keyframes(
            asset, pipeline, metadata, captured_at, location_label, reverse_geocode, started,
            extractor="mlt_semantic",
        )

    def _process_webp_keyframes(self, asset, pipeline, metadata, captured_at, location_label,
                                reverse_geocode, started, extractor):
        """Extract scenes, reconcile overlapping 3-5-frame VLM windows, then import memories."""

        store = pipeline.store
        asset_id = asset["id"]
        data_root = Path(os.getenv("SENTRIX_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
        window_vlm_seconds = 0.0
        if extractor == "mlt_semantic":
            from .mlt_keyframe import merge_and_analyze_windows, run as run_mlt_keyframes
            output = data_root / "derived" / "video" / asset_id / "mlt-semantic"
            frames, merged, manifest = run_mlt_keyframes(asset["path"], output, asset_id)
            preliminary_count = len(merged)
            window_vlm_started = time.perf_counter()
            merged, merge_stats = merge_and_analyze_windows(
                merged, pipeline.gamma,
                {"source_video": asset.get("file_name"), "video_id": asset_id},
                max_window=int(os.getenv("SENTRIX_VIDEO_MLT_VLM_WINDOW", "5")),
                stride=int(os.getenv("SENTRIX_VIDEO_MLT_VLM_STRIDE", "4")),
            )
            window_vlm_seconds = time.perf_counter() - window_vlm_started
            manifest.update({
                "vlm_window_merge_and_memory": True,
                "vlm_window_max_frames": min(5, max(3, int(os.getenv("SENTRIX_VIDEO_MLT_VLM_WINDOW", "5")))),
                "preliminary_scene_count": preliminary_count,
                "merged_scene_count": len(merged),
                "vlm_memory_calls": merge_stats["calls"],
                "vlm_window_seconds": round(window_vlm_seconds, 3),
                "vlm_scenes_merged_away": merge_stats["merged_away"],
                "mlt_strong_boundary_splits": merge_stats.get("strong_boundary_splits", 0),
                "vlm_sliding_windows": merge_stats.get("sliding_windows", False),
                "vlm_window_stride": merge_stats.get("window_stride"),
                "vlm_edge_consensus": merge_stats.get("edge_consensus", []),
                "vlm_windows": merge_stats["windows"],
            })
            (output / "memory_merge.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
            )
        else:
            from .hybrid_keyframe import run as run_hybrid_keyframes
            output = data_root / "derived" / "video" / asset_id / "hybrid-webp"
            frames, merged, manifest = run_hybrid_keyframes(asset["path"], output, asset_id)
        if not merged or not manifest.get("image_integrity_passed"):
            raise RuntimeError(f"{extractor} extractor produced no valid WebP representatives")
        boundary_reviews = []
        if extractor != "mlt_semantic":
            merged, boundary_reviews = _review_scene_boundaries(merged, pipeline.gamma)
            if boundary_reviews:
                manifest["model_scene_boundary_review"] = boundary_reviews
                manifest["model_scene_boundary_review_count"] = len(boundary_reviews)
                manifest["model_scene_boundary_merge_count"] = sum(
                    1 for item in boundary_reviews if item.get("merged")
                )
                manifest["merged_event_count_after_model_review"] = len(merged)
        store.update_asset(asset_id, "video-scene-importing", {
            "video_stage": "video-scene-importing", "keyframe_algorithm": manifest["method"],
            "keyframe_source_count": len(frames), "worldmm_scene_count": len(merged),
            "worldmm_keyframe_count": len(merged), "worldmm_selected_keyframe_count": len(merged),
            "memory_event_merge": True, "memory_duplicate_frame_removal": True,
        })
        scene_ids = []
        keyframe_asset_ids = []
        event_vlm_seconds = window_vlm_seconds
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
                if isinstance(item.get("event_analysis"), dict):
                    event_analysis = dict(item["event_analysis"])
                elif hasattr(pipeline.gamma, "analyze_video_event"):
                    event_analysis = pipeline.gamma.analyze_video_event(
                        evidence_paths,
                        {
                            "source_video": asset.get("file_name"),
                            "start_sec": start_sec, "end_sec": end_sec,
                            "video_duration_sec": float(getattr(metadata, "duration_sec", 0) or 0),
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
                video_duration = float(getattr(metadata, "duration_sec", 0) or 0)
                short_video = 0.0 < video_duration < 30.0
                expanded_retention = short_video or event_duration < 30.0
                # Short videos need more visual evidence because one primary
                # frame can miss an action stage. The final number remains the
                # VLM-selected minimum semantic coverage set; this only raises
                # the candidate/maximum budget and never forces extra frames.
                max_persistent = 3 if expanded_retention else 2 if event_duration < 90.0 else 3
                min_persistent = 1 if event_duration < 45.0 else 2 if event_duration < 150.0 else 3
                selected_indices = (normalized_indices or [fallback_index])[:max_persistent]
                selected_indices, semantic_scene_changed = _semantic_scene_transition(
                    event_analysis, evidence_paths, selected_indices, max_persistent,
                )
                evidence_times = [
                    float(value.get("source_timestamp_sec", start_sec) or start_sec)
                    for value in evidence_records
                ]
                selected_indices, same_view_collapsed = _collapse_same_view_people_change(
                    evidence_paths, evidence_records, selected_indices, item.get("yolo_timeline") or [],
                ) if not semantic_scene_changed else (selected_indices, False)
                if extractor == "mlt_semantic":
                    required_indices = []
                    for value in event_analysis.get("coverage_required_indices") or []:
                        try:
                            index = int(value)
                        except (TypeError, ValueError):
                            continue
                        if 0 <= index < len(evidence_paths) and index not in required_indices:
                            required_indices.append(index)
                    selected_indices = list(dict.fromkeys(selected_indices + required_indices))[:max_persistent]
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
                "summary": f"{start_sec:.1f}s~{end_sec:.1f}s；合并 {item.get('mlt_scene_count') or item['source_frame_count']} 个片段，保留信息帧 {len(representatives)} 张",
                "time_start": _captured_at(captured_at, start_sec), "time_end": _captured_at(captured_at, end_sec),
                "place": location_label, "source_asset_id": asset_id, "source_scene_index": scene_index,
                "source_start_sec": start_sec, "source_end_sec": end_sec,
                "source_metadata": {
                    "keyframe_algorithm": manifest["method"], "memory_event_merge": True,
                    "memory_duplicate_frame_removal": True, "source_event_ids": item["source_event_ids"],
                    "source_frame_count": item["source_frame_count"], "duplicate_frame_count": item["duplicate_frame_count"],
                    "memory_keyframe_count": len(representatives), "semantic_labels": labels[:80],
                    "mlt_scene_count": item.get("mlt_scene_count"),
                    "vlm_merge_reason": item.get("vlm_merge_reason"),
                    "semantic_scene_changed": semantic_scene_changed,
                    "semantic_scene_change_rule": (
                        "visual_semantic_transition" if semantic_scene_changed
                        else "same_view_information_max" if same_view_collapsed
                        else "vlm_minimal_coverage"
                    ),
                    "model_scene_boundary_review": item.get("boundary_model_review") or [],
                    "frame_observations": item.get("frame_observations") or [],
                    "event_detail": {
                        key: event_analysis.get(key)
                        for key in ("caption", "activity", "people", "objects", "clothing",
                                    "spatial_relations", "ocr_text", "facts", "detail")
                        if event_analysis.get(key) not in (None, "", [], {})
                    },
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
                frame_analysis = (
                    _frame_analysis_from_event(
                        event_analysis, representative.get("vlm_selected_evidence_index", evidence_index),
                    ) if extractor == "mlt_semantic" else event_analysis
                )
                # Keep the combined VLM result for the event summary, but do
                # not copy one multi-frame description onto every retained
                # image. When more than one frame survived because the video
                # has a semantic scene transition, each frame gets its own
                # image-grounded observation.
                if extractor != "mlt_semantic" and len(representatives) > 1 and hasattr(pipeline.gamma, "analyze_image"):
                    try:
                        frame_analysis = pipeline.gamma.analyze_image(str(target), {
                            "file_name": target.name,
                            "captured_at": provenance["captured_at"],
                            "captured_location": asset.get("captured_location") or "",
                            "source_video": asset.get("file_name"),
                            "source_timestamp_sec": provenance["source_timestamp_sec"],
                            "source_scene_index": scene_index,
                            "location_context": reverse_geocode or {},
                        })
                    except Exception:
                        frame_analysis = event_analysis
                processed = pipeline.process(
                    keyframe_id, summarize_event=False, forced_event_id=event["id"],
                    image_analysis=frame_analysis,
                )
                if processed.get("status") != "processed":
                    raise RuntimeError(f"WebP keyframe processing failed: {keyframe_id}")
                keyframe_asset_ids.append(keyframe_id)
            if extractor == "mlt_semantic":
                # The 3-5-frame window call already produced the final memory.
                # Persist its event projection directly instead of issuing a
                # second LLM summary request for the same evidence.
                caption = str(event_analysis.get("caption") or "").strip()
                activity = str(event_analysis.get("activity") or "").strip()
                event_type = str(event_analysis.get("event_type") or "视频场景").strip()
                title = (event_type if event_type not in {"视频场景", "家庭记录"} else activity) or caption or "视频场景"
                projection = {
                    "title": title[:20],
                    "event_type": event_type[:20],
                    "activity": (activity or event_type)[:20],
                    "summary": (caption or activity or event_type)[:240],
                    "confidence": event_analysis.get("confidence", 0.65),
                    "model": event_analysis.get("model") or "mlt_window_memory",
                }
                if hasattr(pipeline, "_persist_event_summary"):
                    pipeline._persist_event_summary(event["id"], projection)
                else:
                    store.update_event(event["id"], projection)
            else:
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
            "vlm_window_seconds": round(window_vlm_seconds, 3),
            "preliminary_scene_count": manifest.get("preliminary_scene_count", len(merged)),
            "vlm_memory_calls": manifest.get("vlm_memory_calls", 0),
            "vlm_scenes_merged_away": manifest.get("vlm_scenes_merged_away", 0),
            "mlt_strong_boundary_splits": manifest.get("mlt_strong_boundary_splits", 0),
            "vlm_sliding_windows": manifest.get("vlm_sliding_windows", False),
            "vlm_window_stride": manifest.get("vlm_window_stride"),
            "transient_vlm_frame_count": transient_vlm_frame_count,
            "persistent_keyframe_count": len(keyframe_asset_ids),
            "worldmm_device": os.getenv("SENTRIX_VIDEO_DEVICE", "0"), "vlm_device": "per-keyframe-pipeline",
            "error_stage": None, "error": None, "retryable": True,
        })
