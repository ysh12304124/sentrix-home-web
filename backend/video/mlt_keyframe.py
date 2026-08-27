from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float32)
    return vector / max(float(np.linalg.norm(vector)), 1e-8)


def _robust_unit(values):
    values = np.asarray(values, dtype=np.float32)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(1.4826 * mad, float(np.std(values)) * 0.25, 1e-5)
    return np.clip(0.5 + (values - median) / (6.0 * scale), 0.0, 1.0)


def _sample_video(video_path, sample_fps):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if source_fps <= 0 or frame_count <= 0:
        raise RuntimeError("video has invalid FPS or frame count")
    step = max(1, int(round(source_fps / max(sample_fps, 0.1))))
    sampled = []
    indices = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sampled.append(cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA))
            indices.append(index)
        index += 1
    capture.release()
    return sampled, np.asarray(indices, dtype=np.int64), {
        "source_fps": source_fps, "frame_count": frame_count,
        "width": width, "height": height, "duration_sec": frame_count / source_fps,
        "sample_fps_actual": source_fps / step, "frame_step": step,
    }


@lru_cache(maxsize=2)
def _load_encoder(device):
    import torch
    from torchvision.models import Swin_T_Weights, swin_t

    weights = Swin_T_Weights.DEFAULT
    model = swin_t(weights=weights)
    model.head = torch.nn.Identity()
    model.eval().to(device)
    return model, weights.transforms(), torch


def _encode_frames(frames, device, batch_size):
    from PIL import Image

    model, transform, torch = _load_encoder(device)
    vectors = []
    use_amp = str(device).startswith("cuda")
    with torch.inference_mode():
        for start in range(0, len(frames), batch_size):
            batch = torch.stack([
                transform(Image.fromarray(frame)) for frame in frames[start:start + batch_size]
            ]).to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                output = model(batch)
            output = torch.nn.functional.normalize(output.float(), dim=-1)
            vectors.append(output.cpu().numpy().astype(np.float32))
    return np.concatenate(vectors, axis=0)


def _sparse_clips(fine, times, clip_seconds, stride_seconds):
    centers = []
    vectors = []
    start = float(times[0])
    end = float(times[-1])
    while start <= end:
        stop = start + clip_seconds
        mask = (times >= start) & (times < stop)
        if mask.any():
            vectors.append(_unit(fine[mask].mean(axis=0)))
            centers.append((start + min(stop, end)) / 2.0)
        start += stride_seconds
    return np.asarray(centers, dtype=np.float32), np.asarray(vectors, dtype=np.float32)


def _cosine_distance(left, right):
    return 1.0 - np.sum(left * right, axis=-1)


def _boundary_scores(fine, times, clip_times, clip_vectors):
    count = len(fine)
    appearance = np.zeros(count, dtype=np.float32)
    appearance[1:] = _cosine_distance(fine[1:], fine[:-1])
    delta = fine[1:] - fine[:-1]
    delta /= np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-8)
    differential = np.zeros(count, dtype=np.float32)
    if len(delta) > 1:
        differential[2:] = _cosine_distance(delta[1:], delta[:-1])
    sparse_change = np.zeros(len(clip_vectors), dtype=np.float32)
    if len(clip_vectors) > 1:
        sparse_change[1:] = _cosine_distance(clip_vectors[1:], clip_vectors[:-1])
    if len(clip_times) > 1:
        sparse_frame = np.interp(times, clip_times, sparse_change).astype(np.float32)
    elif len(sparse_change):
        sparse_frame = np.full(count, float(sparse_change[0]), dtype=np.float32)
    else:
        sparse_frame = np.zeros(count, dtype=np.float32)
    score = np.clip(
        0.50 * _robust_unit(appearance)
        + 0.35 * _robust_unit(differential)
        + 0.15 * _robust_unit(sparse_frame),
        0.0, 1.0,
    )
    score[0] = 0.0
    return score, appearance, differential, sparse_frame


def _choose_boundaries(scores, sample_fps, min_scene_seconds, max_scenes):
    core = np.asarray(scores[1:], dtype=np.float32)
    median = float(np.median(core))
    threshold = max(
        float(np.percentile(core, 82)),
        median + 1.8 * float(np.median(np.abs(core - median))),
    )
    candidates = [
        index for index in range(1, len(scores) - 1)
        if scores[index] >= threshold
        and scores[index] >= scores[index - 1]
        and scores[index] >= scores[index + 1]
    ]
    min_gap = max(1, int(round(sample_fps * min_scene_seconds)))
    selected = []
    for index in sorted(candidates, key=lambda value: float(scores[value]), reverse=True):
        if all(abs(index - kept) >= min_gap for kept in selected):
            selected.append(index)
        if len(selected) >= max(1, max_scenes - 1):
            break
    return sorted(selected)


def _read_original_frame(video_path, frame_index):
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"cannot read representative frame {frame_index}")
    return frame


def _write_webp(frame, target, quality):
    max_width = int(os.getenv("SENTRIX_VIDEO_MLT_MAX_IMAGE_WIDTH", "1280"))
    if max_width > 0 and frame.shape[1] > max_width:
        height = max(1, int(round(frame.shape[0] * max_width / frame.shape[1])))
        frame = cv2.resize(frame, (max_width, height), interpolation=cv2.INTER_AREA)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), frame, [cv2.IMWRITE_WEBP_QUALITY, int(quality)]):
        raise RuntimeError(f"cannot write representative image: {target}")


def _merge_items(items, merge_reason):
    primary = dict(items[len(items) // 2]["representative"])
    evidence = []
    for item in items:
        for value in item.get("representatives") or [item["representative"]]:
            if value.get("webp_path") and value.get("webp_path") not in {
                row.get("webp_path") for row in evidence
            }:
                evidence.append(dict(value))
    primary["vlm_evidence"] = evidence[:5]
    return {
        "event_id": "+".join(item["event_id"] for item in items),
        "source_event_ids": [value for item in items for value in item.get("source_event_ids", [])],
        "start_sec": min(float(item["start_sec"]) for item in items),
        "end_sec": max(float(item["end_sec"]) for item in items),
        "representative": primary, "representatives": [primary],
        "objects": list(dict.fromkeys(value for item in items for value in item.get("objects", []))),
        "actions": list(dict.fromkeys(value for item in items for value in item.get("actions", []))),
        "expressions": list(dict.fromkeys(value for item in items for value in item.get("expressions", []))),
        "yolo_timeline": [value for item in items for value in item.get("yolo_timeline", [])],
        "source_frame_count": sum(int(item.get("source_frame_count") or 1) for item in items),
        "duplicate_frame_count": sum(int(item.get("duplicate_frame_count") or 0) for item in items),
        "visual_duplicate_count": sum(int(item.get("visual_duplicate_count") or 0) for item in items),
        "memory_keyframe_count": 1,
        "mlt_scene_count": sum(int(item.get("mlt_scene_count") or 1) for item in items),
        "vlm_merge_reason": merge_reason,
    }


def _sliding_window_spans(count, maximum=5, stride=None):
    """Return 3-5 item windows with one or two overlapping anchor frames."""
    maximum = max(3, min(5, int(maximum)))
    stride = max(1, min(maximum - 1, int(stride or maximum - 1)))
    if count <= maximum:
        return [(0, count)]
    spans = []
    start = 0
    while start < count:
        end = min(count, start + maximum)
        spans.append((start, end))
        if end >= count:
            break
        next_start = start + stride
        if count - next_start < 3:
            next_start = max(start + 1, count - 3)
        start = next_start
    return spans


def _mlt_distance(left, right):
    left_embedding = left.get("_mlt_embedding")
    right_embedding = right.get("_mlt_embedding")
    if left_embedding is None or right_embedding is None:
        return None
    return float(1.0 - np.dot(_unit(left_embedding), _unit(right_embedding)))


def _component_memory(records, component, strong_guarded_edges):
    """Reconcile memories from overlapping calls without another model request."""
    component_set = set(component)
    relevant = [record for record in records if component_set.intersection(record["global_indices"])]
    relevant.sort(key=lambda record: (
        len(component_set.intersection(record["global_indices"])),
        -len(set(record["global_indices"]) - component_set),
    ), reverse=True)
    base = dict((relevant[0] if relevant else {}).get("memory") or {})
    exact = next((record for record in relevant if set(record["global_indices"]) == component_set), None)
    if exact is not None:
        base = dict(exact["memory"])

    observations_by_index = {}
    selected_globals = []
    for record in relevant:
        for observation in record.get("frame_observations") or []:
            index = observation.get("index")
            if index in component_set and index not in observations_by_index:
                observations_by_index[index] = dict(observation)
        for index in record.get("representative_indices") or []:
            if index in component_set and index not in selected_globals:
                selected_globals.append(index)
    observations = [observations_by_index[index] for index in component if index in observations_by_index]
    complete_observation_coverage = len(observations_by_index) == len(component)

    def joined_observations(key, limit):
        values = list(dict.fromkeys(
            str(value.get(key) or "").strip() for value in observations
            if str(value.get(key) or "").strip()
        ))
        if not values:
            values = list(dict.fromkeys(
                str(record["memory"].get(key) or "").strip() for record in relevant
                if str(record["memory"].get(key) or "").strip()
            ))
        return "；".join(values)[:limit]

    reconciled = exact is None or any(edge in strong_guarded_edges for edge in component[:-1])
    if reconciled:
        base["caption"] = joined_observations("caption", 160)
        base["activity"] = joined_observations("activity", 60)
        base["place"] = joined_observations("place", 40)
        base["people"] = list(dict.fromkeys(
            str(value) for observation in observations for value in observation.get("people") or []
        ))[:12]
        base["objects"] = list(dict.fromkeys(
            str(value) for observation in observations for value in observation.get("objects") or []
        ))[:20]
        base["semantic"] = {}
        base["facts"] = []
        base["spatial_relations"] = []
        base["ocr_text"] = ""
        uncertainties = ["重叠滑动窗口结果已进行全局连续性汇总"]
        if any(edge in strong_guarded_edges for edge in component[:-1]):
            uncertainties.append("大模型合并跨越MLT强视觉边界，已保守拆分")
        base["detail"] = {
            "schema_version": 1, "visible_details": [], "regions": [], "text_blocks": [],
            "uncertainties": uncertainties,
        }
        base["confidence"] = min(float(base.get("confidence", 0.65) or 0.65), 0.7)
    base["frame_observations"] = [
        {**observation, "index": component.index(observation["index"])}
        for observation in observations
    ]
    if 1 < len(component) <= 3 and complete_observation_coverage:
        selected_globals = list(component)
        base["coverage_required_indices"] = list(range(len(component)))
    else:
        selected_globals = sorted(dict.fromkeys(selected_globals))[:3]
    base["representative_indices"] = [
        component.index(index) for index in selected_globals[:3]
    ] or [len(component) // 2]
    base.pop("indices", None)
    base.pop("merge_reason", None)
    return base


def merge_and_analyze_windows(items, gamma, context=None, max_window=5, stride=None):
    """Analyze overlapping 3-5 frame windows and reconcile one global timeline."""
    ordered = list(items or [])
    if not ordered or not hasattr(gamma, "analyze_video_scene_window"):
        return ordered, {"calls": 0, "merged_away": 0, "windows": []}
    windows = []
    calls = 0
    records = []
    edge_votes = [{"same": 0, "different": 0} for _ in range(max(0, len(ordered) - 1))]
    maximum = max(3, min(5, int(max_window)))
    stride = max(1, min(maximum - 1, int(stride or maximum - 1)))
    spans = _sliding_window_spans(len(ordered), maximum, stride)
    previous_end = 0
    for window_index, (start, end) in enumerate(spans):
        window = ordered[start:end]
        paths = [Path(str(item["representative"]["webp_path"])) for item in window]
        metadata = {
            **(context or {}),
            "sliding_window": {
                "window_index": window_index, "global_start_index": start,
                "global_end_index": end - 1, "overlap_with_previous": max(0, previous_end - start),
            },
            "scenes": [{
                "index": offset, "global_index": start + offset,
                "start_sec": item["start_sec"], "end_sec": item["end_sec"],
                "boundary_confidence": item.get("boundary_confidence"),
            } for offset, item in enumerate(window)],
        }
        analysis = gamma.analyze_video_scene_window(paths, metadata)
        calls += 1
        groups = list(analysis.get("groups") or [])
        group_map = [-1] * len(window)
        window_groups = []
        for group_index, group in enumerate(groups):
            indices = [int(value) for value in group.get("indices") or []
                       if 0 <= int(value) < len(window)]
            indices = sorted(dict.fromkeys(indices))
            if not indices:
                continue
            for index in indices:
                group_map[index] = group_index
            global_indices = [start + index for index in indices]
            memory = dict(group)
            frame_observations = []
            for observation in group.get("frame_observations") or []:
                if not isinstance(observation, dict):
                    continue
                try:
                    local_index = int(observation.get("index"))
                except (TypeError, ValueError):
                    continue
                if local_index in indices:
                    frame_observations.append({**observation, "index": start + local_index})
            representative_indices = []
            for value in group.get("representative_indices") or []:
                try:
                    local_index = int(value)
                except (TypeError, ValueError):
                    continue
                if local_index in indices:
                    representative_indices.append(start + local_index)
            records.append({
                "global_indices": global_indices, "memory": memory,
                "frame_observations": frame_observations,
                "representative_indices": representative_indices,
                "merge_reason": str(group.get("merge_reason") or "")[:500],
            })
            window_groups.append({
                "indices": indices, "global_indices": global_indices,
                "source_event_ids": [
                    value for index in global_indices for value in ordered[index].get("source_event_ids", [])
                ],
                "merge_reason": str(group.get("merge_reason") or "")[:500],
            })
        for local_index in range(len(window) - 1):
            edge = start + local_index
            same = group_map[local_index] >= 0 and group_map[local_index] == group_map[local_index + 1]
            edge_votes[edge]["same" if same else "different"] += 1
        windows.append({
            "window_index": window_index, "start_index": start, "end_index": end - 1,
            "window_size": len(window), "overlap_with_previous": max(0, previous_end - start),
            "groups": window_groups,
        })
        previous_end = end

    threshold = float(os.getenv("SENTRIX_VIDEO_MLT_MERGE_MAX_DISTANCE", "0.75"))
    merge_edges = []
    strong_guarded_edges = set()
    edge_consensus = []
    for edge, votes in enumerate(edge_votes):
        merge = votes["same"] > 0 and votes["different"] == 0
        distance = _mlt_distance(ordered[edge], ordered[edge + 1])
        guarded = bool(merge and distance is not None and distance > threshold)
        if guarded:
            merge = False
            strong_guarded_edges.add(edge)
        merge_edges.append(merge)
        edge_consensus.append({
            "left_index": edge, "right_index": edge + 1,
            "same_votes": votes["same"], "different_votes": votes["different"],
            "mlt_distance": round(distance, 6) if distance is not None else None,
            "mlt_strong_boundary_guard": guarded, "merge": merge,
        })

    components = []
    current = [0]
    for edge, merge in enumerate(merge_edges):
        if merge:
            current.append(edge + 1)
        else:
            components.append(current)
            current = [edge + 1]
    components.append(current)

    result = []
    for component in components:
        reasons = list(dict.fromkeys(
            record["merge_reason"] for record in records
            if set(component).intersection(record["global_indices"]) and record["merge_reason"]
        ))
        merged = _merge_items(
            [ordered[index] for index in component], "；".join(reasons)[:500],
        )
        merged["event_analysis"] = _component_memory(records, component, strong_guarded_edges)
        merged["frame_observations"] = list(merged["event_analysis"].get("frame_observations") or [])
        result.append(merged)
    return result, {
        "calls": calls, "merged_away": max(0, len(ordered) - len(result)),
        "windows": windows, "sliding_windows": True,
        "window_stride": stride,
        "strong_boundary_splits": len(strong_guarded_edges),
        "edge_consensus": edge_consensus,
    }


def run(video_path, output_dir, video_id):
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sample_fps = float(os.getenv("SENTRIX_VIDEO_MLT_SAMPLE_FPS", "2"))
    clip_seconds = float(os.getenv("SENTRIX_VIDEO_MLT_CLIP_SECONDS", "8"))
    clip_stride = float(os.getenv("SENTRIX_VIDEO_MLT_CLIP_STRIDE", "4"))
    min_scene_seconds = float(os.getenv("SENTRIX_VIDEO_MLT_MIN_SCENE_SECONDS", "2"))
    max_scenes = int(os.getenv("SENTRIX_VIDEO_MLT_MAX_SCENES", "160"))
    batch_size = int(os.getenv("SENTRIX_VIDEO_MLT_BATCH_SIZE", "8"))
    device = os.getenv("SENTRIX_VIDEO_MLT_DEVICE", os.getenv("SENTRIX_VIDEO_DEVICE", "cpu"))
    if str(device).isdigit():
        device = f"cuda:{device}"
    frames, frame_indices, meta = _sample_video(video_path, sample_fps)
    if len(frames) < 2:
        raise RuntimeError("MLT extractor sampled fewer than two frames")
    times = frame_indices / meta["source_fps"]
    fine = _encode_frames(frames, device, batch_size)
    clip_times, clip_vectors = _sparse_clips(fine, times, clip_seconds, clip_stride)
    scores, appearance, differential, sparse_change = _boundary_scores(
        fine, times, clip_times, clip_vectors,
    )
    boundaries = _choose_boundaries(
        scores, meta["sample_fps_actual"], min_scene_seconds, max_scenes,
    )
    cuts = [0] + boundaries + [len(frames)]
    webp_dir = output / "webp"
    rows = []
    quality = int(os.getenv("SENTRIX_VIDEO_WEBP_QUALITY", "80"))
    for scene_index, (start, end) in enumerate(zip(cuts[:-1], cuts[1:]), 1):
        local = np.arange(start, end)
        center = (start + end - 1) / 2.0
        center_distance = np.abs(local - center) / max(1.0, (end - start) / 2.0)
        representative_index = int(local[np.argmax((1.0 - scores[local]) - 0.15 * center_distance)])
        source_frame = int(frame_indices[representative_index])
        source_time = float(times[representative_index])
        target = webp_dir / f"scene_{scene_index:04d}_frame_{source_frame:08d}_t{source_time:010.3f}.webp"
        _write_webp(_read_original_frame(video_path, source_frame), target, quality)
        start_score = float(scores[start]) if start else 1.0
        row = {
            "event_id": f"mlt_scene_{scene_index:04d}",
            "source_event_ids": [f"mlt_scene_{scene_index:04d}"],
            "event_start_sec": float(times[start]),
            "event_end_sec": float(times[end - 1] + 1.0 / meta["sample_fps_actual"]),
            "start_sec": float(times[start]),
            "end_sec": float(times[end - 1] + 1.0 / meta["sample_fps_actual"]),
            "source_timestamp_sec": source_time, "source_frame_index": source_frame,
            "webp_path": str(target), "webp_bytes": target.stat().st_size,
            "objects": [], "actions": [], "expressions": [], "yolo_timeline": [],
            "boundary_confidence": start_score,
            "source_frame_count": int(end - start),
        }
        item = {
            "event_id": row["event_id"], "source_event_ids": row["source_event_ids"],
            "start_sec": row["start_sec"], "end_sec": row["end_sec"],
            "representative": row, "representatives": [row],
            "objects": [], "actions": [], "expressions": [], "yolo_timeline": [],
            "source_frame_count": int(end - start), "duplicate_frame_count": 0,
            "visual_duplicate_count": 0, "memory_keyframe_count": 1,
            "mlt_scene_count": 1, "boundary_confidence": start_score,
            "_mlt_embedding": fine[representative_index],
        }
        rows.append((row, item))
    np.savez_compressed(
        output / "mlt_features.npz", times=times, frame_indices=frame_indices,
        fine_frame_embeddings=fine, sparse_clip_times=clip_times,
        sparse_clip_embeddings=clip_vectors, boundary_score=scores,
        appearance=appearance, differential=differential, sparse_change=sparse_change,
    )
    manifest = {
        "method": "mlt_dedup_semantic_swin_t_differential_v1",
        "paper": "https://arxiv.org/abs/2606.12215v1",
        "adaptation": "self-video semantic boundary localization",
        "encoder": "torchvision ImageNet Swin-T",
        "video_id": str(video_id), "video": meta,
        "sampled_frame_count": len(frames), "sparse_clip_count": len(clip_vectors),
        "preliminary_scene_count": len(rows), "boundary_count": len(boundaries),
        "image_integrity_passed": all(Path(row[0]["webp_path"]).is_file() for row in rows),
        "config": {
            "sample_fps": sample_fps, "clip_seconds": clip_seconds,
            "clip_stride": clip_stride, "min_scene_seconds": min_scene_seconds,
            "max_scenes": max_scenes, "device": device,
        },
    }
    (output / "mlt_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return [row for row, _item in rows], [item for _row, item in rows], manifest
