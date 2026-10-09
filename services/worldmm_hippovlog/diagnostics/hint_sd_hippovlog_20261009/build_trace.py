#!/usr/bin/env python3
"""Materialize provenance from saved VSD/YOLO/eSVD/Event/WorldMM outputs.

This is an audit of the exact historical source run used by qa_trace.py; it
does not alter or rerun any keyframe-selection or memory algorithm.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

SOURCE = Path("/home/sscy/evaluations/sentrix-worldmm-hippovlog-20260928/services/worldmm_hippovlog/results/20260928-163006/svd_esvd")
VIDEOS = ("6Z_qEtbmK34", "Ei7hTKr8Ins")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sec(clock: str) -> float:
    value = str(clock).zfill(8)
    return int(value[:2]) * 3600 + int(value[2:4]) * 60 + int(value[4:6]) + int(value[6:]) / 100


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(parents=True, exist_ok=True)
    traces = out / "stage_trace.jsonl"
    conflicts = []
    summary = {}
    with traces.open("w", encoding="utf-8") as handle:
        def emit(row):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        for video_id in VIDEOS:
            root = SOURCE / "keyframes" / video_id
            analysis = load(root / "analysis.json")
            manifest = load(root / "manifest.json")
            yolo = load(root / "yolo_scan.json")
            candidates = load(root / "yolo_scan_candidates.json")
            captions = load(SOURCE / "captions" / video_id / "10sec.json")
            caption_metrics = load(SOURCE / "captions" / video_id / "metrics.json")
            semantic = load(SOURCE / "metadata/semantic_memory" / video_id / "semantic_consolidation_results_qwen3vl-4b.json")
            metrics = analysis["metrics"]
            video = analysis["video"]
            first = analysis["fold_assignments"]
            second = analysis["second_pass_fold_assignments"]
            timestamps = metrics["timestamps"]
            sampled_raw = {item["frame_index"]: (index, timestamps[index]) for index, item in enumerate(first)}
            with (out / f"raw_sampling_{video_id}.jsonl").open("w", encoding="utf-8") as raw_handle:
                for raw_index in range(video["frame_count"]):
                    sample = sampled_raw.get(raw_index)
                    raw_handle.write(json.dumps({
                        "video_id": video_id, "stage": "RAW_TO_10FPS", "frame_id": f"{video_id}:{raw_index}",
                        "frame_index": raw_index,
                        "timestamp_sec": sample[1] if sample else raw_index / video["fps"],
                        "timestamp_estimated": sample is None,
                        "input_count": 1, "output_count": int(sample is not None),
                        "decision": "KEEP" if sample else "DROP",
                        "reason": "10fps_source_sampling" if sample else "not_in_10fps_sample",
                        "provenance": {"sample_index": sample[0] if sample else None},
                    }, ensure_ascii=False) + "\n")
            scores = metrics["score"]
            observations = {row["sample_index"]: row for row in yolo["samples"]}
            candidate_set = {row["sample_index"] for row in candidates}
            selected_set = set(analysis["selected_indices"])
            events = metrics["event_generation"]["events"]
            event_by_sample = [None] * len(timestamps)
            for event in events:
                for index in range(event["start_index"], event["end_index"]):
                    event_by_sample[index] = event["event_id"]
            keyframes = {round(row["timestamp_sec"], 5): row for row in manifest["keyframes"]}
            caption_ranges = [(sec(row["start_time"]), sec(row["end_time"]), i, row)
                              for i, row in enumerate(captions)]
            group_observations = defaultdict(list)
            for index, observation in observations.items():
                group_observations[second[index]["group_id"]].append(observation)
            conflict_groups = set()
            for group_id, items in group_observations.items():
                signatures = {tuple(sorted(item.get("labels") or [])) for item in items}
                if len(items) < 2 or len(signatures) < 2:
                    continue
                if not any(second[item["sample_index"]]["status"] == "folded" for item in items):
                    continue
                conflict_groups.add(group_id)
                conflicts.append({
                    "video_id": video_id, "code": "EVSD_SEMANTIC_CONFLICT", "fold_group_id": group_id,
                    "observations": [{"frame_index": item["frame_index"], "timestamp_sec": item["timestamp_sec"],
                                      "labels": item.get("labels", []), "detections": item.get("detections", []),
                                      "fold_status": second[item["sample_index"]]["status"]} for item in items],
                    "review_status": "candidate_not_ground_truth",
                })

            first_counts = Counter(item["status"] for item in first)
            second_counts = Counter(item["status"] for item in second)
            for index, (timestamp, fold1, fold2) in enumerate(zip(timestamps, first, second)):
                frame = fold1["frame_index"]
                event_id = event_by_sample[index]
                base = {"video_id": video_id, "sample_index": index,
                        "frame_id": f"{video_id}:{frame}", "frame_index": frame,
                        "timestamp_sec": timestamp}
                emit({**base, "stage": "RAW_SAMPLE", "decision": "KEEP", "input_count": 1,
                      "output_count": 1, "reason": "10fps_source_sample", "provenance": {"raw_frame_index": frame}})
                candidate = index in candidate_set
                if candidate:
                    vsd_decision, reason = "KEEP", "highest_scoring_first_pass_representative_in_1fps_bucket"
                elif fold1["status"] == "folded":
                    vsd_decision, reason = "FOLD", "first_pass_low_rank_group"
                else:
                    vsd_decision, reason = "DROP", "not_highest_score_in_1fps_scan_bucket"
                emit({**base, "stage": "VSD", "decision": vsd_decision, "input_count": 1,
                      "output_count": int(candidate), "reason": reason,
                      "score": scores[index], "reconstruction_error": metrics["reconstruction_error"][index],
                      "subspace_change": metrics["subspace_change"][index],
                      "spectrum_change": metrics["spectrum_change"][index],
                      "first_fold_group_id": fold1["group_id"],
                      "fold_residual": fold1.get("reconstruction_residual"),
                      "provenance": {"raw_frame_index": frame, "yolo_candidate_sample_index": index if candidate else None},
                      "note": "not YOLO-scanned does not necessarily mean absent from final event selection"})
                if candidate:
                    observation = observations.get(index)
                    if observation is None:
                        raise RuntimeError(f"missing YOLO observation for {video_id}:{index}")
                    emit({**base, "stage": "YOLO", "decision": "KEEP", "input_count": 1,
                          "output_count": len(observation["detections"]),
                          "reason": "model_inference_complete", "detections": observation["detections"],
                          "labels": observation["labels"], "model": yolo["model"], "device": yolo["device"],
                          "provenance": {"vsd_sample_index": index, "event_id": event_id}})
                status = fold2["status"]
                emit({**base, "stage": "EVSD", "decision": "FOLD" if status == "folded" else "KEEP",
                      "input_count": 1, "output_count": int(status != "folded"),
                      "reason": "event_local_low_rank_fold", "event_id": event_id,
                      "fold_group_id": fold2["group_id"], "fold_residual": fold2.get("reconstruction_residual"),
                      "similarity_to_group": fold2.get("similarity_to_group"),
                      "first_fold_group_id": fold1["group_id"], "selected_as_final_keyframe": index in selected_set,
                      "diagnostic_flag": "EVSD_SEMANTIC_CONFLICT" if fold2["group_id"] in conflict_groups else None,
                      "provenance": {"vsd_sample_index": index, "yolo_event_id": event_id,
                                     "event_fold_group_id": fold2["group_id"]}})

            for event in events:
                event_id = event["event_id"]
                chosen = [row for row in manifest["keyframes"] if row["raw"].get("yolo_event_id") == event_id]
                if not chosen:
                    raise RuntimeError(f"event {event_id} has no keyframe")
                emit({"video_id": video_id, "stage": "Event", "event_id": event_id,
                      "frame_id": f"{video_id}:{chosen[0]['frame_index']}",
                      "timestamp_sec": chosen[0]["timestamp_sec"],
                      "start_sec": event["start_sec"], "end_sec": event["end_sec"],
                      "input_count": event["end_index"] - event["start_index"],
                      "output_count": len(chosen),
                      "decision": "MERGE" if event.get("camera_motion_merged") else "KEEP",
                      "reason": event.get("boundary_reason"), "camera_motion_merged": bool(event.get("camera_motion_merged")),
                      "semantic_labels": event.get("semantic_labels"),
                      "mean_label_confidence": event.get("mean_label_confidence"),
                      "boundary_score": scores[event["start_index"]],
                      "provenance": {"sample_indices": [event["start_index"], event["end_index"]],
                                     "selected_keyframe_indices": [row["frame_index"] for row in chosen]}})

            for start, end, caption_index, row in caption_ranges:
                source_frames = [item for item in manifest["keyframes"] if start <= item["timestamp_sec"] < end]
                fallback = not source_frames
                emit({"video_id": video_id, "stage": "Memory", "memory_type": "episodic_10sec",
                      "memory_id": f"10sec_{caption_index}", "frame_id": None,
                      "timestamp_sec": start, "start_sec": start, "end_sec": end,
                      "input_count": len(source_frames), "output_count": int(bool(row.get("text"))),
                      "decision": "KEEP", "reason": "WorldMM_10sec_caption" + ("_midpoint_fallback" if fallback else ""),
                      "text": row.get("text"), "provenance": {
                          "event_ids": sorted({item["raw"].get("yolo_event_id") for item in source_frames}),
                          "keyframe_ids": [item["code"] for item in source_frames],
                          "fallback_midpoint_raw_frame": fallback,
                      }})
            for memory_time, item in semantic.items():
                triples = item.get("consolidated_semantic_triples") or []
                emit({"video_id": video_id, "stage": "Memory", "memory_type": "semantic_triples",
                      "memory_id": f"semantic_{memory_time}", "frame_id": None, "timestamp_sec": None,
                      "input_count": None, "output_count": len(triples), "decision": "KEEP",
                      "reason": "WorldMM_semantic_consolidation", "triples": triples,
                      "provenance": {"multiscale_timestamp": memory_time}})

            summary[video_id] = {
                "video": video, "raw_frame_count": video["frame_count"],
                "sampled_frame_count": len(timestamps),
                "unsampled_raw_frame_count": video["frame_count"] - len(timestamps),
                "vsd_candidate_count": len(candidate_set), "vsd_first_fold_status": dict(first_counts),
                "yolo_observation_count": len(observations),
                "yolo_detection_count": sum(len(row["detections"]) for row in observations.values()),
                "yolo_zero_detection_observations": sum(not row["detections"] for row in observations.values()),
                "evsd_fold_status": dict(second_counts), "evsd_conflict_candidate_groups": len(conflict_groups),
                "event_count": len(events), "event_merged_count": sum(bool(row.get("camera_motion_merged")) for row in events),
                "final_keyframe_count": len(manifest["keyframes"]),
                "episodic_10sec_count": len(captions),
                "episodic_midpoint_fallback_count": sum(not any(
                    start <= item["timestamp_sec"] < end for item in manifest["keyframes"])
                    for start, end, _, _ in caption_ranges),
                "semantic_triple_count": sum(len(item.get("consolidated_semantic_triples") or []) for item in semantic.values()),
                "visual_memory_file_exists": (SOURCE / "metadata/visual_memory" / video_id / "visual_embeddings.pkl").is_file(),
                "timing_seconds": {
                    "vsd_yolo_evsd_event_total": metrics.get("runtime_seconds"),
                    "first_pass_fold": metrics["folding"].get("runtime_seconds"),
                    "second_pass_evsd_fold": metrics["folding_passes"]["second_pass_per_event"].get("runtime_seconds"),
                    "caption_and_multiscale": caption_metrics.get("caption_and_multiscale_seconds"),
                    "memory_build": None,
                },
                "historical_gpu_peak_mib": None,
                "notes": [
                    "VSD output count denotes sparse YOLO scan candidates; non-candidates may still become final event keyframes.",
                    "RAW sampling trace records exact timestamps for sampled frames and frame_index/fps estimates for unsampled frames.",
                    "Historical VSD/YOLO/Event timing breakdown and GPU/RAM peaks were not persisted.",
                    "Semantic evidence retention cannot be inferred solely from structural provenance.",
                ],
            }
    with (out / "evsd_semantic_conflicts.jsonl").open("w", encoding="utf-8") as handle:
        for row in conflicts:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(out / "stage_summary.json", summary)
    print(json.dumps({video: {key: value for key, value in data.items() if key in {
        "raw_frame_count", "sampled_frame_count", "vsd_candidate_count", "event_count", "final_keyframe_count", "evsd_conflict_candidate_groups"
    }} for video, data in summary.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
