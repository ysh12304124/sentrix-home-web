#!/usr/bin/env python3
"""Run parameter and component ablations from a cached MTSW state scan.

This deliberately reuses the measured YOLO/Pose state cache.  It does not call
any VLM/LLM and does not decode the source video again, so every row isolates
the state-selection policy rather than changing detector or decode cost.
"""

from __future__ import annotations

import argparse
import copy
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.video.mtsw import MTSW_CONFIG, MTSW_METHOD_VERSION, FrameState, MTSWEngine


def load_states(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("cache_version") != "mtsw-state-v1":
        raise ValueError(f"unsupported cache version: {payload.get('cache_version')}")
    states = []
    for item in payload["states"]:
        states.append(FrameState(
            frame_index=int(item["frame_index"]),
            timestamp=float(item["timestamp"]),
            image_path=str(item.get("image_path") or ""),
            objects=list(item.get("objects") or []),
            people=list(item.get("people") or []),
            pose_state=str(item.get("pose_state") or "unknown"),
            scene_embedding=list(item.get("scene_embedding") or []),
            appearance_embedding=list(item.get("appearance_embedding") or []),
            sharpness=float(item.get("sharpness") or 0),
            phash=str(item.get("phash") or ""),
            black_frame=bool(item.get("black_frame", False)),
        ))
    return states


def measure(states, changes):
    config = {**MTSW_CONFIG, **changes}
    result = MTSWEngine(config).analyze(copy.deepcopy(states))
    metrics = dict(result["metrics"])
    metrics["z_threshold"] = config["z_threshold"]
    metrics["persistence_frames"] = config["persistence_frames"]
    metrics["phash_max_distance"] = config["phash_max_distance"]
    metrics["interaction_weight"] = config["interaction_weight"]
    metrics["pose_weight"] = config["pose_weight"]
    metrics["micro_window_sec"] = config["micro_window_sec"]
    metrics["event_short_window_sec"] = config["event_short_window_sec"]
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cache = Path(args.cache)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    states = load_states(cache)

    parameter_rows = []
    for threshold in [1.3, 1.5, 1.7, 2.0, 2.3]:
        parameter_rows.append({"family": "z_threshold", "value": threshold, "metrics": measure(states, {"z_threshold": threshold})})
    for persistence in [1, 2, 3]:
        parameter_rows.append({"family": "persistence_frames", "value": persistence, "metrics": measure(states, {"persistence_frames": persistence})})
    for phash_distance in [2, 4, 8]:
        parameter_rows.append({"family": "phash_max_distance", "value": phash_distance, "metrics": measure(states, {"phash_max_distance": phash_distance})})
    for micro_window in [1.0, 2.0, 3.0]:
        parameter_rows.append({"family": "micro_window_sec", "value": micro_window, "metrics": measure(states, {"micro_window_sec": micro_window})})
    for event_window in [3.0, 10.0, 15.0]:
        parameter_rows.append({"family": "event_short_window_sec", "value": event_window, "metrics": measure(states, {"event_short_window_sec": event_window})})

    component_rows = [
        {"component": "baseline_v2.0", "status": "frozen_reference", "metrics": {"keyframes": 180, "events": 180, "source": "measured_baseline_run"}},
        {"component": "eventagg_v2.1", "status": "frozen_reference", "metrics": {"keyframes": 180, "events": 131, "timeline_compression": 0.272222, "source": "measured_eventagg_run"}},
        {"component": "mtsw_default", "status": "measured", "metrics": measure(states, {})},
        {"component": "mtsw_no_dedup", "status": "measured", "metrics": measure(states, {"phash_max_distance": -1, "visual_similarity": 2.0})},
        {"component": "mtsw_interaction_disabled", "status": "measured", "metrics": measure(states, {"interaction_weight": 0.0})},
        {"component": "mtsw_pose_disabled", "status": "measured", "metrics": measure(states, {"pose_weight": 0.0})},
        {"component": "tracker", "status": "not_enabled_in_v2.3", "metrics": {}},
        {"component": "osnet_identity", "status": "not_enabled_in_v2.3", "metrics": {}},
        {"component": "places365_scene", "status": "not_enabled_in_v2.3", "metrics": {}},
        {"component": "clothing_color", "status": "available_heuristic_not_used_as_detector", "metrics": {}},
    ]
    payload = {
        "method_version": MTSW_METHOD_VERSION,
        "cache": str(cache),
        "scan_frames": len(states),
        "vlm_calls": 0,
        "llm_calls": 0,
        "parameter_rows": parameter_rows,
        "component_rows": component_rows,
        "interpretation": {
            "fairness": "all selection rows reuse the same measured low-rate YOLO/Pose state cache",
            "not_enabled": ["tracker", "osnet_identity", "places365_scene"],
            "note": "micro/event window fields are recorded configuration knobs; the current event grouping is continuity/state based, so rows may be identical until window gating is enabled in a future version.",
        },
    }
    (output / "v23_parameter_sweep.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for row in parameter_rows:
        metrics = row["metrics"]
        rows.append("<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            html.escape(str(row["family"])), html.escape(str(row["value"])), metrics.get("state_transitions", ""), metrics.get("final_keyframes", ""), metrics.get("events", ""), metrics.get("duplicate_removed", ""), metrics.get("singleton_rate", "")))
    component_html = []
    for row in component_rows:
        metrics = row["metrics"]
        component_html.append("<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(html.escape(row["component"]), html.escape(row["status"]), metrics.get("keyframes", metrics.get("final_keyframes", "")), metrics.get("events", ""), metrics.get("duplicate_removed", "")))
    style = "body{font:14px system-ui;margin:24px;color:#182016;background:#f7faf4}table{border-collapse:collapse;background:#fff;width:100%;margin:14px 0}th,td{border:1px solid #dfe8d7;padding:7px;text-align:left}th{background:#eaf4df}"
    document = "<!doctype html><meta charset='utf-8'><style>{}</style><h1>v2.3 MTSW parameter sweep</h1><p>Cache: {} · scan frames: {} · VLM/LLM: 0/0</p><h2>Parameter sweep</h2><table><tr><th>family</th><th>value</th><th>transitions</th><th>keyframes</th><th>events</th><th>dedup removed</th><th>singleton rate</th></tr>{}</table><h2>Component ablation</h2><table><tr><th>component</th><th>status</th><th>keyframes</th><th>events</th><th>dedup removed</th></tr>{}</table><p>{}</p>".format(style, html.escape(str(cache)), len(states), "".join(rows), "".join(component_html), html.escape(payload["interpretation"]["note"]))
    (output / "v23_parameter_sweep.html").write_text(document, encoding="utf-8")
    print(json.dumps({"output": str(output), "scan_frames": len(states), "parameter_rows": len(parameter_rows), "component_rows": len(component_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
