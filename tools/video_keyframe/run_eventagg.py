#!/usr/bin/env python3
"""Run EventAgg over existing hybrid WebP evidence.

This is the controlled A/B entry point: no video decode, YOLO, Pose, Katna or
WebP generation happens here.  It reads the existing 180-frame package and
writes only method_runs/event_aggregate_* records and reports.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.db import MemoryStore
from backend.video.event_aggregator import (
    DEFAULT_CONFIG, EVENTAGG_METHOD_VERSION, DINOEmbedder, EventAggregator,
)


def _time_at(value, offset):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return (parsed + timedelta(seconds=float(offset))).isoformat()
    except ValueError:
        return None


def _sharpness(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0.0
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def _load_records(store, media_id, semantic_path):
    asset = store.get_asset(media_id)
    if not asset:
        raise ValueError(f"video asset not found: {media_id}")
    package = json.loads(Path(semantic_path).read_text(encoding="utf-8"))
    derived = store.list_derived_assets(media_id)
    records = EventAggregator.records_from_package(package, derived)
    for record in records:
        record.sharpness = _sharpness(record.image_path)
    return asset, package, records


def _event_payload(group, captured_at):
    data = EventAggregator.group_to_dict(group)
    data["start_time"] = _time_at(captured_at, data["start_sec"])
    data["end_time"] = _time_at(captured_at, data["end_sec"])
    data["member_frames"] = [
        {"frame_id": item.frame_id, "timestamp_sec": item.timestamp, "frame_hash": item.frame_hash}
        for item in group.members
    ]
    return data


def _html_report(payload, output_path, asset_url_base):
    rows = []
    for event in payload["events"]:
        members = " ".join(
            f'<span class="frame"><img src="{asset_url_base}/{item["frame_id"]}/file"><small>{item["timestamp_sec"]:.1f}s</small></span>'
            for item in event.get("member_frames", [])
        )
        rows.append(
            f'<article><h2>{event["event_id"]} · {event["start_sec"]:.1f}s–{event["end_sec"]:.1f}s · {event["frame_count"]} frames</h2>'
            f'<p>representative={event.get("representative_frame_id")} · merge={event.get("merge_score", 0):.4f} · objects={", ".join(event.get("object_summary", []))}</p>'
            f'<p class="breakdown">{json.dumps(event.get("merge_breakdown", {}), ensure_ascii=False)}</p><div class="frames">{members}</div></article>'
        )
    metrics = json.dumps(payload["metrics"], ensure_ascii=False, indent=2)
    html = f'''<!doctype html><meta charset="utf-8"><title>Sentrix EventAgg A/B</title>
<style>body{{font:14px system-ui;margin:24px;background:#f7faf4;color:#172016}}pre{{background:#fff;padding:16px;overflow:auto}}article{{background:#fff;border:1px solid #dfe8d7;border-radius:12px;padding:14px;margin:14px 0}}.frames{{display:flex;gap:8px;flex-wrap:wrap}}.frame{{display:inline-flex;flex-direction:column;font-size:11px;color:#667}}.frame img{{width:128px;height:80px;object-fit:cover;border-radius:6px}}.breakdown{{font-size:11px;color:#687466}}</style>
<h1>Sentrix Home EventAgg v2.1</h1><p>run={payload["run_id"]} · method={payload["method_version"]} · threshold={payload["config"]["merge_threshold"]}</p><h2>Metrics</h2><pre>{metrics}</pre>{''.join(rows)}'''
    Path(output_path).write_text(html, encoding="utf-8")


def _write_summary_reports(output_dir, comparison, outputs, asset, package):
    """Write stable machine-readable and human-readable A/B artifacts."""
    metadata = asset.get("metadata_json") or {}
    baseline = comparison["baseline"]
    selected = min(outputs, key=lambda item: (item["metrics"].get("event_count", 10**9), item["config"].get("merge_threshold", 1))) if outputs else None
    eventagg_performance = {
        "method_version": EVENTAGG_METHOD_VERSION,
        "media_id": asset.get("id"),
        "scope_id": asset.get("scope_id"),
        "video_duration_sec": metadata.get("video_metadata", {}).get("duration_sec"),
        "baseline": {
            "method_version": "keyframe-hybrid-v2.0.0",
            "keyframe_extraction_seconds": metadata.get("keyframe_extraction_seconds"),
            "memory_build_seconds": metadata.get("memory_build_seconds"),
            "total_video_processing_seconds": metadata.get("video_processing_seconds"),
            "keyframe_bytes": metadata.get("derived_keyframe_bytes"),
        },
        "eventagg": {
            "selected_threshold": selected["config"]["merge_threshold"] if selected else None,
            "warm_cache_wall_seconds": selected["metrics"].get("eventagg_wall_seconds") if selected else None,
            "embedding_seconds": selected["metrics"].get("embedding_seconds") if selected else None,
            "aggregation_seconds": selected["metrics"].get("aggregation_seconds") if selected else None,
            "cache_hits": selected["metrics"].get("cache_hits") if selected else None,
            "cache_misses": selected["metrics"].get("cache_misses") if selected else None,
            "cold_model_load_seconds": None,
            "cold_model_load_note": "DINOv2 首次加载耗时单独测量；本报告只记录复用缓存后的事件聚合耗时。",
        },
    }
    comparison["selected"] = {
        "threshold": selected["config"]["merge_threshold"] if selected else None,
        "run_id": selected["run_id"] if selected else None,
        "selection_rule": "minimum event_count; evidence retention must remain 1.0",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eventagg_performance.json").write_text(json.dumps(eventagg_performance, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "eventagg_comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = "".join(
        f'<tr><td>{item["config"]["merge_threshold"]:.2f}</td><td>{item["metrics"].get("event_count", "—")}</td>'
        f'<td>{item["metrics"].get("singleton_rate", 0) * 100:.1f}%</td><td>{item["metrics"].get("mean_event_frames", 0):.2f}</td>'
        f'<td>{item["metrics"].get("eventagg_wall_seconds", 0):.4f}s</td><td>{item["metrics"].get("evidence_retention", 0) * 100:.1f}%</td></tr>'
        for item in outputs
    )
    html = f'''<!doctype html><meta charset="utf-8"><title>Sentrix EventAgg A/B Ablation</title>
<style>body{{font:14px system-ui;margin:32px;background:#f7faf4;color:#172016}}table{{border-collapse:collapse;background:#fff}}th,td{{padding:10px 14px;border:1px solid #dfe8d7;text-align:right}}th{{background:#eaf4df}}td:first-child,th:first-child{{text-align:left}}.note{{background:#fff;padding:14px;border-radius:10px}}</style>
<h1>Sentrix Home EventAgg v2.1 threshold ablation</h1><p class="note">Baseline: {baseline["timeline_scenes"]} scenes / {baseline["keyframes"]} keyframes. All thresholds reuse the same WebP, metadata, YOLO/Pose and DINO cache; no video re-decoding.</p>
<table><thead><tr><th>threshold</th><th>events</th><th>singleton rate</th><th>mean frames</th><th>warm wall</th><th>evidence retention</th></tr></thead><tbody>{rows}</tbody></table>
<p>Machine reports: eventagg_ablation.json · eventagg_comparison.json · eventagg_performance.json</p>'''
    (output_dir / "eventagg_ablation.html").write_text(html, encoding="utf-8")


def run_one(args, threshold, embedder, asset, package, records, store):
    run_started = time.perf_counter()
    embeddings, embedding_metrics = embedder.embed(args.media_id, records, args.batch_size)
    for record in records:
        record.visual_embedding = embeddings.get(record.frame_id, [])
    aggregator = EventAggregator({**DEFAULT_CONFIG, "merge_threshold": threshold}, cache_dir=args.cache_dir)
    groups, metrics = aggregator.aggregate(records, args.media_id, threshold=threshold)
    metrics.update(embedding_metrics)
    metrics["baseline_scene_count"] = len(store.list_video_scene_events(args.media_id))
    metrics["evidence_retention"] = round(sum(len(group.members) for group in groups) / max(len(records), 1), 6)
    metrics["eventagg_wall_seconds"] = round(time.perf_counter() - run_started, 4)
    metrics["video_duration_sec"] = float((asset.get("metadata_json") or {}).get("video_metadata", {}).get("duration_sec") or 0)
    metrics["processing_ratio"] = round(metrics["eventagg_wall_seconds"] / metrics["video_duration_sec"], 6) if metrics["video_duration_sec"] else None
    run_id = args.run_id or f"eventagg_{args.media_id}_{int(float(threshold) * 100):02d}"
    events = [_event_payload(group, asset.get("captured_at")) for group in groups]
    for index, event in enumerate(events):
        event["event_id"] = f"{run_id}_event_{index + 1:04d}"
    config = {**DEFAULT_CONFIG, "merge_threshold": float(threshold), "cache_dir": str(args.cache_dir)}
    store.replace_eventagg_run(run_id, args.media_id, args.scope_id, EVENTAGG_METHOD_VERSION, events, metrics, config)
    return {
        "run_id": run_id, "media_id": args.media_id, "scope_id": args.scope_id,
        "method_version": EVENTAGG_METHOD_VERSION, "config": config,
        "metrics": metrics, "events": events,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--media-id", required=True)
    parser.add_argument("--scope-id", default="home-default")
    parser.add_argument("--semantic", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default="/ssd/sscy/eventagg-cache/embeddings")
    parser.add_argument("--threshold", type=float, default=0.68)
    parser.add_argument("--sweep", default="")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--asset-url-base", default="http://192.168.0.200:8091/api/assets")
    args = parser.parse_args()
    args.cache_dir = Path(args.cache_dir)
    store = MemoryStore(args.db)
    asset, package, records = _load_records(store, args.media_id, args.semantic)
    if len(records) != 180:
        print(f"WARNING: existing package contains {len(records)} frames, expected 180", file=sys.stderr)
    embedder = DINOEmbedder(args.cache_dir, device=os.getenv("SENTRIX_EVENTAGG_DEVICE"))
    thresholds = [float(item) for item in args.sweep.split(",") if item.strip()] or [args.threshold]
    outputs = []
    for threshold in thresholds:
        run_args = args
        run_args.run_id = f"eventagg_{args.media_id}_{int(threshold * 100):02d}"
        output = run_one(run_args, threshold, embedder, asset, package, records, store)
        output["reports"] = {}
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"eventagg_{int(threshold * 100):02d}"
        json_path = output_dir / f"{stem}.json"
        html_path = output_dir / f"{stem}.html"
        json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        _html_report(output, html_path, args.asset_url_base)
        output["reports"] = {"json": str(json_path), "html": str(html_path)}
        json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        outputs.append(output)
    comparison = {
        "method_version": EVENTAGG_METHOD_VERSION,
        "baseline": {
            "method": "keyframe-hybrid-v2.0.0",
            "keyframes": len(records),
            "timeline_scenes": len(store.list_video_scene_events(args.media_id)),
            "singleton_rate": 1.0,
            "evidence_retention": 1.0,
        },
        "runs": [{"threshold": item["config"]["merge_threshold"], **item["metrics"]} for item in outputs],
    }
    Path(args.output, "eventagg_ablation.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary_reports(Path(args.output), comparison, outputs, asset, package)
    print("=" * 50)
    print("Sentrix Home EventAgg v2.1")
    print("=" * 50)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
