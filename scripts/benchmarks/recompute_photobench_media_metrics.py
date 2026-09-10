"""Recompute PhotoBench media metrics from saved retrieval IDs without LLM calls.

This is primarily used after metric-normalization changes such as projecting a
derived video keyframe to its parent video.  The script is dry-run by default;
``--write`` updates run.json/results.jsonl after making timestamped backups.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR_PATH = (
    REPO_ROOT / "services" / "photobench" / "backend" / "benchmark_orchestrator.py"
)


def _load_orchestrator():
    spec = importlib.util.spec_from_file_location("photobench_metric_recompute", ORCHESTRATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _assets_by_name(db_path: Path, scope_id: str) -> dict[str, list[dict]]:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT * FROM assets WHERE scope_id = ?", (scope_id,),
        ).fetchall()
    finally:
        connection.close()
    result: dict[str, list[dict]] = {}
    for row in rows:
        asset = dict(row)
        try:
            asset["metadata_json"] = json.loads(asset.get("metadata_json") or "{}")
        except (TypeError, json.JSONDecodeError):
            asset["metadata_json"] = {}
        result.setdefault(Path(asset.get("file_name") or "").name, []).append(asset)
    return result


def _recompute_item(module, item: dict, assets_by_name: dict) -> None:
    gt_refs = module._normalize_media_refs(item, "retrieval")
    selected_media = module._resolve_predicted_media(
        list(item.get("selected_asset_ids") or []), assets_by_name,
    )
    retrieved_media = module._resolve_predicted_media(
        list(item.get("retrieved_asset_ids") or []), assets_by_name,
    )
    evidence_media = module._resolve_predicted_media(
        list(item.get("evidence_asset_ids") or []), assets_by_name,
    )
    metrics = module._modality_metrics(gt_refs, retrieved_media)
    rank_metrics = module._ranked_retrieval_metrics(gt_refs, retrieved_media)
    gt_media = module._resolve_gt_media(gt_refs, assets_by_name, retrieved_media)

    retrieved_keys = {module._evaluation_media_key(value) for value in retrieved_media}
    evidence_keys = {module._evaluation_media_key(value) for value in evidence_media}
    selected_keys = {module._evaluation_media_key(value) for value in selected_media}
    matched = sorted(
        entry["file_name"] for entry in gt_media
        if module._media_key(entry["media_type"], entry["media_id"]) in retrieved_keys
    )
    evidence_matched = sorted(
        entry["file_name"] for entry in gt_media
        if module._media_key(entry["media_type"], entry["media_id"]) in evidence_keys
    )
    delivery_matched = sorted(
        entry["file_name"] for entry in gt_media
        if module._media_key(entry["media_type"], entry["media_id"]) in selected_keys
    )
    selected_names = sorted({value["file_name"] for value in selected_media})
    retrieved_names = sorted({value["file_name"] for value in retrieved_media})
    evidence_names = sorted({value["file_name"] for value in evidence_media})

    item.update({
        "predicted_media": selected_media,
        "retrieved_candidate_media": retrieved_media,
        "evidence_source_media": evidence_media,
        "predicted_images": [value for value in selected_media if value["media_type"] == "image"],
        "retrieved_candidate_images": [
            value for value in retrieved_media if value["media_type"] == "image"
        ],
        "evidence_source_images": [
            value for value in evidence_media if value["media_type"] == "image"
        ],
        "predicted_file_names": selected_names,
        "retrieved_file_names": retrieved_names,
        "evidence_source_file_names": evidence_names,
        "matched_file_names": matched,
        "retrieved_matched_file_names": matched,
        "evidence_matched_file_names": evidence_matched,
        "delivery_matched_file_names": delivery_matched,
        "selected_delivery_file_names": selected_names,
        "media_retrieval_counts": metrics["media"],
        "image_retrieval_counts": metrics["image"],
        "video_retrieval_counts": metrics["video"],
        "media_retrieval_recall": metrics["media"]["recall"],
        "media_retrieval_precision": metrics["media"]["precision"],
        "media_retrieval_f1": metrics["media"]["f1"],
        "image_retrieval_recall": metrics["image"]["recall"],
        "image_retrieval_precision": metrics["image"]["precision"],
        "image_retrieval_f1": metrics["image"]["f1"],
        "video_retrieval_recall": metrics["video"]["recall"],
        "video_retrieval_precision": metrics["video"]["precision"],
        "video_retrieval_f1": metrics["video"]["f1"],
        "retrieval_recall": metrics["media"]["recall"],
        "retrieval_precision": metrics["media"]["precision"],
        "retrieval_f1": metrics["media"]["f1"],
        **rank_metrics,
        "gt_media": gt_media,
        "gt_images": [value for value in gt_media if value["media_type"] == "image"],
    })
    item["attribution"] = module.BenchmarkRun._derive_attribution(item)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-root", type=Path,
                        default=REPO_ROOT / "services" / "photobench" / "results")
    parser.add_argument("--db", type=Path, default=REPO_ROOT / "data" / "sentrix.db")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    module = _load_orchestrator()
    run_dir = (args.results_root / args.run_id).resolve()
    run_path = run_dir / "run.json"
    results_path = run_dir / "results.jsonl"
    if not run_path.is_file():
        raise FileNotFoundError(run_path)
    state = json.loads(run_path.read_text(encoding="utf-8"))
    if str(state.get("status") or "").lower() in {"pending", "running", "cancelling"}:
        raise RuntimeError("refusing to rescore an active PhotoBench run")
    scope_id = str(state.get("scope_id") or "").strip()
    if not scope_id:
        raise RuntimeError("run has no scope_id")

    assets_by_name = _assets_by_name(args.db.resolve(), scope_id)
    if not assets_by_name:
        raise RuntimeError(f"no assets found for scope {scope_id}")
    items = state.get("items") or []
    before = dict(state.get("summary") or {})
    for item in items:
        _recompute_item(module, item, assets_by_name)
    module.OrchestratorRepository._refresh_summary(state)
    after = state.get("summary") or {}

    report = {
        "run_id": args.run_id,
        "scope_id": scope_id,
        "items": len(items),
        "assets": sum(len(values) for values in assets_by_name.values()),
        "before": {
            "media_retrieval_recall_macro": before.get("media_retrieval_recall_macro"),
            "media_retrieval_metric_count": before.get("media_retrieval_metric_count"),
            "retrieval_excluded_unanswerable_count": before.get(
                "retrieval_excluded_unanswerable_count"
            ),
            "retrieval_r_at_1": before.get("retrieval_r_at_1"),
            "retrieval_r_at_3": before.get("retrieval_r_at_3"),
            "retrieval_r_at_5": before.get("retrieval_r_at_5"),
            "retrieval_r_at_8": before.get("retrieval_r_at_8"),
            "retrieval_mrr": before.get("retrieval_mrr"),
        },
        "after": {
            "media_retrieval_recall_macro": after.get("media_retrieval_recall_macro"),
            "media_retrieval_metric_count": after.get("media_retrieval_metric_count"),
            "retrieval_excluded_unanswerable_count": after.get(
                "retrieval_excluded_unanswerable_count"
            ),
            "retrieval_r_at_1": after.get("retrieval_r_at_1"),
            "retrieval_r_at_3": after.get("retrieval_r_at_3"),
            "retrieval_r_at_5": after.get("retrieval_r_at_5"),
            "retrieval_r_at_8": after.get("retrieval_r_at_8"),
            "retrieval_mrr": after.get("retrieval_mrr"),
        },
        "written": bool(args.write),
    }

    if args.write:
        suffix = datetime.now().strftime("%Y%m%d-%H%M%S")
        shutil.copy2(run_path, run_dir / f"run.before-media-rescore-{suffix}.json")
        if results_path.is_file():
            shutil.copy2(results_path, run_dir / f"results.before-media-rescore-{suffix}.jsonl")
        module.atomic_json(run_path, state)
        temporary = results_path.with_suffix(".jsonl.tmp")
        temporary.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items),
            encoding="utf-8",
        )
        temporary.replace(results_path)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
