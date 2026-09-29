"""Benchmark a vision model's per-image Sentrix semantic-observation contract.

The script is intentionally read-only: it calls the same ``GammaClient`` used by
the ingestion pipeline but never creates assets, observations, or database rows.
It is a precondition for changing the default semantic enrichment model.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.model_clients import GammaClient


REQUIRED_SCALAR_FIELDS = ("caption", "activity", "event_type")
EVIDENCE_FIELDS = ("place", "people", "objects", "clothing", "emotions", "spatial_relations", "ocr_text")
DEFAULT_BASELINE_SECONDS = 18.14


def _has_value(value):
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)


def evaluate_images(image_paths, client, baseline_seconds=DEFAULT_BASELINE_SECONDS, min_speedup=5.0):
    """Return per-image contract and latency measurements without persisting data."""
    rows = []
    for image_path in image_paths:
        path = Path(image_path)
        started_at = time.perf_counter()
        try:
            analysis = client.analyze_image(path, {"file_name": path.name})
            elapsed = time.perf_counter() - started_at
            missing = [field for field in REQUIRED_SCALAR_FIELDS if not _has_value(analysis.get(field))]
            evidence = [field for field in EVIDENCE_FIELDS if _has_value(analysis.get(field))]
            chinese_caption = any("\u4e00" <= character <= "\u9fff" for character in str(analysis.get("caption", "")))
            rows.append({
                "image": str(path),
                "seconds": round(elapsed, 4),
                "valid_json": isinstance(analysis, dict),
                "missing_required_fields": missing,
                "evidence_fields": evidence,
                "caption_is_chinese": chinese_caption,
                "passed": not missing and bool(evidence) and chinese_caption,
            })
        except Exception as error:  # Keep all sample failures visible in one report.
            rows.append({"image": str(path), "seconds": round(time.perf_counter() - started_at, 4), "error": str(error), "passed": False})

    successful_seconds = [row["seconds"] for row in rows if row.get("passed")]
    mean_seconds = statistics.mean(successful_seconds) if successful_seconds else None
    speedup = baseline_seconds / mean_seconds if mean_seconds is not None and mean_seconds > 0 else float("inf") if mean_seconds == 0 else 0.0
    passed = len(rows) > 0 and len(successful_seconds) == len(rows) and speedup >= min_speedup
    return {
        "model": client.model,
        "samples": rows,
        "summary": {
            "sample_count": len(rows),
            "passed_samples": len(successful_seconds),
            "mean_seconds": round(mean_seconds, 4) if mean_seconds else None,
            "baseline_seconds": baseline_seconds,
            "speedup": round(speedup, 4) if speedup != float("inf") else "infinite",
            "minimum_speedup": min_speedup,
            "passed": passed,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:11435")
    parser.add_argument("--model", default="gemma4:12b")
    parser.add_argument("--baseline-seconds", type=float, default=DEFAULT_BASELINE_SECONDS)
    parser.add_argument("--min-speedup", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    missing = [str(path) for path in args.images if not path.is_file()]
    if missing:
        parser.error("image files not found: " + ", ".join(missing))
    result = evaluate_images(args.images, GammaClient(args.base_url, args.model, args.timeout), args.baseline_seconds, args.min_speedup)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["summary"]["passed"] else 1)


if __name__ == "__main__":
    main()
