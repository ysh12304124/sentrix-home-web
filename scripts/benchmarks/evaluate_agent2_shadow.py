#!/usr/bin/env python3
"""Validate Agent 2's generalization manifest and select comparable run cases.

This utility never forwards expected answers to Sentrix.  Execution remains in
PhotoBench; this tool selects case ids and writes a reproducible evaluation
envelope for a legacy or shadow profile.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_STRATA = (
    "baseline_regression", "multi_hop_composition", "ambiguity_recovery",
    "evidence_boundary", "safety_provenance",
)
FORBIDDEN_CASE_KEYS = {"answer", "reference_answer", "expected_answer", "ground_truth"}


def validate_case_manifest(payload: dict) -> dict:
    strata = payload.get("strata") if isinstance(payload, dict) else None
    if not isinstance(strata, dict) or set(strata) != set(REQUIRED_STRATA):
        raise ValueError("case manifest must contain every required stratum")
    for name in REQUIRED_STRATA:
        rows = strata[name]
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"stratum {name} must be non-empty")
        for row in rows:
            if not isinstance(row, dict) or not str(row.get("case_id") or ""):
                raise ValueError(f"stratum {name} needs case_id")
            if FORBIDDEN_CASE_KEYS & set(row):
                raise ValueError("case manifest must not contain answer labels")
    return strata


def load_case_manifest(path: Path) -> dict:
    return validate_case_manifest(json.loads(path.read_text(encoding="utf-8")))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=("legacy", "goal_driven_shadow"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("agent2_shadow_cases.json"))
    args = parser.parse_args()
    strata = load_case_manifest(args.cases)
    envelope = {"profile": args.profile, "strata": strata, "execution": "photobench_required"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(envelope, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
