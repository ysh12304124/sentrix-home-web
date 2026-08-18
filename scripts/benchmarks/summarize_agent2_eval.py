#!/usr/bin/env python3
"""Compare two Agent 2 evaluation envelopes after PhotoBench execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--gates", required=True, type=Path)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    gates = json.loads(args.gates.read_text(encoding="utf-8"))
    report = {"passed": False, "reason": "PhotoBench metrics are required before comparison",
              "baseline_profile": baseline.get("profile"), "candidate_profile": candidate.get("profile"),
              "gates": gates.get("gates") or {}}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
