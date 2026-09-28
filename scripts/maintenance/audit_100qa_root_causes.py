#!/usr/bin/env python3
"""Produce a deterministic root-cause ledger from a Photobench run.

This audit never calls a model and never changes the benchmark database.  It
keeps judge validity, retrieval recall, tool routing and answer-context signals
separate so an invalid or stale run cannot be treated as a retrieval verdict.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path


def _score(item):
    value = (item.get("judge") or {}).get("score")
    return value if value in {0, 1, 2} else None


def _tool_names(item):
    return [str(row.get("tool") or "") for row in item.get("tool_trace") or []
            if isinstance(row, dict)]


def audit(run_path: Path) -> dict:
    run = json.loads(run_path.read_text(encoding="utf-8"))
    items = run.get("items") or []
    scored = [item for item in items if _score(item) is not None]
    full_recall = [item for item in scored if float(item.get("retrieval_recall") or 0) >= 1.0]
    full_recall_aq0 = [item for item in full_recall if _score(item) == 0]
    non_full = [item for item in scored if float(item.get("retrieval_recall") or 0) < 1.0]
    tool_counts = Counter(tool for item in items for tool in _tool_names(item))
    return {
        "run": {
            "run_id": run.get("run_id"),
            "path": str(run_path),
            "status": run.get("status"),
            "run_valid": run.get("run_valid"),
            "qa_count": run.get("qa_count"),
            "summary": run.get("summary") or {},
        },
        "validity": {
            "items": len(items),
            "judge_valid": len(scored),
            "judge_missing": len(items) - len(scored),
            "all_agents_completed": all(item.get("agent_status") in {None, "completed", "complete"}
                                         for item in items),
        },
        "retrieval": {
            "full_recall_scored": len(full_recall),
            "full_recall_aq0": len(full_recall_aq0),
            "full_recall_aq0_rate": round(len(full_recall_aq0) / len(full_recall), 4) if full_recall else None,
            "non_full_recall_scored": len(non_full),
            "mean_recall_scored": round(sum(float(i.get("retrieval_recall") or 0) for i in scored) / len(scored), 4) if scored else None,
        },
        "tool_routing": {
            "counts": dict(tool_counts),
            "search_calls": tool_counts.get("search_memories", 0),
            "page_calls": tool_counts.get("get_result_page", 0),
            "inspect_calls": tool_counts.get("inspect_photo", 0),
            "ocr_calls": tool_counts.get("read_photo_text", 0),
        },
        "sample_failures": [
            {
                "qa_id": item.get("qa_id"),
                "question": item.get("question"),
                "score": _score(item),
                "retrieval_recall": item.get("retrieval_recall"),
                "predicted_file_names": item.get("predicted_file_names"),
                "gt_images": item.get("gt_images"),
                "tools": _tool_names(item),
                "answer": item.get("answer"),
            }
            for item in full_recall_aq0[:20]
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = audit(args.run)
    try:
        payload["source"] = {
            "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "status": subprocess.check_output(["git", "status", "--short"], text=True).splitlines(),
        }
    except Exception:
        payload["source"] = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "run_id": payload["run"]["run_id"],
        "judge_valid": payload["validity"]["judge_valid"],
        "full_recall_aq0": payload["retrieval"]["full_recall_aq0"],
        "full_recall_aq0_rate": payload["retrieval"]["full_recall_aq0_rate"],
        "out": str(args.out),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
