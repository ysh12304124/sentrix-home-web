#!/usr/bin/env python3
"""Audit persisted Agent2 traces for the production evidence contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PUBLIC = {
    "memory_asset", "user_statement", "confirmed_identity", "photo_identity",
    "temporal_metadata", "location_metadata", "structured_fact",
    "visual_observation", "visible_text",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    stats = {"items": 0, "trace_items": 0, "missing_trace_fields": 0,
             "invalid_evidence_types": 0, "handle_out_of_chain": 0,
             "duplicate_tool_calls": 0, "final_gate_noncomplete": 0,
             "writer_missing": 0}
    details = []
    for line in args.results.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        stats["items"] += 1
        trace = item.get("agent2_trace") or {}
        if not trace:
            continue
        stats["trace_items"] += 1
        seen_calls = set()
        handles = set()
        # `runtime_turns[].debug_trace` is the authoritative persisted Agent2
        # trace.  `tool_trace` is a presentation projection and intentionally
        # omits some contract fields.
        tool_steps = []
        for runtime_turn in item.get("runtime_turns") or []:
            tool_steps.extend(x for x in runtime_turn.get("debug_trace") or []
                              if x.get("type") == "tool")
        if not tool_steps:
            tool_steps = [x for x in item.get("tool_trace") or [] if x.get("type") == "tool"]
        for step in tool_steps:
            if step.get("type") != "tool":
                continue
            args0 = step.get("arguments") or {}
            sig = (step.get("tool"), json.dumps(args0, sort_keys=True, ensure_ascii=False))
            # A bounded auto-resolution retry is deliberately allowed to use
            # the same handle after an uncertain/partial result; it is not an
            # uncontrolled duplicate loop.
            if sig in seen_calls and not step.get("auto_resolution"):
                stats["duplicate_tool_calls"] += 1
            if not step.get("auto_resolution"):
                seen_calls.add(sig)
            obs = step.get("observation") or {}
            for preview in obs.get("preview") or []:
                if preview.get("handle"):
                    handles.add(str(preview["handle"]))
            if step.get("tool") == "inspect_photo":
                handle = str(args0.get("asset_handle") or "")
                if handle and handle not in handles:
                    stats["handle_out_of_chain"] += 1
        entries = (trace.get("evidence_ledger") or {}).get("entries") or []
        for entry in entries:
            if entry.get("evidence_type") not in PUBLIC:
                stats["invalid_evidence_types"] += 1
        for step in tool_steps:
            required = {"raw_arguments", "normalized_arguments", "task_status_before",
                        "requirement_status_before", "standardized_evidence", "evidence_ids",
                        "task_status_after", "requirement_status_after"}
            if not required.issubset(step):
                stats["missing_trace_fields"] += 1
        final_gate = trace.get("final_gate") or {}
        if final_gate and final_gate.get("status") not in {"complete", "answered"}:
            stats["final_gate_noncomplete"] += 1
        if trace.get("task_status") == "complete" and not trace.get("writer_output"):
            # Older persisted traces expose writer output only in execution_trace;
            # count as missing only when neither representation exists.
            if not any(x.get("stage") == "writer" for x in item.get("execution_trace") or []):
                stats["writer_missing"] += 1
        if stats["missing_trace_fields"] or stats["invalid_evidence_types"] or stats["handle_out_of_chain"]:
            details.append({"qa_id": item.get("qa_id"), "trace": trace,
                            "tool_trace": tool_steps})
    stats["structural_pass"] = not any(stats[k] for k in (
        "missing_trace_fields", "invalid_evidence_types", "handle_out_of_chain",
        "final_gate_noncomplete", "writer_missing"))
    payload = {"stats": stats, "details": details[:20]}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
