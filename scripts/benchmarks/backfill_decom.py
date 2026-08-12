#!/usr/bin/env python3
"""F9 backfill — 给历史 QA run 补 decompose 分层 + tool_perf 聚合，并重新上传 Dashboard。

用法：
  python backfill_decom.py --runs 20260811_180254_phasee_final3,20260811_184229_phasef_f1
  python backfill_decom.py --all
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decompose_layers import decompose_row, decompose_summary, aggregate_tool_perf

DEFAULT_QA_DIR = Path("~/Downloads/sentrix_qa_report").expanduser()
DEFAULT_BASE = "http://192.168.0.153:4174"
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def recompute_summary(rows):
    n = len(rows)
    statuses = Counter(r.get("status") or "error" for r in rows)
    judged = [r for r in rows if (r.get("judge") or {}).get("judged")]
    verdicts = Counter((r.get("judge") or {}).get("verdict") for r in judged)
    ev_has = [r for r in rows if (r.get("evidence") or {}).get("has_gold")]
    ev_hit = sum(1 for r in ev_has if (r.get("evidence") or {}).get("hit"))
    ev_recall = round(sum((r.get("evidence") or {}).get("recall") or 0 for r in ev_has) / max(1, len(ev_has)), 3)
    usage = Counter()
    for r in rows:
        for t in r.get("tools") or []:
            usage[t] += 1
    return {
        "total": n,
        "errored": sum(1 for r in rows if r.get("error")),
        "statuses": dict(statuses),
        "avg_latency_s": round(sum(r.get("latency_s") or 0 for r in rows) / max(1, n), 1),
        "judged": len(judged),
        "verdicts": dict(verdicts),
        "evidence_questions": len(ev_has),
        "evidence_hit": ev_hit,
        "evidence_recall_avg": ev_recall,
        "tool_usage": dict(usage),
        "decom": decompose_summary(rows),
        "tool_perf": aggregate_tool_perf(rows),
    }


def upload(base, run_id, payload):
    req = urllib.request.Request(
        f"{base}/api/qa/runs/upload",
        data=json.dumps(payload).encode(),
        method="POST",
        headers={"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa-dir", default=str(DEFAULT_QA_DIR))
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--runs", default="", help="逗号分隔 run_id；空则全部")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()

    qa_dir = Path(args.qa_dir).expanduser() / "runs"
    run_ids = [x.strip() for x in args.runs.split(",") if x.strip()]
    if args.all:
        run_ids = [e.name for e in sorted(qa_dir.iterdir()) if e.is_dir()]
    if not run_ids:
        run_ids = [e.name for e in sorted(qa_dir.iterdir()) if e.is_dir()]

    for run_id in run_ids:
        rp = qa_dir / run_id / "qa_result.json"
        if not rp.is_file():
            print(f"[skip] {run_id}: 无 qa_result.json")
            continue
        data = json.loads(rp.read_text(encoding="utf-8"))
        rows = data.get("rows") or []
        changed = 0
        for r in rows:
            if not r.get("decom"):
                r["decom"] = decompose_row(r)
                changed += 1
        data["summary"] = recompute_summary(rows)
        rp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ok] {run_id}: 补 decom {changed} 行，summary 已刷新")
        if not args.no_upload:
            meta = data.get("meta") or {}
            res = upload(args.base, run_id, {
                "run_id": run_id, "meta": meta, "summary": data["summary"],
                "rows": rows, "asset_map": data.get("asset_map") or {},
                "tag": meta.get("tag") or run_id,
                "note": meta.get("note", ""),
                "profile": (meta.get("profile") or ""),
            })
            print(f"     upload -> {res.get('status')} ({res.get('run_id')})")


if __name__ == "__main__":
    main()
