#!/usr/bin/env python3
"""A5 — Offline A/B：Tool-Loop Shadow vs Canonical RX Pipeline。

同一 DB snapshot / 同一 Retrieval Kernel / 同一 12B，唯一主变量是编排方式：
- Tool-Loop：模型自主选择 Tool（shadow 结果文件）
- Canonical RX：固定 Parser -> Router -> Evidence -> AnswerBrief -> Writer（本脚本运行）

只读，不写数据库。用法:
  python evaluate_tool_loop_ab.py --cases shadow_cases_v1.json \
      --tool-loop /tmp/shadow_v12.json --out /tmp/ab_v1.json --scope album2_e2b
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

# Canonical RX 标志（与 8092 validation 实例一致；validation ledger 不影响答案路径）。
RX_ENV = {
    "SENTRIX_RX_V1": "1",
    "SENTRIX_ANSWER_BRIEF_V1": "1",
    "SENTRIX_RESPONSE_PLAN_V1": "1",
    "SENTRIX_VISIBLE_EVIDENCE_V1": "1",
    "SENTRIX_RESPONSE_WRITER_V2": "1",
    "SENTRIX_RESPONSE_VALIDATOR_V1": "1",
    "SENTRIX_THIN_AGENT_V1": "1",
    "SENTRIX_EVIDENCE_MULTI_RETRIEVER_V1": "1",
    "SENTRIX_MODEL_SPLIT_V1": "1",
    "SENTRIX_AGENT_MODEL_PROFILE": "quality_12b",
    "SENTRIX_EVIDENCE_ANSWER_12B": "1",
    "SENTRIX_AGENT_STAGE_TRACE": "1",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--tool-loop", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scope", default="album2_e2b")
    ap.add_argument("--base", default="http://127.0.0.1:8100/v1")
    args = ap.parse_args()

    for k, v in RX_ENV.items():
        os.environ[k] = v

    from backend.db import MemoryStore
    from backend.model_clients import GammaClient
    from backend.thin_agent import ThinAgentRuntime

    store = MemoryStore(os.getenv("SENTRIX_DB_PATH", os.path.join(ROOT, "data", "sentrix.db")))
    gamma = GammaClient(base_url=args.base, backend="openai")
    agent = ThinAgentRuntime(store, gamma=gamma)

    cases = json.load(open(args.cases, encoding="utf-8"))
    rx_results = []
    for case in cases:
        t0 = time.time()
        try:
            result = agent.answer_turn(case["query"], scope_id=args.scope, viewer_id="owner")
            trace = (result or {}).get("retrieval_trace") or []
            rx_trace = [t for t in trace if t.get("stage") == "rx"]
            rx_results.append({
                "id": case["id"], "query": case["query"],
                "answer": (result or {}).get("answer") or "",
                "mode": (result or {}).get("response_mode") or "",
                "rx_used": bool(rx_trace),
                "rx_fallback_used": bool((result or {}).get("rx_fallback_used")),
                "degraded": bool(((result or {}).get("perf") or {}).get("degraded")),
                "latency_s": round(time.time() - t0, 2),
                "perf": (result or {}).get("perf"),
            })
        except Exception as exc:
            rx_results.append({
                "id": case["id"], "query": case["query"],
                "answer": "", "mode": "error", "rx_used": False,
                "degraded": False, "latency_s": round(time.time() - t0, 2),
                "error": str(exc)[:200],
            })
        r = rx_results[-1]
        print(f"{r['id']} mode={r['mode']} rx={r['rx_used']} fallback={r.get('rx_fallback_used')} {r['latency_s']}s | {r['answer'][:44]}")
        sys.stdout.flush()

    tl = json.load(open(args.tool_loop, encoding="utf-8"))
    by_id = {r["id"]: r for r in rx_results}
    paired = []
    for c in tl:
        paired.append({
            "id": c["id"], "category": c.get("category"), "query": c.get("query"),
            "tool_loop": {
                "status": c.get("status"), "reason": c.get("reason"),
                "answer": c.get("final_answer"), "latency_s": c.get("latency_s"),
                "steps": c.get("steps"),
            },
            "canonical_rx": {
                "mode": by_id.get(c["id"], {}).get("mode"),
                "rx_used": by_id.get(c["id"], {}).get("rx_used"),
                "rx_fallback_used": by_id.get(c["id"], {}).get("rx_fallback_used"),
                "answer": by_id.get(c["id"], {}).get("answer"),
                "latency_s": by_id.get(c["id"], {}).get("latency_s"),
                "error": by_id.get(c["id"], {}).get("error"),
            },
        })
    summary = {
        "scope": args.scope,
        "canonical_env": RX_ENV,
        "tool_loop_file": args.tool_loop,
        "tool_loop_complete": sum(1 for c in tl if c.get("status") == "complete"),
        "tool_loop_blocked": sum(1 for c in tl if c.get("status") == "blocked_by_guard"),
        "tool_loop_partial": sum(1 for c in tl if c.get("status") == "partial"),
        "tool_loop_total": len(tl),
        "canonical_answered": sum(1 for r in rx_results if r.get("answer")),
        "canonical_rx_used": sum(1 for r in rx_results if r.get("rx_used")),
        "canonical_degraded": sum(1 for r in rx_results if r.get("degraded")),
        "canonical_total": len(rx_results),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "cases": paired, "rx_raw": rx_results},
                  f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
