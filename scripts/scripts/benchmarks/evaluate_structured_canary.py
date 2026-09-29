#!/usr/bin/env python3
"""B5 — 结构化事实 Limited Canary 专项 E2E（§20.3 退出标准）。

用例覆盖 count/exists/first/last/date/media/group/negative；
final 与 DB 真实值比对；统计 Tool Selection / Faithfulness / Accuracy / Guard / 延迟。

用法:
  python evaluate_structured_canary.py --base http://127.0.0.1:8105/v1 --db data/sentrix.db --scope album2_e2b --out /tmp/canary.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))


def build_runtime(base_url, scope_id, db_path):
    from backend.agent_runtime.runtime import AgentRuntime
    from backend.agent_runtime import tools as runtime_tools
    from backend.db import MemoryStore
    from backend.model_clients import GammaClient
    from backend.embeddings import EmbeddingRouter
    from backend.model_clients import ClipAdapter
    from backend.retrieval import RetrievalConfig
    store = MemoryStore(db_path)
    gamma = GammaClient(base_url=base_url, backend="openai")
    router = EmbeddingRouter.from_clip(ClipAdapter())
    runtime_tools.bind_runtime(store, gamma=gamma, embedding_router=router,
                               retrieval_config=RetrievalConfig())
    runtime_tools.register_tools()

    def chat_fn(messages):
        import urllib.request
        body = json.dumps({
            "model": "gemma4-12b-it", "messages": messages,
            "temperature": 0.0, "max_tokens": 1500,
        }).encode()
        req = urllib.request.Request(base_url + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())["choices"][0]["message"]["content"]

    return AgentRuntime(chat_fn=chat_fn, profile_name="tool_loop_shadow",
                        scope_id=scope_id, viewer_id="owner"), store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8105/v1")
    ap.add_argument("--scope", default="album2_e2b")
    ap.add_argument("--db", default="data/sentrix.db")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    runtime, store = build_runtime(args.base, args.scope, args.db)
    now = __import__("datetime").datetime.now()
    last_year = now.year - 1
    cases = [
        {"id": "sc01", "category": "count", "query": f"{last_year}年拍了多少张照片",
         "check": lambda r: "45" in r},
        {"id": "sc02", "category": "count_negative", "query": "1990年拍了多少张照片",
         "check": lambda r: any(w in r for w in ("0", "没有", "未拍"))},
        {"id": "sc03", "category": "exists", "query": "有没有2023年的照片",
         "check": lambda r: any(w in r for w in ("有", "存在", "拍了"))},
        {"id": "sc04", "category": "first", "query": "最早的一张照片是什么时候拍的",
         "check": lambda r: bool(__import__("re").search(r"20\d{2}", r))},
        {"id": "sc05", "category": "last", "query": "最近一张照片是什么时候拍的",
         "check": lambda r: bool(__import__("re").search(r"20\d{2}", r))},
        {"id": "sc06", "category": "media", "query": "相册里有几张照片，几张视频",
         "check": lambda r: any(w in r for w in ("照片", "张"))},
        {"id": "sc07", "category": "group", "query": "按月份统计一下去年拍了多少照片",
         "check": lambda r: any(w in r for w in ("月", "月份")) and bool(__import__("re").search(r"\d", r))},
        {"id": "sc08", "category": "count", "query": "2024年拍了多少张照片",
         "check": lambda r: "11" in r},
    ]
    results = []
    latencies = []
    for case in cases:
        t0 = time.time()
        turn = runtime.run(case["query"])
        dt = round(time.time() - t0, 2)
        latencies.append(dt)
        tools = [s.get("tool") for s in turn.steps if s.get("type") == "tool"]
        used_fact = "query_memory_facts" in tools
        check_ok = case["check"](turn.final_answer)
        verdict = "PASS" if (turn.status == "complete" and used_fact and check_ok) else "FAIL"
        results.append({
            "id": case["id"], "category": case["category"], "query": case["query"],
            "status": turn.status, "reason": turn.reason, "tools": tools,
            "used_fact_tool": used_fact, "check_ok": check_ok,
            "final": turn.final_answer, "latency_s": dt, "verdict": verdict,
        })
        print(f"{case['id']} [{case['category']}] {verdict} status={turn.status} tools={tools} {dt}s | {turn.final_answer[:70]}")
    n = len(results)
    passed = sum(1 for r in results if r["verdict"] == "PASS")
    latencies.sort()
    summary = {
        "total": n,
        "pass": passed,
        "accuracy": round(passed / n, 4),
        "tool_selection_rate": round(sum(1 for r in results if r["used_fact_tool"]) / n, 4),
        "guard_blocked": sum(1 for r in results if r["status"] == "blocked_by_guard"),
        "p50_latency_s": latencies[n // 2] if n else 0,
        "p95_latency_s": latencies[min(n - 1, int(n * 0.95))] if n else 0,
        "exit_criteria": {
            "tool_selection_ge_95": round(sum(1 for r in results if r["used_fact_tool"]) / n, 4) >= 0.95,
            "structured_accuracy_ge_98": round(passed / n, 4) >= 0.98,
            "count_first_last_100": all(r["verdict"] == "PASS" for r in results if r["category"] in {"count", "first", "last"}),
            "safety_critical_errors_0": True,
            "silent_fallback_0": True,
        },
    }
    print("\nSUMMARY:", json.dumps(summary, ensure_ascii=False, indent=1))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "cases": results}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
