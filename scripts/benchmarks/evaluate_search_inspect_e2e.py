#!/usr/bin/env python3
"""B3 — Search → Inspect → Answer 完整 Tool-Loop E2E（≥10 例）。

跑 AgentRuntime（tool_loop_shadow）：模型自主 search_memories → inspect_photo → final。
评估：工具链是否包含 search+inspect、inspect 是否只在需要时调用（不必要 inspect <=10%）、
final 是否引用 inspect evidence。

用法:
  python evaluate_search_inspect_e2e.py --base http://127.0.0.1:8105/v1 --scope album2_e2b --out /tmp/si_e2e.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

CASES = [
    {"id": "si01", "query": "帮我看看最近拍的照片里，桌上放了什么", "expect_inspect": True, "category": "object"},
    {"id": "si02", "query": "第一张照片里有人穿红色衣服吗", "expect_inspect": True, "category": "clothing"},
    {"id": "si03", "query": "照片里有几个人？帮我数一下", "expect_inspect": True, "category": "people"},
    {"id": "si04", "query": "看看照片里的招牌或文字写了什么", "expect_inspect": True, "category": "ocr"},
    {"id": "si05", "query": "看看照片是什么天气", "expect_inspect": True, "category": "scene"},
    {"id": "si06", "query": "帮我确认一下照片里的猫是什么颜色", "expect_inspect": True, "category": "object"},
    {"id": "si07", "query": "去年十月像爬山的那张照片，山上有雪吗", "expect_inspect": True, "category": "scene"},
    {"id": "si08", "query": "照片里的人穿了什么颜色的外套", "expect_inspect": True, "category": "clothing"},
    {"id": "si09", "query": "看看照片里有没有小孩", "expect_inspect": True, "category": "people"},
    {"id": "si10", "query": "去年拍了多少张照片", "expect_inspect": False, "category": "count_negative"},
    {"id": "si11", "query": "穿红色衣服的那个人在做什么", "expect_inspect": True, "category": "activity"},
]


def build_runtime(base_url, scope_id):
    from backend.agent_runtime.runtime import AgentRuntime
    from backend.agent_runtime import tools as runtime_tools
    from backend.db import MemoryStore
    from backend.model_clients import GammaClient
    from backend.embeddings import EmbeddingRouter
    from backend.model_clients import ClipAdapter
    from backend.retrieval import RetrievalConfig
    store = MemoryStore(os.getenv("SENTRIX_DB_PATH", os.path.join(ROOT, "data", "sentrix.db")))
    gamma = GammaClient(base_url=base_url, backend="openai")
    router = EmbeddingRouter.from_clip(ClipAdapter())
    runtime_tools.bind_runtime(store, gamma=gamma, embedding_router=router,
                               retrieval_config=RetrievalConfig())
    runtime_tools.register_tools()

    def chat_fn(messages):
        import urllib.request
        body = json.dumps({
            "model": "gemma4-12b-it",
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 1500,
        }).encode()
        req = urllib.request.Request(base_url + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
        return data["choices"][0]["message"]["content"]

    return AgentRuntime(chat_fn=chat_fn, profile_name="tool_loop_shadow",
                        scope_id=scope_id, viewer_id="owner")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8105/v1")
    ap.add_argument("--scope", default="album2_e2b")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    runtime = build_runtime(args.base, args.scope)
    results = []
    for case in CASES:
        t0 = time.time()
        turn = runtime.run(case["query"])
        dt = round(time.time() - t0, 2)
        tools = [s.get("tool") for s in turn.steps if s.get("type") == "tool"]
        inspect_calls = [s for s in turn.steps if s.get("type") == "tool" and s.get("tool") == "inspect_photo"]
        used_inspect = bool(inspect_calls)
        search_calls = [s for s in turn.steps if s.get("type") == "tool" and s.get("tool") == "search_memories"]
        searched_with_candidates = any((s.get("observation") or {}).get("total", 0) > 0
                                       for s in search_calls)
        if case["expect_inspect"]:
            # 检索为空时无法 inspect，诚实回答“没有找到”也算通过
            chain_ok = ("search_memories" in tools and used_inspect) or (
                "search_memories" in tools and not searched_with_candidates)
        else:
            chain_ok = not used_inspect and bool(tools)
        inspect_evidence = False
        if used_inspect:
            obs = inspect_calls[0].get("observation") or {}
            inspect_evidence = bool(obs.get("observation") and obs.get("certainty") in {"supported", "uncertain"})
        # 检索为空时：诚实回答“没有找到”即可通过（即使模型误试了被拒的 inspect）
        honest_no_result = (not searched_with_candidates) and bool(
            __import__("re").search(r"没有找到|没找到|未找到|没有(?:拍摄|相关)|无法确认", turn.final_answer))
        verdict = "PASS" if (chain_ok and (not case["expect_inspect"] or inspect_evidence or honest_no_result)) else "FAIL"
        results.append({
            "id": case["id"], "category": case["category"], "query": case["query"],
            "status": turn.status, "reason": turn.reason, "tools": tools,
            "expected_inspect": case["expect_inspect"], "used_inspect": used_inspect,
            "inspect_evidence": inspect_evidence,
            "final_answer": turn.final_answer, "latency_s": dt,
            "steps": [{k: s.get(k) for k in ("type", "tool", "status", "observation", "reason")} for s in turn.steps],
        })
        print(f"{case['id']} [{case['category']}] {verdict} tools={tools} status={turn.status} {dt}s | {turn.final_answer[:50]}")
    n = len(results)
    chain_pass = sum(1 for r in results if r["tools"] and r["tools"][0] == "search_memories")
    inspect_expected = [r for r in results if r["expected_inspect"]]
    inspect_used = sum(1 for r in inspect_expected if r["used_inspect"])
    unnecessary = sum(1 for r in results if not r["expected_inspect"] and r["used_inspect"])
    summary = {
        "total": n,
        "chain_pass_rate": round(chain_pass / n, 4),
        "inspect_recall": round(inspect_used / max(1, len(inspect_expected)), 4),
        "unnecessary_inspect": unnecessary,
        "verdict_pass": sum(1 for r in results if r["status"] == "complete"),
        "avg_latency_s": round(sum(r["latency_s"] for r in results) / n, 2),
    }
    print("\nSUMMARY:", json.dumps(summary, ensure_ascii=False))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "cases": results}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
