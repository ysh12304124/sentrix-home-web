#!/usr/bin/env python3
"""A3/A5 — Tool-Loop 离线 Shadow 评估。

对固定 case 集顺序执行 AgentRuntime（tool_loop_shadow profile），
保存完整 trajectory / 工具调用 / 答案 / 延迟，供与 Canonical RX Pipeline 对比。
只读，不写数据库，不改生产行为。

用法:
  python evaluate_tool_loop_shadow.py --cases cases.json --out shadow_result.json [--base http://127.0.0.1:8100/v1]
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)


ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["tool_call", "final"]},
        "tool": {"type": "string",
                 "enum": ["query_memory_facts", "search_memories", "get_original_photos", "inspect_photo"]},
        "arguments": {"type": "object"},
        "public_status": {"type": "string"},
        "answer": {"type": "string"},
        "evidence_refs": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["action"],
}


def build_runtime(base_url, scope_id="home-default", guided_json=False):
    from backend.agent_runtime.runtime import AgentRuntime
    from backend.agent_runtime import tools as runtime_tools
    from backend.db import MemoryStore
    from backend.model_clients import GammaClient

    store = MemoryStore(os.getenv("SENTRIX_DB_PATH", os.path.join(ROOT, "data", "sentrix.db")))
    gamma = GammaClient(base_url=base_url, backend="openai")
    try:
        from backend.embeddings import EmbeddingRouter
        from backend.model_clients import ClipAdapter
        from backend.retrieval import RetrievalConfig
        embedding_router = EmbeddingRouter.from_clip(ClipAdapter())
        retrieval_config = RetrievalConfig()
    except Exception:
        embedding_router = None
        retrieval_config = None
    runtime_tools.bind_runtime(store, gamma=gamma, embedding_router=embedding_router,
                               retrieval_config=retrieval_config)
    runtime_tools.register_tools()

    def chat_fn(messages):
        # 把 messages 渲染成单 prompt 调 12B（shadow 阶段用 OpenAI 兼容 chat）
        import urllib.request
        payload = {
            "model": "gemma4-12b-it",
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 1500,
        }
        if guided_json:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "action", "schema": ACTION_SCHEMA},
            }
        body = json.dumps(payload).encode()
        req = urllib.request.Request(base_url + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
        return data["choices"][0]["message"]["content"]

    runtime = AgentRuntime(chat_fn=chat_fn, profile_name="tool_loop_shadow",
                           scope_id=scope_id, viewer_id="owner")
    return runtime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="http://127.0.0.1:8100/v1")
    ap.add_argument("--scope", default="home-default")
    ap.add_argument("--guided-json", action="store_true")
    args = ap.parse_args()

    cases = json.load(open(args.cases, encoding="utf-8"))
    runtime = build_runtime(args.base, scope_id=args.scope, guided_json=args.guided_json)
    results = []
    for case in cases:
        t0 = time.time()
        turn = runtime.run(case["query"], history=case.get("history", ""))
        results.append({
            "id": case["id"], "category": case.get("category", ""),
            "query": case["query"], "expected_tool": case.get("expected", ""),
            "status": turn.status, "reason": turn.reason,
            "final_answer": turn.final_answer,
            "steps": turn.steps, "public_progress": turn.public_progress,
            "budget": turn.budget.as_dict(),
            "latency_s": round(time.time() - t0, 2),
        })
        print(f"{case['id']} [{case.get('category','')}] status={turn.status} "
              f"steps={len(turn.steps)} {round(time.time()-t0,1)}s | {turn.final_answer[:40]}")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    n = len(results)
    ok = sum(1 for r in results if r["status"] == "complete")
    print(f"\nSHADOW SUMMARY: {ok}/{n} complete, "
          f"avg {sum(r['latency_s'] for r in results)/max(1,n):.1f}s/turn")


if __name__ == "__main__":
    main()
