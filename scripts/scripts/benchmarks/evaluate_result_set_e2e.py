#!/usr/bin/env python3
"""B3.1/B3.2 — ResultSet 分页/持久化/原图交付验收。

单元级：全量稳定 handle、page() 分页、TTL 过期、scope 校验。
工具级：search -> get_result_page(page2) -> get_original_photos(handle) 授权原图 URL。
模型级：12B 自主 search -> get_result_page -> final（分页续接）。

用法:
  python evaluate_result_set_e2e.py --base http://127.0.0.1:8105/v1 --db data/sentrix.db --scope album2_e2b --out /tmp/rs_e2e.json
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
                        scope_id=scope_id, viewer_id="owner")


def unit_tests():
    """单元级：全量 handle / page / TTL / scope。"""
    from backend.agent_runtime.result_set import ResultSetStore, ResultSet
    checks = []
    store = ResultSetStore(None, ttl_s=3600)
    rs = ResultSet(result_set_id="rs_unit", scope_id="s1", query="q",
                   asset_ids=[f"a{i}" for i in range(25)], total=25)
    store.save(rs)
    handles = rs.handles()
    checks.append(("full_handle_map", len(handles) == 25 and handles["photo_25"] == "a24"))
    p1 = rs.page(1, 6)
    p4 = rs.page(4, 6)
    p5 = rs.page(5, 6)
    checks.append(("page_bounds", len(p1) == 6 and len(p4) == 6 and len(p5) == 1
                   and p4[0]["handle"] == "photo_19" and p5[0]["handle"] == "photo_25"))
    checks.append(("stable_handle", p1[0]["handle"] == "photo_1" and p5[0]["handle"] == "photo_25"))
    checks.append(("resolve_cross_page", store.resolve_handle("rs_unit", "photo_25") == "a24"))
    checks.append(("resolve_bad_handle", store.resolve_handle("rs_unit", "photo_99") is None))
    # TTL 过期
    store2 = ResultSetStore(None, ttl_s=0.05)
    rs2 = ResultSet(result_set_id="rs_ttl", scope_id="s1", query="q",
                    asset_ids=["a1"], total=1)
    store2.save(rs2)
    time.sleep(0.1)
    checks.append(("ttl_expired", store2.get("rs_ttl") is None))
    # scope 校验在工具层（get_result_page）
    return checks


def tool_tests(scope_id, db_path):
    """工具级：search -> page2 -> original(handle)。"""
    from backend.agent_runtime import tools as runtime_tools
    from backend.db import MemoryStore
    from backend.model_clients import GammaClient
    from backend.embeddings import EmbeddingRouter
    from backend.model_clients import ClipAdapter
    from backend.retrieval import RetrievalConfig
    store = MemoryStore(db_path)
    runtime_tools.bind_runtime(store, gamma=GammaClient(), embedding_router=EmbeddingRouter.from_clip(ClipAdapter()),
                               retrieval_config=RetrievalConfig())
    runtime_tools.register_tools()
    checks = []
    ctx = {"scope_id": scope_id, "viewer_id": "owner", "task_state": {}}
    obs = runtime_tools._search_memories({"query": "风景", "mode": "best"}, context=ctx)
    rid = obs["result_set_id"]
    checks.append(("search_has_more", obs["has_more"] is True))
    checks.append(("search_total", obs["total"] > 6))
    p2 = runtime_tools._get_result_page({"result_set_id": rid, "page": 2, "page_size": 6}, context=ctx)
    checks.append(("page2_shown", p2["shown"] == min(p2["total"], 12)))
    checks.append(("page2_handle_global", p2["preview"][0]["handle"] == "photo_7"))
    checks.append(("page2_remaining", p2["remaining"] == max(0, p2["total"] - p2["shown"])))
    orig = runtime_tools._get_original_photos({"result_set_id": rid, "handle": p2["preview"][0]["handle"]}, context=ctx)
    checks.append(("original_delivered", orig["delivered"] == 1))
    checks.append(("original_url", "/api/assistant/result-set/" in orig.get("url", "")))
    # scope 越权
    other_ctx = {"scope_id": "other_scope", "viewer_id": "owner", "task_state": {}}
    denied = runtime_tools._get_result_page({"result_set_id": rid, "page": 1}, context=other_ctx)
    checks.append(("page_scope_denied", denied.get("blocked") == ["scope_mismatch"]))
    denied2 = runtime_tools._get_original_photos({"result_set_id": rid, "handle": "photo_1"}, context=other_ctx)
    checks.append(("original_scope_denied", denied2.get("blocked") == ["scope_mismatch"]))
    # 过期结果集
    runtime_tools._RUNTIME["result_sets"]._memory[rid].expires_at = time.time() - 1
    expired = runtime_tools._get_result_page({"result_set_id": rid, "page": 1}, context=ctx)
    checks.append(("page_expired", expired.get("blocked") == ["unknown_result_set"]))
    return checks


MODEL_CASES = [
    {"id": "rp01_t1", "query": "帮我找一些户外照片", "expect_page": False, "turn": 1},
    {"id": "rp01_t2", "query": "还有吗，再给我看下一页", "expect_page": True, "turn": 2},
    {"id": "rp02", "query": "去年拍了多少张照片", "expect_page": False, "turn": 1},
]


def model_tests(base_url, scope_id, db_path):
    runtime = build_runtime(base_url, scope_id, db_path)
    results = []
    prev_state = None
    for case in MODEL_CASES:
        t0 = time.time()
        turn = runtime.run(case["query"], task_state=prev_state)
        dt = round(time.time() - t0, 2)
        tools = [s.get("tool") for s in turn.steps if s.get("type") == "tool"]
        used_page = "get_result_page" in tools
        verdict = "PASS" if (case["expect_page"] == used_page) else "FAIL"
        results.append({"id": case["id"], "query": case["query"], "status": turn.status,
                        "tools": tools, "used_page": used_page, "expect_page": case["expect_page"],
                        "final": turn.final_answer, "latency_s": dt, "verdict": verdict,
                        "rs_ctx": turn.task_state.get("current_result_set")})
        prev_state = turn.task_state
        print(f"{case['id']} {verdict} tools={tools} status={turn.status} {dt}s | {turn.final_answer[:60]}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8105/v1")
    ap.add_argument("--scope", default="album2_e2b")
    ap.add_argument("--db", default="data/sentrix.db")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    unit = unit_tests()
    print("UNIT:", [(name, ok) for name, ok in unit])
    tool = tool_tests(args.scope, args.db)
    print("TOOL:", [(name, ok) for name, ok in tool])
    model = model_tests(args.base, args.scope, args.db)
    summary = {
        "unit_pass": sum(1 for _, ok in unit if ok), "unit_total": len(unit),
        "tool_pass": sum(1 for _, ok in tool if ok), "tool_total": len(tool),
        "model_pass": sum(1 for r in model if r["verdict"] == "PASS"), "model_total": len(model),
        "model": model,
    }
    print("SUMMARY:", json.dumps({k: v for k, v in summary.items() if k != "model"}, ensure_ascii=False))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"unit": unit, "tool": tool, **summary}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
