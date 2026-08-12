#!/usr/bin/env python3
"""Phase F v2 F7 — inspect_result_set Spike（在 153 上运行，真实 DB + 8100 VLM）。

对比同一批多图/选图问题的两种解析方式：
  A. 当前 LLM 循环（baseline）：AgentRuntime 自主决定逐张 inspect_photo
  B. 候选 Tool 级批量（spike）：search_memories → 结果集内按序逐张 VLM "supports?" →
     首个支持即停（一次 LLM 回合内完成）

指标：Correct Asset Selection / Answer Found Rate / Images Inspected / VLM Calls / Latency。

用法（153）：
  cd /home/asus/Github/Sentrix-Home-Web
  PYTHONPATH=. python scripts/benchmarks/inspect_result_set_spike.py \
      --questions validation-album3-026-q01,validation-album3-047-q07 \
      --out /tmp/inspect_result_set_spike.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from backend.db import MemoryStore                      # noqa: E402
from backend.model_clients import GammaClient, ClipAdapter, parse_json_response  # noqa: E402
from backend.embeddings import EmbeddingRouter          # noqa: E402
from backend.retrieval import RetrievalConfig           # noqa: E402
from backend.agent_runtime import tools as runtime_tools  # noqa: E402
from backend.agent_runtime.runtime import AgentRuntime  # noqa: E402

SUPPORTS_PROMPT = """你是 Sentrix 家庭照片归档助手。用户问题：{question}
请判断这张照片是否【直接回答了用户问题】（照片本身包含问题要求的证据，例如指定的人物/物体/文字/地点/数量）。
只输出 JSON，不要多余文字：{{"supports": true或false, "reason": "一句话理由"}}"""

SPIKE_QUESTIONS = [
    # 多图/选图/需要从结果集中挑出正确照片的问题（final3 实测失败或仅 1 次 inspect）
    "validation-album3-012-q02",   # 两张合影
    "validation-album3-012-q08",   # 黄色镂空衣服
    "validation-album3-024-q02",   # 大圣葱油拌面
    "validation-album3-024-q05",   # 白色背心
    "validation-album3-026-q01",   # 1974 年份照片
    "validation-album3-047-q03",   # 火把
    "validation-album3-047-q07",   # 有火把的照片
]


def load_qa(qa_path) -> dict:
    out = {}
    for line in Path(qa_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out[d["qa_id"]] = d
    return out


def gold_names(qa: dict) -> list:
    return [Path(x).name.lower() for x in (qa.get("answer_evidence_image_ids") or [])]


def asset_names(store) -> dict:
    rows = store.connection.execute("SELECT id, file_name FROM assets").fetchall()
    return {r["id"]: (r["file_name"] or "") for r in rows}


def run_baseline(rt, question, scope_id):
    t0 = time.time()
    turn = rt.run(question, history="", task_state=None, progress_callback=None,
                  selected_handle=None, selected_result_set_id=None,
                  conversation_summary="")
    latency = time.time() - t0
    trace = [s for s in (turn.steps or []) if s.get("type") == "tool"]
    vlm_calls = sum(1 for t in trace if t.get("tool") in ("inspect_photo", "read_photo_text"))
    for t in trace:
        if t.get("tool") == "read_photo_text":
            vlm_calls += 4  # 2x2 = 4 tiles（整图 1 次已计入）
    return {
        "latency_s": round(latency, 1),
        "tools": [t.get("tool") for t in trace],
        "vlm_calls": vlm_calls,
        "answer": (turn.final_answer or "")[:120],
        "status": turn.status,
    }


def run_spike_batch(store, gamma, question, scope_id, max_images=5):
    ctx = {"scope_id": scope_id, "viewer_id": "owner", "task_state": {"user_goal": question}}
    t0 = time.time()
    obs = runtime_tools._search_memories({"query": question, "mode": "best"}, context=ctx)
    preview = obs.get("preview") or []
    inspected, matched = [], None
    for p in preview[:max_images]:
        handle = p.get("handle")
        aid = runtime_tools._handle_to_asset_id(handle)
        if not aid:
            continue
        row = store.connection.execute("SELECT path FROM assets WHERE id = ?", (aid,)).fetchone()
        if not row or not Path(row["path"]).is_file():
            continue
        import base64
        with open(row["path"], "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        raw = gamma.chat(SUPPORTS_PROMPT.format(question=question),
                         images=[{"base64": b64, "mime_type": "image/jpeg"}],
                         json_mode=True, role="inspect")
        parsed = parse_json_response(raw) or {}
        inspected.append({"handle": handle, "asset_id": aid, "supports": bool(parsed.get("supports")),
                          "reason": (parsed.get("reason") or "")[:80]})
        if parsed.get("supports"):
            matched = inspected[-1]
            break
    latency = time.time() - t0
    return {
        "latency_s": round(latency, 1),
        "inspected": len(inspected),
        "vlm_calls": len(inspected),
        "matched_handle": matched and matched["handle"],
        "matched_asset_id": matched and matched["asset_id"],
        "matched_reason": matched and matched["reason"],
        "stop_reason": "found_support" if matched else "budget_exhausted",
        "result_set_total": obs.get("total"),
        "preview_size": len(preview),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default="/home/asus/Downloads/album3/qa/full-album3.jsonl")
    ap.add_argument("--scope", default="album3")
    ap.add_argument("--questions", default="", help="逗号分隔 qa_id；空则用内置 SPIKE_QUESTIONS")
    ap.add_argument("--max-images", type=int, default=5)
    ap.add_argument("--out", default="/tmp/inspect_result_set_spike.json")
    ap.add_argument("--base", default="http://192.168.0.153:8100/v1")
    args = ap.parse_args()

    qa_map = load_qa(args.qa)
    qids = [x.strip() for x in args.questions.split(",") if x.strip()] or SPIKE_QUESTIONS

    store = MemoryStore(os.getenv("SENTRIX_DB_PATH", str(ROOT / "data" / "sentrix.db")))
    gamma = GammaClient(base_url=args.base)
    try:
        embedding_router = EmbeddingRouter.from_clip(ClipAdapter())
        retrieval_config = RetrievalConfig()
    except Exception as exc:
        print(f"[warn] embedding 不可用，降级确定性检索: {exc}")
        embedding_router = None
        retrieval_config = None
    runtime_tools.bind_runtime(store, gamma=gamma, embedding_router=embedding_router,
                               retrieval_config=retrieval_config)
    runtime_tools.register_tools()
    names = asset_names(store)

    def chat_fn(messages):
        payload = {"model": getattr(gamma, "model", "gemma4-12b-it"),
                   "messages": messages, "temperature": 0.0, "max_tokens": 1500}
        import httpx
        resp = httpx.post(f"{gamma.base_url}/chat/completions", json=payload, timeout=120)
        resp.raise_for_status()
        return (resp.json().get("choices") or [{}])[0].get("message", {}).get("content", "")

    rt = AgentRuntime(chat_fn=chat_fn, profile_name="tool_loop", scope_id=args.scope,
                      viewer_id="owner")

    out = []
    print(f"inspect_result_set Spike | scope={args.scope} | {len(qids)} 题")
    for qid in qids:
        qa = qa_map.get(qid)
        if not qa:
            print(f"[skip] {qid} 不在 QA 文件")
            continue
        gold = gold_names(qa)
        print(f"[{qid}] 问题: {qa['question'][:40]}… gold={gold}")
        base = run_baseline(rt, qa["question"], args.scope)
        spike = run_spike_batch(store, gamma, qa["question"], args.scope, args.max_images)
        spike_gold_hit = None
        if spike.get("matched_asset_id"):
            fn = names.get(spike["matched_asset_id"], "").lower()
            spike_gold_hit = bool(fn and Path(fn).name in gold)
        out.append({"qa_id": qid, "question": qa["question"], "gold_images": gold,
                    "baseline": base, "spike_batch": {**spike, "gold_hit": spike_gold_hit}})
        print(f"  baseline: tools={base['tools']} vlm={base['vlm_calls']} {base['latency_s']}s status={base['status']}")
        print(f"  spike   : inspected={spike['inspected']}/{spike.get('preview_size')} "
              f"vlm={spike['vlm_calls']} {spike['latency_s']}s "
              f"stop={spike['stop_reason']} gold_hit={spike_gold_hit}")
        print(f"           answer={base['answer'][:80]}")
    payload = {"meta": {"created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "scope": args.scope, "vlm": gamma.base_url,
                        "questions": qids},
               "rows": out}
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"输出: {args.out}")


if __name__ == "__main__":
    main()
