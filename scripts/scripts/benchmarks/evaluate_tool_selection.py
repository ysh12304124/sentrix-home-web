#!/usr/bin/env python3
"""A0.5 12B Tool Selection Capability Spike.

直接调用 vLLM 8100（OpenAI 兼容），评估 12B 在中文家庭问题上选择
Tool 的能力。仅用于能力评估，不注入任何 runtime。

用法:
  python evaluate_tool_selection.py --set tool_selection_set.json \
      --base http://127.0.0.1:8100/v1 --model gemma4-12b-it --out result.json
"""
import argparse
import json
import time
import urllib.request
from collections import Counter

SYSTEM = """你是 Sentrix 家庭记忆助手。你的任务：根据用户这句话，选择下一步动作，并输出一个 JSON 对象，不要输出其他内容。

可选动作：
1. {"action": "final", "answer": "直接回答"} — 纯聊天、寒暄、写作、翻译、不需要查家庭记忆的问题，直接回答。
2. {"action": "tool_call", "tool": "query_memory_facts", "arguments": {"operation": "count|exists|first|last|group|date|media", "filters": {"time": "", "person": "", "place": "", "media": ""}}, "public_status": "我在查记忆事实。"} — 数量、时间、日期、首次/最后一次出现、地点聚合、媒体类型等结构化事实问题。
3. {"action": "tool_call", "tool": "search_memories", "arguments": {"query": "检索词", "mode": "best|all|representative", "filters": {"time": "", "place": "", "person": ""}}, "public_status": "我在找相关照片。"} — 找照片、图片语义（衣着/颜色/物体/场景）、混合查询。
4. {"action": "tool_call", "tool": "get_original_photos", "arguments": {"result_set_id": "", "handle": ""}, "public_status": "我在准备原图。"} — 用户要原图/当前结果集中的某张图，且上下文已有结果集。
5. {"action": "tool_call", "tool": "clarify", "arguments": {"question": "澄清问题"}, "public_status": "我需要确认一下。"} — 请求太模糊，无法判断查什么。

规则：
- 聊天/写作/翻译/寒暄/闲聊 → final。
- 问数量/时间/首次/最后一次/月份/地点统计 → query_memory_facts。
- 找照片、照片内容（衣服颜色/物体/场景/合影）、介绍某个人（如"介绍一下明哥"）、"我和X有合影吗" → search_memories。
- 结果集 follow-up："还有吗""继续""更多" → search_memories（在已有结果集内继续找）。
- 用户说"把第N张原图给我""刚才那张的原图" → get_original_photos（假设结果集存在）。
- 只说"第二张""第一张"（数字序号）→ search_memories（在当前结果集内定位），不是原图请求。
- "帮我看看照片""随便看看" 这类宽泛请求 → search_memories。
- 完全无法推测（如"之前聊到的那个"且无上下文）→ clarify。
只输出一个 JSON 对象，不要用 markdown 代码块包裹，不要输出解释。严格示例：
{"action": "final", "answer": "你好"}
{"action": "tool_call", "tool": "query_memory_facts", "arguments": {"operation": "count", "filters": {"time": "去年"}}, "public_status": "我在查记忆事实。"}
{"action": "tool_call", "tool": "search_memories", "arguments": {"query": "黄色睡衣", "mode": "best", "filters": {}}, "public_status": "我在找相关照片。"}
{"action": "tool_call", "tool": "get_original_photos", "arguments": {"result_set_id": "rs", "handle": "photo_2"}, "public_status": "我在准备原图。"}
{"action": "tool_call", "tool": "clarify", "arguments": {"question": "您具体想看什么？"}, "public_status": "我需要确认一下。"}"""


def call_llm(base, model, user_msg, timeout=120):
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.0,
        "max_tokens": 300,
    }).encode()
    req = urllib.request.Request(base + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode())
    text = data["choices"][0]["message"]["content"]
    return text, time.time() - t0


def parse_action(text):
    text = (text or "").strip()
    try:
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3]
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
        return json.loads(text)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--base", default="http://127.0.0.1:8100/v1")
    ap.add_argument("--model", default="gemma4-12b-it")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dataset = json.load(open(args.set, encoding="utf-8"))
    cases = dataset["cases"]
    rows = []
    stats = Counter()
    for c in cases:
        raw, sec = call_llm(args.base, args.model, c["query"])
        action = parse_action(raw)
        tool = None
        schema_ok = False
        if isinstance(action, dict):
            a = action.get("action")
            if a == "final":
                tool, schema_ok = "final", True
            elif a == "tool_call" and isinstance(action.get("tool"), str):
                tool = action["tool"]
                args_ok = isinstance(action.get("arguments"), dict)
                status_ok = isinstance(action.get("public_status"), str)
                schema_ok = args_ok and status_ok
        correct = (tool == c["expected"])
        rows.append({
            "id": c["id"], "category": c["category"], "query": c["query"],
            "expected": c["expected"], "tool": tool, "correct": correct,
            "schema_ok": schema_ok, "raw": raw, "latency": round(sec, 2),
        })
        stats["total"] += 1
        if schema_ok:
            stats["schema_ok"] += 1
        if correct:
            stats["correct"] += 1
        if tool != c["expected"] and c["expected"] in {"final"}:
            stats["unnecessary_tool"] += 1
        if not correct:
            stats.setdefault("miss_" + c["expected"], 0)
            stats["miss_" + c["expected"]] += 1
        print(f"{c['id']} [{c['category']}] expect={c['expected']} got={tool} "
              f"{'OK' if correct else 'X'} schema={schema_ok} {round(sec,1)}s")
        time.sleep(0.1)

    n = stats["total"]
    report = {
        "set": args.set, "model": args.model, "base": args.base,
        "total": n,
        "schema_validity": round(stats["schema_ok"] / n, 4),
        "primary_action_accuracy": round(stats["correct"] / n, 4),
        "unnecessary_tool_call_rate": round(stats.get("unnecessary_tool", 0) / n, 4),
        "by_category": {},
        "confusion": {k: v for k, v in stats.items() if k.startswith("miss_")},
        "rows": rows,
    }
    cat_acc = Counter()
    cat_tot = Counter()
    for r in rows:
        cat_tot[r["category"]] += 1
        if r["correct"]:
            cat_acc[r["category"]] += 1
    for cat in sorted(set(cat_tot) | set(cat_acc)):
        report["by_category"][cat] = round(cat_acc.get(cat, 0) / cat_tot[cat], 3)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n==== REPORT ====")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
