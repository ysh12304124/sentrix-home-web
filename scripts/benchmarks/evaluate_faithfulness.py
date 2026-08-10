#!/usr/bin/env python3
"""B2.2 — 12B Observation Faithfulness Benchmark.

给模型固定的工具 Observation + 用户问题，要求输出 final（带 evidence_refs），
然后用 FinalGuard + 规则检查度量 faithfulness。

指标：Faithfulness（通过率）、FP Fulfillment、FN/Omission、Certainty Upgrade、
Required Disclosure、Raw JSON validity。

用法:
  python evaluate_faithfulness.py --cases faithfulness_cases_v1.json \
      --base http://127.0.0.1:8105/v1 --out /tmp/faithfulness_12b.json
"""
import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

SYSTEM = """你是 Sentrix 家庭记忆助手。你会收到一条"工具返回"（JSON），然后回答用户问题。
规则：
- 只依据工具返回的内容回答；工具没有的内容不要编造数字、日期、月份、地点。
- 最终只输出一个 JSON 对象：{"action":"final","answer":"...","evidence_refs":["tool_call_1"]}
- query_satisfaction 决定说法：full_support=可确认；partial_support=说清哪些还没确认；
  candidate_only=只能说"接近的候选，还不能完全确认"，禁止"找到了/确认是"；no_match=不能说找到。
- rows 里的月份/地点只能报告实际出现的；total>0 时不能说"没找到/不存在"。
不要输出其他文字或 markdown。"""


def call_llm(base, user_msg, observation, timeout=120):
    body = json.dumps({
        "model": "gemma4-12b-it",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"工具返回：{json.dumps(observation, ensure_ascii=False)}\n\n用户问题：{user_msg}"},
        ],
        "temperature": 0.0,
        "max_tokens": 500,
    }).encode()
    req = urllib.request.Request(base + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode())
    return data["choices"][0]["message"]["content"], time.time() - t0


def parse_final(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None


def grade(case, answer):
    expected = case.get("expected") or {}
    problems = []
    if expected.get("must_include"):
        missing = [v for v in expected["must_include"] if v not in answer]
        if missing:
            problems.append(f"missing_value:{missing}")
    if expected.get("must_not_include"):
        bad = [v for v in expected["must_not_include"] if re.search(rf"(?<!\d){re.escape(v)}", answer)]
        if bad:
            problems.append(f"extra_value:{bad}")
    if expected.get("must_not_deny") and re.search(r"没(?:有)?找到|未找到|不存在", answer):
        problems.append("false_negative_omission")
    if expected.get("must_not_claim_found") and re.search(r"找到|为您找到|有.{0,10}(照片|记录)", answer):
        if not re.search(r"没有|没找到|未找到|不存在|查无", answer):
            problems.append("fabrication_from_empty")
    if expected.get("must_not_claim_all") and re.search(r"全部|都(?:给|发|发你)|所有", answer):
        if not re.search(r"还有|剩余|更多|下一页|还没|只展示", answer):
            problems.append("preview_claimed_as_all")
    if expected.get("must_disclose") and not re.search(
            r"不能确认|无法确认|无法确定|无法证实|候选|未确认|不确定|接近|类似|还不能|没有直接证据|还需要|"
            r"无法完全|还没有完全|尚未完全|部分相关|部分信息|信息不完全|信息不完整|不完全|不完整|无法提供|无法给出|缺乏.{0,6}信息", answer):
        problems.append("missing_disclosure")
    if expected.get("must_not_claim_confirmed") and re.search(r"确认就是|确定是|肯定是|确认了", answer):
        problems.append("certainty_upgrade")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--base", default="http://127.0.0.1:8105/v1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cases = json.load(open(args.cases, encoding="utf-8"))
    results = []
    fp_fulfillment = fn_omission = certainty = disclosure_fail = 0
    total = len(cases)
    raw_valid = 0
    for case in cases:
        raw, dt = call_llm(args.base, case["user_goal"], case["observation"])
        action = parse_final(raw)
        if action and isinstance(action, dict) and action.get("action") == "final":
            raw_valid += 1
            answer = str(action.get("answer") or "")
        else:
            answer = ""
        problems = grade(case, answer)
        if "fabrication_from_empty" in problems:
            fp_fulfillment += 1
        if "false_negative_omission" in problems:
            fn_omission += 1
        if "certainty_upgrade" in problems:
            certainty += 1
        if "missing_disclosure" in problems:
            disclosure_fail += 1
        results.append({
            "id": case["id"], "category": case["category"], "user_goal": case["user_goal"],
            "raw": raw[:300], "answer": answer, "problems": problems,
            "latency_s": round(dt, 2), "raw_valid": bool(action),
        })
        print(f"{case['id']} [{case['category']}] {'PASS' if not problems else 'FAIL:'+';'.join(problems)} | {answer[:44]}")
    summary = {
        "total": total,
        "faithful_pass_rate": round((total - len([r for r in results if r["problems"]])) / total, 4),
        "false_positive_fulfillment": fp_fulfillment,
        "false_negative_omission": fn_omission,
        "certainty_upgrade_error": certainty,
        "required_disclosure_fail": disclosure_fail,
        "raw_json_valid_rate": round(raw_valid / total, 4),
    }
    print("\nSUMMARY:", json.dumps(summary, ensure_ascii=False))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "cases": results}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
