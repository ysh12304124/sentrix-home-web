#!/usr/bin/env python3
"""分析 Tool-Loop shadow 结果中 raw action JSON 的修复率与失败模式（B2.3）。

用法:
  python analyze_action_repair.py --result /tmp/b0_shadow.json
"""
import argparse
import json


def strict_parse(raw: str) -> dict | None:
    raw = (raw or "").strip()
    if not raw or raw.startswith("```"):
        return None
    try:
        obj = json.loads(raw)
    except Exception:
        return None
    if isinstance(obj, dict) and obj.get("action") in ("tool_call", "final"):
        return obj
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    args = ap.parse_args()
    data = json.load(open(args.result, encoding="utf-8"))
    total = raw_ok = repaired = 0
    patterns = {}
    per_case = {}
    for r in data:
        tc = {"model": 0, "raw_ok": 0, "repaired": 0}
        for s in r.get("steps", []):
            if s.get("type") != "model":
                continue
            raw = s.get("raw", "")
            total += 1
            tc["model"] += 1
            if strict_parse(raw):
                raw_ok += 1
                tc["raw_ok"] += 1
            else:
                repaired += 1
                tc["repaired"] += 1
                if "```" in raw:
                    patterns["markdown_fence"] = patterns.get("markdown_fence", 0) + 1
                elif raw.count("{") > raw.count("}"):
                    patterns["unclosed_brace"] = patterns.get("unclosed_brace", 0) + 1
                elif raw.count('"') % 2 == 1:
                    patterns["unclosed_string"] = patterns.get("unclosed_string", 0) + 1
                elif '"' in raw and "{" not in raw:
                    patterns["no_json_object"] = patterns.get("no_json_object", 0) + 1
                else:
                    patterns["other"] = patterns.get("other", 0) + 1
                per_case.setdefault(r["id"], {"status": r.get("status"), "reason": r.get("reason"),
                                             "repaired": 0, "raw": raw[:120]})
                per_case[r["id"]]["repaired"] += 1
    print(json.dumps({
        "total_model_outputs": total,
        "raw_valid": raw_ok,
        "repaired": repaired,
        "repair_rate_pct": round(repaired / total * 100, 1) if total else 0,
        "pattern_distribution": patterns,
        "repairing_cases": per_case,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
