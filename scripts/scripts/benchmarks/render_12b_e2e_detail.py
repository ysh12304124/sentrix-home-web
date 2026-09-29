#!/usr/bin/env python3
"""Render the detailed 12B-FC E2E report (question / answer / evidence per case).

Reads sentrix-12b-full-chain-cases.json and writes a verbose markdown report
with the FULL user question, FULL agent answer, evidence details, model calls
and route for every case — the "what did you ask, what did the agent say, what
evidence came back" report.

Run locally:
  python3 scripts/benchmarks/render_12b_e2e_detail.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _md_table(rows):
    if not rows:
        return "（无）"
    out = ["| 字段 | 值 |", "|---|---|"]
    for k, v in rows:
        out.append(f"| {k} | {str(v).replace(chr(10), ' ') if not isinstance(v, str) else v.replace(chr(10), ' ')} |")
    return "\n".join(out)


def main():
    src = REPO_ROOT / "docs" / "baseline" / "sentrix-12b-full-chain-cases.json"
    out = REPO_ROOT / "docs" / "baseline" / "sentrix-12b-full-chain-e2e-detail.md"
    data = json.loads(src.read_text(encoding="utf-8"))
    lines = ["# 12B 完整链路 E2E — 逐 case 详细测试报告（问题 / 答案 / 证据）",
             "",
             f"**日期**：2026-08-06　**API**：{data.get('api')}　**结果**：{data['passed']}/{data['count']} 通过",
             "",
             "---", ""]
    for c in data["cases"]:
        lines.append(f"## Case: {c['name']}")
        lines.append("")
        lines.append(f"- **判定**: **{c['verdict']}**　| **Route/status**: `{c['evidence_status']}`　| **耗时**: {c['latency_s']}s")
        lines.append(f"- **问题（用户输入原文）**: `{c['message']}`")
        lines.append(f"- **Scope**: `{c['scope']}`")
        lines.append(f"- **期望模型角色**: {c['expected_model_roles']}　| **实际模型角色**: {c['actual_model_roles']}　| 缺失: {c['missing_roles']}")
        lines.append(f"- **all_models_match**: {c['all_models_match']}　| **degradation_used**: {c['degradation_used']}　| **assertion({c['assertion']})**: {c['assertion_ok']}")
        lines.append("")
        if c.get("model_calls"):
            lines.append("**模型调用（ModelCallLedger）**：")
            lines.append("")
            lines.append("| role | actual_model | latency_ms | json_valid | fallback | error |")
            lines.append("|---|---|---:|---:|---:|---|")
            for mc in c["model_calls"]:
                lines.append(f"| {mc['role']} | {mc['actual']} | {mc['latency_ms']} | {mc['json']} | {mc['fb']} | {mc['err']} |")
            lines.append("")
        lines.append("**Agent 回答（answer 原文）**：")
        lines.append("")
        lines.append(f"> {c['answer']}")
        lines.append("")
        lines.append(f"**Evidence 数量**: {c['evidence_count']}")
        ev = c.get("evidence") or []
        if ev:
            lines.append("")
            lines.append("**Evidence 明细**：")
            for e in ev:
                conds = "; ".join(f"{k.split(':',1)[-1]}={v.get('status')}" for k, v in (e.get("condition_results") or {}).items())
                lines.append(f"- `{e['asset_id']}` file=`{e['file_name']}` level=`{e['level']}` recall={e.get('recall_strength')} cond=[{conds}]")
        gaps = c.get("gaps") or []
        if gaps:
            lines.append("")
            lines.append(f"**Gaps**: {json.dumps(gaps, ensure_ascii=False)}")
        claims = c.get("claims") or []
        if claims:
            lines.append("")
            lines.append(f"**Claims ({len(claims)})**: {json.dumps(claims, ensure_ascii=False)[:400]}")
        if c.get("image_results"):
            lines.append("")
            lines.append(f"**image_results**: {[i.get('asset_id') for i in c['image_results']]}")
        if c.get("issues"):
            lines.append("")
            lines.append(f"**issues**: {c['issues']}")
        if c.get("error"):
            lines.append("")
            lines.append(f"**error**: {c['error']}")
        lines.append("")
        lines.append("---")
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
