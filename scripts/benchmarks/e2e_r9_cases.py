#!/usr/bin/env python3
"""Phase R9-6 — end-to-end case verification against the real API (§10).

Runs the R9 acceptance cases through POST /api/assistant/turn and asserts route /
evidence / model-call expectations.  Each case records PASS/FAIL so the R9
closing report can cite concrete end-to-end evidence.  Requires the API on 153
with SENTRIX_AGENT_STAGE_TRACE=1 for model-call assertions.

Usage:
  PYTHONPATH=. .venv-mac/bin/python scripts/benchmarks/e2e_r9_cases.py --api http://127.0.0.1:8091
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _check(condition, message, detail=""):
    return {"pass": bool(condition), "message": message, "detail": detail}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default=os.getenv("SENTRIX_API_URL", "http://127.0.0.1:8091"))
    parser.add_argument("--report", default="docs/baseline/e2e_r9_cases.json")
    args = parser.parse_args()

    import httpx
    client = httpx.Client(timeout=60)

    def turn(message, scope_id):
        resp = client.post(f"{args.api}/api/assistant/turn",
                           json={"message": message, "scope_id": scope_id})
        return resp.json()

    checks = []

    def run_case(name, message, scope, asserts):
        body = turn(message, scope)
        perf = body.get("perf") or {}
        calls = perf.get("model_calls") or {}
        evidence = body.get("evidence") or []
        answer = str(body.get("answer") or "")
        status = body.get("evidence_status")
        results = []
        for check in asserts:
            results.append(check(body, calls, evidence, answer, status))
        checks.append({"name": name, "message": message, "evidence_status": status,
                       "evidence_count": len(evidence), "model_calls": calls,
                       "answer": answer[:120], "checks": results})
        print(f"[{'PASS' if all(r['pass'] for r in results) else 'FAIL'}] {name}: "
              f"status={status} evidence={len(evidence)} calls={calls}", flush=True)

    # 1. 人物介绍 -> evidence/clarify (never normal_chat).  Complex chain needs
    #    a confirmed 明哥 entity — the benchmark albums have none, so clarify is
    #    the safe fallback; the hard gate is "not dropped to normal chat".
    run_case("person_intro", "介绍一下明哥", "album1",
             [lambda b, c, e, a, s: _check(s != "not_applicable", "not dropped to normal chat", f"status={s}")])

    # 2. 写作 -> none, zero memory.
    def c2(b, c, e, a, s):
        return _check(s == "not_applicable" and not b.get("memory_used"), "writing has no memory", f"status={s}")
    run_case("writing", "帮我写一段生日祝福", "album1", [c2])

    # 3. 短语无命中 -> clarify/refusal, never product talk.
    def c3(b, c, e, a, s):
        safe = s in {"clarify", "anchored", "gap"} or "照片" in a or "记忆" in a or "无法确认" in a
        no_product = "商品" not in a and "购物" not in a
        return _check(safe and no_product, "ambiguous phrase -> clarify/refusal, no product talk", f"status={s} ans={a[:60]}")
    run_case("short_visual_phrase", "银色心形手镯", "album3", [c3])

    # 4. 照片里写着什么 -> household, not writing.
    def c4(b, c, e, a, s):
        return _check(b.get("memory_used"), "photo-reads query uses memory", f"status={s}")
    run_case("photo_reads", "照片里写着什么？", "album1", [c4])

    # 5. 简单 evidence -> deterministic answer, parser 1, no writer.
    def c5(b, c, e, a, s):
        parser_ok = c.get("parser", 0) in (0, 1)
        writer_skipped = c.get("answer", 0) == 0 and "complex_chain" not in (b.get("perf") or {})
        return _check(parser_ok and writer_skipped, "simple evidence skips writer", f"calls={c}")
    run_case("simple_evidence", "2024年5月厨房里做晚饭", "album1", [c5])

    # 6. 为什么...照片 -> evidence (general verb does not decide none).
    def c6(b, c, e, a, s):
        return _check(b.get("memory_used"), "why+date+person stays household", f"status={s}")
    run_case("general_verb_household", "为什么去年春节没有小黑的照片", "album1", [c6])

    # 7. 海豚 -> allow_approximate disclosure or safe refusal.
    def c7(b, c, e, a, s):
        disclosed = ("无法确认" in a) or ("近似" in a) or ("可能" in a) or ("没有找到" in a)
        return _check(disclosed or not e, "approximate disclosed or refused", f"status={s} ans={a[:60]}")
    run_case("allow_approximate", "水族馆海豚跃出水面", "album3", [c7])

    # 8. 贵阳 -> strict_empty refusal, 0 evidence.
    def c8(b, c, e, a, s):
        return _check(not e and ("没有找到" in a or "未找到" in a), "strict_empty refusal", f"status={s} ans={a[:60]}")
    run_case("strict_empty", "贵阳夜晚步行街", "album1", [c8])

    # 9. 会话后续 -> evidence/clarify (focus reused), not none.
    def c9(b, c, e, a, s):
        return _check(s != "not_applicable", "session follow-up is not chat", f"status={s}")
    run_case("session_followup", "上次说的那件黄色的", "album1", [c9])

    # 10. 概念问题 -> none/chat, no memory.
    def c10(b, c, e, a, s):
        return _check(s == "not_applicable" and not b.get("memory_used"), "concept question is chat", f"status={s}")
    run_case("concept_question", "解释一下量子纠缠", "album1", [c10])

    summary = {"api": args.api, "cases": len(checks),
               "passed": sum(1 for c in checks if all(r["pass"] for r in c["checks"])),
               "failed": sum(1 for c in checks if not all(r["pass"] for r in c["checks"]))}
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "cases": checks}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{summary}")
    print(f"wrote {out}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
