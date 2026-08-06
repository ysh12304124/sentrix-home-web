#!/usr/bin/env python3
"""Phase 12B-FC V1 — role-level direct probes for every Agent LLM role.

Each role calls the 12B model DIRECTLY (bypassing the Agent Router) with the
sentrix role context, records model / latency / JSON legality, and verifies the
actual model is the configured 12B.  Every role must complete >= ``--warm`` runs
at 100% success with fallback_used=false for the phase to pass.

Roles: parser, answer, evidence_answer, writer, claim, verifier, repairer.

Run on 153:
  PYTHONPATH=. .venv/bin/python scripts/maintenance/probe_12b_roles.py
Output: docs/baseline/sentrix-12b-role-probes.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
MODEL = os.getenv("SENTRIX_PARSE_MODEL", "gemma4:12b")

# Role -> (json_mode, prompt).  These are structural sentrix-role probes, not
# the full production prompts — they verify the 12B can PERFORM each role.
ROLE_PROBES = {
    "parser": (True, "你是 Sentrix 查询解析器。把用户消息解析为 QueryParseDraft JSON（含 mode/actions/facets/time/media），只输出 JSON。用户消息：2024年5月厨房里做晚饭的照片"),
    "answer": (False, "你是 Sentrix，一个自然、克制的家庭数字助手。用户：今天感觉怎么样？请直接自然回答，不要提到数据库、检索或工具。"),
    "evidence_answer": (False, "你是 Sentrix。以下是你记忆里的证据：有一条照片记录，地点厨房，时间2024年5月，活动做晚饭（确定）；衣物颜色无法确认。用户问：2024年5月厨房里做了什么？只用证据回答，明确确定项和无法确认项。"),
    "writer": (False, "你是 Sentrix 人物总结 Writer。人物：明哥。记录：多次出现在厨房做饭（确定）、去过公园（确定）、性格是否外向（未知）。写一段自然的人物介绍，不新增任何事实。"),
    "claim": (True, "你是 Claim 提取器。从下面文本提取所有家庭主张（含否定和未知），输出 JSON claims 数组（每项 text/intended_type）。文本：明哥去年在厨房做过晚饭，但不确定他是否喜欢做饭；他没有去过北京。"),
    "verifier": (True, "你是 Verifier。对每条 claim 根据 canonical evidence 判断 supported/unsupported，输出 JSON（claim_id/status/reason）。claims:[{id:c1,text:明哥在厨房做过晚饭}],evidence:[{id:e1,text:2024年5月厨房做晚饭照片记录}]"),
    "repairer": (True, "你是 Repairer。下面 claim 过度断言，只局部降低断言强度，不整段重写，输出 JSON（revised_text/reason）。claim：明哥每天都做饭。证据只支持他做过几次。"),
}


def _json_ok(text):
    try:
        json.loads(text)
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm", type=int, default=3)
    parser.add_argument("--report", default=str(REPO_ROOT / "docs" / "baseline" / "sentrix-12b-role-probes.json"))
    args = parser.parse_args()

    import httpx
    client = httpx.Client(timeout=120)
    roles = {}
    for role, (json_mode, prompt) in ROLE_PROBES.items():
        runs = []
        for i in range(args.warm):
            payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "stream": False, "options": {"temperature": 0, "num_predict": 512, "num_ctx": 4096},
                       "think": False}
            if json_mode:
                payload["format"] = "json"
            t0 = time.monotonic()
            try:
                resp = client.post(f"{BASE}/api/chat", json=payload)
                body = resp.json()
                text = body.get("message", {}).get("content") or ""
                actual_model = body.get("model")
                runs.append({
                    "run": i, "status": resp.status_code, "actual_model": actual_model,
                    "latency_s": round(time.monotonic() - t0, 2),
                    "json_valid": _json_ok(text) if json_mode else None,
                    "input_size": len(prompt), "output_size": len(text),
                    "response_sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
                    "fallback_used": False, "cache_hit": False,
                    "circuit_breaker_state": "closed", "error": None,
                    "sample": text[:120],
                })
            except Exception as exc:
                runs.append({"run": i, "status": "error", "actual_model": None,
                             "latency_s": round(time.monotonic() - t0, 2), "json_valid": False,
                             "error": str(exc)[:200], "fallback_used": True, "sample": ""})
        success = sum(1 for r in runs if r["status"] == 200)
        model_ok = all(r["actual_model"] == MODEL for r in runs if r["status"] == 200)
        json_ok = all(r["json_valid"] for r in runs if r["json_valid"] is not None and r["status"] == 200)
        roles[role] = {
            "json_mode": json_mode,
            "success_rate": round(success / len(runs), 4),
            "all_models_match": model_ok,
            "json_legal_rate": json_ok,
            "warm_p50_s": round(sorted(r["latency_s"] for r in runs)[len(runs) // 2], 2) if runs else None,
            "runs": runs,
        }

    report = {"model": MODEL, "endpoint": BASE, "warm_runs": args.warm, "roles": roles,
              "overall": {
                  "all_roles_success": all(r["success_rate"] == 1.0 for r in roles.values()),
                  "all_roles_model_match": all(r["all_models_match"] for r in roles.values()),
                  "all_json_roles_legal": all(r["json_legal_rate"] for r in roles.values()
                                              if r["json_mode"]),
                  "fallback_used_anywhere": any(any(x.get("fallback_used") for x in r["runs"])
                                                for r in roles.values()),
              }}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "roles"}, ensure_ascii=False, indent=2))
    for role, r in roles.items():
        print(f"{role:15} success={r['success_rate']} model_match={r['all_models_match']} "
              f"json_legal={r['json_legal_rate']} warm_p50={r['warm_p50_s']}s")
    return 0 if report["overall"]["all_roles_success"] and report["overall"]["all_roles_model_match"] else 1


if __name__ == "__main__":
    sys.exit(main())
